"""Config-driven pipeline runner — declare every component + its params in one file.

``prefscope run --config pipeline.yaml`` runs the per-lens analysis chain
(``name → verify → classify-role → cluster → win-relevance``) where each stage's component is resolved
by name through the registry. The config is the declarative front-end over the same
swappable components the subcommands use, so changing a verifier or clustering algorithm
is a one-line edit:

    lens_dir: lenses/indiv_8b
    corpus:   corpora/arena.parquet        # needed by win-relevance; optional otherwise
    out_dir:  results/run1
    stages: [name, verify, classify-role, cluster, win-relevance]
    llm: {backend: openai, model: deepseek/deepseek-v3.2}
    interpreter: {name: auto, n_active: 12}
    verifier:    {name: auto, n_per_bucket: 12}
    role_classifier: {n_top: 6, n_random: 2}
    clusterer:   {name: mi-leiden, resolution: 1.2, knn: 6}

Outputs are written under ``out_dir`` with the canonical artifact names and threaded
forward (``verify`` reads ``name``'s csv; ``cluster``/``win-relevance`` read the
fidelity csv). Prompt lenses support ``name``, ``verify``, and ``cluster``. Role
classification and preference relevance are completion-only; role classification also
requires an individual-response lens.
"""
from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from prefscope.analysis.presence import annotation_flag

# Register only the lightweight built-ins used by this runner. SAE and dataset plug-ins
# remain lazy so importing the config runner does not import torch.
from prefscope.pipeline import cluster as _clusterers  # noqa: F401
from prefscope.artifacts import (
    FEATURE_CLUSTERS, FEATURE_FIDELITY, FEATURE_NAMES, FEATURE_ROLES, LLM_USAGE,
    LLM_USAGE_EVENTS, MANIFEST, PROMPT_FEATURE_CLUSTERS, PROMPT_FEATURE_FIDELITY,
    PROMPT_FEATURE_NAMES, WIN_RELEVANCE)
from prefscope.core import registry
from prefscope.core.plugins import load_plugins, normalize_plugin_modules
from prefscope.interpret.llm import DEFAULT_API_BASE, DEFAULT_MODEL, LLMClient, UsageTracker
from prefscope.interpret.strategy import (
    LensCodes,
    NameStrategy,
    VerifyCodes,
    VerifyStrategy,
    resolve_name_mode,
    resolve_verify_mode,
)

# Stage -> output filename, per lens kind. A completion lens runs the full chain; a prompt
# lens runs name/verify/cluster over z_prompt (win-relevance is completion-only because
# it measures associations between response features and observed preferences).
_COMPLETION_OUTPUTS = {
    "name": FEATURE_NAMES,
    "verify": FEATURE_FIDELITY,
    "classify-role": FEATURE_ROLES,
    "cluster": FEATURE_CLUSTERS,
    "win-relevance": WIN_RELEVANCE,
}
_PROMPT_OUTPUTS = {
    "name": PROMPT_FEATURE_NAMES,
    "verify": PROMPT_FEATURE_FIDELITY,
    "cluster": PROMPT_FEATURE_CLUSTERS,
}
KNOWN_STAGES = ("name", "verify", "classify-role", "cluster", "win-relevance")
DEFAULT_COMPLETION_STAGES = ("name", "verify", "cluster", "win-relevance")
DEFAULT_PROMPT_STAGES = ("name", "verify", "cluster")


def _stage_outputs(lens_kind: str) -> dict:
    return _PROMPT_OUTPUTS if lens_kind == "prompt" else _COMPLETION_OUTPUTS


@dataclass
class StageConfig:
    """A component selection (registry name) plus its constructor params."""
    component: str = "auto"
    params: dict = field(default_factory=dict)

    @classmethod
    def parse(cls, raw, *, default: str = "auto") -> "StageConfig":
        """Accept a bare name (``verifier: pairwise``) or a mapping with a ``name`` key
        (``verifier: {name: pairwise, n_per_bucket: 12}``)."""
        if raw is None:
            return cls(default, {})
        if isinstance(raw, str):
            return cls(raw, {})
        if not isinstance(raw, dict):
            raise ValueError(
                f"component config must be a name or a mapping, got {type(raw).__name__}")
        d = dict(raw)
        component = d.pop("name", None) or d.pop("component", None) or default
        return cls(component, d)


@dataclass
class LLMConfig:
    backend: str = "openai"
    model: str = DEFAULT_MODEL
    api_base: str = DEFAULT_API_BASE
    api_key_env: str = "OPENROUTER_API_KEY"

    _KEYS = ("backend", "model", "api_base", "api_key_env")

    @classmethod
    def parse(cls, raw) -> "LLMConfig":
        d = dict(raw or {})
        unknown = set(d) - set(cls._KEYS)
        if unknown:
            raise ValueError(
                f"unknown llm keys: {', '.join(sorted(unknown))}; "
                f"allowed: {', '.join(cls._KEYS)}")
        return cls(**d)

    def client(self, *, usage_tracker: UsageTracker | None = None,
               usage_stage: str = "llm") -> LLMClient:
        return LLMClient(backend=self.backend, model=self.model,
                         api_base=self.api_base, api_key_env=self.api_key_env,
                         usage_tracker=usage_tracker, usage_stage=usage_stage)


# Cluster-stage keys that steer the runner, not the clusterer constructor (popped before make).
_CLUSTER_CONTROL = ("cluster_on", "fidelity_only", "name_clusters", "concurrency")
_WIN_RELEVANCE_KEYS = ("all_features", "group_col")
_ROLE_CLASSIFIER_KEYS = (
    "linkage", "all_named", "features", "pole", "n_top", "n_random", "batch_size",
    "min_valid_examples", "seed", "concurrency",
)


def _accepted_params(kind: str, name: str) -> set | None:
    """The keyword params a component's ``__init__`` declares — its config contract.

    Validated against this so a misspelled or wrong-component param (e.g. ``n_clusters``
    on ``mi-leiden``, which only the k-means clusterers take) is rejected up front rather
    than silently swallowed by a ``**kwargs`` catch-all. ``auto`` shares the base
    ``__init__`` across the concrete strategies, so its base strategy defines the set.
    Returns ``None`` when the name can't be resolved — ``registry.make`` then raises the
    friendly "no such component" error instead."""
    if name == "auto":
        cls = {"interpreter": NameStrategy, "verifier": VerifyStrategy}.get(kind)
        if cls is None:
            return None
    else:
        try:
            cls = registry.get(kind, name)
        except KeyError:
            return None
    return {p.name for p in inspect.signature(cls.__init__).parameters.values()
            if p.name != "self" and p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)}


def _check_params(kind: str, sc: "StageConfig", *, extra=()) -> None:
    accepted = _accepted_params(kind, sc.component)
    if accepted is None:
        return
    allowed = accepted | set(extra)
    unknown = set(sc.params) - allowed
    if unknown:
        raise ValueError(
            f"unknown {kind} param(s) for {sc.component!r}: {', '.join(sorted(unknown))}; "
            f"allowed: {', '.join(sorted(allowed))}")


_TOP_KEYS = {"lens_dir", "out_dir", "stages", "corpus", "annotations", "lens_kind",
             "plugins",
             "llm", "name_llm", "verify_llm", "role_llm", "cluster_llm",
             "interpreter", "verifier", "role_classifier", "clusterer",
             "win_relevance"}


@dataclass
class PipelineConfig:
    """Typed, validated view of a pipeline config file."""
    lens_dir: str
    out_dir: str
    stages: list = field(default_factory=lambda: list(DEFAULT_COMPLETION_STAGES))
    corpus: str | None = None
    annotations: list | None = None
    lens_kind: str = "completion"
    llm: LLMConfig = field(default_factory=LLMConfig)
    name_llm: LLMConfig | None = None
    verify_llm: LLMConfig | None = None
    role_llm: LLMConfig | None = None
    cluster_llm: LLMConfig | None = None
    interpreter: StageConfig = field(default_factory=StageConfig)
    verifier: StageConfig = field(default_factory=StageConfig)
    role_classifier: dict = field(default_factory=dict)
    clusterer: StageConfig = field(default_factory=lambda: StageConfig("spherical-kmeans"))
    win_relevance: dict = field(default_factory=dict)
    plugins: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, d: dict) -> "PipelineConfig":
        if not isinstance(d, dict):
            raise ValueError("config root must be a mapping")
        unknown = set(d) - _TOP_KEYS
        if unknown:
            raise ValueError(
                f"unknown config keys: {', '.join(sorted(unknown))}; "
                f"allowed: {', '.join(sorted(_TOP_KEYS))}")
        for req in ("lens_dir", "out_dir"):
            if not d.get(req):
                raise ValueError(f"config missing required key: {req!r}")

        plugins = normalize_plugin_modules(d.get("plugins"))

        lens_kind = d.get("lens_kind", "completion")
        if lens_kind not in ("completion", "prompt"):
            raise ValueError(
                f"lens_kind must be 'completion' or 'prompt', got {lens_kind!r}")

        outputs = _stage_outputs(lens_kind)
        default_stages = (DEFAULT_PROMPT_STAGES if lens_kind == "prompt"
                          else DEFAULT_COMPLETION_STAGES)
        stages = list(d.get("stages") or default_stages)
        bad = [s for s in stages if s not in outputs]
        if bad:
            why = (" (win-relevance is completion-only)"
                   if lens_kind == "prompt" and "win-relevance" in bad else "")
            raise ValueError(
                f"unsupported stage(s) for lens_kind={lens_kind}: {', '.join(bad)}; "
                f"allowed: {', '.join(outputs)}{why}")

        annotations = d.get("annotations")
        if isinstance(annotations, str):
            annotations = [annotations]

        interpreter = StageConfig.parse(d.get("interpreter"))
        verifier = StageConfig.parse(d.get("verifier"))
        clusterer = StageConfig.parse(d.get("clusterer"), default="spherical-kmeans")

        role_classifier = dict(d.get("role_classifier") or {})
        unknown_role = set(role_classifier) - set(_ROLE_CLASSIFIER_KEYS)
        if unknown_role:
            raise ValueError(
                f"unknown role_classifier key(s): {', '.join(sorted(unknown_role))}; "
                f"allowed: {', '.join(_ROLE_CLASSIFIER_KEYS)}")
        if role_classifier.get("pole") not in (None, "positive"):
            raise ValueError("role_classifier.pole must be 'positive' when set")
        if "all_named" in role_classifier and not isinstance(
                role_classifier["all_named"], bool):
            raise ValueError("role_classifier.all_named must be a boolean")
        if "features" in role_classifier and not isinstance(
                role_classifier["features"], (list, tuple)):
            raise ValueError("role_classifier.features must be a list of feature IDs")

        win_relevance = dict(d.get("win_relevance") or {})
        unknown_wr = set(win_relevance) - set(_WIN_RELEVANCE_KEYS)
        if unknown_wr:
            raise ValueError(
                f"unknown win_relevance key(s): {', '.join(sorted(unknown_wr))}; "
                f"allowed: {', '.join(_WIN_RELEVANCE_KEYS)}")

        llm = LLMConfig.parse(d.get("llm"))
        name_llm = LLMConfig.parse(d["name_llm"]) if d.get("name_llm") is not None else None
        verify_llm = (
            LLMConfig.parse(d["verify_llm"])
            if d.get("verify_llm") is not None else None)
        role_llm = LLMConfig.parse(d["role_llm"]) if d.get("role_llm") is not None else None
        cluster_llm = (
            LLMConfig.parse(d["cluster_llm"])
            if d.get("cluster_llm") is not None else None)

        # Import trusted extension code only after all registry-independent config
        # validation has succeeded.
        load_plugins(plugins)
        _check_params("interpreter", interpreter)
        _check_params("verifier", verifier)
        _check_params("clusterer", clusterer, extra=_CLUSTER_CONTROL)

        return cls(
            lens_dir=d["lens_dir"], out_dir=d["out_dir"], stages=stages,
            corpus=d.get("corpus"), annotations=annotations, lens_kind=lens_kind,
            plugins=plugins, llm=llm, name_llm=name_llm, verify_llm=verify_llm,
            role_llm=role_llm, cluster_llm=cluster_llm,
            interpreter=interpreter, verifier=verifier, role_classifier=role_classifier,
            clusterer=clusterer,
            win_relevance=win_relevance)

    @classmethod
    def load(cls, path) -> "PipelineConfig":
        text = Path(path).read_text()
        if str(path).endswith((".yaml", ".yml")):
            import yaml
            raw = yaml.safe_load(text)
        else:
            raw = json.loads(text)
        return cls.from_dict(raw)


def _save(df: pd.DataFrame, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out


def _fidelity_features(names: pd.DataFrame | None, *, restrict: bool) -> list | None:
    """Feature ids to keep: fidelity-passing ones when a fidelity csv is threaded in."""
    if names is None or not restrict or "fidelity_pass" not in names.columns:
        return None
    return names.loc[
        names["fidelity_pass"].map(annotation_flag), "feature_id"
    ].astype(int).tolist()


def preflight(cfg: PipelineConfig) -> None:
    """Fail fast with a clear message before running any stage: a config typo or a missing
    lens/corpus should not surface as a mid-pipeline traceback."""
    if not (Path(cfg.lens_dir) / MANIFEST).exists():
        raise FileNotFoundError(f"no lens at {cfg.lens_dir!r} (missing {MANIFEST})")
    manifest = json.loads((Path(cfg.lens_dir) / MANIFEST).read_text())
    single = cfg.lens_kind == "completion" and manifest.get("dataset_mode") == "single"
    if "classify-role" in cfg.stages:
        if manifest.get("input_rep") != "individual":
            raise ValueError("classify-role needs an individual completion lens")
        linkage = cfg.role_classifier.get("linkage")
        if linkage and not Path(linkage).exists():
            raise FileNotFoundError(f"role-classifier linkage not found: {linkage!r}")
    if cfg.lens_kind == "prompt":
        # prompt naming/verify map prompt text from the corpus (cluster reads z_prompt only).
        if any(s in cfg.stages for s in ("name", "verify")) and not cfg.corpus:
            raise ValueError(
                "prompt-lens name/verify stages need corpus: in the config (to fetch prompt text)")
    else:
        if single and "win-relevance" in cfg.stages:
            raise ValueError(
                "win-relevance is pairwise-only; remove it from stages for a "
                "single-response lens")
        # These stages re-attach text; win-relevance additionally needs human_pref.
        text_stages = [s for s in cfg.stages
                       if s in ("name", "verify", "classify-role", "win-relevance")]
        if text_stages and not single:
            if bool(cfg.annotations) == bool(cfg.corpus):
                raise ValueError(
                    f"stage(s) {text_stages} need exactly one of corpus: or annotations: "
                    "in the config (to re-attach battle text / labels)")
            if "win-relevance" in cfg.stages and not cfg.corpus:
                raise ValueError("win-relevance needs corpus: (with human_pref) in the config")
    if cfg.corpus and not Path(cfg.corpus).exists():
        raise FileNotFoundError(f"corpus not found: {cfg.corpus!r}")


def run_pipeline(cfg: PipelineConfig, *, client=None, verbose: bool = True) -> dict:
    """Execute ``cfg.stages`` in canonical order, threading artifacts under ``out_dir``.

    ``client`` overrides the LLM client (the config's ``llm`` builds one lazily on first
    LLM stage otherwise) — tests inject a fake. Returns ``{stage: output_path}``.
    """
    preflight(cfg)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outmap = _stage_outputs(cfg.lens_kind)            # stage -> filename, per lens kind
    outputs: dict[str, Path] = {}
    _client_boxes: dict[str, object] = {}
    usage_tracker = (None if client is not None
                     else UsageTracker(out_dir / LLM_USAGE_EVENTS))

    def log(msg: str) -> None:
        if verbose:
            print(msg)

    def get_client(role: str):
        if client is not None:
            return client
        override = getattr(cfg, f"{role}_llm")
        # Keep one client per role even when the configuration is shared so every
        # request carries the correct name/verify/cluster stage in the usage ledger.
        if role not in _client_boxes:
            _client_boxes[role] = (override or cfg.llm).client(
                usage_tracker=usage_tracker,
                usage_stage="classify-role" if role == "role" else role)
        return _client_boxes[role]

    def names_csv(stage: str) -> Path:
        """Path produced by an upstream stage, or the on-disk default if it didn't run."""
        return outputs.get(stage, out_dir / outmap[stage])

    def best_names_path() -> Path | None:
        """Concepts to attach downstream: the verify fidelity csv if present (richer —
        carries fidelity_pass for filtering), else the raw name csv, else nothing."""
        for stage in ("verify", "name"):
            p = names_csv(stage)
            if p.exists():
                return p
        return None

    # Run in canonical order regardless of the order listed in the config; skip stages not
    # in this run and stages that don't apply to the lens kind (e.g. win-relevance on prompt).
    for stage in KNOWN_STAGES:
        if stage not in cfg.stages or stage not in outmap:
            continue

        if stage == "name":
            codes = LensCodes.load(cfg.lens_dir, cfg.annotations, corpus=cfg.corpus,
                                   lens_kind=cfg.lens_kind)
            mode = resolve_name_mode(cfg.interpreter.component, codes.input_rep, cfg.lens_kind)
            strategy = registry.make("interpreter", mode, **cfg.interpreter.params)
            df = strategy.name(codes, get_client("name"))
            outputs[stage] = _save(df, out_dir / outmap[stage])
            log(f"[name] {mode}: wrote {len(df)} feature names -> {outputs[stage]}")

        elif stage == "verify":
            names_path = names_csv("name")
            if not names_path.exists():
                raise FileNotFoundError(
                    f"verify needs feature names; run the 'name' stage or place a CSV at "
                    f"{names_path}")
            vcodes = VerifyCodes.load(cfg.lens_dir, cfg.annotations, corpus=cfg.corpus,
                                      lens_kind=cfg.lens_kind)
            mode = resolve_verify_mode(cfg.verifier.component, vcodes.input_rep, cfg.lens_kind)
            strategy = registry.make("verifier", mode, **cfg.verifier.params)
            df = strategy.verify(vcodes, pd.read_csv(names_path), get_client("verify"))
            outputs[stage] = _save(df, out_dir / outmap[stage])
            log(f"[verify] {mode}: {int(df['fidelity_pass'].sum())}/{len(df)} pass "
                f"-> {outputs[stage]}")

        elif stage == "classify-role":
            outputs[stage] = _run_role_classification(
                cfg, out_dir, best_names_path(), get_client("role"), log, outmap)

        elif stage == "cluster":
            outputs[stage] = _run_cluster(
                cfg, out_dir, best_names_path(), lambda: get_client("cluster"),
                log, outmap)

        elif stage == "win-relevance":
            outputs[stage] = _run_win_relevance(cfg, out_dir, best_names_path(),
                                                outputs.get("cluster"), log, outmap)

        # Keep a human-readable checkpoint current after every completed LLM stage.
        # The JSONL ledger is appended per response and therefore also survives a
        # mid-stage interruption.
        if usage_tracker is not None and _client_boxes:
            usage_tracker.write_summary(out_dir / LLM_USAGE)

    if usage_tracker is not None and _client_boxes:
        usage_path = usage_tracker.write_summary(out_dir / LLM_USAGE)
        log(f"[llm-usage] {usage_tracker.progress()} -> {usage_path}")
    return outputs


def _run_role_classification(cfg, out_dir, names_path, client, log, outmap) -> Path:
    """Classify verified individual-response concepts using prompt/response evidence."""
    from prefscope.interpret.role import classify_response_roles

    if names_path is None:
        raise FileNotFoundError(
            "classify-role needs feature names; run name/verify first or place "
            f"{FEATURE_FIDELITY} in {out_dir}")
    codes = VerifyCodes.load(
        cfg.lens_dir, cfg.annotations, corpus=cfg.corpus, lens_kind="completion"
    )
    if codes.input_rep != "individual" or codes.z_a is None:
        raise ValueError("classify-role needs an individual completion lens")

    params = dict(cfg.role_classifier)
    pole = params.pop("pole", None)
    if codes.activation_polarity == "signed" and pole != "positive":
        raise ValueError(
            "signed lenses require role_classifier.pole: positive for classify-role")
    all_named = bool(params.pop("all_named", False))
    requested = params.pop("features", None)
    linkage_path = params.pop("linkage", None)

    names = pd.read_csv(names_path)
    if not {"feature_id", "concept"} <= set(names.columns):
        raise ValueError("classify-role names need feature_id and concept columns")
    names["feature_id"] = pd.to_numeric(names["feature_id"], errors="raise").astype(int)
    if not all_named:
        if "fidelity_pass" not in names.columns:
            raise ValueError(
                "classify-role needs fidelity results by default; run verify first or set "
                "role_classifier.all_named: true")
        names = names[names["fidelity_pass"].map(annotation_flag)].copy()

    available = set(names["feature_id"].tolist())
    if requested is not None:
        requested = list(dict.fromkeys(int(feature_id) for feature_id in requested))
        missing = sorted(set(requested).difference(available))
        if missing:
            raise ValueError(
                f"classify-role requested feature IDs unavailable after filtering: {missing}")
        names = names[names["feature_id"].isin(requested)].copy()
    linkage = pd.read_csv(linkage_path) if linkage_path else None

    df = classify_response_roles(
        codes.battles,
        codes.z_a,
        codes.z_b,
        names,
        client,
        instruction_ids=codes.instruction_ids,
        features=requested,
        linkage=linkage,
        **params,
    )
    path = _save(df, out_dir / outmap["classify-role"])
    counts = df.get("behavior_scope", pd.Series(dtype=str)).value_counts().to_dict()
    log(f"[classify-role] wrote {len(df)} feature roles {counts} -> {path}")
    return path


def _run_cluster(cfg, out_dir, names_path, get_client, log, outmap) -> Path:
    from prefscope.pipeline.cluster import (
        cluster_run_diagnostics, load_cofiring_codes, name_clusters,
        summarize_clusters)

    params = dict(cfg.clusterer.params)
    cluster_on = params.pop("cluster_on", "difference")
    fidelity_only = params.pop("fidelity_only", False)
    do_name = params.pop("name_clusters", False)
    concurrency = params.pop("concurrency", 1)

    names = pd.read_csv(names_path) if names_path is not None else None
    z = load_cofiring_codes(cfg.lens_dir, lens_kind=cfg.lens_kind, cluster_on=cluster_on)
    clusterer = registry.make("clusterer", cfg.clusterer.component, **params)
    clusters = clusterer.cluster(z, features=_fidelity_features(names, restrict=fidelity_only))
    summary = summarize_clusters(clusters, names=names)
    diagnostics = cluster_run_diagnostics(clusters)

    if do_name:
        labels = name_clusters(summary, get_client(), concurrency=concurrency)
        mapped_labels = summary["cluster_id"].map(labels)
        has_label = mapped_labels.notna() & mapped_labels.astype(str).str.strip().ne("")
        summary["behavior"] = mapped_labels.where(has_label, summary["behavior"])

    out = clusters.copy()
    if names is not None and "concept" in names.columns:
        out = out.merge(names[["feature_id", "concept"]], on="feature_id", how="left")
    out = out.merge(summary[["cluster_id", "behavior"]], on="cluster_id", how="left")
    path = _save(out, out_dir / outmap["cluster"])
    _save(summary, out_dir / outmap["cluster"].replace(".csv", "_summary.csv"))
    _save(diagnostics, out_dir / outmap["cluster"].replace(".csv", "_diagnostics.csv"))
    log(f"[cluster] {cfg.clusterer.component}: {len(clusters)} features -> "
        f"{clusters['cluster_id'].nunique()} behaviors -> {path}")
    return path


def _run_win_relevance(cfg, out_dir, names_path, clusters_path, log, outmap) -> Path:
    from prefscope.analysis.grouping import resolve_group_ids
    from prefscope.interpret.io import load_lens_battles
    from prefscope.pipeline.winrelevance import win_relevance, win_relevance_logistic

    if not cfg.corpus:
        raise ValueError("win-relevance needs a corpus with human_pref "
                         "(build-corpus --keep-labels)")
    battles, z_diff, _ = load_lens_battles(cfg.lens_dir, cfg.annotations, corpus=cfg.corpus)
    if "human_pref" not in battles.columns or battles["human_pref"].isna().all():
        raise ValueError("corpus has no human_pref; rebuild with build-corpus --keep-labels")

    names = pd.read_csv(names_path) if names_path is not None else None
    restrict = not cfg.win_relevance.get("all_features", False)
    feats = _fidelity_features(names, restrict=restrict)

    hp = battles["human_pref"].to_numpy()
    wc = lambda s: battles[s].fillna("").str.split().str.len().to_numpy()  # noqa: E731
    length = wc("completion_a") - wc("completion_b")
    group_ids = resolve_group_ids(
        battles, group_col=cfg.win_relevance.get("group_col"))
    df = win_relevance(z_diff, hp, features=feats, group_ids=group_ids)
    logistic = win_relevance_logistic(
        z_diff, hp, length, features=feats, group_ids=group_ids).rename(columns={
            "n_groups": "delta_win_n_groups",
            "n_independent_groups": "delta_win_n_independent_groups",
            "estimand": "delta_win_estimand",
            "inference_test": "delta_win_inference_test",
        })
    df = df.merge(logistic, on="feature_id", how="left")
    if names is not None and "concept" in names.columns:
        df = df.merge(names[["feature_id", "concept"]], on="feature_id", how="left")
        df = df[["feature_id", "concept"]
                + [c for c in df.columns if c not in ("feature_id", "concept")]]
    df = df.sort_values("win_assoc", ascending=False).reset_index(drop=True)
    path = _save(df, out_dir / outmap["win-relevance"])
    log(f"[win-relevance] {int(df['significant'].sum())}/{len(df)} significant -> {path}")

    if clusters_path is not None and Path(clusters_path).exists():
        from prefscope.pipeline.winrelevance import cluster_win_relevance
        cdf = cluster_win_relevance(
            z_diff, hp, length, pd.read_csv(clusters_path), group_ids=group_ids)
        cout = out_dir / outmap["win-relevance"].replace(".csv", "_clusters.csv")
        _save(cdf, cout)
        log(f"[win-relevance] {len(cdf)} cluster-level rows -> {cout}")
    return path
