"""Config-driven pipeline runner — declare every component + its params in one file.

``prefscope run --config pipeline.yaml`` runs the per-lens analysis chain
(``name → verify → cluster → win-relevance``) where each stage's component is resolved
by name through the registry. The config is the declarative front-end over the same
swappable components the subcommands use, so changing a verifier or clustering algorithm
is a one-line edit:

    lens_dir: lenses/indiv_8b
    corpus:   corpora/arena.parquet        # needed by win-relevance; optional otherwise
    out_dir:  results/run1
    stages: [name, verify, cluster, win-relevance]
    llm: {backend: openai, model: deepseek/deepseek-v3.2}
    interpreter: {name: auto, n_active: 12}
    verifier:    {name: auto, n_per_bucket: 12}
    clusterer:   {name: mi-leiden, resolution: 1.2, knn: 6}

Outputs are written under ``out_dir`` with the canonical artifact names and threaded
forward (``verify`` reads ``name``'s csv; ``cluster``/``win-relevance`` read the
fidelity csv). Only ``lens_kind: completion`` is supported here — point prompt lenses at
the dedicated ``interpret``/``cluster-features`` subcommands.
"""
from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import prefscope.adapters  # noqa: F401  — registers all built-in components
from prefscope.artifacts import (
    FEATURE_CLUSTERS, FEATURE_FIDELITY, FEATURE_NAMES, MANIFEST, PROMPT_FEATURE_CLUSTERS,
    PROMPT_FEATURE_FIDELITY, PROMPT_FEATURE_NAMES, WIN_RELEVANCE)
from prefscope.core import registry
from prefscope.interpret.llm import DEFAULT_API_BASE, DEFAULT_MODEL, LLMClient
from prefscope.interpret.strategy import (
    LensCodes, VerifyCodes, resolve_name_mode, resolve_verify_mode)

# Stage -> output filename, per lens kind. A completion lens runs the full chain; a prompt
# lens runs name/verify/cluster over z_prompt (win-relevance is a completion-only notion —
# it scores which response features humans reward).
_COMPLETION_OUTPUTS = {
    "name": FEATURE_NAMES,
    "verify": FEATURE_FIDELITY,
    "cluster": FEATURE_CLUSTERS,
    "win-relevance": WIN_RELEVANCE,
}
_PROMPT_OUTPUTS = {
    "name": PROMPT_FEATURE_NAMES,
    "verify": PROMPT_FEATURE_FIDELITY,
    "cluster": PROMPT_FEATURE_CLUSTERS,
}
KNOWN_STAGES = ("name", "verify", "cluster", "win-relevance")


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

    def client(self) -> LLMClient:
        return LLMClient(backend=self.backend, model=self.model,
                         api_base=self.api_base, api_key_env=self.api_key_env)


# Cluster-stage keys that steer the runner, not the clusterer constructor (popped before make).
_CLUSTER_CONTROL = ("cluster_on", "fidelity_only", "name_clusters", "concurrency")
_WIN_RELEVANCE_KEYS = ("all_features",)


def _accepted_params(kind: str, name: str) -> set | None:
    """The keyword params a component's ``__init__`` declares — its config contract.

    Validated against this so a misspelled or wrong-component param (e.g. ``n_clusters``
    on ``mi-leiden``, which only the k-means clusterers take) is rejected up front rather
    than silently swallowed by a ``**kwargs`` catch-all. ``auto`` shares the base
    ``__init__`` across the concrete strategies, so any registered name gives the same set.
    Returns ``None`` when the name can't be resolved — ``registry.make`` then raises the
    friendly "no such component" error instead."""
    resolved = name
    if name == "auto":
        avail = registry.available(kind)
        if not avail:
            return None
        resolved = avail[0]
    try:
        cls = registry.get(kind, resolved)
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
             "llm", "name_llm", "verify_llm", "cluster_llm",
             "interpreter", "verifier", "clusterer", "win_relevance"}


@dataclass
class PipelineConfig:
    """Typed, validated view of a pipeline config file."""
    lens_dir: str
    out_dir: str
    stages: list = field(default_factory=lambda: list(KNOWN_STAGES))
    corpus: str | None = None
    annotations: list | None = None
    lens_kind: str = "completion"
    llm: LLMConfig = field(default_factory=LLMConfig)
    name_llm: LLMConfig | None = None
    verify_llm: LLMConfig | None = None
    cluster_llm: LLMConfig | None = None
    interpreter: StageConfig = field(default_factory=StageConfig)
    verifier: StageConfig = field(default_factory=StageConfig)
    clusterer: StageConfig = field(default_factory=lambda: StageConfig("spherical-kmeans"))
    win_relevance: dict = field(default_factory=dict)

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

        lens_kind = d.get("lens_kind", "completion")
        if lens_kind not in ("completion", "prompt"):
            raise ValueError(
                f"lens_kind must be 'completion' or 'prompt', got {lens_kind!r}")

        outputs = _stage_outputs(lens_kind)
        stages = list(d.get("stages") or outputs)
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
        _check_params("interpreter", interpreter)
        _check_params("verifier", verifier)
        _check_params("clusterer", clusterer, extra=_CLUSTER_CONTROL)

        win_relevance = dict(d.get("win_relevance") or {})
        unknown_wr = set(win_relevance) - set(_WIN_RELEVANCE_KEYS)
        if unknown_wr:
            raise ValueError(
                f"unknown win_relevance key(s): {', '.join(sorted(unknown_wr))}; "
                f"allowed: {', '.join(_WIN_RELEVANCE_KEYS)}")

        return cls(
            lens_dir=d["lens_dir"], out_dir=d["out_dir"], stages=stages,
            corpus=d.get("corpus"), annotations=annotations, lens_kind=lens_kind,
            llm=LLMConfig.parse(d.get("llm")),
            name_llm=(LLMConfig.parse(d["name_llm"]) if d.get("name_llm") is not None
                      else None),
            verify_llm=(LLMConfig.parse(d["verify_llm"]) if d.get("verify_llm") is not None
                        else None),
            cluster_llm=(LLMConfig.parse(d["cluster_llm"]) if d.get("cluster_llm") is not None
                         else None),
            interpreter=interpreter, verifier=verifier, clusterer=clusterer,
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
    return names.loc[names["fidelity_pass"].astype(bool), "feature_id"].astype(int).tolist()


def preflight(cfg: PipelineConfig) -> None:
    """Fail fast with a clear message before running any stage: a config typo or a missing
    lens/corpus should not surface as a mid-pipeline traceback."""
    if not (Path(cfg.lens_dir) / MANIFEST).exists():
        raise FileNotFoundError(f"no lens at {cfg.lens_dir!r} (missing {MANIFEST})")
    manifest = json.loads((Path(cfg.lens_dir) / MANIFEST).read_text())
    single = cfg.lens_kind == "completion" and manifest.get("dataset_mode") == "single"
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
        # name/verify re-attach text; win-relevance needs human_pref — all need a battle source.
        text_stages = [s for s in cfg.stages if s in ("name", "verify", "win-relevance")]
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

    def log(msg: str) -> None:
        if verbose:
            print(msg)

    def get_client(role: str):
        if client is not None:
            return client
        override = getattr(cfg, f"{role}_llm")
        key = role if override is not None else "shared"
        if key not in _client_boxes:
            _client_boxes[key] = (override or cfg.llm).client()
        return _client_boxes[key]

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

        elif stage == "cluster":
            outputs[stage] = _run_cluster(
                cfg, out_dir, best_names_path(), lambda: get_client("cluster"),
                log, outmap)

        elif stage == "win-relevance":
            outputs[stage] = _run_win_relevance(cfg, out_dir, best_names_path(),
                                                outputs.get("cluster"), log, outmap)

    return outputs


def _run_cluster(cfg, out_dir, names_path, get_client, log, outmap) -> Path:
    from prefscope.pipeline.cluster import (
        load_cofiring_codes, name_clusters, summarize_clusters)

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

    if do_name:
        labels = name_clusters(summary, get_client(), concurrency=concurrency)
        summary["behavior"] = summary["cluster_id"].map(labels).where(
            summary["cluster_id"].map(labels).astype(bool), summary["behavior"])

    out = clusters.copy()
    if names is not None and "concept" in names.columns:
        out = out.merge(names[["feature_id", "concept"]], on="feature_id", how="left")
    out = out.merge(summary[["cluster_id", "behavior"]], on="cluster_id", how="left")
    path = _save(out, out_dir / outmap["cluster"])
    _save(summary, out_dir / outmap["cluster"].replace(".csv", "_summary.csv"))
    log(f"[cluster] {cfg.clusterer.component}: {len(clusters)} features -> "
        f"{clusters['cluster_id'].nunique()} behaviors -> {path}")
    return path


def _run_win_relevance(cfg, out_dir, names_path, clusters_path, log, outmap) -> Path:
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
    df = win_relevance(z_diff, hp, features=feats)
    df = df.merge(win_relevance_logistic(z_diff, hp, length, features=feats),
                  on="feature_id", how="left")
    if names is not None and "concept" in names.columns:
        df = df.merge(names[["feature_id", "concept"]], on="feature_id", how="left")
        df = df[["feature_id", "concept"]
                + [c for c in df.columns if c not in ("feature_id", "concept")]]
    df = df.sort_values("win_assoc", ascending=False).reset_index(drop=True)
    path = _save(df, out_dir / outmap["win-relevance"])
    log(f"[win-relevance] {int(df['significant'].sum())}/{len(df)} significant -> {path}")

    if clusters_path is not None and Path(clusters_path).exists():
        from prefscope.pipeline.winrelevance import cluster_win_relevance
        cdf = cluster_win_relevance(z_diff, hp, length, pd.read_csv(clusters_path))
        cout = out_dir / outmap["win-relevance"].replace(".csv", "_clusters.csv")
        _save(cdf, cout)
        log(f"[win-relevance] {len(cdf)} cluster-level rows -> {cout}")
    return path
