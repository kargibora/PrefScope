"""Apply published prompt/response lenses to a mapped dataset from one config.

This module is orchestration, not a second implementation of the analyses. It composes
the existing dataset adapter, frozen-lens encoder, concept exporter, relation analysis,
paired comparison, preference association, and viewer exporter.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from prefscope.analysis.presence import annotation_flag
from prefscope.artifacts import (
    BATTLES,
    FEATURE_CALIBRATION,
    FEATURE_CLUSTERS,
    FEATURE_CONTEXT,
    FEATURE_FIDELITY,
    FEATURE_NAMES,
    FEATURE_ROLES,
    MANIFEST,
    PROMPT_FEATURE_CLUSTERS,
    PROMPT_FEATURE_FIDELITY,
    PROMPT_FEATURE_NAMES,
    SAE_MODEL,
    WIN_RELEVANCE,
)
from prefscope.core.manifest import LensManifest
from prefscope.pipeline.analyze_config import (
    AnalyzeConfig,
    LensSource,
    apply_set_overrides,
    set_config_value,
)
from prefscope.pipeline.compare import compare_encoded_responses
from prefscope.pipeline.concepts import export_concepts_from_codes
from prefscope.pipeline.elicit import run_elicitation
from prefscope.pipeline.encode_dataset import run_encode_dataset
from prefscope.pipeline.prepare_dataset import mapping_from_spec, prepare_dataset
from prefscope.pipeline.winrelevance import win_relevance, win_relevance_logistic


_ANALYZE_WORKFLOW_VERSION = 2



_LENS_CONTRACT_FILES = (
    MANIFEST,
    SAE_MODEL,
    "whiten.npz",
    FEATURE_NAMES,
    FEATURE_FIDELITY,
    FEATURE_ROLES,
    FEATURE_CALIBRATION,
    FEATURE_CONTEXT,
    FEATURE_CLUSTERS,
    PROMPT_FEATURE_NAMES,
    PROMPT_FEATURE_FIDELITY,
    PROMPT_FEATURE_CLUSTERS,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fingerprint_path(path, *, lens: bool = False) -> dict:
    """Content fingerprint for a local input that can invalidate resumed stages."""
    root = Path(path).expanduser().resolve()
    if not root.exists():
        raise ValueError(f"analysis input does not exist: {root}")
    if root.is_file():
        return {"path": str(root), "sha256": _sha256_file(root)}
    names = _LENS_CONTRACT_FILES if lens else ()
    files = [root / name for name in names if (root / name).is_file()]
    if not files:
        raise ValueError(f"local lens has no fingerprintable contract files: {root}")
    return {
        "path": str(root),
        "files": {str(path.relative_to(root)): _sha256_file(path) for path in files},
    }


def _input_fingerprints(cfg: "AnalyzeConfig") -> dict:
    from prefscope.api.hub import resolve_hf_revision, split_hf_source

    fingerprints = {}
    source = cfg.data["source"]
    if source["type"] == "local":
        fingerprints["data"] = _fingerprint_path(source["path"])
    else:
        token_env = source.get("token_env")
        resolved = resolve_hf_revision(
            source["dataset_id"], revision=source.get("revision"), repo_type="dataset",
            token=os.environ.get(token_env) if token_env else None)
        fingerprints["data"] = {
            "repo_id": source["dataset_id"],
            "requested_revision": source.get("revision"),
            "resolved_revision": resolved,
        }
    resolved_hub: dict[tuple, str] = {}
    for name, lens_source in (
        ("completion_lens", cfg.completion_lens),
        ("prompt_lens", cfg.prompt_lens),
    ):
        if lens_source is None:
            continue
        if str(lens_source.source).startswith("hf://"):
            repo_id, source_subfolder = split_hf_source(lens_source.source)
            token_env = cfg.lens_options.get("token_env")
            key = (repo_id, lens_source.revision)
            if key not in resolved_hub:
                resolved_hub[key] = resolve_hf_revision(
                    repo_id, revision=lens_source.revision,
                    token=os.environ.get(token_env) if token_env else None,
                    local_files_only=bool(cfg.lens_options.get("local_files_only", False)),
                )
            fingerprints[name] = {
                "repo_id": repo_id,
                "subfolder": lens_source.subfolder or source_subfolder,
                "requested_revision": lens_source.revision,
                "resolved_revision": resolved_hub[key],
            }
        else:
            fingerprints[name] = _fingerprint_path(lens_source.source, lens=True)
        if lens_source.annotations:
            fingerprints[f"{name}_annotations"] = _fingerprint_path(
                lens_source.annotations, lens=Path(lens_source.annotations).is_dir())
    return fingerprints


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text)
    os.replace(temporary, path)


def _reset_managed_output(out: Path, *, protected_paths=()) -> None:
    """Remove only a recognized PrefScope analysis directory for ``--fresh``."""
    if not out.exists():
        return
    if not out.is_dir():
        raise ValueError(f"analysis out_dir exists but is not a directory: {out}")
    if not any(out.iterdir()):
        return
    resolved = out.resolve()
    dangerous = {Path(resolved.anchor).resolve(), Path.home().resolve(), Path.cwd().resolve()}
    if resolved in dangerous or (resolved / ".git").exists():
        raise ValueError(f"refusing to remove unsafe analysis out_dir: {resolved}")
    for value in protected_paths:
        if value is None or str(value).startswith("hf://"):
            continue
        protected = Path(value).expanduser().resolve()
        if protected == resolved or protected.is_relative_to(resolved):
            raise ValueError(
                f"refusing to remove analysis out_dir {resolved}: it contains input {protected}")
    state_path = resolved / "analysis_state.json"
    if not state_path.is_file():
        raise ValueError(
            f"refusing to remove unrecognized non-empty directory {resolved}; "
            "choose a dedicated out_dir or remove it manually")
    try:
        state = json.loads(state_path.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"refusing to remove {resolved}: analysis_state.json is invalid") from exc
    if state.get("schema_version") != 1 or not state.get("config_sha256"):
        raise ValueError(
            f"refusing to remove {resolved}: analysis_state.json is not a PrefScope run")
    shutil.rmtree(resolved)




def _load_lens(
    spec: LensSource, cfg: AnalyzeConfig, *, resolved_revision: str | None = None,
):
    from prefscope import load_lens

    hub = spec.source.startswith("hf://")
    token_env = cfg.lens_options.get("token_env")
    token = os.environ.get(token_env) if token_env and hub else None
    lens = load_lens(
        spec.source,
        device=cfg.device,
        revision=(resolved_revision or spec.revision) if hub else None,
        cache_dir=cfg.lens_options.get("cache_dir") if hub else None,
        token=token,
        local_files_only=(cfg.lens_options.get("local_files_only", False) if hub else False),
        subfolder=spec.subfolder if hub else None,
        annotations=spec.annotations,
        embedding_cache=cfg.embedding.get("cache_dir"),
        embed_backend=cfg.embedding.get("backend", "hf"),
        embed_batch_size=cfg.embedding.get("batch_size"),
    )
    if hub and resolved_revision:
        lens.requested_revision = spec.revision
        lens.resolved_revision = resolved_revision
        lens.pretrained_revision = spec.revision
        lens.pretrained_resolved_revision = resolved_revision
    return lens


def _outcome_analysis(
    encoded_dir, features: pd.DataFrame, out, *, spec: dict,
    group_col: str | None = None,
) -> pd.DataFrame:
    from prefscope.analysis.grouping import resolve_group_ids
    from prefscope.analysis.outcomes import associate_outcomes, normalize_outcomes

    encoded_dir = Path(encoded_dir)
    meta = pd.read_parquet(encoded_dir / "meta.parquet")
    columns = list(spec["columns"])
    missing = [column for column in columns if column not in meta.columns]
    if missing:
        raise ValueError(
            f"configured outcome columns are absent from encoded metadata: {missing}; "
            "retain them with data.columns.metadata")
    code_path = encoded_dir / f"{spec['code_array']}.npy"
    if not code_path.exists():
        raise ValueError(
            f"configured outcome code array is absent: {code_path.name}")
    codes = np.load(code_path, mmap_mode="r")
    values = meta[columns]
    if len(columns) == 1 and spec["kind"] != "multi_continuous":
        values = values.iloc[:, 0]
    outcomes = normalize_outcomes(
        values, kind=spec["kind"], names=columns,
        normalization=spec["normalization"])
    groups = resolve_group_ids(meta, group_col=group_col)
    result = associate_outcomes(
        codes, outcomes, group_ids=groups, min_units=int(spec["min_units"]))
    table = result.table.merge(
        features[[column for column in ("feature_id", "concept")
                  if column in features.columns]].drop_duplicates("feature_id"),
        on="feature_id", how="left")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    sidecar = Path(out).with_name(f"{Path(out).stem}_outcomes.json")
    _atomic_write_text(sidecar, json.dumps({
        "schema_version": 1,
        "code_array": spec["code_array"],
        "outcome_kind": outcomes.kind,
        "outcome_names": list(outcomes.names),
        "normalization": outcomes.normalization,
        "center": outcomes.center.tolist(),
        "scale": outcomes.scale.tolist(),
        "association_normalization": table[[
            "outcome", "association_outcome_center", "association_outcome_scale",
        ]].drop_duplicates("outcome").to_dict(orient="records"),
        "grouped": result.grouped,
        "method": result.method,
        "estimand": result.estimand,
    }, indent=2))
    return table


def _preference_analysis(encoded_dir, features: pd.DataFrame, out, *,
                         fidelity_only: bool, group_col: str | None = None) -> pd.DataFrame:
    from prefscope.analysis.grouping import resolve_group_ids

    encoded_dir = Path(encoded_dir)
    meta = pd.read_parquet(encoded_dir / "meta.parquet")
    z_diff = np.load(encoded_dir / "z_diff.npy", mmap_mode="r")
    if "human_pref" not in meta or meta["human_pref"].isna().all():
        raise ValueError("preference analysis needs mapped human_pref labels")
    labels = pd.to_numeric(meta["human_pref"], errors="raise")
    valid = labels.notna().to_numpy()
    meta = meta.loc[valid].reset_index(drop=True)
    z = np.asarray(z_diff[valid])
    y = labels[valid].to_numpy(dtype=float)
    ids = None
    if fidelity_only:
        if "fidelity_pass" not in features:
            raise ValueError("fidelity_only preference analysis needs fidelity annotations")
        ids = features.loc[
            features["fidelity_pass"].map(annotation_flag), "feature_id"
        ].astype(int).tolist()
    group_ids = resolve_group_ids(meta, group_col=group_col)
    result = win_relevance(z, y, features=ids, group_ids=group_ids)
    lengths = (
        meta["completion_a"].fillna("").str.split().str.len().to_numpy()
        - meta["completion_b"].fillna("").str.split().str.len().to_numpy()
    )
    logistic = win_relevance_logistic(
        z, y, lengths, features=ids, group_ids=group_ids).rename(columns={
            "n_groups": "delta_win_n_groups",
            "n_independent_groups": "delta_win_n_independent_groups",
            "estimand": "delta_win_estimand",
            "inference_test": "delta_win_inference_test",
        })
    result = result.merge(logistic, on="feature_id", how="left")
    if "concept" in features:
        names = features[["feature_id", "concept"]].drop_duplicates("feature_id")
        result = result.merge(names, on="feature_id", how="left")
        result = result[["feature_id", "concept", *[
            column for column in result if column not in {"feature_id", "concept"}
        ]]]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)
    return result


def materialize_applied_lens(lens, encoded_dir, out, *, overwrite: bool = False) -> Path:
    """Transactionally combine a frozen lens with corpus-aligned codes."""
    from prefscope.pipeline.encode_dataset import (
        _manifest_digest,
        _reject_output_overlap,
        _transactional_output,
        _validate_output_destination,
    )

    encoded_dir, out = Path(encoded_dir), Path(out)
    lens_dir = Path(lens.lens_dir)
    _reject_output_overlap(
        out, (("source lens", lens_dir), ("encoded bundle", encoded_dir)))
    _validate_output_destination(out, overwrite=overwrite)
    source_manifest_path = lens_dir / MANIFEST
    if not source_manifest_path.is_file():
        raise FileNotFoundError(f"source lens has no {MANIFEST}: {lens_dir}")
    source_manifest_digest = _manifest_digest(
        json.loads(source_manifest_path.read_text()))

    code_manifest_path = encoded_dir / MANIFEST
    if not code_manifest_path.is_file():
        raise FileNotFoundError(f"encoded bundle has no {MANIFEST}: {encoded_dir}")
    code_manifest = json.loads(code_manifest_path.read_text())
    encoded_source_digest = code_manifest.get("source_lens_manifest_sha256")
    if not isinstance(encoded_source_digest, str) or not encoded_source_digest:
        raise ValueError(
            "encoded bundle manifest must bind source_lens_manifest_sha256")
    if encoded_source_digest != source_manifest_digest:
        raise ValueError(
            "encoded bundle was produced by a different source lens manifest")
    arrays = list(code_manifest.get("output_arrays") or [])
    if not arrays or len(arrays) != len(set(arrays)):
        raise ValueError("encoded bundle manifest must declare unique output arrays")
    n_rows = code_manifest.get("n_rows")
    if not isinstance(n_rows, int) or isinstance(n_rows, bool) or n_rows <= 0:
        raise ValueError("encoded bundle manifest n_rows must be a positive integer")

    meta_path = encoded_dir / "meta.parquet"
    battles_path = encoded_dir / "battles.parquet"
    if not meta_path.is_file() or not battles_path.is_file():
        raise FileNotFoundError(
            "encoded bundle needs aligned meta.parquet and battles.parquet")
    encoded_meta = pd.read_parquet(meta_path)
    encoded_battles = pd.read_parquet(battles_path)
    if len(encoded_meta) != n_rows or len(encoded_battles) != n_rows:
        raise ValueError(
            f"encoded metadata row mismatch: manifest={n_rows}, "
            f"meta={len(encoded_meta)}, battles={len(encoded_battles)}")
    if not encoded_meta.equals(encoded_battles):
        raise ValueError(
            "encoded meta.parquet and battles.parquet must be identical and row-aligned")

    def build_applied(staging: Path) -> None:
        # `Lens.save` assembles the compact inference contract in this clean staging path.
        lens.save(staging, overwrite=True, inference_only=True)
        base_files = {path.name for path in staging.iterdir()}
        source_manifest = json.loads((staging / MANIFEST).read_text())
        manifest = LensManifest.from_dict(source_manifest)
        m_total = manifest.m_total
        if not isinstance(m_total, int) or m_total <= 0:
            raise ValueError("source lens manifest m_total must be a positive integer")

        observed_shapes = {}
        for name in arrays:
            source = encoded_dir / f"{name}.npy"
            if not source.is_file():
                raise FileNotFoundError(
                    f"encoded bundle manifest declares missing array {name}.npy")
            values = np.load(source, mmap_mode="r")
            if values.ndim != 2 or values.shape != (n_rows, m_total):
                raise ValueError(
                    f"encoded array {name} must have shape {(n_rows, m_total)}, "
                    f"got {values.shape}")
            for start in range(0, n_rows, 4096):
                if not np.isfinite(values[start:start + 4096]).all():
                    raise ValueError(f"encoded array {name} contains non-finite values")
            shutil.copy2(source, staging / f"{name}.npy")
            observed_shapes[name] = list(values.shape)
        declared_shapes = code_manifest.get("array_shapes")
        if declared_shapes is not None and declared_shapes != observed_shapes:
            raise ValueError(
                f"encoded manifest array_shapes {declared_shapes} disagree with "
                f"observed {observed_shapes}")
        declared_width = code_manifest.get("m_total")
        if declared_width is not None and declared_width != m_total:
            raise ValueError(
                f"encoded manifest m_total={declared_width} disagrees with source lens "
                f"m_total={m_total}")
        shutil.copy2(meta_path, staging / "meta.parquet")
        shutil.copy2(battles_path, staging / BATTLES)

        manifest.output_arrays = arrays
        manifest.array_shapes = observed_shapes
        manifest.n_battles = n_rows
        if manifest.dataset_hash is not None:
            manifest.extra.setdefault("source_lens_dataset_hash", manifest.dataset_hash)
        if code_manifest.get("dataset_hash") is not None:
            manifest.dataset_hash = code_manifest["dataset_hash"]
        manifest.extra["source_lens_manifest_sha256"] = source_manifest_digest
        manifest.extra["artifact_scope"] = "analysis"
        manifest.extra["dataset_mode"] = (
            "paired" if "z_b" in arrays else "single")
        # Never persist local source/cache paths in a portable artifact.
        manifest.extra.pop("applied_from", None)
        manifest.extra.pop("lens_dir", None)
        manifest.require_complete()
        projector = getattr(lens, "projector", None)
        if projector is not None:
            manifest.validate_projector(projector)
        manifest.validate_arrays(staging)

        # Write the final applied manifest last, then validate the serialized contract.
        (staging / MANIFEST).write_text(json.dumps(manifest.to_dict(), indent=2))
        reloaded = LensManifest.from_dict(
            json.loads((staging / MANIFEST).read_text()), strict=True)
        if projector is not None:
            reloaded.validate_projector(projector)
        reloaded.validate_arrays(staging)
        expected_files = base_files | {
            "meta.parquet", BATTLES, *(f"{name}.npy" for name in arrays)}
        actual_files = {path.name for path in staging.iterdir()}
        if actual_files != expected_files:
            raise ValueError(
                "applied lens contains missing or undeclared artifacts: "
                f"missing={sorted(expected_files - actual_files)}, "
                f"extra={sorted(actual_files - expected_files)}")

    _transactional_output(out, build_applied, overwrite=overwrite)
    return out


def _enabled(value, applicable: bool, *, stage: str) -> bool:
    if value is False:
        return False
    if applicable:
        return True
    if value is True:
        raise ValueError(f"analysis.{stage}=true, but this dataset/lens cannot run it")
    return False


def _viewer_args(cfg: AnalyzeConfig, completion_dir: Path, prompt_dir: Path | None,
                 dataset: Path, relation_path: Path | None,
                 comparison_dir: Path | None) -> list[str]:
    v = cfg.viewer
    args = [
        "--lens-dir", str(completion_dir),
        "--analysis-dir", str(completion_dir),
        "--corpus", str(dataset),
        "--out", str(_viewer_output_dir(cfg)),
        "--examples-per-feature", str(v["examples_per_feature"]),
        "--examples-per-group", str(v["examples_per_group"]),
        "--examples-random", str(v["examples_random"]),
        "--examples-boundary", str(v["examples_boundary"]),
        "--map-sample", str(v["map_sample"]),
        "--map-sample-mode", str(v["map_sample_mode"]),
        "--coactivation-top-k", str(v["coactivation_top_k"]),
        "--coactivation-max-pairs", str(v["coactivation_max_pairs"]),
    ]
    if v["feature_map"]:
        args.append("--feature-map")
    if v["response_map"]:
        args.append("--response-map")
    if relation_path is not None:
        args.extend(["--elicitation", str(relation_path)])
    if comparison_dir is not None:
        args.extend(["--comparison-dir", str(comparison_dir)])
    if prompt_dir is not None:
        args.extend([
            "--prompt-lens", str(prompt_dir),
            "--prompt-interpret-dir", str(prompt_dir),
            "--prompt-examples-per-feature", str(v["prompt_examples_per_feature"]),
            "--prompt-examples-per-group", str(v["prompt_examples_per_group"]),
            "--prompt-examples-random", str(v["prompt_examples_random"]),
            "--prompt-examples-boundary", str(v["prompt_examples_boundary"]),
        ])
        if v["prompt_feature_map"]:
            args.append("--prompt-feature-map")
        if v["joint_examples"] and relation_path is not None:
            args.append("--joint-examples")
    return args


def _viewer_output_dir(cfg: AnalyzeConfig) -> Path:
    return Path(cfg.viewer.get("output_dir") or (Path(cfg.out_dir) / "viewer-data"))


def run_analysis(cfg: AnalyzeConfig, *, fresh: bool = False, log=print) -> dict:
    """Run the reusable frozen-lens workflow, resuming completed stages by default."""
    out = Path(cfg.out_dir)
    resolved = cfg.to_dict()
    protected = [
        cfg.data["source"].get("path"),
        cfg.completion_lens.source,
        cfg.completion_lens.annotations,
    ]
    if cfg.prompt_lens is not None:
        protected.extend([cfg.prompt_lens.source, cfg.prompt_lens.annotations])
    if fresh:
        _reset_managed_output(out, protected_paths=protected)
    out.mkdir(parents=True, exist_ok=True)
    fingerprints = _input_fingerprints(cfg)

    signature = {
        "workflow_version": _ANALYZE_WORKFLOW_VERSION,
        "config": resolved,
        "inputs": fingerprints,
    }
    digest = hashlib.sha256(
        json.dumps(signature, sort_keys=True, default=str).encode()).hexdigest()
    state_path = out / "analysis_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state.get("config_sha256") != digest:
            raise ValueError(
                f"{out} contains a run with different settings; choose another out_dir "
                "or pass --fresh")
    else:
        occupied = [path for path in out.iterdir() if path.name != "analyze.resolved.yaml"]
        if occupied:
            raise ValueError(
                f"{out} is not an empty analysis directory; choose another out_dir or "
                "pass --fresh")
        state = {
            "schema_version": 1,
            "workflow_version": _ANALYZE_WORKFLOW_VERSION,
            "config_sha256": digest,
            "input_fingerprints": fingerprints,
            "completed": [],
        }
    _atomic_write_text(
        out / "analyze.resolved.yaml", yaml.safe_dump(resolved, sort_keys=False))

    def save_state():
        _atomic_write_text(state_path, json.dumps(state, indent=2))

    # Publish the ownership marker before the first stage so an interrupted new run can
    # later be restarted safely with --fresh.
    save_state()

    def stage(name: str, expected, function):
        paths = [Path(path) for path in expected]
        if name in state["completed"] and all(path.exists() for path in paths):
            log(f"resume: {name} already complete")
            return
        log(f"[{name}]")
        function()
        missing = [str(path) for path in paths if not path.exists()]
        if missing:
            raise RuntimeError(f"stage {name} did not create expected output(s): {missing}")
        if name not in state["completed"]:
            state["completed"].append(name)
        save_state()

    dataset = out / "dataset.parquet"

    def prepare():
        source = cfg.data["source"]
        kwargs = {
            "out": dataset,
            "mapping": mapping_from_spec(cfg.data),
            "drop_empty": bool(cfg.data.get("drop_empty", True)),
            "limit": source.get("limit"),
        }
        if source["type"] == "local":
            kwargs["data"] = source["path"]
        else:
            kwargs.update(
                hf_dataset=source["dataset_id"],
                hf_name=source.get("name"),
                split=source.get("split", "train"),
                revision=source.get("revision"),
                resolved_revision=fingerprints["data"]["resolved_revision"],
                token_env=source.get("token_env"),
                streaming=bool(source.get("streaming", False)),
            )
        prepare_dataset(**kwargs)

    stage("prepare", [dataset], prepare)
    table = pd.read_parquet(dataset)
    paired = "completion_b" in table.columns
    if cfg.data["mode"] == "paired" and not paired:
        raise ValueError(
            "data.mode=paired, but the mapped dataset has no completion_b; configure "
            "data.columns.response_b")
    if cfg.data["mode"] == "single" and paired:
        raise ValueError(
            "data.mode=single, but the mapping produced completion_b; remove the second "
            "response mapping or set data.mode=paired")
    labelled = paired and "human_pref" in table and table["human_pref"].notna().any()

    completion_revision = fingerprints.get("completion_lens", {}).get("resolved_revision")
    completion = (
        _load_lens(
            cfg.completion_lens, cfg, resolved_revision=completion_revision)
        if completion_revision else _load_lens(cfg.completion_lens, cfg)
    )
    if completion.input_rep != "individual":
        raise ValueError(
            "analyze needs an individual completion lens so it can describe each "
            "response; a difference-only lens cannot describe a single answer")
    prompt_revision = fingerprints.get("prompt_lens", {}).get("resolved_revision")
    prompt = (
        _load_lens(cfg.prompt_lens, cfg, resolved_revision=prompt_revision)
        if cfg.prompt_lens and prompt_revision
        else (_load_lens(cfg.prompt_lens, cfg) if cfg.prompt_lens else None)
    )
    if prompt is not None and prompt.input_rep != "prompt":
        raise ValueError("configured prompt lens is not a prompt lens")

    metadata = [column for column in ("item_id", "language", "source")
                if column in table.columns]
    configured_group = cfg.analysis.get("group_col")
    canonical_encoded = {
        "row_id", "battle_id", "prompt", "completion_a", "completion_b",
        "model_a", "model_b", "human_pref",
    }
    if (
        configured_group and configured_group in table.columns
        and configured_group not in metadata
        and configured_group not in canonical_encoded
    ):
        metadata.append(configured_group)
    for column in cfg.data.get("columns", {}).get("metadata", []):
        if column in table.columns and column not in metadata:
            metadata.append(column)
    completion_codes = out / "codes" / "completion"
    prompt_codes = out / "codes" / "prompt"

    def encode(lens, target):
        run_encode_dataset(
            lens.lens_dir, dataset, target, embedder=lens.embedder,
            prompt_col="prompt", response_col="completion_a",
            response2_col="completion_b" if paired else None,
            model_col="model_a" if "model_a" in table else None,
            model2_col="model_b" if "model_b" in table else None,
            label_col="human_pref" if labelled else None,
            metadata_cols=metadata, device=cfg.device, overwrite=True)

    completion_code_outputs = [completion_codes / MANIFEST, completion_codes / "z_a.npy"]
    if paired:
        completion_code_outputs.extend([
            completion_codes / "z_b.npy", completion_codes / "z_diff.npy"])
    stage(
        "encode_completion", completion_code_outputs,
        lambda: encode(completion, completion_codes))
    if prompt is not None:
        stage(
            "encode_prompt", [prompt_codes / MANIFEST, prompt_codes / "z_prompt.npy"],
            lambda: encode(prompt, prompt_codes))

    concept_kwargs = dict(cfg.concepts)
    completion_concepts = out / "response_concepts.parquet"
    stage(
        "response_concepts", [completion_concepts],
        lambda: export_concepts_from_codes(
            completion, completion_codes, completion_concepts, **concept_kwargs))
    prompt_concepts = None
    if prompt is not None:
        prompt_concepts = out / "prompt_concepts.parquet"
        stage(
            "prompt_concepts", [prompt_concepts],
            lambda: export_concepts_from_codes(
                prompt, prompt_codes, prompt_concepts, **concept_kwargs))

    relation_path = None
    if _enabled(
            cfg.analysis["relationships"], prompt is not None,
            stage="relationships"):
        relation_path = out / "prompt_response_relations.csv"

        def relationships():
            edges = run_elicitation(
                completion_codes, prompt_codes,
                completion_names=completion.feature_table,
                completion_fidelity=completion.feature_table,
                prompt_names=prompt.feature_table,
                prompt_fidelity=prompt.feature_table,
                min_support=int(cfg.analysis["min_support"]),
                min_cooccur=int(cfg.analysis["min_cooccur"]),
                group_col=cfg.analysis.get("group_col"), log=log)
            edges.to_csv(relation_path, index=False)

        stage("relationships", [relation_path], relationships)

    comparison_dir = None
    if _enabled(cfg.analysis["comparison"], paired, stage="comparison"):
        comparison_dir = out / "comparison"

        def comparison():
            result = compare_encoded_responses(
                completion_codes, features=completion.feature_table,
                prompt_dir=prompt_codes if prompt is not None else None,
                prompt_features=prompt.feature_table if prompt is not None else None,
                side_a_name=cfg.analysis["side_a_name"],
                side_b_name=cfg.analysis["side_b_name"],
                presence_policy=cfg.concepts["presence_policy"],
                prompt_presence_policy=cfg.concepts["presence_policy"],
                fidelity_only=bool(cfg.concepts["fidelity_only"]),
                named_only=bool(cfg.concepts["named_only"]),
                min_context_pairs=int(cfg.analysis["min_context_pairs"]),
                group_col=cfg.analysis.get("group_col"),
                examples_per_direction=int(cfg.analysis["examples_per_direction"]))
            result.save(comparison_dir)

        stage("comparison", [comparison_dir / "comparison.json"], comparison)

    preference_path = None
    if _enabled(cfg.analysis["preference"], labelled, stage="preference"):
        preference_path = out / WIN_RELEVANCE
        stage(
            "preference", [preference_path],
            lambda: _preference_analysis(
                completion_codes, completion.feature_table, preference_path,
                fidelity_only=bool(cfg.concepts["fidelity_only"]),
                group_col=cfg.analysis.get("group_col")))

    outcomes_path = None
    outcome_spec = cfg.analysis.get("outcomes")
    if outcome_spec is not None:
        if outcome_spec["code_array"] == "z_prompt" and prompt is None:
            raise ValueError(
                "analysis.outcomes.code_array=z_prompt requires a configured prompt lens")
        outcomes_path = out / outcome_spec["output"]
        outcome_sidecar = outcomes_path.with_name(
            f"{outcomes_path.stem}_outcomes.json")
        outcome_codes = (
            prompt_codes if outcome_spec["code_array"] == "z_prompt"
            else completion_codes)
        outcome_features = (
            prompt.feature_table if outcome_spec["code_array"] == "z_prompt" and prompt
            else completion.feature_table)
        stage(
            "outcomes", [outcomes_path, outcome_sidecar],
            lambda: _outcome_analysis(
                outcome_codes, outcome_features, outcomes_path, spec=outcome_spec,
                group_col=cfg.analysis.get("group_col")))

    applied_completion = applied_prompt = None
    if cfg.viewer["enabled"]:
        applied_completion = out / "applied" / "completion"
        applied_prompt = out / "applied" / "prompt" if prompt is not None else None
        expected = [
            applied_completion / MANIFEST,
            applied_completion / "sae_model.pt",
            applied_completion / "z_a.npy",
        ]
        if paired:
            expected.extend([
                applied_completion / "z_b.npy", applied_completion / "z_diff.npy"])
        if applied_prompt is not None:
            expected.extend([
                applied_prompt / MANIFEST,
                applied_prompt / "sae_model.pt",
                applied_prompt / "z_prompt.npy",
            ])

        def materialize():
            materialize_applied_lens(
                completion, completion_codes, applied_completion, overwrite=True)
            if preference_path is not None:
                shutil.copy2(preference_path, applied_completion / WIN_RELEVANCE)
            if prompt is not None:
                materialize_applied_lens(
                    prompt, prompt_codes, applied_prompt, overwrite=True)

        stage("materialize_viewer_lenses", expected, materialize)
        viewer_out = _viewer_output_dir(cfg)

        def viewer_export():
            from prefscope.viewer_export.cli import main as export_viewer

            code = export_viewer(_viewer_args(
                cfg, applied_completion, applied_prompt, dataset,
                relation_path, comparison_dir))
            if code:
                raise RuntimeError(f"viewer exporter returned status {code}")

        stage("viewer", [viewer_out / "bundle_manifest.json"], viewer_export)

    outputs = {
        "dataset": str(dataset),
        "response_codes": str(completion_codes),
        "prompt_codes": str(prompt_codes) if prompt is not None else None,
        "response_concepts": str(completion_concepts),
        "prompt_concepts": str(prompt_concepts) if prompt_concepts else None,
        "relationships": str(relation_path) if relation_path else None,
        "comparison": str(comparison_dir) if comparison_dir else None,
        "preference": str(preference_path) if preference_path else None,
        "outcomes": str(outcomes_path) if outcomes_path else None,
        "viewer_data": str(_viewer_output_dir(cfg)) if cfg.viewer["enabled"] else None,
    }
    state["outputs"] = outputs
    save_state()
    log(f"analysis complete: {out}")
    return outputs


__all__ = [
    "AnalyzeConfig", "LensSource", "apply_set_overrides", "materialize_applied_lens",
    "run_analysis", "set_config_value",
]
