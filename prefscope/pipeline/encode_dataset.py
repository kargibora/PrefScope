"""Encode an arbitrary ``(prompt, response[, response_2])`` dataset into sparse codes
with an already-trained lens.

This is the inference counterpart to ``build-lens``: it never trains and never writes into
the lens directory. It reproduces the lens's exact embedding (the same embedder the lens
manifest records) and projection (``SAEProjector``, which re-applies the training-time
whitening), so the codes are consistent with the codes the lens saved at build time.

Two shapes:
  - **absolute** ``(prompt, response)``            -> per-response codes (individual lens only)
  - **battle**   ``(prompt, response, response_2)`` -> contrast codes ``z_diff`` (plus
    ``z_a``/``z_b`` for an individual lens)
  - **prompt**   ``(prompt)``                       -> prompt codes ``z_prompt``

The lens directory and the dataset stay separate: outputs go to ``out``, never the lens.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from prefscope.data import pair_schema
from prefscope.core.manifest import LensManifest
from prefscope.pipeline.lens_rep import get_lens_rep

_SUPPORTED_SUFFIXES = (".parquet", ".csv", ".jsonl", ".json")
_EMBEDDER_PROVENANCE_FIELDS = (
    "embed_model_id", "embed_model_revision", "max_tokens", "embed_instruction",
    "pooling", "normalization", "dtype", "backend",
)


def SAEProjector(*args, **kwargs):
    """Construct the heavy projector lazily; retained as a patchable boundary."""
    from prefscope.encode.sae import SAEProjector as projector_class

    return projector_class(*args, **kwargs)


def _remove_output_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def _paths_overlap(left, right) -> bool:
    """Whether two resolved paths are equal or one contains the other."""
    left = Path(left).expanduser().resolve()
    right = Path(right).expanduser().resolve()
    return left == right or left in right.parents or right in left.parents


def _reject_output_overlap(out, sources) -> None:
    for label, source in sources:
        if _paths_overlap(out, source):
            raise ValueError(
                f"output path {Path(out)} overlaps {label} {Path(source)}; "
                "source and output paths must be disjoint")


def _validate_output_destination(out, *, overwrite: bool) -> Path:
    destination = Path(out)
    if destination.exists() and not destination.is_dir():
        raise FileExistsError(
            f"output destination exists and is not a directory: {destination}")
    if destination.exists() and not overwrite and any(destination.iterdir()):
        raise FileExistsError(
            f"output destination is not empty: {destination}; pass overwrite=True")
    return destination


def _transactional_output(out, builder, *, overwrite: bool = True):
    """Build a clean sibling directory and publish it with rollback."""
    destination = _validate_output_destination(out, overwrite=overwrite)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{destination.name}.tmp-", dir=destination.parent))
    backup = destination.parent / f".{destination.name}.bak-{uuid.uuid4().hex}"
    try:
        result = builder(staging)
        had_destination = destination.exists()
        if had_destination:
            os.replace(destination, backup)
        try:
            os.replace(staging, destination)
        except BaseException:
            if had_destination and backup.exists():
                os.replace(backup, destination)
            raise
        if backup.exists():
            _remove_output_path(backup)
        return result
    finally:
        if staging.exists() or staging.is_symlink():
            _remove_output_path(staging)


def _validate_encoded_bundle(bundle: Path, expected_manifest: dict) -> None:
    """Validate the exact staged bundle that downstream consumers will read."""
    manifest_path = bundle / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("encoded bundle did not write manifest.json")
    manifest = json.loads(manifest_path.read_text())
    if manifest != expected_manifest:
        raise ValueError("serialized encoded manifest differs from the validated manifest")

    arrays = list(manifest.get("output_arrays") or [])
    if not arrays or len(arrays) != len(set(arrays)):
        raise ValueError("encoded manifest must declare unique output arrays")
    expected_files = {
        "manifest.json", "meta.parquet", "battles.parquet",
        *(f"{name}.npy" for name in arrays),
    }
    actual_files = {path.name for path in bundle.iterdir()}
    if actual_files != expected_files:
        raise ValueError(
            "encoded bundle contains missing or undeclared artifacts: "
            f"missing={sorted(expected_files - actual_files)}, "
            f"extra={sorted(actual_files - expected_files)}")

    meta = pd.read_parquet(bundle / "meta.parquet")
    battles = pd.read_parquet(bundle / "battles.parquet")
    n_rows = manifest.get("n_rows")
    m_total = manifest.get("m_total")
    if not isinstance(n_rows, int) or isinstance(n_rows, bool) or n_rows <= 0:
        raise ValueError("encoded manifest n_rows must be a positive integer")
    if not isinstance(m_total, int) or isinstance(m_total, bool) or m_total <= 0:
        raise ValueError("encoded manifest m_total must be a positive integer")
    if len(meta) != n_rows or len(battles) != n_rows:
        raise ValueError(
            f"encoded metadata row mismatch: manifest={n_rows}, meta={len(meta)}, "
            f"battles={len(battles)}")
    if not meta.equals(battles):
        raise ValueError("meta.parquet and battles.parquet must be identical and row-aligned")

    observed_shapes = {}
    for name in arrays:
        path = bundle / f"{name}.npy"
        if not path.is_file():
            raise ValueError(f"encoded manifest declares missing array {name}.npy")
        array = np.load(path, mmap_mode="r")
        if array.ndim != 2 or array.shape != (n_rows, m_total):
            raise ValueError(
                f"encoded array {name} must have shape {(n_rows, m_total)}, "
                f"got {array.shape}")
        for start in range(0, n_rows, 4096):
            if not np.isfinite(array[start:start + 4096]).all():
                raise ValueError(f"encoded array {name} contains non-finite values")
        observed_shapes[name] = list(array.shape)
    if manifest.get("array_shapes") != observed_shapes:
        raise ValueError(
            f"encoded manifest array_shapes {manifest.get('array_shapes')} disagree "
            f"with observed {observed_shapes}")
    dataset_hash = manifest.get("dataset_hash")
    try:
        valid_hash = len(dataset_hash) == 64 and int(dataset_hash, 16) >= 0
    except (TypeError, ValueError):
        valid_hash = False
    if not valid_hash:
        raise ValueError("encoded manifest dataset_hash must be a SHA-256 hex digest")


def _validated_matrix(values, name: str, *, n_rows: int | None = None,
                      n_cols: int | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2-D matrix; got shape {array.shape}")
    if array.shape[1] == 0:
        raise ValueError(f"{name} must have at least one feature column")
    if n_rows is not None and array.shape[0] != n_rows:
        raise ValueError(
            f"{name} has {array.shape[0]} rows but expected {n_rows}; encoded rows, "
            "projected rows, and metadata must align 1:1")
    if n_cols is not None and array.shape[1] != n_cols:
        raise ValueError(f"{name} has feature dim {array.shape[1]} but expected {n_cols}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _validate_embedder_provenance(embedder, manifest: dict, *, prompt: bool) -> None:
    """Require every known lens embedding setting to match before embedding."""
    provenance_fn = getattr(embedder, "provenance", None)
    if not callable(provenance_fn):
        raise ValueError(
            "embedder must expose provenance(prompt=...) so its full embedding "
            "contract can be checked against the lens manifest")
    try:
        actual = dict(provenance_fn(prompt=prompt))
    except TypeError as exc:
        raise ValueError(
            "embedder provenance must accept prompt=... and return the full embedding "
            "contract") from exc
    mismatches = []
    for field in _EMBEDDER_PROVENANCE_FIELDS:
        expected = manifest.get(field)
        if expected is not None and actual.get(field) != expected:
            mismatches.append(
                f"{field}: embedder={actual.get(field)!r}, lens={expected!r}")
    if mismatches:
        raise ValueError(
            "embedder provenance does not match the lens manifest: "
            + "; ".join(mismatches))


def _validated_outputs(arrays: dict, *, n_rows: int, m_total: int) -> dict:
    if not arrays:
        raise ValueError("projection returned no output arrays")
    return {
        name: _validated_matrix(array, name, n_rows=n_rows, n_cols=m_total)
        for name, array in arrays.items()
    }


def _read_manifest(lens_dir: Path) -> dict:
    mf = lens_dir / "manifest.json"
    if not mf.exists():
        raise FileNotFoundError(f"no manifest.json in lens dir {lens_dir}")
    return json.loads(mf.read_text())


def _manifest_digest(manifest: dict) -> str:
    payload = json.dumps(
        manifest, sort_keys=True, ensure_ascii=False,
        separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def codes_from_embeddings(lens_dir, e_a, e_b=None, *, device: str = "cpu") -> dict:
    """Frozen-lens codes for already-computed embeddings.

    Reads ``input_rep`` from the lens manifest and builds a ``SAEProjector`` (which
    re-applies the lens's whitening). Battle (``e_b`` given) returns the rep's contrast
    arrays — for a battle this is exactly ``rep.output_arrays``, the same primitive
    ``build_lens`` uses. Absolute (``e_b`` None) returns the single-response code
    (individual lens only — a difference lens raises with guidance)."""
    lens_dir = Path(lens_dir)
    manifest = _read_manifest(lens_dir)
    typed = LensManifest.from_dict(manifest)
    rep = get_lens_rep(typed.input_rep)
    proj = SAEProjector(lens_dir, device=device)
    typed.validate_projector(proj)
    input_dim = typed.input_dim or getattr(proj, "input_dim", None)
    m_total = typed.m_total or getattr(proj, "m_total", None)
    if not isinstance(input_dim, int) or input_dim <= 0:
        raise ValueError("lens/projector does not declare a valid input_dim")
    if not isinstance(m_total, int) or m_total <= 0:
        raise ValueError("lens/projector does not declare a valid m_total")
    e_a = _validated_matrix(e_a, "e_a", n_cols=input_dim)
    if e_b is None:
        arrays = rep.single_output_arrays(proj, e_a)
    else:
        e_b = _validated_matrix(
            e_b, "e_b", n_rows=e_a.shape[0], n_cols=input_dim)
        arrays = rep.output_arrays(proj, e_a, e_b)
    return _validated_outputs(arrays, n_rows=e_a.shape[0], m_total=m_total)


def _load_table(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf == ".parquet":
        return pd.read_parquet(path)
    if suf == ".csv":
        return pd.read_csv(path)
    if suf in (".jsonl", ".json"):
        return pd.read_json(path, lines=(suf == ".jsonl"))
    raise ValueError(f"unsupported data format {suf!r}; use one of {list(_SUPPORTED_SUFFIXES)}")


def _nonempty(s: pd.Series) -> pd.Series:
    return s.notna() & (s.astype(str).str.strip() != "")


def _code_stats(z: np.ndarray) -> dict:
    nz = z != 0
    return {"n_rows": int(z.shape[0]), "m_total": int(z.shape[1]),
            "mean_l0": round(float(nz.sum(axis=1).mean()), 3),
            "n_all_zero_rows": int((~nz.any(axis=1)).sum())}


def run_encode_dataset(lens_dir, data, out, *, embedder,
                       prompt_col: str = "prompt", response_col: str = "response",
                       response2_col: str | None = None,
                       model_col: str | None = None, model2_col: str | None = None,
                       label_col: str | None = None, metadata_cols=(),
                       device: str = "cpu", overwrite: bool = False) -> dict:
    """Encode a dataset file into a codes bundle. Returns the written manifest.

    ``embedder`` is any object with ``.encode(prompts, completions) -> (N, D) array``; the
    caller builds it with the lens manifest's ``embed_model_id`` (see the CLI). Writes
    ``z_*.npy`` + ``meta.parquet`` (aligned 1:1 with the code rows; ``row_id`` traces each
    back to the input file) + ``manifest.json`` (provenance copied from the lens).

    ``meta.parquet`` uses the CANONICAL pair-schema names (``prompt`` / ``completion_a`` /
    ``completion_b`` / ``model_a`` / ``model_b`` / ``human_pref``), whatever the source
    columns were called — so the downstream analytics (orientation, win-relevance, report)
    work on a BYO dataset exactly as on the Arena corpus. On input, a default column name
    that is absent falls back to its canonical twin (``response`` -> ``completion_a`` etc.),
    so an already-canonical dataset needs no flags."""
    lens_dir, data, out = Path(lens_dir), Path(data), Path(out)
    _reject_output_overlap(out, (("lens directory", lens_dir), ("input data", data)))
    _validate_output_destination(out, overwrite=overwrite)
    manifest = _read_manifest(lens_dir)
    typed = LensManifest.from_dict(manifest)
    input_rep = typed.input_rep

    df = _load_table(Path(data))
    # Column resolution, two paths:
    #  explicit flag  — keep it, falling back to its canonical twin when absent (so
    #                   --response-2-col response_2 also matches completion_b);
    #  no flag (None) — probe the generic encode-dataset name AND the canonical corpus
    #                   name, so both a (response_2/model/label) dataset and an
    #                   already-canonical one encode with NO flags. Battle mode is then
    #                   detected from the data, not from whether a flag was typed.
    def _resolve(col, *defaults):
        if col is not None:
            alias = pair_schema.ENCODE_ALIASES.get(col)
            if col not in df.columns and alias in df.columns:
                return alias
            return col
        return next((c for c in defaults if c in df.columns), None)
    response_col = _resolve(response_col, "response", pair_schema.RESPONSE_A)
    response2_col = _resolve(response2_col, "response_2", pair_schema.RESPONSE_B)
    model_col = _resolve(model_col, "model", pair_schema.MODEL_A)
    model2_col = _resolve(model2_col, "model_2", pair_schema.MODEL_B)
    label_col = _resolve(label_col, "label", pair_schema.LABEL)
    prompt_only = input_rep == "prompt"
    battle = response2_col is not None and not prompt_only

    # A same-dimension model with any different preprocessing setting produces silently
    # invalid codes. Check every provenance field that the lens knows before embedding.
    _validate_embedder_provenance(embedder, manifest, prompt=prompt_only)

    # Absolute mode needs a per-response encoder; a contrast-only lens can't code a lone
    #     response — refuse now, before the (expensive) embedding pass.
    if not prompt_only and not battle and not get_lens_rep(input_rep).per_side:
        raise ValueError(
            f"this lens ({input_rep!r}) has no single-response code — a contrast lens can "
            "only code an A/B pair. Provide --response-2-col (battle mode) or use an "
            "'individual' lens.")
    required = [prompt_col] if prompt_only else (
        [prompt_col, response_col] + ([response2_col] if battle else []))
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"data is missing column(s) {missing}; found columns: {list(df.columns)}")
    metadata_cols = list(dict.fromkeys(str(column) for column in metadata_cols))
    reserved_meta = {
        "row_id", "battle_id", pair_schema.PROMPT, pair_schema.RESPONSE_A,
        pair_schema.RESPONSE_B, pair_schema.MODEL_A, pair_schema.MODEL_B,
        pair_schema.LABEL,
    }
    conflicting = sorted(reserved_meta.intersection(metadata_cols))
    if conflicting:
        raise ValueError(
            f"metadata columns use reserved output name(s) {conflicting}; canonical "
            "fields are copied automatically")
    for src in (model_col, model2_col, label_col, *metadata_cols):
        if src is not None and src not in df.columns:
            raise ValueError(f"column {src!r} not in data; found columns: {list(df.columns)}")

    # keep only rows with a non-empty prompt + response(s); row_id traces originals so a
    # gap in row_id tells the user which input rows were dropped.
    n_total = len(df)
    df = df.assign(row_id=np.arange(n_total))
    keep = _nonempty(df[prompt_col])
    if not prompt_only:
        keep &= _nonempty(df[response_col])
    if battle:
        keep = keep & _nonempty(df[response2_col])
    kept = df[keep].reset_index(drop=True)
    n_dropped = n_total - len(kept)
    if not len(kept):
        raise ValueError("no rows left after dropping rows with an empty prompt/response")

    prompts = kept[prompt_col].astype(str).tolist()
    if prompt_only:
        raw_a = embedder.encode_prompts(prompts)
    else:
        raw_a = embedder.encode(prompts, kept[response_col].astype(str).tolist())
    e_a = _validated_matrix(
        raw_a, "e_a", n_rows=len(kept), n_cols=typed.input_dim)
    e_b = None
    if battle:
        raw_b = embedder.encode(
            prompts, kept[response2_col].astype(str).tolist())
        e_b = _validated_matrix(
            raw_b, "e_b", n_rows=len(kept), n_cols=e_a.shape[1])

    if prompt_only:
        projector = SAEProjector(lens_dir, device=device)
        typed.validate_projector(projector)
        input_dim = typed.input_dim or getattr(projector, "input_dim", None)
        m_total = typed.m_total or getattr(projector, "m_total", None)
        if not isinstance(input_dim, int) or input_dim <= 0:
            raise ValueError("lens/projector does not declare a valid input_dim")
        if not isinstance(m_total, int) or m_total <= 0:
            raise ValueError("lens/projector does not declare a valid m_total")
        e_a = _validated_matrix(
            e_a, "e_prompt", n_rows=len(kept), n_cols=input_dim)
        arrays = _validated_outputs(
            {"z_prompt": projector.project(e_a)},
            n_rows=len(kept), m_total=m_total)
    else:
        arrays = codes_from_embeddings(lens_dir, e_a, e_b, device=device)

    # Emit CANONICAL pair-schema names regardless of the source column names.
    # Files are written only after the complete in-memory result has validated.
    meta = {"row_id": kept["row_id"], pair_schema.PROMPT: kept[prompt_col]}
    # A stable cross-bundle id lets prompt and completion encodings of the same
    # prepared dataset be aligned by `elicit`, even when source rows were filtered.
    if "battle_id" in kept.columns:
        meta["battle_id"] = kept["battle_id"].astype(str)
    elif "item_id" in kept.columns:
        meta["battle_id"] = kept["item_id"].astype(str)
    else:
        meta["battle_id"] = kept["row_id"].astype(str)
    if not prompt_only:
        meta[pair_schema.RESPONSE_A] = kept[response_col]
    if battle:
        meta[pair_schema.RESPONSE_B] = kept[response2_col]
    elif prompt_only:
        # Preserve canonical response text when present. Prompt encoding ignores it,
        # but keeping the aligned metadata makes the bundle inspectable.
        for column in (pair_schema.RESPONSE_A, pair_schema.RESPONSE_B):
            if column in kept.columns:
                meta[column] = kept[column]
    for src, dst in ((model_col, pair_schema.MODEL_A), (model2_col, pair_schema.MODEL_B),
                     (label_col, pair_schema.LABEL)):
        if src is not None:
            meta[dst] = kept[src]
    for column in metadata_cols:
        meta[column] = kept[column]
    meta_df = pd.DataFrame(meta)
    if len(meta_df) != len(kept):
        raise ValueError(
            f"output metadata has {len(meta_df)} rows but codes have {len(kept)}; "
            "codes and metadata must align 1:1")
    for name, array in arrays.items():
        if array.shape[0] != len(meta_df):
            raise ValueError(
                f"{name} has {array.shape[0]} rows but output metadata has "
                f"{len(meta_df)}")
    _, has_preference = pair_schema.normalize_pair_columns(meta_df)

    stats_src = arrays.get("z_a", arrays.get("z_diff", arrays.get("z_prompt")))
    array_shapes = {name: list(array.shape) for name, array in arrays.items()}
    from prefscope.core.provenance import ordered_dataset_hash

    dataset_hash = ordered_dataset_hash(meta_df, arrays)
    written = {
        "schema_version": 1,
        "dataset": data.name,
        "mode": "prompt" if prompt_only else ("battle" if battle else "absolute"),
        "n_rows": int(len(kept)),
        "n_dropped": int(n_dropped),
        "has_preference": bool(has_preference),
        "output_arrays": list(arrays),
        "array_shapes": array_shapes,
        "dataset_hash": dataset_hash,
        # Portable provenance copied from the lens — never a local source path.
        "lens_input_rep": input_rep,
        "source_lens_manifest_sha256": _manifest_digest(manifest),
        "source_lens_dataset_hash": manifest.get("dataset_hash"),
        "embed_model_id": typed.embed_model_id,
        "input_dim": int(e_a.shape[1]),
        "m_total": int(next(iter(arrays.values())).shape[1]),
        "k": typed.k,
        "activation_polarity": typed.activation_polarity,
        "code_semantics": typed.code_semantics,
        "selection_rule": typed.selection_rule,
        "whiten": typed.whiten,
        "code_stats": _code_stats(np.asarray(stats_src)),
    }

    def write_bundle(staging: Path) -> dict:
        for name, array in arrays.items():
            np.save(staging / f"{name}.npy", np.asarray(array, dtype=np.float32))
        meta_df.to_parquet(staging / "meta.parquet", index=False)
        # Existing two-lens analyses use the historical battles.parquet filename.
        meta_df.to_parquet(staging / "battles.parquet", index=False)
        # The manifest is the commit record inside the staged bundle and is written last.
        (staging / "manifest.json").write_text(json.dumps(written, indent=2))
        _validate_encoded_bundle(staging, written)
        return written

    return _transactional_output(out, write_bundle, overwrite=overwrite)
