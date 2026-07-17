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

The lens directory and the dataset stay separate: outputs go to ``out``, never the lens.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from prefscope.data import pair_schema
from prefscope.encode.sae import SAEProjector
from prefscope.pipeline.lens_rep import get_lens_rep

_SUPPORTED_SUFFIXES = (".parquet", ".csv", ".jsonl", ".json")


def _read_manifest(lens_dir: Path) -> dict:
    mf = lens_dir / "manifest.json"
    if not mf.exists():
        raise FileNotFoundError(f"no manifest.json in lens dir {lens_dir}")
    return json.loads(mf.read_text())


def codes_from_embeddings(lens_dir, e_a, e_b=None, *, device: str = "cpu") -> dict:
    """Frozen-lens codes for already-computed embeddings.

    Reads ``input_rep`` from the lens manifest and builds a ``SAEProjector`` (which
    re-applies the lens's whitening). Battle (``e_b`` given) returns the rep's contrast
    arrays — for a battle this is exactly ``rep.output_arrays``, the same primitive
    ``build_lens`` uses. Absolute (``e_b`` None) returns the single-response code
    (individual lens only — a difference lens raises with guidance)."""
    lens_dir = Path(lens_dir)
    manifest = _read_manifest(lens_dir)
    input_rep = manifest.get("input_rep")
    if not input_rep:
        raise ValueError(f"lens manifest {lens_dir / 'manifest.json'} has no 'input_rep'")
    rep = get_lens_rep(input_rep)
    proj = SAEProjector(lens_dir, device=device)
    if e_b is None:
        return rep.single_output_arrays(proj, e_a)
    return rep.output_arrays(proj, e_a, e_b)


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
                       label_col: str | None = None, device: str = "cpu") -> dict:
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
    lens_dir, out = Path(lens_dir), Path(out)
    manifest = _read_manifest(lens_dir)

    # Fail BEFORE the (expensive) embedding pass, like build_lens' up-front rep check:
    #  1. the embedder must be the lens's own — a same-dim but different model would embed
    #     silently-wrong codes that the projector's dim check can't catch.
    expected_model = manifest.get("embed_model_id")
    actual_model = getattr(embedder, "model_id", None)
    if expected_model and actual_model is not None and actual_model != expected_model:
        raise ValueError(
            f"embedder model_id {actual_model!r} != lens embed_model_id {expected_model!r}; "
            "the dataset must be embedded with the lens's own embedder for the codes to be "
            "valid. (The CLI reads the model id from the lens manifest — build the embedder "
            "the same way.)")
    input_rep = manifest.get("input_rep")
    if not input_rep:
        raise ValueError(f"lens manifest {lens_dir / 'manifest.json'} has no 'input_rep'")

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
    battle = response2_col is not None

    #  2. absolute mode needs a per-response encoder; a contrast-only lens can't code a lone
    #     response — refuse now, before the (expensive) embedding pass.
    if not battle and not get_lens_rep(input_rep).per_side:
        raise ValueError(
            f"this lens ({input_rep!r}) has no single-response code — a contrast lens can "
            "only code an A/B pair. Provide --response-2-col (battle mode) or use an "
            "'individual' lens.")
    required = [prompt_col, response_col] + ([response2_col] if battle else [])
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"data is missing column(s) {missing}; found columns: {list(df.columns)}")
    for src in (model_col, model2_col, label_col):
        if src is not None and src not in df.columns:
            raise ValueError(f"column {src!r} not in data; found columns: {list(df.columns)}")

    # keep only rows with a non-empty prompt + response(s); row_id traces originals so a
    # gap in row_id tells the user which input rows were dropped.
    n_total = len(df)
    df = df.assign(row_id=np.arange(n_total))
    keep = _nonempty(df[prompt_col]) & _nonempty(df[response_col])
    if battle:
        keep = keep & _nonempty(df[response2_col])
    kept = df[keep].reset_index(drop=True)
    n_dropped = n_total - len(kept)
    if not len(kept):
        raise ValueError("no rows left after dropping rows with an empty prompt/response")

    prompts = kept[prompt_col].astype(str).tolist()
    e_a = np.asarray(embedder.encode(prompts, kept[response_col].astype(str).tolist()),
                     dtype=np.float32)
    e_b = None
    if battle:
        e_b = np.asarray(embedder.encode(prompts, kept[response2_col].astype(str).tolist()),
                         dtype=np.float32)

    arrays = codes_from_embeddings(lens_dir, e_a, e_b, device=device)

    out.mkdir(parents=True, exist_ok=True)
    for name, arr in arrays.items():
        np.save(out / f"{name}.npy", np.asarray(arr, dtype=np.float32))

    # emit CANONICAL pair-schema names regardless of the source column names
    meta = {"row_id": kept["row_id"], pair_schema.PROMPT: kept[prompt_col],
            pair_schema.RESPONSE_A: kept[response_col]}
    if battle:
        meta[pair_schema.RESPONSE_B] = kept[response2_col]
    for src, dst in ((model_col, pair_schema.MODEL_A), (model2_col, pair_schema.MODEL_B),
                     (label_col, pair_schema.LABEL)):
        if src is not None:
            meta[dst] = kept[src]
    meta_df = pd.DataFrame(meta)
    meta_df.to_parquet(out / "meta.parquet", index=False)
    _, has_preference = pair_schema.normalize_pair_columns(meta_df)

    stats_src = arrays.get("z_a", arrays.get("z_diff"))
    written = {
        "dataset": Path(data).name,
        "mode": "battle" if battle else "absolute",
        "n_rows": int(len(kept)),
        "n_dropped": int(n_dropped),
        "has_preference": bool(has_preference),
        "output_arrays": list(arrays),
        # provenance copied from the lens — never hardcoded
        "lens_dir": str(lens_dir),
        "lens_input_rep": manifest.get("input_rep"),
        "embed_model_id": manifest.get("embed_model_id"),
        "input_dim": manifest.get("input_dim"),
        "m_total": manifest.get("m_total"),
        "k": manifest.get("k"),
        "whiten": manifest.get("whiten"),
        "code_stats": _code_stats(np.asarray(stats_src)),
    }
    (out / "manifest.json").write_text(json.dumps(written, indent=2))
    return written
