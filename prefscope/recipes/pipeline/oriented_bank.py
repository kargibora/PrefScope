"""Global oriented-code bank: every battle projected in BOTH orientations.

The per-model diagnosis (``diagnose.py``) and the predictive validation
(``validate.py``) both need a cross-model *baseline*: "how much does the rest of
the pool express this concept?". A single battle only tells us A-vs-B; to build a
pool baseline we must orient every battle around *each* of its two models.

Because the SAE is non-linear (the BatchTopK threshold), the two orientations are
NOT sign-flips of each other: ``project(e_a - e_b) != -project(e_b - e_a)``. So we
project both explicitly. For a difference lens each battle yields two rows:

    A-as-self:  z = project(e_a - e_b),  self=model_a, other=model_b, win = y
    B-as-self:  z = project(e_b - e_a),  self=model_b, other=model_a, win = 1 - y

where ``y = P(A preferred)`` (judge ``y_judge`` or human ``human_pref``) and the
numeric outcome equals that preference probability (1 win / 0.5 tie / 0 loss).

The result is a ``(2N, M)`` code matrix plus a meta frame tagged by ``self_model``
and ``orientation`` ("a"/"b"). Slicing ``self_model == X`` gives X's diagnosis
codes; ``self_model != X`` gives the pool baseline; the ``orientation == "a"``
rows reproduce the lens's natural ``z_diff`` (with ``win`` = ``y``), which is what
``win_relevance`` consumes.

Embeddings are read from a ``--dump-embeddings`` directory (``e_a.npy`` /
``e_b.npy`` / ``meta.parquet``), so building the bank is a cheap CPU-only SAE
forward pass — no re-embedding, no GPU.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

_BANK_META = "bank_meta.parquet"
_BANK_CODES = "bank_codes.npy"
_BANK_MANIFEST = "bank_manifest.json"


def build_oriented_codes(e_a, e_b, battles: pd.DataFrame, projector, *,
                         input_rep: str = "difference",
                         label_col: str = "y_judge"):
    """Project every battle in both orientations through ``projector``.

    e_a, e_b: (N, D) completion embeddings, row-aligned to ``battles``.
    battles:  must have ``model_a``, ``model_b`` and ``label_col`` (= P(A
              preferred) in {0, 0.5, 1}); ``instruction_id``/``source`` carried
              through if present. If ``completion_a``/``completion_b`` are present,
              a per-row ``length`` column (oriented word-count gap self-minus-other)
              is added so the validation LOO can length-control its refit.
    Returns ``(Z, meta)`` with ``Z`` shape ``(2N, M)`` and ``meta`` rows aligned
    to ``Z`` (first N rows are A-as-self, next N are B-as-self).
    """
    e_a = np.asarray(e_a, dtype=np.float32)
    e_b = np.asarray(e_b, dtype=np.float32)
    if e_a.shape != e_b.shape:
        raise ValueError(f"e_a {e_a.shape} and e_b {e_b.shape} must match")
    for col in ("model_a", "model_b", label_col):
        if col not in battles.columns:
            raise ValueError(f"battles missing required column {col!r}")
    n = len(battles)
    if len(e_a) != n:
        raise ValueError(f"row mismatch: {len(e_a)} embeddings vs {n} battles")

    from prefscope.pipeline.lens_rep import get_lens_rep
    z_a_self, z_b_self = get_lens_rep(input_rep).oriented_codes(projector, e_a, e_b)

    # NOTE: intentionally NOT data.pair_schema.orient_by_label — that helper keeps one
    # winner-oriented row per battle (ties dropped), while the bank needs BOTH orientations
    # (2N rows, ties kept, non-linear re-projection per side). Refactoring onto it would
    # change behavior.
    y = np.asarray(battles[label_col], dtype=float)            # P(A preferred)
    bad = ~np.isin(y, [0.0, 0.5, 1.0])
    if bad.any():
        raise ValueError(
            f"{label_col} has values outside {{0,0.5,1}}: "
            f"{sorted(set(y[bad].tolist()))[:5]}")

    extra = {c: battles[c].to_numpy() for c in ("instruction_id", "source")
             if c in battles.columns}

    # per-battle length = oriented word-count gap (self − other), matching the
    # self-minus-other code orientation of each row. Needs the completion text;
    # if it's absent we still emit a length column (all zeros) so downstream code
    # can detect length-control is unavailable.
    if {"completion_a", "completion_b"} <= set(battles.columns):
        wc = lambda s: battles[s].fillna("").str.split().str.len().to_numpy(dtype=float)  # noqa: E731
        len_a = wc("completion_a") - wc("completion_b")       # a-as-self: wc(a) − wc(b)
    else:
        len_a = np.zeros(n, dtype=float)

    a = pd.DataFrame({
        "orientation": "a",
        "self_model": battles["model_a"].to_numpy(),
        "other_model": battles["model_b"].to_numpy(),
        "win": y,
        "length": len_a,
        **extra,
    })
    b = pd.DataFrame({
        "orientation": "b",
        "self_model": battles["model_b"].to_numpy(),
        "other_model": battles["model_a"].to_numpy(),
        "win": 1.0 - y,
        "length": -len_a,                                     # b-as-self: wc(b) − wc(a)
        **extra,
    })
    meta = pd.concat([a, b], ignore_index=True)
    Z = np.vstack([z_a_self, z_b_self]).astype(np.float32)
    return Z, meta


def save_bank(out_dir, Z: np.ndarray, meta: pd.DataFrame, *,
              lens_dir=None, label_col="y_judge", input_rep="difference") -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / _BANK_CODES, np.asarray(Z, dtype=np.float32))
    meta.reset_index(drop=True).to_parquet(out_dir / _BANK_META)
    manifest = {
        "n_rows": int(Z.shape[0]),
        "n_battles": int(Z.shape[0] // 2),
        "m_total": int(Z.shape[1]),
        "n_models": int(meta["self_model"].nunique()),
        "label_col": label_col,
        "input_rep": input_rep,
        "lens_dir": str(lens_dir) if lens_dir is not None else None,
    }
    (out_dir / _BANK_MANIFEST).write_text(json.dumps(manifest, indent=2))
    return manifest


def load_bank(bank_dir):
    """Return ``(Z, meta, manifest)`` for a saved oriented-code bank."""
    bank_dir = Path(bank_dir)
    Z = np.load(bank_dir / _BANK_CODES)
    meta = pd.read_parquet(bank_dir / _BANK_META)
    manifest = json.loads((bank_dir / _BANK_MANIFEST).read_text())
    if len(meta) != len(Z):
        raise ValueError(f"bank row mismatch: {len(meta)} meta vs {len(Z)} codes")
    return Z, meta, manifest
