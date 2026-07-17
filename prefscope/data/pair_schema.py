"""Canonical pair schema: ONE set of column names for a preference pair.

The corpus format (``prefscope/data/corpus.py``) already fixes the on-disk names —
``prompt / model_a / model_b / completion_a / completion_b`` plus the optional
``human_pref`` label — so those ARE the canonical names; nothing is renamed there.
This module gives them symbolic constants, maps the generalized names that
``encode-dataset`` accepts on input (``response`` / ``response_2`` / ``model`` /
``model_2`` / ``label``) onto them, and hosts the one shared orientation helper so
BYO datasets flow into the same analytics as the Arena corpus.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Canonical column names (== the corpus schema; see corpus.CONTENT_COLS/OPTIONAL_COLS).
PROMPT = "prompt"
RESPONSE_A = "completion_a"
RESPONSE_B = "completion_b"
MODEL_A = "model_a"
MODEL_B = "model_b"
LABEL = "human_pref"          # y = P(A preferred): A wins -> 1.0, B -> 0.0, tie -> 0.5

# encode-dataset's generalized input names -> canonical. ``prompt`` is already canonical.
ENCODE_ALIASES = {
    "response": RESPONSE_A,
    "response_2": RESPONSE_B,
    "model": MODEL_A,
    "model_2": MODEL_B,
    "label": LABEL,
}


def normalize_pair_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """Rename encode-dataset columns to the canonical pair schema.

    Accepts a frame with either canonical names or encode-dataset names (or a mix)
    and returns ``(renamed_copy, has_preference)``. An alias is only renamed when
    its canonical twin is absent, so the call is idempotent on already-canonical
    frames and never clobbers an existing canonical column. ``has_preference`` is
    True iff the label column exists with at least one non-null value.
    """
    rename = {alias: canon for alias, canon in ENCODE_ALIASES.items()
              if alias in df.columns and canon not in df.columns}
    out = df.rename(columns=rename)          # rename always returns a copy
    has_preference = LABEL in out.columns and bool(out[LABEL].notna().any())
    return out, has_preference


def orient_by_label(y, diff, *, drop_ties: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Orient per-battle A-minus-B values toward the labeled winner.

    ``y`` is P(A preferred) per battle (1 = A wins, 0 = B wins, 0.5 = tie, NaN =
    unlabeled); ``diff`` is an (N, ...) array in A-minus-B convention (e.g. a
    lens's ``z_diff``). Rows are flipped by ``sign(y - 0.5)`` so + = the preferred
    response expresses the value more. Unlabeled rows are always dropped; ties are
    dropped when ``drop_ties`` (a tie has no winner — with ``drop_ties=False`` tie
    rows are kept but zeroed by the sign). Returns ``(oriented, keep)`` where
    ``keep`` is the (N,) bool mask of retained rows, for subsetting any row-aligned
    companions (prompt codes, battle ids, labels).
    """
    y = np.asarray(y, dtype=float)
    diff = np.asarray(diff)
    if len(y) != len(diff):
        raise ValueError(f"label/value row mismatch: {len(y)} labels vs {len(diff)} rows")
    keep = ~np.isnan(y)
    if drop_ties:
        keep &= y != 0.5
    sign = np.sign(y[keep] - 0.5)
    oriented = diff[keep] * sign.reshape((-1,) + (1,) * (diff.ndim - 1))
    return oriented, keep
