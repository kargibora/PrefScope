"""Generic analyses over (codes, meta) — independent of how the codes were made.

``codes`` are signed self-minus-other SAE codes (N, M): z>0 means the "self"
response expresses the feature more than its opponent. ``meta`` must carry a
``pref`` column = P(self preferred) per row, aligned to ``codes`` rows. These
wrap the generic cores in ``pipeline`` so any (codes, meta) — from a LoadedLens
or supplied directly — can be diagnosed without a file path or the OpenJury format.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from prefscope.pipeline.diagnose import diagnose_features
from prefscope.pipeline.winrelevance import win_relevance


def _win_vector(meta: pd.DataFrame, codes: np.ndarray) -> np.ndarray:
    if "pref" not in meta.columns:
        raise ValueError("meta must have a 'pref' column (P(self preferred) per row)")
    if len(meta) != len(codes):
        raise ValueError(f"codes ({len(codes)}) and meta ({len(meta)}) length mismatch")
    return np.asarray(meta["pref"], dtype=float)


def _fidelity_feats(names, fidelity_only):
    if names is not None and fidelity_only and "fidelity_pass" in names.columns:
        return names.loc[names["fidelity_pass"].astype(bool),
                         "feature_id"].astype(int).tolist()
    return None


def _attach_names(df: pd.DataFrame, names) -> pd.DataFrame:
    if names is None or "feature_id" not in getattr(names, "columns", []):
        return df
    keep = [c for c in ("feature_id", "concept", "concept_abbrev") if c in names.columns]
    df = df.merge(names[keep], on="feature_id", how="left")
    front = [c for c in ("feature_id", "concept", "concept_abbrev") if c in df.columns]
    return df[front + [c for c in df.columns if c not in front]]


def diagnose(codes, meta, *, names=None, fidelity_only: bool = False) -> pd.DataFrame:
    """Per-feature over/under-expression + outcome association, sorted by
    net_direction (descending). Optionally attach concept names and restrict to
    fidelity-passing features."""
    codes = np.asarray(codes, dtype=np.float32)
    win = _win_vector(meta, codes)
    feats = _fidelity_feats(names, fidelity_only)
    df = diagnose_features(codes, win, features=feats)
    df = _attach_names(df, names)
    return df.sort_values("net_direction", ascending=False).reset_index(drop=True)


def feature_preference_relevance(codes, meta, *, names=None) -> pd.DataFrame:
    """Per-feature univariate human-preference relevance (which directions humans
    reward). Wraps ``win_relevance``."""
    codes = np.asarray(codes, dtype=np.float32)
    win = _win_vector(meta, codes)
    df = win_relevance(codes, win)
    return _attach_names(df, names)
