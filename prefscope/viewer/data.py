"""Pure data assembly for the viewer (no Streamlit import, so it is testable).

`feature_table` summarises every axis (concept, fidelity, corpus activation
stats); `top_battles` selects the battles that drive a chosen axis. The app
layer wraps these with widgets and plots.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

_TEXT_COLS = ("prompt", "completion_a", "completion_b")


def load_lens_for_view(lens_dir, annotations=None, corpus=None):
    """Load a lens for the viewer; completion text is optional.

    With ``corpus`` (a merged-corpus parquet) or ``annotations`` (OpenJury JSON)
    we re-attach the A/B completions for the Feature-detail view. Without either
    we fall back to the lens's own ``battles.parquet`` meta (already row-aligned
    to ``z_diff``), so the viewer still runs from the lens folder alone —
    directions, fidelity, and activations work; text shows as blank.
    """
    lens_dir = Path(lens_dir)
    if corpus or annotations:
        from prefscope.interpret.io import load_lens_battles
        if corpus:
            return load_lens_battles(lens_dir, corpus=corpus)
        return load_lens_battles(lens_dir, list(annotations))
    z_diff = np.load(lens_dir / "z_diff.npy")
    manifest = json.loads((lens_dir / "manifest.json").read_text())
    battles = pd.read_parquet(lens_dir / "battles.parquet")
    if "instruction_id" not in battles.columns and "battle_id" in battles.columns:
        battles["instruction_id"] = battles["battle_id"]
    for c in _TEXT_COLS:
        if c not in battles.columns:
            battles[c] = ""
    if len(battles) != len(z_diff):
        raise ValueError(
            f"row mismatch: {len(battles)} meta rows vs {len(z_diff)} z_diff rows")
    return battles.reset_index(drop=True), z_diff, manifest


def feature_table(z_diff: np.ndarray, names: pd.DataFrame | None = None,
                  fidelity: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per SAE axis with corpus-level activation stats + any metadata."""
    z_diff = np.asarray(z_diff, dtype=np.float32)
    n, m = z_diff.shape
    rows = []
    for f in range(m):
        col = z_diff[:, f]
        rows.append({
            "feature_id": f,
            "fire_rate": float((col != 0).mean()) if n else float("nan"),
            "self_more_rate": float((col > 0).mean()) if n else float("nan"),
            "self_less_rate": float((col < 0).mean()) if n else float("nan"),
            "mean_abs_z": float(np.abs(col).mean()) if n else float("nan"),
        })
    df = pd.DataFrame(rows)
    if names is not None:
        keep = [c for c in ("feature_id", "concept", "concept_abbrev") if c in names.columns]
        df = df.merge(names[keep], on="feature_id", how="left")
    if fidelity is not None:
        keep = [c for c in ("feature_id", "correlation", "p_bonferroni", "fidelity_pass")
                if c in fidelity.columns]
        df = df.merge(fidelity[keep], on="feature_id", how="left")
    front = [c for c in ("feature_id", "concept", "concept_abbrev", "fidelity_pass",
                         "correlation") if c in df.columns]
    return df[front + [c for c in df.columns if c not in front]]


def top_battles(z_diff: np.ndarray, battles: pd.DataFrame, feature_id: int, *,
                mode: str = "abs", n: int = 20) -> pd.DataFrame:
    """Battles ranked by this axis's activation.

    mode: 'abs' (largest |z|), 'pos' (most positive), 'neg' (most negative).
    Only battles where the axis fires (z != 0) are returned.
    """
    if mode not in ("abs", "pos", "neg"):
        raise ValueError(f"mode must be 'abs', 'pos' or 'neg', got {mode!r}")
    col = np.asarray(z_diff[:, feature_id], dtype=np.float32)
    if mode == "pos":
        order = np.argsort(-col)
        order = order[col[order] > 0]
    elif mode == "neg":
        order = np.argsort(col)
        order = order[col[order] < 0]
    else:
        order = np.argsort(-np.abs(col))
        order = order[col[order] != 0]
    order = order[:n]
    out = battles.iloc[order].copy()
    out.insert(0, "z", col[order])
    return out.reset_index(drop=True)


def diagnosis_battles(per_battle: pd.DataFrame, feature_id: int, *,
                      mode: str = "more", n: int = 10) -> pd.DataFrame:
    """Rank the diagnosed model's battles by a feature's activation.

    Reads the ``z{feature_id}`` column from a `diagnose --battles-out` parquet
    (target-minus-opponent codes). mode: 'more' (target over-expresses, z>0),
    'less' (under-expresses, z<0), 'abs' (largest magnitude). Lets you eyeball
    the model's own responses where it most does / doesn't express the concept.
    """
    if mode not in ("more", "less", "abs"):
        raise ValueError(f"mode must be 'more', 'less' or 'abs', got {mode!r}")
    col = f"z{int(feature_id)}"
    if col not in per_battle.columns:
        raise ValueError(f"{col} not in per-battle frame; was it diagnosed?")
    z = per_battle[col].to_numpy(dtype=float)
    if mode == "more":
        order = np.argsort(-z); order = order[z[order] > 0]
    elif mode == "less":
        order = np.argsort(z); order = order[z[order] < 0]
    else:
        order = np.argsort(-np.abs(z)); order = order[z[order] != 0]
    order = order[:n]
    out = per_battle.iloc[order].copy()
    out.insert(0, "z", z[order])
    return out.reset_index(drop=True)
