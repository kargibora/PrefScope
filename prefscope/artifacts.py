"""Canonical artifact filenames + small shared readers.

Centralizes the magic filename literals (``feature_names.csv``, ``z_diff.npy``, …)
and the battle-id column convention that were copy-pasted across the package and
``scripts/``. Import these instead of re-typing the strings, so a rename happens in
one place. (Migration is incremental — new/edited code should use these; old literals
are equivalent until touched.)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# --- lens directory artifacts ---
MANIFEST = "manifest.json"
SAE_MODEL = "sae_model.pt"
BATTLES = "battles.parquet"
FEATURE_NAMES = "feature_names.csv"
FEATURE_FIDELITY = "feature_fidelity.csv"
FEATURE_CLUSTERS = "feature_clusters.csv"
WIN_RELEVANCE = "win_relevance.csv"
Z_DIFF = "z_diff.npy"
Z_A = "z_a.npy"
Z_B = "z_b.npy"
Z_PROMPT = "z_prompt.npy"

# --- prompt-lens interpret artifacts ---
PROMPT_FEATURE_NAMES = "prompt_feature_names.csv"
PROMPT_FEATURE_FIDELITY = "prompt_feature_fidelity.csv"
PROMPT_FEATURE_CLUSTERS = "prompt_feature_clusters.csv"


def battle_id_col(df: pd.DataFrame) -> str:
    """The per-row battle-id column. The framework writes either ``battle_id`` or
    ``instruction_id`` (it sets ``instruction_id = battle_id``); prefer the former."""
    if "battle_id" in df.columns:
        return "battle_id"
    if "instruction_id" in df.columns:
        return "instruction_id"
    raise KeyError(f"no battle_id/instruction_id column in {list(df.columns)}")


def lens_battle_ids(source: "pd.DataFrame | str | Path") -> np.ndarray:
    """Per-row battle ids as strings, from a battles DataFrame OR a lens dir/path
    (in which case ``battles.parquet`` is read)."""
    if isinstance(source, (str, Path)):
        source = pd.read_parquet(Path(source) / BATTLES)
    return source[battle_id_col(source)].astype(str).to_numpy()
