"""Normalized, label-free battle corpus for training a difference SAE.

A corpus row is just a pairwise comparison of two completions to the same
prompt — no human/judge label is needed, because the SAE learns the core
differences between completion A and B directly from ``e_a - e_b``. Per-arena
adapters (see ``prefscope/data/arenas.py``) map a raw HuggingFace dataset into
this schema; ``merge_corpora`` concatenates and de-duplicates across arenas.

Schema (all strings):
    battle_id · source · language · prompt · model_a · model_b · completion_a · completion_b

``battle_id`` hashes the *content only* (prompt + models + completions), not the
source — so the same battle appearing in two arenas (e.g. the 100k/140k overlap)
collapses to one row.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

# Fields an adapter must produce; also the fields battle_id is hashed over.
CONTENT_COLS = ["prompt", "model_a", "model_b", "completion_a", "completion_b"]
CORPUS_COLS = ["battle_id", "source", "language"] + CONTENT_COLS
# Optional columns carried through when present (e.g. human preference labels).
# y = P(A preferred): model_a wins -> 1.0, model_b -> 0.0, tie -> 0.5.
OPTIONAL_COLS = ["human_pref"]


def make_battle_id(row: dict) -> str:
    """Stable id from content (source-independent), so duplicates dedup across arenas."""
    raw = "␟".join(str(row.get(c, "")) for c in CONTENT_COLS)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def normalize(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Validate + clean an adapter's output into the corpus schema.

    Adds ``source``/``language`` and a content ``battle_id`` (unless the adapter
    already set one); drops rows with an empty prompt, model, or completion.
    """
    missing = [c for c in CONTENT_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"adapter for {source!r} produced no {missing} column(s)")
    out = df.copy()
    out["source"] = source
    out["language"] = out["language"].astype("string") if "language" in out.columns else ""
    for c in CONTENT_COLS:
        out[c] = out[c].astype("string").str.strip()
    good = (out["prompt"].str.len() > 0) & (out["completion_a"].str.len() > 0) \
        & (out["completion_b"].str.len() > 0) \
        & out["model_a"].notna() & out["model_b"].notna() \
        & (out["model_a"].str.len() > 0) & (out["model_b"].str.len() > 0)
    out = out[good].copy()
    if "battle_id" not in out.columns:
        out["battle_id"] = [make_battle_id(r) for r in out[CONTENT_COLS].to_dict("records")]
    out["language"] = out["language"].fillna("")
    keep = CORPUS_COLS + [c for c in OPTIONAL_COLS if c in out.columns]
    return out[keep].reset_index(drop=True)


def merge_corpora(frames) -> pd.DataFrame:
    """Concatenate normalized frames and dedup on battle_id (de-overlaps 100k/140k)."""
    frames = [f for f in frames if f is not None and len(f)]
    if not frames:
        return pd.DataFrame(columns=CORPUS_COLS)
    merged = pd.concat(frames, ignore_index=True)
    return merged.drop_duplicates("battle_id").reset_index(drop=True)


def write_corpus(df: pd.DataFrame, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cols = CORPUS_COLS + [c for c in OPTIONAL_COLS if c in df.columns]
    df[cols].to_parquet(path, index=False)


def load_corpus(path) -> pd.DataFrame:
    """Load a merged corpus and expose ``instruction_id`` so build-lens can use it."""
    df = pd.read_parquet(path)
    missing = [c for c in CORPUS_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: not a corpus file (missing {missing})")
    df = df.copy()
    df["instruction_id"] = df["battle_id"]
    return df
