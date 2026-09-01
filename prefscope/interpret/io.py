"""Load a built lens and re-attach prompts/completions from the annotation JSON.

The lens battles.parquet stores meta only; interpretation needs the text, so we
re-load it from the annotations and align to the lens's z_diff row order.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from prefscope.artifacts import Z_A, Z_DIFF
from prefscope.data.ingest import load_battles


def load_lens_battles(lens_dir, annotations=None, *, corpus=None):
    """Return (battles_df, codes, manifest), battles row-aligned to the codes.

    Codes are the lens's paired contrast (``z_diff.npy``) when present, otherwise the
    single-response codes (``z_a.npy``).

    Re-attach the prompts/completions from whichever source the lens was built
    from: an OpenJury annotation JSON (``annotations``) or a merged battle
    corpus parquet (``corpus``). Exactly one must be given.
    """
    if bool(annotations) == bool(corpus):
        raise ValueError("provide exactly one of annotations or corpus")
    lens_dir = Path(lens_dir)
    # Interpretation reads a handful of feature columns at a time.  Memory-map large
    # code matrices so a one-feature naming run does not materialize multi-GB artifacts.
    code_path = next(
        (lens_dir / name for name in (Z_DIFF, Z_A) if (lens_dir / name).exists()), None)
    if code_path is None:
        raise FileNotFoundError(f"{lens_dir} has neither {Z_DIFF} nor {Z_A}")
    z_diff = np.load(code_path, mmap_mode="r")
    manifest = json.loads((lens_dir / "manifest.json").read_text())
    lens_meta = pd.read_parquet(lens_dir / "battles.parquet")
    order = lens_meta["instruction_id"].astype(str).tolist()

    if corpus:
        from prefscope.data.corpus import load_corpus
        full = load_corpus(corpus)
    else:
        full = load_battles(annotations)
    full["instruction_id"] = full["instruction_id"].astype(str)
    indexed = full.set_index("instruction_id")
    missing = [i for i in order if i not in indexed.index]
    if missing:
        raise ValueError(
            f"{len(missing)} lens battles missing from annotations "
            f"(e.g. {missing[:3]}); wrong annotation file?")
    battles = indexed.loc[order].reset_index()
    if len(battles) != len(z_diff):
        raise ValueError(
            f"row mismatch: {len(battles)} battles vs {len(z_diff)} z_diff rows")
    return battles, z_diff, manifest
