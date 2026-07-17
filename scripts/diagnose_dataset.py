#!/usr/bin/env python
"""Example: diagnose a preference dataset from a trained difference lens.

A user-level recipe over the framework primitives in ``prefscope.analysis``
(deliberately a script, not framework code). It loads a difference lens built
with ``build-lens`` in difference mode (chosen = completion_a, rejected =
completion_b), and a manually chosen set of UNDESIRABLE feature ids, and writes:

  - dataset_reward.csv      : per-feature reward direction r_f + split-half stability
  - problematic_samples.csv : per-example spurious_share + label_inconsistency

It then prints the most confound-driven and most label-inconsistent samples.
CPU-only: it reads cached codes (z_diff.npy) — no model, no GPU.

``feature_names.csv`` is NOT produced by ``build-lens``; generate it with
``interpret name`` and place it in LENS_DIR to get concept labels (optional —
without it the report uses bare feature ids).

Constants below must match the lens you built.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prefscope.analysis import diagnose_dataset  # noqa: E402

log = logging.getLogger(__name__)

# --- constants -------------------------------------------------------------
LENS_DIR = Path("lenses/dataset_diff")     # a difference lens (z_diff.npy + battles.parquet)
UNDESIRABLE = [2]                          # feature ids judged spurious (from `interpret name`)
TOP = 20                                   # rows to print per ranking


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, datefmt="%H:%M:%S",
        format="%(asctime)s %(levelname)s %(name)s | %(message)s")

    z = np.load(LENS_DIR / "z_diff.npy")
    battles = pd.read_parquet(LENS_DIR / "battles.parquet")
    id_col = "instruction_id" if "instruction_id" in battles.columns else "battle_id"
    ids = battles[id_col].tolist() if id_col in battles.columns else None
    names_path = LENS_DIR / "feature_names.csv"
    names = pd.read_csv(names_path) if names_path.exists() else None
    if names is None:
        log.warning("no feature_names.csv in %s — run `interpret name` and place it "
                    "there for concept labels; using bare feature ids", LENS_DIR)

    log.info("diagnosing %d samples x %d features; undesirable=%s",
             z.shape[0], z.shape[1], UNDESIRABLE)
    per_feature, per_sample = diagnose_dataset(z, UNDESIRABLE, ids=ids, names=names)

    per_feature.to_csv(LENS_DIR / "dataset_reward.csv", index=False)
    per_sample.to_csv(LENS_DIR / "problematic_samples.csv", index=False)

    print("\nMost CONFOUND-DRIVEN samples (high spurious_share):")
    for _, r in per_sample.sort_values("spurious_share", ascending=False).head(TOP).iterrows():
        print(f"  share={r['spurious_share']:.3f}  id={r['id']}")
    print("\nMost LABEL-INCONSISTENT samples (low a_i — chosen weaker on quality):")
    for _, r in per_sample.sort_values("label_inconsistency").head(TOP).iterrows():
        print(f"  a={r['label_inconsistency']:+.3f}  id={r['id']}")
    log.info("saved dataset_reward.csv + problematic_samples.csv under %s", LENS_DIR)


if __name__ == "__main__":
    main()
