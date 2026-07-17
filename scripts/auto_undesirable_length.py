#!/usr/bin/env python
"""Example: auto-flag spurious (length-driven) features in a preference dataset.

Demonstrates ``auto_undesirable`` (design §3.4 option 2): builds the classic
length-difference surrogate len(chosen) - len(rejected) per example, correlates
each feature's reward direction with it, and reports the features whose preference
direction tracks length — the canonical confound. Feed the returned ids as the
``undesirable`` set to scripts/diagnose_dataset.py.

Requires a DIFFERENCE lens built with chosen = completion_a (so both z_diff and
the surrogate share that orientation), and the labeled SOURCE it was built from
(OpenJury annotations) to recover completion lengths. The surrogate is aligned to
z_diff rows by ``instruction_id`` (not row order), so it is robust to reordering.
CPU-only.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prefscope.analysis import auto_undesirable, feature_confound_correlation  # noqa: E402
from prefscope.data.ingest import load_battles  # noqa: E402

log = logging.getLogger(__name__)

# --- constants -------------------------------------------------------------
LENS_DIR = Path("lenses/dataset_diff")      # difference lens (chosen=A): z_diff + battles.parquet
ANNOTATIONS = "path/to/annotations.json"    # the labeled source it was built from
THRESHOLD = 0.3                             # |corr| cutoff to flag a feature spurious


def main() -> None:
    logging.basicConfig(level=logging.INFO, datefmt="%H:%M:%S",
                        format="%(asctime)s %(levelname)s %(name)s | %(message)s")

    manifest = json.loads((LENS_DIR / "manifest.json").read_text())
    if manifest.get("input_rep") != "difference":
        raise SystemExit(f"{LENS_DIR} must be a difference lens (chosen=A); got "
                         f"input_rep={manifest.get('input_rep')!r}")

    z = np.load(LENS_DIR / "z_diff.npy")
    battles_meta = pd.read_parquet(LENS_DIR / "battles.parquet")   # instruction_id, z_diff order
    names_path = LENS_DIR / "feature_names.csv"
    names = pd.read_csv(names_path) if names_path.exists() else None

    # length-diff surrogate (chosen = completion_a), aligned to z_diff by instruction_id
    src = load_battles(ANNOTATIONS)
    len_by_id = dict(zip(src["instruction_id"].astype(str),
                         (src["completion_a"].str.len() - src["completion_b"].str.len())))
    length_diff = battles_meta["instruction_id"].astype(str).map(len_by_id)
    if length_diff.isna().any():
        raise SystemExit("could not align every battle to the source by instruction_id")
    length_diff = length_diff.to_numpy(dtype=float)

    corr = feature_confound_correlation(z, length_diff)
    flagged = auto_undesirable(z, length_diff, threshold=THRESHOLD)
    log.info("%d / %d features track length-diff (|corr| >= %.2f)",
             len(flagged), z.shape[1], THRESHOLD)

    if names is not None:
        corr = corr.merge(names[["feature_id", "concept"]], on="feature_id", how="left")
    print(f"\nFeatures whose preference direction tracks LENGTH (auto-U = {flagged}):\n")
    for _, r in corr.head(15).iterrows():
        tag = "  <-- flagged" if int(r["feature_id"]) in flagged else ""
        concept = r.get("concept", "") if names is not None else ""
        print(f"  corr={r['corr']:+.3f}  feature {int(r['feature_id'])}  {concept}{tag}")


if __name__ == "__main__":
    main()
