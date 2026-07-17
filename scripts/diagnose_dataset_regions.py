#!/usr/bin/env python
"""Example: cluster a preference dataset into behavior regions and score each.

A user-level recipe over the framework primitives (deliberately a script). It
loads a difference lens (``z_diff.npy``; and ``z_a``/``z_b`` if the lens was built
in individual mode), clusters the examples into behavior regions B_k on their
symmetric activity, and for each region reports the concepts it most rewards or
penalizes (Δ_{k,m}, split-half stable + Bonferroni). This is the feature-conditioned
("which subsets of the data reward which behaviors?") view of *Anatomy of
Post-Training* (App. B.1 / §feature-conditioned). CPU-only — no model, no GPU.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prefscope.analysis import region_behavior_contrast, symmetric_activity  # noqa: E402
from prefscope.pipeline.cluster import cluster_examples  # noqa: E402

log = logging.getLogger(__name__)

# --- constants -------------------------------------------------------------
LENS_DIR = Path("lenses/dataset_diff")
N_REGIONS = 8          # number of behavior regions B_k
ALPHA = 0.05           # Bonferroni-corrected Welch p cutoff
TOP_PER_REGION = 8     # concepts to print per region


def _concept(names, feature_id):
    if names is not None and "concept" in names.columns:
        hit = names.loc[names["feature_id"] == feature_id, "concept"]
        if len(hit) and isinstance(hit.iloc[0], str):
            return hit.iloc[0]
    return f"feature {feature_id}"


def main() -> None:
    logging.basicConfig(level=logging.INFO, datefmt="%H:%M:%S",
                        format="%(asctime)s %(levelname)s %(name)s | %(message)s")

    z_diff = np.load(LENS_DIR / "z_diff.npy")          # signed chosen-minus-rejected
    z_a_path, z_b_path = LENS_DIR / "z_a.npy", LENS_DIR / "z_b.npy"
    if z_a_path.exists() and z_b_path.exists():
        profile = symmetric_activity(np.load(z_a_path), np.load(z_b_path))
    else:
        log.info("no z_a/z_b (difference-mode lens); clustering on |z_diff|")
        profile = np.abs(z_diff)
    names_path = LENS_DIR / "feature_names.csv"
    names = pd.read_csv(names_path) if names_path.exists() else None

    log.info("clustering %d examples into %d regions", z_diff.shape[0], N_REGIONS)
    clusters = cluster_examples(profile, n_clusters=N_REGIONS)
    regions = region_behavior_contrast(z_diff, clusters["cluster_id"].to_numpy())
    regions = regions[regions["stable"] & (regions["p_bonferroni"] < ALPHA)]
    if names is not None:
        regions = regions.merge(names[["feature_id", "concept"]], on="feature_id", how="left")
    regions.to_csv(LENS_DIR / "region_behaviors.csv", index=False)

    for k in sorted(clusters["cluster_id"].unique()):
        sub = (regions[regions["cluster_id"] == k]
               .sort_values("delta", key=abs, ascending=False).head(TOP_PER_REGION))
        size = int((clusters["cluster_id"] == k).sum())
        print(f"\n== region {k}  ({size} examples) ==")
        for _, r in sub.iterrows():
            verb = "rewards" if r["delta"] > 0 else "penalizes"
            print(f"  {verb} ({r['delta']:+.2f})  {_concept(names, int(r['feature_id']))}")
    log.info("saved region_behaviors.csv under %s", LENS_DIR)


if __name__ == "__main__":
    main()
