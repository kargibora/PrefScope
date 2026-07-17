#!/usr/bin/env python
"""Example: prompt-conditioned dataset diagnosis (which prompt topics teach which
response shifts) — the local view of *Anatomy of Post-Training* (App. B, §3.5).

Composes existing primitives (no new framework code):
  cluster_examples(z_prompt)            -> prompt regions A_k (one per battle)
  region_behavior_contrast(z_diff, A_k) -> Δ_{k,f}: response-concept f rewarded
                                           when the prompt is in region A_k vs not
                                           (split-half stable + Bonferroni)

INPUTS — both are per-battle, row-aligned, from the SAME labeled dataset:
  DIFF_LENS_DIR/z_diff.npy     a DIFFERENCE lens (input_rep="difference",
                               chosen = completion_a) so sign(z_diff[i,f]) is the
                               reward direction (f stronger in the chosen response).
  PROMPT_LENS_DIR/z_prompt.npy a prompt lens built per-battle (NO prompt dedup) so
                               its rows align one-to-one with z_diff.
Using the unsupervised concept lens from lmsys_concept_pipeline.py here would be
wrong: its z_diff is an arbitrary A-minus-B contrast, not chosen-minus-rejected.

CPU-only; no model, no GPU.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prefscope.analysis import region_behavior_contrast  # noqa: E402
from prefscope.pipeline.cluster import cluster_examples  # noqa: E402

log = logging.getLogger(__name__)

# --- constants -------------------------------------------------------------
DIFF_LENS_DIR = Path("lenses/dataset_diff")      # difference lens (chosen=A): z_diff.npy
PROMPT_LENS_DIR = Path("lenses/dataset_prompt")  # per-battle prompt lens: z_prompt.npy
N_PROMPT_CLUSTERS = 8
ALPHA = 0.05
TOP_PER_CLUSTER = 8


def _concept(names, feature_id):
    if names is not None and "concept" in names.columns:
        hit = names.loc[names["feature_id"] == feature_id, "concept"]
        if len(hit) and isinstance(hit.iloc[0], str):
            return hit.iloc[0]
    return f"feature {feature_id}"


def main() -> None:
    logging.basicConfig(level=logging.INFO, datefmt="%H:%M:%S",
                        format="%(asctime)s %(levelname)s %(name)s | %(message)s")

    manifest = json.loads((DIFF_LENS_DIR / "manifest.json").read_text())
    if manifest.get("input_rep") != "difference":
        raise SystemExit(f"{DIFF_LENS_DIR} must be a difference lens (chosen=A); got "
                         f"input_rep={manifest.get('input_rep')!r}")

    z_diff = np.load(DIFF_LENS_DIR / "z_diff.npy")        # (N, Mr) chosen-rejected reward
    z_prompt = np.load(PROMPT_LENS_DIR / "z_prompt.npy")  # (N, Mp) per-battle prompt codes
    if z_prompt.shape[0] != z_diff.shape[0]:
        raise SystemExit("z_prompt and z_diff are not row-aligned; build the prompt "
                         "lens per-battle (no prompt dedup) on the same dataset")
    names_path = DIFF_LENS_DIR / "feature_names.csv"
    resp_names = pd.read_csv(names_path) if names_path.exists() else None

    region_per_battle = cluster_examples(
        z_prompt, n_clusters=N_PROMPT_CLUSTERS)["cluster_id"].to_numpy()

    log.info("scoring %d response concepts across %d prompt regions over %d battles",
             z_diff.shape[1], N_PROMPT_CLUSTERS, z_diff.shape[0])
    regions = region_behavior_contrast(z_diff, region_per_battle)
    regions = regions[regions["stable"] & (regions["p_bonferroni"] < ALPHA)]
    if resp_names is not None:
        regions = regions.merge(resp_names[["feature_id", "concept"]],
                                on="feature_id", how="left")
    regions.to_csv(DIFF_LENS_DIR / "prompt_conditioned.csv", index=False)

    for k in sorted(np.unique(region_per_battle)):
        size = int((region_per_battle == k).sum())
        sub = (regions[regions["cluster_id"] == k]
               .sort_values("delta", key=abs, ascending=False).head(TOP_PER_CLUSTER))
        print(f"\n== prompt region {k}  ({size} battles) teaches ==")
        for _, r in sub.iterrows():
            verb = "reward" if r["delta"] > 0 else "penalize"
            print(f"  {verb} ({r['delta']:+.2f})  {_concept(resp_names, int(r['feature_id']))}")
    log.info("saved prompt_conditioned.csv under %s", DIFF_LENS_DIR)


if __name__ == "__main__":
    main()
