#!/usr/bin/env python
"""Example 03 — cluster features into behaviors + win-relevance (Python API).

Mirrors `prefscope cluster-features` and `win-relevance`. The LLM is used only
for `--name-clusters`. win-relevance needs a corpus WITH human_pref
(build it with `build-corpus --keep-labels`).

    python scripts/examples/03_cluster_and_winrelevance.py \
        --lens-dir artifacts/lenses/completion \
        --corpus   data/corpus.parquet \
        --names    artifacts/interpretation/completion/feature_fidelity.csv \
        --out-dir  artifacts/interpretation/completion --name-clusters
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from prefscope.interpret.io import load_lens_battles                          # noqa: E402
from prefscope.interpret.llm import LLMClient                                 # noqa: E402
from prefscope.pipeline.cluster import (                                      # noqa: E402
    cluster_features, summarize_clusters, name_clusters)
from prefscope.pipeline.winrelevance import win_relevance                     # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="cluster features + win-relevance")
    ap.add_argument("--lens-dir", required=True)
    ap.add_argument("--corpus", required=True, help="corpus WITH human_pref (--keep-labels)")
    ap.add_argument("--names", default=None, help="feature_fidelity.csv (concepts + fidelity)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-clusters", type=int, default=12)
    ap.add_argument("--method", default="spherical-kmeans",
                    choices=["spherical-kmeans", "agglomerative"])
    ap.add_argument("--name-clusters", action="store_true", help="LLM-name each behavior")
    ap.add_argument("--model", default="deepseek/deepseek-v3.2")
    ap.add_argument("--backend", default="openai")
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    z = np.load(Path(args.lens_dir) / "z_diff.npy")
    names = pd.read_csv(args.names) if args.names else None

    # --- cluster by coactivation: distance = 1 - |corr| between feature columns ---
    clusters = cluster_features(z, n_clusters=args.n_clusters, method=args.method)
    summary = summarize_clusters(clusters, names=names)   # one row per behavior
    if args.name_clusters:
        labels = name_clusters(summary, LLMClient(backend=args.backend, model=args.model),
                               concurrency=args.concurrency)
        summary["behavior"] = summary["cluster_id"].map(labels)
        clusters = clusters.merge(summary[["cluster_id", "behavior"]],
                                  on="cluster_id", how="left")
    clusters.to_csv(out / "feature_clusters.csv", index=False)
    summary.to_csv(out / "feature_clusters_summary.csv", index=False)
    print(f"{clusters['cluster_id'].nunique()} behaviors over {len(clusters)} features")

    # --- win relevance: which features humans reward (z>0 = A; needs human_pref) ---
    battles, z_diff, _ = load_lens_battles(args.lens_dir, corpus=args.corpus)
    if "human_pref" not in battles.columns or battles["human_pref"].isna().all():
        print("corpus has no human_pref; rebuild with `build-corpus --keep-labels` — "
              "skipping win-relevance")
        return
    wr = win_relevance(z_diff, battles["human_pref"].to_numpy())
    if names is not None and "concept" in names.columns:
        wr = wr.merge(names[["feature_id", "concept"]], on="feature_id", how="left")
    wr.to_csv(out / "win_relevance.csv", index=False)
    print(f"win_relevance for {len(wr)} features -> {out/'win_relevance.csv'}")


if __name__ == "__main__":
    main()
