#!/usr/bin/env python
"""Shuffled-names negative control for the fidelity gate.

Builds a names CSV in which every feature keeps its id but receives ANOTHER
feature's concept text (a derangement — no feature keeps its own name). Running
``interpret verify`` on this CSV measures the gate's false-pass rate: how often
an LLM verifier confirms a *wrong* name on held-out activations. A well-behaved
gate should pass ~none of these; the observed rate calibrates how much the real
pass count (e.g. 82/380) could owe to verifier leniency.

If a ``--clusters`` CSV is given, the shuffle additionally avoids donors from
the same co-activation cluster where possible — otherwise near-synonym features
would trade names and "falsely" pass for the honest reason that the swapped
name is still true, overstating the false-pass rate.

    python scripts/make_shuffled_names.py \
        --names "$LENS/feature_names.csv" --clusters "$LENS/feature_clusters.csv" \
        --out "$LENS/validation/names_shuffled.csv" --seed 0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def derange(ids: list, clusters: dict, rng: np.random.Generator,
            max_restarts: int = 500) -> dict:
    """id -> donor id; no fixed points, avoiding same-cluster donors *per feature*.

    A global "zero same-cluster donors across all features at once" permutation is
    combinatorially unreachable for hundreds of features, so uniform-rejection
    sampling silently degrades to a plain derangement that ignores clusters. Instead
    assign donors greedily in random order: each feature draws from its allowed set
    (not itself, different cluster), and only falls back to a same-cluster donor when
    no clean donor is left. This minimises same-cluster reuse *locally* rather than
    demanding a globally-perfect permutation, so cluster-avoidance actually happens.
    """
    n = len(ids)
    cl = [clusters.get(i) for i in ids]

    def _different_cluster(i: int, j: int) -> bool:
        return cl[i] is None or cl[j] is None or cl[i] != cl[j]

    for _ in range(max_restarts):
        order = list(rng.permutation(n))
        used = [False] * n
        assign = [-1] * n
        ok = True
        for i in order:
            clean = [j for j in range(n)
                     if not used[j] and j != i and _different_cluster(i, j)]
            cands = clean or [j for j in range(n) if not used[j] and j != i]
            if not cands:                       # dead end (last slot is self) — restart
                ok = False
                break
            j = int(rng.choice(cands))
            assign[i] = j
            used[j] = True
        if ok:                                  # j != i enforced ⇒ guaranteed derangement
            return {ids[i]: ids[assign[i]] for i in range(n)}
    sys.exit("could not build a derangement (too few features?)")


def main() -> None:
    ap = argparse.ArgumentParser(description="shuffled-names verify control")
    ap.add_argument("--names", required=True, help="feature_names.csv (feature_id, concept)")
    ap.add_argument("--clusters", default=None,
                    help="feature_clusters.csv — avoid same-cluster donors")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    names = pd.read_csv(args.names)
    named = names[names["concept"].notna() & (names["concept"].astype(str).str.strip() != "")]
    ids = named["feature_id"].astype(int).tolist()
    concept = dict(zip(named["feature_id"].astype(int), named["concept"]))

    clusters: dict = {}
    if args.clusters and Path(args.clusters).exists():
        cl = pd.read_csv(args.clusters)
        clusters = dict(zip(cl["feature_id"].astype(int), cl["cluster_id"]))

    donor = derange(ids, clusters, np.random.default_rng(args.seed))
    n_same = sum(1 for i in ids if clusters and clusters.get(i) is not None
                 and clusters.get(i) == clusters.get(donor[i]))

    out = named.copy()
    out["concept"] = out["feature_id"].astype(int).map(lambda i: concept[donor[i]])
    # co-shuffle the abbreviation so it agrees with the donated concept (else the
    # verifier sees a name/abbrev that disagree). Only if the column exists.
    if "concept_abbrev" in out.columns:
        abbrev = dict(zip(named["feature_id"].astype(int), named["concept_abbrev"]))
        out["concept_abbrev"] = out["feature_id"].astype(int).map(lambda i: abbrev[donor[i]])
    out["donor_feature_id"] = out["feature_id"].astype(int).map(donor)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    if clusters and n_same:
        print(f"WARNING: {n_same}/{len(ids)} donors share a cluster with their "
              f"recipient (no clean donor was available). Same-cluster swaps can be "
              f"coincidentally true, so exclude them from the false-pass numerator "
              f"via donor_feature_id when calibrating.", file=sys.stderr)
    print(f"wrote {len(out)} shuffled names to {args.out} "
          f"(seed {args.seed}, same-cluster donors: {n_same})")


if __name__ == "__main__":
    main()
