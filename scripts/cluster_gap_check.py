#!/usr/bin/env python
"""Does pooling features into behaviour clusters widen the winner-loser gap?

At feature resolution the winner-minus-loser prevalence gap on producing a concept is
tiny (~2-5pp), so "what a model lacks that wins" is a weak signal. This checks whether
aggregating features into co-activation CLUSTERS (behaviours) makes the gap large enough
to matter. For each prompt type X and cluster C, the behaviour's expression level is the
fraction of C's members that fire; we compare that between winning and losing responses.

No new LLM calls — just the completion codes, prompt-lens argmax, human_pref, clusters.

    python scripts/cluster_gap_check.py --completion-lens "$L" --prompt-lens "$LP" \
        --corpus "$CORPUS" --clusters "$L/feature_clusters.csv"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prefscope.artifacts import Z_PROMPT, lens_battle_ids            # noqa: E402
from prefscope.data.corpus import load_corpus                        # noqa: E402
from prefscope.data.pair_schema import LABEL, MODEL_A, MODEL_B       # noqa: E402


def gap_stats(win_expr, lose_expr, X, min_battles):
    """Per prompt type, winner-minus-loser mean expression; return the pooled gap array."""
    gaps = []
    for x in np.unique(X):
        m = X == x
        if int(m.sum()) < min_battles:
            continue
        g = win_expr[m].mean(axis=0) - lose_expr[m].mean(axis=0)   # (#units,)
        gaps.append(g)
    return np.abs(np.concatenate(gaps)) if gaps else np.array([])


def main() -> None:
    ap = argparse.ArgumentParser(description="feature vs cluster winner-loser gap")
    ap.add_argument("--completion-lens", required=True, dest="completion_lens")
    ap.add_argument("--prompt-lens", required=True, dest="prompt_lens")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--clusters", required=True, help="feature_clusters.csv (feature_id, cluster_id)")
    ap.add_argument("--fidelity", default=None, help="restrict features to verified")
    ap.add_argument("--min-battles", type=int, default=300, dest="min_battles")
    args = ap.parse_args()

    clens, plens = Path(args.completion_lens), Path(args.prompt_lens)
    fa = np.load(clens / "z_a.npy") > 0
    fb = np.load(clens / "z_b.npy") > 0
    z_prompt = np.load(plens / Z_PROMPT)
    cb, pb = lens_battle_ids(clens), lens_battle_ids(plens)
    if not (len(cb) == len(pb) and bool((cb == pb).all())):
        common = pd.Index(cb).intersection(pd.Index(pb))
        cpos = {b: i for i, b in enumerate(cb)}; ppos = {b: i for i, b in enumerate(pb)}
        ci = np.array([cpos[b] for b in common])
        fa, fb = fa[ci], fb[ci]
        z_prompt = z_prompt[np.array([ppos[b] for b in common])]
        cb = common.to_numpy()
    X = z_prompt.argmax(axis=1)

    corp = load_corpus(args.corpus).assign(battle_id=lambda d: d["battle_id"].astype(str)).set_index("battle_id")
    y = pd.Series(cb).astype(str).map(corp[LABEL]).to_numpy(dtype=float)
    keep = np.isfinite(y) & (y != 0.5)
    fa, fb, X, y = fa[keep], fb[keep], X[keep], y[keep]
    a_won = y == 1.0
    print(f"{len(y)} decisive battles, {len(set(X.tolist()))} prompt types", flush=True)

    # winner / loser FEATURE fire (0/1 per feature)
    win_f = np.where(a_won[:, None], fa, fb).astype(np.float32)
    lose_f = np.where(a_won[:, None], fb, fa).astype(np.float32)

    # optional: restrict to verified features
    feat_ids = np.arange(fa.shape[1])
    if args.fidelity and Path(args.fidelity).exists():
        fid = pd.read_csv(args.fidelity)
        if "fidelity_pass" in fid.columns:
            feat_ids = np.array(sorted(fid[fid["fidelity_pass"].astype(bool)]["feature_id"].astype(int)))
    wf, lf = win_f[:, feat_ids], lose_f[:, feat_ids]

    # cluster expression = fraction of a cluster's members that fire
    cl = pd.read_csv(args.clusters).dropna(subset=["cluster_id"])
    cl["cluster_id"] = cl["cluster_id"].astype(int); cl["feature_id"] = cl["feature_id"].astype(int)
    members = {c: [f for f in g["feature_id"] if 0 <= f < fa.shape[1]]
               for c, g in cl.groupby("cluster_id")}
    members = {c: fs for c, fs in members.items() if fs}
    win_c = np.column_stack([win_f[:, fs].mean(axis=1) for fs in members.values()])
    lose_c = np.column_stack([lose_f[:, fs].mean(axis=1) for fs in members.values()])

    fgap = gap_stats(wf, lf, X, args.min_battles)
    cgap = gap_stats(win_c, lose_c, X, args.min_battles)

    def summarize(g, label):
        if not len(g):
            print(f"{label}: no cells"); return
        print(f"{label}: n={len(g)}  median={np.median(g):.3f}  p90={np.percentile(g,90):.3f}  "
              f"p99={np.percentile(g,99):.3f}  max={g.max():.3f}  |  >5pp={np.mean(g>0.05):.1%}  "
              f">10pp={np.mean(g>0.10):.1%}  >15pp={np.mean(g>0.15):.1%}")

    print("\n=== winner-loser prevalence gap, |Δ| across (prompt type × unit) ===")
    summarize(fgap, "FEATURE  ")
    summarize(cgap, f"CLUSTER ({len(members)} behaviours)")

    # name the biggest cluster gaps
    beh = None
    if "behavior" in cl.columns:
        beh = cl.dropna(subset=["behavior"]).groupby("cluster_id")["behavior"].first().to_dict()
    cids = list(members)
    rows = []
    for x in np.unique(X):
        m = X == x
        if int(m.sum()) < args.min_battles:
            continue
        g = win_c[m].mean(axis=0) - lose_c[m].mean(axis=0)
        for j, c in enumerate(cids):
            rows.append((abs(g[j]), x, c, g[j]))
    rows.sort(reverse=True)
    print("\ntop cluster winner-loser gaps:")
    for ag, x, c, g in rows[:10]:
        print(f"  prompt {x:>2} · cluster {c:>3} gap={g:+.3f}  {str(beh.get(c,'') if beh else '')[:46]}")


if __name__ == "__main__":
    main()
