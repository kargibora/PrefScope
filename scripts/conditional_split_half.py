#!/usr/bin/env python
"""Split-half reliability of the conditional δ_{f,k} rankings.

Go/no-go for prompt-conditional feature selection: within each prompt type, is
the ranking of features by δ_{f,k} *reliably* different from the pooled ranking,
or does it just look different because small strata are noisy?

For each prompt type k the battles are split in half R times. Each half yields a
δ ranking. Per type:

  - **self**:  mean Spearman between the two half rankings — the reliability
    ceiling of the type's own δ vector;
  - **cross_matched**: mean Spearman between a half ranking and a δ computed on a
    subsample of the non-type-k battles **of the same size as the half**. Matching
    the sample size matches the precision, so under the null (the type-k ranking
    equals the pooled ranking) E[self] == E[cross_matched] and the go/no-go
    quantity ``excess = self − cross_matched`` is centred at 0. A positive excess
    for a meaningful set of types ⇒ the conditional structure is real.

(Comparing self to the *full* non-k pool — as an earlier version did — is a
precision mismatch: the huge pool is near-noiseless, so the null itself predicts
cross > self by a √-reliability gap, and "self ≈ cross → noise" would be
backwards. ``cross_full`` is still reported, but only as descriptive alignment to
the stable global ranking, never as the test.) Top-5 overlaps mirror all three.

    python scripts/conditional_split_half.py \
        --completion-lens "$LM" --prompt-lens "$LP" --corpus "$CORPUS" \
        --conditional "$LM/conditional_win_relevance.csv" --out-dir "$LM/validation"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prefscope.artifacts import Z_DIFF, Z_PROMPT, lens_battle_ids  # noqa: E402
from prefscope.data.corpus import load_corpus                      # noqa: E402
from prefscope.data.pair_schema import LABEL, RESPONSE_A, RESPONSE_B  # noqa: E402
from prefscope.pipeline.winrelevance import win_relevance_logistic  # noqa: E402

TOPK = 5


def _delta(z, y, length, feats) -> pd.Series:
    wr = win_relevance_logistic(z, y, length, features=feats)
    return wr.set_index("feature_id")["delta_win_rate"]


def _top(s: pd.Series, k: int = TOPK) -> set:
    return set(s.dropna().sort_values(ascending=False).head(k).index)


def split_half_reliability(idx, pool_idx, delta_fn, rng, *, n_splits=5,
                           min_valid=10, top_k=TOPK):
    """Precision-matched split-half reliability of a per-type δ ranking.

    ``delta_fn(indices) -> pd.Series`` maps a set of battle rows to a feature→δ
    ranking. ``idx`` are the type-k rows; ``pool_idx`` the rows NOT of type k.

    Returns per-type means of:
      - ``self``: Spearman between the two type-k halves (the type's reliability);
      - ``cross_matched``: Spearman between a half and a pool subsample **of the
        same size as the half** — precision-matched, so under the null (type-k
        ranking == pooled ranking) ``E[self] == E[cross_matched]`` and the
        go/no-go quantity ``self − cross_matched`` is centred at 0. (Comparing to
        the *full* pool instead inflates cross by √-reliability, so the null would
        predict cross > self and a "no signal" read would be unsupported.)
      - ``cross_full``: Spearman between a half and the full pool ranking — kept
        only as a descriptive "alignment to the stable global ranking", NOT the
        significance test.
    plus the top-``k`` overlap analogues.
    """
    pool_full = delta_fn(pool_idx)
    keys = ("self", "cross_matched", "cross_full")
    rho = {k: [] for k in keys}
    ov = {k: [] for k in keys}
    for _ in range(n_splits):
        half = rng.permutation(idx)
        h1, h2 = half[: len(half) // 2], half[len(half) // 2:]
        if min(len(h1), len(h2)) < 1:
            continue
        d1, d2 = delta_fn(h1), delta_fn(h2)
        m = min(len(h1), len(pool_idx))
        pool_sub = delta_fn(rng.choice(pool_idx, size=m, replace=False))
        ok = d1.notna() & d2.notna() & pool_full.notna() & pool_sub.notna()
        if ok.sum() < min_valid:
            continue
        rho["self"].append(spearmanr(d1[ok], d2[ok])[0])
        rho["cross_matched"].append(
            np.mean([spearmanr(d[ok], pool_sub[ok])[0] for d in (d1, d2)]))
        rho["cross_full"].append(
            np.mean([spearmanr(d[ok], pool_full[ok])[0] for d in (d1, d2)]))
        ov["self"].append(len(_top(d1) & _top(d2)) / top_k)
        ov["cross_matched"].append(
            np.mean([len(_top(d) & _top(pool_sub)) / top_k for d in (d1, d2)]))
        ov["cross_full"].append(
            np.mean([len(_top(d) & _top(pool_full)) / top_k for d in (d1, d2)]))
    if not rho["self"]:
        return None
    out = {f"{k}_spearman": float(np.mean(rho[k])) for k in keys}
    out.update({f"{k}_top5_overlap": float(np.mean(ov[k])) for k in keys})
    out["n_valid_splits"] = len(rho["self"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="split-half reliability of δ_{f,k}")
    ap.add_argument("--completion-lens", required=True, dest="completion_lens")
    ap.add_argument("--prompt-lens", required=True, dest="prompt_lens")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--conditional", required=True,
                    help="conditional_win_relevance.csv — defines the tested "
                         "features and prompt types")
    ap.add_argument("--out-dir", required=True, dest="out_dir")
    ap.add_argument("--n-splits", type=int, default=5, dest="n_splits")
    ap.add_argument("--min-valid", type=int, default=10, dest="min_valid",
                    help="minimum features with a finite δ in both halves for a "
                         "split to count (a rank correlation on <10 points is noise)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cond = pd.read_csv(args.conditional)
    feats = sorted(cond["feature_id"].astype(int).unique().tolist())
    types = sorted(cond["prompt_concept"].astype(int).unique().tolist())
    print(f"{len(feats)} features x {len(types)} prompt types from {args.conditional}",
          flush=True)

    # --- align completion and prompt lenses by battle id (as prompt_delta does) ---
    clens, plens = Path(args.completion_lens), Path(args.prompt_lens)
    z_diff = np.load(clens / Z_DIFF)
    z_prompt = np.load(plens / Z_PROMPT)
    cb, pb = lens_battle_ids(clens), lens_battle_ids(plens)
    if not (len(cb) == len(pb) and bool((cb == pb).all())):
        common = pd.Index(cb).intersection(pd.Index(pb))
        cpos = {b: i for i, b in enumerate(cb)}
        ppos = {b: i for i, b in enumerate(pb)}
        z_diff = z_diff[np.array([cpos[b] for b in common])]
        z_prompt = z_prompt[np.array([ppos[b] for b in common])]
        cb = common.to_numpy()

    corp = load_corpus(args.corpus)
    if LABEL not in corp.columns:
        sys.exit("corpus has no human_pref; rebuild with `build-corpus --keep-labels`")
    ci = corp.assign(battle_id=corp["battle_id"].astype(str)).set_index("battle_id")
    bids = pd.Series(cb).astype(str)
    y = bids.map(ci[LABEL]).to_numpy(dtype=float)
    _wc = lambda c: ci[c].reindex(bids).fillna("").astype(str).str.split().str.len().to_numpy(float)  # noqa: E731
    length = _wc(RESPONSE_A) - _wc(RESPONSE_B)
    concept = z_prompt.argmax(axis=1)                    # raw prompt concepts

    # The reconstructed types are argmax prompt-feature ids (matching how
    # prompt_delta.py builds the RAW conditional_win_relevance.csv). The *clustered*
    # variant stores prompt-cluster ids in a different id space; validating it here
    # would silently compare mismatched keys. Refuse rather than mislead.
    universe = set(np.unique(concept).tolist())
    unknown = [k for k in types if k not in universe]
    if len(unknown) > len(types) // 2:
        sys.exit(f"{len(unknown)}/{len(types)} prompt_concept ids in "
                 f"{args.conditional} are not argmax prompt-feature ids — this looks "
                 "like the CLUSTERED conditional CSV. Pass the raw "
                 "conditional_win_relevance.csv (argmax ids).")

    keep = np.isfinite(y)
    z_diff, y, length, concept = z_diff[keep], y[keep], length[keep], concept[keep]
    print(f"{len(y)} labeled battles", flush=True)

    rng = np.random.default_rng(args.seed)
    delta_fn = lambda ind: _delta(z_diff[ind], y[ind], length[ind], feats)  # noqa: E731
    rows = []
    for k in types:
        mask = concept == k
        idx = np.flatnonzero(mask)
        pool_idx = np.flatnonzero(~mask)                 # non-type-k rows (held out)
        rel = split_half_reliability(idx, pool_idx, delta_fn, rng,
                                     n_splits=args.n_splits, min_valid=args.min_valid)
        if rel is None:
            continue
        rows.append({"prompt_concept": k, "n_battles": int(mask.sum()), **rel})
        r = rows[-1]
        print(f"type {k:>3} (n={r['n_battles']:>6}): self ρ={r['self_spearman']:+.3f} "
              f"cross_m ρ={r['cross_matched_spearman']:+.3f} "
              f"(full {r['cross_full_spearman']:+.3f}) | "
              f"excess={r['self_spearman'] - r['cross_matched_spearman']:+.3f}", flush=True)

    df = pd.DataFrame(rows)
    if df.empty:
        sys.exit("no prompt type had enough support for a split-half estimate; "
                 "lower --min-valid or --n-splits, or check the conditional CSV")
    # go/no-go uses the precision-MATCHED cross (null-centred at 0); cross_full is
    # reported for context only.
    df["excess_spearman"] = df["self_spearman"] - df["cross_matched_spearman"]
    df["excess_top5"] = df["self_top5_overlap"] - df["cross_matched_top5_overlap"]
    df.to_csv(out_dir / "conditional_split_half.csv", index=False)

    summary = {
        "n_types": len(df), "n_features": len(feats), "n_splits": args.n_splits,
        "median_self_spearman": float(df["self_spearman"].median()),
        "median_cross_matched_spearman": float(df["cross_matched_spearman"].median()),
        "median_cross_full_spearman": float(df["cross_full_spearman"].median()),
        "median_excess_spearman": float(df["excess_spearman"].median()),
        "n_types_excess_gt_0": int((df["excess_spearman"] > 0).sum()),
        "n_types_excess_gt_0.1": int((df["excess_spearman"] > 0.1).sum()),
        "median_excess_top5": float(df["excess_top5"].median()),
    }
    (out_dir / "conditional_split_half_summary.json").write_text(
        json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nwrote conditional_split_half.csv + summary to {out_dir}")


if __name__ == "__main__":
    main()
