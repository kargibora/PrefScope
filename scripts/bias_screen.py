#!/usr/bin/env python
"""Confound screen — which rewarded completion features co-vary with length/style.

For each completion feature this reports, over the battles where it fires:
  - win_assoc / correlation : how strongly its A-vs-B direction tracks human_pref
  - corr_confound_len        : correlation of its direction with the length surrogate
                               len(completion_a) - len(completion_b)
  - correlation_resid_len    : the reward correlation AFTER partialling out length
                               (partial correlation) — the honest "still rewarded
                               once length is controlled" signal
  - confound_entangled       : flag = strong length covariance AND the reward
                               correlation largely collapses after residualizing

This is a SCREENING tool, not a bias verdict: a high corr_confound_len means you
cannot separate genuine quality from the length artifact, not that the feature is
fake. Confirming a bias needs an intervention (downweight + refit), out of scope here.

    python scripts/bias_screen.py --lens-dir "$LENS" --corpus "$CORPUS" \
        --names "$OUT/feature_fidelity.csv" --out "$OUT/bias_screen.csv" --permute 50
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prefscope.analysis.dataset import feature_confound_correlation  # noqa: E402
from prefscope.interpret.io import load_lens_battles                 # noqa: E402
from prefscope.pipeline.winrelevance import win_relevance            # noqa: E402

CONFOUND_THR = 0.3       # |corr| with length at/above this = strong covariance
COLLAPSE_FRAC = 0.5      # reward corr shrinks below this fraction after residualizing


def _partial_corr(x, y, z) -> float:
    """Partial correlation of x and y controlling for z (NaN if undefined)."""
    if min(np.std(x), np.std(y), np.std(z)) == 0:
        return float("nan")
    rxy = np.corrcoef(x, y)[0, 1]
    rxz = np.corrcoef(x, z)[0, 1]
    ryz = np.corrcoef(y, z)[0, 1]
    denom = np.sqrt((1 - rxz**2) * (1 - ryz**2))
    return float((rxy - rxz * ryz) / denom) if denom > 0 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description="length-residualized confound screen")
    ap.add_argument("--lens-dir", required=True, help="completion lens dir (z_diff.npy)")
    ap.add_argument("--corpus", required=True, help="corpus WITH human_pref (--keep-labels)")
    ap.add_argument("--names", default=None, help="feature_fidelity.csv (concept + fidelity_pass)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--permute", type=int, default=0, metavar="N",
                    help="shuffle human_pref N times -> null count of 'significant' features")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    battles, z_diff, _ = load_lens_battles(args.lens_dir, corpus=args.corpus)
    if "human_pref" not in battles.columns or battles["human_pref"].isna().all():
        sys.exit("corpus has no human_pref; rebuild with `build-corpus --keep-labels`")
    y = battles["human_pref"].to_numpy(dtype=float)
    yc = 2.0 * y - 1.0                                   # +1 A preferred, -1 B, 0 tie

    # length surrogate: chosen/rejected sit in A/B; z>0 = A. char length is fine.
    la = battles["completion_a"].fillna("").str.len().to_numpy(dtype=float)
    lb = battles["completion_b"].fillna("").str.len().to_numpy(dtype=float)
    length_diff = la - lb

    wr = win_relevance(z_diff, y)[["feature_id", "win_assoc", "correlation",
                                   "n_fire", "significant"]]
    conf = feature_confound_correlation(z_diff, length_diff).rename(
        columns={"corr": "corr_confound_len"})

    # partial correlation reward|length, per feature, over firing battles
    resid = []
    for f in range(z_diff.shape[1]):
        col = z_diff[:, f]
        m = col != 0
        resid.append({"feature_id": f,
                      "correlation_resid_len": _partial_corr(
                          np.sign(col[m]), yc[m], length_diff[m])})
    resid = pd.DataFrame(resid)

    df = wr.merge(conf, on="feature_id").merge(resid, on="feature_id")
    df["confound_entangled"] = (
        (df["corr_confound_len"].abs() >= CONFOUND_THR)
        & (df["correlation"].abs() > 0)
        & (df["correlation_resid_len"].abs() < COLLAPSE_FRAC * df["correlation"].abs()))

    if args.names and Path(args.names).exists():
        names = pd.read_csv(args.names)
        keep = [c for c in ("feature_id", "concept", "fidelity_pass") if c in names.columns]
        df = df.merge(names[keep], on="feature_id", how="left")
        front = [c for c in ("feature_id", "concept", "fidelity_pass") if c in df.columns]
        df = df[front + [c for c in df.columns if c not in front]]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df)} features to {args.out}; "
          f"{int(df['confound_entangled'].sum())} length-entangled "
          f"(strong length covariance + reward collapses after residualizing)")

    if args.permute > 0:
        rng = np.random.default_rng(args.seed)
        n_obs = int(wr["significant"].sum())
        null = np.array([int(win_relevance(z_diff, rng.permutation(y))["significant"].sum())
                         for _ in range(args.permute)])
        exceed = int((null >= n_obs).sum())
        print(f"\nhuman_pref-permutation null ({args.permute} shuffles): "
              f"significant features mean={null.mean():.1f}, "
              f"95th pct={np.percentile(null, 95):.0f}, max={null.max()}")
        print(f"observed={n_obs}  |  empirical p = {(exceed + 1) / (args.permute + 1):.4f}")


if __name__ == "__main__":
    main()
