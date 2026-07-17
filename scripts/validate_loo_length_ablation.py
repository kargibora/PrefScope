#!/usr/bin/env python
"""LOO predictive validation + length-feature ablation.

Runs the honest headline number for the per-model diagnosis — leave-one-model-out
R^2 (with bootstrap CI and permutation p, from ``validate_diagnosis``) — and then
re-runs it with length-tracking features REMOVED from the predictor.

Why the ablation: the LOO weights are length-controlled (logistic AME), but the
per-model ``net_direction`` profile is not — a model that simply writes longer
answers over-expresses every length-tracking feature, so a high R^2 could be
verbosity rediscovered through the lens. If R^2 survives with those features
dropped, the diagnosis carries signal beyond length.

Feature-length tracking comes from the bias screen (``scripts/bias_screen.py``):
``corr_confound_len`` = correlation of the feature's A-vs-B direction with the
A-minus-B length gap. Features with |corr| >= each ``--thresholds`` value are
dropped in turn (0.3 matches ``auto_undesirable``'s default; 0.1 is aggressive).

    python scripts/validate_loo_length_ablation.py \
        --bank "$LENS/bank" --win-relevance "$LENS/win_relevance.csv" \
        --bias-screen "$LENS/bias_screen.csv" --out-dir "$LENS/validation"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prefscope.pipeline.oriented_bank import load_bank      # noqa: E402
from prefscope.pipeline.validate import validate_diagnosis  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="LOO validation + length ablation")
    ap.add_argument("--bank", required=True, help="oriented-code bank dir (build-bank)")
    ap.add_argument("--win-relevance", required=True, dest="win_relevance")
    ap.add_argument("--bias-screen", required=True, dest="bias_screen",
                    help="bias_screen.csv with corr_confound_len per feature")
    ap.add_argument("--out-dir", required=True, dest="out_dir")
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.3, 0.1],
                    help="drop features with |corr_confound_len| >= t, one run per t")
    ap.add_argument("--weight-col", default="delta_win_rate", dest="weight_col",
                    help="win-relevance column to weight features by")
    ap.add_argument("--min-battles", type=int, default=20, dest="min_battles")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bank_Z, bank_meta, bank_manifest = load_bank(args.bank)
    wr = pd.read_csv(args.win_relevance)
    bias = pd.read_csv(args.bias_screen)
    if "corr_confound_len" not in bias.columns:
        sys.exit(f"{args.bias_screen} has no corr_confound_len column")
    len_corr = bias.set_index("feature_id")["corr_confound_len"]

    # A feature absent from the bias screen is KEPT (treated as "no evidence of
    # length tracking"). That's only safe if the screen covers every win-relevance
    # feature; otherwise a length-tracker could silently survive the ablation.
    missing = set(wr["feature_id"].astype(int)) - set(bias["feature_id"].astype(int))
    if missing:
        sys.exit(f"{len(missing)} win-relevance features are absent from the bias "
                 f"screen (e.g. {sorted(missing)[:5]}) — cannot decide if they track "
                 "length. Regenerate bias_screen.csv over all features.")

    # per-model mean length gap (self − other, word count) — the verbosity signal
    # the ablation is meant to strip. If R^2 survives ablation but the surviving
    # predicted score still tracks this, length wasn't really removed.
    model_len = (bank_meta.groupby("self_model")["length"].mean()
                 if "length" in bank_meta.columns else None)

    runs: dict[str, dict] = {}

    def _run(tag: str, wr_sub: pd.DataFrame) -> None:
        print(f"\n=== {tag}: {len(wr_sub)} features, LOO over "
              f"{bank_manifest['n_models']} models ===", flush=True)
        df, summary = validate_diagnosis(
            bank_Z, bank_meta, wr_sub, weight_col=args.weight_col,
            min_battles=args.min_battles, loo=True, seed=args.seed)
        # empirical length-leak check: does the predicted per-model score still
        # correlate with how much longer that model's answers are?
        if model_len is not None:
            score_col = "predicted_score_loo" if "predicted_score_loo" in df else "predicted_score"
            merged = df.assign(_len=df["model"].map(model_len)).dropna(subset=["_len", score_col])
            if len(merged) > 2:
                summary["pred_vs_length_r"] = float(
                    merged[score_col].corr(merged["_len"]))
                summary["pred_vs_length_n"] = int(len(merged))
        df.to_csv(out_dir / f"per_model_{tag}.csv", index=False)
        summary["n_features"] = int(len(wr_sub))
        runs[tag] = summary
        print(json.dumps(summary, indent=2, default=str), flush=True)
        # persist after every run so a wall-clock kill loses nothing finished
        (out_dir / "validation_summary.json").write_text(
            json.dumps(runs, indent=2, default=str))

    _run("baseline", wr)
    for t in args.thresholds:
        # features absent from the bias screen are KEPT (no evidence of tracking)
        tracked = len_corr[len_corr.abs() >= t].index
        wr_sub = wr[~wr["feature_id"].isin(tracked)].reset_index(drop=True)
        print(f"\nthreshold {t}: dropping {len(wr) - len(wr_sub)} length-tracking "
              f"features ({len(wr_sub)} remain)", flush=True)
        _run(f"len_ablated_{t:g}", wr_sub)

    print("\n=== comparison ===")
    header = (f"{'run':<20} {'n_feat':>6} {'LOO R2':>8} {'[95% CI]':>18} "
              f"{'perm p':>8} {'rho':>7} {'r(len)':>7}")
    print(header)
    for tag, s in runs.items():
        print(f"{tag:<20} {s['n_features']:>6} {s.get('loo_r2', float('nan')):>8.3f} "
              f"[{s.get('loo_r2_ci_lo', float('nan')):.3f}, "
              f"{s.get('loo_r2_ci_hi', float('nan')):.3f}]"
              f" {s.get('loo_r2_perm_p', float('nan')):>8.4f}"
              f" {s.get('loo_spearman', float('nan')):>7.3f}"
              f" {s.get('pred_vs_length_r', float('nan')):>7.3f}")
    print(f"\nwrote per-model CSVs + validation_summary.json to {out_dir}")


if __name__ == "__main__":
    main()
