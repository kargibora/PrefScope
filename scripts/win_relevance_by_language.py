#!/usr/bin/env python
"""Language-stratified win-relevance — a Simpson's-paradox screen.

The pooled length-controlled Δwin-rate mixes battles across languages. If a
feature tracks a language (e.g. fires on Russian answers) and win rates differ
by language for other reasons, the pooled estimate can carry the wrong sign for
every single language. This script re-runs the same length-controlled logistic
AME (``win_relevance_logistic``) *within* each language stratum and reports,
per feature: the per-language estimates and whether any adequately-supported
stratum disagrees in sign with the pooled estimate.

Only features that are pooled-significant (``delta_win_significant``) are
re-estimated — the screen is about whether the *reported* effects survive
stratification, not about discovering new ones.

    python scripts/win_relevance_by_language.py \
        --lens-dir "$LENS" --corpus "$CORPUS" \
        --win-relevance "$LENS/win_relevance.csv" --out-dir "$LENS/validation"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prefscope.interpret.io import load_lens_battles            # noqa: E402
from prefscope.pipeline.winrelevance import win_relevance_logistic  # noqa: E402

# the corpus mixes ISO codes and English names (two source datasets)
_LANG_ALIASES = {
    "english": "en", "russian": "ru", "chinese": "zh", "german": "de",
    "polish": "pl", "spanish": "es", "french": "fr", "portuguese": "pt",
    "japanese": "ja", "korean": "ko", "italian": "it", "vietnamese": "vi",
    "turkish": "tr", "ukrainian": "uk", "arabic": "ar", "czech": "cs",
    "dutch": "nl", "indonesian": "id", "persian": "fa", "unknown": "und",
}


def normalize_language(s: pd.Series) -> pd.Series:
    low = s.fillna("und").astype(str).str.strip().str.lower()
    return low.map(lambda v: _LANG_ALIASES.get(v, v))


def main() -> None:
    ap = argparse.ArgumentParser(description="language-stratified win-relevance")
    ap.add_argument("--lens-dir", required=True, dest="lens_dir")
    ap.add_argument("--corpus", required=True, help="corpus WITH human_pref")
    ap.add_argument("--win-relevance", required=True, dest="win_relevance",
                    help="pooled win_relevance.csv (delta_win_rate + significance)")
    ap.add_argument("--out-dir", required=True, dest="out_dir")
    ap.add_argument("--min-decisive", type=int, default=2000, dest="min_decisive",
                    help="minimum decisive battles for a language stratum")
    ap.add_argument("--min-fire", type=int, default=30, dest="min_fire",
                    help="minimum firing battles in a stratum for a cell to be "
                         "eligible for the flip test (below this the LR test is "
                         "underpowered and its verdict is not trusted)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    battles, z_diff, _ = load_lens_battles(args.lens_dir, corpus=args.corpus)
    for col in ("human_pref", "language"):
        if col not in battles.columns or battles[col].isna().all():
            sys.exit(f"battles have no usable {col!r} column")

    pooled = pd.read_csv(args.win_relevance)
    if "delta_win_significant" not in pooled.columns:
        sys.exit(f"{args.win_relevance} has no delta_win_significant (need the "
                 "logistic win-relevance output)")
    sig = pooled[pooled["delta_win_significant"].astype(bool)]
    feats = sig["feature_id"].astype(int).tolist()
    pooled_delta = sig.set_index("feature_id")["delta_win_rate"]
    print(f"{len(feats)} pooled-significant features to re-estimate", flush=True)

    y = battles["human_pref"].to_numpy(dtype=float)
    # word-count length gap A−B — the SAME nuisance covariate the canonical
    # win_relevance_logistic / win_relevance.csv controls for (not char length),
    # so the per-stratum re-estimates are directly comparable to the pooled deltas.
    la = battles["completion_a"].fillna("").str.split().str.len().to_numpy(dtype=float)
    lb = battles["completion_b"].fillna("").str.split().str.len().to_numpy(dtype=float)
    length = la - lb
    lang = normalize_language(battles["language"])

    decisive = y != 0.5
    counts = lang[decisive].value_counts()
    strata = [l for l in counts.index if counts[l] >= args.min_decisive and l != "und"]
    print(f"strata (>= {args.min_decisive} decisive): "
          f"{[(l, int(counts[l])) for l in strata]}", flush=True)

    if not strata:
        sys.exit("no language stratum meets --min-decisive; lower the threshold")

    per_lang = []
    for l in strata:
        mask = (lang == l).to_numpy()
        zm = z_diff[mask]
        wr = win_relevance_logistic(zm, y[mask], length[mask], features=feats)
        wr["language"] = l
        wr["n_decisive"] = int((mask & decisive).sum())
        # support within the stratum: battles where the feature actually fires
        wr["n_fire"] = [int((zm[:, f] != 0).sum()) for f in feats]
        per_lang.append(wr)
    strat = pd.concat(per_lang, ignore_index=True)
    strat["pooled_delta"] = strat["feature_id"].map(pooled_delta)
    strat.to_csv(out_dir / "win_relevance_by_language.csv", index=False)

    # A cell (feature × stratum) is only *powered* to test a flip when the feature
    # actually fires enough within the stratum; below --min-fire the within-stratum
    # LR test is unreliable, so "no significant flip" there is absence of power, not
    # evidence of robustness. Classify every cell three ways and only count flips
    # among powered cells.
    powered = strat["n_fire"] >= args.min_fire
    opposite = (np.sign(strat["delta_win_rate"]) != np.sign(strat["pooled_delta"])) \
        & (strat["pooled_delta"].abs() > 0)
    flips = strat[powered & strat["delta_win_significant"].astype(bool) & opposite]
    n_cells = int(len(strat))
    n_underpowered = int((~powered).sum())
    summary = {
        "n_features": len(feats),
        "strata": {l: int(counts[l]) for l in strata},
        "min_fire": args.min_fire,
        "n_cells": n_cells,
        "n_cells_powered": int(powered.sum()),
        "n_cells_underpowered": n_underpowered,
        "n_sign_flips_significant": int(flips["feature_id"].nunique()),
        "flip_feature_ids": sorted(flips["feature_id"].astype(int).unique().tolist()),
        # honesty flag: if most cells are underpowered, a zero flip count is weak
        # evidence — the screen simply couldn't see a reversal.
        "underpowered_fraction": round(n_underpowered / n_cells, 3) if n_cells else None,
    }
    (out_dir / "language_stratification_summary.json").write_text(
        json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if len(flips):
        print("\nsign-flipping cells (significant within stratum, opposite to pooled):")
        cols = ["feature_id", "language", "delta_win_rate", "pooled_delta",
                "n_fire", "delta_win_p_bonferroni"]
        print(flips[cols].to_string(index=False))
    print(f"\nwrote win_relevance_by_language.csv + summary to {out_dir}")


if __name__ == "__main__":
    main()
