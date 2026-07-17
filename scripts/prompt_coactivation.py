#!/usr/bin/env python
"""Prompt-concept co-activation map — which prompt concepts fire together above chance.

Phase 1 of the compound-prompt-concept analysis: a "compound vocabulary" (english ∧
coding, translation ∧ bilingual). Descriptive/co-occurrence only (lift is symmetric,
not causal). See prefscope/analysis/coactivation.py.

    python scripts/prompt_coactivation.py --prompt-lens "$LP" \
        --prompt-names "$POUT/prompt_feature_names.csv" \
        --fidelity "$POUT/prompt_feature_fidelity.csv" --out "$LP/prompt_coactivation.csv"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prefscope.analysis.coactivation import prompt_coactivation      # noqa: E402
from prefscope.artifacts import Z_PROMPT                             # noqa: E402


def _names(path) -> dict:
    if not path or not Path(path).exists():
        return {}
    df = pd.read_csv(path)
    col = "concept" if "concept" in df.columns else df.columns[1]
    return {int(f): str(c) for f, c in zip(df["feature_id"], df[col])}


def main() -> None:
    ap = argparse.ArgumentParser(description="prompt-concept co-activation map")
    ap.add_argument("--prompt-lens", required=True, dest="prompt_lens",
                    help="prompt lens dir (z_prompt.npy)")
    ap.add_argument("--prompt-names", default=None, dest="prompt_names")
    ap.add_argument("--fidelity", default=None,
                    help="prompt_feature_fidelity.csv — restrict to verified prompt concepts")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-support", type=int, default=30, dest="min_support")
    ap.add_argument("--min-cooccur", type=int, default=5, dest="min_cooccur")
    args = ap.parse_args()

    z_prompt = np.load(Path(args.prompt_lens) / Z_PROMPT)
    features = None
    if args.fidelity and Path(args.fidelity).exists():
        fid = pd.read_csv(args.fidelity)
        if "fidelity_pass" in fid.columns:
            features = sorted(fid[fid["fidelity_pass"].astype(bool)]["feature_id"].astype(int))
            print(f"restricting to {len(features)} verified prompt concepts", flush=True)

    df = prompt_coactivation(z_prompt, features=features,
                             min_support=args.min_support, min_cooccur=args.min_cooccur)
    names = _names(args.prompt_names)
    df.insert(1, "name_a", df["concept_a"].map(lambda i: names.get(int(i), f"prompt {i}")))
    df.insert(3, "name_b", df["concept_b"].map(lambda i: names.get(int(i), f"prompt {i}")))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    sig = df[df["significant"].astype(bool)] if "significant" in df.columns else df
    print(f"wrote {len(df)} pairs ({len(sig)} significant) to {args.out}")
    print("\ntop co-firing prompt-concept pairs:")
    for r in df.sort_values("lift", ascending=False).head(15).itertuples():
        mark = "*" if getattr(r, "significant", False) else " "
        print(f" {mark} lift {r.lift:5.2f}  {str(r.name_a)[:34]:34s} ∧  {str(r.name_b)[:34]}")


if __name__ == "__main__":
    main()
