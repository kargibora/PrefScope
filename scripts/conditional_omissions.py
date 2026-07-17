#!/usr/bin/env python
"""Run the conditional concept-omission diagnosis on a built lens.

Joins the completion lens's per-side codes (z_a/z_b) with the prompt lens's
per-battle prompt type (argmax z_prompt, aligned by battle id) and the two gate
tables (elicitation + conditional Δwin), then computes, per (model, prompt-type,
concept), the paired-opponent production shortfall and flags the survivors.
See ``prefscope/analysis/omission.py`` for the statistic.

    python scripts/conditional_omissions.py \
        --completion-lens "$LM" --prompt-lens "$LP" --corpus "$CORPUS" \
        --elicitation "$LM/prompt_response_elicitation.csv" \
        --conditional "$LM/conditional_win_relevance.csv" \
        --out "$LM/omissions.csv"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prefscope.analysis.omission import conditional_omissions, gate_candidates  # noqa: E402
from prefscope.artifacts import Z_PROMPT, lens_battle_ids                        # noqa: E402
from prefscope.data.corpus import load_corpus                                    # noqa: E402
from prefscope.data.pair_schema import LABEL, MODEL_A, MODEL_B                    # noqa: E402


def _load(clens: Path, name: str) -> np.ndarray:
    return np.load(clens / name)


def main() -> None:
    ap = argparse.ArgumentParser(description="conditional concept-omission diagnosis")
    ap.add_argument("--completion-lens", required=True, dest="completion_lens")
    ap.add_argument("--prompt-lens", required=True, dest="prompt_lens")
    ap.add_argument("--corpus", required=True, help="corpus with human_pref + model_a/model_b")
    ap.add_argument("--elicitation", required=True, help="prompt_response_elicitation.csv")
    ap.add_argument("--conditional", required=True, help="conditional_win_relevance.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-battles", type=int, default=300, dest="min_battles")
    ap.add_argument("--base-floor", type=float, default=0.15, dest="base_floor")
    ap.add_argument("--min-shortfall", type=float, default=0.05, dest="min_shortfall")
    args = ap.parse_args()

    clens, plens = Path(args.completion_lens), Path(args.prompt_lens)
    z_a, z_b = _load(clens, "z_a.npy"), _load(clens, "z_b.npy")
    z_prompt = np.load(plens / Z_PROMPT)
    cb, pb = lens_battle_ids(clens), lens_battle_ids(plens)

    # align completion rows to prompt rows by battle id (as conditional_split_half does)
    if not (len(cb) == len(pb) and bool((cb == pb).all())):
        common = pd.Index(cb).intersection(pd.Index(pb))
        cpos = {b: i for i, b in enumerate(cb)}
        ppos = {b: i for i, b in enumerate(pb)}
        ci = np.array([cpos[b] for b in common])
        z_a, z_b = z_a[ci], z_b[ci]
        z_prompt = z_prompt[np.array([ppos[b] for b in common])]
        cb = common.to_numpy()
    prompt_type = z_prompt.argmax(axis=1)

    # corpus attrs aligned by battle id
    corp = load_corpus(args.corpus).assign(battle_id=lambda d: d["battle_id"].astype(str))
    corp = corp.set_index("battle_id")
    bids = pd.Series(cb).astype(str)
    model_a = bids.map(corp[MODEL_A]).to_numpy()
    model_b = bids.map(corp[MODEL_B]).to_numpy()
    human_pref = bids.map(corp[LABEL]).to_numpy(dtype=float)

    keep = ~pd.isna(model_a) & ~pd.isna(model_b) & np.isfinite(human_pref)
    z_a, z_b, prompt_type = z_a[keep], z_b[keep], prompt_type[keep]
    model_a, model_b, human_pref = model_a[keep], model_b[keep], human_pref[keep]
    print(f"{keep.sum()} labeled battles, {len(set(prompt_type.tolist()))} prompt types, "
          f"{len(set(model_a) | set(model_b))} models", flush=True)

    cands = gate_candidates(pd.read_csv(args.elicitation), pd.read_csv(args.conditional))
    print(f"{len(cands)} candidate (prompt, response) cells "
          "(elicited & rewarded)", flush=True)
    if not cands:
        sys.exit("no candidate cells — nothing is both elicited and rewarded; check the gate CSVs")

    df = conditional_omissions(
        z_a, z_b, model_a, model_b, prompt_type, human_pref, cands,
        min_battles=args.min_battles, base_floor=args.base_floor,
        min_shortfall=args.min_shortfall)
    df.to_csv(args.out, index=False)

    n_flag = int(df["flagged"].sum()) if not df.empty else 0
    print(f"\nwrote {len(df)} evaluated cells to {args.out} — {n_flag} flagged omissions")
    if n_flag:
        top = df[df["flagged"]].head(15)
        print(top[["model", "prompt_concept", "feature_id", "n", "expected",
                   "produced", "shortfall", "won_when_fired", "won_when_not"]].to_string(index=False))


if __name__ == "__main__":
    main()
