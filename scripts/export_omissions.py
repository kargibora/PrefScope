#!/usr/bin/env python
"""Export omissions.csv -> omissions.json for the report card's per-model view.

Keyed by model so ReportCard can show, for the selected model, the response
concepts it under-produces per prompt type (flagged, corroboration-annotated).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _names(path: Path) -> dict:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    col = "concept" if "concept" in df.columns else df.columns[1]
    return {int(f): str(c) for f, c in zip(df["feature_id"], df[col])}


def main() -> None:
    ap = argparse.ArgumentParser(description="export omissions.json")
    ap.add_argument("--omissions", required=True, help="omissions.csv")
    ap.add_argument("--feature-names", required=True, dest="feature_names",
                    help="completion feature_names.csv (response concepts)")
    ap.add_argument("--prompt-names", required=True, dest="prompt_names",
                    help="prompt_feature_names.csv (prompt concepts)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--flagged-only", action="store_true", default=True)
    args = ap.parse_args()

    o = pd.read_csv(args.omissions)
    flagged = o[o["flagged"].astype(bool)] if "flagged" in o.columns else o
    cn = _names(Path(args.feature_names))
    pn = _names(Path(args.prompt_names))

    by_model: dict[str, list] = {}
    for r in flagged.itertuples():
        cell = {
            "x": int(r.prompt_concept), "f": int(r.feature_id), "n": int(r.n),
            "expected": round(float(r.expected), 3), "produced": round(float(r.produced), 3),
            "shortfall": round(float(r.shortfall), 3),
            "wf": None if pd.isna(r.won_when_fired) else round(float(r.won_when_fired), 3),
            "wn": None if pd.isna(r.won_when_not) else round(float(r.won_when_not), 3),
            "corroborated": bool((r.won_when_fired or 0) > (r.won_when_not or 0)),
        }
        # winners' prevalence (dual bar: winners vs this model), present with the new gate
        if "p_win" in flagged.columns:
            cell["p_win"] = None if pd.isna(r.p_win) else round(float(r.p_win), 3)
            cell["p_lose"] = None if pd.isna(r.p_lose) else round(float(r.p_lose), 3)
        by_model.setdefault(str(r.model), []).append(cell)
    for m in by_model:
        by_model[m].sort(key=lambda d: -d["shortfall"])

    concepts = {str(f): cn.get(f, f"feature {f}")
                for f in sorted(set(flagged["feature_id"].astype(int)))}
    prompt_concepts = {str(x): pn.get(x, f"prompt concept {x}")
                       for x in sorted(set(flagged["prompt_concept"].astype(int)))}
    out = {"concepts": concepts, "prompt_concepts": prompt_concepts,
           "by_model": by_model, "n_flagged": int(len(flagged)),
           "n_models_with_flags": len(by_model)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out))
    print(f"wrote {args.out}: {len(flagged)} flags across {len(by_model)} models")


if __name__ == "__main__":
    main()
