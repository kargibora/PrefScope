#!/usr/bin/env python
"""JudgeArena task-run annotations.csv -> PrefScope corpus.

A JudgeArena *task run* (``--task alpaca-eval`` / ``arena-hard``) writes a
self-contained ``*-annotations.csv`` in its result folder: one row per battle
with the prompt, both completions inline, both model names, and per-criterion
judge scores (``adherence_A``/``adherence_B``, ``helpfulness_A``/…). That is
already a preference pair — this adapter maps it onto the PrefScope corpus schema
(``prompt / model_a / model_b / completion_a / completion_b`` + ``human_pref``)
and hands off to ``prefscope.data.corpus.normalize`` for id-hashing/validation.

``human_pref`` (= P(model_a preferred)) is derived exactly the way JudgeArena
aggregates criteria (``evaluate.parse_criteria_scores``: each side's score is the
mean over criteria), then hard-labelled by sign:

    score_a = mean(<criterion>_A over criteria) ; score_b = mean(<criterion>_B)
    human_pref = 1.0 if score_a > score_b  (A preferred)
                 0.0 if score_a < score_b  (B preferred)
                 0.5 if equal              (tie)

Note PrefScope's label is A-oriented (1 = A), the opposite of JudgeArena's own
B-oriented ``pref``; we derive from the scores directly so no flip is needed. The
criterion columns are auto-detected (any ``X_A``/``X_B`` numeric pair except the
``completion_``/``model_`` text columns), so this is preset-agnostic.

The graded per-side scores are a richer signal (relevant to the Soft-Elo work);
they are dropped here to keep the corpus to the standard schema, but see
``--keep-scores`` to carry them through as extra columns.

    python scripts/judgearena_to_corpus.py \
        --annotations $JA/results/<run>/<run>-annotations.csv \
        --out corpora/judgearena_qwen.parquet --source judgearena:alpaca-eval
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prefscope.data.corpus import CONTENT_COLS, make_battle_id, normalize  # noqa: E402

# annotations columns that end in _A/_B but are NOT criteria (they're the pair text)
_NON_CRITERION_BASES = {"completion", "model"}


def detect_criteria(cols: list[str]) -> list[str]:
    """Base names with a numeric X_A/X_B pair (the per-criterion judge scores)."""
    bases = []
    for c in cols:
        if c.endswith("_A"):
            base = c[:-2]
            if base in _NON_CRITERION_BASES:
                continue
            if f"{base}_B" in cols:
                bases.append(base)
    return bases


def _parse_score_scores(judge_completion: object) -> tuple[float, float]:
    """Parse ``score_A: X / score_B: Y`` from a raw judge completion (``default``/
    ``score`` preset), mirroring JudgeArena's ``PairScore._parse_score_scores``:
    strip <think> blocks, lower-case, regex-grab, reject out-of-[0,10] mis-grabs."""
    import re
    text = re.sub(r"<think>.*?</think>", " ", str(judge_completion), flags=re.S | re.I).lower()

    def grab(pat: str) -> float:
        m = re.search(pat, text)
        if not m:
            return np.nan
        v = float(m.group(1))
        return v if 0.0 <= v <= 10.0 else np.nan

    return grab(r'score.*?a[": *\n]*(-?\d+)'), grab(r'score.*?b[": *\n]*(-?\d+)')


def side_scores(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, list[str]]:
    """Mean per-side judge score (A, B), from whichever preset the run used:
    the expanded ``X_A``/``X_B`` criteria columns, else the ``judge_completion`` text."""
    criteria = detect_criteria(list(df.columns))
    if criteria:                                     # criteria preset -> averaged columns
        a = df[[f"{c}_A" for c in criteria]].apply(pd.to_numeric, errors="coerce")
        b = df[[f"{c}_B" for c in criteria]].apply(pd.to_numeric, errors="coerce")
        return a.mean(axis=1), b.mean(axis=1), criteria
    if "judge_completion" in df.columns:             # default/score preset -> parse text
        parsed = df["judge_completion"].apply(_parse_score_scores)
        sa = pd.Series([t[0] for t in parsed], index=df.index)
        sb = pd.Series([t[1] for t in parsed], index=df.index)
        return sa, sb, ["overall(score_A/score_B)"]
    sys.exit("no per-criterion X_A/X_B columns and no judge_completion to parse — "
             "cannot derive a preference from this annotations.csv.")


def load_annotations(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"instruction", "completion_A", "completion_B", "model_A", "model_B"}
    missing = need - set(df.columns)
    if missing:
        sys.exit(f"{path}: not a JudgeArena task-run annotations.csv "
                 f"(missing {sorted(missing)}). This adapter needs a task run "
                 "(alpaca-eval / arena-hard), not the ELO/arena battles.parquet.")

    score_a, score_b, criteria = side_scores(df)                 # preset-agnostic
    pref = np.where(score_a > score_b, 1.0,
                    np.where(score_a < score_b, 0.0, 0.5))
    pref = np.where(score_a.isna() | score_b.isna(), np.nan, pref)  # unscored -> unlabeled

    out = pd.DataFrame({
        "prompt": df["instruction"],
        "model_a": df["model_A"], "model_b": df["model_B"],
        "completion_a": df["completion_A"], "completion_b": df["completion_B"],
        "human_pref": pref,
    })
    if "language" in df.columns:
        out["language"] = df["language"]
    out["_score_a"], out["_score_b"] = score_a.to_numpy(), score_b.to_numpy()
    out.attrs["criteria"] = criteria
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="JudgeArena annotations.csv -> PrefScope corpus")
    ap.add_argument("--annotations", required=True, nargs="+",
                    help="one or more JudgeArena task-run *-annotations.csv files")
    ap.add_argument("--out", required=True, help="output corpus .parquet")
    ap.add_argument("--source", default="judgearena",
                    help="source tag stored on every battle (e.g. judgearena:alpaca-eval)")
    ap.add_argument("--language", default="en",
                    help="fallback language when the annotations have no language column")
    ap.add_argument("--keep-scores", action="store_true",
                    help="also carry the mean per-side judge scores (_score_a/_score_b)")
    args = ap.parse_args()

    frames, crit = [], None
    for p in args.annotations:
        raw = load_annotations(Path(p))
        crit = crit or raw.attrs.get("criteria")
        if "language" not in raw.columns:
            raw["language"] = args.language
        base = raw.drop(columns=["_score_a", "_score_b"])
        norm = normalize(base, args.source)        # recomputes content battle_id, drops empties
        if args.keep_scores:
            sc = raw[["_score_a", "_score_b"]].copy()
            sc["battle_id"] = [make_battle_id(r) for r in base[CONTENT_COLS].to_dict("records")]
            norm = norm.merge(sc.drop_duplicates("battle_id"), on="battle_id", how="left")
        frames.append(norm)
        print(f"  {Path(p).name}: {len(norm)} battles kept", flush=True)

    corpus = pd.concat(frames, ignore_index=True).drop_duplicates("battle_id").reset_index(drop=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    corpus.to_parquet(out, index=False)

    y = corpus["human_pref"]
    n_a = int((y == 1.0).sum()); n_b = int((y == 0.0).sum())
    n_tie = int((y == 0.5).sum()); n_na = int(y.isna().sum())
    print(f"\nwrote {len(corpus)} battles to {out}")
    print(f"  criteria used: {crit}")
    print(f"  human_pref — A wins: {n_a}  B wins: {n_b}  tie: {n_tie}  unlabeled: {n_na}")
    print(f"  models: {sorted(set(corpus['model_a']) | set(corpus['model_b']))}")
    print(f"  languages: {sorted(corpus['language'].dropna().unique().tolist())}")


if __name__ == "__main__":
    main()
