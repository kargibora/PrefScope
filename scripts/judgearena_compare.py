#!/usr/bin/env python
"""A-vs-B concept comparison from an ``encode-dataset`` codes bundle.

Given a battle codes bundle (``z_a``/``z_b``/``z_diff`` + ``meta.parquet`` from
``prefscope encode-dataset``) and the lens's ``feature_names.csv``, report, per
named concept:

  - **A power** / **B power**  — each side's mean activation and fire rate on the
    concept (``z_a`` is model_a's response code, ``z_b`` is model_b's). This is the
    "what each model does" view.
  - **contrast**  = mean(z_diff) = mean(z_a − z_b). >0: model_a expresses the
    concept more; <0: model_b does. The head-to-head "what differs".
  - **winner_aligned**  — mean of z_diff oriented toward the judged winner
    (``orient_by_label``): >0 means the *preferred* side expresses the concept
    more, i.e. the concept is (descriptively) rewarded here. With a handful of
    battles this is a direction, NOT a significant effect — it's a pipeline check.

Only NAMED concepts are shown (and, with ``--fidelity``, only verified ones), so
the table never leaks raw feature ids. Everything is descriptive: this is built to
confirm the frozen lens produces a sensible concept read on out-of-distribution
data, not to make statistical claims on a tiny sample.

    python scripts/judgearena_compare.py \
        --encoded $OUT/encoded_qwen --lens $LENS \
        --out $OUT/encoded_qwen/ab_compare.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prefscope.data.pair_schema import LABEL, MODEL_A, MODEL_B, orient_by_label  # noqa: E402


def _one_model(meta: pd.DataFrame, col: str, default: str) -> str:
    if col in meta.columns:
        vals = meta[col].dropna().unique()
        if len(vals) == 1:
            return str(vals[0])
    return default


def main() -> None:
    ap = argparse.ArgumentParser(description="A-vs-B concept comparison")
    ap.add_argument("--encoded", required=True,
                    help="encode-dataset output dir (z_a.npy/z_b.npy/z_diff.npy + meta.parquet)")
    ap.add_argument("--lens", required=True, help="lens dir (for feature_names.csv)")
    ap.add_argument("--fidelity", default=None,
                    help="feature_fidelity.csv — restrict to verified concepts if given")
    ap.add_argument("--out", required=True, help="output CSV")
    ap.add_argument("--top", type=int, default=15, help="rows to print per section")
    args = ap.parse_args()

    enc = Path(args.encoded)
    z_a = np.load(enc / "z_a.npy")
    z_b = np.load(enc / "z_b.npy")
    z_diff = np.load(enc / "z_diff.npy") if (enc / "z_diff.npy").exists() else z_a - z_b
    meta = pd.read_parquet(enc / "meta.parquet")
    n = z_a.shape[0]

    names = pd.read_csv(Path(args.lens) / "feature_names.csv")
    named = names[names["concept"].notna() & (names["concept"].astype(str).str.strip() != "")]
    keep_ids = set(named["feature_id"].astype(int))
    if args.fidelity and Path(args.fidelity).exists():
        fid = pd.read_csv(args.fidelity)
        col = "fidelity_pass" if "fidelity_pass" in fid.columns else None
        if col:
            passed = set(fid[fid[col].astype(bool)]["feature_id"].astype(int))
            keep_ids &= passed
    concept = dict(zip(named["feature_id"].astype(int), named["concept"].astype(str)))

    ma = _one_model(meta, MODEL_A, "model_a")
    mb = _one_model(meta, MODEL_B, "model_b")

    # winner-oriented contrast (drops ties/unlabeled); descriptive on small n
    y = meta[LABEL].to_numpy(dtype=float) if LABEL in meta.columns else np.full(n, np.nan)
    oriented, keep = orient_by_label(y, z_diff, drop_ties=True)
    mean_oriented = oriented.mean(axis=0) if oriented.shape[0] else np.full(z_diff.shape[1], np.nan)

    rows = []
    for f in sorted(keep_ids):
        rows.append({
            "feature_id": f, "concept": concept[f],
            f"power_{ma}_mean": float(z_a[:, f].mean()),
            f"power_{mb}_mean": float(z_b[:, f].mean()),
            f"fire_{ma}": float((z_a[:, f] != 0).mean()),
            f"fire_{mb}": float((z_b[:, f] != 0).mean()),
            "contrast_a_minus_b": float(z_diff[:, f].mean()),
            "winner_aligned": float(mean_oriented[f]),
        })
    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    def _show(title, sort_col, ascending=False, cols=None):
        print(f"\n## {title}")
        sub = df.sort_values(sort_col, ascending=ascending).head(args.top)
        print(sub[cols or ["concept", sort_col]].to_string(index=False))

    print(f"{len(df)} named{' verified' if args.fidelity else ''} concepts over "
          f"{n} battles ({int(keep.sum())} decisive).  A={ma}  B={mb}")
    _show(f"{ma} expresses MORE than {mb} (contrast > 0)", "contrast_a_minus_b", False,
          ["concept", "contrast_a_minus_b", f"fire_{ma}", f"fire_{mb}"])
    _show(f"{mb} expresses MORE than {ma} (contrast < 0)", "contrast_a_minus_b", True,
          ["concept", "contrast_a_minus_b", f"fire_{ma}", f"fire_{mb}"])
    _show(f"{ma} individual — top concepts by fire rate", f"fire_{ma}", False,
          ["concept", f"fire_{ma}", f"power_{ma}_mean"])
    _show(f"{mb} individual — top concepts by fire rate", f"fire_{mb}", False,
          ["concept", f"fire_{mb}", f"power_{mb}_mean"])
    if int(keep.sum()):
        _show("Descriptively rewarded here (winner expresses more) — TINY n, direction only",
              "winner_aligned", False, ["concept", "winner_aligned", "contrast_a_minus_b"])
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
