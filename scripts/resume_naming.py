"""Resume / extend feature naming without recomputing already-named features.

`prefscope interpret name` overwrites feature_names.csv and has no fire-rate
top-N selector. This helper closes both gaps:

  list   compute the top-N features by fire rate (fraction of responses where the
         feature fires, averaged over z_a/z_b), drop any already named, and print
         the remaining IDs as a space-separated list for `--features`.

  merge  fold a freshly-named CSV (from a --features run) into the canonical
         feature_names.csv (or any *_fidelity.csv), dedup on feature_id keeping
         the new rows. Same overwrite-safe merge works for feature_fidelity.csv.

Fire rate is chunked over rows so it is safe on the full Arena z arrays.

Usage:
    # 1. which unnamed features are in the top 300 by fire rate?
    python scripts/resume_naming.py list --lens-dir $LENS --top 300
    #    -> prints: 512 88 1907 ...   (feed straight into --features)

    # 2. after `interpret name --features <those> --out names_new.csv`
    python scripts/resume_naming.py merge --into $LENS/feature_names.csv \
        --new names_new.csv
    # 3. same after `interpret verify --features <those> --out fidelity_new.csv`
    python scripts/resume_naming.py merge --into $LENS/feature_fidelity.csv \
        --new fidelity_new.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from prefscope.artifacts import FEATURE_NAMES, Z_A, Z_B  # noqa: E402


def _fire_rate(lens_dir: Path, *, chunk: int = 20000) -> np.ndarray:
    """Per-feature fraction of responses (both sides) where the code is non-zero."""
    za = np.load(lens_dir / Z_A, mmap_mode="r")
    zb = np.load(lens_dir / Z_B, mmap_mode="r")
    m = za.shape[1]
    fired = np.zeros(m, dtype=np.int64)
    n = 0
    for arr in (za, zb):
        for i in range(0, arr.shape[0], chunk):
            block = np.asarray(arr[i : i + chunk])
            fired += (block != 0).sum(axis=0)
            n += block.shape[0]
    return fired / max(n, 1)


def _cmd_list(args: argparse.Namespace) -> None:
    lens = Path(args.lens_dir)
    fire = _fire_rate(lens)
    order = np.argsort(-fire)[: args.top]
    named: set[int] = set()
    names_csv = Path(args.names) if args.names else lens / FEATURE_NAMES
    if names_csv.exists():
        df = pd.read_csv(names_csv)
        # a feature counts as "named" only if it actually has a concept string
        col = "concept" if "concept" in df.columns else None
        rows = df[df[col].notna() & (df[col].astype(str) != "")] if col else df
        named = set(rows["feature_id"].astype(int))
    todo = [int(i) for i in order if int(i) not in named]
    print(
        f"# top {args.top} by fire rate: {len(order)} feats, "
        f"{len(named)} already named, {len(todo)} to name",
        file=sys.stderr,
    )
    print(" ".join(map(str, todo)))


def _cmd_merge(args: argparse.Namespace) -> None:
    into = Path(args.into)
    new = pd.read_csv(args.new)
    if into.exists():
        old = pd.read_csv(into)
        merged = pd.concat([old, new], ignore_index=True).drop_duplicates(
            "feature_id", keep="last"
        )
    else:
        merged = new
    merged = merged.sort_values("feature_id").reset_index(drop=True)
    merged.to_csv(into, index=False)
    print(f"merged {len(new)} new rows -> {into} ({len(merged)} total)", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    pl = sub.add_parser("list", help="print unnamed top-N-by-fire-rate feature IDs")
    pl.add_argument("--lens-dir", required=True)
    pl.add_argument("--top", type=int, default=300)
    pl.add_argument("--names", default=None, help="feature_names.csv (default: <lens>/feature_names.csv)")
    pl.set_defaults(func=_cmd_list)

    pm = sub.add_parser("merge", help="fold a new CSV into a canonical one (dedup on feature_id)")
    pm.add_argument("--into", required=True)
    pm.add_argument("--new", required=True)
    pm.set_defaults(func=_cmd_merge)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
