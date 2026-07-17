#!/usr/bin/env python
"""Export a self-contained ``model_compare.json`` for the viewer's Model-vs-Model tab.

Consumes an ``encode-dataset`` codes bundle (``z_a``/``z_b``/``z_diff`` + ``meta``)
and the lens's names, and emits, PER DATASET (``source``, joined from the corpus):

  - the **battle graph** — which models fought which (adjacency ``pairs`` with
    battle counts), so the viewer can offer only *reachable* opponents and split
    disconnected datasets into tabs;
  - **per-model power** — each model's mean activation and fire rate on every named
    concept, aggregated over all its responses in that dataset ("what it does");
  - **per-pair contrast** — for each battled pair, the mean ``z_diff`` (a expresses
    more / b expresses more) and each side's power over their shared battles, plus
    the winner-oriented contrast (descriptively rewarded).

Only NAMED (and, by default, VERIFIED) concepts are emitted, so the viewer never
shows a raw feature id.

    python scripts/export_model_compare.py \
        --encoded results_judgearena/encoded_qwen --lens results_judgearena/lens \
        --corpus corpora/judgearena_qwen.parquet \
        --out viewer-web/public/data/model_compare.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prefscope.data.pair_schema import LABEL, MODEL_A, MODEL_B, orient_by_label  # noqa: E402

R = 5  # rounding


def _round(x) -> float | None:
    v = float(x)
    return None if not np.isfinite(v) else round(v, R)


def _per_concept(mean_vec, fire_vec, ids) -> list[dict]:
    """Compact per-concept records for the named ids, sorted by fire rate desc."""
    rows = [{"f": int(f), "mean": _round(mean_vec[f]), "fire": _round(fire_vec[f])}
            for f in ids]
    return sorted(rows, key=lambda r: (r["fire"] or 0.0), reverse=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="export model_compare.json")
    ap.add_argument("--encoded", required=True, help="encode-dataset output dir")
    ap.add_argument("--lens", required=True, help="lens dir (feature_names.csv[, feature_fidelity.csv])")
    ap.add_argument("--corpus", required=True, help="corpus parquet (for the source/dataset per battle)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--all-features", action="store_true",
                    help="keep all named concepts, not just fidelity-verified ones")
    ap.add_argument("--min-pair-battles", type=int, default=1,
                    help="drop model pairs with fewer than this many battles")
    args = ap.parse_args()

    enc = Path(args.encoded)
    z_a = np.load(enc / "z_a.npy")
    z_b = np.load(enc / "z_b.npy")
    z_diff = np.load(enc / "z_diff.npy") if (enc / "z_diff.npy").exists() else z_a - z_b
    meta = pd.read_parquet(enc / "meta.parquet").reset_index(drop=True)
    n = len(meta)

    # dataset (source) per battle: join the encoded rows back to the corpus by row_id
    corpus = pd.read_parquet(args.corpus)
    if "row_id" in meta.columns and "source" in corpus.columns:
        src = corpus["source"].to_numpy()
        source = np.array([str(src[i]) if 0 <= int(i) < len(src) else "unknown"
                           for i in meta["row_id"].to_numpy()])
    else:
        source = np.array(["unknown"] * n)

    model_a = meta[MODEL_A].astype(str).to_numpy()
    model_b = meta[MODEL_B].astype(str).to_numpy()
    y = meta[LABEL].to_numpy(dtype=float) if LABEL in meta.columns else np.full(n, np.nan)

    # named (+ verified) concepts
    names = pd.read_csv(Path(args.lens) / "feature_names.csv")
    named = names[names["concept"].notna() & (names["concept"].astype(str).str.strip() != "")]
    keep = set(named["feature_id"].astype(int))
    fid_path = Path(args.lens) / "feature_fidelity.csv"
    if not args.all_features and fid_path.exists():
        fid = pd.read_csv(fid_path)
        if "fidelity_pass" in fid.columns:
            keep &= set(fid[fid["fidelity_pass"].astype(bool)]["feature_id"].astype(int))
    ids = sorted(keep)
    concepts = [{"f": int(fid_), "concept": str(dict(zip(named["feature_id"].astype(int),
                 named["concept"].astype(str)))[fid_])} for fid_ in ids]

    datasets = []
    for s in sorted(set(source.tolist())):
        smask = source == s
        sa, sb = model_a[smask], model_b[smask]
        za, zb, zd = z_a[smask], z_b[smask], z_diff[smask]
        ys = y[smask]

        models = sorted(set(sa.tolist()) | set(sb.tolist()))

        # per-model power: stack every response of the model (as A -> z_a, as B -> z_b)
        model_power, model_battles = {}, {}
        for m in models:
            am, bm = sa == m, sb == m
            stacked = np.concatenate([za[am], zb[bm]], axis=0) if (am.any() or bm.any()) else za[:0]
            model_battles[m] = int(am.sum() + bm.sum())
            fire = (stacked > 0).mean(axis=0) if len(stacked) else np.zeros(za.shape[1])  # +pole = present
            mean = stacked.mean(axis=0) if len(stacked) else np.zeros(za.shape[1])
            model_power[m] = _per_concept(mean, fire, ids)

        # per-pair contrast over shared battles (a = first in the sorted key)
        pairs, pair_contrast = [], {}
        seen = set()
        for i in range(len(sa)):
            key = tuple(sorted((sa[i], sb[i])))
            if key in seen:
                continue
            seen.add(key)
            a, b = key
            rows = ((sa == a) & (sb == b)) | ((sa == b) & (sb == a))
            nb = int(rows.sum())
            if nb < args.min_pair_battles:
                continue
            # orient every shared battle to (a - b): flip rows where meta has b as model_a
            flip = np.where((sa[rows] == a), 1.0, -1.0).reshape(-1, 1)
            d_ab = zd[rows] * flip                     # a-minus-b per battle
            za_ab = np.where((sa[rows] == a).reshape(-1, 1), za[rows], zb[rows])  # a's codes
            zb_ab = np.where((sa[rows] == a).reshape(-1, 1), zb[rows], za[rows])  # b's codes
            # winner-oriented (toward the judged winner), relative to a
            y_ab = np.where((sa[rows] == a), ys[rows], 1.0 - ys[rows])
            oriented, kmask = orient_by_label(y_ab, d_ab, drop_ties=True)
            mo = oriented.mean(axis=0) if oriented.shape[0] else np.full(za.shape[1], np.nan)
            contrast = d_ab.mean(axis=0)
            fa, fb = (za_ab > 0).mean(axis=0), (zb_ab > 0).mean(axis=0)  # +pole = present (not != 0)
            pa, pb = za_ab.mean(axis=0), zb_ab.mean(axis=0)
            pairs.append({"a": a, "b": b, "n": nb, "n_decisive": int(kmask.sum())})
            pair_contrast[f"{a}|{b}"] = [{
                "f": int(f), "contrast": _round(contrast[f]),
                "pa": _round(pa[f]), "pb": _round(pb[f]),
                "fa": _round(fa[f]), "fb": _round(fb[f]),
                "won": _round(mo[f])} for f in ids]

        datasets.append({
            "source": s, "n_battles": int(smask.sum()),
            "models": [{"name": m, "n": model_battles[m]} for m in models],
            "pairs": pairs, "model_power": model_power, "pair_contrast": pair_contrast,
        })

    out = {"concepts": concepts, "n_concepts": len(concepts),
           "verified_only": (not args.all_features), "datasets": datasets}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out))
    print(f"wrote {args.out}")
    for d in datasets:
        print(f"  {d['source']}: {len(d['models'])} models, {len(d['pairs'])} pairs, "
              f"{d['n_battles']} battles")


if __name__ == "__main__":
    main()
