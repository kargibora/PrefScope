#!/usr/bin/env python
"""Build a per-dataset report-card OVERLAY through an already-trained lens.

One SAE, many datasets: the concept basis (features) is shared; each dataset is an
overlay of per-model diagnosis computed through the SAME lens. This turns an
``encode-dataset`` codes bundle into a real ``diagnosis.json`` + ``meta.json`` that
the viewer's Report card renders exactly like the Arena dataset — including prompt
concepts (prompt_types / relations) when prompt codes are supplied.

For the individual (completion) lens the oriented bank codes are ``z_a − z_b`` /
``z_b − z_a`` (lens_rep.IndividualLensRep.oriented_codes), so the bank is built from
the saved codes with NO re-embedding.

    python scripts/build_dataset_overlay.py \
        --encoded results_judgearena/encoded_qwen --lens results_judgearena/lens \
        --corpus corpora/judgearena_qwen.parquet \
        --prompt-enc results_judgearena/prompt_enc \
        --prompt-names results_judgearena/prompt_feature_names.csv \
        --dataset-id qwen --label "Qwen 9B vs 27B (alpaca smoke)" \
        --out-dir viewer-data/datasets/qwen
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prefscope.pipeline.oriented_bank import save_bank                # noqa: E402
from prefscope.viewer_export.diagnosis import export_diagnosis        # noqa: E402


def _clean(o):
    if isinstance(o, float):
        return None if (math.isnan(o) or math.isinf(o)) else o
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    return o


def build_bank_from_codes(z_a, z_b, battles: pd.DataFrame, out: Path) -> None:
    """Oriented bank from saved individual-lens codes (no embeddings/projector)."""
    z_a_self = (z_a - z_b).astype(np.float32)
    z_b_self = (z_b - z_a).astype(np.float32)
    y = battles["human_pref"].to_numpy(dtype=float)
    wc = lambda s: battles[s].fillna("").str.split().str.len().to_numpy(dtype=float)  # noqa: E731
    len_a = wc("completion_a") - wc("completion_b")
    extra = {c: battles[c].to_numpy() for c in ("instruction_id", "source")
             if c in battles.columns}
    a = pd.DataFrame({"orientation": "a", "self_model": battles["model_a"].to_numpy(),
                      "other_model": battles["model_b"].to_numpy(), "win": y,
                      "length": len_a, **extra})
    b = pd.DataFrame({"orientation": "b", "self_model": battles["model_b"].to_numpy(),
                      "other_model": battles["model_a"].to_numpy(), "win": 1.0 - y,
                      "length": -len_a, **extra})
    meta = pd.concat([a, b], ignore_index=True)
    Z = np.vstack([z_a_self, z_b_self]).astype(np.float32)
    save_bank(out, Z, meta, label_col="human_pref", input_rep="individual")


def main() -> None:
    ap = argparse.ArgumentParser(description="build a per-dataset report-card overlay")
    ap.add_argument("--encoded", required=True, help="encode-dataset dir (z_a/z_b + meta)")
    ap.add_argument("--lens", required=True, help="lens dir (feature_names.csv, feature_fidelity.csv, manifest.json)")
    ap.add_argument("--corpus", required=True, help="corpus parquet (battle_id + source)")
    ap.add_argument("--prompt-enc", default=None, dest="prompt_enc",
                    help="prompt codes dir (z_prompt.npy + meta.parquet w/ battle_id)")
    ap.add_argument("--prompt-names", default=None, dest="prompt_names")
    ap.add_argument("--dataset-id", required=True, dest="dataset_id")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out-dir", required=True, dest="out_dir")
    ap.add_argument("--min-battles", type=int, default=1, dest="min_battles",
                    help="min battles per model to keep (1 for BYO smoke; 20 for Arena)")
    args = ap.parse_args()

    enc, lens = Path(args.encoded), Path(args.lens)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    z_a, z_b = np.load(enc / "z_a.npy"), np.load(enc / "z_b.npy")
    meta = pd.read_parquet(enc / "meta.parquet").reset_index(drop=True)

    # battle_id + source from the corpus (encoded meta lacks them), by row_id
    corp = pd.read_parquet(args.corpus)
    bid = corp["battle_id"].astype(str).to_numpy()
    src = corp["source"].astype(str).to_numpy() if "source" in corp.columns else None
    ridx = meta["row_id"].to_numpy()
    battles = meta.copy()
    battles["instruction_id"] = [str(bid[i]) for i in ridx]
    if src is not None:
        battles["source"] = [str(src[i]) for i in ridx]

    # 1) bank from codes -> scratch lens dir with a bank/
    scratch = out / "_lens"
    build_bank_from_codes(z_a, z_b, battles, scratch / "bank")

    # 2) prompt-lens dir (z_prompt + battles.parquet with battle_id), if given
    prompt_lens = None
    prompt_names = None
    if args.prompt_enc:
        pe = Path(args.prompt_enc)
        pdir = out / "_prompt"; pdir.mkdir(parents=True, exist_ok=True)
        shutil.copy(pe / "z_prompt.npy", pdir / "z_prompt.npy")
        pm = pd.read_parquet(pe / "meta.parquet")
        pm.to_parquet(pdir / "battles.parquet")     # lens_battle_ids reads battle_id here
        prompt_lens = pdir
        if args.prompt_names and Path(args.prompt_names).exists():
            pn = pd.read_csv(args.prompt_names)
            col = "concept" if "concept" in pn.columns else pn.columns[1]
            prompt_names = {int(f): str(c) for f, c in zip(pn["feature_id"], pn[col])}

    # 3) shared features (concepts + fidelity) for the diagnosis filter
    fn = pd.read_csv(lens / "feature_names.csv")
    feats = fn[["feature_id", "concept"]].copy()
    fid_path = lens / "feature_fidelity.csv"
    if fid_path.exists():
        fd = pd.read_csv(fid_path)[["feature_id", "fidelity_pass"]]
        feats = feats.merge(fd, on="feature_id", how="left")
        feats["fidelity_pass"] = feats["fidelity_pass"].fillna(False)

    diag = export_diagnosis(scratch, feats, min_battles=args.min_battles,
                            prompt_lens=prompt_lens, prompt_names=prompt_names)
    if diag is None:
        sys.exit("export_diagnosis returned None (no bank?) — aborting")
    (out / "diagnosis.json").write_text(json.dumps(_clean(diag)))

    lm = json.loads((enc / "manifest.json").read_text())
    n_models = len(diag.get("models", []))
    meta_json = {
        "dataset_id": args.dataset_id, "dataset_label": args.label,
        "lens": lens.name, "input_rep": lm.get("lens_input_rep"),
        "embed_model_id": lm.get("embed_model_id"), "m_total": lm.get("m_total"),
        "k": lm.get("k"), "input_dim": lm.get("input_dim"),
        "n_battles": int(len(meta)), "n_models": n_models, "has_preference": True,
    }
    (out / "meta.json").write_text(json.dumps(meta_json))

    # cleanup scratch (bank + prompt dir kept small; remove to keep the bundle clean)
    shutil.rmtree(scratch, ignore_errors=True)
    if prompt_lens:
        shutil.rmtree(prompt_lens, ignore_errors=True)
    print(f"wrote {out}/diagnosis.json + meta.json — {n_models} models: {diag.get('models')}")
    for m in diag.get("models", []):
        r = diag["rows"][m]
        print(f"  {m}: win {r['win_rate']:.2f}, {r['n_battles']} battles, "
              f"{len(r['prompt_types'])} prompt types, {len(r['relations'])} relations")


if __name__ == "__main__":
    main()
