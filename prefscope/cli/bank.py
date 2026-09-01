from __future__ import annotations

import sys
from pathlib import Path


from prefscope.cli.common import _save


def _cmd_build_bank(args) -> int:
    import json as _json

    import numpy as np
    import pandas as pd

    from prefscope.encode.sae import SAEProjector
    from prefscope.core.manifest import LensManifest
    from prefscope.pipeline.oriented_bank import build_oriented_codes, save_bank

    emb = Path(args.from_embeddings)
    e_a = np.load(emb / "e_a.npy")
    e_b = np.load(emb / "e_b.npy")
    meta = pd.read_parquet(emb / "meta.parquet").reset_index(drop=True)
    manifest = _json.loads((Path(args.lens_dir) / "manifest.json").read_text())
    input_rep = LensManifest.from_dict(manifest).input_rep

    label_col = "y_judge"
    corp = None
    if args.label == "human":
        if not args.corpus:
            print(
                "--label human needs --corpus carrying human_pref "
                "(build-corpus --keep-labels)",
                file=sys.stderr,
            )
            return 2
        from prefscope.data.corpus import load_corpus

        corp = load_corpus(args.corpus)
        if "human_pref" not in corp.columns:
            print(
                "corpus has no human_pref; rebuild with build-corpus --keep-labels",
                file=sys.stderr,
            )
            return 2
        corp["instruction_id"] = corp["instruction_id"].astype(str)
        lut = corp.set_index("instruction_id")["human_pref"]
        meta["human_pref"] = meta["instruction_id"].astype(str).map(lut)
        label_col = "human_pref"

    # attach completion text so build_oriented_codes can persist a per-battle
    # `length` (word-count gap) for length-controlled validation. The dumped
    # meta.parquet doesn't carry it; the corpus does. If no corpus is supplied,
    # length falls back to 0.0 (build_oriented_codes notes this).
    if "completion_a" not in meta.columns:
        if corp is None and args.corpus:
            from prefscope.data.corpus import load_corpus

            corp = load_corpus(args.corpus)
            corp["instruction_id"] = corp["instruction_id"].astype(str)
        if corp is not None and {"completion_a", "completion_b"} <= set(corp.columns):
            ca = corp.set_index("instruction_id")["completion_a"]
            cb = corp.set_index("instruction_id")["completion_b"]
            iid = meta["instruction_id"].astype(str)
            meta["completion_a"] = iid.map(ca)
            meta["completion_b"] = iid.map(cb)
        else:
            print(
                "note: no corpus completion text available; bank `length` = 0.0 "
                "(validation LOO will not be length-controlled)"
            )

    if label_col not in meta.columns:
        print(
            f"embedding meta has no {label_col!r} column "
            f"(dump came from a label-free corpus?)",
            file=sys.stderr,
        )
        return 2

    keep = meta[label_col].isin([0.0, 0.5, 1.0]).to_numpy()
    if not keep.all():
        print(f"dropping {int((~keep).sum())} rows with missing/invalid {label_col}")
        e_a, e_b = e_a[keep], e_b[keep]
        meta = meta[keep].reset_index(drop=True)

    projector = SAEProjector(args.lens_dir, device=args.device)
    Z, bank_meta = build_oriented_codes(
        e_a, e_b, meta, projector, input_rep=input_rep, label_col=label_col
    )
    out_manifest = save_bank(
        args.out,
        Z,
        bank_meta,
        lens_dir=args.lens_dir,
        label_col=label_col,
        input_rep=input_rep,
    )
    print(_json.dumps(out_manifest, indent=2, default=str))
    print(
        f"wrote oriented-code bank ({Z.shape[0]} rows, "
        f"{out_manifest['n_models']} models) to {args.out}"
    )
    return 0


def _cmd_validate_diagnosis(args) -> int:
    import json as _json

    import pandas as pd

    from prefscope.pipeline.oriented_bank import load_bank
    from prefscope.pipeline.validate import validate_diagnosis

    bank_Z, bank_meta, _ = load_bank(args.bank)
    wr = pd.read_csv(args.win_relevance)
    df, summary = validate_diagnosis(
        bank_Z,
        bank_meta,
        wr,
        weight_col=args.weight_col,
        significant_only=not args.all_features,
        min_battles=args.min_battles,
        loo=args.loo,
        seed=args.seed,
    )
    _save(df, args.out)
    print(_json.dumps(summary, indent=2, default=str))
    r2 = summary.get("loo_r2") if args.loo else summary.get("insample_r2")
    tag = "LOO" if args.loo else "in-sample"
    print(
        f"\n{tag} R^2 = {r2:.3f} over {summary['n_models']} models "
        f"(predicted deficit vs actual win rate)"
    )
    print(f"wrote {len(df)} per-model rows to {args.out}")
    return 0
