from __future__ import annotations

import json
from pathlib import Path



def _cmd_extract_activations(args) -> int:
    from prefscope.activations.cache import ActivationCache
    from prefscope.activations.extract import ActivationExtractor
    from prefscope.data.corpus import load_corpus

    # Refuse an existing output before loading a multi-billion-parameter model.
    ActivationCache.ensure_empty_output(args.out)
    battles = load_corpus(args.corpus)
    if args.n_battles and args.n_battles < len(battles):
        battles = battles.sample(n=args.n_battles, random_state=args.seed).reset_index(
            drop=True
        )
    print(
        f"extracting activations for {len(battles)} battles "
        f"({args.model_id} layer {args.layer})",
        flush=True,
    )
    ext = ActivationExtractor(
        args.model_id,
        args.layer,
        max_tokens=args.max_tokens,
        outlier_norm_mult=args.outlier_norm_mult,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    cache = ActivationCache(args.out, hidden_dim=ext.hidden_dim)
    n_done = 0
    for vectors, rows in ext.iter_battle_activations(battles):
        cache.append(vectors, rows)
        n_done += 1
        if n_done % 500 == 0:
            print(f"  {n_done} spans appended ({cache._n} tokens)", flush=True)
    meta_cols = [
        c
        for c in ("battle_id", "model_a", "model_b", "source", "language", "human_pref")
        if c in battles.columns
    ]
    import pandas as pd

    cache.finalize(
        extra_manifest={
            "model_id": args.model_id,
            "layer": args.layer,
            "max_tokens": args.max_tokens,
            "outlier_norm_mult": args.outlier_norm_mult,
            "n_battles": int(len(battles)),
        }
    )
    pd.DataFrame(battles[meta_cols]).to_parquet(Path(args.out) / "battle_meta.parquet")
    print(f"done: {cache._n} tokens cached to {args.out}")
    return 0


def _cmd_train_token_sae(args) -> int:
    import torch
    from prefscope.activations.cache import ActivationCache
    from prefscope.activations.train import train_token_sae

    cache = ActivationCache.open(args.cache)
    m_total = args.m_total if args.m_total else args.expansion * cache.hidden_dim
    model, config, log = train_token_sae(
        cache,
        m_total=m_total,
        k=args.k,
        matryoshka_prefix=tuple(args.matryoshka_prefix),
        val_frac=args.val_frac,
        max_train_tokens=args.max_train_tokens,
        n_epochs=args.epochs,
        batch=args.batch,
        seed=args.seed,
        device=args.device,
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "config": config}, out / "sae_model.pt"
    )
    import pandas as pd

    pd.DataFrame(log).to_csv(out / "sae_training_log.csv", index=False)
    manifest = {
        "source_cache": str(args.cache),
        "m_total": int(m_total),
        "k": int(args.k),
        "input_dim": int(cache.hidden_dim),
        "sae_type": config["sae_type"],
        "activation_polarity": config["activation_polarity"],
        "code_semantics": config["code_semantics"],
        "selection_rule": config["selection_rule"],
        "matryoshka_prefix_lengths": config["matryoshka_prefix_lengths"],
        "optimizer": config["optimizer"],
        "weight_decay": config["weight_decay"],
        "seed": config["seed"],
        "best_val_norm_mse": config["best_val_norm_mse"],
        "best_val_ev": config["best_val_ev"],
        "dead_neurons": config["dead_neurons"],
        "n_train_tokens": config["n_train_tokens"],
        "n_val_tokens": config["n_val_tokens"],
        "model_id": cache.manifest.get("model_id"),
        "layer": cache.manifest.get("layer"),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(json.dumps(manifest, indent=2, default=str))
    return 0


def _cmd_summarize_activations(args) -> int:
    from prefscope.activations.cache import ActivationCache
    from prefscope.activations.summarize import summarize_spans
    from prefscope.encode.sae import SAEProjector

    cache = ActivationCache.open(args.cache)
    projector = SAEProjector(args.sae, device=args.device)
    summaries, span_meta = summarize_spans(cache, projector, batch=args.batch)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summaries.to_parquet(out / "span_summaries.parquet")
    bm_path = Path(args.cache) / "battle_meta.parquet"
    if bm_path.exists():
        import pandas as pd

        bm = pd.read_parquet(bm_path)
        bm["battle_id"] = bm["battle_id"].astype(str)
        span_meta["battle_id"] = span_meta["battle_id"].astype(str)
        span_meta = span_meta.merge(bm, on="battle_id", how="left")
    span_meta.to_parquet(out / "span_meta.parquet")
    print(
        f"wrote {len(summaries)} (battle,span,feature) rows + "
        f"{len(span_meta)} span-meta rows to {out}"
    )
    return 0
