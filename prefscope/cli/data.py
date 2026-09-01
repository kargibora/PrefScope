from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from prefscope.config import CONFIG
from prefscope.data.ingest import load_battles
from prefscope.encode.cache import NpyCache
from prefscope.encode.embed import Embedder
from prefscope.pipeline.inspect import summarize


def _cmd_inspect(args) -> int:
    if bool(args.corpus) == bool(args.annotations):
        print("provide exactly one of --corpus or --annotations", file=sys.stderr)
        return 2
    if args.corpus:
        from prefscope.data.corpus import load_corpus

        battles = load_corpus(args.corpus)
    else:
        battles = load_battles(args.annotations)
    print(json.dumps(summarize(battles), indent=2, default=str))
    return 0


def _cmd_init_demo(args) -> int:
    from prefscope.data.demo import create_demo

    paths = create_demo(args.out, force=args.force)
    print(f"wrote synthetic corpus: {paths['corpus']}")
    print(f"wrote pipeline config: {paths['config']}")
    print(
        "next: prefscope build-lens --corpus "
        f"{paths['corpus']} --out {paths['lens']} --m-total 16 --k 4 "
        "--input-rep individual --embed-model-id Qwen/Qwen3-Embedding-0.6B "
        "--device cpu"
    )
    return 0


def _cmd_build_corpus(args) -> int:
    from prefscope.data.arenas import SOURCES, load_arena
    from prefscope.data.corpus import merge_corpora, write_corpus

    token = os.environ.get(args.hf_token_env) if args.hf_token_env else None
    unknown = [s for s in args.source if s not in SOURCES]
    if unknown:
        print(f"unknown source(s) {unknown}; known: {sorted(SOURCES)}", file=sys.stderr)
        return 2
    if "comparia" in args.source and not token:
        print(
            f"warning: comparia is gated; set ${args.hf_token_env} or it will fail",
            file=sys.stderr,
        )
    cap = (
        f" (limit {args.limit})"
        if args.limit
        else " (full split — large first-time download)"
    )
    frames = []
    for src in args.source:
        print(f"  loading {src} from {SOURCES[src]['hf_id']}{cap}…", flush=True)
        df = load_arena(
            src,
            split=args.split,
            limit=args.limit,
            token=token,
            keep_labels=args.keep_labels,
        )
        print(f"    -> {len(df)} battles", flush=True)
        frames.append(df)
    merged = merge_corpora(frames)
    write_corpus(merged, args.out)
    by_src = merged["source"].value_counts().to_dict()
    extra = " (+human_pref)" if "human_pref" in merged.columns else ""
    print(f"wrote {len(merged)} battles to {args.out}  (by source: {by_src}){extra}")
    return 0


def _cmd_prepare_dataset(args) -> int:
    """Materialize a local/Hugging Face dataset in the canonical pair schema."""
    from prefscope.pipeline.prepare_dataset import (
        load_dataset_spec,
        mapping_from_spec,
        override_mapping,
        prepare_dataset,
    )

    spec = load_dataset_spec(args.spec) if args.spec else {}
    source_spec = dict(spec.get("source") or {})
    source_type = str(source_spec.get("type", "")).casefold()
    data = args.data
    hf_dataset = args.hf_dataset
    if data is None and hf_dataset is None:
        if source_type in {"huggingface", "hf", "hub"}:
            hf_dataset = source_spec.get("path") or source_spec.get("dataset_id")
        elif source_type in {"local", "file"}:
            data = source_spec.get("path")

    def _tokens(values):
        if values is None:
            return None
        return tuple(
            token.strip()
            for value in values
            for token in str(value).split(",")
            if token.strip()
        )

    mapping = mapping_from_spec(spec)
    mapping = override_mapping(
        mapping,
        prompt=args.prompt_col,
        response_a=args.response_col,
        response_b=args.response2_col,
        label=args.label_col,
        model_a=args.model_col,
        model_b=args.model2_col,
        item_id=args.id_col,
        language=args.language_col,
        metadata=args.keep_columns,
        prompt_role=args.prompt_role,
        response_a_role=args.response_role,
        response_b_role=args.response2_role,
        label_mode=args.label_mode,
        a_values=_tokens(args.a_wins_values),
        b_values=_tokens(args.b_wins_values),
        tie_values=_tokens(args.tie_values),
        auto_pair=False if args.single else None,
    )
    split = args.split or source_spec.get("split") or "train"
    limit = args.limit if args.limit is not None else source_spec.get("limit")
    streaming = bool(args.streaming or source_spec.get("streaming", False))
    summary = prepare_dataset(
        args.out,
        data=data,
        hf_dataset=hf_dataset,
        hf_name=args.hf_name or source_spec.get("name"),
        split=split,
        revision=args.hf_revision or source_spec.get("revision"),
        token_env=args.hf_token_env or source_spec.get("token_env"),
        streaming=streaming,
        limit=limit,
        mapping=mapping,
        drop_empty=not args.fail_on_empty,
    )
    print(json.dumps(summary, indent=2))
    return 0


def _cmd_build_lens(args) -> int:
    from prefscope.pipeline.build_lens import build_lens  # lazy (torch)

    common = dict(
        m_total=args.m_total,
        k=args.k,
        matryoshka_prefix=tuple(args.matryoshka_prefix),
        input_rep=args.input_rep,
        val_frac=args.val_frac,
        device=args.device,
        embed_model_id=args.embed_model_id,
        batch=args.batch,
        n_epochs=args.n_epochs,
        seed=args.seed,
        whiten=args.whiten,
        whiten_eps=args.whiten_eps,
        sae_type=args.sae_type,
        sparsity_coef=args.sparsity_coef,
        bandwidth=args.bandwidth,
        sparsity_warmup_steps=args.sparsity_warmup_steps,
        max_train_rows=args.max_train_rows,
    )

    if args.from_embeddings:
        # retrain from a previously dumped embedding set — no corpus, no cache scan
        from prefscope.pipeline.build_lens import build_lens_from_embeddings

        print(f"training from dumped embeddings {args.from_embeddings}")
        manifest = build_lens_from_embeddings(args.from_embeddings, args.out, **common)
        print(json.dumps(manifest, indent=2, default=str))
        return 0

    if bool(args.corpus) == bool(args.annotations):
        print("provide exactly one of --annotations or --corpus", file=sys.stderr)
        return 2
    if args.corpus:
        from prefscope.data.corpus import load_corpus

        battles = load_corpus(args.corpus)
        print(f"loaded {len(battles)} battles from corpus {args.corpus}")
    else:
        battles = load_battles(args.annotations)
        print(f"loaded {len(battles)} battles from {len(args.annotations)} file(s)")
    cache = NpyCache(args.cache_dir or CONFIG.cache_dir)
    embedder = Embedder(
        cache,
        model_id=args.embed_model_id,
        model_revision=args.embed_model_revision,
        device=args.device,
        max_tokens=args.max_tokens,
        batch_size=args.embed_batch_size,
        cache_workers=args.cache_workers,
        backend=args.embed_backend,
        tensor_parallel_size=args.tensor_parallel_size,
        api_base=args.embed_api_base,
        api_key_env=args.embed_api_key_env,
    )
    manifest = build_lens(
        battles, embedder, args.out, dump_embeddings=args.dump_embeddings, **common
    )
    print(json.dumps(manifest, indent=2, default=str))
    return 0


def _cmd_encode_dataset(args) -> int:
    """Encode an arbitrary (prompt, response[, response_2]) dataset with a trained lens.

    The embedder is chosen from the LENS manifest's embed_model_id (never a flag), so the
    dataset is embedded exactly as the lens was — the codes stay consistent."""
    from prefscope.pipeline.encode_dataset import run_encode_dataset

    from prefscope import load_lens

    if args.lens is not None:
        token = os.environ.get(args.hf_token_env) if args.hf_token_env else None
        loaded_lens = load_lens(
            args.lens,
            device=args.device,
            revision=args.revision,
            cache_dir=args.hub_cache_dir,
            token=token,
            local_files_only=args.local_files_only,
            subfolder=args.subfolder,
            embedding_cache=args.cache_dir or CONFIG.cache_dir,
            embed_backend=args.embed_backend,
            embed_batch_size=args.embed_batch_size,
        )
        lens_dir = Path(loaded_lens.lens_dir)
    else:
        lens_dir = Path(args.lens_dir)
        loaded_lens = load_lens(
            lens_dir,
            device=args.device,
            embedding_cache=args.cache_dir or CONFIG.cache_dir,
            embed_backend=args.embed_backend,
            embed_batch_size=args.embed_batch_size,
        )
    mf = lens_dir / "manifest.json"
    if not mf.exists():
        print(f"no manifest.json in lens dir {lens_dir}", file=sys.stderr)
        return 2
    embed_model_id = json.loads(mf.read_text()).get("embed_model_id")
    if not embed_model_id:
        print(
            f"lens manifest {mf} has no embed_model_id — cannot pick the embedder",
            file=sys.stderr,
        )
        return 2

    embedder = loaded_lens.embedder
    embedder.cache_workers = args.cache_workers
    embedder.tensor_parallel_size = args.tensor_parallel_size
    embedder.api_base = args.embed_api_base
    embedder.api_key_env = args.embed_api_key_env
    manifest = run_encode_dataset(
        lens_dir,
        args.data,
        args.out,
        embedder=embedder,
        prompt_col=args.prompt_col,
        response_col=args.response_col,
        response2_col=args.response2_col,
        model_col=args.model_col,
        model2_col=args.model2_col,
        label_col=args.label_col,
        metadata_cols=args.metadata_cols,
        device=args.device,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, indent=2, default=str))
    return 0


def _cmd_compare_responses(args) -> int:
    """Compare two prompt-aligned response sets without using preference labels."""
    from prefscope.pipeline.compare import compare_encoded_responses

    comparison = compare_encoded_responses(
        args.encoded_dir,
        features=args.features,
        prompt_dir=args.prompt_encoded_dir,
        prompt_features=args.prompt_features,
        prompt_clusters=args.prompt_clusters,
        side_a_name=args.side_a_name,
        side_b_name=args.side_b_name,
        presence_policy=args.presence_policy,
        prompt_presence_policy=args.prompt_presence_policy,
        fidelity_only=not args.include_unverified,
        named_only=not args.include_unnamed,
        min_context_pairs=args.min_context_pairs,
        group_col=args.group_col,
        examples_per_direction=args.examples_per_direction,
        confidence=args.confidence,
    )
    out = comparison.save(args.out)
    counts = comparison.scope["response_scope"].value_counts().to_dict()
    print(f"wrote paired response comparison to {out}")
    print(f"  {len(comparison.overall)} concepts: {counts}")
    print("  preference labels were not used")
    return 0


def _cmd_concepts(args) -> int:
    """Export all active concepts for a BYO prompt/response table."""
    from prefscope import load_lens
    from prefscope.pipeline.concepts import export_concepts

    token = os.environ.get(args.hf_token_env) if args.hf_token_env else None
    lens = load_lens(
        args.lens,
        device=args.device,
        revision=args.revision,
        cache_dir=args.hub_cache_dir,
        token=token,
        local_files_only=args.local_files_only,
        subfolder=args.subfolder,
        annotations=args.annotations,
    )
    result = export_concepts(
        lens,
        args.data,
        args.out,
        prompt_col=args.prompt_col,
        response_col=args.response_col,
        response2_col=args.response2_col,
        batch_size=args.batch_size,
        active_only=not args.include_zero,
        pole=args.pole,
        min_abs_activation=args.min_abs_activation,
        top_k=args.top_k,
        fidelity_only=args.fidelity_only,
        semantic_presence_only=args.semantic_presence_only,
        include_text=args.include_text,
    )
    print(json.dumps(result, indent=2))
    return 0


def _cmd_embed_corpus(args) -> int:
    """Embed one shard of a corpus into the shared cache (no training).

    Run N of these in parallel (one GPU each, CUDA_VISIBLE_DEVICES) to fill the
    per-completion cache across all GPUs, then a single `build-lens` reads the
    cache and trains. The cache is keyed per (model, text), so concurrent shards
    write disjoint keys safely and the job is resumable.
    """
    from prefscope.data.corpus import load_corpus

    battles = load_corpus(args.corpus)
    if args.num_shards > 1:
        sel = (battles.index.to_numpy() % args.num_shards) == args.shard
        battles = battles[sel].reset_index(drop=True)
    print(f"shard {args.shard}/{args.num_shards}: {len(battles)} battles", flush=True)

    cache = NpyCache(args.cache_dir or CONFIG.cache_dir)
    embedder = Embedder(
        cache,
        model_id=args.embed_model_id,
        model_revision=args.embed_model_revision,
        device=args.device,
        max_tokens=args.max_tokens,
        batch_size=args.embed_batch_size,
        cache_workers=args.cache_workers,
        backend=args.embed_backend,
        tensor_parallel_size=args.tensor_parallel_size,
        api_base=args.embed_api_base,
        api_key_env=args.embed_api_key_env,
    )
    prompts = battles["prompt"].tolist()
    print("embedding completion A…", flush=True)
    embedder.encode(prompts, battles["completion_a"].tolist())
    if "completion_b" in battles.columns:
        print("embedding completion B…", flush=True)
        embedder.encode(prompts, battles["completion_b"].tolist())
    print(f"shard {args.shard}/{args.num_shards} done: cached {len(battles)} rows")
    return 0


def _cmd_embed_prompts(args) -> int:
    """Embed prompts ALONE → a battle_id-aligned e_prompt.npy for the prompt lens.

    The dump is row-aligned to the corpus and carries ``battle_id``, so it joins
    back to z_diff / the responses on ``battle_id`` — every prompt vector matches
    exactly the query whose responses the difference-lens saw. With --num-shards
    it only warms the cache for its shard (parallel multi-GPU pre-pass), like
    embed-corpus; a final unsharded run reads the warm cache and writes the dump.
    """
    from pathlib import Path

    import numpy as np

    from prefscope.data.corpus import load_corpus

    out = Path(args.out)
    battles = load_corpus(args.corpus)
    if args.num_shards > 1:
        sel = (battles.index.to_numpy() % args.num_shards) == args.shard
        battles = battles[sel].reset_index(drop=True)
        print(
            f"shard {args.shard}/{args.num_shards}: {len(battles)} prompts", flush=True
        )

    cache = NpyCache(args.cache_dir or CONFIG.cache_dir)
    embedder = Embedder(
        cache,
        model_id=args.embed_model_id,
        model_revision=args.embed_model_revision,
        device=args.device,
        max_tokens=args.max_tokens,
        batch_size=args.embed_batch_size,
        cache_workers=args.cache_workers,
        backend=args.embed_backend,
        tensor_parallel_size=args.tensor_parallel_size,
        api_base=args.embed_api_base,
        api_key_env=args.embed_api_key_env,
    )
    print("embedding prompts…", flush=True)
    e = embedder.encode_prompts(battles["prompt"].tolist())

    if args.num_shards > 1:
        print(
            f"shard {args.shard}/{args.num_shards}: cache warmed ({len(battles)} prompts)"
        )
        return 0

    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "e_prompt.npy", np.asarray(e, dtype=np.float32))
    (out / "embedding_manifest.json").write_text(
        json.dumps(embedder.provenance(prompt=True), indent=2))
    cols = [
        c
        for c in (
            "battle_id",
            "instruction_id",
            "group_id",
            "model_a",
            "model_b",
            "source",
            "language",
            "human_pref",
        )
        if c in battles.columns
    ]
    battles[cols].reset_index(drop=True).to_parquet(out / "meta.parquet")
    print(f"wrote {len(e)} prompt embeddings (dim {e.shape[1]}) to {out}")
    print("  meta.parquet carries battle_id — join to z_diff / responses on battle_id")
    return 0


def _cmd_build_prompt_lens(args) -> int:
    from prefscope.pipeline.build_lens import build_prompt_lens

    manifest = build_prompt_lens(
        args.from_embeddings,
        args.out,
        m_total=args.m_total,
        k=args.k,
        matryoshka_prefix=tuple(args.matryoshka_prefix),
        val_frac=args.val_frac,
        device=args.device,
        embed_model_id=args.embed_model_id,
        max_train_rows=args.max_train_rows,
        batch=args.batch,
        n_epochs=args.n_epochs,
        seed=args.seed,
        sae_type=args.sae_type,
        sparsity_coef=args.sparsity_coef,
        bandwidth=args.bandwidth,
        sparsity_warmup_steps=args.sparsity_warmup_steps,
    )
    print(json.dumps(manifest, indent=2, default=str))
    return 0
