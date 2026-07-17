"""PrefScope command-line entry point.

    prefscope inspect    <annotations.json> [more.json ...]
    prefscope build-lens --annotations <json> [...] --out <dir>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from prefscope.config import CONFIG
from prefscope.data.ingest import load_battles
from prefscope.encode.cache import NpyCache
from prefscope.encode.embed import Embedder
from prefscope.interpret.io import load_lens_battles
from prefscope.interpret.llm import LLMClient, DEFAULT_API_BASE, DEFAULT_MODEL
from prefscope.pipeline.inspect import summarize

# NOTE: build_lens / run_diagnose are imported lazily inside their command handlers
# (below), not here — importing them pulls in torch, and `prefscope --help` (and every
# non-training command) must work without a torch install present.


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


def _cmd_build_corpus(args) -> int:
    from prefscope.data.arenas import SOURCES, load_arena
    from prefscope.data.corpus import merge_corpora, write_corpus

    token = os.environ.get(args.hf_token_env) if args.hf_token_env else None
    unknown = [s for s in args.source if s not in SOURCES]
    if unknown:
        print(f"unknown source(s) {unknown}; known: {sorted(SOURCES)}", file=sys.stderr)
        return 2
    if "comparia" in args.source and not token:
        print(f"warning: comparia is gated; set ${args.hf_token_env} or it will fail",
              file=sys.stderr)
    cap = f" (limit {args.limit})" if args.limit else " (full split — large first-time download)"
    frames = []
    for src in args.source:
        print(f"  loading {src} from {SOURCES[src]['hf_id']}{cap}…", flush=True)
        df = load_arena(src, split=args.split, limit=args.limit, token=token,
                        keep_labels=args.keep_labels)
        print(f"    -> {len(df)} battles", flush=True)
        frames.append(df)
    merged = merge_corpora(frames)
    write_corpus(merged, args.out)
    by_src = merged["source"].value_counts().to_dict()
    extra = " (+human_pref)" if "human_pref" in merged.columns else ""
    print(f"wrote {len(merged)} battles to {args.out}  (by source: {by_src}){extra}")
    return 0


def _cmd_build_lens(args) -> int:
    from prefscope.pipeline.build_lens import build_lens   # lazy (torch)
    common = dict(m_total=args.m_total, k=args.k,
                  matryoshka_prefix=tuple(args.matryoshka_prefix),
                  input_rep=args.input_rep, val_frac=args.val_frac,
                  device=args.device, embed_model_id=args.embed_model_id,
                  batch=args.batch, n_epochs=args.n_epochs, seed=args.seed,
                  whiten=args.whiten, whiten_eps=args.whiten_eps,
                  sae_type=args.sae_type, sparsity_coef=args.sparsity_coef,
                  bandwidth=args.bandwidth, max_train_rows=args.max_train_rows)

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
    embedder = Embedder(cache, model_id=args.embed_model_id,
                        device=args.device, max_tokens=args.max_tokens,
                        batch_size=args.embed_batch_size,
                        cache_workers=args.cache_workers,
                        backend=args.embed_backend,
                        tensor_parallel_size=args.tensor_parallel_size,
                        api_base=args.embed_api_base,
                        api_key_env=args.embed_api_key_env)
    manifest = build_lens(battles, embedder, args.out,
                          dump_embeddings=args.dump_embeddings, **common)
    print(json.dumps(manifest, indent=2, default=str))
    return 0


def _cmd_encode_dataset(args) -> int:
    """Encode an arbitrary (prompt, response[, response_2]) dataset with a trained lens.

    The embedder is chosen from the LENS manifest's embed_model_id (never a flag), so the
    dataset is embedded exactly as the lens was — the codes stay consistent."""
    from prefscope.pipeline.encode_dataset import run_encode_dataset

    lens_dir = Path(args.lens_dir)
    mf = lens_dir / "manifest.json"
    if not mf.exists():
        print(f"no manifest.json in lens dir {lens_dir}", file=sys.stderr)
        return 2
    embed_model_id = json.loads(mf.read_text()).get("embed_model_id")
    if not embed_model_id:
        print(f"lens manifest {mf} has no embed_model_id — cannot pick the embedder",
              file=sys.stderr)
        return 2

    cache = NpyCache(args.cache_dir or CONFIG.cache_dir)
    embedder = Embedder(cache, model_id=embed_model_id, device=args.device,
                        max_tokens=args.max_tokens, batch_size=args.embed_batch_size,
                        cache_workers=args.cache_workers, backend=args.embed_backend,
                        tensor_parallel_size=args.tensor_parallel_size,
                        api_base=args.embed_api_base, api_key_env=args.embed_api_key_env)
    manifest = run_encode_dataset(
        lens_dir, args.data, args.out, embedder=embedder,
        prompt_col=args.prompt_col, response_col=args.response_col,
        response2_col=args.response2_col, model_col=args.model_col,
        model2_col=args.model2_col, label_col=args.label_col, device=args.device)
    print(json.dumps(manifest, indent=2, default=str))
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
    embedder = Embedder(cache, model_id=args.embed_model_id,
                        device=args.device, max_tokens=args.max_tokens,
                        batch_size=args.embed_batch_size,
                        cache_workers=args.cache_workers,
                        backend=args.embed_backend,
                        tensor_parallel_size=args.tensor_parallel_size,
                        api_base=args.embed_api_base,
                        api_key_env=args.embed_api_key_env)
    prompts = battles["prompt"].tolist()
    print("embedding completion A…", flush=True)
    embedder.encode(prompts, battles["completion_a"].tolist())
    print("embedding completion B…", flush=True)
    embedder.encode(prompts, battles["completion_b"].tolist())
    print(f"shard {args.shard}/{args.num_shards} done: cached {len(battles)} battles")
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
        print(f"shard {args.shard}/{args.num_shards}: {len(battles)} prompts", flush=True)

    cache = NpyCache(args.cache_dir or CONFIG.cache_dir)
    embedder = Embedder(cache, model_id=args.embed_model_id,
                        device=args.device, max_tokens=args.max_tokens,
                        batch_size=args.embed_batch_size,
                        cache_workers=args.cache_workers,
                        backend=args.embed_backend,
                        tensor_parallel_size=args.tensor_parallel_size,
                        api_base=args.embed_api_base,
                        api_key_env=args.embed_api_key_env)
    print("embedding prompts…", flush=True)
    e = embedder.encode_prompts(battles["prompt"].tolist())

    if args.num_shards > 1:
        print(f"shard {args.shard}/{args.num_shards}: cache warmed ({len(battles)} prompts)")
        return 0

    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "e_prompt.npy", np.asarray(e, dtype=np.float32))
    cols = [c for c in ("battle_id", "instruction_id", "model_a", "model_b",
                        "source", "language", "human_pref") if c in battles.columns]
    battles[cols].reset_index(drop=True).to_parquet(out / "meta.parquet")
    print(f"wrote {len(e)} prompt embeddings (dim {e.shape[1]}) to {out}")
    print("  meta.parquet carries battle_id — join to z_diff / responses on battle_id")
    return 0


def _cmd_build_prompt_lens(args) -> int:
    from prefscope.pipeline.build_lens import build_prompt_lens

    manifest = build_prompt_lens(
        args.from_embeddings, args.out, m_total=args.m_total, k=args.k,
        matryoshka_prefix=tuple(args.matryoshka_prefix), val_frac=args.val_frac,
        device=args.device, embed_model_id=args.embed_model_id,
        max_train_rows=args.max_train_rows,
        batch=args.batch, n_epochs=args.n_epochs, seed=args.seed)
    print(json.dumps(manifest, indent=2, default=str))
    return 0


def _cmd_name_prompts(args) -> int:
    from pathlib import Path

    import numpy as np
    import pandas as pd

    from prefscope.data.corpus import load_corpus
    from prefscope.interpret.prompt_name import name_prompt_features

    lens = Path(args.lens_dir)
    z = np.load(lens / "z_prompt.npy")
    meta = pd.read_parquet(lens / "battles.parquet")
    bid = (meta["battle_id"] if "battle_id" in meta.columns
           else meta["instruction_id"]).astype(str)
    corp = load_corpus(args.corpus)
    corp["battle_id"] = corp["battle_id"].astype(str)
    prompts = bid.map(corp.set_index("battle_id")["prompt"]).tolist()
    if any(p is None or (isinstance(p, float) and np.isnan(p)) for p in prompts):
        print("warning: some prompts missing from corpus (battle_id mismatch?)",
              file=sys.stderr)
        prompts = ["" if (p is None or (isinstance(p, float) and np.isnan(p))) else p
                   for p in prompts]

    df = name_prompt_features(prompts, z, _make_client(args), features=args.features,
                              n_active=args.n_active, n_zero=args.n_zero,
                              concurrency=args.concurrency, instruction_ids=bid.tolist())
    _save(df, args.out)
    print(f"wrote {len(df)} prompt-feature names to {args.out}")
    return 0


def _save(df, out, *, index: bool = False) -> None:
    """Write a DataFrame to a CSV/parquet path, creating parent dirs if needed.

    Output paths are user-supplied files; their parent directory may not exist
    yet (e.g. a fresh `.../interpret/<lens>/feature_names.csv`). Create it so the
    pipeline never fails just because the enclosing folder is missing."""
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix == ".parquet":
        df.to_parquet(p, index=index)
    else:
        df.to_csv(p, index=index)


def _make_client(args) -> "LLMClient":
    return LLMClient(backend=args.backend, model=args.model,
                     api_base=args.api_base, api_key_env=args.api_key_env,
                     reasoning_effort=getattr(args, "reasoning_effort", None))


def _cmd_run(args) -> int:
    from prefscope.pipeline.run import PipelineConfig, preflight, run_pipeline

    try:
        cfg = PipelineConfig.load(args.config)
        preflight(cfg)
        print(f"running stages {cfg.stages} on lens {cfg.lens_dir} -> {cfg.out_dir}")
        outputs = run_pipeline(cfg)
    except (ValueError, FileNotFoundError) as e:
        # config typos surface here too: an unknown component name (registry.make),
        # a missing names CSV for verify, or a lens that can't supply z_a/z_b.
        print(f"config error: {e}", file=sys.stderr)
        return 2
    print(f"\npipeline complete: {len(outputs)} stage(s) written to {cfg.out_dir}")
    return 0


def _cmd_interpret_name(args) -> int:
    from prefscope.core import registry
    from prefscope.interpret.strategy import LensCodes, resolve_name_mode

    codes = LensCodes.load(args.lens_dir, args.annotations, corpus=args.corpus)
    mode = resolve_name_mode(args.name_mode, codes.input_rep)
    src = f"auto, from lens input_rep={codes.input_rep!r}" if mode != args.name_mode else "explicit"
    print(f"naming mode: {mode} ({src})")
    strategy = registry.make(
        "interpreter", mode, features=args.features, n_active=args.n_active,
        n_zero=args.n_zero, verify_frac=args.verify_frac, seed=args.seed,
        abbreviate=args.abbreviate, concurrency=args.concurrency,
        debug_dir=args.debug_responses, negatives=args.negatives)
    df = strategy.name(codes, _make_client(args))
    _save(df, args.out)
    print(f"wrote {len(df)} feature names to {args.out}")
    return 0


def _cmd_interpret_verify(args) -> int:
    import pandas as pd

    from prefscope.core import registry
    from prefscope.interpret.strategy import VerifyCodes, resolve_verify_mode

    lens_kind = getattr(args, "lens_kind", "completion")
    if lens_kind == "prompt" and not args.corpus:
        print("prompt-lens verify needs --corpus (to fetch prompt text)", file=sys.stderr)
        return 2
    codes = VerifyCodes.load(args.lens_dir, args.annotations, corpus=args.corpus,
                             lens_kind=lens_kind)
    mode = resolve_verify_mode(args.verify_mode, codes.input_rep, lens_kind)
    src = (f"auto, from lens input_rep={codes.input_rep!r}"
           if mode != args.verify_mode and lens_kind != "prompt"
           else (f"--lens-kind {lens_kind}" if lens_kind == "prompt" else "explicit"))
    print(f"verify mode: {mode} ({src})")
    strategy = registry.make(
        "verifier", mode, n_per_bucket=args.n_per_bucket, verify_frac=args.verify_frac,
        seed=args.seed, fidelity_threshold=args.fidelity_threshold,
        concurrency=args.concurrency, negatives=getattr(args, "negatives", "random"),
        embeddings=getattr(args, "embeddings", None))
    df = strategy.verify(codes, pd.read_csv(args.names), _make_client(args))
    _save(df, args.out)
    print(f"wrote {len(df)} fidelity rows ({int(df['fidelity_pass'].sum())} pass) to {args.out}")
    return 0


def _print_diagnosis(summary, df, top: int) -> None:
    print(f"\n{summary['model']} vs pool — {summary['n_battles']} battles, "
          f"win rate {summary['win_rate']:.3f}, {summary['n_features']} features\n")
    has_concept = "concept" in df.columns
    label = (lambda r: str(r["concept"] or f"feature {int(r['feature_id'])}")) if has_concept \
        else (lambda r: f"feature {int(r['feature_id'])}")

    has_pool = "delta_vs_pool" in df.columns
    # headline helps-win signal is length-controlled: global delta_win_rate
    # (helps_win) if merged, else the within-model length-controlled AME
    # (outcome_assoc_lc), falling back to the raw outcome_assoc.
    assoc_col = next((c for c in ("helps_win", "outcome_assoc_lc", "outcome_assoc")
                      if c in df.columns), "outcome_assoc")
    assoc_label = "helps-win" if assoc_col != "outcome_assoc" else "win-assoc"

    def _line(r):
        assoc = r.get(assoc_col, float("nan"))
        assoc_s = "  n/a   " if assoc != assoc else f"{assoc:+.2f}"
        pool_s = ""
        if has_pool:
            d = r["delta_vs_pool"]
            star = "*" if r.get("welch_p_bonferroni", 1.0) < 0.05 else " "
            pool_s = f"Δpool {d:+.2f}{star} "
        return (f"  {r['net_direction']:+.2f}  differs {r['fire_rate']:5.0%}  "
                f"{pool_s}{assoc_label} {assoc_s}  {label(r)}")

    print(f"Most OVER-expressed vs peers (does MORE than others):")
    for _, r in df.head(top).iterrows():
        print(_line(r))
    print(f"\nMost UNDER-expressed vs peers (does LESS than others):")
    for _, r in df.tail(top).iloc[::-1].iterrows():
        print(_line(r))


def _cmd_diagnose(args) -> int:
    import json as _json
    from pathlib import Path

    import pandas as pd

    from prefscope.encode.sae import SAEProjector

    battles = load_battles(args.annotations)
    manifest = _json.loads((Path(args.lens_dir) / "manifest.json").read_text())
    input_rep = manifest.get("input_rep", "difference")
    # the manifest is the source of truth for which embedder this lens expects;
    # only fall back to the config default if the user explicitly overrode it.
    embed_model_id = args.embed_model_id or manifest.get("embed_model_id") or CONFIG.embed_model_id
    if not args.embed_model_id and manifest.get("embed_model_id"):
        print(f"embedder: {embed_model_id} (from lens manifest)")
    elif not args.embed_model_id:
        print(f"warning: lens manifest has no embed_model_id; falling back to config "
              f"default {embed_model_id} — pass --embed-model-id if this lens used another.",
              file=sys.stderr)

    cache = NpyCache(args.cache_dir or CONFIG.cache_dir)
    embedder = Embedder(cache, model_id=embed_model_id,
                        device=args.device, max_tokens=args.max_tokens,
                        batch_size=args.embed_batch_size,
                        cache_workers=args.cache_workers,
                        backend=args.embed_backend,
                        tensor_parallel_size=args.tensor_parallel_size,
                        api_base=args.embed_api_base,
                        api_key_env=args.embed_api_key_env)
    projector = SAEProjector(args.lens_dir, device=args.device)

    baseline_z = None
    if args.bank:
        from prefscope.pipeline.oriented_bank import load_bank
        bank_Z, bank_meta, _ = load_bank(args.bank)
        other = (bank_meta["self_model"] != args.model).to_numpy()
        baseline_z = bank_Z[other]
        print(f"baseline: {int(other.sum())} pool rows from bank {args.bank}")

    names = pd.read_csv(args.fidelity) if args.fidelity else None
    win_rel = pd.read_csv(args.win_relevance) if args.win_relevance else None
    from prefscope.pipeline.diagnose import run_diagnose   # lazy (torch)
    result = run_diagnose(battles, args.model, embedder, projector,
                          input_rep=input_rep, names=names,
                          fidelity_only=not args.all_features,
                          return_battles=bool(args.battles_out),
                          baseline_z=baseline_z, win_relevance=win_rel)
    if args.battles_out:
        df, summary, per_battle = result
        _save(per_battle, args.battles_out)
    else:
        df, summary = result
    _save(df, args.out)
    print(_json.dumps(summary, indent=2, default=str))
    _print_diagnosis(summary, df, args.top)
    print(f"\nwrote {len(df)} feature diagnoses to {args.out}")
    if args.battles_out:
        print(f"wrote {len(per_battle)} per-battle evidence rows to {args.battles_out}")
    return 0


def _build_diagnose_embedder(args):
    """Embedder + projector wired exactly like `diagnose` (manifest-driven embedder)."""
    import json as _json

    from prefscope.encode.sae import SAEProjector

    manifest = _json.loads((Path(args.lens_dir) / "manifest.json").read_text())
    input_rep = manifest.get("input_rep", "difference")
    embed_model_id = args.embed_model_id or manifest.get("embed_model_id") or CONFIG.embed_model_id
    if not args.embed_model_id and manifest.get("embed_model_id"):
        print(f"embedder: {embed_model_id} (from lens manifest)")
    elif not args.embed_model_id:
        print(f"warning: lens manifest has no embed_model_id; falling back to config "
              f"default {embed_model_id} — pass --embed-model-id if this lens used another.",
              file=sys.stderr)
    cache = NpyCache(args.cache_dir or CONFIG.cache_dir)
    embedder = Embedder(cache, model_id=embed_model_id,
                        device=args.device, max_tokens=args.max_tokens,
                        batch_size=args.embed_batch_size,
                        cache_workers=args.cache_workers,
                        backend=args.embed_backend,
                        tensor_parallel_size=args.tensor_parallel_size,
                        api_base=args.embed_api_base,
                        api_key_env=args.embed_api_key_env)
    projector = SAEProjector(args.lens_dir, device=args.device)
    return embedder, projector, input_rep


def _cmd_report(args) -> int:
    import pandas as pd

    from prefscope.pipeline.report import (format_report, prompt_concept_winrates,
                                           prompt_to_response_winrates)

    if bool(args.corpus) == bool(args.annotations):
        print("provide exactly one of --corpus or --annotations", file=sys.stderr)
        return 2
    if args.corpus:
        from prefscope.data.corpus import load_corpus
        battles = load_corpus(args.corpus)
    else:
        battles = load_battles(args.annotations)

    embedder, projector, input_rep = _build_diagnose_embedder(args)

    baseline_z = None
    if args.bank:
        from prefscope.pipeline.oriented_bank import load_bank
        bank_Z, bank_meta, _ = load_bank(args.bank)
        other = (bank_meta["self_model"] != args.model).to_numpy()
        baseline_z = bank_Z[other]
        print(f"baseline: {int(other.sum())} pool rows from bank {args.bank}")

    names = pd.read_csv(args.names) if args.names else None
    win_rel = pd.read_csv(args.win_relevance) if args.win_relevance else None
    want_battles = bool(args.prompt_lens)
    from prefscope.pipeline.diagnose import run_diagnose   # lazy (torch)
    result = run_diagnose(battles, args.model, embedder, projector,
                          input_rep=input_rep, names=names,
                          fidelity_only=not args.all_features,
                          return_battles=want_battles,
                          baseline_z=baseline_z, win_relevance=win_rel)
    if want_battles:
        df, summary, per_battle = result
    else:
        df, summary = result

    prompt_wr = relations = None
    if args.prompt_lens:
        prompt_names = pd.read_csv(args.prompt_names) if args.prompt_names else None
        bids = per_battle["instruction_id"].tolist()
        wins = per_battle["win"].to_numpy()
        prompt_wr = prompt_concept_winrates(
            args.prompt_lens, bids, wins, prompt_names=prompt_names,
            min_battles=args.min_battles)
        # per-model prompt→response: pull the per-battle z{f} codes back into an array
        feat_ids = [int(c[1:]) for c in per_battle.columns if c.startswith("z")
                    and c[1:].isdigit()]
        if feat_ids:
            resp_codes = per_battle[[f"z{f}" for f in feat_ids]].to_numpy()
            resp_names = (df[["feature_id", "concept"]] if "concept" in df else None)
            relations = prompt_to_response_winrates(
                args.prompt_lens, bids, resp_codes, feat_ids, wins,
                prompt_names=prompt_names, response_names=resp_names,
                min_support=args.min_battles, top=args.top)

    md = format_report(df, model=summary["model"], n_battles=summary["n_battles"],
                       win_rate=summary["win_rate"], top=args.top,
                       prompt_winrates=prompt_wr, relations=relations)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    feats_csv = out.with_name(f"{out.stem}_features.csv")
    _save(df, feats_csv)
    print(md)
    print(f"\nwrote report to {out} and per-feature diagnosis to {feats_csv}")
    return 0


def _cmd_win_relevance(args) -> int:
    import pandas as pd

    from prefscope.pipeline.winrelevance import win_relevance, win_relevance_logistic

    battles, z_diff, _ = load_lens_battles(args.lens_dir, corpus=args.corpus)
    if "human_pref" not in battles.columns or battles["human_pref"].isna().all():
        print("corpus has no human_pref; rebuild with `build-corpus --keep-labels`",
              file=sys.stderr)
        return 2
    feats = None
    names = pd.read_csv(args.names) if args.names else None
    if names is not None and "fidelity_pass" in names.columns and not args.all_features:
        feats = names.loc[names["fidelity_pass"].astype(bool), "feature_id"].astype(int).tolist()
    hp = battles["human_pref"].to_numpy()
    df = win_relevance(z_diff, hp, features=feats)
    # WIMHF length-controlled Δwin-rate (App. A.2): word-count difference A−B
    wc = lambda s: battles[s].fillna("").str.split().str.len().to_numpy()  # noqa: E731
    length = wc("completion_a") - wc("completion_b")
    dwr = win_relevance_logistic(z_diff, hp, length, features=feats)
    df = df.merge(dwr, on="feature_id", how="left")
    if names is not None and "concept" in names.columns:
        df = df.merge(names[["feature_id", "concept"]], on="feature_id", how="left")
        df = df[["feature_id", "concept"] + [c for c in df.columns
                                             if c not in ("feature_id", "concept")]]
    df = df.sort_values("win_assoc", ascending=False).reset_index(drop=True)
    _save(df, args.out)
    n_sig = int(df["significant"].sum())
    print(f"wrote {len(df)} feature win-relevances ({n_sig} significant) to {args.out}")

    # Anatomy-style cluster-level win-relevance (same logistic, aggregated unit)
    if args.clusters:
        from prefscope.pipeline.winrelevance import cluster_win_relevance
        cl = pd.read_csv(args.clusters)
        cdf = cluster_win_relevance(z_diff, hp, length, cl)
        cout = str(args.out).replace(".csv", "_clusters.csv")
        _save(cdf, cout)
        csig = int(cdf["delta_win_significant"].sum()) if len(cdf) else 0
        print(f"wrote {len(cdf)} cluster win-relevances ({csig} significant) to {cout}")
    return 0


def _cmd_elicit(args) -> int:
    from pathlib import Path

    from prefscope.pipeline.elicit import run_elicitation

    edges = run_elicitation(
        args.completion_lens, args.prompt_lens,
        completion_names=args.completion_names, completion_fidelity=args.completion_fidelity,
        prompt_names=args.prompt_names, prompt_fidelity=args.prompt_fidelity,
        min_support=args.min_support, min_cooccur=args.min_cooccur)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    edges.to_csv(args.out, index=False)
    nsig = int(edges["significant"].sum()) if len(edges) else 0
    print(f"wrote {len(edges)} prompt→response edges to {args.out}; "
          f"{nsig} significant (Bonferroni over {edges.attrs.get('n_tested', len(edges))} cells)")
    return 0


def _cmd_conditional_delta(args) -> int:
    from prefscope.pipeline.prompt_delta import run_prompt_conditioned_delta

    run_prompt_conditioned_delta(
        args.completion_lens, args.prompt_lens, args.out, corpus=args.corpus,
        completion_names=args.completion_names, prompt_names=args.prompt_names,
        prompt_clusters=args.prompt_clusters, conditional_out=args.conditional_out,
        completion_fidelity=args.completion_fidelity, seed=args.seed,
        permute=args.permute, jobs=args.jobs)
    return 0


def _cmd_sae_metrics(args) -> int:
    import pandas as pd

    from prefscope.analysis.sae_metrics import lens_metrics

    m = lens_metrics(args.lens_dir)
    print(json.dumps(m, indent=2, default=str))
    if args.out:
        row = pd.DataFrame([m])
        if Path(args.out).exists():                       # append a row for M-sweeps
            row = pd.concat([pd.read_csv(args.out), row], ignore_index=True)
        _save(row, args.out)
        print(f"wrote metrics row to {args.out}")
    return 0


def _cmd_cluster_features(args) -> int:
    import pandas as pd

    from prefscope.pipeline.cluster import load_cofiring_codes, summarize_clusters

    # Cluster on co-firing in INDIVIDUAL responses (z_a/z_b stacked) — semantic
    # co-occurrence, à la Anatomy — NOT on the difference. In z_diff, antonym features
    # (e.g. "refuses" and "elaborates") co-fire by construction on the same battle, so
    # MI clustering on z_diff merges opposites; clustering on individual codes doesn't.
    z = load_cofiring_codes(args.lens_dir, lens_kind=getattr(args, "lens_kind", "completion"),
                            cluster_on=getattr(args, "cluster_on", "difference"))
    names = pd.read_csv(args.names) if args.names else None
    feats = None
    if names is not None and args.fidelity_only and "fidelity_pass" in names.columns:
        feats = names.loc[names["fidelity_pass"].astype(bool), "feature_id"].astype(int).tolist()

    from prefscope.core import registry
    clusterer = registry.make("clusterer", args.method, n_clusters=args.n_clusters,
                              resolution=args.resolution, knn=args.knn,
                              min_cluster_size=args.min_cluster_size)
    clusters = clusterer.cluster(z, features=feats)
    summary = summarize_clusters(clusters, names=names)        # uses raw clusters + names

    if args.name_clusters:
        from prefscope.pipeline.cluster import name_clusters
        labels = name_clusters(summary, _make_client(args), concurrency=args.concurrency)
        summary["behavior"] = summary["cluster_id"].map(labels).where(
            summary["cluster_id"].map(labels).astype(bool), summary["behavior"])

    out = clusters.copy()
    if names is not None and "concept" in names.columns:
        out = out.merge(names[["feature_id", "concept"]], on="feature_id", how="left")
    out = out.merge(summary[["cluster_id", "behavior"]], on="cluster_id", how="left")
    _save(out, args.out)
    _save(summary, str(args.out).replace(".csv", "_summary.csv"))

    print(f"\n{len(clusters)} features -> {clusters['cluster_id'].nunique()} behaviors "
          f"({args.method})\n")
    for _, r in summary.iterrows():
        vtag = f"  [{r['n_verified']} verified]" if r.get("n_verified") is not None else ""
        print(f"  behavior {r['cluster_id']}  ({r['n_features']} feats{vtag}): {r['behavior']}")
        if r["member_concepts"]:
            print(f"      {r['member_concepts']}")
    print(f"\nwrote {args.out} (+ _summary.csv)")
    return 0


def _cmd_build_bank(args) -> int:
    import json as _json
    from pathlib import Path

    import numpy as np
    import pandas as pd

    from prefscope.encode.sae import SAEProjector
    from prefscope.pipeline.oriented_bank import build_oriented_codes, save_bank

    emb = Path(args.from_embeddings)
    e_a = np.load(emb / "e_a.npy")
    e_b = np.load(emb / "e_b.npy")
    meta = pd.read_parquet(emb / "meta.parquet").reset_index(drop=True)
    manifest = _json.loads((Path(args.lens_dir) / "manifest.json").read_text())
    input_rep = manifest.get("input_rep", "difference")

    label_col = "y_judge"
    corp = None
    if args.label == "human":
        if not args.corpus:
            print("--label human needs --corpus carrying human_pref "
                  "(build-corpus --keep-labels)", file=sys.stderr)
            return 2
        from prefscope.data.corpus import load_corpus
        corp = load_corpus(args.corpus)
        if "human_pref" not in corp.columns:
            print("corpus has no human_pref; rebuild with build-corpus --keep-labels",
                  file=sys.stderr)
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
            print("note: no corpus completion text available; bank `length` = 0.0 "
                  "(validation LOO will not be length-controlled)")

    if label_col not in meta.columns:
        print(f"embedding meta has no {label_col!r} column "
              f"(dump came from a label-free corpus?)", file=sys.stderr)
        return 2

    keep = meta[label_col].isin([0.0, 0.5, 1.0]).to_numpy()
    if not keep.all():
        print(f"dropping {int((~keep).sum())} rows with missing/invalid {label_col}")
        e_a, e_b = e_a[keep], e_b[keep]
        meta = meta[keep].reset_index(drop=True)

    projector = SAEProjector(args.lens_dir, device=args.device)
    Z, bank_meta = build_oriented_codes(e_a, e_b, meta, projector,
                                        input_rep=input_rep, label_col=label_col)
    out_manifest = save_bank(args.out, Z, bank_meta, lens_dir=args.lens_dir,
                             label_col=label_col, input_rep=input_rep)
    print(_json.dumps(out_manifest, indent=2, default=str))
    print(f"wrote oriented-code bank ({Z.shape[0]} rows, "
          f"{out_manifest['n_models']} models) to {args.out}")
    return 0


def _cmd_validate_diagnosis(args) -> int:
    import json as _json

    import pandas as pd

    from prefscope.pipeline.oriented_bank import load_bank
    from prefscope.pipeline.validate import validate_diagnosis

    bank_Z, bank_meta, _ = load_bank(args.bank)
    wr = pd.read_csv(args.win_relevance)
    df, summary = validate_diagnosis(
        bank_Z, bank_meta, wr, weight_col=args.weight_col,
        significant_only=not args.all_features, min_battles=args.min_battles,
        loo=args.loo, seed=args.seed)
    _save(df, args.out)
    print(_json.dumps(summary, indent=2, default=str))
    r2 = summary.get("loo_r2") if args.loo else summary.get("insample_r2")
    tag = "LOO" if args.loo else "in-sample"
    print(f"\n{tag} R^2 = {r2:.3f} over {summary['n_models']} models "
          f"(predicted deficit vs actual win rate)")
    print(f"wrote {len(df)} per-model rows to {args.out}")
    return 0


def _cmd_extract_activations(args) -> int:
    from prefscope.activations.cache import ActivationCache
    from prefscope.activations.extract import ActivationExtractor
    from prefscope.data.corpus import load_corpus

    battles = load_corpus(args.corpus)
    if args.n_battles and args.n_battles < len(battles):
        battles = battles.sample(n=args.n_battles, random_state=args.seed
                                 ).reset_index(drop=True)
    print(f"extracting activations for {len(battles)} battles "
          f"({args.model_id} layer {args.layer})", flush=True)
    ext = ActivationExtractor(args.model_id, args.layer, max_tokens=args.max_tokens,
                              outlier_norm_mult=args.outlier_norm_mult,
                              device=args.device, dtype=args.dtype,
                              attn_implementation=args.attn_implementation)
    cache = ActivationCache(args.out, hidden_dim=ext.hidden_dim)
    n_done = 0
    for vectors, rows in ext.iter_battle_activations(battles):
        cache.append(vectors, rows)
        n_done += 1
        if n_done % 500 == 0:
            print(f"  {n_done} spans appended ({cache._n} tokens)", flush=True)
    meta_cols = [c for c in ("battle_id", "model_a", "model_b", "source",
                             "language", "human_pref") if c in battles.columns]
    import pandas as pd
    cache.finalize(extra_manifest={"model_id": args.model_id, "layer": args.layer,
                                   "max_tokens": args.max_tokens,
                                   "outlier_norm_mult": args.outlier_norm_mult,
                                   "n_battles": int(len(battles))})
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
        cache, m_total=m_total, k=args.k,
        matryoshka_prefix=tuple(args.matryoshka_prefix), val_frac=args.val_frac,
        max_train_tokens=args.max_train_tokens, n_epochs=args.epochs,
        batch=args.batch, seed=args.seed, device=args.device)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": config}, out / "sae_model.pt")
    import pandas as pd
    pd.DataFrame(log).to_csv(out / "sae_training_log.csv", index=False)
    manifest = {"source_cache": str(args.cache), "m_total": int(m_total), "k": int(args.k),
                "input_dim": int(cache.hidden_dim),
                "best_val_norm_mse": config["best_val_norm_mse"],
                "best_val_ev": config["best_val_ev"], "dead_neurons": config["dead_neurons"],
                "n_train_tokens": config["n_train_tokens"],
                "n_val_tokens": config["n_val_tokens"],
                "model_id": cache.manifest.get("model_id"),
                "layer": cache.manifest.get("layer")}
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
    print(f"wrote {len(summaries)} (battle,span,feature) rows + "
          f"{len(span_meta)} span-meta rows to {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="prefscope")
    sub = p.add_subparsers(dest="command", required=True)

    pi = sub.add_parser("inspect", help="battle-table sanity summary (corpus or annotations)")
    pi.add_argument("--corpus", default=None,
                    help="merged corpus parquet from build-corpus (label-free)")
    pi.add_argument("--annotations", nargs="+", default=None,
                    help="OpenJury annotation JSON(s)")
    pi.set_defaults(func=_cmd_inspect)

    pc = sub.add_parser("build-corpus",
                        help="build a merged label-free battle corpus from HF arenas")
    pc.add_argument("--source", nargs="+", required=True,
                    help="arena sources: lmarena-100k lmarena-140k comparia")
    pc.add_argument("--out", required=True, help="output corpus parquet")
    pc.add_argument("--split", default="train")
    pc.add_argument("--limit", type=int, default=None,
                    help="cap battles per source (for quick trials)")
    pc.add_argument("--hf-token-env", default="HF_TOKEN", dest="hf_token_env",
                    help="env var holding an HF token (needed for gated comparia)")
    pc.add_argument("--keep-labels", action="store_true", dest="keep_labels",
                    help="carry the human vote as human_pref (y=P(A preferred)) "
                         "for win-relevance analysis")
    pc.set_defaults(func=_cmd_build_corpus)

    pb = sub.add_parser("build-lens", help="embed + train a frozen SAE lens")
    pb.add_argument("--annotations", nargs="+", default=None,
                    help="OpenJury annotation JSON(s)")
    pb.add_argument("--corpus", default=None,
                    help="merged corpus parquet from build-corpus (label-free)")
    pb.add_argument("--dump-embeddings", default=None, dest="dump_embeddings",
                    help="also save assembled embeddings here (e_a/e_b/meta) so "
                         "later --from-embeddings can retrain without re-embedding")
    pb.add_argument("--from-embeddings", default=None, dest="from_embeddings",
                    help="train from a dumped embedding set (skip corpus + cache "
                         "scan + embedding); for fast M/K sweeps")
    pb.add_argument("--out", required=True, help="output lens directory")
    pb.add_argument("--m-total", type=int, default=128, dest="m_total")
    pb.add_argument("--k", type=int, default=16)
    pb.add_argument("--matryoshka-prefix", type=int, nargs="+", default=[8],
                    dest="matryoshka_prefix",
                    help="nested Matryoshka prefix lengths; m_total is appended "
                         "automatically (WIMHF default: 8)")
    pb.add_argument("--whiten", choices=["none", "standardize", "pca"], default="none",
                    help="whiten inputs before the SAE (anisotropic embeddings): "
                         "'standardize' per-dim, 'pca' full PCA whitening (arXiv:2511.13981). "
                         "Stored with the lens and re-applied at projection.")
    pb.add_argument("--whiten-eps", type=float, default=1e-5, dest="whiten_eps")
    pb.add_argument("--sae-type", default="batchtopk", dest="sae_type",
                    help="SAE architecture: any registered SAE (built-in: batchtopk, "
                         "jumprelu, simple-topk); unknown -> error lists all. jumprelu "
                         "(JumpReLU SAE, arXiv:2407.14435 — learned per-feature thresholds; "
                         "pair with --input-rep individual); simple-topk is an ablation.")
    pb.add_argument("--sparsity-coef", type=float, default=1e-3, dest="sparsity_coef",
                    help="jumprelu: L0 sparsity penalty lambda (tune this; higher = sparser)")
    pb.add_argument("--bandwidth", type=float, default=1e-3,
                    help="jumprelu: straight-through-estimator rectangle-kernel bandwidth epsilon")
    pb.add_argument("--input-rep", choices=["difference", "individual"],
                    default="difference", dest="input_rep",
                    help="SAE input: 'difference' (e_a-e_b, WIMHF-style, default) "
                         "or 'individual' (pooled e_a,e_b)")
    pb.add_argument("--val-frac", type=float, default=0.1, dest="val_frac")
    pb.add_argument("--batch", type=int, default=512)
    pb.add_argument("--n-epochs", type=int, default=200, dest="n_epochs")
    pb.add_argument("--max-train-rows", type=int, default=None, dest="max_train_rows",
                    help="reservoir cap: train the SAE on at most N randomly-sampled "
                         "rows (the dataset is usually far larger than a small "
                         "dictionary needs)")
    pb.add_argument("--seed", type=int, default=0)
    pb.add_argument("--device", default="cuda", choices=["cuda", "mps", "cpu"])
    pb.add_argument("--embed-model-id", default=CONFIG.embed_model_id,
                    dest="embed_model_id")
    pb.add_argument("--embed-batch-size", type=int,
                    default=CONFIG.embed_batch_size, dest="embed_batch_size")
    pb.add_argument("--cache-workers", type=int, default=32, dest="cache_workers",
                    help="parallel threads for reading cached embeddings "
                         "(big speedup on parallel filesystems)")
    pb.add_argument("--embed-backend", choices=["hf", "vllm", "vllm-server"],
                    default="hf", dest="embed_backend",
                    help="'hf' (default transformers), 'vllm' (in-process), or "
                         "'vllm-server' (HTTP to a vLLM OpenAI server, e.g. a "
                         "Singularity container) — no GPU torch needed host-side")
    pb.add_argument("--tensor-parallel-size", type=int, default=1,
                    dest="tensor_parallel_size",
                    help="vLLM tensor-parallel GPUs (model must be split; for an "
                         "8B model that fits on one GPU, prefer data-parallel sharding)")
    pb.add_argument("--embed-api-base", default=None, dest="embed_api_base",
                    help="vllm-server: OpenAI-compatible /v1 URL "
                         "(e.g. http://localhost:8000/v1)")
    pb.add_argument("--embed-api-key-env", default="OPENAI_API_KEY",
                    dest="embed_api_key_env",
                    help="env var holding the server API key (vLLM ignores the value)")
    pb.add_argument("--max-tokens", type=int, default=CONFIG.max_tokens,
                    dest="max_tokens")
    pb.add_argument("--cache-dir", default=None, dest="cache_dir")
    pb.set_defaults(func=_cmd_build_lens)

    pe = sub.add_parser(
        "embed-corpus",
        help="embed one shard of a corpus into the cache (parallel multi-GPU "
             "pre-pass; then run build-lens to train from the warm cache)")
    pe.add_argument("--corpus", required=True, help="merged corpus parquet")
    pe.add_argument("--shard", type=int, default=0,
                    help="this shard index in [0, num-shards)")
    pe.add_argument("--num-shards", type=int, default=1, dest="num_shards",
                    help="total shards (= number of parallel GPU processes)")
    pe.add_argument("--device", default="cuda", choices=["cuda", "mps", "cpu"])
    pe.add_argument("--embed-model-id", default=CONFIG.embed_model_id,
                    dest="embed_model_id")
    pe.add_argument("--embed-batch-size", type=int,
                    default=CONFIG.embed_batch_size, dest="embed_batch_size")
    pe.add_argument("--max-tokens", type=int, default=CONFIG.max_tokens,
                    dest="max_tokens")
    pe.add_argument("--cache-dir", default=None, dest="cache_dir")
    pe.add_argument("--cache-workers", type=int, default=32, dest="cache_workers")
    pe.add_argument("--embed-backend", choices=["hf", "vllm", "vllm-server"],
                    default="hf", dest="embed_backend")
    pe.add_argument("--tensor-parallel-size", type=int, default=1,
                    dest="tensor_parallel_size")
    pe.add_argument("--embed-api-base", default=None, dest="embed_api_base",
                    help="vllm-server: OpenAI-compatible /v1 URL")
    pe.add_argument("--embed-api-key-env", default="OPENAI_API_KEY",
                    dest="embed_api_key_env")
    pe.set_defaults(func=_cmd_embed_corpus)

    pep = sub.add_parser(
        "embed-prompts",
        help="embed prompts alone -> battle_id-aligned e_prompt.npy for the prompt lens")
    pep.add_argument("--corpus", required=True, help="merged corpus parquet")
    pep.add_argument("--out", required=True,
                     help="output dir for e_prompt.npy + meta.parquet")
    pep.add_argument("--shard", type=int, default=0)
    pep.add_argument("--num-shards", type=int, default=1, dest="num_shards",
                     help=">1: only warm the cache for this shard (multi-GPU pre-pass)")
    pep.add_argument("--device", default="cuda", choices=["cuda", "mps", "cpu"])
    pep.add_argument("--embed-model-id", default=CONFIG.embed_model_id,
                     dest="embed_model_id")
    pep.add_argument("--embed-batch-size", type=int,
                     default=CONFIG.embed_batch_size, dest="embed_batch_size")
    pep.add_argument("--max-tokens", type=int, default=CONFIG.max_tokens,
                     dest="max_tokens")
    pep.add_argument("--cache-dir", default=None, dest="cache_dir")
    pep.add_argument("--cache-workers", type=int, default=32, dest="cache_workers")
    pep.add_argument("--embed-backend", choices=["hf", "vllm", "vllm-server"],
                     default="hf", dest="embed_backend")
    pep.add_argument("--tensor-parallel-size", type=int, default=1,
                     dest="tensor_parallel_size")
    pep.add_argument("--embed-api-base", default=None, dest="embed_api_base",
                     help="vllm-server: OpenAI-compatible /v1 URL")
    pep.add_argument("--embed-api-key-env", default="OPENAI_API_KEY",
                     dest="embed_api_key_env")
    pep.set_defaults(func=_cmd_embed_prompts)

    ppl = sub.add_parser(
        "build-prompt-lens",
        help="train a standard SAE on prompt embeddings (the prompt-concept matrix)")
    ppl.add_argument("--from-embeddings", required=True, dest="from_embeddings",
                     help="embed-prompts dump dir (e_prompt.npy + meta.parquet)")
    ppl.add_argument("--out", required=True, help="output prompt-lens directory")
    ppl.add_argument("--m-total", type=int, default=64, dest="m_total")
    ppl.add_argument("--k", type=int, default=8)
    ppl.add_argument("--matryoshka-prefix", type=int, nargs="+", default=[8],
                     dest="matryoshka_prefix")
    ppl.add_argument("--val-frac", type=float, default=0.1, dest="val_frac")
    ppl.add_argument("--batch", type=int, default=512)
    ppl.add_argument("--n-epochs", type=int, default=200, dest="n_epochs")
    ppl.add_argument("--max-train-rows", type=int, default=None, dest="max_train_rows",
                     help="reservoir cap: train the SAE on at most N randomly-sampled "
                          "rows (the dataset is usually far larger than a small "
                          "dictionary needs)")
    ppl.add_argument("--seed", type=int, default=0)
    ppl.add_argument("--device", default="cpu", choices=["cuda", "mps", "cpu"])
    ppl.add_argument("--embed-model-id", default=CONFIG.embed_model_id,
                     dest="embed_model_id", help="label only (recorded in manifest)")
    ppl.set_defaults(func=_cmd_build_prompt_lens)

    ped = sub.add_parser(
        "encode-dataset",
        help="encode an arbitrary (prompt, response[, response_2]) dataset into sparse "
             "codes with a trained lens (no training; embedder taken from the lens manifest)")
    ped.add_argument("--lens-dir", required=True, dest="lens_dir",
                     help="trained lens dir (sae_model.pt + manifest.json)")
    ped.add_argument("--data", required=True,
                     help="dataset file (.parquet / .csv / .jsonl)")
    ped.add_argument("--out", required=True, help="output dir for codes + meta + manifest")
    ped.add_argument("--prompt-col", default="prompt", dest="prompt_col")
    ped.add_argument("--response-col", default="response", dest="response_col")
    ped.add_argument("--response-2-col", default=None, dest="response2_col",
                     help="second response column; its presence switches to battle mode")
    ped.add_argument("--model-col", default=None, dest="model_col",
                     help="optional; copied to meta.parquet for later phases")
    ped.add_argument("--model-2-col", default=None, dest="model2_col",
                     help="optional; the second model's name (battle mode)")
    ped.add_argument("--label-col", default=None, dest="label_col",
                     help="optional preference/winner column; copied to meta.parquet")
    # embed knobs — model id is read from the lens manifest, NOT a flag
    ped.add_argument("--device", default="cuda", choices=["cuda", "mps", "cpu"],
                     help="device for the embedder (cuda also covers ROCm builds)")
    ped.add_argument("--embed-batch-size", type=int, default=CONFIG.embed_batch_size,
                     dest="embed_batch_size")
    ped.add_argument("--max-tokens", type=int, default=CONFIG.max_tokens, dest="max_tokens")
    ped.add_argument("--cache-dir", default=None, dest="cache_dir")
    ped.add_argument("--cache-workers", type=int, default=32, dest="cache_workers")
    ped.add_argument("--embed-backend", choices=["hf", "vllm", "vllm-server"],
                     default="hf", dest="embed_backend")
    ped.add_argument("--tensor-parallel-size", type=int, default=1,
                     dest="tensor_parallel_size")
    ped.add_argument("--embed-api-base", default=None, dest="embed_api_base",
                     help="vllm-server: OpenAI-compatible /v1 URL")
    ped.add_argument("--embed-api-key-env", default="OPENAI_API_KEY",
                     dest="embed_api_key_env")
    ped.set_defaults(func=_cmd_encode_dataset)

    pnp = sub.add_parser(
        "name-prompts",
        help="LLM-name prompt-lens features from their top-activating prompts")
    pnp.add_argument("--lens-dir", required=True, help="prompt lens dir (z_prompt.npy)")
    pnp.add_argument("--corpus", required=True, help="corpus parquet (prompt text by battle_id)")
    pnp.add_argument("--out", required=True, help="output prompt_feature_names.csv")
    pnp.add_argument("--features", type=int, nargs="*", default=None)
    pnp.add_argument("--n-active", type=int, default=12, dest="n_active")
    pnp.add_argument("--n-zero", type=int, default=8, dest="n_zero")
    pnp.add_argument("--backend", choices=["openai", "claude-cli", "codex-cli"],
                     default="openai")
    pnp.add_argument("--model", default=DEFAULT_MODEL)
    pnp.add_argument("--api-base", default=DEFAULT_API_BASE)
    pnp.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    pnp.add_argument("--concurrency", type=int, default=1)
    pnp.add_argument("--reasoning-effort", default=None, dest="reasoning_effort",
                     choices=["minimal", "low", "medium", "high"],
                     help="curb reasoning-model thinking tokens (minimal/low) to avoid "
                          "truncation and cut cost on this simple task")
    pnp.set_defaults(func=_cmd_name_prompts)

    prun = sub.add_parser(
        "run",
        help="run a config-driven pipeline (name/verify/cluster/win-relevance) from a "
             "YAML/JSON file; every component is selected by name + params in the config")
    prun.add_argument("--config", required=True,
                      help="pipeline config (.yaml/.yml/.json)")
    prun.set_defaults(func=_cmd_run)

    pn = sub.add_parser("interpret", help="name + verify SAE difference-axes")
    isub = pn.add_subparsers(dest="interpret_command", required=True)

    def _add_common(p):
        p.add_argument("--lens-dir", required=True)
        p.add_argument("--annotations", nargs="+", default=None,
                       help="annotation JSON(s) the lens was built from")
        p.add_argument("--corpus", default=None,
                       help="merged corpus parquet the lens was built from")
        p.add_argument("--out", required=True)
        p.add_argument("--backend", choices=["openai", "claude-cli", "codex-cli"],
                       default="openai")
        p.add_argument("--model", default=DEFAULT_MODEL)
        p.add_argument("--api-base", default=DEFAULT_API_BASE,
                       help="OpenAI-compatible base URL (OpenRouter default; "
                            "set to a local vLLM endpoint to run offline)")
        p.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
        p.add_argument("--verify-frac", type=float, default=0.2, dest="verify_frac")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--concurrency", type=int, default=1,
                       help="number of features to send to the LLM in parallel "
                            "(thread pool; 1 = sequential)")
        p.add_argument("--reasoning-effort", default=None, dest="reasoning_effort",
                       choices=["minimal", "low", "medium", "high"],
                       help="for reasoning models (gpt-5-mini, o-series): curb thinking "
                            "tokens. Naming isn't a reasoning task, so 'minimal'/'low' "
                            "avoids truncation (reasoning eating the token budget) and cuts "
                            "cost. Omit to use the provider default.")

    pnn = isub.add_parser("name", help="label each feature from top pairs")
    _add_common(pnn)
    pnn.add_argument("--features", type=int, nargs="*", default=None)
    pnn.add_argument("--name-mode", default="auto", dest="name_mode",
                     help="interpreter strategy: auto (default) picks individual vs pairwise "
                          "from the lens manifest's input_rep; or name any registered "
                          "strategy (built-in: individual, pairwise). Unknown -> error lists all.")
    pnn.add_argument("--n-active", type=int, default=10, dest="n_active")
    pnn.add_argument("--n-zero", type=int, default=10, dest="n_zero")
    pnn.add_argument("--negatives", choices=["random", "close"], default="random",
                     help="non-activating controls: 'random' silent responses (default) or "
                          "'close' HARD negatives — silent responses whose other concepts "
                          "resemble the activators, so the name isolates THIS feature instead "
                          "of a generic trait (e.g. formatting) the controls also share")
    pnn.add_argument("--abbreviate", action="store_true",
                     help="run the WIMHF abbreviate-concept step")
    pnn.add_argument("--debug-responses", default=None, dest="debug_responses",
                     help="dir to dump each feature's raw LLM response (feature_<id>.txt) "
                          "for debugging empty/garbage concepts")
    pnn.set_defaults(func=_cmd_interpret_name)

    pnv = isub.add_parser("verify", help="held-out fidelity of named axes")
    _add_common(pnv)
    pnv.add_argument("--names", required=True, help="feature_names.csv from `name`")
    pnv.add_argument("--verify-mode", default="auto", dest="verify_mode",
                     help="verifier strategy: auto (default) picks individual vs pairwise from "
                          "the lens manifest's input_rep; or name any registered strategy "
                          "(built-in: individual, pairwise, prompt). --lens-kind prompt forces prompt.")
    pnv.add_argument("--n-per-bucket", type=int, default=10, dest="n_per_bucket")
    pnv.add_argument("--fidelity-threshold", type=float, default=0.3,
                     dest="fidelity_threshold",
                     help="min POSITIVE correlation to pass (with Bonferroni p<0.05); a "
                          "flipped-polarity name (negative correlation) fails")
    pnv.add_argument("--lens-kind", choices=["completion", "prompt"], default="completion",
                     dest="lens_kind",
                     help="'prompt' verifies prompt-lens concepts on z_prompt + prompt text "
                          "(needs --corpus; folds the old verify_prompts.py)")
    pnv.add_argument("--negatives", default="random",
                     help="prompt verify: 'random' silent prompts or 'close' (needs --embeddings)")
    pnv.add_argument("--embeddings", default=None,
                     help="prompt verify: e_prompt.npy for 'close' negatives")
    pnv.set_defaults(func=_cmd_interpret_verify)

    pd_ = sub.add_parser(
        "diagnose",
        help="aggregate a target model's contrast codes into per-feature tendencies")
    pd_.add_argument("--lens-dir", required=True, help="frozen lens directory")
    pd_.add_argument("--annotations", nargs="+", required=True,
                     help="OpenJury annotation JSON(s) containing the target model")
    pd_.add_argument("--model", required=True, help="target model name to diagnose")
    pd_.add_argument("--out", required=True, help="output diagnosis CSV")
    pd_.add_argument("--battles-out", default=None, dest="battles_out",
                     help="optional parquet of per-battle evidence (target vs "
                          "opponent text, outcome, per-axis activation) for the viewer")
    pd_.add_argument("--bank", default=None,
                     help="oriented-code bank dir (from `build-bank`); adds the "
                          "inside-vs-outside Welch contrast vs the model pool and "
                          "sorts by distinctiveness (delta_vs_pool)")
    pd_.add_argument("--fidelity", default=None,
                     help="feature_fidelity.csv from `interpret verify`; attaches "
                          "concept names and (by default) restricts to passing axes")
    pd_.add_argument("--win-relevance", default=None, dest="win_relevance",
                     help="win-relevance CSV (from `win-relevance`); merges the global "
                          "length-controlled delta_win_rate as the headline `helps_win` "
                          "signal")
    pd_.add_argument("--all-features", action="store_true",
                     help="diagnose every feature, not just fidelity-passing ones")
    pd_.add_argument("--top", type=int, default=10,
                     help="how many over/under-expressed features to print")
    pd_.add_argument("--device", default="cuda", choices=["cuda", "mps", "cpu"])
    pd_.add_argument("--embed-model-id", default=None, dest="embed_model_id",
                     help="override the embedder; defaults to the lens manifest's "
                          "embed_model_id (recommended — leave unset)")
    pd_.add_argument("--embed-batch-size", type=int,
                     default=CONFIG.embed_batch_size, dest="embed_batch_size")
    pd_.add_argument("--max-tokens", type=int, default=CONFIG.max_tokens,
                     dest="max_tokens")
    pd_.add_argument("--cache-dir", default=None, dest="cache_dir")
    pd_.add_argument("--cache-workers", type=int, default=32, dest="cache_workers")
    pd_.add_argument("--embed-backend", choices=["hf", "vllm", "vllm-server"],
                     default="hf", dest="embed_backend",
                     help="MUST match the lens's embedder model (vectors are cached "
                          "by model_id, so the backend can differ if cache is warm)")
    pd_.add_argument("--tensor-parallel-size", type=int, default=1,
                     dest="tensor_parallel_size")
    pd_.add_argument("--embed-api-base", default=None, dest="embed_api_base",
                     help="vllm-server: OpenAI-compatible /v1 URL")
    pd_.add_argument("--embed-api-key-env", default="OPENAI_API_KEY",
                     dest="embed_api_key_env")
    pd_.set_defaults(func=_cmd_diagnose)

    prp = sub.add_parser(
        "report",
        help="human-readable per-model concept report card (markdown) over the diagnosis")
    prp.add_argument("--lens-dir", required=True, help="frozen lens directory")
    prp.add_argument("--model", required=True, help="target model to report on")
    prp.add_argument("--annotations", nargs="+", default=None,
                     help="OpenJury annotation JSON(s) containing the target model")
    prp.add_argument("--corpus", default=None,
                     help="merged corpus parquet (alternative to --annotations)")
    prp.add_argument("--names", default=None,
                     help="feature_fidelity/feature_names CSV (concept names; by default "
                          "restricts to fidelity-passing axes)")
    prp.add_argument("--win-relevance", default=None, dest="win_relevance",
                     help="win-relevance CSV -> surfaces rewarded gaps (helps_win)")
    prp.add_argument("--prompt-lens", default=None, dest="prompt_lens",
                     help="prompt lens dir (z_prompt.npy) -> adds the strong/weak "
                          "prompt-types section")
    prp.add_argument("--prompt-names", default=None, dest="prompt_names",
                     help="prompt_feature_names.csv to label prompt concepts")
    prp.add_argument("--bank", default=None,
                     help="oriented-code bank dir (build-bank); under-expression "
                          "measured vs the pool (delta_vs_pool)")
    prp.add_argument("--out", required=True, help="output markdown report path")
    prp.add_argument("--top", type=int, default=15,
                     help="how many concepts to list per section")
    prp.add_argument("--min-battles", type=int, default=20, dest="min_battles",
                     help="min battles per prompt concept for the prompt-types section")
    prp.add_argument("--all-features", action="store_true",
                     help="report on every feature, not just fidelity-passing ones")
    prp.add_argument("--device", default="cuda", choices=["cuda", "mps", "cpu"])
    prp.add_argument("--embed-model-id", default=None, dest="embed_model_id",
                     help="override the embedder; defaults to the lens manifest's value")
    prp.add_argument("--embed-batch-size", type=int,
                     default=CONFIG.embed_batch_size, dest="embed_batch_size")
    prp.add_argument("--max-tokens", type=int, default=CONFIG.max_tokens,
                     dest="max_tokens")
    prp.add_argument("--cache-dir", default=None, dest="cache_dir")
    prp.add_argument("--cache-workers", type=int, default=32, dest="cache_workers")
    prp.add_argument("--embed-backend", choices=["hf", "vllm", "vllm-server"],
                     default="hf", dest="embed_backend")
    prp.add_argument("--tensor-parallel-size", type=int, default=1,
                     dest="tensor_parallel_size")
    prp.add_argument("--embed-api-base", default=None, dest="embed_api_base")
    prp.add_argument("--embed-api-key-env", default="OPENAI_API_KEY",
                     dest="embed_api_key_env")
    prp.set_defaults(func=_cmd_report)

    pw = sub.add_parser(
        "win-relevance",
        help="which features humans reward (activation vs human_pref on the corpus)")
    pw.add_argument("--lens-dir", required=True)
    pw.add_argument("--corpus", required=True,
                    help="corpus parquet WITH human_pref (build-corpus --keep-labels)")
    pw.add_argument("--names", default=None,
                    help="feature_names/fidelity CSV to attach concepts + filter")
    pw.add_argument("--all-features", action="store_true",
                    help="score every feature, not just fidelity-passing ones")
    pw.add_argument("--clusters", default=None,
                    help="feature_clusters.csv -> ALSO emit cluster-level win-relevance "
                         "(<out>_clusters.csv): Anatomy-style per-behavior Δwin-rate")
    pw.add_argument("--out", required=True, help="output win-relevance CSV")
    pw.set_defaults(func=_cmd_win_relevance)

    pel = sub.add_parser(
        "elicit",
        help="prompt-concept -> response-concept co-activation lift (preference-independent): "
             "which response concepts appear when a prompt concept is present")
    pel.add_argument("--completion-lens", required=True, dest="completion_lens",
                     help="individual lens dir (z_a.npy + z_b.npy)")
    pel.add_argument("--prompt-lens", required=True, dest="prompt_lens",
                     help="prompt lens dir (z_prompt.npy)")
    pel.add_argument("--completion-names", default=None, dest="completion_names",
                     help="feature_names.csv (response concepts)")
    pel.add_argument("--completion-fidelity", default=None, dest="completion_fidelity",
                     help="feature_fidelity.csv -> restrict to verified response axes")
    pel.add_argument("--prompt-names", default=None, dest="prompt_names",
                     help="prompt_feature_names.csv")
    pel.add_argument("--prompt-fidelity", default=None, dest="prompt_fidelity",
                     help="prompt_feature_fidelity.csv -> restrict to verified prompt axes")
    pel.add_argument("--min-support", type=int, default=30, dest="min_support",
                     help="min responses where the prompt feature fires to test a cell")
    pel.add_argument("--min-cooccur", type=int, default=5, dest="min_cooccur",
                     help="min co-occurrences to test a cell")
    pel.add_argument("--out", required=True, help="output elicitation CSV")
    pel.set_defaults(func=_cmd_elicit)

    pcd = sub.add_parser(
        "conditional-delta",
        help="prompt-conditioned completion delta Δ_{k,f} (which response properties "
             "distinguish the winner per prompt type) + optional conditional δ_{f,k}")
    pcd.add_argument("--completion-lens", required=True, dest="completion_lens",
                     help="completion lens dir (z_diff.npy)")
    pcd.add_argument("--prompt-lens", required=True, dest="prompt_lens",
                     help="prompt lens dir (z_prompt.npy)")
    pcd.add_argument("--corpus", default=None,
                     help="corpus WITH human_pref — orients z_diff toward the winner "
                          "(without it Δ ~ 0; required for --conditional-out)")
    pcd.add_argument("--completion-names", default=None, dest="completion_names")
    pcd.add_argument("--prompt-names", default=None, dest="prompt_names")
    pcd.add_argument("--prompt-clusters", default=None, dest="prompt_clusters",
                     help="prompt_feature_clusters.csv -> condition on prompt CLUSTERS "
                          "(fewer Bonferroni tests, more power) instead of raw concepts")
    pcd.add_argument("--conditional-out", default=None, dest="conditional_out",
                     help="ALSO emit the length-controlled conditional win-rate δ_{f,k}")
    pcd.add_argument("--completion-fidelity", default=None, dest="completion_fidelity",
                     help="feature_fidelity.csv -> restrict the conditional table to verified axes")
    pcd.add_argument("--seed", type=int, default=0)
    pcd.add_argument("--permute", type=int, default=0, metavar="N",
                     help="label-permutation null: shuffle prompt-concept labels N times")
    pcd.add_argument("--jobs", type=int, default=1, metavar="N",
                     help="parallelize the permutation null across N processes")
    pcd.add_argument("--out", required=True, help="output Δ_{k,f} CSV")
    pcd.set_defaults(func=_cmd_conditional_delta)

    psm = sub.add_parser(
        "sae-metrics",
        help="redundancy + fit health metrics for a lens (decoder cosine, MI, FVU, "
             "dead-frac, L0) — track across an M sweep. NOT an absorption score.")
    psm.add_argument("--lens-dir", required=True)
    psm.add_argument("--out", default=None,
                     help="CSV to append a metrics row to (for M-sweep tables)")
    psm.set_defaults(func=_cmd_sae_metrics)

    pcl = sub.add_parser(
        "cluster-features",
        help="group co-activating SAE features into higher-level behaviors")
    pcl.add_argument("--lens-dir", required=True, help="lens dir with z_diff.npy")
    pcl.add_argument("--names", default=None,
                     help="feature_fidelity/feature_names CSV (concepts + fidelity)")
    pcl.add_argument("--n-clusters", type=int, default=10, dest="n_clusters")
    pcl.add_argument("--method", default="spherical-kmeans",
                     help="clusterer component (built-in: mi-leiden, spherical-kmeans, "
                          "agglomerative; or any registered). mi-leiden: Anatomy-style MI "
                          "co-firing graph + Leiden, count emerges. Unknown -> error lists all.")
    pcl.add_argument("--resolution", type=float, default=1.0,
                     help="mi-leiden resolution (higher -> more, smaller communities)")
    pcl.add_argument("--knn", type=int, default=0,
                     help="mi-leiden: sparsify graph to each feature's top-knn edges (0=dense)")
    pcl.add_argument("--min-cluster-size", type=int, default=1, dest="min_cluster_size",
                     help="mi-leiden: fold communities smaller than this into one bucket")
    pcl.add_argument("--fidelity-only", action="store_true", dest="fidelity_only",
                     help="cluster only fidelity-passing features")
    pcl.add_argument("--cluster-on", choices=["difference", "individual"], default="difference",
                     dest="cluster_on",
                     help="co-firing space: 'difference' (z_diff, default) or 'individual' "
                          "(z_a/z_b stacked — Anatomy-style semantic co-occurrence; avoids "
                          "merging antonym features that co-fire only in the contrast)")
    pcl.add_argument("--lens-kind", choices=["completion", "prompt"], default="completion",
                     dest="lens_kind",
                     help="'prompt' clusters z_prompt.npy on a prompt lens (folds the old "
                          "cluster_prompts.py); 'completion' (default) uses --cluster-on")
    pcl.add_argument("--name-clusters", action="store_true", dest="name_clusters",
                     help="LLM-name each behavior from its member concepts")
    pcl.add_argument("--backend", choices=["openai", "claude-cli", "codex-cli"],
                     default="openai")
    pcl.add_argument("--model", default=DEFAULT_MODEL)
    pcl.add_argument("--api-base", default=DEFAULT_API_BASE)
    pcl.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    pcl.add_argument("--concurrency", type=int, default=1)
    pcl.add_argument("--out", required=True, help="output feature_clusters.csv")
    pcl.set_defaults(func=_cmd_cluster_features)

    pbk = sub.add_parser(
        "build-bank",
        help="project every battle in BOTH orientations -> pool baseline for "
             "diagnose --bank and validate-diagnosis")
    pbk.add_argument("--lens-dir", required=True, help="frozen lens directory")
    pbk.add_argument("--from-embeddings", required=True, dest="from_embeddings",
                     help="dumped embedding dir (e_a.npy/e_b.npy/meta.parquet) from "
                          "build-lens --dump-embeddings")
    pbk.add_argument("--label", choices=["judge", "human"], default="judge",
                     help="orient outcomes by judge y_judge (default) or by human "
                          "preference (needs --corpus with human_pref)")
    pbk.add_argument("--corpus", default=None,
                     help="corpus parquet with human_pref (for --label human)")
    pbk.add_argument("--out", required=True, help="output bank directory")
    pbk.add_argument("--device", default="cpu", choices=["cuda", "mps", "cpu"],
                     help="device for the SAE forward pass (CPU is fine)")
    pbk.set_defaults(func=_cmd_build_bank)

    pv = sub.add_parser(
        "validate-diagnosis",
        help="does the diagnosed deficit predict actual win rate? (R^2 across models)")
    pv.add_argument("--bank", required=True, help="oriented-code bank dir (build-bank)")
    pv.add_argument("--win-relevance", required=True, dest="win_relevance",
                    help="win-relevance CSV (feature reward weights)")
    pv.add_argument("--out", required=True, help="output per-model CSV")
    pv.add_argument("--weight-col", default="delta_win_rate", dest="weight_col",
                    help="win-relevance column to weight features by "
                         "(default delta_win_rate: length-controlled AME)")
    pv.add_argument("--all-features", action="store_true",
                    help="weight by every feature, not just significant ones")
    pv.add_argument("--min-battles", type=int, default=20, dest="min_battles",
                    help="skip models with fewer oriented battles than this")
    pv.add_argument("--seed", type=int, default=0,
                    help="RNG seed for the bootstrap CI and permutation null")
    pv.add_argument("--loo", action="store_true",
                    help="leave-one-model-out: refit reward weights excluding each "
                         "model's own battles (honest held-out R^2)")
    pv.set_defaults(func=_cmd_validate_diagnosis)

    pxa = sub.add_parser(
        "extract-activations",
        help="extract layer-L token activations from any HF causal LM into a memmap cache")
    pxa.add_argument("--corpus", required=True)
    pxa.add_argument("--out", required=True, help="output cache dir")
    pxa.add_argument("--model-id", default="meta-llama/Llama-3.1-8B-Instruct",
                     dest="model_id")
    pxa.add_argument("--layer", type=int, default=24,
                     help="hidden layer to extract (default 24, as in Anatomy of "
                          "Post-Training for Llama-3.1-8B-Instruct; ~0.75x depth)")
    pxa.add_argument("--n-battles", type=int, default=30000, dest="n_battles",
                     help="random subsample size; 0 = all")
    pxa.add_argument("--max-tokens", type=int, default=512, dest="max_tokens")
    pxa.add_argument("--outlier-norm-mult", type=float, default=6.0,
                     dest="outlier_norm_mult")
    pxa.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    pxa.add_argument("--dtype", default="bfloat16")
    pxa.add_argument("--attn-implementation", default="sdpa",
                     dest="attn_implementation",
                     help="HF attn backend; 'sdpa' (default) works on CUDA+ROCm, "
                          "'eager' is the safe fallback on AMD/ROCm")
    pxa.add_argument("--seed", type=int, default=0)
    pxa.set_defaults(func=_cmd_extract_activations)

    pts = sub.add_parser(
        "train-token-sae",
        help="stream-train a BatchTopK SAE from an activation cache")
    pts.add_argument("--cache", required=True, help="extract-activations cache dir")
    pts.add_argument("--out", required=True, help="output SAE dir")
    pts.add_argument("--expansion", type=int, default=8,
                     help="m_total = expansion * hidden_dim (ignored if --m-total set)")
    pts.add_argument("--m-total", type=int, default=0, dest="m_total",
                     help="explicit feature count; overrides --expansion")
    pts.add_argument("--k", type=int, default=64)
    pts.add_argument("--matryoshka-prefix", type=int, nargs="+", default=[8],
                     dest="matryoshka_prefix")
    pts.add_argument("--val-frac", type=float, default=0.05, dest="val_frac")
    pts.add_argument("--max-train-tokens", type=int, default=40_000_000,
                     dest="max_train_tokens", help="reservoir cap on training rows")
    pts.add_argument("--epochs", type=int, default=2)
    pts.add_argument("--batch", type=int, default=4096)
    pts.add_argument("--seed", type=int, default=0)
    pts.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    pts.set_defaults(func=_cmd_train_token_sae)

    psa = sub.add_parser(
        "summarize-activations",
        help="project cached activations through the SAE -> per-span X^max/X^freq")
    psa.add_argument("--cache", required=True)
    psa.add_argument("--sae", required=True, help="train-token-sae output dir")
    psa.add_argument("--out", required=True, help="output summaries dir")
    psa.add_argument("--batch", type=int, default=8192)
    psa.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    psa.set_defaults(func=_cmd_summarize_activations)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
