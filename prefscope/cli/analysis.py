from __future__ import annotations

import json
import sys
from pathlib import Path

from prefscope.analysis.presence import annotation_flag
from prefscope.config import CONFIG
from prefscope.data.ingest import load_battles
from prefscope.encode.cache import NpyCache
from prefscope.encode.embed import Embedder
from prefscope.interpret.io import load_lens_battles

from prefscope.cli.common import (
    _save,
    _tracked_client,
    _write_usage,
)


def _cmd_context_profile(args) -> int:
    """Export feature prompt-dependence and model cross-context stability."""
    import numpy as np
    import pandas as pd

    from prefscope.analysis.context import (
        profile_feature_context,
        profile_prompt_linkage,
    )
    from prefscope.analysis.presence import concept_presence
    from prefscope.analysis.prompt_regions import (
        prompt_region_membership,
        regions_from_feature_presence,
    )
    from prefscope.artifacts import BATTLES, Z_A, Z_B, Z_PROMPT, lens_battle_ids

    completion_dir = Path(args.completion_lens)
    prompt_dir = Path(args.prompt_lens)
    completion_meta = pd.read_parquet(completion_dir / BATTLES)
    prompt_meta = pd.read_parquet(prompt_dir / BATTLES)
    completion_ids = lens_battle_ids(completion_meta)
    prompt_ids = lens_battle_ids(prompt_meta)
    if len(completion_ids) != len(prompt_ids) or not np.array_equal(
        completion_ids, prompt_ids
    ):
        print(
            "completion and prompt lenses must contain identical battle IDs in the same "
            "order; rebuild them from the same corpus",
            file=sys.stderr,
        )
        return 2
    missing_models = {"model_a", "model_b"}.difference(completion_meta.columns)
    if missing_models:
        print(
            f"completion battles are missing model columns: {sorted(missing_models)}",
            file=sys.stderr,
        )
        return 2
    for path in (completion_dir / Z_A, completion_dir / Z_B, prompt_dir / Z_PROMPT):
        if not path.exists():
            print(f"missing required lens artifact: {path}", file=sys.stderr)
            return 2
    z_a = np.load(completion_dir / Z_A, mmap_mode="r")
    z_b = np.load(completion_dir / Z_B, mmap_mode="r")
    z_prompt = np.load(prompt_dir / Z_PROMPT, mmap_mode="r")
    if len(z_prompt) != len(completion_meta):
        print("prompt code rows do not align with completion battles", file=sys.stderr)
        return 2
    calibration = pd.read_csv(args.calibration) if args.calibration else None
    names = pd.read_csv(args.names) if args.names else None
    prompt_names = pd.read_csv(args.prompt_names) if args.prompt_names else None
    prompt_tables = [
        table
        for table in (
            prompt_names,
            pd.read_csv(args.prompt_fidelity) if args.prompt_fidelity else None,
            pd.read_csv(args.prompt_calibration) if args.prompt_calibration else None,
        )
        if table is not None
    ]
    if prompt_tables:
        prompt_annotations = pd.concat(prompt_tables, ignore_index=True, sort=False)
        prompt_annotations["feature_id"] = prompt_annotations["feature_id"].astype(int)
        prompt_annotations = prompt_annotations.groupby(
            "feature_id", as_index=False, sort=True
        ).last()
    else:
        prompt_annotations = pd.DataFrame(
            {"feature_id": np.arange(z_prompt.shape[1], dtype=int)}
        )
    prompt_ids = prompt_annotations["feature_id"].astype(int).tolist()
    if "fidelity_pass" in prompt_annotations.columns:
        prompt_ids = (
            prompt_annotations.loc[
                prompt_annotations["fidelity_pass"].map(annotation_flag), "feature_id"
            ]
            .astype(int)
            .tolist()
        )
    if not prompt_ids:
        print(
            "no prompt concepts remain after fidelity filtering",
            file=sys.stderr,
        )
        return 2
    cluster_table = pd.read_csv(args.prompt_clusters) if args.prompt_clusters else None
    context_names = prompt_names
    if cluster_table is not None and "behavior" in cluster_table.columns:
        context_names = (
            cluster_table.dropna(subset=["cluster_id", "behavior"])
            .groupby("cluster_id", as_index=False)["behavior"]
            .first()
            .rename(columns={"cluster_id": "feature_id", "behavior": "concept"})
        )
    if calibration is None:
        if args.model_out:
            print("--model-out requires --calibration", file=sys.stderr)
            return 2
        prompt_context_ids, _, prompt_context_scores = prompt_region_membership(
            z_prompt,
            feature_ids=prompt_ids,
            clusters=cluster_table,
        )
        if not len(prompt_context_ids):
            print("no non-empty prompt regions remain after filtering", file=sys.stderr)
            return 2
        print(
            f"prompt contexts: {len(prompt_context_ids)} high-activation axes"
        )
        features = profile_prompt_linkage(
            z_a,
            z_b,
            prompt_context_scores,
            features=names,
            prompt_names=context_names,
            prompt_context_ids=prompt_context_ids,
            top_n=args.top_n,
            min_top_examples=args.min_top_examples,
            prompt_tail_fractions=args.prompt_tail_fractions,
            min_tail_overlap=args.min_tail_overlap,
            min_context_lift=args.min_link_lift,
            q_threshold=args.link_q_threshold,
            min_stable_scales=args.min_link_scales,
        )
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        features.to_csv(args.out, index=False)
        counts = (
            features["prompt_scope"].value_counts().to_dict()
            if len(features)
            else {}
        )
        print(
            f"wrote {len(features)} LLM-free prompt-link profiles "
            f"to {args.out}: {counts}"
        )
        return 0
    if not args.model_out:
        print("calibrated context profiling requires --model-out", file=sys.stderr)
        return 2
    prompt_presence = concept_presence(
        z_prompt,
        prompt_annotations,
        feature_ids=prompt_ids,
        policy=args.prompt_presence_policy,
    )
    if not len(prompt_presence.feature_ids):
        print(
            "no prompt concepts remain under the requested presence policy; pass "
            "--prompt-presence-policy mixed or provide prompt calibration",
            file=sys.stderr,
        )
        return 2
    prompt_context_ids, prompt_context = regions_from_feature_presence(
        prompt_presence.values, prompt_presence.feature_ids, clusters=cluster_table
    )
    if not len(prompt_context_ids):
        print("no non-empty prompt regions remain after filtering", file=sys.stderr)
        return 2
    print(
        f"prompt contexts: {len(prompt_context_ids)} overlapping regions, "
        f"{prompt_context.sum(axis=1).mean():.2f} memberships/prompt"
    )
    features, models = profile_feature_context(
        z_a,
        z_b,
        calibration,
        prompt_context,
        completion_meta["model_a"].astype(str).to_numpy(),
        completion_meta["model_b"].astype(str).to_numpy(),
        names=names,
        prompt_names=context_names,
        prompt_context_ids=prompt_context_ids,
        min_context_occurrences=args.min_context_occurrences,
        min_model_context_battles=args.min_model_context_battles,
        min_model_context_discordant=args.min_model_context_discordant,
        min_stable_contexts=args.min_stable_contexts,
        consistency_threshold=args.consistency_threshold,
        q_threshold=args.q_threshold,
        general_min_contexts=args.general_min_contexts,
        general_max_context_share=args.general_max_context_share,
        general_max_prompt_dependence=args.general_max_prompt_dependence,
        min_choice_ratio=args.min_choice_ratio,
        prompt_content_max_choice=args.prompt_content_max_choice,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.out, index=False)
    models.to_parquet(args.model_out, index=False)
    counts = (
        features["behavior_category"].value_counts().to_dict() if len(features) else {}
    )
    print(f"wrote {len(features)} feature context profiles to {args.out}: {counts}")
    print(f"wrote {len(models)} model-feature stability rows to {args.model_out}")
    return 0


def _print_diagnosis(summary, df, top: int) -> None:
    print(
        f"\n{summary['model']} vs pool — {summary['n_battles']} battles, "
        f"win rate {summary['win_rate']:.3f}, {summary['n_features']} features\n"
    )
    has_concept = "concept" in df.columns
    label = (
        (lambda r: str(r["concept"] or f"feature {int(r['feature_id'])}"))
        if has_concept
        else (lambda r: f"feature {int(r['feature_id'])}")
    )

    has_pool = "delta_vs_pool" in df.columns
    # headline helps-win signal is length-controlled: global delta_win_rate
    # (helps_win) if merged, else the within-model length-controlled AME
    # (outcome_assoc_lc), falling back to the raw outcome_assoc.
    assoc_col = next(
        (
            c
            for c in ("helps_win", "outcome_assoc_lc", "outcome_assoc")
            if c in df.columns
        ),
        "outcome_assoc",
    )
    assoc_label = "helps-win" if assoc_col != "outcome_assoc" else "win-assoc"

    def _line(r):
        assoc = r.get(assoc_col, float("nan"))
        assoc_s = "  n/a   " if assoc != assoc else f"{assoc:+.2f}"
        pool_s = ""
        if has_pool:
            d = r["delta_vs_pool"]
            star = "*" if r.get("welch_p_bonferroni", 1.0) < 0.05 else " "
            pool_s = f"Δpool {d:+.2f}{star} "
        return (
            f"  {r['net_direction']:+.2f}  differs {r['fire_rate']:5.0%}  "
            f"{pool_s}{assoc_label} {assoc_s}  {label(r)}"
        )

    print("Most OVER-expressed vs peers (does MORE than others):")
    for _, r in df.head(top).iterrows():
        print(_line(r))
    print("\nMost UNDER-expressed vs peers (does LESS than others):")
    for _, r in df.tail(top).iloc[::-1].iterrows():
        print(_line(r))


def _cmd_diagnose(args) -> int:
    import json as _json
    from pathlib import Path

    import pandas as pd

    from prefscope.encode.sae import SAEProjector
    from prefscope.core.manifest import LensManifest

    battles = load_battles(args.annotations)
    manifest = _json.loads((Path(args.lens_dir) / "manifest.json").read_text())
    typed = LensManifest.from_dict(manifest)
    input_rep = typed.input_rep
    # the manifest is the source of truth for which embedder this lens expects;
    # only fall back to the config default if the user explicitly overrode it.
    embed_model_id = (
        args.embed_model_id or manifest.get("embed_model_id") or CONFIG.embed_model_id
    )
    if not args.embed_model_id and manifest.get("embed_model_id"):
        print(f"embedder: {embed_model_id} (from lens manifest)")
    elif not args.embed_model_id:
        print(
            f"warning: lens manifest has no embed_model_id; falling back to config "
            f"default {embed_model_id} — pass --embed-model-id if this lens used another.",
            file=sys.stderr,
        )

    cache = NpyCache(args.cache_dir or CONFIG.cache_dir)
    embedder = Embedder(
        cache,
        model_id=embed_model_id,
        model_revision=typed.embed_model_revision,
        device=args.device,
        max_tokens=typed.max_tokens or args.max_tokens,
        batch_size=args.embed_batch_size,
        cache_workers=args.cache_workers,
        backend=args.embed_backend,
        tensor_parallel_size=args.tensor_parallel_size,
        api_base=args.embed_api_base,
        api_key_env=args.embed_api_key_env,
        embed_instruction=typed.embed_instruction or CONFIG.embed_instruction,
        pooling=typed.pooling or "last-token",
        normalization=typed.normalization or "l2",
        dtype=typed.dtype,
    )
    projector = SAEProjector(args.lens_dir, device=args.device)
    typed.validate_projector(projector)

    baseline_z = None
    if args.bank:
        from prefscope.pipeline.oriented_bank import load_bank

        bank_Z, bank_meta, _ = load_bank(args.bank)
        other = (bank_meta["self_model"] != args.model).to_numpy()
        baseline_z = bank_Z[other]
        print(f"baseline: {int(other.sum())} pool rows from bank {args.bank}")

    names = pd.read_csv(args.fidelity) if args.fidelity else None
    win_rel = pd.read_csv(args.win_relevance) if args.win_relevance else None
    from prefscope.pipeline.diagnose import run_diagnose  # lazy (torch)

    result = run_diagnose(
        battles,
        args.model,
        embedder,
        projector,
        input_rep=input_rep,
        names=names,
        fidelity_only=not args.all_features,
        return_battles=bool(args.battles_out),
        baseline_z=baseline_z,
        win_relevance=win_rel,
    )
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
    from prefscope.core.manifest import LensManifest

    manifest = _json.loads((Path(args.lens_dir) / "manifest.json").read_text())
    typed = LensManifest.from_dict(manifest)
    input_rep = typed.input_rep
    embed_model_id = (
        args.embed_model_id or manifest.get("embed_model_id") or CONFIG.embed_model_id
    )
    if not args.embed_model_id and manifest.get("embed_model_id"):
        print(f"embedder: {embed_model_id} (from lens manifest)")
    elif not args.embed_model_id:
        print(
            f"warning: lens manifest has no embed_model_id; falling back to config "
            f"default {embed_model_id} — pass --embed-model-id if this lens used another.",
            file=sys.stderr,
        )
    cache = NpyCache(args.cache_dir or CONFIG.cache_dir)
    embedder = Embedder(
        cache,
        model_id=embed_model_id,
        model_revision=typed.embed_model_revision,
        device=args.device,
        max_tokens=typed.max_tokens or args.max_tokens,
        batch_size=args.embed_batch_size,
        cache_workers=args.cache_workers,
        backend=args.embed_backend,
        tensor_parallel_size=args.tensor_parallel_size,
        api_base=args.embed_api_base,
        api_key_env=args.embed_api_key_env,
        embed_instruction=typed.embed_instruction or CONFIG.embed_instruction,
        pooling=typed.pooling or "last-token",
        normalization=typed.normalization or "l2",
        dtype=typed.dtype,
    )
    projector = SAEProjector(args.lens_dir, device=args.device)
    typed.validate_projector(projector)
    return embedder, projector, input_rep


def _cmd_report(args) -> int:
    import pandas as pd

    from prefscope.pipeline.report import (
        format_report,
        prompt_concept_winrates,
        prompt_to_response_winrates,
    )

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
    from prefscope.pipeline.diagnose import run_diagnose  # lazy (torch)

    result = run_diagnose(
        battles,
        args.model,
        embedder,
        projector,
        input_rep=input_rep,
        names=names,
        fidelity_only=not args.all_features,
        return_battles=want_battles,
        baseline_z=baseline_z,
        win_relevance=win_rel,
    )
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
            args.prompt_lens,
            bids,
            wins,
            prompt_names=prompt_names,
            min_battles=args.min_battles,
            min_prompt_activation=args.min_prompt_activation,
        )
        # per-model prompt→response: pull the per-battle z{f} codes back into an array
        feat_ids = [
            int(c[1:])
            for c in per_battle.columns
            if c.startswith("z") and c[1:].isdigit()
        ]
        if feat_ids:
            resp_codes = per_battle[[f"z{f}" for f in feat_ids]].to_numpy()
            resp_names = df[["feature_id", "concept"]] if "concept" in df else None
            relations = prompt_to_response_winrates(
                args.prompt_lens,
                bids,
                resp_codes,
                feat_ids,
                wins,
                prompt_names=prompt_names,
                response_names=resp_names,
                min_support=args.min_battles,
                top=args.top,
                min_prompt_activation=args.min_prompt_activation,
            )

    md = format_report(
        df,
        model=summary["model"],
        n_battles=summary["n_battles"],
        win_rate=summary["win_rate"],
        top=args.top,
        prompt_winrates=prompt_wr,
        relations=relations,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    feats_csv = out.with_name(f"{out.stem}_features.csv")
    _save(df, feats_csv)
    print(md)
    print(f"\nwrote report to {out} and per-feature diagnosis to {feats_csv}")
    return 0


def _cmd_associate_outcomes(args) -> int:
    """Associate frozen sparse codes with binary, preference, or rating outcomes."""
    import numpy as np
    import pandas as pd

    from prefscope.analysis.grouping import resolve_group_ids
    from prefscope.analysis.outcomes import associate_outcomes, normalize_outcomes

    encoded = Path(args.encoded_dir)
    meta_path = encoded / "meta.parquet"
    if not meta_path.exists():
        print(f"encoded bundle is missing {meta_path}", file=sys.stderr)
        return 2
    meta = pd.read_parquet(meta_path)
    requested = args.code_array
    candidates = [requested] if requested != "auto" else ["z_a", "z_diff", "z_prompt"]
    code_path = next((encoded / f"{name}.npy" for name in candidates
                      if (encoded / f"{name}.npy").exists()), None)
    if code_path is None:
        print(
            f"encoded bundle has none of {[f'{name}.npy' for name in candidates]}",
            file=sys.stderr,
        )
        return 2
    missing = [column for column in args.outcome_col if column not in meta.columns]
    if missing:
        print(f"outcome columns are absent: {missing}", file=sys.stderr)
        return 2
    codes = np.load(code_path, mmap_mode="r")
    if codes.ndim != 2 or len(codes) != len(meta):
        print(
            f"encoded code/metadata mismatch: {getattr(codes, 'shape', None)} vs "
            f"{len(meta)} rows", file=sys.stderr)
        return 2
    values = meta[args.outcome_col]
    if len(args.outcome_col) == 1 and args.outcome_kind != "multi_continuous":
        values = values.iloc[:, 0]
    try:
        outcomes = normalize_outcomes(
            values, kind=args.outcome_kind, names=args.outcome_col,
            normalization=args.normalization)
        groups = None if args.no_grouping else resolve_group_ids(
            meta, group_col=args.group_col)
        result = associate_outcomes(
            codes, outcomes, group_ids=groups, min_units=args.min_units)
    except ValueError as exc:
        print(f"outcome analysis error: {exc}", file=sys.stderr)
        return 2
    table = result.table
    if args.names:
        names = pd.read_csv(args.names)
        if not {"feature_id", "concept"} <= set(names.columns):
            print("--names must contain feature_id and concept", file=sys.stderr)
            return 2
        table = table.merge(
            names[["feature_id", "concept"]].drop_duplicates("feature_id"),
            on="feature_id", how="left")
    _save(table, args.out)
    sidecar = Path(args.out).with_name(f"{Path(args.out).stem}_outcomes.json")
    sidecar.write_text(json.dumps({
        "schema_version": 1,
        "code_array": code_path.stem,
        "outcome_kind": outcomes.kind,
        "outcome_names": list(outcomes.names),
        "normalization": outcomes.normalization,
        "center": outcomes.center.tolist(),
        "scale": outcomes.scale.tolist(),
        "association_normalization": table[[
            "outcome", "association_outcome_center", "association_outcome_scale",
        ]].drop_duplicates("outcome").to_dict(orient="records"),
        "grouped": result.grouped,
        "method": result.method,
        "estimand": result.estimand,
    }, indent=2))
    print(
        f"wrote {len(table)} feature/outcome associations to {args.out} "
        f"({outcomes.n_attributes} outcomes; {'grouped' if result.grouped else 'row-level'})")
    return 0


def _cmd_win_relevance(args) -> int:
    import numpy as np
    import pandas as pd

    from prefscope.analysis.grouping import resolve_group_ids
    from prefscope.pipeline.winrelevance import win_relevance, win_relevance_logistic

    if args.encoded_dir:
        if args.lens_dir or args.corpus:
            print(
                "--encoded-dir cannot be combined with --lens-dir/--corpus",
                file=sys.stderr,
            )
            return 2
        encoded = Path(args.encoded_dir)
        meta_path = encoded / "meta.parquet"
        codes_path = encoded / "z_diff.npy"
        missing = [str(path) for path in (meta_path, codes_path) if not path.exists()]
        if missing:
            print(f"encoded bundle is missing {missing}", file=sys.stderr)
            return 2
        battles = pd.read_parquet(meta_path)
        z_diff = np.load(codes_path, mmap_mode="r")
        if len(battles) != len(z_diff):
            print(
                f"encoded bundle row mismatch: {len(battles)} metadata rows vs "
                f"{len(z_diff)} code rows",
                file=sys.stderr,
            )
            return 2
    else:
        if not args.lens_dir or not args.corpus:
            print(
                "provide either --encoded-dir or both --lens-dir and --corpus",
                file=sys.stderr,
            )
            return 2
        battles, z_diff, _ = load_lens_battles(args.lens_dir, corpus=args.corpus)
    if "human_pref" not in battles.columns or battles["human_pref"].isna().all():
        print(
            "data has no human_pref; map a preference label with `prepare-dataset`",
            file=sys.stderr,
        )
        return 2
    try:
        human_pref = pd.to_numeric(battles["human_pref"], errors="raise")
    except (TypeError, ValueError):
        print(
            "human_pref must be numeric P(A preferred); normalize winner labels with "
            "`prepare-dataset --label-mode ...`",
            file=sys.stderr,
        )
        return 2
    invalid = human_pref.notna() & ~human_pref.between(0.0, 1.0)
    if invalid.any():
        print("human_pref contains values outside [0, 1]", file=sys.stderr)
        return 2
    labeled = human_pref.notna().to_numpy()
    battles = battles.loc[labeled].reset_index(drop=True)
    z_diff = np.asarray(z_diff[labeled])
    hp = human_pref[labeled].to_numpy(dtype=float)

    feats = None
    names = pd.read_csv(args.names) if args.names else None
    if names is not None and "fidelity_pass" in names.columns and not args.all_features:
        feats = (
            names.loc[names["fidelity_pass"].map(annotation_flag), "feature_id"]
            .astype(int)
            .tolist()
        )
    try:
        group_ids = resolve_group_ids(battles, group_col=args.group_col)
    except ValueError as exc:
        print(f"invalid grouping: {exc}", file=sys.stderr)
        return 2
    df = win_relevance(z_diff, hp, features=feats, group_ids=group_ids)
    # WIMHF length-controlled Δwin-rate (App. A.2): word-count difference A−B
    if "completion_b" not in battles.columns:
        print("error: win-relevance compares two responses and this data has only one "
              "(no completion_b column)", file=sys.stderr)
        return 2
    wc = lambda s: battles[s].fillna("").str.split().str.len().to_numpy()  # noqa: E731
    length = wc("completion_a") - wc("completion_b")
    dwr = win_relevance_logistic(
        z_diff, hp, length, features=feats, group_ids=group_ids).rename(columns={
            "n_groups": "delta_win_n_groups",
            "n_independent_groups": "delta_win_n_independent_groups",
            "estimand": "delta_win_estimand",
            "inference_test": "delta_win_inference_test",
        })
    df = df.merge(dwr, on="feature_id", how="left")
    if names is not None and "concept" in names.columns:
        df = df.merge(names[["feature_id", "concept"]], on="feature_id", how="left")
        df = df[
            ["feature_id", "concept"]
            + [c for c in df.columns if c not in ("feature_id", "concept")]
        ]
    decisive = hp != 0.5
    outcome_varies = len(np.unique(hp[decisive])) > 1 if decisive.any() else False
    if outcome_varies:
        df = df.sort_values("win_assoc", ascending=False).reset_index(drop=True)
    else:
        order = (
            df["preferred_minus_rejected_mean"]
            .abs()
            .sort_values(ascending=False, na_position="last")
            .index
        )
        df = df.reindex(order).reset_index(drop=True)
    _save(df, args.out)
    n_sig = int(df["significant"].sum())
    print(f"wrote {len(df)} feature win-relevances ({n_sig} significant) to {args.out}")

    summary = {
        "n_labeled": int(len(hp)),
        "n_decisive": int(decisive.sum()),
        "n_ties": int((hp == 0.5).sum()),
        # Treat a tie as half a win; this is exactly mean(P(A preferred)).
        "a_win_rate_ties_half": float(hp.mean()),
        "a_win_rate_decisive": (
            float((hp[decisive] > 0.5).mean()) if decisive.any() else None
        ),
        "a_b_outcome_varies": outcome_varies,
        "label_semantics": "human_pref = P(A preferred)",
    }
    if {"model_a", "model_b"} <= set(battles.columns):
        side_a = pd.DataFrame(
            {
                "model": battles["model_a"].astype("string"),
                "score": hp,
            }
        )
        side_b = pd.DataFrame(
            {
                "model": battles["model_b"].astype("string"),
                "score": 1.0 - hp,
            }
        )
        model_rows = pd.concat([side_a, side_b], ignore_index=True)
        model_rows = model_rows[
            model_rows["model"].notna() & (model_rows["model"].str.strip() != "")
        ]
        if not model_rows.empty:
            models = (
                model_rows.groupby("model", as_index=False)
                .agg(
                    n_battles=("score", "size"),
                    win_rate_ties_half=("score", "mean"),
                    n_wins=("score", lambda values: int((values > 0.5).sum())),
                    n_losses=("score", lambda values: int((values < 0.5).sum())),
                    n_ties=("score", lambda values: int((values == 0.5).sum())),
                )
                .sort_values(
                    ["win_rate_ties_half", "n_battles"],
                    ascending=[False, False],
                )
                .reset_index(drop=True)
            )
            models_path = Path(args.out).with_name(f"{Path(args.out).stem}_models.csv")
            models.to_csv(models_path, index=False)
            summary["models_output"] = str(models_path)
            summary["n_models"] = int(len(models))
    summary_path = Path(args.out).with_name(f"{Path(args.out).stem}_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(
        f"A win rate: {summary['a_win_rate_ties_half']:.3f} (ties=0.5; "
        f"{summary['n_labeled']} labeled rows); wrote {summary_path}"
    )
    if decisive.any() and not outcome_varies:
        print(
            "note: A/B outcome is constant; use the winner-oriented "
            "preferred_minus_rejected_* columns. Correlation and logistic "
            "delta-win fields are undefined."
        )

    # Anatomy-style cluster-level win-relevance (same logistic, aggregated unit)
    if args.clusters:
        from prefscope.pipeline.winrelevance import cluster_win_relevance

        cl = pd.read_csv(args.clusters)
        cdf = cluster_win_relevance(
            z_diff, hp, length, cl, group_ids=group_ids)
        cout = str(args.out).replace(".csv", "_clusters.csv")
        _save(cdf, cout)
        csig = int(cdf["delta_win_significant"].sum()) if len(cdf) else 0
        print(f"wrote {len(cdf)} cluster win-relevances ({csig} significant) to {cout}")
    return 0


def _cmd_screen_confounds(args) -> int:
    """Screen preference-associated concepts for response-length entanglement."""
    import pandas as pd

    from prefscope.analysis.grouping import resolve_group_ids
    from prefscope.pipeline.confounds import screen_length_confound

    battles, z_diff, _ = load_lens_battles(args.lens_dir, corpus=args.corpus)
    needed = {"human_pref", "completion_a", "completion_b"}
    missing = needed.difference(battles.columns)
    if missing:
        print(
            f"confound screen is missing corpus columns: {sorted(missing)}",
            file=sys.stderr,
        )
        return 2
    human_pref = pd.to_numeric(battles["human_pref"], errors="coerce")
    valid = human_pref.notna().to_numpy()
    if not valid.any():
        print("corpus has no usable human_pref labels", file=sys.stderr)
        return 2
    invalid = human_pref[valid].lt(0.0) | human_pref[valid].gt(1.0)
    if invalid.any():
        print("human_pref contains values outside [0, 1]", file=sys.stderr)
        return 2
    selected = battles.loc[valid].reset_index(drop=True)
    codes = z_diff[valid]
    word_count = lambda column: (  # noqa: E731
        selected[column].fillna("").str.split().str.len().to_numpy(dtype=float)
    )
    length = word_count("completion_a") - word_count("completion_b")
    annotations = pd.read_csv(args.names) if args.names else None
    try:
        group_ids = resolve_group_ids(selected, group_col=args.group_col)
    except ValueError as exc:
        print(f"invalid grouping: {exc}", file=sys.stderr)
        return 2
    result, summary = screen_length_confound(
        codes,
        human_pref[valid].to_numpy(dtype=float),
        length,
        annotations=annotations,
        group_ids=group_ids,
        confound_threshold=args.confound_threshold,
        collapse_fraction=args.collapse_fraction,
        permutations=args.permute,
        seed=args.seed,
    )
    _save(result, args.out)
    print(json.dumps(summary, indent=2))
    print(
        f"wrote {len(result)} features to {args.out}; "
        f"{summary['n_confound_entangled']} length-entangled"
    )
    return 0


def _cmd_elicit(args) -> int:
    from pathlib import Path

    from prefscope.pipeline.elicit import run_elicitation

    edges = run_elicitation(
        args.completion_lens,
        args.prompt_lens,
        completion_names=args.completion_names,
        completion_fidelity=args.completion_fidelity,
        prompt_names=args.prompt_names,
        prompt_fidelity=args.prompt_fidelity,
        min_support=args.min_support,
        min_cooccur=args.min_cooccur,
        group_col=args.group_col,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    edges.to_csv(args.out, index=False)
    nsig = int(edges["significant"].sum()) if len(edges) else 0
    print(
        f"wrote {len(edges)} prompt→response edges to {args.out}; "
        f"{nsig} significant (Bonferroni over {edges.attrs.get('n_tested', len(edges))} cells)"
    )
    return 0


def _cmd_conditional_delta(args) -> int:
    from prefscope.pipeline.prompt_delta import run_prompt_conditioned_delta

    run_prompt_conditioned_delta(
        args.completion_lens,
        args.prompt_lens,
        args.out,
        corpus=args.corpus,
        completion_names=args.completion_names,
        prompt_names=args.prompt_names,
        prompt_clusters=args.prompt_clusters,
        conditional_out=args.conditional_out,
        completion_fidelity=args.completion_fidelity,
        prompt_fidelity=args.prompt_fidelity,
        min_prompt_activation=args.min_prompt_activation,
        min_prompt_support=args.min_prompt_support,
        seed=args.seed,
        permute=args.permute,
        jobs=args.jobs,
        group_col=args.group_col,
    )
    return 0


def _cmd_sae_metrics(args) -> int:
    import pandas as pd

    from prefscope.analysis.sae_metrics import lens_metrics

    m = lens_metrics(args.lens_dir)
    print(json.dumps(m, indent=2, default=str))
    if args.out:
        row = pd.DataFrame([m])
        if Path(args.out).exists():  # append a row for M-sweeps
            row = pd.concat([pd.read_csv(args.out), row], ignore_index=True)
        _save(row, args.out)
        print(f"wrote metrics row to {args.out}")
    return 0


def _cmd_select_lens(args) -> int:
    import pandas as pd

    from prefscope.analysis.sae_selection import expansion_ratio, recommend_config

    sweep = pd.read_csv(args.sweep)
    rec = recommend_config(sweep, n_rows=args.n_rows)
    table = rec.pop("table")
    cols = [c for c in ("m_total", "k", "fvu", "dead_frac", "l0_mean",
                        "decoder_cos_mean_max", "admissible", "rejected_because")
            if c in table.columns]
    print(table[cols].to_string(index=False))
    print()
    if not rec["admissible"]:
        print("!! no configuration passed every check; showing the least-bad one",
              file=sys.stderr)
    print(f"recommended: M={rec['m_total']} K={rec['k']} (FVU {rec['fvu']:.3f}, "
          f"{rec['n_admissible']}/{rec['n_evaluated']} admissible)")
    if args.input_dim:
        ratio = expansion_ratio(rec["m_total"], args.input_dim)
        regime = "overcomplete" if ratio >= 1 else "UNDERCOMPLETE"
        print(f"expansion ratio {ratio:.2f}x of {args.input_dim} dims ({regime})")
    if args.out:
        _save(table, args.out)
        print(f"wrote annotated sweep to {args.out}")
    return 0


def _cmd_feature_relations(args) -> int:
    """Export non-destructive activation/name/decoder feature relationships."""
    import pandas as pd

    from prefscope.analysis.feature_graph import (
        feature_relationship_summary,
        feature_relationships,
        load_decoder_directions,
    )
    from prefscope.pipeline.cluster import load_cofiring_codes

    z = load_cofiring_codes(
        args.lens_dir,
        lens_kind=getattr(args, "lens_kind", "completion"),
        cluster_on=getattr(args, "cluster_on", "individual"),
    )
    names = pd.read_csv(args.names) if args.names else None
    features = None
    if names is not None and args.fidelity_only and "fidelity_pass" in names.columns:
        features = (
            names.loc[names["fidelity_pass"].map(annotation_flag), "feature_id"]
            .astype(int)
            .tolist()
        )
    decoder = None
    if not args.no_decoder:
        checkpoint = Path(args.lens_dir) / "sae_model.pt"
        if checkpoint.exists():
            try:
                decoder = load_decoder_directions(args.lens_dir)
            except ImportError as exc:
                print(f"warning: {exc}; continuing without decoder relationships",
                      file=sys.stderr)
        else:
            print(
                f"warning: {checkpoint} is absent; continuing without decoder relationships",
                file=sys.stderr,
            )
    relations = feature_relationships(
        z,
        names=names,
        decoder=decoder,
        features=features,
        pole=args.cofire_pole,
        min_cooccur=args.min_cooccur,
        min_jaccard=args.min_jaccard,
        min_containment=args.min_containment,
        min_phi=args.min_phi,
        min_lift=args.min_lift,
        min_name_similarity=args.min_name_similarity,
        min_decoder_cosine=args.min_decoder_cosine,
    )
    summary = feature_relationship_summary(relations)
    _save(relations, args.out)
    out_path = Path(args.out)
    summary_path = out_path.with_name(f"{out_path.stem}_summary.csv")
    _save(summary, summary_path)

    n_features = len(features) if features is not None else z.shape[1]
    print(f"\n{n_features} features -> {len(relations)} candidate relationships")
    for row in summary.itertuples(index=False):
        print(f"  {row.relation}: {row.n_pairs}")
    collisions = relations[relations["needs_relabel"]].head(12)
    if len(collisions):
        print("\nTop same-name pairs needing contrastive relabelling:")
        for row in collisions.itertuples(index=False):
            print(
                f"  {row.feature_a} / {row.feature_b}: {row.concept_a!r} / "
                f"{row.concept_b!r}  J={row.jaccard:.3f}, "
                f"containment={row.containment_a_in_b:.2f}/{row.containment_b_in_a:.2f}"
            )
    return 0


def _cmd_cluster_features(args) -> int:
    import pandas as pd

    from prefscope.pipeline.cluster import (
        cluster_run_diagnostics,
        load_cofiring_codes,
        summarize_clusters,
    )

    # Cluster on co-firing in INDIVIDUAL responses (z_a/z_b stacked) — semantic
    # co-occurrence, à la Anatomy — NOT on the difference. In z_diff, antonym features
    # (e.g. "refuses" and "elaborates") co-fire by construction on the same battle, so
    # MI clustering on z_diff merges opposites; clustering on individual codes doesn't.
    z = load_cofiring_codes(
        args.lens_dir,
        lens_kind=getattr(args, "lens_kind", "completion"),
        cluster_on=getattr(args, "cluster_on", "difference"),
    )
    names = pd.read_csv(args.names) if args.names else None
    feats = None
    if names is not None and args.fidelity_only and "fidelity_pass" in names.columns:
        feats = (
            names.loc[names["fidelity_pass"].map(annotation_flag), "feature_id"]
            .astype(int)
            .tolist()
        )

    from prefscope.core import registry

    clusterer = registry.make(
        "clusterer",
        args.method,
        n_clusters=args.n_clusters,
        resolution=args.resolution,
        knn=args.knn,
        min_cluster_size=args.min_cluster_size,
        affinity_metric=getattr(args, "affinity_metric", "phi"),
        pole=getattr(args, "cofire_pole", "positive"),
        min_cooccur=getattr(args, "min_cooccur", 30),
        knn_mode=getattr(args, "knn_mode", "mutual"),
        small_community_policy=getattr(args, "small_community_policy", "preserve"),
        stability_runs=getattr(args, "stability_runs", 5),
        super_resolution=getattr(args, "super_resolution", None),
        super_knn=getattr(args, "super_knn", 4),
    )
    clusters = clusterer.cluster(z, features=feats)
    summary = summarize_clusters(clusters, names=names)  # uses raw clusters + names
    diagnostics = cluster_run_diagnostics(clusters)

    usage_client = usage_path = None
    if args.name_clusters:
        from prefscope.pipeline.cluster import name_clusters

        usage_client, usage_path = _tracked_client(args, "cluster")
        labels = name_clusters(summary, usage_client, concurrency=args.concurrency)
        mapped_labels = summary["cluster_id"].map(labels)
        has_label = mapped_labels.notna() & mapped_labels.astype(str).str.strip().ne("")
        summary["behavior"] = mapped_labels.where(has_label, summary["behavior"])

    out = clusters.copy()
    if names is not None and "concept" in names.columns:
        out = out.merge(names[["feature_id", "concept"]], on="feature_id", how="left")
    out = out.merge(summary[["cluster_id", "behavior"]], on="cluster_id", how="left")
    _save(out, args.out)
    _save(summary, str(args.out).replace(".csv", "_summary.csv"))
    _save(diagnostics, str(args.out).replace(".csv", "_diagnostics.csv"))
    if usage_client is not None:
        _write_usage(usage_client, usage_path)

    print(
        f"\n{len(clusters)} features -> {clusters['cluster_id'].nunique()} behaviors "
        f"({args.method})\n"
    )
    for _, r in summary.iterrows():
        vtag = (
            f"  [{r['n_verified']} verified]" if r.get("n_verified") is not None else ""
        )
        print(
            f"  behavior {r['cluster_id']}  ({r['n_features']} feats{vtag}): {r['behavior']}"
        )
        if r["member_concepts"]:
            print(f"      {r['member_concepts']}")
    ari = diagnostics.iloc[0].get("seed_ari_mean") if len(diagnostics) else None
    stable = f"; seed ARI={ari:.3f}" if pd.notna(ari) else ""
    print(f"\nwrote {args.out} (+ _summary.csv, _diagnostics.csv){stable}")
    return 0
