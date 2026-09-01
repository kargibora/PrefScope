from __future__ import annotations

import sys

from prefscope.analysis.presence import annotation_flag

from prefscope.cli.common import (
    _file_digest,
    _interpret_signature,
    _path_identity,
    _requested_features,
    _reset_usage,
    _tracked_client,
    _write_usage,
)


def _cmd_name_prompts(args) -> int:
    """Compatibility alias for ``interpret name --lens-kind prompt``.

    Keeping one implementation is important here: prompt naming must have the same
    per-feature checkpoint, resume validation, and usage ledger as completion naming.
    """
    args.annotations = None
    args.name_mode = "single-text"
    args.lens_kind = "prompt"
    args.abbreviate = False
    return _cmd_interpret_name(args)


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
    from prefscope.interpret.checkpoint import FeatureCheckpoint
    from prefscope.interpret.strategy import LensCodes, resolve_name_mode

    lens_kind = getattr(args, "lens_kind", "completion")
    if lens_kind == "prompt" and not args.corpus:
        print(
            "prompt-lens naming needs --corpus (to fetch prompt text)", file=sys.stderr
        )
        return 2
    codes = LensCodes.load(
        args.lens_dir, args.annotations, corpus=args.corpus, lens_kind=lens_kind
    )
    mode = resolve_name_mode(args.name_mode, codes.input_rep, lens_kind)
    src = (
        f"--lens-kind {lens_kind}"
        if lens_kind == "prompt"
        else (
            f"auto, from lens input_rep={codes.input_rep!r}"
            if mode != args.name_mode
            else "explicit"
        )
    )
    print(f"naming mode: {mode} ({src})")
    requested = _requested_features(codes, args.features)
    signature = _interpret_signature(
        args,
        "name",
        name_mode=mode,
        n_active=args.n_active,
        n_zero=args.n_zero,
        negatives=args.negatives,
        abbreviate=args.abbreviate,
        pole=args.pole,
        lens_kind=lens_kind,
    )
    try:
        checkpoint = FeatureCheckpoint(args.out, signature, fresh=args.fresh)
    except ValueError as exc:
        print(f"resume error: {exc}", file=sys.stderr)
        return 2
    completed = checkpoint.completed_ids
    remaining = [f for f in requested if f not in completed]
    print(
        f"resume: {len(requested) - len(remaining)}/{len(requested)} requested features "
        f"already complete; {len(remaining)} remaining"
    )
    if not remaining:
        print(f"nothing to do; checkpoint is complete: {args.out}")
        return 0
    if args.fresh:
        _reset_usage(args.out)
    client, usage_path = _tracked_client(args, "name", resume=not args.fresh)

    def save_result(row) -> None:
        checkpoint.record(row)
        write = getattr(client, "write_usage", None)
        if callable(write):
            write(usage_path)

    strategy = registry.make(
        "interpreter",
        mode,
        features=remaining,
        n_active=args.n_active,
        n_zero=args.n_zero,
        verify_frac=args.verify_frac,
        seed=args.seed,
        abbreviate=args.abbreviate,
        concurrency=args.concurrency,
        debug_dir=args.debug_responses,
        negatives=args.negatives,
        pole=args.pole,
        on_result=save_result,
    )
    df = strategy.name(codes, client)
    checkpoint.merge(df)
    _write_usage(client, usage_path)
    total = len(checkpoint.completed_ids.intersection(requested))
    print(f"wrote {total}/{len(requested)} requested feature names to {args.out}")
    return 0


def _cmd_interpret_verify(args) -> int:
    import pandas as pd

    from prefscope.core import registry
    from prefscope.interpret.checkpoint import FeatureCheckpoint
    from prefscope.interpret.strategy import VerifyCodes, resolve_verify_mode

    lens_kind = getattr(args, "lens_kind", "completion")
    if lens_kind == "prompt" and not args.corpus:
        print(
            "prompt-lens verify needs --corpus (to fetch prompt text)", file=sys.stderr
        )
        return 2
    codes = VerifyCodes.load(
        args.lens_dir, args.annotations, corpus=args.corpus, lens_kind=lens_kind
    )
    mode = resolve_verify_mode(args.verify_mode, codes.input_rep, lens_kind)
    src = (
        f"auto, from lens input_rep={codes.input_rep!r}"
        if mode != args.verify_mode and lens_kind != "prompt"
        else (f"--lens-kind {lens_kind}" if lens_kind == "prompt" else "explicit")
    )
    print(f"verify mode: {mode} ({src})")
    names = pd.read_csv(args.names)
    if "feature_id" not in names.columns:
        print("names CSV has no feature_id column", file=sys.stderr)
        return 2
    available = list(dict.fromkeys(names["feature_id"].astype(int).tolist()))
    if args.features is None:
        requested = available
    else:
        requested = list(dict.fromkeys(int(f) for f in args.features))
        missing = sorted(set(requested).difference(available))
        if missing:
            print(
                f"requested feature IDs are absent from --names: {missing}",
                file=sys.stderr,
            )
            return 2
        names = names[names["feature_id"].astype(int).isin(requested)].copy()
    signature = _interpret_signature(
        args,
        "verify",
        verify_mode=mode,
        lens_kind=lens_kind,
        names_sha256=_file_digest(args.names),
        n_per_bucket=args.n_per_bucket,
        fidelity_threshold=args.fidelity_threshold,
        negatives=args.negatives,
        embeddings=_path_identity(args.embeddings),
        pole=args.pole,
        sampling=args.sampling,
        n_examples=args.n_examples,
        min_success_rate=args.min_success_rate,
        min_bucket=args.min_bucket,
        features=requested,
    )
    try:
        checkpoint = FeatureCheckpoint(args.out, signature, fresh=args.fresh)
    except ValueError as exc:
        print(f"resume error: {exc}", file=sys.stderr)
        return 2
    completed = checkpoint.completed_ids
    remaining = [f for f in requested if f not in completed]
    print(
        f"resume: {len(requested) - len(remaining)}/{len(requested)} requested features "
        f"already complete; {len(remaining)} remaining"
    )
    if not remaining:
        final = checkpoint.frame()
        passed = int(final["fidelity_pass"].map(annotation_flag).sum())
        print(
            f"nothing to do; checkpoint is complete ({passed}/{len(final)} pass): {args.out}"
        )
        return 0
    if args.fresh:
        _reset_usage(args.out)
    client, usage_path = _tracked_client(args, "verify", resume=not args.fresh)

    def save_result(row) -> None:
        checkpoint.record(row)
        write = getattr(client, "write_usage", None)
        if callable(write):
            write(usage_path)

    strategy = registry.make(
        "verifier",
        mode,
        n_per_bucket=args.n_per_bucket,
        verify_frac=args.verify_frac,
        seed=args.seed,
        fidelity_threshold=args.fidelity_threshold,
        concurrency=args.concurrency,
        negatives=getattr(args, "negatives", "random"),
        embeddings=getattr(args, "embeddings", None),
        features=remaining,
        pole=args.pole,
        sampling=args.sampling,
        n_examples=args.n_examples,
        min_success_rate=args.min_success_rate,
        min_bucket=args.min_bucket,
        on_result=save_result,
    )
    df = strategy.verify(codes, names, client)
    checkpoint.merge(df)
    _write_usage(client, usage_path)
    final = checkpoint.frame()
    print(
        f"wrote {len(final)} fidelity rows "
        f"({int(final['fidelity_pass'].map(annotation_flag).sum())} pass) to {args.out}"
    )
    return 0


def _cmd_interpret_calibrate_presence(args) -> int:
    """Learn per-feature semantic presence thresholds from ordinary activations."""
    import pandas as pd

    from prefscope.interpret.calibrate import calibrate_single_text_features
    from prefscope.interpret.checkpoint import FeatureCheckpoint
    from prefscope.interpret.strategy import VerifyCodes

    lens_kind = args.lens_kind
    if lens_kind == "prompt" and not args.corpus:
        print("prompt-lens calibration needs --corpus", file=sys.stderr)
        return 2
    codes = VerifyCodes.load(
        args.lens_dir, args.annotations, corpus=args.corpus, lens_kind=lens_kind
    )
    if lens_kind == "completion" and (
        codes.input_rep != "individual" or codes.z_a is None
    ):
        print(
            "presence calibration needs an individual completion lens with z_a/z_b",
            file=sys.stderr,
        )
        return 2
    if codes.activation_polarity == "signed" and args.pole != "positive":
        print(
            "signed lenses require --pole positive for presence calibration",
            file=sys.stderr,
        )
        return 2
    names = pd.read_csv(args.names)
    if "feature_id" not in names.columns or "concept" not in names.columns:
        print("--names must contain feature_id and concept", file=sys.stderr)
        return 2
    if args.fidelity:
        fidelity = pd.read_csv(args.fidelity)
        if (
            "feature_id" not in fidelity.columns
            or "fidelity_pass" not in fidelity.columns
        ):
            print(
                "--fidelity must contain feature_id and fidelity_pass", file=sys.stderr
            )
            return 2
        keep = fidelity[["feature_id", "fidelity_pass"]].copy()
        keep["feature_id"] = keep["feature_id"].astype(int)
        names["feature_id"] = names["feature_id"].astype(int)
        names = names.merge(
            keep, on="feature_id", how="left", suffixes=("", "_verified")
        )
        if not args.all_named:
            pass_col = (
                "fidelity_pass_verified"
                if "fidelity_pass_verified" in names.columns
                else "fidelity_pass"
            )
            names = names[names[pass_col].map(annotation_flag)].copy()
    elif not args.all_named:
        print("provide --fidelity, or explicitly pass --all-named", file=sys.stderr)
        return 2
    available = list(dict.fromkeys(names["feature_id"].astype(int).tolist()))
    requested = (
        available
        if args.features is None
        else list(dict.fromkeys(int(f) for f in args.features))
    )
    missing = sorted(set(requested).difference(available))
    if missing:
        print(
            f"requested feature IDs are unavailable after fidelity filtering: {missing}",
            file=sys.stderr,
        )
        return 2
    names = names[names["feature_id"].astype(int).isin(requested)].copy()
    signature = _interpret_signature(
        args,
        "calibrate-presence",
        lens_kind=lens_kind,
        names_sha256=_file_digest(args.names),
        fidelity_sha256=_file_digest(args.fidelity) if args.fidelity else None,
        all_named=args.all_named,
        features=requested,
        pole=args.pole,
        n_per_bin=args.n_per_bin,
        n_top=args.n_top,
        n_zero=args.n_zero,
        batch_size=args.batch_size,
        target_precision=args.target_precision,
        min_above=args.min_above,
        max_silent_rate=args.max_silent_rate,
    )
    try:
        checkpoint = FeatureCheckpoint(args.out, signature, fresh=args.fresh)
    except ValueError as exc:
        print(f"resume error: {exc}", file=sys.stderr)
        return 2
    remaining = [f for f in requested if f not in checkpoint.completed_ids]
    print(
        f"resume: {len(requested) - len(remaining)}/{len(requested)} calibrated; "
        f"{len(remaining)} remaining"
    )
    if not remaining:
        final = checkpoint.frame()
        passed = int(final["presence_pass"].map(annotation_flag).sum())
        print(
            f"nothing to do; calibration complete ({passed}/{len(final)} usable): {args.out}"
        )
        return 0
    if args.fresh:
        _reset_usage(args.out)
    client, usage_path = _tracked_client(
        args, "calibrate-presence", resume=not args.fresh
    )

    def save_result(row) -> None:
        checkpoint.record(row)
        write = getattr(client, "write_usage", None)
        if callable(write):
            write(usage_path)

    if lens_kind == "prompt":
        texts, contexts, z = codes.prompts, None, codes.z_prompt
        ids = codes.instruction_ids
    else:
        paired = codes.z_b is not None
        texts = codes.battles["completion_a"].astype(str).tolist()
        if paired:
            texts += codes.battles["completion_b"].astype(str).tolist()
        contexts = codes.battles["prompt"].astype(str).tolist() * (2 if paired else 1)
        z = (codes.z_a, codes.z_b) if paired else codes.z_a
        ids = list(codes.instruction_ids) * (2 if paired else 1)
    frame = calibrate_single_text_features(
        texts,
        z,
        names,
        client,
        contexts=contexts,
        instruction_ids=ids,
        features=remaining,
        n_per_bin=args.n_per_bin,
        n_top=args.n_top,
        n_zero=args.n_zero,
        batch_size=args.batch_size,
        target_precision=args.target_precision,
        min_above=args.min_above,
        max_silent_rate=args.max_silent_rate,
        seed=args.seed,
        concurrency=args.concurrency,
        on_result=save_result,
    )
    checkpoint.merge(frame)
    _write_usage(client, usage_path)
    final = checkpoint.frame()
    passed = int(final["presence_pass"].map(annotation_flag).sum())
    print(f"wrote {len(final)} semantic calibrations ({passed} usable) to {args.out}")
    return 0


def _cmd_interpret_classify_role(args) -> int:
    """Classify named response properties without conflating role and linkage."""
    import pandas as pd

    from prefscope.interpret.checkpoint import FeatureCheckpoint
    from prefscope.interpret.role import classify_response_roles
    from prefscope.interpret.strategy import VerifyCodes

    codes = VerifyCodes.load(
        args.lens_dir, args.annotations, corpus=args.corpus, lens_kind="completion"
    )
    if codes.input_rep != "individual" or codes.z_a is None:
        print(
            "role classification needs an individual completion lens",
            file=sys.stderr,
        )
        return 2
    if codes.activation_polarity == "signed" and args.pole != "positive":
        print(
            "signed lenses require --pole positive for role classification",
            file=sys.stderr,
        )
        return 2
    names = pd.read_csv(args.names)
    if not {"feature_id", "concept"} <= set(names.columns):
        print("--names must contain feature_id and concept", file=sys.stderr)
        return 2
    names["feature_id"] = names["feature_id"].astype(int)
    if not args.all_named:
        if "fidelity_pass" not in names.columns:
            print(
                "--names has no fidelity_pass column; provide feature_fidelity.csv or "
                "explicitly pass --all-named",
                file=sys.stderr,
            )
            return 2
        names = names[names["fidelity_pass"].map(annotation_flag)].copy()
    available = list(dict.fromkeys(names["feature_id"].astype(int).tolist()))
    requested = (
        available
        if args.features is None
        else list(dict.fromkeys(int(feature_id) for feature_id in args.features))
    )
    missing = sorted(set(requested).difference(available))
    if missing:
        print(
            f"requested feature IDs are unavailable after fidelity filtering: {missing}",
            file=sys.stderr,
        )
        return 2
    names = names[names["feature_id"].isin(requested)].copy()
    linkage = pd.read_csv(args.linkage) if args.linkage else None
    signature = _interpret_signature(
        args,
        "classify-role",
        names_sha256=_file_digest(args.names),
        linkage_sha256=_file_digest(args.linkage) if args.linkage else None,
        features=requested,
        all_named=args.all_named,
        pole=args.pole,
        n_top=args.n_top,
        n_random=args.n_random,
        batch_size=args.batch_size,
        min_valid_examples=args.min_valid_examples,
    )
    try:
        checkpoint = FeatureCheckpoint(args.out, signature, fresh=args.fresh)
    except ValueError as exc:
        print(f"resume error: {exc}", file=sys.stderr)
        return 2
    remaining = [feature_id for feature_id in requested
                 if feature_id not in checkpoint.completed_ids]
    print(
        f"resume: {len(requested) - len(remaining)}/{len(requested)} classified; "
        f"{len(remaining)} remaining"
    )
    if not remaining:
        final = checkpoint.frame()
        print(f"nothing to do; role classification complete: {args.out} ({len(final)} rows)")
        return 0
    if args.fresh:
        _reset_usage(args.out)
    client, usage_path = _tracked_client(args, "classify-role", resume=not args.fresh)

    def save_result(row) -> None:
        checkpoint.record(row)
        write = getattr(client, "write_usage", None)
        if callable(write):
            write(usage_path)

    frame = classify_response_roles(
        codes.battles,
        codes.z_a,
        codes.z_b,
        names,
        client,
        instruction_ids=codes.instruction_ids,
        features=remaining,
        linkage=linkage,
        n_top=args.n_top,
        n_random=args.n_random,
        batch_size=args.batch_size,
        min_valid_examples=args.min_valid_examples,
        seed=args.seed,
        concurrency=args.concurrency,
        on_result=save_result,
    )
    checkpoint.merge(frame)
    _write_usage(client, usage_path)
    final = checkpoint.frame()
    counts = final.get("behavior_scope", pd.Series(dtype=str)).value_counts().to_dict()
    print(f"wrote {len(final)} feature-role rows to {args.out}: {counts}")
    return 0
