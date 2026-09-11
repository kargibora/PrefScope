from __future__ import annotations

import sys
from pathlib import Path

from prefscope.recipes.analysis.presence import annotation_flag

from prefscope.cli.common import (
    _file_digest,
    _interpret_signature,
    _requested_features,
    _reset_usage,
    _tracked_client,
    _write_usage,
)


def _cmd_interpret_name(args) -> int:
    from prefscope.core import registry
    from prefscope.interpret.checkpoint import FeatureCheckpoint
    from prefscope.interpret.strategy import LensCodes, resolve_name_mode

    lens_kind = getattr(args, "lens_kind", "completion")
    names_filename = (
        "prompt_feature_names.csv"
        if lens_kind == "prompt"
        else "feature_names.csv"
    )
    if Path(args.out).resolve() == (Path(args.lens_dir).resolve() / names_filename):
        print(
            "name output must not overwrite the lens bundle; write to a results "
            "directory, then publish with Lens.save(..., annotations=...)",
            file=sys.stderr,
        )
        return 2
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
