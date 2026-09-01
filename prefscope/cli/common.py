from __future__ import annotations

import hashlib
from pathlib import Path

from prefscope.interpret.llm import (
    LLMClient,
    UsageTracker,
)


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


def _usage_paths(out) -> tuple[Path, Path]:
    p = Path(out)
    base = p.with_suffix("") if p.suffix else p
    return (
        base.with_name(base.name + ".usage.json"),
        base.with_name(base.name + ".usage.jsonl"),
    )


def _path_identity(path) -> dict | None:
    """Cheap identity for resume validation without hashing a multi-GB corpus."""
    if path is None:
        return None
    p = Path(path).expanduser().resolve()
    stat = p.stat()
    return {"path": str(p), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _file_digest(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _interpret_implementation_digest() -> str:
    """Detect prompt/interpreter edits so a resume cannot silently mix algorithms."""
    root = Path(__file__).parents[1] / "interpret"
    files = sorted(p for p in root.rglob("*") if p.suffix in (".py", ".txt"))
    h = hashlib.sha256()
    for path in files:
        h.update(str(path.relative_to(root)).encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def _interpret_signature(args, stage: str, **extra) -> dict:
    lens = Path(args.lens_dir).expanduser().resolve()
    manifest = lens / "manifest.json"
    signature = {
        "schema_version": 1,
        "stage": stage,
        "implementation_sha256": _interpret_implementation_digest(),
        "lens_dir": str(lens),
        "lens_manifest_sha256": _file_digest(manifest),
        "corpus": _path_identity(args.corpus),
        "annotations": [_path_identity(p) for p in (args.annotations or [])],
        "backend": args.backend,
        "model": args.model,
        "api_base": args.api_base,
        "max_tokens": args.max_tokens,
        "reasoning_effort": args.reasoning_effort,
        "verify_frac": args.verify_frac,
        "seed": args.seed,
    }
    signature.update(extra)
    return signature


def _requested_features(codes, features) -> list[int]:
    matrix = next(
        (x for x in (codes.z_prompt, codes.z_a, codes.z_diff) if x is not None), None
    )
    if matrix is None:
        raise ValueError("lens has no feature-code matrix")
    n_features = int(matrix.shape[1])
    requested = (
        list(range(n_features)) if features is None else list(dict.fromkeys(features))
    )
    invalid = [int(f) for f in requested if int(f) < 0 or int(f) >= n_features]
    if invalid:
        raise ValueError(f"feature ids outside [0, {n_features - 1}]: {invalid[:8]}")
    return [int(f) for f in requested]


def _reset_usage(out) -> None:
    for path in _usage_paths(out):
        path.unlink(missing_ok=True)


def _make_client(
    args, *, usage_tracker: UsageTracker | None = None, usage_stage: str = "llm"
) -> "LLMClient":
    return LLMClient(
        backend=args.backend,
        model=args.model,
        api_base=args.api_base,
        api_key_env=args.api_key_env,
        max_tokens=getattr(args, "max_tokens", 512),
        reasoning_effort=getattr(args, "reasoning_effort", None),
        usage_tracker=usage_tracker,
        usage_stage=usage_stage,
    )


def _tracked_client(
    args, stage: str, *, resume: bool = False
) -> tuple[LLMClient, Path]:
    summary, events = _usage_paths(args.out)
    tracker = UsageTracker(events, resume=resume)
    return _make_client(args, usage_tracker=tracker, usage_stage=stage), summary


def _write_usage(client: LLMClient, path: Path) -> None:
    write = getattr(client, "write_usage", None)
    progress = getattr(client, "usage_progress", None)
    # Custom/test clients satisfy the strategy's small ``raw`` protocol but do not
    # necessarily implement PrefScope's optional accounting extension.
    if callable(write):
        write(path)
        detail = progress() if callable(progress) else "summary written"
        print(f"LLM usage: {detail} -> {path}")
