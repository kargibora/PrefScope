"""Handlers for high-level, config-driven workflows."""
from __future__ import annotations

from pathlib import Path

import yaml

from prefscope.pipeline.analyze import (
    AnalyzeConfig,
    apply_set_overrides,
    run_analysis,
    set_config_value,
)


def _cmd_analyze(args) -> int:
    config_path = Path(args.config).expanduser().resolve()
    raw = yaml.safe_load(config_path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{config_path}: config root must be a mapping")
    raw = apply_set_overrides(raw, args.set)

    # Named flags are shortcuts for common --set overrides and take final precedence.
    cwd = Path.cwd()
    if args.data is not None:
        source = dict((raw.get("data") or {}).get("source") or {})
        source.pop("dataset_id", None)
        source.update({
            "type": "local", "path": str((cwd / args.data).resolve())
            if not Path(args.data).expanduser().is_absolute()
            else str(Path(args.data).expanduser()),
        })
        set_config_value(raw, "data.source", source)
    elif args.hf_dataset is not None:
        source = dict((raw.get("data") or {}).get("source") or {})
        source.pop("path", None)
        source.update({
            "type": "huggingface", "dataset_id": args.hf_dataset,
        })
        set_config_value(raw, "data.source", source)
    if args.out is not None:
        path = Path(args.out).expanduser()
        set_config_value(raw, "out_dir", str(path if path.is_absolute() else (cwd / path).resolve()))
    if args.repo is not None:
        lenses = dict(raw.get("lenses") or {})
        lenses.pop("completion", None)
        lenses.pop("prompt", None)
        lenses["repo"] = args.repo
        raw["lenses"] = lenses
    for name, value in (
        ("completion_subfolder", args.completion_subfolder),
        ("prompt_subfolder", args.prompt_subfolder),
        ("revision", args.revision),
    ):
        if value is not None:
            set_config_value(raw, f"lenses.{name}", value)
    for name, value in (
        ("completion", args.completion_lens), ("prompt", args.prompt_lens)
    ):
        if value is not None:
            if not str(value).startswith("hf://"):
                path = Path(value).expanduser()
                value = str(path if path.is_absolute() else (cwd / path).resolve())
            set_config_value(raw, f"lenses.{name}", value)
    if args.device is not None:
        set_config_value(raw, "device", args.device)
    if args.presence_policy is not None:
        set_config_value(raw, "concepts.presence_policy", args.presence_policy)
    if args.top_k is not None:
        set_config_value(raw, "concepts.top_k", args.top_k)
    if args.viewer is not None:
        set_config_value(raw, "viewer.enabled", args.viewer)

    cfg = AnalyzeConfig.from_dict(raw, base_dir=config_path.parent)
    run_analysis(cfg, fresh=args.fresh)
    return 0


__all__ = ["_cmd_analyze"]
