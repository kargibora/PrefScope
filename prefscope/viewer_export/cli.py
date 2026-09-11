"""Command-line entry point for the static Viewer bundle export."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from prefscope import load_feature_batch, load_feature_catalog
from prefscope.viewer_export.export import export_viewer_bundle


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".json", ".jsonl"}:
        return pd.read_json(path, lines=suffix == ".jsonl")
    raise ValueError(f"unsupported table format: {path}")


def _tables(values) -> dict[str, pd.DataFrame]:
    result = {}
    for value in values or ():
        if "=" not in value:
            raise ValueError("--table expects NAME=PATH")
        name, raw_path = value.split("=", 1)
        if not name or name in result:
            raise ValueError("--table names must be non-empty and unique")
        result[name] = _read_table(Path(raw_path))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prefscope-export-viewer",
        description="Package existing feature data with a built static Viewer.",
    )
    parser.add_argument(
        "--features", required=True, help="saved FeatureBatch directory"
    )
    parser.add_argument("--catalog", default=None, help="optional FeatureCatalog JSON")
    parser.add_argument(
        "--prompt-features",
        default=None,
        help="optional aligned prompt FeatureBatch directory",
    )
    parser.add_argument(
        "--prompt-catalog", default=None, help="optional prompt FeatureCatalog JSON"
    )
    parser.add_argument(
        "--table",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="optional caller-computed CSV, Parquet, JSON, or JSONL table",
    )
    parser.add_argument(
        "--viewer-dist",
        required=True,
        help="built Viewer asset directory containing viewer-build.json",
    )
    parser.add_argument("--out", required=True, help="new output bundle directory")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    features = load_feature_batch(args.features)
    catalog = load_feature_catalog(args.catalog) if args.catalog else None
    export_viewer_bundle(
        features,
        args.out,
        viewer_dist=args.viewer_dist,
        catalog=catalog,
        tables=_tables(args.table),
        prompt_features=(
            load_feature_batch(args.prompt_features) if args.prompt_features else None
        ),
        prompt_catalog=(
            load_feature_catalog(args.prompt_catalog) if args.prompt_catalog else None
        ),
    )
    return 0


__all__ = ["build_parser", "main"]
