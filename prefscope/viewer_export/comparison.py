"""Export a generic paired-response comparison for the web viewer."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .sanitize import _round


def export_paired_comparison(directory) -> dict | None:
    directory = Path(directory) if directory else None
    if directory is None:
        return None
    manifest_path = directory / "comparison.json"
    scope_path = directory / "response_scope.parquet"
    if not manifest_path.exists() or not scope_path.exists():
        raise FileNotFoundError(
            f"{directory} is not a paired comparison (needs comparison.json and "
            "response_scope.parquet)")
    manifest = json.loads(manifest_path.read_text())
    overall = pd.read_parquet(scope_path)
    conditional_path = directory / "concept_shift_by_context.parquet"
    examples_path = directory / "paired_examples.parquet"
    conditional = (pd.read_parquet(conditional_path)
                   if conditional_path.exists() else pd.DataFrame())
    examples = pd.read_parquet(examples_path) if examples_path.exists() else pd.DataFrame()
    return {
        "meta": manifest,
        "concepts": _round(overall),
        "contexts": _round(conditional),
        "examples": _round(examples),
    }
