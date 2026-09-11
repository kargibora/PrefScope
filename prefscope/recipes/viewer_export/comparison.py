"""Export a generic paired-response comparison for the external visualization clients."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .sanitize import _label_or_id, _round


def _normalize_comparison_labels(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for label_column, id_column, prefix in (
        ("concept", "feature_id", "feature"),
        ("region_concept", "region_id", "prompt feature"),
    ):
        if {label_column, id_column} <= set(frame.columns):
            frame[label_column] = [
                _label_or_id(value, feature_id, prefix=prefix)
                for value, feature_id in zip(frame[label_column], frame[id_column])
            ]
    return frame


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
    overall = _normalize_comparison_labels(pd.read_parquet(scope_path))
    conditional_path = directory / "concept_shift_by_context.parquet"
    examples_path = directory / "paired_examples.parquet"
    conditional = _normalize_comparison_labels(
        pd.read_parquet(conditional_path)
        if conditional_path.exists()
        else pd.DataFrame()
    )
    examples = _normalize_comparison_labels(
        pd.read_parquet(examples_path)
        if examples_path.exists()
        else pd.DataFrame()
    )
    return {
        "meta": manifest,
        "concepts": _round(overall),
        "contexts": _round(conditional),
        "examples": _round(examples),
    }
