import json

import pandas as pd
import pytest

from prefscope.viewer_export.comparison import export_paired_comparison


def test_export_paired_comparison_combines_durable_artifacts(tmp_path):
    (tmp_path / "comparison.json").write_text(json.dumps({
        "schema_version": 1,
        "side_a_name": "base",
        "side_b_name": "tuned",
        "preference_labels_used": False,
    }))
    pd.DataFrame({
        "feature_id": [7],
        "concept": ["uses headings"],
        "delta_b_minus_a": [0.1256789],
        "response_scope": ["general_tendency"],
    }).to_parquet(tmp_path / "response_scope.parquet", index=False)
    pd.DataFrame({
        "feature_id": [7], "region_id": [3], "delta_b_minus_a": [0.2],
    }).to_parquet(tmp_path / "concept_shift_by_context.parquet", index=False)
    pd.DataFrame({
        "feature_id": [7], "direction": ["b_only"],
        "prompt": ["Explain this"], "response_a": ["plain"],
        "response_b": ["## Structured"],
    }).to_parquet(tmp_path / "paired_examples.parquet", index=False)

    exported = export_paired_comparison(tmp_path)

    assert exported["meta"]["preference_labels_used"] is False
    assert exported["concepts"][0]["delta_b_minus_a"] == pytest.approx(0.12568)
    assert exported["contexts"][0]["region_id"] == 3
    assert exported["examples"][0]["response_b"] == "## Structured"


def test_export_paired_comparison_rejects_incomplete_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="paired comparison"):
        export_paired_comparison(tmp_path)
