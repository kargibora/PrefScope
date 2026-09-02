from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from prefscope.api.analysis_contracts import AnalysisArtifact, AnalysisComponent, OutcomeSpec
from prefscope.api import analysis_execution as analysis_execution_module
from prefscope.api.analysis_execution import AnalysisPlan, analyze_dataset
from prefscope.api.encoded import save_feature_batch
from prefscope.core.features import FeatureBatch, FeatureMatrix
from prefscope.observability import observe_run
from prefscope.observability import runtime as runtime_module
from prefscope.reporting import source as source_module
from prefscope.reporting.source import FeatureBundleReader


def _events(path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.fixture(autouse=True)
def _clean_automatic_runtime(monkeypatch):
    runtime_module._close_environment_recorder()
    monkeypatch.delenv("PREFSCOPE_EVENTS_PATH", raising=False)
    yield
    runtime_module._close_environment_recorder()


def _features() -> FeatureMatrix:
    return FeatureMatrix(
        np.arange(12, dtype=np.float32).reshape(6, 2),
        row_ids=tuple(f"private-row-{index}" for index in range(6)),
        role="private-response-content",
        provenance={"note": "private-prompt-text"},
    )


def _bundle(tmp_path):
    batch = FeatureBatch(
        row_ids=("private-row-a", "private-row-b", "private-row-c"),
        arrays={
            "private_secret_view": np.arange(6, dtype=np.float32).reshape(3, 2),
            "presence": np.array(
                [[True, False], [False, True], [True, True]], dtype=bool
            ),
        },
        roles={
            "private_secret_view": "response",
            "presence": "semantic_presence",
        },
        orientations={"private_secret_view": "none", "presence": "none"},
        metadata={
            "group_id": ("private-group-a", "private-group-a", "private-group-b"),
            "prompt": ("private prompt one", "private prompt two", "private prompt three"),
        },
        activation_polarity="nonnegative",
        code_semantics="numerical_activity",
        provenance={"note": "private-source-content"},
    )
    return save_feature_batch(batch, tmp_path / "private-source-path")


def test_analyze_dataset_records_only_safe_completion_counts(tmp_path):
    feature_matrix = _features()
    outcomes = OutcomeSpec(
        [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        row_ids=feature_matrix.row_ids,
        kind="binary",
    )
    path = tmp_path / "events.jsonl"

    with observe_run(path, durable=False):
        result = analyze_dataset(
            {"private-config-name": feature_matrix},
            {"private-spec-name": outcomes},
            group_ids=("private-group-a",) * 3 + ("private-group-b",) * 3,
        )

    assert result.dataset.n_rows == 6
    rows = [row for row in _events(path) if row["stage"] == "analyze_dataset"]
    assert [row["status"] for row in rows] == ["started", "completed"]
    assert set(rows[0]["data"]) == {"operation_id", "parent_operation_id"}
    completed = rows[1]["data"]
    assert completed["n_rows"] == 6
    assert completed["n_features"] == 2
    assert completed["n_groups"] == 2
    assert completed["n_views"] == 1
    assert completed["artifact_count"] == 1

    text = path.read_text()
    for private_value in (
        "private-row",
        "private-group",
        "private-config-name",
        "private-spec-name",
        "private-response-content",
        "private-prompt-text",
    ):
        assert private_value not in text


def test_feature_bundle_open_records_safe_shape_and_view_metadata(tmp_path):
    root = _bundle(tmp_path)
    path = tmp_path / "events.jsonl"

    with observe_run(path, durable=False):
        source = FeatureBundleReader.open(root)

    assert source.n_rows == 3
    rows = [row for row in _events(path) if row["stage"] == "load_feature_source"]
    assert [row["status"] for row in rows] == ["started", "completed"]
    assert set(rows[0]["data"]) == {"operation_id", "parent_operation_id"}
    completed = rows[1]["data"]
    assert set(completed) == {
        "n_rows",
        "n_features",
        "n_views",
        "artifact_count",
        "operation_id",
        "parent_operation_id",
        "duration_seconds",
    }
    assert completed["n_rows"] == 3
    assert completed["n_features"] == 2
    assert completed["n_views"] == 2
    assert completed["artifact_count"] == 5

    text = path.read_text()
    for private_value in (
        "private-source-path",
        "private_secret_view",
        "private-row",
        "private-group",
        "private prompt",
        "private-source-content",
    ):
        assert private_value not in text


class _FailingAnalysis(AnalysisComponent):
    name = "failing_analysis"

    def run(self, dataset):
        del dataset
        raise RuntimeError("private prompt and private-row must not be stored")


def test_analysis_failure_is_sanitized_and_reraised(tmp_path):
    path = tmp_path / "events.jsonl"

    with pytest.raises(RuntimeError, match="private prompt"):
        with observe_run(path, durable=False):
            analyze_dataset(_features(), plan=AnalysisPlan((_FailingAnalysis(),)))

    rows = [row for row in _events(path) if row["stage"] == "analyze_dataset"]
    assert [row["status"] for row in rows] == ["started", "failed"]
    assert rows[1]["message"] == "analyze_dataset failed"
    assert rows[1]["data"]["error_type"] == "RuntimeError"
    assert "private prompt" not in path.read_text()
    assert "private-row" not in path.read_text()


def test_feature_source_failure_does_not_record_path_or_exception_message(tmp_path):
    path = tmp_path / "events.jsonl"
    missing = tmp_path / "private-missing-source"

    with pytest.raises(FileNotFoundError, match="private-missing-source"):
        with observe_run(path, durable=False):
            FeatureBundleReader.open(missing)

    rows = [row for row in _events(path) if row["stage"] == "load_feature_source"]
    assert [row["status"] for row in rows] == ["started", "failed"]
    assert rows[1]["message"] == "load_feature_source failed"
    assert rows[1]["data"]["error_type"] == "FileNotFoundError"
    assert "private-missing-source" not in path.read_text()


def test_completion_metadata_keyboard_interrupt_does_not_change_successful_operations(
    tmp_path, monkeypatch
):
    root = _bundle(tmp_path)
    path = tmp_path / "events.jsonl"

    def fail_metadata(_):
        raise KeyboardInterrupt("private completion metadata interruption")

    monkeypatch.setattr(
        analysis_execution_module, "_analysis_completion_data", fail_metadata
    )
    monkeypatch.setattr(
        source_module, "_feature_source_completion_data", fail_metadata
    )

    with observe_run(path, durable=False):
        result = analyze_dataset(
            _features(), plan=AnalysisPlan((_SuccessfulAnalysis(),))
        )
        source = FeatureBundleReader.open(root)

    assert result.artifact("successful_analysis").table.iloc[0]["value"] == 1
    assert source.n_rows == 3
    rows = _events(path)
    assert [(row["stage"], row["status"]) for row in rows] == [
        ("analyze_dataset", "started"),
        ("analyze_dataset", "completed"),
        ("load_feature_source", "started"),
        ("load_feature_source", "completed"),
    ]
    assert "private completion metadata interruption" not in path.read_text()


def test_operations_remain_noops_without_a_recorder_or_environment(
    tmp_path, monkeypatch
):
    root = _bundle(tmp_path)

    def unexpected_metadata(_):
        pytest.fail("disabled spans must not derive completion metadata")

    monkeypatch.setattr(
        analysis_execution_module, "_analysis_completion_data", unexpected_metadata
    )
    monkeypatch.setattr(
        source_module, "_feature_source_completion_data", unexpected_metadata
    )

    result = analyze_dataset(
        _features(),
        plan=AnalysisPlan((_SuccessfulAnalysis(),)),
    )
    source = FeatureBundleReader.open(root)

    assert result.artifact("successful_analysis").table.iloc[0]["value"] == 1
    assert source.n_rows == 3
    assert runtime_module._ENVIRONMENT_RUN is None


class _SuccessfulAnalysis(AnalysisComponent):
    name = "successful_analysis"

    def run(self, dataset):
        del dataset
        return AnalysisArtifact(
            self.name,
            pd.DataFrame({"value": [1]}),
            "synthetic test result",
        )
