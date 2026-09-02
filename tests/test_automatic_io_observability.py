from __future__ import annotations

import json

import pytest

from prefscope.api import analysis_io as analysis_io_module
from prefscope.api import encoded as encoded_module
from prefscope.api.analysis_io import load_analysis_result, save_analysis_result
from prefscope.api.encoded import load_feature_batch, save_feature_batch
from prefscope.observability import observe_run
from prefscope.observability import runtime as runtime_module
from prefscope.reporting import io as reporting_io_module
from prefscope.reporting.io import (
    json_payload,
    load_report_bundle,
    write_report_bundle,
)
from tests.test_analysis_result_io import _result
from tests.test_encoded import _batch
from tests.test_reporting_bundle_io import _artifact, _manifest, _policy


def _events(path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_durable_io_operations_emit_bounded_safe_summaries(tmp_path) -> None:
    events_path = tmp_path / "events.jsonl"
    feature_path = tmp_path / "private-feature-directory"
    analysis_path = tmp_path / "private-analysis-directory"
    report_path = tmp_path / "private-report-directory"
    analysis = _result()
    policy = _policy()
    payload = json_payload({"score": 0.5}, privacy_policy=policy)
    artifact = _artifact(
        payload,
        artifact_id="private_artifact_id",
        path="private/payload.json",
    )
    manifest = _manifest((artifact,), policy)

    with observe_run(events_path, durable=False):
        save_feature_batch(_batch(), feature_path)
        load_feature_batch(feature_path, arrays=("z_a",))
        save_analysis_result(analysis, analysis_path)
        load_analysis_result(analysis_path, dataset=analysis.dataset)
        write_report_bundle(
            report_path,
            manifest,
            {artifact.artifact_id: payload},
        )
        load_report_bundle(report_path)

    rows = _events(events_path)
    expected_stages = [
        "feature_bundle.save",
        "feature_bundle.load",
        "analysis_result.save",
        "analysis_result.load",
        "report_bundle.write",
        "report_bundle.load",
    ]
    assert [(row["stage"], row["status"]) for row in rows] == [
        (stage, status)
        for stage in expected_stages
        for status in ("started", "completed")
    ]

    started = {row["stage"]: row["data"] for row in rows if row["status"] == "started"}
    completed = {
        row["stage"]: row["data"] for row in rows if row["status"] == "completed"
    }
    assert started["feature_bundle.save"]["overwrite"] is False
    assert started["feature_bundle.load"]["requested_view_count"] == 1
    assert "requested_views" not in started["feature_bundle.load"]
    assert "z_a" not in events_path.read_text()
    assert started["analysis_result.load"]["attached"] is True
    assert started["report_bundle.write"]["overwrite"] is False

    assert completed["feature_bundle.load"]["n_rows"] == 2
    assert completed["feature_bundle.load"]["n_features"] == 2
    assert completed["feature_bundle.load"]["n_arrays"] == 1
    assert completed["feature_bundle.load"]["shapes"] == [[2, 2]]
    assert completed["analysis_result.load"]["n_rows"] == 4
    assert completed["analysis_result.load"]["n_groups"] == 3
    assert completed["analysis_result.load"]["artifact_count"] == 2
    assert completed["analysis_result.load"]["shapes"] == [[2, 3], [1, 2]]
    assert completed["report_bundle.load"]["n_rows"] == 5
    assert completed["report_bundle.load"]["n_groups"] == 1
    assert completed["report_bundle.load"]["artifact_count"] == 1
    assert completed["report_bundle.load"]["status"] == "ready"
    assert completed["report_bundle.load"]["profile"] == "shareable"
    assert completed["report_bundle.load"]["feature_view_count"] == 0
    assert completed["report_bundle.load"]["evidence_layer_count"] == 1

    serialized = events_path.read_text()
    for private_value in (
        str(feature_path),
        str(analysis_path),
        str(report_path),
        feature_path.name,
        analysis_path.name,
        report_path.name,
        artifact.artifact_id,
        artifact.path,
        artifact.source_refs[0],
        analysis.dataset.row_ids[0],
    ):
        assert private_value not in serialized


def test_io_failure_reraises_original_exception_without_private_details(
    tmp_path, monkeypatch
) -> None:
    events_path = tmp_path / "events.jsonl"
    error = RuntimeError(
        "private/path/payload.json row_id=private-row group_id=private-group"
    )

    def fail_load(path, *, arrays=None):
        raise error

    monkeypatch.setattr(encoded_module, "_load_feature_batch", fail_load)
    with observe_run(events_path, durable=False):
        with pytest.raises(RuntimeError) as raised:
            load_feature_batch(tmp_path / "private-input")

    assert raised.value is error
    rows = _events(events_path)
    assert [row["status"] for row in rows] == ["started", "failed"]
    assert rows[1]["stage"] == "feature_bundle.load"
    assert rows[1]["data"]["error_type"] == "RuntimeError"
    assert "error_message" not in rows[1]["data"]
    assert rows[1]["message"] == "feature_bundle.load failed"
    serialized = events_path.read_text()
    for private_value in (
        "private/path/payload.json",
        "private-row",
        "private-group",
        "private-input",
    ):
        assert private_value not in serialized


def test_durable_io_is_unchanged_without_a_recorder(tmp_path, monkeypatch) -> None:
    runtime_module._close_environment_recorder()
    monkeypatch.delenv("PREFSCOPE_EVENTS_PATH", raising=False)
    try:
        feature_path = tmp_path / "feature"
        saved = save_feature_batch(_batch(), feature_path)
        loaded = load_feature_batch(saved, arrays=("z_a",))
        assert loaded.array("z_a").shape == (2, 2)
        assert runtime_module._ENVIRONMENT_RUN is None
        assert not (tmp_path / "events.jsonl").exists()
    finally:
        runtime_module._close_environment_recorder()


def test_completion_metadata_is_lazy_and_best_effort(
    tmp_path, monkeypatch
) -> None:
    runtime_module._close_environment_recorder()
    monkeypatch.delenv("PREFSCOPE_EVENTS_PATH", raising=False)
    calls: list[str] = []

    def fail_observation(name, error):
        def fail(value):
            calls.append(name)
            raise error

        return fail

    monkeypatch.setattr(
        encoded_module,
        "_feature_batch_observation",
        fail_observation(
            "feature", KeyboardInterrupt("metadata must not affect publication")
        ),
    )
    monkeypatch.setattr(
        analysis_io_module,
        "_analysis_result_observation",
        fail_observation(
            "analysis", SystemExit("metadata must not affect publication")
        ),
    )
    monkeypatch.setattr(
        reporting_io_module,
        "_report_bundle_observation",
        fail_observation(
            "report", KeyboardInterrupt("metadata must not affect publication")
        ),
    )

    analysis = _result()
    policy = _policy()
    payload = json_payload({"score": 0.5}, privacy_policy=policy)
    artifact = _artifact(payload)
    manifest = _manifest((artifact,), policy)

    assert save_feature_batch(_batch(), tmp_path / "disabled-feature").is_dir()
    assert save_analysis_result(
        analysis, tmp_path / "disabled-analysis"
    ).is_dir()
    assert write_report_bundle(
        tmp_path / "disabled-report",
        manifest,
        {artifact.artifact_id: payload},
    ).root.is_dir()
    assert calls == []

    events_path = tmp_path / "events.jsonl"
    with observe_run(events_path, durable=False):
        assert save_feature_batch(
            _batch(), tmp_path / "active-feature"
        ).is_dir()
        assert save_analysis_result(
            analysis, tmp_path / "active-analysis"
        ).is_dir()
        assert write_report_bundle(
            tmp_path / "active-report",
            manifest,
            {artifact.artifact_id: payload},
        ).root.is_dir()

    assert calls == ["feature", "analysis", "report"]
    completed = [
        row for row in _events(events_path) if row["status"] == "completed"
    ]
    assert [row["stage"] for row in completed] == [
        "feature_bundle.save",
        "analysis_result.save",
        "report_bundle.write",
    ]
    assert all("n_rows" not in row["data"] for row in completed)
