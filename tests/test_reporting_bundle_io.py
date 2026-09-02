from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from prefscope.core.table_schema import TableContract
from prefscope.reporting.contracts import (
    ArtifactPrivacy,
    ArtifactStatus,
    CompilerProvenance,
    DatasetLineage,
    EvidenceLayer,
    ReportArtifact,
    ReportCapabilities,
    ReportError,
    ReportManifest,
    ReportLineage,
    ReportMode,
    ReportStatus,
    SamplingProvenance,
    SectionContract,
    SectionKind,
    SectionOrientation,
    SectionStatus,
    SourceArtifactReference,
    StatusReason,
    table_to_json_table,
)
from prefscope.reporting.io import (
    MANIFEST_FILENAME,
    PathPayload,
    artifact_sha256,
    canonical_json_bytes,
    json_payload,
    load_report_bundle,
    write_report_bundle,
)
from prefscope.reporting.privacy import PrivacyPolicy, TextPolicy


def _policy(*, local: bool = False) -> PrivacyPolicy:
    cls = PrivacyPolicy.local if local else PrivacyPolicy.shareable
    return cls(
        allow_fields=("score", "feature_id"),
        id_fields=("row_id",),
        cell_count_fields=("n_rows",),
        categorical_fields={"scope": ("dataset",)},
    )


def _section() -> SectionContract:
    return SectionContract(
        section_id="summary",
        kind=SectionKind.TYPED_TABLE,
        version=1,
        title="Summary",
        evidence_layer=EvidenceLayer.DESCRIPTIVE,
        orientation=SectionOrientation.AS_DECLARED,
        coordinates={"scope": "dataset"},
    )


def _artifact(
    payload: bytes,
    *,
    artifact_id: str = "summary_data",
    path: str = "data/summary.json",
    media_type: str = "application/json",
    table_contract: TableContract | None = None,
    privacy: ArtifactPrivacy = ArtifactPrivacy.AGGREGATE,
) -> ReportArtifact:
    schema_name = table_contract.schema_name if table_contract else "summary"
    schema_version = table_contract.schema_version if table_contract else 1
    orientation = table_contract.orientation if table_contract else "as_declared"
    units = dict(table_contract.units) if table_contract else {}
    return ReportArtifact(
        artifact_id=artifact_id,
        schema_name=schema_name,
        schema_version=schema_version,
        section_id="summary",
        evidence_layer=EvidenceLayer.DESCRIPTIVE,
        orientation=orientation,
        coordinates={"scope": "dataset"},
        status=ArtifactStatus.READY,
        reason=None,
        error=None,
        source_refs=("source_data",),
        path=path,
        media_type=media_type,
        sha256=artifact_sha256(payload),
        table_contract=table_contract,
        estimand="descriptive score",
        units=units,
        support={"n_rows": 5},
        missing="reported as null",
        tie="not applicable",
        test="not applicable",
        multiplicity="not applicable",
        privacy=privacy,
    )


def _lineage() -> ReportLineage:
    return ReportLineage(
        dataset=DatasetLineage(
            dataset_sha256="a" * 64,
            row_ids_sha256="b" * 64,
            group_partition_sha256="c" * 64,
            group_source="row_id",
            n_rows=5,
            n_groups=1,
        ),
        sources=(SourceArtifactReference(
            source_id="source_data",
            artifact_type="analysis.result",
            schema_version=1,
            sha256="d" * 64,
        ),),
        compiler=CompilerProvenance(
            compiler_name="test_compiler",
            compiler_version="1.0",
            report_spec_name="test_report",
            report_spec_version=1,
            report_spec_sha256="e" * 64,
        ),
        sampling=SamplingProvenance(
            method="none",
            sampling_frame_sha256="f" * 64,
            seed=None,
            population_count=5,
            sampled_count=0,
            max_examples_per_feature=0,
        ),
    )


def _manifest(
    artifacts: tuple[ReportArtifact, ...],
    policy: PrivacyPolicy,
    *,
    status: ReportStatus = ReportStatus.READY,
) -> ReportManifest:
    sections = []
    for section_id in dict.fromkeys(item.section_id for item in artifacts):
        owned = [item for item in artifacts if item.section_id == section_id]
        section = replace(
            _section(),
            section_id=section_id,
            title="Summary" if section_id == "summary" else "Failed section",
        )
        error_artifacts = [
            item for item in owned if item.status is ArtifactStatus.ERROR]
        ready_artifacts = [
            item for item in owned if item.status is ArtifactStatus.READY]
        if error_artifacts:
            section = replace(
                section,
                status=SectionStatus.ERROR,
                reason=StatusReason.PROCESSING_ERROR,
                error=error_artifacts[0].error,
            )
        elif not ready_artifacts:
            reason = owned[0].reason if owned else StatusReason.INPUT_ABSENT
            section = replace(
                section,
                status=SectionStatus.UNAVAILABLE,
                reason=reason or StatusReason.INPUT_ABSENT,
            )
        sections.append(section)
    if not sections:
        sections.append(replace(
            _section(),
            status=SectionStatus.UNAVAILABLE,
            reason=StatusReason.INPUT_ABSENT,
        ))
    evidence_layers = tuple({
        item.evidence_layer for item in sections
        if item.status is SectionStatus.READY
    })
    return ReportManifest(
        name="test_report",
        title="Test report",
        status=status,
        sections=tuple(sections),
        capabilities=ReportCapabilities(
            mode=ReportMode.TABLE_ONLY,
            n_rows=5,
            n_groups=1,
            evidence_layers=evidence_layers,
            table_only=True,
        ),
        lineage=_lineage(),
        artifacts=artifacts,
        errors=(),
        privacy=policy.to_manifest(),
    )


def test_ready_bundle_round_trip_and_manifest_name(tmp_path):
    policy = _policy()
    payload = json_payload({"score": 0.5}, privacy_policy=policy)
    artifact = _artifact(payload)
    bundle = write_report_bundle(
        tmp_path / "report", _manifest((artifact,), policy),
        {artifact.artifact_id: payload},
    )
    assert (bundle.root / MANIFEST_FILENAME).is_file()
    assert bundle.artifact("summary_data") == artifact
    assert bundle.read_json("summary_data") == {"score": 0.5}
    assert bundle.read_bytes("summary_data") == payload


def test_all_failed_and_partial_bundles_require_only_ready_payloads(tmp_path):
    policy = _policy()
    payload = json_payload({"score": 1.0}, privacy_policy=policy)
    ready = _artifact(payload)
    error = ReportError(code="serialization_failed", message="Serialization failed")
    failed = replace(
        ready,
        artifact_id="failed_data",
        section_id="failed",
        status=ArtifactStatus.ERROR,
        reason=StatusReason.PROCESSING_ERROR,
        error=error,
        path=None,
        media_type=None,
        sha256=None,
    )
    failed_manifest = _manifest((failed,), policy, status=ReportStatus.FAILED)
    loaded = write_report_bundle(tmp_path / "failed", failed_manifest, {})
    assert loaded.artifact("failed_data").status is ArtifactStatus.ERROR
    with pytest.raises(ValueError, match="no ready payload"):
        loaded.read_bytes("failed_data")

    partial = _manifest((ready, failed), policy, status=ReportStatus.PARTIAL)
    bundle = write_report_bundle(
        tmp_path / "partial", partial, {ready.path: payload})
    assert bundle.read_json(ready.artifact_id) == {"score": 1.0}


def test_payload_keys_are_exact_and_unambiguous(tmp_path):
    policy = _policy()
    payload = json_payload({"score": 1.0}, privacy_policy=policy)
    first = _artifact(payload, artifact_id="first", path="second")
    second = _artifact(payload, artifact_id="second", path="first")
    manifest = _manifest((first, second), policy)
    with pytest.raises(ValueError, match="ambiguous"):
        write_report_bundle(
            tmp_path / "ambiguous", manifest,
            {"first": payload, "second": payload})
    with pytest.raises(ValueError, match="exactly match"):
        write_report_bundle(
            tmp_path / "extra", manifest,
            {"first": payload, "second": payload, "x": b""})


def test_noncanonical_duplicate_and_nonfinite_json_are_refused(tmp_path):
    policy = _policy()
    bad_values = (
        b'{"score": 1.0}\n',
        b'{"score":1.0,"score":2.0}\n',
        b'{"score":1e400}\n',
        b'{"score":NaN}\n',
    )
    for number, payload in enumerate(bad_values):
        artifact = _artifact(payload)
        with pytest.raises(ValueError, match="JSON|canonical|finite|strict"):
            write_report_bundle(
                tmp_path / f"bad-{number}", _manifest((artifact,), policy),
                {artifact.artifact_id: payload},
            )


def test_raw_missing_values_require_json_payload_and_objects_are_pre_sanitized(tmp_path):
    policy = _policy()
    expected = json_payload({"score": float("nan")}, privacy_policy=policy)
    artifact = _artifact(expected)
    bundle = write_report_bundle(
        tmp_path / "missing", _manifest((artifact,), policy),
        {artifact.artifact_id: expected},
    )
    assert bundle.read_json(artifact.artifact_id) == {"score": None}
    with pytest.raises(ValueError, match="finite"):
        write_report_bundle(
            tmp_path / "raw-object", _manifest((artifact,), policy),
            {artifact.artifact_id: {"score": float("nan")}},
        )
    with pytest.raises(ValueError, match="finite"):
        canonical_json_bytes({"score": float("inf")}, privacy_policy=policy)



def test_object_payload_is_validated_without_resanitizing_opaque_ids(tmp_path):
    policy = _policy()
    sanitized = policy.sanitize({"row_id": "raw-row", "score": 1.0})
    payload = canonical_json_bytes(sanitized)
    artifact = _artifact(payload, privacy=ArtifactPrivacy.OPAQUE_ROWS)
    bundle = write_report_bundle(
        tmp_path / "opaque", _manifest((artifact,), policy),
        {artifact.artifact_id: sanitized},
    )
    assert bundle.read_json(artifact.artifact_id) == sanitized


def test_table_object_with_id_column_is_table_aware_and_not_rehashed(tmp_path):
    policy = _policy()
    contract = TableContract(
        schema_name="row_scores",
        schema_version=1,
        required_columns=("row_id", "score"),
        dtypes={"row_id": "string", "score": "float"},
        unique_key=("row_id",),
        orientation="as_declared",
        units={"score": "unitless"},
    )
    wire = table_to_json_table(
        pd.DataFrame({"row_id": ["raw-row"], "score": [0.5]}),
        contract,
        policy,
    )
    opaque_id = wire["records"][0]["row_id"]
    payload = canonical_json_bytes(wire)
    artifact = _artifact(
        payload,
        table_contract=contract,
        privacy=ArtifactPrivacy.OPAQUE_ROWS,
    )
    bundle = write_report_bundle(
        tmp_path / "table-object", _manifest((artifact,), policy),
        {artifact.artifact_id: wire},
    )
    assert bundle.read_json(artifact.artifact_id)["records"][0]["row_id"] == opaque_id


def test_read_json_is_bounded_and_does_not_delegate_to_unbounded_read_bytes(
    tmp_path, monkeypatch,
):
    import prefscope.reporting.io as report_io

    policy = _policy()
    payload = json_payload({"score": 1.0}, privacy_policy=policy)
    artifact = _artifact(payload)
    bundle = write_report_bundle(
        tmp_path / "report", _manifest((artifact,), policy),
        {artifact.artifact_id: payload},
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("read_json delegated to read_bytes")

    monkeypatch.setattr(report_io.ReportBundle, "read_bytes", forbidden)
    assert bundle.read_json(artifact.artifact_id) == {"score": 1.0}
    monkeypatch.setattr(report_io, "MAX_JSON_BYTES", len(payload) - 1)
    with pytest.raises(ValueError, match="byte"):
        bundle.read_json(artifact.artifact_id)


def test_loaded_bundle_rejects_replaced_root_on_later_reads(tmp_path):
    policy = _policy()
    first_payload = json_payload({"score": 1.0}, privacy_policy=policy)
    second_payload = json_payload({"score": 2.0}, privacy_policy=policy)
    first_artifact = _artifact(first_payload)
    second_artifact = _artifact(second_payload)
    first = tmp_path / "first"
    second = tmp_path / "second"
    bundle = write_report_bundle(
        first, _manifest((first_artifact,), policy),
        {first_artifact.artifact_id: first_payload})
    write_report_bundle(
        second, _manifest((second_artifact,), policy),
        {second_artifact.artifact_id: second_payload})
    os.replace(first, tmp_path / "original-first")
    os.replace(second, first)
    with pytest.raises(ValueError, match="replaced"):
        bundle.read_json(first_artifact.artifact_id)


def test_loader_inventory_uses_root_fd_not_os_walk(tmp_path, monkeypatch):
    import prefscope.reporting.io as report_io

    policy = _policy()
    payload = json_payload({"score": 1.0}, privacy_policy=policy)
    artifact = _artifact(payload)
    destination = tmp_path / "report"
    write_report_bundle(
        destination, _manifest((artifact,), policy),
        {artifact.artifact_id: payload})

    def forbidden(*args, **kwargs):
        raise AssertionError("path-based os.walk inventory was used")

    monkeypatch.setattr(report_io.os, "walk", forbidden)
    assert load_report_bundle(destination).read_json(artifact.artifact_id) == {"score": 1.0}


def test_path_component_and_aggregate_directory_limits(tmp_path, monkeypatch):
    import prefscope.reporting.io as report_io

    policy = _policy()
    payload = json_payload({"score": 1.0}, privacy_policy=policy)
    deep_path = "/".join(["directory"] * (report_io.MAX_PATH_COMPONENTS + 1))
    deep = _artifact(payload, path=deep_path)
    with pytest.raises(ValueError, match="component limit"):
        write_report_bundle(
            tmp_path / "deep", _manifest((deep,), policy),
            {deep.artifact_id: payload})

    first = _artifact(payload, artifact_id="first", path="one/value.json")
    second = _artifact(payload, artifact_id="second", path="two/value.json")
    monkeypatch.setattr(report_io, "MAX_BUNDLE_DIRECTORIES", 1)
    with pytest.raises(ValueError, match="director"):
        write_report_bundle(
            tmp_path / "too-many-directories", _manifest((first, second), policy),
            {"first": payload, "second": payload})


def test_text_snippet_artifact_uses_html_neutral_semantic_validator(tmp_path):
    policy = PrivacyPolicy.shareable(
        text=TextPolicy.SNIPPETS,
        snippet_chars=4,
        allow_fields=("score", "feature_id"),
        text_fields=("prompt_text",),
        id_fields=("row_id",),
        cell_count_fields=("n_rows",),
        categorical_fields={"scope": ("dataset",)},
    )
    sanitized = policy.sanitize({"prompt_text": "&&&&"})
    assert len(sanitized["prompt_text"]) > policy.snippet_chars + 1
    payload = canonical_json_bytes(sanitized)
    artifact = _artifact(payload, privacy=ArtifactPrivacy.TEXT_SNIPPETS)
    bundle = write_report_bundle(
        tmp_path / "snippet", _manifest((artifact,), policy),
        {artifact.artifact_id: sanitized},
    )
    assert bundle.read_json(artifact.artifact_id) == sanitized


def test_loader_detects_root_replacement_during_validation(tmp_path, monkeypatch):
    import prefscope.reporting.io as report_io

    policy = _policy()
    payload = json_payload({"score": 1.0}, privacy_policy=policy)
    artifact = _artifact(payload)
    target = tmp_path / "target"
    replacement = tmp_path / "replacement"
    write_report_bundle(
        target, _manifest((artifact,), policy), {artifact.artifact_id: payload})
    write_report_bundle(
        replacement, _manifest((artifact,), policy), {artifact.artifact_id: payload})
    original = tmp_path / "original-target"
    real_validate = report_io._validate_json_artifact
    injected = False

    def validate_and_replace(*args, **kwargs):
        nonlocal injected
        result = real_validate(*args, **kwargs)
        if not injected:
            os.replace(target, original)
            os.replace(replacement, target)
            injected = True
        return result

    monkeypatch.setattr(report_io, "_validate_json_artifact", validate_and_replace)
    with pytest.raises(ValueError, match="root changed"):
        report_io.load_report_bundle(target)

def test_shareable_privacy_leaks_and_non_json_are_refused(tmp_path):
    policy = _policy()
    leak = canonical_json_bytes({"email": "person@example.com"})
    artifact = _artifact(leak)
    with pytest.raises(ValueError, match="PII|field|email"):
        write_report_bundle(
            tmp_path / "leak", _manifest((artifact,), policy),
            {artifact.artifact_id: leak},
        )

    binary = b"opaque bytes"
    binary_artifact = _artifact(
        binary, path="data/blob.bin", media_type="application/octet-stream")
    with pytest.raises(ValueError, match="SHAREABLE|canonical JSON"):
        write_report_bundle(
            tmp_path / "binary", _manifest((binary_artifact,), policy),
            {binary_artifact.artifact_id: binary},
        )


def _table_contract(name: str = "scores") -> TableContract:
    return TableContract(
        schema_name=name,
        schema_version=1,
        required_columns=("feature_id", "score"),
        dtypes={"feature_id": "integer", "score": "float"},
        unique_key=("feature_id",),
        orientation="as_declared",
        units={"score": "unitless"},
    )


def test_table_contract_is_validated_on_write_and_load(tmp_path):
    policy = _policy()
    declared = _table_contract()
    wrong = _table_contract("other_scores")
    wire = table_to_json_table(
        pd.DataFrame({"feature_id": [1], "score": [0.5]}), wrong, policy)
    payload = canonical_json_bytes(wire)
    artifact = _artifact(payload, table_contract=declared)
    with pytest.raises(ValueError, match="contract does not match"):
        write_report_bundle(
            tmp_path / "table", _manifest((artifact,), policy),
            {artifact.artifact_id: payload},
        )

    binary = _artifact(
        b"x", media_type="application/octet-stream",
        path="data/table.bin", table_contract=declared)
    with pytest.raises(ValueError, match="table_contract"):
        write_report_bundle(
            tmp_path / "bad-media", _manifest((binary,), _policy(local=True)),
            {binary.artifact_id: b"x"},
        )


@pytest.mark.parametrize(
    "path",
    [
        "../escape.json",
        "data\\escape.json",
        "data/a:b.json",
        "data/CON.json",
        "data/trailing. ",
        "data/ünicode.json",
        "BUNDLE_MANIFEST.JSON",
    ],
)
def test_path_portability_is_strict(tmp_path, path):
    policy = _policy()
    payload = json_payload({"score": 1.0}, privacy_policy=policy)
    with pytest.raises(ValueError, match="path|portable|reserved|manifest|POSIX"):
        artifact = _artifact(payload, path=path)
        write_report_bundle(
            tmp_path / "unsafe", _manifest((artifact,), policy),
            {artifact.artifact_id: payload},
        )


def test_corruption_stale_files_and_symlinks_are_rejected(tmp_path):
    policy = _policy()
    payload = json_payload({"score": 1.0}, privacy_policy=policy)
    artifact = _artifact(payload)
    destination = tmp_path / "report"
    write_report_bundle(destination, _manifest((artifact,), policy), {"summary_data": payload})

    (destination / "stale.txt").write_text("stale")
    with pytest.raises(ValueError, match="mismatch"):
        load_report_bundle(destination)
    (destination / "stale.txt").unlink()

    target = destination / artifact.path
    target.unlink()
    target.symlink_to(destination / MANIFEST_FILENAME)
    with pytest.raises(ValueError, match="symlink|regular|safely"):
        load_report_bundle(destination)



def test_artifact_byte_corruption_is_rejected(tmp_path):
    policy = _policy()
    payload = json_payload({"score": 1.0}, privacy_policy=policy)
    artifact = _artifact(payload)
    destination = tmp_path / "report"
    write_report_bundle(
        destination, _manifest((artifact,), policy), {"summary_data": payload})
    (destination / artifact.path).write_bytes(
        json_payload({"score": 2.0}, privacy_policy=policy))
    with pytest.raises(ValueError, match="corrupt"):
        load_report_bundle(destination)


def test_case_insensitive_artifact_path_collision_is_rejected(tmp_path):
    policy = _policy()
    payload = json_payload({"score": 1.0}, privacy_policy=policy)
    upper = _artifact(payload, artifact_id="upper", path="data/Value.json")
    lower = _artifact(payload, artifact_id="lower", path="data/value.json")
    with pytest.raises(ValueError, match="case-insensitively"):
        write_report_bundle(
            tmp_path / "collision", _manifest((upper, lower), policy),
            {"upper": payload, "lower": payload})


def test_manifest_member_symlink_is_rejected(tmp_path):
    policy = _policy()
    payload = json_payload({"score": 1.0}, privacy_policy=policy)
    artifact = _artifact(payload)
    destination = tmp_path / "report"
    write_report_bundle(
        destination, _manifest((artifact,), policy), {"summary_data": payload})
    manifest_copy = tmp_path / "manifest-copy.json"
    os.replace(destination / MANIFEST_FILENAME, manifest_copy)
    (destination / MANIFEST_FILENAME).symlink_to(manifest_copy)
    with pytest.raises(ValueError, match="regular|safely"):
        load_report_bundle(destination)


def test_on_disk_json_table_tamper_fails_contract_even_with_updated_hash(tmp_path):
    policy = _policy()
    contract = _table_contract()
    wire = table_to_json_table(
        pd.DataFrame({"feature_id": [1], "score": [0.5]}), contract, policy)
    payload = canonical_json_bytes(wire)
    artifact = _artifact(payload, table_contract=contract)
    destination = tmp_path / "table-report"
    write_report_bundle(
        destination, _manifest((artifact,), policy), {artifact.artifact_id: payload})

    tampered = dict(wire)
    tampered["records"] = [{"feature_id": 1, "score": 2}]
    tampered_payload = canonical_json_bytes(tampered)
    (destination / artifact.path).write_bytes(tampered_payload)
    manifest_path = destination / MANIFEST_FILENAME
    raw_manifest = json.loads(manifest_path.read_text())
    raw_manifest["artifacts"][0]["sha256"] = artifact_sha256(tampered_payload)
    manifest_path.write_bytes(canonical_json_bytes(raw_manifest))
    with pytest.raises(ValueError, match="invalid values|represented"):
        load_report_bundle(destination)


def test_writer_rejects_immediate_parent_symlink(tmp_path):
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    policy = _policy()
    payload = json_payload({"score": 1.0}, privacy_policy=policy)
    artifact = _artifact(payload)
    with pytest.raises(ValueError, match="parent.*symlink"):
        write_report_bundle(
            linked_parent / "report", _manifest((artifact,), policy),
            {artifact.artifact_id: payload})

def test_unmanaged_overwrite_is_refused(tmp_path):
    destination = tmp_path / "report"
    destination.mkdir()
    (destination / "important.txt").write_text("do not delete")
    policy = _policy()
    payload = json_payload({"score": 1.0}, privacy_policy=policy)
    artifact = _artifact(payload)
    with pytest.raises(ValueError, match="manifest|bundle"):
        write_report_bundle(
            destination, _manifest((artifact,), policy),
            {artifact.artifact_id: payload}, overwrite=True)
    assert (destination / "important.txt").read_text() == "do not delete"


def test_no_overwrite_final_no_replace_leaves_late_empty_directory(
    tmp_path, monkeypatch,
):
    import prefscope.reporting.io as report_io

    destination = tmp_path / "report"
    policy = _policy()
    payload = json_payload({"score": 1.0}, privacy_policy=policy)
    artifact = _artifact(payload)
    real_rename = report_io._rename_no_replace

    def collide_at_final_install(source, target):
        destination.mkdir()
        return real_rename(source, target)

    monkeypatch.setattr(report_io, "_rename_no_replace", collide_at_final_install)
    with pytest.raises(FileExistsError, match="appeared"):
        report_io.write_report_bundle(
            destination, _manifest((artifact,), policy),
            {artifact.artifact_id: payload})
    assert destination.is_dir()
    assert list(destination.iterdir()) == []


def test_report_bundle_root_is_absolute_and_survives_cwd_change(tmp_path, monkeypatch):
    working = tmp_path / "working"
    working.mkdir()
    policy = _policy()
    payload = json_payload({"score": 1.0}, privacy_policy=policy)
    artifact = _artifact(payload)
    monkeypatch.chdir(working)
    bundle = write_report_bundle(
        "report", _manifest((artifact,), policy), {"summary_data": payload})
    assert bundle.root.is_absolute()
    monkeypatch.chdir(tmp_path)
    assert bundle.read_json("summary_data") == {"score": 1.0}


def test_overwrite_revalidates_destination_immediately_before_backup(
    tmp_path, monkeypatch,
):
    import prefscope.reporting.io as report_io

    destination = tmp_path / "report"
    stolen = tmp_path / "stolen-valid-report"
    policy = _policy()
    old_payload = json_payload({"score": 1.0}, privacy_policy=policy)
    old_artifact = _artifact(old_payload)
    report_io.write_report_bundle(
        destination, _manifest((old_artifact,), policy),
        {"summary_data": old_payload})

    new_payload = json_payload({"score": 2.0}, privacy_policy=policy)
    new_artifact = _artifact(new_payload)
    real_load = report_io.load_report_bundle
    injected = False

    def load_and_replace_before_backup(path, *args, **kwargs):
        nonlocal injected
        result = real_load(path, *args, **kwargs)
        if not injected and Path(path).name.startswith(".report.tmp-"):
            os.replace(destination, stolen)
            destination.mkdir()
            (destination / "intruder.txt").write_text("mine")
            injected = True
        return result

    monkeypatch.setattr(report_io, "load_report_bundle", load_and_replace_before_backup)
    with pytest.raises(ValueError, match="manifest|bundle"):
        report_io.write_report_bundle(
            destination, _manifest((new_artifact,), policy),
            {"summary_data": new_payload}, overwrite=True)
    assert (destination / "intruder.txt").read_text() == "mine"
    assert real_load(stolen).read_json("summary_data") == {"score": 1.0}
    assert not list(tmp_path.glob(".report.bak-*"))

def test_overwrite_no_replace_leaves_late_empty_dir_and_retains_backup(
    tmp_path, monkeypatch,
):
    import prefscope.reporting.io as report_io

    destination = tmp_path / "report"
    policy = _policy()
    old_payload = json_payload({"score": 1.0}, privacy_policy=policy)
    old_artifact = _artifact(old_payload)
    old_manifest = _manifest((old_artifact,), policy)
    report_io.write_report_bundle(
        destination, old_manifest, {"summary_data": old_payload})

    new_payload = json_payload({"score": 2.0}, privacy_policy=policy)
    new_artifact = _artifact(new_payload)
    real_rename = report_io._rename_no_replace

    def collide_at_final_install(source, target):
        destination.mkdir()
        return real_rename(source, target)

    monkeypatch.setattr(report_io, "_rename_no_replace", collide_at_final_install)
    with pytest.raises(FileExistsError, match="appeared"):
        report_io.write_report_bundle(
            destination, _manifest((new_artifact,), policy),
            {"summary_data": new_payload}, overwrite=True)
    assert destination.is_dir()
    assert list(destination.iterdir()) == []
    backups = list(tmp_path.glob(".report.bak-*"))
    assert len(backups) == 1
    assert report_io.load_report_bundle(backups[0]).read_json("summary_data") == {
        "score": 1.0}
    assert not list(tmp_path.glob(".report.quarantine-*"))


def test_final_publication_failure_restores_old_bundle(tmp_path, monkeypatch):
    import prefscope.reporting.io as report_io

    destination = tmp_path / "report"
    policy = _policy()
    old_payload = json_payload({"score": 1.0}, privacy_policy=policy)
    old_artifact = _artifact(old_payload)
    report_io.write_report_bundle(
        destination, _manifest((old_artifact,), policy),
        {"summary_data": old_payload})

    new_payload = json_payload({"score": 2.0}, privacy_policy=policy)
    new_artifact = _artifact(new_payload)
    def fail_final_install(source, target):
        raise OSError("injected final install failure")

    monkeypatch.setattr(report_io, "_rename_no_replace", fail_final_install)
    with pytest.raises(OSError, match="injected"):
        report_io.write_report_bundle(
            destination, _manifest((new_artifact,), policy),
            {"summary_data": new_payload}, overwrite=True)
    restored = report_io.load_report_bundle(destination)
    assert restored.read_json("summary_data") == {"score": 1.0}
    assert not list(tmp_path.glob(".report.tmp-*"))
    assert not list(tmp_path.glob(".report.bak-*"))

def test_orphan_backup_is_recovered_by_next_writer(tmp_path):
    destination = tmp_path / "report"
    policy = _policy()
    payload = json_payload({"score": 1.0}, privacy_policy=policy)
    artifact = _artifact(payload)
    manifest = _manifest((artifact,), policy)
    write_report_bundle(destination, manifest, {"summary_data": payload})
    backup = tmp_path / ".report.bak-crash"
    os.replace(destination, backup)

    with pytest.raises(FileExistsError):
        write_report_bundle(destination, manifest, {"summary_data": payload})
    assert load_report_bundle(destination).read_json("summary_data") == {"score": 1.0}
    assert not backup.exists()


def test_local_binary_path_payload_streams_and_hashes_off_skips_read(tmp_path, monkeypatch):
    import prefscope.reporting.io as report_io

    source = tmp_path / "source.bin"
    source.write_bytes(b"0123456789" * 200_000)
    path_payload = PathPayload(source)
    artifact = _artifact(
        b"", path="data/blob.bin", media_type="application/octet-stream")
    artifact = replace(artifact, sha256=artifact_sha256(path_payload))
    policy = _policy(local=True)
    destination = tmp_path / "local"
    write_report_bundle(
        destination, _manifest((artifact,), policy),
        {artifact.artifact_id: path_payload})

    def should_not_hash(*args, **kwargs):
        raise AssertionError("binary payload was read with verify_hashes=False")

    monkeypatch.setattr(report_io, "_hash_relative", should_not_hash)
    loaded = report_io.load_report_bundle(destination, verify_hashes=False)
    assert loaded.artifact(artifact.artifact_id).sha256 == artifact.sha256
    assert loaded.read_bytes(artifact.artifact_id) == source.read_bytes()




def test_reporting_bundle_import_is_torch_free():
    code = """
import builtins
real_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == 'torch' or name.startswith('torch.'):
        raise AssertionError('report bundle I/O imported torch')
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
from prefscope.reporting.io import load_report_bundle
"""
    result = subprocess.run(
        [sys.executable, "-c", code], text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
