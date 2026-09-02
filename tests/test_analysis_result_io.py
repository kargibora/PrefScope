from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

import prefscope.api.analysis_io as analysis_io
from prefscope.api.analysis_contracts import (
    AnalysisArtifact,
    AnalysisDataset,
    OutcomeSpec,
)
from prefscope.api.analysis_execution import DatasetAnalysisResult
from prefscope.api.analysis_io import (
    AnalysisDatasetReference,
    LoadedAnalysisResult,
    load_analysis_result,
    save_analysis_result,
)
from prefscope.core.features import FeatureMatrix
from prefscope.core.table_schema import TableContract


def _dataset(*, row_ids=("r0", "r1", "r2", "r3"), groups=("a", "a", "b", "c")):
    features = FeatureMatrix(
        np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32),
        row_ids=row_ids,
        role="response",
        orientation="none",
        provenance={"source": "synthetic-test"},
    )
    outcome = OutcomeSpec(
        [0.0, 0.0, 1.0, 1.0], row_ids=row_ids, kind="binary")
    return AnalysisDataset(
        features={"response": features},
        outcomes={"accepted": outcome},
        group_ids=groups,
    )


def _summary_contract():
    return TableContract(
        schema_name="group_summary",
        schema_version=1,
        required_columns=("group", "estimate", "supported"),
        dtypes={"group": "string", "estimate": "float", "supported": "boolean"},
        unique_key=("group",),
        orientation="declared_outcome",
        units={"estimate": "probability points"},
    )


def _diagnostics_contract():
    return TableContract(
        schema_name="run_diagnostics",
        schema_version=1,
        required_columns=("metric", "value"),
        dtypes={"metric": "string", "value": "integer"},
        unique_key=("metric",),
        orientation="none",
        units={},
    )


def _result(dataset=None):
    dataset = _dataset() if dataset is None else dataset
    summary = AnalysisArtifact(
        name="group_summary",
        table=pd.DataFrame({
            "group": ["a", "b"],
            "estimate": np.array([0.25, 0.75], dtype=float),
            "supported": [True, False],
        }),
        estimand="descriptive group-level probability difference",
        metadata={
            "component": "synthetic-group-summary",
            "multiplicity": "none; deterministic fixture",
        },
        table_contract=_summary_contract(),
    )
    diagnostics = AnalysisArtifact(
        name="run_diagnostics",
        table=pd.DataFrame({"metric": ["n_groups"], "value": [3]}),
        estimand="deterministic analysis-run diagnostics",
        metadata={"component": "synthetic-diagnostics", "inference": "none"},
        table_contract=_diagnostics_contract(),
    )
    return DatasetAnalysisResult(
        dataset=dataset,
        artifacts={summary.name: summary, diagnostics.name: diagnostics},
    )


def _manifest(path):
    return json.loads((path / "manifest.json").read_text())


def _write_manifest(path, manifest):
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))


def test_detached_round_trip_has_distinct_result_type_and_report_boundary(tmp_path):
    source = _result()
    destination = save_analysis_result(source, tmp_path / "analysis")

    assert {entry.name for entry in destination.iterdir()} == {
        "manifest.json", "group_summary.parquet", "run_diagnostics.parquet",
    }
    detached = load_analysis_result(destination)
    assert isinstance(detached, LoadedAnalysisResult)
    assert not isinstance(detached, DatasetAnalysisResult)
    assert not hasattr(detached, "dataset")
    assert isinstance(detached.dataset_reference, AnalysisDatasetReference)
    assert detached.dataset_reference.row_ids == source.dataset.row_ids
    assert detached.dataset_reference.n_rows == 4
    assert detached.dataset_reference.group_codes == (0, 0, 1, 2)
    assert detached.dataset_reference.n_groups == 3
    assert detached.to_manifest()["row_ids_sha256"] == source.to_manifest()[
        "row_ids_sha256"]

    for name in source.artifacts:
        pd.testing.assert_frame_equal(
            detached.artifact(name).table,
            source.artifact(name).table.reset_index(drop=True),
        )
        assert detached.artifact(name).estimand == source.artifact(name).estimand
        assert dict(detached.artifact(name).metadata) == dict(
            source.artifact(name).metadata)
        assert detached.artifact(name).table_contract == source.artifact(
            name).table_contract

    # Report adapters can consume the common artifacts/artifact(name) surface, while
    # only an explicitly reattached result exposes the complete analysis dataset.
    assert tuple(detached.artifacts) == ("group_summary", "run_diagnostics")
    attached = load_analysis_result(destination, dataset=source.dataset)
    assert isinstance(attached, DatasetAnalysisResult)
    assert attached.dataset is source.dataset
    assert attached.artifact("group_summary").name == detached.artifact(
        "group_summary").name


def test_manifest_persists_canonical_group_codes_and_validates_reattachment(tmp_path):
    source = _result()
    destination = save_analysis_result(source, tmp_path / "analysis")
    manifest = _manifest(destination)
    assert manifest["group_source"] == "explicit"
    assert manifest["group_codes"] == [0, 0, 1, 2]
    assert len(manifest["group_partition_sha256"]) == 64

    wrong_order = _dataset(
        row_ids=("r1", "r0", "r2", "r3"), groups=("a", "a", "b", "c"))
    with pytest.raises(ValueError, match="row_ids do not exactly match"):
        load_analysis_result(destination, dataset=wrong_order)

    # Labels may differ, but the scientific partition must be identical.
    relabeled = _dataset(groups=(10, 10, 20, 30))
    assert load_analysis_result(destination, dataset=relabeled).dataset is relabeled
    wrong_partition = _dataset(groups=("a", "b", "b", "c"))
    with pytest.raises(ValueError, match="group partition does not match"):
        load_analysis_result(destination, dataset=wrong_partition)

    manifest["group_codes"] = [1, 1, 0, 2]
    _write_manifest(destination, manifest)
    with pytest.raises(ValueError, match="canonical in first-appearance order"):
        load_analysis_result(destination)


def test_save_requires_contract_and_default_unnamed_range_index(tmp_path):
    dataset = _dataset()
    uncontracted = AnalysisArtifact(
        name="uncontracted",
        table=pd.DataFrame({"value": [1]}),
        estimand="uncontracted test value",
    )
    result = DatasetAnalysisResult(
        dataset=dataset, artifacts={uncontracted.name: uncontracted})
    with pytest.raises(ValueError, match="needs a TableContract"):
        save_analysis_result(result, tmp_path / "uncontracted")

    table = pd.DataFrame({
        "group": ["a", "b"], "estimate": [0.25, 0.75],
        "supported": [True, False],
    }, index=pd.Index(["row-a", "row-b"], name="result_row"))
    indexed = AnalysisArtifact(
        name="group_summary",
        table=table,
        estimand="descriptive group-level probability difference",
        table_contract=_summary_contract(),
    )
    result = DatasetAnalysisResult(
        dataset=dataset, artifacts={indexed.name: indexed})
    with pytest.raises(ValueError, match="default unnamed RangeIndex"):
        save_analysis_result(result, tmp_path / "indexed")


def test_load_uses_logical_contract_without_physical_dtype_schema(tmp_path):
    destination = save_analysis_result(_result(), tmp_path / "analysis")
    manifest = _manifest(destination)
    artifact = manifest["artifacts"][0]
    assert "physical_dtypes" not in artifact and "dtypes" not in artifact
    loaded = load_analysis_result(destination)
    assert loaded.artifact("group_summary").table_contract == _summary_contract()

    # Unknown physical dtype provenance cannot be smuggled into the strict schema.
    artifact["physical_dtypes"] = {
        column: "untrusted" for column in artifact["columns"]
    }
    _write_manifest(destination, manifest)
    with pytest.raises(ValueError, match=r"extra=\['physical_dtypes'\]"):
        load_analysis_result(destination)
    artifact.pop("physical_dtypes")

    # Even with a valid new file checksum, logical unique-key corruption fails closed.
    table_path = destination / artifact["file"]
    table = pd.read_parquet(table_path)
    table.loc[1, "group"] = table.loc[0, "group"]
    table.to_parquet(table_path, index=False)
    artifact["sha256"] = hashlib.sha256(table_path.read_bytes()).hexdigest()
    _write_manifest(destination, manifest)
    with pytest.raises(ValueError, match="unique key contains duplicates"):
        load_analysis_result(destination)

def test_manifest_json_rejects_duplicate_keys_and_nonfinite_constants(tmp_path):
    duplicate = save_analysis_result(_result(), tmp_path / "duplicate")
    raw = (duplicate / "manifest.json").read_text()
    raw = raw.replace(
        '"artifact_type": "prefscope.dataset_analysis_result",',
        '"artifact_type": "prefscope.dataset_analysis_result",\n'
        '  "artifact_type": "prefscope.dataset_analysis_result",',
        1,
    )
    (duplicate / "manifest.json").write_text(raw)
    with pytest.raises(ValueError, match="duplicate key 'artifact_type'"):
        load_analysis_result(duplicate)

    nonfinite = save_analysis_result(_result(), tmp_path / "nonfinite")
    raw = (nonfinite / "manifest.json").read_text()
    raw = raw.replace('"n_rows": 4', '"n_rows": NaN', 1)
    (nonfinite / "manifest.json").write_text(raw)
    with pytest.raises(ValueError, match="non-portable constant NaN"):
        load_analysis_result(nonfinite)


def test_loader_rejects_corruption_unsafe_paths_extras_and_symlinks(tmp_path):
    corrupted = save_analysis_result(_result(), tmp_path / "corrupted")
    table_path = corrupted / "group_summary.parquet"
    table_path.write_bytes(table_path.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="content does not match sha256"):
        load_analysis_result(corrupted)

    unsafe = save_analysis_result(_result(), tmp_path / "unsafe")
    manifest = _manifest(unsafe)
    manifest["artifacts"][0]["file"] = "../group_summary.parquet"
    _write_manifest(unsafe, manifest)
    with pytest.raises(ValueError, match="non-canonical or unsafe file path"):
        load_analysis_result(unsafe)

    extra = save_analysis_result(_result(), tmp_path / "extra")
    (extra / "stale.parquet").write_bytes(b"stale")
    with pytest.raises(ValueError, match="undeclared artifacts"):
        load_analysis_result(extra)

    linked = save_analysis_result(_result(), tmp_path / "linked")
    artifact_path = linked / "group_summary.parquet"
    outside = tmp_path / "outside.parquet"
    artifact_path.replace(outside)
    artifact_path.symlink_to(outside)
    with pytest.raises(ValueError, match="must be a regular file"):
        load_analysis_result(linked)


def test_parquet_hash_and_parse_use_one_stable_open_snapshot(tmp_path, monkeypatch):
    destination = save_analysis_result(_result(), tmp_path / "analysis")
    artifact_path = destination / "group_summary.parquet"
    original_read_parquet = analysis_io.pd.read_parquet
    observed = {"spooled": False, "rolled": False}
    monkeypatch.setattr(analysis_io, "_SPOOL_MEMORY_BYTES", 1)

    def replace_path_during_parse(source, *args, **kwargs):
        observed["spooled"] = isinstance(source, tempfile.SpooledTemporaryFile)
        observed["rolled"] = bool(getattr(source, "_rolled", False))
        if observed["spooled"] and artifact_path.exists():
            old_path = destination / "old.parquet"
            os.replace(artifact_path, old_path)
            artifact_path.write_bytes(old_path.read_bytes())
        return original_read_parquet(source, *args, **kwargs)

    monkeypatch.setattr(analysis_io.pd, "read_parquet", replace_path_during_parse)
    with pytest.raises(ValueError, match="changed while parsed"):
        load_analysis_result(destination)
    assert observed == {"spooled": True, "rolled": True}


def test_publication_race_quarantines_intruder_and_restores_backup(
    tmp_path, monkeypatch,
):
    destination = tmp_path / "analysis"
    destination.mkdir()

    def racing_rename(source, target):
        destination.mkdir()
        (destination / "racer.txt").write_text("new untrusted occupant")
        raise OSError("synthetic final rename race")

    monkeypatch.setattr(analysis_io, "_rename_no_replace", racing_rename)
    with pytest.raises(OSError, match="synthetic final rename race"):
        save_analysis_result(_result(), destination)

    assert destination.is_dir() and list(destination.iterdir()) == []
    quarantines = list(tmp_path.glob(".analysis.quarantine-*"))
    assert len(quarantines) == 1
    assert (quarantines[0] / "racer.txt").read_text() == "new untrusted occupant"
    assert not list(tmp_path.glob(".analysis.tmp-*"))


def test_failed_backup_restore_keeps_backup_and_quarantine(tmp_path, monkeypatch):
    destination = tmp_path / "analysis"
    destination.mkdir()
    real_replace = analysis_io.os.replace

    def racing_rename(source, target):
        destination.mkdir()
        (destination / "racer.txt").write_text("intruder")
        raise OSError("publish failed")

    def failing_restore(source, target):
        if ".analysis.bak-" in os.fspath(source):
            raise OSError("restore failed")
        return real_replace(source, target)

    monkeypatch.setattr(analysis_io, "_rename_no_replace", racing_rename)
    monkeypatch.setattr(analysis_io.os, "replace", failing_restore)
    with pytest.raises(OSError, match="restore failed"):
        save_analysis_result(_result(), destination)
    assert len(list(tmp_path.glob(".analysis.bak-*"))) == 1
    assert len(list(tmp_path.glob(".analysis.quarantine-*"))) == 1
    assert not destination.exists()


def test_initially_absent_late_destination_is_left_untouched(tmp_path, monkeypatch):
    destination = tmp_path / "analysis"
    observed = {}

    def late_appearance(source, target):
        destination.mkdir()
        observed["identity"] = (
            destination.stat().st_dev, destination.stat().st_ino)
        raise FileExistsError("late valid destination")

    monkeypatch.setattr(analysis_io, "_rename_no_replace", late_appearance)
    with pytest.raises(FileExistsError, match="late valid destination"):
        save_analysis_result(_result(), destination)

    assert destination.is_dir() and list(destination.iterdir()) == []
    assert (destination.stat().st_dev, destination.stat().st_ino) == observed["identity"]
    assert not list(tmp_path.glob(".analysis.bak-*"))
    assert not list(tmp_path.glob(".analysis.quarantine-*"))


def test_loader_rejects_root_symlink_swap_after_descriptor_reads(
    tmp_path, monkeypatch,
):
    destination = save_analysis_result(_result(), tmp_path / "analysis")
    moved = tmp_path / "original-analysis"
    original_read = analysis_io.pd.read_parquet
    swapped = {"done": False}

    def swap_root(source, *args, **kwargs):
        if not swapped["done"]:
            os.replace(destination, moved)
            destination.symlink_to(moved, target_is_directory=True)
            swapped["done"] = True
        return original_read(source, *args, **kwargs)

    monkeypatch.setattr(analysis_io.pd, "read_parquet", swap_root)
    with pytest.raises(ValueError, match="root directory changed while loading"):
        load_analysis_result(destination)
    assert swapped["done"] and destination.is_symlink()

def test_save_rejects_immediate_symlink_parent(tmp_path):
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(ValueError, match="parent must not be a symlink"):
        save_analysis_result(_result(), linked_parent / "analysis")
    assert not (real_parent / "analysis").exists()

def test_save_refuses_nonempty_and_failed_staging_preserves_empty_destination(
    tmp_path, monkeypatch,
):
    nonempty = tmp_path / "nonempty"
    nonempty.mkdir()
    (nonempty / "old.txt").write_text("old")
    with pytest.raises(FileExistsError, match="not empty"):
        save_analysis_result(_result(), nonempty)
    assert (nonempty / "old.txt").read_text() == "old"

    empty = tmp_path / "empty"
    empty.mkdir()

    def fail_write(*args, **kwargs):
        raise RuntimeError("synthetic parquet failure")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail_write)
    with pytest.raises(RuntimeError, match="synthetic parquet failure"):
        save_analysis_result(_result(), empty)
    assert empty.is_dir() and list(empty.iterdir()) == []
    assert not list(tmp_path.glob(".empty.tmp-*"))
    assert not list(tmp_path.glob(".empty.bak-*"))





def test_decoded_cell_budgets_reject_low_row_footers_before_pandas(
    tmp_path, monkeypatch,
):
    destination = save_analysis_result(_result(), tmp_path / "analysis")

    def must_not_parse(*args, **kwargs):
        raise AssertionError("decoded-cell overflow reached pandas")

    monkeypatch.setattr(analysis_io.pd, "read_parquet", must_not_parse)
    # The first synthetic artifact has only 2 rows, but 2 x 3 footer cells still
    # exceeds this monkeypatched per-artifact budget.
    monkeypatch.setattr(analysis_io, "_MAX_ARTIFACT_CELLS", 5)
    with pytest.raises(ValueError, match="exceeds 5 decoded cells"):
        load_analysis_result(destination)

    # Each artifact now fits alone (6 and 2 cells), but the manifest total does not.
    monkeypatch.setattr(analysis_io, "_MAX_ARTIFACT_CELLS", 100)
    monkeypatch.setattr(analysis_io, "_MAX_TOTAL_ARTIFACT_CELLS", 7)
    with pytest.raises(ValueError, match="aggregate decoded cell budget"):
        load_analysis_result(destination)

def test_parquet_metadata_budgets_fail_before_pandas(tmp_path, monkeypatch):
    destination = save_analysis_result(_result(), tmp_path / "analysis")

    class OversizedMetadata:
        num_rows = 2
        num_columns = analysis_io._MAX_ARTIFACT_COLUMNS + 1
        num_row_groups = 0

    class FakeParquetFile:
        def __init__(self, source):
            self.metadata = OversizedMetadata()

    def must_not_parse(*args, **kwargs):
        raise AssertionError("oversized metadata reached pandas")

    monkeypatch.setattr(analysis_io.pq, "ParquetFile", FakeParquetFile)
    monkeypatch.setattr(analysis_io.pd, "read_parquet", must_not_parse)
    with pytest.raises(ValueError, match="exceeds .* columns"):
        load_analysis_result(destination)


def test_save_memory_and_serialized_size_budgets_precede_publication(
    tmp_path, monkeypatch,
):
    result = _result()

    def oversized_memory(self, *args, **kwargs):
        return pd.Series([analysis_io._MAX_TABLE_MEMORY_BYTES + 1])

    def must_not_serialize(*args, **kwargs):
        raise AssertionError("oversized table reached to_parquet")

    monkeypatch.setattr(pd.DataFrame, "memory_usage", oversized_memory)
    monkeypatch.setattr(pd.DataFrame, "to_parquet", must_not_serialize)
    with pytest.raises(ValueError, match="in-memory summary budget"):
        save_analysis_result(result, tmp_path / "memory")

    monkeypatch.undo()
    monkeypatch.setattr(analysis_io, "_MAX_ARTIFACT_BYTES", 10)

    def write_oversized(self, path, *args, **kwargs):
        path.write_bytes(b"x" * 11)

    monkeypatch.setattr(pd.DataFrame, "to_parquet", write_oversized)
    with pytest.raises(ValueError, match="exceeds 10 bytes"):
        save_analysis_result(result, tmp_path / "serialized")
    assert not (tmp_path / "serialized").exists()

def test_bounded_manifest_and_artifact_preflights_run_before_parse(
    tmp_path, monkeypatch,
):
    destination = save_analysis_result(_result(), tmp_path / "analysis")
    manifest_size = (destination / "manifest.json").stat().st_size
    monkeypatch.setattr(analysis_io, "_MAX_MANIFEST_BYTES", manifest_size - 1)
    with pytest.raises(ValueError, match="manifest.json.*exceeds"):
        load_analysis_result(destination)

    monkeypatch.setattr(analysis_io, "_MAX_MANIFEST_BYTES", 16 * 1024 * 1024)
    artifact_path = destination / "group_summary.parquet"
    monkeypatch.setattr(
        analysis_io, "_MAX_ARTIFACT_BYTES", artifact_path.stat().st_size - 1)

    def must_not_parse(*args, **kwargs):
        raise AssertionError("oversized artifact reached Parquet parser")

    monkeypatch.setattr(analysis_io.pd, "read_parquet", must_not_parse)
    with pytest.raises(ValueError, match="group_summary.parquet.*exceeds"):
        load_analysis_result(destination)


def test_parquet_row_count_is_checked_from_footer_before_table_allocation(
    tmp_path, monkeypatch,
):
    destination = save_analysis_result(_result(), tmp_path / "analysis")
    manifest = _manifest(destination)
    manifest["artifacts"][0]["n_rows"] = 1
    _write_manifest(destination, manifest)

    def must_not_allocate(*args, **kwargs):
        raise AssertionError("row-count mismatch reached full Parquet allocation")

    monkeypatch.setattr(analysis_io.pd, "read_parquet", must_not_allocate)
    with pytest.raises(ValueError, match="row count disagrees with manifest"):
        load_analysis_result(destination)


def test_artifact_count_and_row_limits_are_preflighted_on_save_and_load(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(analysis_io, "_MAX_ARTIFACT_COUNT", 1)
    with pytest.raises(ValueError, match="exceeds 1 artifacts"):
        save_analysis_result(_result(), tmp_path / "too-many")

    monkeypatch.setattr(analysis_io, "_MAX_ARTIFACT_COUNT", 1024)
    monkeypatch.setattr(analysis_io, "_MAX_ARTIFACT_ROWS", 1)
    with pytest.raises(ValueError, match="group_summary.*exceeds 1 rows"):
        save_analysis_result(_result(), tmp_path / "too-many-rows")

    monkeypatch.setattr(analysis_io, "_MAX_ARTIFACT_ROWS", 100_000_000)
    destination = save_analysis_result(_result(), tmp_path / "analysis")
    monkeypatch.setattr(analysis_io, "_MAX_ARTIFACT_COUNT", 1)
    with pytest.raises(ValueError, match="at most 1 entries"):
        load_analysis_result(destination)


def test_save_fsyncs_files_manifest_staging_and_parent(tmp_path, monkeypatch):
    calls = {"files": [], "manifest": [], "directories": []}
    real_file = analysis_io._fsync_file
    real_manifest = analysis_io._write_fsynced_manifest
    real_directory = analysis_io._fsync_directory

    def track_file(path):
        calls["files"].append(path.name)
        return real_file(path)

    def track_manifest(path, manifest):
        calls["manifest"].append(path.name)
        return real_manifest(path, manifest)

    def track_directory(path):
        calls["directories"].append(path.name)
        return real_directory(path)

    monkeypatch.setattr(analysis_io, "_fsync_file", track_file)
    monkeypatch.setattr(analysis_io, "_write_fsynced_manifest", track_manifest)
    monkeypatch.setattr(analysis_io, "_fsync_directory", track_directory)
    destination = save_analysis_result(_result(), tmp_path / "analysis")

    assert set(calls["files"]) == {
        "group_summary.parquet", "run_diagnostics.parquet",
    }
    assert calls["manifest"] == ["manifest.json"]
    assert any(name.startswith(".analysis.tmp-") for name in calls["directories"])
    assert tmp_path.name in calls["directories"]
    assert destination.is_dir()

def test_analysis_io_import_is_torch_free():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import prefscope.api.analysis_io; "
            "assert 'torch' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
