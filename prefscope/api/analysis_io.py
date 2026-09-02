"""Durable, Torch-free I/O for task-centered dataset analysis results."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import ctypes
from dataclasses import dataclass
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import tempfile
from types import MappingProxyType
from typing import Mapping
import uuid

import pandas as pd
import pyarrow.parquet as pq

from prefscope.analysis.grouping import factorize_group_ids
from prefscope.api._lens_publication import _publication_lock, _recover_orphan_backup
from prefscope.api.analysis_contracts import AnalysisArtifact, AnalysisDataset
from prefscope.api.analysis_execution import DatasetAnalysisResult
from prefscope.core.representation import validate_row_ids
from prefscope.core.table_schema import TableContract
from prefscope.observability.runtime import automatic_stage


_FORMAT = "prefscope.dataset_analysis_result"
_SCHEMA_VERSION = 1
_GROUP_SOURCES = frozenset({
    "row", "explicit", "canonical_group_id", "normalized_prompt_hash",
})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ARTIFACT_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
_MAX_MANIFEST_BYTES = 16 * 1024 * 1024
_MAX_ARTIFACT_BYTES = 128 * 1024 * 1024
_MAX_ARTIFACT_COUNT = 64
_MAX_ARTIFACT_ROWS = 500_000
_MAX_ARTIFACT_COLUMNS = 256
_MAX_ARTIFACT_ROW_GROUPS = 2048
_MAX_ARTIFACT_CELLS = 8_000_000
_MAX_ARTIFACT_UNCOMPRESSED_BYTES = 256 * 1024 * 1024
_MAX_TOTAL_ARTIFACT_BYTES = 256 * 1024 * 1024
_MAX_TOTAL_ARTIFACT_ROWS = 1_000_000
_MAX_TOTAL_ARTIFACT_CELLS = 16_000_000
_MAX_TOTAL_UNCOMPRESSED_BYTES = 512 * 1024 * 1024
_MAX_TABLE_MEMORY_BYTES = 128 * 1024 * 1024
_MAX_TOTAL_TABLE_MEMORY_BYTES = 256 * 1024 * 1024
_SPOOL_MEMORY_BYTES = 2 * 1024 * 1024
_SUPPRESS_LOAD_OBSERVATION: ContextVar[bool] = ContextVar(
    "prefscope_analysis_result_suppress_load_observation", default=False
)


@dataclass(frozen=True)
class AnalysisDatasetReference:
    """Detached identity of the aligned dataset used for saved estimates."""

    row_ids: tuple[str, ...]
    group_source: str
    group_codes: tuple[int, ...]
    row_ids_sha256: str
    group_partition_sha256: str

    def __post_init__(self) -> None:
        ids = validate_row_ids(self.row_ids)
        codes = _validate_group_codes(self.group_codes, n_rows=len(ids))
        if self.group_source not in _GROUP_SOURCES:
            raise ValueError(f"unsupported analysis group_source {self.group_source!r}")
        if self.group_source == "row" and codes != tuple(range(len(ids))):
            raise ValueError("row-grouped analysis must declare one canonical group per row")
        for name, value in (
            ("row_ids_sha256", self.row_ids_sha256),
            ("group_partition_sha256", self.group_partition_sha256),
        ):
            if not isinstance(value, str) or not _SHA256.fullmatch(value):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if _row_ids_hash(ids) != self.row_ids_sha256:
            raise ValueError("analysis row_ids do not match row_ids_sha256")
        if _group_codes_hash(codes) != self.group_partition_sha256:
            raise ValueError("analysis group_codes do not match group_partition_sha256")
        object.__setattr__(self, "row_ids", ids)
        object.__setattr__(self, "group_codes", codes)

    @property
    def n_rows(self) -> int:
        return len(self.row_ids)

    @property
    def n_groups(self) -> int:
        return len(set(self.group_codes))


@dataclass(frozen=True, eq=False)
class LoadedAnalysisResult:
    """Detached analysis artifacts plus dataset identity, without input arrays."""

    dataset_reference: AnalysisDatasetReference
    artifacts: Mapping[str, AnalysisArtifact]

    def __post_init__(self) -> None:
        if not isinstance(self.dataset_reference, AnalysisDatasetReference):
            raise ValueError("dataset_reference must be an AnalysisDatasetReference")
        artifacts = dict(self.artifacts)
        if (
            not artifacts
            or any(name != artifact.name for name, artifact in artifacts.items())
            or not all(isinstance(artifact, AnalysisArtifact) for artifact in artifacts.values())
            or any(artifact.table_contract is None for artifact in artifacts.values())
        ):
            raise ValueError(
                "loaded artifacts must be a non-empty name-aligned contracted mapping")
        object.__setattr__(self, "artifacts", MappingProxyType(artifacts))

    def artifact(self, name: str) -> AnalysisArtifact:
        try:
            return self.artifacts[name]
        except KeyError:
            raise ValueError(f"unknown analysis artifact {name!r}") from None

    def to_manifest(self) -> dict[str, object]:
        """Return the same portable in-memory summary shape as an attached result."""
        manifest = {
            "schema_version": 1,
            "n_rows": self.dataset_reference.n_rows,
            "row_ids_sha256": self.dataset_reference.row_ids_sha256,
            "group_source": self.dataset_reference.group_source,
            "artifacts": [artifact.to_manifest() for artifact in self.artifacts.values()],
        }
        json.dumps(manifest, sort_keys=True, allow_nan=False)
        return manifest


def _row_ids_hash(row_ids: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    digest.update(b"prefscope-analysis-rows-v1\0")
    for row_id in row_ids:
        encoded = row_id.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _canonical_group_codes(dataset: AnalysisDataset) -> tuple[int, ...]:
    if dataset.group_ids is None:
        return tuple(range(dataset.n_rows))
    codes, _ = factorize_group_ids(dataset.group_ids)
    return tuple(int(code) for code in codes)


def _validate_group_codes(value, *, n_rows: int) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError("analysis group_codes must be an array")
    codes = tuple(value)
    if len(codes) != n_rows or any(type(code) is not int or code < 0 for code in codes):
        raise ValueError(
            "analysis group_codes must contain one non-negative integer per row")
    seen: set[int] = set()
    for code in codes:
        if code not in seen:
            if code != len(seen):
                raise ValueError(
                    "analysis group_codes must be canonical in first-appearance order")
            seen.add(code)
    return codes


def _group_codes_hash(codes: tuple[int, ...]) -> str:
    digest = hashlib.sha256()
    digest.update(b"prefscope-analysis-groups-v1\0")
    digest.update(len(codes).to_bytes(8, "big"))
    for code in codes:
        digest.update(code.to_bytes(8, "big", signed=True))
    return digest.hexdigest()


def _require_exact_keys(value, expected: set[str], *, where: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be a JSON object")
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{where} fields do not match schema v{_SCHEMA_VERSION}: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    return value


def _reject_json_constant(value: str):
    raise ValueError(f"analysis result JSON contains non-portable constant {value}")


def _parse_finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"analysis result JSON contains non-finite number {value}")
    return parsed


def _reject_duplicate_keys(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"analysis result JSON contains duplicate key {key!r}")
        value[key] = item
    return value


def _strict_json_loads(raw: bytes) -> object:
    try:
        text = raw.decode("utf-8")
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
            parse_float=_parse_finite_json_float,
        )
    except UnicodeDecodeError as exc:
        raise ValueError("analysis result manifest is not valid UTF-8 JSON") from exc
    except json.JSONDecodeError as exc:
        raise ValueError("analysis result manifest is not valid UTF-8 JSON") from exc


def _stable_signature(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        info.st_dev, info.st_ino, info.st_size,
        info.st_mtime_ns, info.st_ctime_ns,
    )


def _open_regular_no_follow(path: Path) -> tuple[int, os.stat_result]:
    try:
        path_info = path.lstat()
    except FileNotFoundError:
        raise FileNotFoundError(f"analysis result is missing {path.name}") from None
    if not stat.S_ISREG(path_info.st_mode):
        raise ValueError(f"analysis result member {path.name!r} must be a regular file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(
            f"analysis result member {path.name!r} could not be opened safely") from exc
    opened = os.fstat(descriptor)
    if (
        not stat.S_ISREG(opened.st_mode)
        or (opened.st_dev, opened.st_ino) != (path_info.st_dev, path_info.st_ino)
    ):
        os.close(descriptor)
        raise ValueError(f"analysis result member {path.name!r} changed while opening")
    return descriptor, opened


def _path_still_matches(path: Path, opened: os.stat_result, current: os.stat_result) -> bool:
    try:
        path_info = path.lstat()
    except FileNotFoundError:
        return False
    return (
        stat.S_ISREG(path_info.st_mode)
        and (path_info.st_dev, path_info.st_ino) == (opened.st_dev, opened.st_ino)
        and _stable_signature(current) == _stable_signature(opened)
    )


def _open_root_directory(path: Path) -> tuple[int, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        raise FileNotFoundError(f"analysis result directory does not exist: {path}") from None
    except OSError as exc:
        raise ValueError(f"analysis result path is not a safe directory: {path}") from exc
    opened = os.fstat(descriptor)
    if not stat.S_ISDIR(opened.st_mode):
        os.close(descriptor)
        raise ValueError(f"analysis result path must be a directory: {path}")
    return descriptor, opened


def _root_still_matches(path: Path, descriptor: int, opened: os.stat_result) -> bool:
    try:
        current_path = path.lstat()
        current_fd = os.fstat(descriptor)
    except FileNotFoundError:
        return False
    return (
        stat.S_ISDIR(current_path.st_mode)
        and (current_path.st_dev, current_path.st_ino) == (opened.st_dev, opened.st_ino)
        and (current_fd.st_dev, current_fd.st_ino) == (opened.st_dev, opened.st_ino)
    )


def _open_member_no_follow(
    root_descriptor: int, name: str
) -> tuple[int, os.stat_result]:
    try:
        path_info = os.stat(name, dir_fd=root_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        raise FileNotFoundError(f"analysis result is missing {name}") from None
    if not stat.S_ISREG(path_info.st_mode):
        raise ValueError(f"analysis result member {name!r} must be a regular file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=root_descriptor)
    except OSError as exc:
        raise ValueError(
            f"analysis result member {name!r} could not be opened safely") from exc
    opened = os.fstat(descriptor)
    if (
        not stat.S_ISREG(opened.st_mode)
        or (opened.st_dev, opened.st_ino) != (path_info.st_dev, path_info.st_ino)
    ):
        os.close(descriptor)
        raise ValueError(f"analysis result member {name!r} changed while opening")
    return descriptor, opened


def _member_still_matches(
    root_descriptor: int, name: str, opened: os.stat_result, current: os.stat_result
) -> bool:
    try:
        member = os.stat(name, dir_fd=root_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return (
        stat.S_ISREG(member.st_mode)
        and (member.st_dev, member.st_ino) == (opened.st_dev, opened.st_ino)
        and _stable_signature(current) == _stable_signature(opened)
    )


def _read_stable_regular_bytes(
    root_descriptor: int, name: str, *, max_bytes: int
) -> bytes:
    descriptor, opened = _open_member_no_follow(root_descriptor, name)
    try:
        if opened.st_size > max_bytes:
            raise ValueError(
                f"analysis result member {name!r} exceeds {max_bytes} bytes")
        chunks = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, max_bytes - total + 1))
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise ValueError(
                    f"analysis result member {name!r} exceeds {max_bytes} bytes")
            chunks.append(chunk)
        current = os.fstat(descriptor)
        if not _member_still_matches(root_descriptor, name, opened, current):
            raise ValueError(f"analysis result member {name!r} changed while read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _stream_file_sha256(path: Path, *, max_bytes: int) -> str:
    descriptor, opened = _open_regular_no_follow(path)
    try:
        if opened.st_size > max_bytes:
            raise ValueError(
                f"analysis result member {path.name!r} exceeds {max_bytes} bytes")
        digest = hashlib.sha256()
        total = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise ValueError(
                    f"analysis result member {path.name!r} exceeds {max_bytes} bytes")
            digest.update(chunk)
        current = os.fstat(descriptor)
        if not _path_still_matches(path, opened, current):
            raise ValueError(f"analysis result member {path.name!r} changed while hashed")
        return digest.hexdigest()
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class _ParquetStats:
    file_bytes: int
    rows: int
    columns: int
    row_groups: int
    cells: int
    uncompressed_bytes: int


@contextmanager
def _verified_parquet_snapshot(
    root_descriptor: int, name: str, expected_sha256: str
):
    descriptor, opened = _open_member_no_follow(root_descriptor, name)
    try:
        if opened.st_size > _MAX_ARTIFACT_BYTES:
            raise ValueError(
                f"analysis artifact {name!r} exceeds {_MAX_ARTIFACT_BYTES} bytes")
        digest = hashlib.sha256()
        total = 0
        with tempfile.SpooledTemporaryFile(
            max_size=_SPOOL_MEMORY_BYTES, mode="w+b"
        ) as snapshot:
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > _MAX_ARTIFACT_BYTES:
                    raise ValueError(
                        f"analysis artifact {name!r} exceeds {_MAX_ARTIFACT_BYTES} bytes")
                snapshot.write(chunk)
                digest.update(chunk)
            after_read = os.fstat(descriptor)
            if not _member_still_matches(
                root_descriptor, name, opened, after_read
            ):
                raise ValueError(f"analysis artifact {name!r} changed while read")
            if digest.hexdigest() != expected_sha256:
                raise ValueError(
                    f"analysis artifact {Path(name).stem!r} content does not match sha256")
            snapshot.seek(0)
            yield snapshot, opened.st_size
            after_use = os.fstat(descriptor)
            if not _member_still_matches(
                root_descriptor, name, opened, after_use
            ):
                raise ValueError(f"analysis artifact {name!r} changed while parsed")
    finally:
        os.close(descriptor)


def _inspect_parquet(
    snapshot, *, name: str, file_bytes: int, expected_rows: int
) -> _ParquetStats:
    snapshot.seek(0)
    try:
        metadata = pq.ParquetFile(snapshot).metadata
        rows = int(metadata.num_rows)
        columns = int(metadata.num_columns)
        row_groups = int(metadata.num_row_groups)
        uncompressed = 0
        for group_index in range(row_groups):
            group = metadata.row_group(group_index)
            for column_index in range(group.num_columns):
                size = int(group.column(column_index).total_uncompressed_size)
                if size < 0:
                    raise ValueError("negative Parquet uncompressed size")
                uncompressed += size
    except Exception as exc:
        raise ValueError(
            f"analysis artifact {Path(name).stem!r} is not readable Parquet") from exc
    if rows != expected_rows:
        raise ValueError(
            f"analysis artifact {Path(name).stem!r} row count disagrees with manifest")
    if rows > _MAX_ARTIFACT_ROWS:
        raise ValueError(
            f"analysis artifact {Path(name).stem!r} exceeds {_MAX_ARTIFACT_ROWS} rows")
    if columns > _MAX_ARTIFACT_COLUMNS:
        raise ValueError(
            f"analysis artifact {Path(name).stem!r} exceeds "
            f"{_MAX_ARTIFACT_COLUMNS} columns")
    if row_groups > _MAX_ARTIFACT_ROW_GROUPS:
        raise ValueError(
            f"analysis artifact {Path(name).stem!r} exceeds "
            f"{_MAX_ARTIFACT_ROW_GROUPS} row groups")
    cells = rows * columns
    if cells > _MAX_ARTIFACT_CELLS:
        raise ValueError(
            f"analysis artifact {Path(name).stem!r} exceeds "
            f"{_MAX_ARTIFACT_CELLS} decoded cells")
    if uncompressed > _MAX_ARTIFACT_UNCOMPRESSED_BYTES:
        raise ValueError(
            f"analysis artifact {Path(name).stem!r} exceeds uncompressed byte budget")
    return _ParquetStats(
        file_bytes, rows, columns, row_groups, cells, uncompressed)


def _preflight_parquet(
    root_descriptor: int, name: str, expected_sha256: str, *, expected_rows: int
) -> _ParquetStats:
    with _verified_parquet_snapshot(
        root_descriptor, name, expected_sha256
    ) as (snapshot, file_bytes):
        return _inspect_parquet(
            snapshot, name=name, file_bytes=file_bytes, expected_rows=expected_rows)


def _read_verified_parquet(
    root_descriptor: int, name: str, expected_sha256: str, *, expected_rows: int
) -> pd.DataFrame:
    with _verified_parquet_snapshot(
        root_descriptor, name, expected_sha256
    ) as (snapshot, file_bytes):
        _inspect_parquet(
            snapshot, name=name, file_bytes=file_bytes, expected_rows=expected_rows)
        snapshot.seek(0)
        try:
            return pd.read_parquet(snapshot)
        except Exception as exc:
            raise ValueError(
                f"analysis artifact {Path(name).stem!r} is not readable Parquet") from exc


def _table_contract_from_manifest(value, *, artifact_name: str) -> TableContract:
    try:
        contract = TableContract.from_manifest(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"artifact {artifact_name!r} has an invalid table_schema") from exc
    if contract.schema_name != artifact_name:
        raise ValueError("artifact table_schema name does not match artifact name")
    return contract


def _validate_artifact_manifest(value) -> tuple[dict, TableContract]:
    manifest = _require_exact_keys(
        value,
        {
            "name", "estimand", "n_rows", "columns", "metadata",
            "table_schema", "file", "sha256",
        },
        where="analysis artifact manifest",
    )
    name = manifest["name"]
    if not isinstance(name, str) or not _ARTIFACT_NAME.fullmatch(name):
        raise ValueError(f"unsafe analysis artifact name: {name!r}")
    if manifest["file"] != f"{name}.parquet":
        raise ValueError(f"artifact {name!r} has a non-canonical or unsafe file path")
    if not isinstance(manifest["estimand"], str) or not manifest["estimand"]:
        raise ValueError(f"artifact {name!r} estimand must be a non-empty string")
    n_rows = manifest["n_rows"]
    if type(n_rows) is not int or not 0 <= n_rows <= _MAX_ARTIFACT_ROWS:
        raise ValueError(
            f"artifact {name!r} n_rows must be in [0, {_MAX_ARTIFACT_ROWS}]")
    columns = manifest["columns"]
    if (
        not isinstance(columns, list)
        or len(columns) > _MAX_ARTIFACT_COLUMNS
        or len(columns) != len(set(columns))
        or any(not isinstance(column, str) or not column for column in columns)
    ):
        raise ValueError(f"artifact {name!r} columns must be unique non-empty strings")
    if not isinstance(manifest["metadata"], dict):
        raise ValueError(f"artifact {name!r} metadata must be a JSON object")
    if not isinstance(manifest["sha256"], str) or not _SHA256.fullmatch(
        manifest["sha256"]
    ):
        raise ValueError(f"artifact {name!r} sha256 is invalid")
    return manifest, _table_contract_from_manifest(
        manifest["table_schema"], artifact_name=name)


def _read_manifest(root_descriptor: int) -> dict:
    value = _strict_json_loads(_read_stable_regular_bytes(
        root_descriptor, "manifest.json", max_bytes=_MAX_MANIFEST_BYTES))
    return _require_exact_keys(
        value,
        {
            "artifact_type", "schema_version", "n_rows", "row_ids",
            "row_ids_sha256", "group_source", "group_codes",
            "group_partition_sha256", "artifacts",
        },
        where="analysis result manifest",
    )


def _validate_dataset_identity(manifest: dict) -> AnalysisDatasetReference:
    if manifest["artifact_type"] != _FORMAT:
        raise ValueError(
            f"unsupported analysis result artifact_type {manifest['artifact_type']!r}")
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != _SCHEMA_VERSION:
        raise ValueError(
            f"unsupported analysis result schema {manifest['schema_version']!r}; "
            f"this build supports only {_SCHEMA_VERSION}")
    raw_ids = manifest["row_ids"]
    if not isinstance(raw_ids, list):
        raise ValueError("analysis result row_ids must be a JSON list")
    try:
        ids = validate_row_ids(raw_ids)
    except (TypeError, ValueError) as exc:
        raise ValueError("analysis result row_ids are invalid") from exc
    n_rows = manifest["n_rows"]
    if type(n_rows) is not int or n_rows <= 0 or n_rows != len(ids):
        raise ValueError("analysis result n_rows does not match row_ids")
    return AnalysisDatasetReference(
        row_ids=ids,
        group_source=manifest["group_source"],
        group_codes=manifest["group_codes"],
        row_ids_sha256=manifest["row_ids_sha256"],
        group_partition_sha256=manifest["group_partition_sha256"],
    )


def _has_default_index(table: pd.DataFrame) -> bool:
    index = table.index
    return (
        isinstance(index, pd.RangeIndex)
        and index.start == 0
        and index.stop == len(table)
        and index.step == 1
        and index.name is None
    )


def _file_sha256(path: Path) -> str:
    return _stream_file_sha256(path, max_bytes=_MAX_ARTIFACT_BYTES)


def _fsync_file(path: Path) -> None:
    descriptor, _ = _open_regular_no_follow(path)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_fsynced_manifest(path: Path, manifest: dict) -> None:
    payload = (
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    if len(payload) > _MAX_MANIFEST_BYTES:
        raise ValueError(
            f"analysis result manifest exceeds {_MAX_MANIFEST_BYTES} bytes")
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _rename_no_replace(source: Path, destination: Path) -> None:
    """Atomically rename a directory only if destination is still absent."""
    libc = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    if sys.platform == "darwin" and hasattr(libc, "renamex_np"):
        result = libc.renamex_np(
            ctypes.c_char_p(source_bytes), ctypes.c_char_p(destination_bytes),
            ctypes.c_uint(0x00000004),  # RENAME_EXCL
        )
    elif sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        result = libc.renameat2(
            ctypes.c_int(-100), ctypes.c_char_p(source_bytes),
            ctypes.c_int(-100), ctypes.c_char_p(destination_bytes),
            ctypes.c_uint(0x1),  # RENAME_NOREPLACE
        )
    else:
        raise RuntimeError(
            "safe no-replace directory publication is unavailable on this platform")
    if result != 0:
        error = ctypes.get_errno()
        if error in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FileExistsError(
                f"analysis result output appeared during publication: {destination}")
        raise OSError(error, os.strerror(error), os.fspath(destination))


def _directory_identity(path: Path) -> tuple[int, int]:
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode):
        raise ValueError("analysis result output must remain a directory")
    return info.st_dev, info.st_ino


def _save_analysis_result(result, out) -> Path:
    """Transactionally publish a contracted result as manifest plus Parquet tables."""
    if not isinstance(result, DatasetAnalysisResult):
        raise ValueError("result must be a DatasetAnalysisResult")
    if not isinstance(result.dataset, AnalysisDataset):
        raise ValueError("result must be backed by a complete AnalysisDataset")
    if len(result.artifacts) > _MAX_ARTIFACT_COUNT:
        raise ValueError(
            f"analysis result exceeds {_MAX_ARTIFACT_COUNT} artifacts")
    total_rows = 0
    total_cells = 0
    total_memory = 0
    for artifact in result.artifacts.values():
        if artifact.table_contract is None:
            raise ValueError(
                f"artifact {artifact.name!r} needs a TableContract for durable I/O")
        if len(artifact.table) > _MAX_ARTIFACT_ROWS:
            raise ValueError(
                f"artifact {artifact.name!r} exceeds {_MAX_ARTIFACT_ROWS} rows")
        if len(artifact.table.columns) > _MAX_ARTIFACT_COLUMNS:
            raise ValueError(
                f"artifact {artifact.name!r} exceeds {_MAX_ARTIFACT_COLUMNS} columns")
        memory_bytes = int(artifact.table.memory_usage(index=True, deep=True).sum())
        if memory_bytes > _MAX_TABLE_MEMORY_BYTES:
            raise ValueError(
                f"artifact {artifact.name!r} exceeds in-memory summary budget")
        cells = len(artifact.table) * len(artifact.table.columns)
        if cells > _MAX_ARTIFACT_CELLS:
            raise ValueError(
                f"artifact {artifact.name!r} exceeds decoded cell budget")
        total_rows += len(artifact.table)
        total_cells += cells
        total_memory += memory_bytes
        if total_rows > _MAX_TOTAL_ARTIFACT_ROWS:
            raise ValueError("analysis result exceeds aggregate row budget")
        if total_cells > _MAX_TOTAL_ARTIFACT_CELLS:
            raise ValueError("analysis result exceeds aggregate decoded cell budget")
        if total_memory > _MAX_TOTAL_TABLE_MEMORY_BYTES:
            raise ValueError("analysis result exceeds aggregate in-memory summary budget")
        if not _has_default_index(artifact.table):
            raise ValueError(
                f"artifact {artifact.name!r} must use the default unnamed RangeIndex")
        artifact.table_contract.validate(artifact.table)
    if out is None:
        raise ValueError("save_analysis_result requires an output directory")
    requested = Path(os.path.abspath(Path(out).expanduser()))
    if requested.name in {"", ".", ".."}:
        raise ValueError("analysis result output must name a directory")
    if requested.parent.is_symlink():
        raise ValueError("analysis result output parent must not be a symlink")
    requested.parent.mkdir(parents=True, exist_ok=True)
    if requested.parent.is_symlink():
        raise ValueError("analysis result output parent must not be a symlink")
    root = requested
    if root.is_symlink() or (root.exists() and not root.is_dir()):
        raise ValueError("analysis result output must be a directory path, not a file/symlink")

    with _publication_lock(root):
        _recover_orphan_backup(root)
        if root.is_symlink() or (root.exists() and not root.is_dir()):
            raise ValueError(
                "analysis result output must be a directory path, not a file/symlink")
        initial_exists = root.exists()
        initial_identity = _directory_identity(root) if initial_exists else None
        if initial_exists and any(root.iterdir()):
            raise FileExistsError(
                f"analysis result output already exists and is not empty: {root}")
        staging = root.parent / f".{root.name}.tmp-{uuid.uuid4().hex}"
        backup = root.parent / f".{root.name}.bak-{uuid.uuid4().hex}"
        quarantine = root.parent / f".{root.name}.quarantine-{uuid.uuid4().hex}"
        backup_created = False
        try:
            staging.mkdir()
            artifact_manifests = []
            total_written_bytes = 0
            for artifact in result.artifacts.values():
                file_name = f"{artifact.name}.parquet"
                artifact_path = staging / file_name
                artifact.table.to_parquet(artifact_path, index=False)
                _fsync_file(artifact_path)
                saved = artifact.to_manifest()
                # Physical pandas dtype strings are neither stable nor the portability
                # contract. The versioned logical TableContract is persisted instead.
                saved.pop("dtypes")
                saved["file"] = file_name
                saved["sha256"] = _file_sha256(artifact_path)
                total_written_bytes += artifact_path.stat().st_size
                if total_written_bytes > _MAX_TOTAL_ARTIFACT_BYTES:
                    raise ValueError("analysis result exceeds aggregate artifact byte budget")
                artifact_manifests.append(saved)
            base = result.to_manifest()
            group_codes = _canonical_group_codes(result.dataset)
            manifest = {
                "artifact_type": _FORMAT,
                "schema_version": _SCHEMA_VERSION,
                "n_rows": base["n_rows"],
                "row_ids": list(result.dataset.row_ids),
                "row_ids_sha256": base["row_ids_sha256"],
                "group_source": base["group_source"],
                "group_codes": list(group_codes),
                "group_partition_sha256": _group_codes_hash(group_codes),
                "artifacts": artifact_manifests,
            }
            _write_fsynced_manifest(staging / "manifest.json", manifest)
            _fsync_directory(staging)
            _load_analysis_result_for_validation(
                staging, dataset=result.dataset
            )
            if initial_exists:
                if (
                    not root.exists()
                    or root.is_symlink()
                    or _directory_identity(root) != initial_identity
                    or any(root.iterdir())
                ):
                    raise FileExistsError(
                        "analysis result output changed during publication")
                os.replace(root, backup)
                backup_created = True
                if _directory_identity(backup) != initial_identity:
                    raise RuntimeError(
                        "analysis result output identity changed during backup rename")
                _fsync_directory(root.parent)
            elif root.exists() or root.is_symlink():
                # No-overwrite means even a valid empty directory that appeared late is
                # owned by somebody else and must remain completely untouched.
                raise FileExistsError(
                    f"analysis result output appeared during publication: {root}")
            try:
                _rename_no_replace(staging, root)
                _fsync_directory(root.parent)
            except BaseException as exc:
                if isinstance(exc, FileExistsError) and not initial_exists:
                    raise
                # For replacement of the original empty directory, preserve any
                # unexpected concurrent occupant under a quarantine name, then restore.
                if root.exists() or root.is_symlink():
                    try:
                        os.replace(root, quarantine)
                        _fsync_directory(root.parent)
                    except BaseException:
                        pass
                if backup_created and backup.exists() and not (
                    root.exists() or root.is_symlink()
                ):
                    os.replace(backup, root)
                    backup_created = False
                    _fsync_directory(root.parent)
                raise
            if backup_created:
                # Publication is already durable. Backup cleanup is best-effort and
                # must never turn successful publication into a reported failure.
                shutil.rmtree(backup, ignore_errors=True)
                backup_created = False
        finally:
            if staging.is_symlink():
                staging.unlink()
            elif staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
            # Never remove backup/quarantine here. On an exceptional publication path
            # they may be the only trustworthy copies of either participant's bytes.
    return root

def _load_analysis_result(path, *, dataset=None):
    """Load a strict result; reattach only an exactly matching complete dataset."""
    requested = Path(os.path.abspath(Path(path).expanduser()))
    if requested.is_symlink():
        raise ValueError("analysis result path must not be a symlink")
    root_descriptor, root_opened = _open_root_directory(requested)
    try:
        manifest = _read_manifest(root_descriptor)
        reference = _validate_dataset_identity(manifest)

        raw_artifacts = manifest["artifacts"]
        if (
            not isinstance(raw_artifacts, list)
            or not 1 <= len(raw_artifacts) <= _MAX_ARTIFACT_COUNT
        ):
            raise ValueError(
                "analysis result artifacts must be a non-empty JSON list with at most "
                f"{_MAX_ARTIFACT_COUNT} entries")
        declarations = []
        names = set()
        declared_total_rows = 0
        for raw in raw_artifacts:
            artifact_manifest, contract = _validate_artifact_manifest(raw)
            name = artifact_manifest["name"]
            if name in names:
                raise ValueError(f"analysis result declares duplicate artifact {name!r}")
            names.add(name)
            declared_total_rows += artifact_manifest["n_rows"]
            if declared_total_rows > _MAX_TOTAL_ARTIFACT_ROWS:
                raise ValueError("analysis result exceeds aggregate row budget")
            declarations.append((artifact_manifest, contract))
        expected_files = {
            "manifest.json", *(value[0]["file"] for value in declarations)}
        actual_files = set(os.listdir(root_descriptor))
        if actual_files != expected_files:
            raise ValueError(
                "analysis result contains missing or undeclared artifacts: "
                f"missing={sorted(expected_files - actual_files)}, "
                f"extra={sorted(actual_files - expected_files)}")

        # Inspect and hash every Parquet footer before pandas allocates any table. This
        # makes aggregate compressed/uncompressed budgets real preflight constraints.
        total_file_bytes = 0
        total_cells = 0
        total_uncompressed = 0
        for artifact_manifest, _ in declarations:
            stats = _preflight_parquet(
                root_descriptor,
                artifact_manifest["file"],
                artifact_manifest["sha256"],
                expected_rows=artifact_manifest["n_rows"],
            )
            total_file_bytes += stats.file_bytes
            total_cells += stats.cells
            total_uncompressed += stats.uncompressed_bytes
            if total_file_bytes > _MAX_TOTAL_ARTIFACT_BYTES:
                raise ValueError("analysis result exceeds aggregate artifact byte budget")
            if total_cells > _MAX_TOTAL_ARTIFACT_CELLS:
                raise ValueError(
                    "analysis result exceeds aggregate decoded cell budget")
            if total_uncompressed > _MAX_TOTAL_UNCOMPRESSED_BYTES:
                raise ValueError(
                    "analysis result exceeds aggregate uncompressed byte budget")

        artifacts = {}
        for artifact_manifest, contract in declarations:
            name = artifact_manifest["name"]
            table = _read_verified_parquet(
                root_descriptor,
                artifact_manifest["file"],
                artifact_manifest["sha256"],
                expected_rows=artifact_manifest["n_rows"],
            )
            if len(table) != artifact_manifest["n_rows"]:
                raise ValueError(f"artifact {name!r} row count disagrees with manifest")
            if list(table.columns) != artifact_manifest["columns"]:
                raise ValueError(f"artifact {name!r} columns disagree with manifest")
            if not _has_default_index(table):
                raise ValueError(f"artifact {name!r} Parquet table has a non-default index")
            artifacts[name] = AnalysisArtifact(
                name=name,
                table=table,
                estimand=artifact_manifest["estimand"],
                metadata=artifact_manifest["metadata"],
                table_contract=contract,
            )

        if not _root_still_matches(requested, root_descriptor, root_opened):
            raise ValueError("analysis result root directory changed while loading")
        if dataset is None:
            return LoadedAnalysisResult(dataset_reference=reference, artifacts=artifacts)
        if not isinstance(dataset, AnalysisDataset):
            raise ValueError("dataset must be an AnalysisDataset")
        if dataset.row_ids != reference.row_ids:
            raise ValueError("dataset row_ids do not exactly match saved analysis row order")
        if dataset.group_source != reference.group_source:
            raise ValueError("dataset group_source does not match saved analysis")
        if _canonical_group_codes(dataset) != reference.group_codes:
            raise ValueError(
                "dataset independent-group partition does not match saved analysis")
        return DatasetAnalysisResult(dataset=dataset, artifacts=artifacts)
    finally:
        os.close(root_descriptor)


def _analysis_result_observation(result) -> dict[str, object]:
    if isinstance(result, DatasetAnalysisResult):
        dataset = result.dataset
        n_groups = len(set(_canonical_group_codes(dataset)))
    else:
        dataset = result.dataset_reference
        n_groups = dataset.n_groups
    return {
        "n_rows": len(dataset.row_ids),
        "n_groups": n_groups,
        "artifact_count": len(result.artifacts),
        "shapes": [
            [len(artifact.table), len(artifact.table.columns)]
            for artifact in result.artifacts.values()
        ],
    }


def save_analysis_result(result, out) -> Path:
    """Transactionally publish a contracted result as manifest plus Parquet tables."""
    with automatic_stage("analysis_result.save") as operation:
        path = _save_analysis_result(result, out)
        if operation.active:
            try:
                operation.update(**_analysis_result_observation(result))
            except BaseException:
                pass
        return path


def load_analysis_result(path, *, dataset=None):
    """Load a strict result; reattach only an exactly matching complete dataset."""
    if _SUPPRESS_LOAD_OBSERVATION.get():
        return _load_analysis_result(path, dataset=dataset)
    with automatic_stage(
        "analysis_result.load", {"attached": dataset is not None}
    ) as operation:
        result = _load_analysis_result(path, dataset=dataset)
        if operation.active:
            try:
                operation.update(**_analysis_result_observation(result))
            except BaseException:
                pass
        return result


def _load_analysis_result_for_validation(path, *, dataset):
    token = _SUPPRESS_LOAD_OBSERVATION.set(True)
    try:
        return load_analysis_result(path, dataset=dataset)
    finally:
        _SUPPRESS_LOAD_OBSERVATION.reset(token)


__all__ = [
    "AnalysisDatasetReference", "LoadedAnalysisResult",
    "load_analysis_result", "save_analysis_result",
]
