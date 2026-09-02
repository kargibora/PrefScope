"""Typed loader for reusable encoded feature bundles."""

from __future__ import annotations

import ctypes
from contextvars import ContextVar
import errno
import json
import os
import sys
from pathlib import Path
import re
import shutil
import stat
import uuid

import numpy as np
import pandas as pd

from prefscope.api._lens_publication import _publication_lock, _recover_orphan_backup
from prefscope.core.features import FeatureBatch, _validate_feature_batch_semantics
from prefscope.core.provenance import ordered_dataset_hash
from prefscope.core.redaction import reject_secrets
from prefscope.observability.runtime import automatic_stage


_ROLES = {
    "z_prompt": "prompt",
    "z_a": "response_a",
    "z_b": "response_b",
    "z_diff": "response_difference",
}
_ORIENTATIONS = {
    "z_prompt": "none",
    "z_a": "absolute_a",
    "z_b": "absolute_b",
    "z_diff": "a_minus_b",
}
_SCHEMA2_KEYS = {
    "schema_version",
    "n_rows",
    "feature_width",
    "feature_ids",
    "output_arrays",
    "array_shapes",
    "array_dtypes",
    "metadata_types",
    "roles",
    "orientations",
    "activation_polarity",
    "code_semantics",
    "provenance",
    "dataset_hash",
}
_MAX_ROWS = 1_000_000
_MAX_FEATURES = 1_000_000
_MAX_ARRAYS = 128
_MAX_ARRAY_ELEMENTS = 100_000_000
_MAX_MANIFEST_BYTES = 8 * 1024 * 1024
_MAX_MANIFEST_DEPTH = 64
_MAX_METADATA_COMPRESSED_BYTES = 512 * 1024**2
_MAX_METADATA_COLUMNS = 512
_MAX_METADATA_ROW_GROUPS = 4096
_MAX_METADATA_UNCOMPRESSED_BYTES = 1024**3
_MAX_METADATA_CELLS = 8_000_000
_SUPPRESS_LOAD_OBSERVATION: ContextVar[bool] = ContextVar(
    "prefscope_feature_bundle_suppress_load_observation", default=False
)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"encoded bundle manifest contains non-finite JSON value {value}")


def _parse_json_float(value: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed):
        _reject_json_constant(value)
    return parsed


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    resolved: dict[str, object] = {}
    for key, value in pairs:
        if key in resolved:
            raise ValueError(
                f"encoded bundle manifest contains duplicate JSON key {key!r}"
            )
        resolved[key] = value
    return resolved


def _validate_json_lexical_depth(payload: bytes) -> None:
    depth = 0
    in_string = False
    escaped = False
    for byte in payload:
        if in_string:
            if escaped:
                escaped = False
            elif byte == ord("\\"):
                escaped = True
            elif byte == ord('"'):
                in_string = False
        elif byte == ord('"'):
            in_string = True
        elif byte in {ord("{"), ord("[")}:
            depth += 1
            if depth > _MAX_MANIFEST_DEPTH:
                raise ValueError(
                    "encoded bundle manifest exceeds maximum JSON nesting depth "
                    f"of {_MAX_MANIFEST_DEPTH}"
                )
        elif byte in {ord("}"), ord("]")}:
            depth -= 1


def _validate_json_depth(value: object) -> None:
    stack = [(value, 1)]
    while stack:
        current, depth = stack.pop()
        if depth > _MAX_MANIFEST_DEPTH:
            raise ValueError(
                "encoded bundle manifest exceeds maximum JSON nesting depth "
                f"of {_MAX_MANIFEST_DEPTH}"
            )
        if isinstance(current, dict):
            stack.extend((item, depth + 1) for item in current.values())
        elif isinstance(current, list):
            stack.extend((item, depth + 1) for item in current)


def _read_manifest(descriptor: int) -> dict[str, object]:
    with os.fdopen(os.dup(descriptor), "rb") as stream:
        payload = stream.read(_MAX_MANIFEST_BYTES + 1)
    if len(payload) > _MAX_MANIFEST_BYTES:
        raise ValueError(
            "encoded bundle manifest.json exceeds maximum size "
            f"of {_MAX_MANIFEST_BYTES} bytes"
        )
    _validate_json_lexical_depth(payload)
    try:
        text = payload.decode("utf-8")
        manifest = json.loads(
            text,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
            parse_float=_parse_json_float,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"encoded bundle manifest.json is not valid JSON: {exc.msg}"
        ) from exc
    except UnicodeError as exc:
        raise ValueError("encoded bundle manifest.json must be UTF-8 JSON") from exc
    except RecursionError as exc:
        raise ValueError(
            "encoded bundle manifest exceeds maximum JSON nesting depth"
        ) from exc
    if not isinstance(manifest, dict):
        raise ValueError("encoded bundle manifest.json must contain a JSON object")
    _validate_json_depth(manifest)
    return manifest


def _file_identity(file_stat: os.stat_result) -> tuple[int, int, int]:
    return (file_stat.st_dev, file_stat.st_ino, stat.S_IFMT(file_stat.st_mode))


def _file_fingerprint(file_stat: os.stat_result) -> tuple[int, ...]:
    return (
        file_stat.st_dev,
        file_stat.st_ino,
        stat.S_IFMT(file_stat.st_mode),
        file_stat.st_size,
        file_stat.st_mtime_ns,
        file_stat.st_ctime_ns,
    )


def _path_identity(path: Path) -> tuple[int, int, int] | None:
    try:
        return _file_identity(path.stat(follow_symlinks=False))
    except FileNotFoundError:
        return None


def _open_bundle_root(path) -> tuple[Path, int, tuple[int, int, int]]:
    requested = Path(path).expanduser().absolute()
    if requested.is_symlink():
        raise ValueError(f"encoded bundle path must not be a symlink: {requested}")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(requested, flags)
    except FileNotFoundError:
        raise FileNotFoundError(f"encoded bundle does not exist: {requested}") from None
    except OSError as exc:
        if requested.is_symlink():
            raise ValueError(
                f"encoded bundle path must not be a symlink: {requested}"
            ) from exc
        raise ValueError(
            f"encoded bundle path must be a directory: {requested}"
        ) from exc
    root_stat = os.fstat(descriptor)
    if not stat.S_ISDIR(root_stat.st_mode):
        os.close(descriptor)
        raise ValueError(f"encoded bundle path must be a directory: {requested}")
    return requested, descriptor, _file_identity(root_stat)


def _open_regular_member(root_descriptor: int, name: str) -> int:
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=root_descriptor)
    except OSError as exc:
        raise ValueError(
            f"could not safely open encoded bundle member {name!r}"
        ) from exc
    member_stat = os.fstat(descriptor)
    if not stat.S_ISREG(member_stat.st_mode):
        os.close(descriptor)
        raise ValueError(f"encoded bundle member {name!r} must be a regular file")
    return descriptor


def _regular_member_names(root_descriptor: int) -> set[str]:
    try:
        names = set(os.listdir(root_descriptor))
    except OSError as exc:
        raise ValueError("could not inspect encoded bundle directory") from exc
    for name in names:
        try:
            member_stat = os.stat(name, dir_fd=root_descriptor, follow_symlinks=False)
        except OSError as exc:
            raise ValueError(
                f"could not inspect encoded bundle member {name!r}"
            ) from exc
        if not stat.S_ISREG(member_stat.st_mode):
            kind = (
                "symlink" if stat.S_ISLNK(member_stat.st_mode) else "non-regular file"
            )
            raise ValueError(
                f"encoded bundle member {name!r} must be a regular file, got {kind}"
            )
    return names


def _assert_stable_snapshot(
    root: Path,
    root_descriptor: int,
    root_identity: tuple[int, int, int],
    member_descriptors: dict[str, int],
    expected_files: set[str],
) -> None:
    if _path_identity(root) != root_identity:
        raise ValueError("encoded bundle root changed while it was being loaded")
    try:
        if set(os.listdir(root_descriptor)) != expected_files:
            raise ValueError(
                "encoded bundle members changed while they were being loaded"
            )
        for name, descriptor in member_descriptors.items():
            current = os.stat(name, dir_fd=root_descriptor, follow_symlinks=False)
            if _file_fingerprint(current) != _file_fingerprint(os.fstat(descriptor)):
                raise ValueError(
                    f"encoded bundle member {name!r} changed while it was being loaded"
                )
    except OSError as exc:
        raise ValueError("encoded bundle changed while it was being loaded") from exc


def _preflight_parquet_member(
    descriptor: int,
    name: str,
    *,
    expected_rows: int,
) -> tuple[int, int]:
    import pyarrow.parquet as parquet

    compressed_size = os.fstat(descriptor).st_size
    if compressed_size > _MAX_METADATA_COMPRESSED_BYTES:
        raise ValueError(
            f"encoded metadata compressed bytes exceed {_MAX_METADATA_COMPRESSED_BYTES}"
        )
    with os.fdopen(os.dup(descriptor), "rb") as stream:
        try:
            parquet_metadata = parquet.ParquetFile(stream).metadata
        except Exception as exc:
            raise ValueError(
                f"encoded bundle member {name!r} is not valid Parquet"
            ) from exc
        n_columns = parquet_metadata.num_columns
        n_row_groups = parquet_metadata.num_row_groups
        if n_columns > _MAX_METADATA_COLUMNS:
            raise ValueError(f"encoded metadata columns exceed {_MAX_METADATA_COLUMNS}")
        if n_row_groups > _MAX_METADATA_ROW_GROUPS:
            raise ValueError(
                f"encoded metadata row groups exceed {_MAX_METADATA_ROW_GROUPS}"
            )
        if parquet_metadata.num_rows != expected_rows:
            raise ValueError(f"encoded bundle n_rows does not match {name}")
        if parquet_metadata.num_rows * n_columns > _MAX_METADATA_CELLS:
            raise ValueError(f"encoded metadata cells exceed {_MAX_METADATA_CELLS}")
        try:
            uncompressed_size = sum(
                parquet_metadata.row_group(index).total_byte_size
                for index in range(n_row_groups)
            )
        except Exception as exc:
            raise ValueError(
                f"encoded bundle member {name!r} has invalid Parquet metadata"
            ) from exc
    if uncompressed_size > _MAX_METADATA_UNCOMPRESSED_BYTES:
        raise ValueError(
            "encoded metadata uncompressed bytes exceed "
            f"{_MAX_METADATA_UNCOMPRESSED_BYTES}"
        )
    return compressed_size, uncompressed_size


def _read_parquet_member(descriptor: int, name: str) -> pd.DataFrame:
    import pyarrow.parquet as parquet

    with os.fdopen(os.dup(descriptor), "rb") as stream:
        try:
            return parquet.ParquetFile(stream).read().to_pandas()
        except Exception as exc:
            raise ValueError(
                f"could not read bounded encoded metadata member {name!r}"
            ) from exc


def _validate_array_budget(
    declared: list[str],
    declared_shapes: object,
    *,
    n_rows: int,
    width: int,
) -> dict[str, list[int]]:
    if not isinstance(declared_shapes, dict) or set(declared_shapes) != set(declared):
        raise ValueError("encoded bundle array_shapes must declare every array")
    expected_shape = [n_rows, width]
    total_elements = 0
    for name in declared:
        shape = declared_shapes[name]
        if shape != expected_shape or any(type(value) is not int for value in shape):
            raise ValueError(f"{name} shape disagrees with encoded manifest")
        total_elements += n_rows * width
        if total_elements > _MAX_ARRAY_ELEMENTS:
            raise ValueError(
                f"encoded arrays exceed element budget of {_MAX_ARRAY_ELEMENTS}"
            )
    return declared_shapes


def _load_feature_batch(path, *, arrays=None) -> FeatureBatch:
    """Load and validate an ``encode-dataset`` directory without importing Torch."""
    root, root_descriptor, root_identity = _open_bundle_root(path)
    member_descriptors: dict[str, int] = {}
    try:
        actual_files = _regular_member_names(root_descriptor)
        core_files = {"manifest.json", "meta.parquet", "battles.parquet"}
        if not core_files <= actual_files:
            raise FileNotFoundError(
                "encoded bundle must contain manifest.json, meta.parquet, and "
                f"battles.parquet: {root}"
            )
        member_descriptors["manifest.json"] = _open_regular_member(
            root_descriptor, "manifest.json"
        )
        manifest = _read_manifest(member_descriptors["manifest.json"])
        schema_version = manifest.get("schema_version")
        if type(schema_version) is not int or schema_version not in {1, 2}:
            raise ValueError(f"unsupported encoded bundle schema {schema_version!r}")
        if schema_version == 2 and set(manifest) != _SCHEMA2_KEYS:
            raise ValueError(
                "schema-2 encoded bundle manifest keys must match exactly: "
                f"missing={sorted(_SCHEMA2_KEYS - set(manifest))}, "
                f"extra={sorted(set(manifest) - _SCHEMA2_KEYS)}"
            )
        declared = manifest.get("output_arrays")
        if (
            not isinstance(declared, list)
            or not declared
            or any(
                not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", name)
                for name in declared
            )
            or len(set(declared)) != len(declared)
        ):
            raise ValueError("encoded bundle must declare unique safe output_arrays")
        if len(declared) > _MAX_ARRAYS:
            raise ValueError(f"encoded bundle declares more than {_MAX_ARRAYS} arrays")
        expected_files = {
            "manifest.json",
            "meta.parquet",
            "battles.parquet",
            *(f"{name}.npy" for name in declared),
        }
        if actual_files != expected_files:
            raise ValueError(
                "encoded bundle contains missing or undeclared artifacts: "
                f"missing={sorted(expected_files - actual_files)}, "
                f"extra={sorted(actual_files - expected_files)}"
            )
        for name in sorted(expected_files - {"manifest.json"}):
            member_descriptors[name] = _open_regular_member(root_descriptor, name)

        selected = tuple(declared if arrays is None else arrays)
        if not selected or len(set(selected)) != len(selected):
            raise ValueError("arrays must select at least one unique feature view")
        unknown = set(selected) - set(declared)
        if unknown:
            raise ValueError(
                f"arrays are not declared by the bundle: {sorted(unknown)}"
            )
        if schema_version == 1:
            unsupported = set(selected) - set(_ROLES)
            if unsupported:
                raise ValueError(
                    f"unsupported encoded feature views: {sorted(unsupported)}"
                )
            roles = {name: _ROLES[name] for name in selected}
            orientations = {name: _ORIENTATIONS[name] for name in selected}
        else:
            declared_roles = manifest.get("roles")
            declared_orientations = manifest.get("orientations")
            if (
                not isinstance(declared_roles, dict)
                or set(declared_roles) != set(declared)
                or not isinstance(declared_orientations, dict)
                or set(declared_orientations) != set(declared)
                or any(
                    not isinstance(value, str) or not value.strip()
                    for value in declared_roles.values()
                )
                or any(
                    not isinstance(value, str) or not value.strip()
                    for value in declared_orientations.values()
                )
            ):
                raise ValueError(
                    "schema-2 encoded bundle roles/orientations must name every array "
                    "with non-empty strings"
                )
            roles = {name: declared_roles[name] for name in selected}
            orientations = {name: declared_orientations[name] for name in selected}
        declared_dtypes = (
            {name: "float32" for name in declared}
            if schema_version == 1
            else manifest.get("array_dtypes")
        )
        if (
            not isinstance(declared_dtypes, dict)
            or set(declared_dtypes) != set(declared)
            or any(
                not isinstance(value, str) or value not in {"float32", "bool"}
                for value in declared_dtypes.values()
            )
        ):
            raise ValueError(
                "encoded bundle array_dtypes must declare float32 or bool for every array"
            )
        n_rows = manifest.get("n_rows")
        width = (
            manifest.get("m_total")
            if schema_version == 1
            else manifest.get("feature_width")
        )
        if type(n_rows) is not int or not 0 < n_rows <= _MAX_ROWS:
            raise ValueError(f"encoded bundle n_rows must be between 1 and {_MAX_ROWS}")
        if type(width) is not int or not 0 < width <= _MAX_FEATURES:
            field = "m_total" if schema_version == 1 else "feature_width"
            raise ValueError(
                f"encoded bundle {field} must be between 1 and {_MAX_FEATURES}"
            )
        declared_shapes = _validate_array_budget(
            declared, manifest.get("array_shapes"), n_rows=n_rows, width=width
        )

        _, meta_uncompressed = _preflight_parquet_member(
            member_descriptors["meta.parquet"], "meta.parquet", expected_rows=n_rows
        )
        _, battles_uncompressed = _preflight_parquet_member(
            member_descriptors["battles.parquet"],
            "battles.parquet",
            expected_rows=n_rows,
        )
        if meta_uncompressed + battles_uncompressed > _MAX_METADATA_UNCOMPRESSED_BYTES:
            raise ValueError(
                "encoded metadata aggregate uncompressed bytes exceed "
                f"{_MAX_METADATA_UNCOMPRESSED_BYTES}"
            )
        metadata = _read_parquet_member(
            member_descriptors["meta.parquet"], "meta.parquet"
        )
        battles = _read_parquet_member(
            member_descriptors["battles.parquet"], "battles.parquet"
        )
        try:
            metadata_matches = metadata.equals(battles)
        finally:
            del battles
        if not metadata_matches:
            raise ValueError(
                "meta.parquet and battles.parquet must be exactly identical"
            )
        if len(metadata) != n_rows:
            raise ValueError("encoded bundle n_rows does not match meta.parquet")
        if schema_version == 1:
            feature_ids = tuple(range(width))
        else:
            from prefscope.core.features import validate_feature_ids

            raw_feature_ids = manifest.get("feature_ids")
            if not isinstance(raw_feature_ids, list):
                raise ValueError("schema-2 encoded bundle must declare feature_ids")
            feature_ids = validate_feature_ids(raw_feature_ids, width=width)
        id_column = "battle_id" if "battle_id" in metadata else "row_id"
        if id_column not in metadata:
            raise ValueError("encoded metadata needs battle_id or row_id")
        if (
            metadata[id_column].isna().any()
            or metadata[id_column].astype(str).duplicated().any()
        ):
            raise ValueError(
                f"encoded metadata {id_column} values must be unique and nonmissing"
            )
        loaded = {}
        for name in declared:
            if schema_version == 1 and name not in _ROLES:
                raise ValueError(f"unsupported encoded feature view: {name}")
            array_descriptor = member_descriptors[f"{name}.npy"]
            expected_dtype = np.dtype(declared_dtypes[name])
            maximum_file_size = n_rows * width * expected_dtype.itemsize + 10_128
            if os.fstat(array_descriptor).st_size > maximum_file_size:
                raise ValueError(
                    f"encoded bundle member {name!r} exceeds its array budget"
                )
            with os.fdopen(os.dup(array_descriptor), "rb") as stream:
                try:
                    matrix = np.load(stream, allow_pickle=False, max_header_size=10_000)
                    consumed_bytes = stream.tell()
                except Exception as exc:
                    raise ValueError(
                        f"encoded bundle member {name!r} is not a valid NPY array"
                    ) from exc
            if consumed_bytes != os.fstat(array_descriptor).st_size:
                raise ValueError(f"encoded bundle member {name!r} has trailing bytes")
            if matrix.shape != (n_rows, width):
                raise ValueError(
                    f"{name} has shape {matrix.shape}; expected {(n_rows, width)}"
                )
            if declared_shapes[name] != list(matrix.shape):
                raise ValueError(f"{name} shape disagrees with encoded manifest")
            if matrix.dtype != expected_dtype:
                raise ValueError(
                    f"{name} must contain canonical finite {expected_dtype} values"
                )
            loaded[name] = matrix
        try:
            observed_hash = ordered_dataset_hash(metadata, loaded)
        except ValueError as exc:
            raise ValueError(
                f"encoded bundle contains invalid array values: {exc}"
            ) from exc
        if manifest.get("dataset_hash") != observed_hash:
            raise ValueError(
                "encoded bundle dataset_hash does not match metadata and arrays"
            )
        values = {name: loaded[name] for name in selected}
        raw_metadata_columns = [
            str(column) for column in metadata.columns if column != id_column
        ]
        if schema_version == 2:
            metadata_types = manifest.get("metadata_types")
            if (
                not isinstance(metadata_types, dict)
                or set(metadata_types) != set(raw_metadata_columns)
                or any(
                    not isinstance(value, str)
                    or value not in {"null", "str", "bool", "int", "float"}
                    for value in metadata_types.values()
                )
            ):
                raise ValueError(
                    "schema-2 metadata_types must declare every metadata column"
                )
        else:
            metadata_types = {name: None for name in raw_metadata_columns}
        metadata_columns = {}
        for column in raw_metadata_columns:
            kind = metadata_types[column]
            wire_values = metadata[column].tolist()
            nonmissing = [
                value
                for value in wire_values
                if value is not None
                and not (not isinstance(value, str) and pd.isna(value))
            ]
            valid_kind = (
                (kind is None)
                or (kind == "null" and not nonmissing)
                or (
                    kind == "str"
                    and all(isinstance(value, str) for value in nonmissing)
                )
                or (
                    kind == "bool"
                    and all(isinstance(value, (bool, np.bool_)) for value in nonmissing)
                )
                or (
                    kind == "int"
                    and all(
                        isinstance(value, (int, np.integer))
                        and not isinstance(value, (bool, np.bool_))
                        for value in nonmissing
                    )
                )
                or (
                    kind == "float"
                    and all(
                        isinstance(value, (float, np.floating)) for value in nonmissing
                    )
                )
            )
            if not valid_kind:
                raise ValueError(
                    f"metadata column {column!r} disagrees with metadata_types"
                )
            restored = []
            for value in wire_values:
                if value is None or (not isinstance(value, str) and pd.isna(value)):
                    restored.append(None)
                elif kind == "str":
                    restored.append(str(value))
                elif kind == "bool":
                    restored.append(bool(value))
                elif kind == "int":
                    restored.append(int(value))
                elif kind == "float":
                    restored.append(float(value))
                else:
                    restored.append(value)
            metadata_columns[column] = tuple(restored)
        source_provenance = (
            manifest.get("provenance", {}) if schema_version == 2 else {}
        )
        if not isinstance(source_provenance, dict):
            raise ValueError("encoded bundle provenance must be a mapping")
        if schema_version == 2:
            reject_secrets(source_provenance, where="encoded bundle provenance")
            _validate_feature_batch_semantics(
                declared,
                activation_polarity=manifest.get("activation_polarity"),
                code_semantics=manifest.get("code_semantics"),
                provenance=source_provenance,
            )
        selected_provenance = dict(source_provenance)
        if "views" in selected_provenance:
            descriptors = selected_provenance["views"]
            selected_provenance["views"] = {
                name: descriptors[name] for name in selected if name in descriptors
            }
        result = FeatureBatch(
            row_ids=tuple(metadata[id_column].astype(str)),
            arrays=values,
            roles=roles,
            orientations=orientations,
            feature_ids=feature_ids,
            metadata=metadata_columns,
            activation_polarity=manifest.get("activation_polarity", "unknown"),
            code_semantics=manifest.get("code_semantics", "custom"),
            provenance={
                **selected_provenance,
                "encoded_bundle": {
                    key: value for key, value in manifest.items() if key != "provenance"
                },
            },
        )
        _assert_stable_snapshot(
            root, root_descriptor, root_identity, member_descriptors, expected_files
        )
        return result
    finally:
        for descriptor in member_descriptors.values():
            os.close(descriptor)
        os.close(root_descriptor)


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically rename a directory, failing if the destination exists."""
    libc = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    if sys.platform == "darwin":
        rename = getattr(libc, "renamex_np", None)
        if rename is not None:
            rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
            rename.restype = ctypes.c_int
            result = rename(source_bytes, destination_bytes, 0x00000004)
        else:
            result = -1
            ctypes.set_errno(errno.ENOSYS)
    elif sys.platform.startswith("linux"):
        rename = getattr(libc, "renameat2", None)
        if rename is not None:
            rename.argtypes = [
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_uint,
            ]
            rename.restype = ctypes.c_int
            result = rename(-100, source_bytes, -100, destination_bytes, 1)
        else:
            result = -1
            ctypes.set_errno(errno.ENOSYS)
    elif os.name == "nt":
        os.rename(source, destination)
        return
    else:
        result = -1
        ctypes.set_errno(errno.ENOSYS)
    if result == 0:
        return
    error = ctypes.get_errno()
    if error in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(
            error, "feature bundle destination appeared during staging", destination
        )
    raise OSError(error, os.strerror(error), destination)


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _cleanup_staging(staging: Path) -> None:
    if staging.is_symlink():
        staging.unlink()
    elif staging.exists():
        shutil.rmtree(staging, ignore_errors=True)


def _restore_backup(root: Path, backup: Path, *, backup_created: bool) -> bool:
    if not backup_created or not backup.exists() or root.exists() or root.is_symlink():
        return backup_created
    os.replace(backup, root)
    _fsync_directory(root.parent)
    return False


def _quarantine_unexpected_destination(root: Path) -> Path | None:
    if not root.exists() and not root.is_symlink():
        return None
    quarantine = root.parent / f".{root.name}.unexpected-{uuid.uuid4().hex}"
    os.replace(root, quarantine)
    _fsync_directory(root.parent)
    return quarantine


def _validate_managed_destination(root: Path) -> None:
    try:
        _load_feature_batch_for_validation(root)
    except Exception as exc:
        raise ValueError(
            "overwrite=True may replace only an existing valid managed encoded bundle: "
            f"{root}"
        ) from exc


def _save_feature_batch(batch: FeatureBatch, path, *, overwrite: bool = False) -> Path:
    """Transactionally save any aligned ``FeatureBatch`` as a schema-2 bundle."""
    if not isinstance(batch, FeatureBatch):
        raise ValueError("batch must be a FeatureBatch")
    if not isinstance(overwrite, bool):
        raise ValueError("overwrite must be boolean")
    if len(batch.row_ids) > _MAX_ROWS:
        raise ValueError(f"feature batch rows exceed {_MAX_ROWS}")
    requested = Path(path).expanduser()
    if requested.name in {"", ".", ".."}:
        raise ValueError("feature bundle destination must name a directory")
    parent = requested.parent
    if parent.is_symlink():
        raise ValueError("feature bundle destination parent must not be a symlink")
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink():
        raise ValueError("feature bundle destination parent must not be a symlink")
    # Do not resolve through the final parent or destination component. Their
    # lstat-based symlink checks must apply to the path the caller supplied.
    root = parent.absolute() / requested.name
    if root.is_symlink() or (root.exists() and not root.is_dir()):
        raise ValueError(
            "feature bundle destination must be a directory path, not a file/symlink"
        )
    if "row_id" in batch.metadata or "battle_id" in batch.metadata:
        raise ValueError("feature metadata must not redefine row_id or battle_id")
    for name in batch.arrays:
        if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
            raise ValueError(f"unsafe feature array name: {name!r}")

    with _publication_lock(root):
        _recover_orphan_backup(root)
        initial_identity = _path_identity(root)
        if root.is_symlink() or (initial_identity is not None and not root.is_dir()):
            raise ValueError(
                "feature bundle destination must be a directory path, not a file/symlink"
            )
        if initial_identity is not None:
            if not overwrite:
                raise FileExistsError(
                    f"feature bundle destination already exists: {root}"
                )
            _validate_managed_destination(root)

        staging = root.parent / f".{root.name}.tmp-{uuid.uuid4().hex}"
        backup = root.parent / f".{root.name}.bak-{uuid.uuid4().hex}"
        backup_created = False
        published = False
        try:
            staging.mkdir()
            _fsync_directory(root.parent)
            metadata_types = {}
            metadata_data = {"row_id": list(batch.row_ids)}
            for name, values in batch.metadata.items():
                nonmissing = [value for value in values if value is not None]
                if not nonmissing:
                    kind = "null"
                    column = list(values)
                elif isinstance(nonmissing[0], str):
                    kind = "str"
                    column = pd.array(values, dtype="string")
                elif isinstance(nonmissing[0], bool):
                    kind = "bool"
                    column = pd.array(values, dtype="boolean")
                elif isinstance(nonmissing[0], int):
                    kind = "int"
                    column = pd.array(values, dtype="Int64")
                else:
                    kind = "float"
                    column = pd.array(values, dtype="Float64")
                metadata_types[name] = kind
                metadata_data[name] = column
            metadata = pd.DataFrame(metadata_data)
            for file_name in ("meta.parquet", "battles.parquet"):
                artifact = staging / file_name
                metadata.to_parquet(artifact, index=False)
                _fsync_file(artifact)

            arrays = {}
            array_dtypes = {}
            for name, values in batch.arrays.items():
                source = np.asarray(values)
                dtype = np.dtype(bool) if source.dtype == bool else np.dtype(np.float32)
                array = np.asarray(source, dtype=dtype)
                artifact = staging / f"{name}.npy"
                np.save(artifact, array, allow_pickle=False)
                _fsync_file(artifact)
                arrays[name] = array
                array_dtypes[name] = dtype.name

            manifest = {
                "schema_version": 2,
                "n_rows": len(batch.row_ids),
                "feature_width": len(batch.feature_ids),
                "feature_ids": list(batch.feature_ids),
                "output_arrays": list(batch.arrays),
                "array_shapes": {
                    name: list(values.shape) for name, values in arrays.items()
                },
                "array_dtypes": array_dtypes,
                "metadata_types": metadata_types,
                "roles": dict(batch.roles),
                "orientations": dict(batch.orientations),
                "activation_polarity": batch.activation_polarity,
                "code_semantics": batch.code_semantics,
                "provenance": dict(batch.provenance),
                "dataset_hash": ordered_dataset_hash(metadata, arrays),
            }
            manifest_path = staging / "manifest.json"
            manifest_path.write_text(
                json.dumps(manifest, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            _fsync_file(manifest_path)
            _fsync_directory(staging)
            _load_feature_batch_for_validation(staging)

            if initial_identity is None:
                # No-overwrite publication must never move, remove, or replace a path
                # that another actor created while this writer staged its bundle.
                if _path_identity(root) is not None:
                    raise FileExistsError(
                        f"feature bundle destination appeared during staging: {root}"
                    )
                _rename_noreplace(staging, root)
                _fsync_directory(root.parent)
                published = True
            else:
                # Validation can take long enough for an uncooperative writer to swap
                # the destination. The exact directory validated at lock acquisition
                # is the only directory that may become our recovery backup.
                _validate_managed_destination(root)
                if _path_identity(root) != initial_identity:
                    raise RuntimeError(
                        "feature bundle destination changed during staging; "
                        "the concurrent destination was left untouched"
                    )
                os.replace(root, backup)
                backup_created = True
                _fsync_directory(root.parent)
                try:
                    _rename_noreplace(staging, root)
                    _fsync_directory(root.parent)
                    published = True
                except BaseException:
                    # Never move or delete a destination created by another actor.
                    # If one appeared, retain the backup for explicit next-run recovery.
                    backup_created = _restore_backup(
                        root, backup, backup_created=backup_created
                    )
                    raise

            if backup_created:
                shutil.rmtree(backup)
                backup_created = False
                _fsync_directory(root.parent)
        finally:
            _cleanup_staging(staging)
            # A backup is never removed here. On an exceptional publication path it
            # can be the only trustworthy copy of the previous destination.
            if not published:
                _restore_backup(root, backup, backup_created=backup_created)
    return root


def _requested_view_observation(arrays) -> dict[str, object]:
    """Return a bounded count without inspecting caller-supplied view values."""
    if type(arrays) not in {list, tuple} or not 0 < len(arrays) <= _MAX_ARRAYS:
        return {}
    return {"requested_view_count": len(arrays)}


def _feature_batch_observation(batch: FeatureBatch) -> dict[str, object]:
    return {
        "n_rows": len(batch.row_ids),
        "n_features": len(batch.feature_ids),
        "n_arrays": len(batch.arrays),
        "shapes": [list(values.shape) for values in batch.arrays.values()],
    }


def load_feature_batch(path, *, arrays=None) -> FeatureBatch:
    """Load and validate an ``encode-dataset`` directory without importing Torch."""
    if _SUPPRESS_LOAD_OBSERVATION.get():
        return _load_feature_batch(path, arrays=arrays)
    with automatic_stage(
        "feature_bundle.load", _requested_view_observation(arrays)
    ) as operation:
        result = _load_feature_batch(path, arrays=arrays)
        if operation.active:
            try:
                operation.update(**_feature_batch_observation(result))
            except BaseException:
                pass
        return result


def _load_feature_batch_for_validation(path) -> FeatureBatch:
    token = _SUPPRESS_LOAD_OBSERVATION.set(True)
    try:
        return load_feature_batch(path)
    finally:
        _SUPPRESS_LOAD_OBSERVATION.reset(token)


def save_feature_batch(batch: FeatureBatch, path, *, overwrite: bool = False) -> Path:
    """Transactionally save any aligned ``FeatureBatch`` as a schema-2 bundle."""
    with automatic_stage(
        "feature_bundle.save",
        ({"overwrite": overwrite} if isinstance(overwrite, bool) else {}),
    ) as operation:
        result = _save_feature_batch(batch, path, overwrite=overwrite)
        if operation.active:
            try:
                operation.update(**_feature_batch_observation(batch))
            except BaseException:
                pass
        return result


__all__ = ["load_feature_batch", "save_feature_batch"]
