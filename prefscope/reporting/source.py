"""Validated lazy access to schema-2 encoded feature bundles.

This Torch-free module never constructs a ``FeatureBatch``. Feature arrays are
read-only memory maps, so opening does not copy them into RAM. A memory map is a
live filesystem view: another process can still change the backing file after
``open`` returns. Callers that need an immutable snapshot must copy the bundle
to immutable storage or copy the selected arrays themselves.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from types import MappingProxyType
from typing import BinaryIO, Iterator, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np
import pandas as pd

from prefscope.core.features import validate_feature_ids
from prefscope.core.provenance import ordered_dataset_hash
from prefscope.core.redaction import reject_secrets
from prefscope.core.representation import validate_portable_mapping, validate_row_ids
from prefscope.observability.runtime import automatic_stage


DEFAULT_MAX_ROWS = 1_000_000
DEFAULT_MAX_FEATURES = 1_000_000
DEFAULT_MAX_ELEMENTS = 1_000_000_000
DEFAULT_MAX_MANIFEST_BYTES = 8 * 1024 * 1024
DEFAULT_MAX_METADATA_FILE_BYTES = 512 * 1024 ** 2
DEFAULT_MAX_METADATA_COMPRESSED_BYTES = 1024 * 1024 ** 2
DEFAULT_MAX_METADATA_COLUMNS = 512
DEFAULT_MAX_METADATA_CELLS = 8_000_000
DEFAULT_MAX_METADATA_ROW_GROUPS = 4_096
DEFAULT_MAX_METADATA_UNCOMPRESSED_BYTES = 1024 * 1024 ** 2
_VALIDATION_CHUNK_ELEMENTS = 1_048_576
_ARRAY_NAME = re.compile(r"[a-z][a-z0-9_]*")
_CANONICAL_DTYPES = {"float32": np.dtype(np.float32), "bool": np.dtype(bool)}
_METADATA_TYPES = {"null", "str", "bool", "int", "float"}
_MANIFEST_KEYS = {
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
_OPEN_TOKEN = object()


def _is_finite_in_bounded_chunks(
    values: np.ndarray,
    *,
    chunk_elements: int = _VALIDATION_CHUNK_ELEMENTS,
) -> bool:
    """Check finiteness without allocating a whole-array boolean mask."""
    n_rows, width = values.shape
    rows_per_chunk = max(1, chunk_elements // width)
    for row_start in range(0, n_rows, rows_per_chunk):
        row_stop = min(row_start + rows_per_chunk, n_rows)
        if (row_stop - row_start) * width <= chunk_elements:
            pieces = (values[row_start:row_stop],)
        else:
            pieces = (
                values[row_start:row_stop, column_start:column_start + chunk_elements]
                for column_start in range(0, width, chunk_elements)
            )
        if any(not np.isfinite(piece).all() for piece in pieces):
            return False
    return True


@dataclass(frozen=True)
class FeatureChunk:
    """One bounded, exactly aligned selection from a :class:`FeatureSource`."""

    row_ids: tuple[str, ...]
    positions: range | tuple[int, ...]
    arrays: Mapping[str, np.ndarray]
    metadata: Mapping[str, tuple[object, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        positions = tuple(self.positions)
        if len(positions) != len(self.row_ids):
            raise ValueError("feature chunk positions must match row_ids")
        if len(set(positions)) != len(positions) or any(
            type(position) is not int or position < 0 for position in positions
        ):
            raise ValueError("feature chunk positions must be unique nonnegative integers")
        if isinstance(self.positions, range) and self.positions.step != 1:
            raise ValueError("contiguous feature chunk ranges must have step one")

        raw_arrays = dict(self.arrays)
        if not raw_arrays:
            raise ValueError("feature chunk must contain at least one view")
        checked: dict[str, np.ndarray] = {}
        widths = set()
        for name, values in raw_arrays.items():
            if not isinstance(name, str) or not name:
                raise ValueError("feature chunk view names must be non-empty strings")
            source = values if isinstance(values, np.ndarray) else np.asarray(values)
            if source.ndim != 2 or source.shape[0] != len(self.row_ids):
                raise ValueError(
                    f"feature chunk view {name!r} must have {len(self.row_ids)} rows")
            if (
                source.shape[1] <= 0
                or not (source.dtype == bool or np.issubdtype(source.dtype, np.number))
                or np.issubdtype(source.dtype, np.complexfloating)
            ):
                raise ValueError(
                    "feature chunk arrays must be non-empty real numeric/boolean matrices")
            if not _is_finite_in_bounded_chunks(source):
                raise ValueError("feature chunk arrays must contain only finite values")
            if source.flags.writeable:
                raise ValueError("feature chunk arrays must be read-only")
            widths.add(int(source.shape[1]))
            checked[name] = source
        if len(widths) != 1:
            raise ValueError("feature chunk arrays must have equal feature widths")

        metadata = dict(self.metadata)
        for name, values in metadata.items():
            if not isinstance(name, str) or not name or len(values) != len(self.row_ids):
                raise ValueError(
                    "feature chunk metadata must use non-empty names and align to rows")
            metadata[name] = tuple(values)
        object.__setattr__(self, "arrays", MappingProxyType(checked))
        object.__setattr__(self, "metadata", MappingProxyType(metadata))

    @property
    def row_slice(self) -> slice | None:
        """Return the source slice for a contiguous chunk, otherwise ``None``."""
        if isinstance(self.positions, range):
            return slice(self.positions.start, self.positions.stop)
        if not self.positions:
            return None
        start = self.positions[0]
        if self.positions == tuple(range(start, start + len(self.positions))):
            return slice(start, start + len(self.positions))
        return None

    @property
    def start(self) -> int:
        if not self.row_ids:
            raise ValueError("empty feature chunks have no start position")
        return self.positions.start if isinstance(self.positions, range) else self.positions[0]

    @property
    def stop(self) -> int:
        row_slice = self.row_slice
        if row_slice is None:
            raise ValueError("noncontiguous feature chunks have no single stop position")
        return int(row_slice.stop)

    def array(self, view: str) -> np.ndarray:
        """Return one requested read-only array from the chunk."""
        try:
            return self.arrays[view]
        except KeyError:
            available = ", ".join(self.arrays)
            raise ValueError(
                f"feature chunk has no view {view!r}; available: {available}") from None


@runtime_checkable
class FeatureSource(Protocol):
    """Torch-free protocol for bounded, aligned feature access."""

    @property
    def row_ids(self) -> tuple[str, ...]: ...

    @property
    def feature_ids(self) -> tuple[int, ...]: ...

    @property
    def view_names(self) -> tuple[str, ...]: ...

    @property
    def roles(self) -> Mapping[str, str]: ...

    @property
    def orientations(self) -> Mapping[str, str]: ...

    @property
    def activation_polarities(self) -> Mapping[str, str]: ...

    @property
    def code_semantics_by_view(self) -> Mapping[str, str]: ...

    @property
    def metadata(self) -> Mapping[str, tuple[object, ...]]: ...

    @property
    def provenance(self) -> Mapping[str, object]: ...

    @property
    def n_rows(self) -> int: ...

    @property
    def n_features(self) -> int: ...

    def array(self, view: str) -> np.ndarray: ...

    def assert_row_ids(self, row_ids: Sequence[object]) -> None: ...

    def row_positions(self, row_ids: Sequence[object]) -> tuple[int, ...]: ...

    def iter_chunks(
        self,
        chunk_size: int,
        *,
        views: Sequence[str] | None = None,
        row_ids: Sequence[object] | None = None,
    ) -> Iterator[FeatureChunk]: ...


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"feature bundle manifest contains non-finite JSON value {value}")


def _parse_json_float(value: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed):
        _reject_json_constant(value)
    return parsed


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    resolved: dict[str, object] = {}
    for key, value in pairs:
        if key in resolved:
            raise ValueError(f"feature bundle manifest contains duplicate JSON key {key!r}")
        resolved[key] = value
    return resolved


def _read_manifest(handle: BinaryIO, *, max_bytes: int) -> dict[str, object]:
    try:
        payload_bytes = handle.read(max_bytes + 1)
        if len(payload_bytes) > max_bytes:
            raise ValueError(
                f"feature bundle manifest exceeds max_manifest_bytes {max_bytes}")
        text = payload_bytes.decode("utf-8")
        payload = json.loads(
            text,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
            parse_float=_parse_json_float,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"feature bundle manifest.json is not valid JSON: {exc.msg}") from exc
    except UnicodeError as exc:
        raise ValueError("feature bundle manifest.json must be UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("feature bundle manifest.json must contain a JSON object")
    return payload


def _stat_fingerprint(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _open_regular(path: Path) -> tuple[BinaryIO, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"feature bundle member is not a safe regular file: {path.name}") from exc
    handle = os.fdopen(descriptor, "rb")
    try:
        initial = os.fstat(handle.fileno())
        path_stat = path.lstat()
        if (
            not stat.S_ISREG(initial.st_mode)
            or not stat.S_ISREG(path_stat.st_mode)
            or _stat_fingerprint(initial) != _stat_fingerprint(path_stat)
        ):
            raise ValueError(
                f"feature bundle member is not a stable regular file: {path.name}")
    except Exception:
        handle.close()
        raise
    return handle, initial


def _verify_open_file(
    path: Path,
    handle: BinaryIO,
    initial: os.stat_result,
) -> None:
    try:
        current = os.fstat(handle.fileno())
        path_stat = path.lstat()
    except OSError as exc:
        raise ValueError(
            f"feature bundle member changed while opening: {path.name}") from exc
    if (
        not stat.S_ISREG(current.st_mode)
        or not stat.S_ISREG(path_stat.st_mode)
        or _stat_fingerprint(current) != _stat_fingerprint(initial)
        or _stat_fingerprint(path_stat) != _stat_fingerprint(initial)
    ):
        raise ValueError(f"feature bundle member changed while opening: {path.name}")


def _scan_regular_members(root: Path) -> dict[str, tuple[int, ...]]:
    members: dict[str, tuple[int, ...]] = {}
    try:
        entries = list(os.scandir(root))
    except OSError as exc:
        raise ValueError(f"could not inspect feature bundle directory: {root}") from exc
    for entry in entries:
        try:
            member_stat = entry.stat(follow_symlinks=False)
        except OSError as exc:
            raise ValueError(
                f"could not inspect feature bundle member: {entry.name}") from exc
        if not stat.S_ISREG(member_stat.st_mode):
            kind = "symlink" if stat.S_ISLNK(member_stat.st_mode) else "non-regular file"
            raise ValueError(
                f"feature bundle member {entry.name!r} must be a regular file, got {kind}")
        members[entry.name] = _stat_fingerprint(member_stat)
    return members


def _verify_member_inventory(
    root: Path,
    expected: Mapping[str, tuple[int, ...]],
) -> None:
    observed = _scan_regular_members(root)
    if observed.keys() != expected.keys() or any(
        observed[name] != fingerprint for name, fingerprint in expected.items()
    ):
        raise ValueError("feature bundle members changed while opening")


def _stream_digest(handle: BinaryIO, *, chunk_bytes: int = 1_048_576) -> str:
    handle.seek(0)
    digest = hashlib.sha256()
    while True:
        chunk = handle.read(chunk_bytes)
        if not chunk:
            break
        digest.update(chunk)
    handle.seek(0)
    return digest.hexdigest()


def _read_verified_metadata_twins(
    metadata_path: Path,
    battles_path: Path,
    *,
    expected_rows: int,
    max_file_bytes: int,
    max_compressed_bytes: int,
    max_columns: int,
    max_cells: int,
    max_row_groups: int,
    max_uncompressed_bytes: int,
) -> pd.DataFrame:
    """Preflight identical Parquet twins, then decode only the canonical copy."""
    metadata_handle, metadata_initial = _open_regular(metadata_path)
    try:
        battles_handle, battles_initial = _open_regular(battles_path)
    except Exception:
        metadata_handle.close()
        raise
    try:
        if max(metadata_initial.st_size, battles_initial.st_size) > max_file_bytes:
            raise ValueError(
                "one metadata Parquet file exceeds configured "
                f"max_metadata_file_bytes {max_file_bytes}")
        compressed_bytes = metadata_initial.st_size + battles_initial.st_size
        if compressed_bytes > max_compressed_bytes:
            raise ValueError(
                "metadata Parquet twins exceed configured aggregate "
                f"max_metadata_compressed_bytes {max_compressed_bytes}")
        metadata_digest = _stream_digest(metadata_handle)
        battles_digest = _stream_digest(battles_handle)
        if (
            metadata_initial.st_size != battles_initial.st_size
            or metadata_digest != battles_digest
        ):
            raise ValueError(
                "meta.parquet and battles.parquet must be byte-for-byte identical")

        try:
            import pyarrow.parquet as parquet

            parquet_file = parquet.ParquetFile(metadata_handle)
            parquet_metadata = parquet_file.metadata
            num_columns = int(parquet_metadata.num_columns)
            num_row_groups = int(parquet_metadata.num_row_groups)
            num_rows = int(parquet_metadata.num_rows)
            del parquet_file
        except Exception as exc:
            raise ValueError(
                "feature bundle metadata parquet is unreadable: meta.parquet") from exc

        if num_rows != expected_rows:
            raise ValueError(
                "encoded bundle n_rows does not match metadata Parquet footer")
        if num_columns > max_columns:
            raise ValueError(
                "metadata Parquet exceeds configured "
                f"max_metadata_columns {max_columns}")
        metadata_cells = num_rows * num_columns
        if metadata_cells > max_cells:
            raise ValueError(
                f"metadata Parquet has {metadata_cells} cells, exceeding configured "
                f"max_metadata_cells {max_cells}")
        if num_row_groups > max_row_groups:
            raise ValueError(
                "metadata Parquet exceeds configured "
                f"max_metadata_row_groups {max_row_groups}")
        try:
            uncompressed_bytes = sum(
                int(parquet_metadata.row_group(group).column(column).total_uncompressed_size)
                for group in range(num_row_groups)
                for column in range(num_columns)
            )
        except Exception as exc:
            raise ValueError(
                "feature bundle metadata parquet footer is invalid") from exc
        if uncompressed_bytes < 0:
            raise ValueError("feature bundle metadata parquet footer is invalid")
        if 2 * uncompressed_bytes > max_uncompressed_bytes:
            raise ValueError(
                "metadata Parquet twins exceed configured "
                f"max_metadata_uncompressed_bytes {max_uncompressed_bytes}")

        metadata_handle.seek(0)
        try:
            frame = pd.read_parquet(metadata_handle)
        except Exception as exc:
            raise ValueError(
                "feature bundle metadata parquet is unreadable: meta.parquet") from exc
        _verify_open_file(metadata_path, metadata_handle, metadata_initial)
        _verify_open_file(battles_path, battles_handle, battles_initial)
        return frame
    finally:
        metadata_handle.close()
        battles_handle.close()


def _open_npy_memmap(
    path: Path,
    *,
    expected_shape: tuple[int, int],
    expected_dtype: np.dtype,
) -> tuple[np.memmap, BinaryIO, os.stat_result]:
    handle, initial = _open_regular(path)
    try:
        try:
            version = np.lib.format.read_magic(handle)
            if version == (1, 0):
                shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(
                    handle, max_header_size=10_000)
            elif version == (2, 0):
                shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(
                    handle, max_header_size=10_000)
            else:
                raise ValueError(f"unsupported .npy format version {version!r}")
        except (EOFError, OSError, ValueError) as exc:
            raise ValueError(
                f"could not read feature view {path.stem!r} as a canonical .npy file") from exc
        if shape != expected_shape:
            raise ValueError(
                f"feature view {path.stem!r} has shape {shape}; expected {expected_shape}")
        if np.dtype(dtype) != expected_dtype:
            raise ValueError(
                f"feature view {path.stem!r} has dtype {dtype}; "
                f"expected canonical {expected_dtype}")
        if fortran_order:
            raise ValueError(
                f"feature view {path.stem!r} must use canonical C-contiguous order")
        offset = handle.tell()
        expected_size = offset + expected_dtype.itemsize * expected_shape[0] * expected_shape[1]
        if initial.st_size != expected_size:
            raise ValueError(
                f"feature view {path.stem!r} file size disagrees with its array header")
        values = np.memmap(
            handle,
            dtype=expected_dtype,
            mode="r",
            offset=offset,
            shape=expected_shape,
            order="C",
        )
        _verify_open_file(path, handle, initial)
        return values, handle, initial
    except Exception:
        handle.close()
        raise


def _validate_limit(value: object, *, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _is_missing(value: object) -> bool:
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    if isinstance(value, str):
        return False
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return isinstance(missing, (bool, np.bool_)) and bool(missing)


def _restore_metadata(
    metadata: pd.DataFrame,
    *,
    id_column: str,
    declared_types: object,
) -> Mapping[str, tuple[object, ...]]:
    columns = [column for column in metadata.columns if column != id_column]
    if (
        not isinstance(declared_types, dict)
        or set(declared_types) != set(columns)
        or any(
            not isinstance(kind, str) or kind not in _METADATA_TYPES
            for kind in declared_types.values()
        )
    ):
        raise ValueError(
            "schema-2 metadata_types must declare every metadata column exactly once")

    restored: dict[str, tuple[object, ...]] = {}
    for column in columns:
        kind = declared_types[column]
        values = metadata[column].tolist()
        nonmissing = [value for value in values if not _is_missing(value)]
        valid = (
            (kind == "null" and not nonmissing)
            or (kind == "str" and all(isinstance(value, str) for value in nonmissing))
            or (kind == "bool" and all(
                isinstance(value, (bool, np.bool_)) for value in nonmissing))
            or (kind == "int" and all(
                isinstance(value, (int, np.integer))
                and not isinstance(value, (bool, np.bool_))
                for value in nonmissing))
            or (kind == "float" and all(
                isinstance(value, (float, np.floating))
                and np.isfinite(value)
                for value in nonmissing))
        )
        if not valid:
            raise ValueError(
                f"metadata column {column!r} disagrees with metadata_types[{column!r}]")

        canonical: list[object] = []
        for value in values:
            if _is_missing(value):
                canonical.append(None)
            elif kind == "str":
                canonical.append(str(value))
            elif kind == "bool":
                canonical.append(bool(value))
            elif kind == "int":
                integer = int(value)
                if not -(2 ** 63) <= integer < 2 ** 63:
                    raise ValueError(
                        f"metadata column {column!r} integers must fit signed int64")
                canonical.append(integer)
            elif kind == "float":
                canonical.append(float(value))
            else:
                raise ValueError(
                    f"metadata column {column!r} declared null but contains a value")
        restored[column] = tuple(canonical)
    return MappingProxyType(restored)


def _validate_view_mapping(
    value: object,
    *,
    field: str,
    views: tuple[str, ...],
) -> Mapping[str, str]:
    if not isinstance(value, dict) or set(value) != set(views):
        raise ValueError(
            f"schema-2 {field} must name every declared feature view exactly once")
    if any(not isinstance(item, str) or not item.strip() for item in value.values()):
        raise ValueError(f"schema-2 {field} values must be non-empty strings")
    return MappingProxyType({view: value[view] for view in views})


def _feature_source_completion_data(reader) -> dict[str, int]:
    """Return safe aggregate counts for an active observation span."""
    n_views = len(reader.view_names)
    return {
        "n_rows": reader.n_rows,
        "n_features": reader.n_features,
        "n_views": n_views,
        "artifact_count": 3 + n_views,
    }


class FeatureBundleReader:
    """Validated lazy schema-2 source backed by live read-only memory maps.

    ``open`` detects member changes during validation. It cannot make files
    immutable after returning; external writers may change what an existing
    memory map observes.
    """

    def __init__(self, _token: object = None) -> None:
        if _token is not _OPEN_TOKEN:
            raise TypeError("use FeatureBundleReader.open(path) to construct a reader")
        self._root = Path()
        self._row_ids: tuple[str, ...] = ()
        self._feature_ids: tuple[int, ...] = ()
        self._views: tuple[str, ...] = ()
        self._roles: Mapping[str, str] = MappingProxyType({})
        self._orientations: Mapping[str, str] = MappingProxyType({})
        self._metadata: Mapping[str, tuple[object, ...]] = MappingProxyType({})
        self._provenance: Mapping[str, object] = MappingProxyType({})
        self._manifest: Mapping[str, object] = MappingProxyType({})
        self._arrays: Mapping[str, np.ndarray] = MappingProxyType({})
        self._activation_polarities: Mapping[str, str] = MappingProxyType({})
        self._code_semantics: Mapping[str, str] = MappingProxyType({})
        self._row_lookup: Mapping[str, int] = MappingProxyType({})

    @classmethod
    def open(
        cls,
        path: str | Path,
        *,
        max_rows: int = DEFAULT_MAX_ROWS,
        max_features: int = DEFAULT_MAX_FEATURES,
        max_elements: int = DEFAULT_MAX_ELEMENTS,
        max_manifest_bytes: int = DEFAULT_MAX_MANIFEST_BYTES,
        max_metadata_file_bytes: int = DEFAULT_MAX_METADATA_FILE_BYTES,
        max_metadata_compressed_bytes: int = DEFAULT_MAX_METADATA_COMPRESSED_BYTES,
        max_metadata_columns: int = DEFAULT_MAX_METADATA_COLUMNS,
        max_metadata_cells: int = DEFAULT_MAX_METADATA_CELLS,
        max_metadata_row_groups: int = DEFAULT_MAX_METADATA_ROW_GROUPS,
        max_metadata_uncompressed_bytes: int = DEFAULT_MAX_METADATA_UNCOMPRESSED_BYTES,
    ) -> "FeatureBundleReader":
        """Open schema 2 within explicit manifest, metadata, and array limits.

        Parquet twins are byte-verified and structurally bounded before one is
        decoded. Arrays are scanned once in bounded chunks for finiteness and content
        integrity, then retained as live read-only memory maps. No
        ``FeatureBatch`` or whole-array validation mask is constructed.
        """
        with automatic_stage("load_feature_source") as span:
            reader = cls._open_unobserved(
                path,
                max_rows=max_rows,
                max_features=max_features,
                max_elements=max_elements,
                max_manifest_bytes=max_manifest_bytes,
                max_metadata_file_bytes=max_metadata_file_bytes,
                max_metadata_compressed_bytes=max_metadata_compressed_bytes,
                max_metadata_columns=max_metadata_columns,
                max_metadata_cells=max_metadata_cells,
                max_metadata_row_groups=max_metadata_row_groups,
                max_metadata_uncompressed_bytes=max_metadata_uncompressed_bytes,
            )
            if span.active:
                try:
                    span.update(**_feature_source_completion_data(reader))
                except BaseException:
                    # Observation metadata must not alter a successful open.
                    pass
            return reader

    @classmethod
    def _open_unobserved(
        cls,
        path: str | Path,
        *,
        max_rows: int,
        max_features: int,
        max_elements: int,
        max_manifest_bytes: int,
        max_metadata_file_bytes: int,
        max_metadata_compressed_bytes: int,
        max_metadata_columns: int,
        max_metadata_cells: int,
        max_metadata_row_groups: int,
        max_metadata_uncompressed_bytes: int,
    ) -> "FeatureBundleReader":
        max_rows = _validate_limit(max_rows, name="max_rows")
        max_features = _validate_limit(max_features, name="max_features")
        max_elements = _validate_limit(max_elements, name="max_elements")
        max_manifest_bytes = _validate_limit(
            max_manifest_bytes, name="max_manifest_bytes")
        max_metadata_file_bytes = _validate_limit(
            max_metadata_file_bytes, name="max_metadata_file_bytes")
        max_metadata_compressed_bytes = _validate_limit(
            max_metadata_compressed_bytes, name="max_metadata_compressed_bytes")
        max_metadata_columns = _validate_limit(
            max_metadata_columns, name="max_metadata_columns")
        max_metadata_cells = _validate_limit(
            max_metadata_cells, name="max_metadata_cells")
        max_metadata_row_groups = _validate_limit(
            max_metadata_row_groups, name="max_metadata_row_groups")
        max_metadata_uncompressed_bytes = _validate_limit(
            max_metadata_uncompressed_bytes,
            name="max_metadata_uncompressed_bytes",
        )

        root = Path(path).expanduser().absolute()
        try:
            root_stat = root.lstat()
        except FileNotFoundError:
            raise FileNotFoundError(f"feature bundle does not exist: {root}") from None
        if stat.S_ISLNK(root_stat.st_mode):
            raise ValueError(f"feature bundle root must not be a symlink: {root}")
        if not stat.S_ISDIR(root_stat.st_mode):
            raise ValueError(f"feature bundle path is not a directory: {root}")
        root_fingerprint = _stat_fingerprint(root_stat)
        inventory = _scan_regular_members(root)

        core_names = {"manifest.json", "meta.parquet", "battles.parquet"}
        missing_core = core_names - set(inventory)
        if missing_core:
            raise FileNotFoundError(
                f"feature bundle is missing required artifacts: {sorted(missing_core)}")

        manifest_path = root / "manifest.json"
        manifest_handle, manifest_initial = _open_regular(manifest_path)
        try:
            if manifest_initial.st_size > max_manifest_bytes:
                raise ValueError(
                    "feature bundle manifest exceeds configured "
                    f"max_manifest_bytes {max_manifest_bytes}")
            manifest = _read_manifest(
                manifest_handle, max_bytes=max_manifest_bytes)
            _verify_open_file(manifest_path, manifest_handle, manifest_initial)
        finally:
            manifest_handle.close()

        if set(manifest) != _MANIFEST_KEYS:
            raise ValueError(
                "schema-2 manifest keys must match exactly: "
                f"missing={sorted(_MANIFEST_KEYS - set(manifest))}, "
                f"extra={sorted(set(manifest) - _MANIFEST_KEYS)}")
        schema = manifest["schema_version"]
        if type(schema) is not int or schema != 2:
            raise ValueError(
                f"FeatureBundleReader requires schema_version 2, got {schema!r}")

        declared = manifest["output_arrays"]
        if (
            not isinstance(declared, list)
            or not declared
            or any(not isinstance(name, str) or not _ARRAY_NAME.fullmatch(name)
                   for name in declared)
            or len(set(declared)) != len(declared)
        ):
            raise ValueError(
                "schema-2 output_arrays must be unique safe lower_snake_case names")
        views = tuple(declared)
        expected_files = {
            "manifest.json", "meta.parquet", "battles.parquet",
            *(f"{view}.npy" for view in views),
        }
        if set(inventory) != expected_files:
            raise ValueError(
                "feature bundle contains missing or undeclared artifacts: "
                f"missing={sorted(expected_files - set(inventory))}, "
                f"extra={sorted(set(inventory) - expected_files)}")

        n_rows = manifest["n_rows"]
        width = manifest["feature_width"]
        if type(n_rows) is not int or n_rows <= 0:
            raise ValueError("schema-2 n_rows must be a positive integer")
        if type(width) is not int or width <= 0:
            raise ValueError("schema-2 feature_width must be a positive integer")
        if n_rows > max_rows:
            raise ValueError(
                f"schema-2 n_rows {n_rows} exceeds configured max_rows {max_rows}")
        if width > max_features:
            raise ValueError(
                f"schema-2 feature_width {width} exceeds configured max_features "
                f"{max_features}")
        elements = n_rows * width
        if elements > max_elements:
            raise ValueError(
                f"schema-2 n_rows * feature_width is {elements}, exceeding "
                f"configured max_elements {max_elements}")

        raw_feature_ids = manifest["feature_ids"]
        if not isinstance(raw_feature_ids, list):
            raise ValueError("schema-2 feature_ids must be a list")
        feature_ids = validate_feature_ids(raw_feature_ids, width=width)
        roles = _validate_view_mapping(manifest["roles"], field="roles", views=views)
        orientations = _validate_view_mapping(
            manifest["orientations"], field="orientations", views=views)

        global_polarity = manifest["activation_polarity"]
        global_semantics = manifest["code_semantics"]
        if not isinstance(global_polarity, str) or not global_polarity.strip():
            raise ValueError(
                "schema-2 activation_polarity must be a non-empty string")
        if not isinstance(global_semantics, str) or not global_semantics.strip():
            raise ValueError("schema-2 code_semantics must be a non-empty string")

        metadata = _read_verified_metadata_twins(
            root / "meta.parquet",
            root / "battles.parquet",
            expected_rows=n_rows,
            max_file_bytes=max_metadata_file_bytes,
            max_compressed_bytes=max_metadata_compressed_bytes,
            max_columns=max_metadata_columns,
            max_cells=max_metadata_cells,
            max_row_groups=max_metadata_row_groups,
            max_uncompressed_bytes=max_metadata_uncompressed_bytes,
        )
        if len(metadata) != n_rows:
            raise ValueError(
                f"schema-2 n_rows is {n_rows}, but metadata contains {len(metadata)} rows")
        if not metadata.columns.is_unique or any(
            not isinstance(column, str) or not column for column in metadata.columns
        ):
            raise ValueError(
                "feature bundle metadata columns must be unique non-empty strings")
        id_columns = [
            column for column in ("row_id", "battle_id") if column in metadata
        ]
        if len(id_columns) != 1:
            raise ValueError(
                "feature bundle metadata needs exactly one row_id or battle_id column")
        id_column = id_columns[0]
        try:
            row_ids = validate_row_ids(metadata[id_column].tolist())
        except ValueError as exc:
            raise ValueError(f"invalid feature bundle {id_column}: {exc}") from exc
        restored_metadata = _restore_metadata(
            metadata,
            id_column=id_column,
            declared_types=manifest["metadata_types"],
        )

        declared_dtypes = manifest["array_dtypes"]
        declared_shapes = manifest["array_shapes"]
        if (
            not isinstance(declared_dtypes, dict)
            or set(declared_dtypes) != set(views)
            or any(
                not isinstance(value, str) or value not in _CANONICAL_DTYPES
                for value in declared_dtypes.values()
            )
        ):
            raise ValueError(
                "schema-2 array_dtypes must declare float32 or bool for every view")
        if not isinstance(declared_shapes, dict) or set(declared_shapes) != set(views):
            raise ValueError(
                "schema-2 array_shapes must declare every feature view exactly once")

        arrays: dict[str, np.ndarray] = {}
        open_arrays: list[tuple[Path, BinaryIO, os.stat_result]] = []
        try:
            for view in views:
                expected_shape_list = [n_rows, width]
                declared_shape = declared_shapes[view]
                if (
                    not isinstance(declared_shape, list)
                    or len(declared_shape) != 2
                    or any(type(dimension) is not int for dimension in declared_shape)
                    or declared_shape != expected_shape_list
                ):
                    raise ValueError(
                        f"schema-2 array_shapes[{view!r}] must equal "
                        f"{expected_shape_list}")
                expected_dtype = _CANONICAL_DTYPES[declared_dtypes[view]]
                array_path = root / f"{view}.npy"
                values, handle, initial = _open_npy_memmap(
                    array_path,
                    expected_shape=(n_rows, width),
                    expected_dtype=expected_dtype,
                )
                arrays[view] = values
                open_arrays.append((array_path, handle, initial))

            dataset_hash = manifest["dataset_hash"]
            if (
                not isinstance(dataset_hash, str)
                or re.fullmatch(r"[0-9a-f]{64}", dataset_hash) is None
            ):
                raise ValueError(
                    "schema-2 dataset_hash must be a lowercase SHA-256 hex digest")
            try:
                observed_hash = ordered_dataset_hash(metadata, arrays)
            except ValueError as exc:
                raise ValueError(f"invalid feature bundle array contents: {exc}") from exc
            if dataset_hash != observed_hash:
                raise ValueError(
                    "feature bundle dataset_hash does not match metadata and arrays")

            raw_provenance = manifest["provenance"]
            if not isinstance(raw_provenance, dict):
                raise ValueError("schema-2 provenance must be a mapping")
            reject_secrets(raw_provenance, where="feature bundle provenance")
            source_provenance = validate_portable_mapping(
                raw_provenance, where="feature bundle provenance")
            descriptors = source_provenance.get("views", {})
            if not isinstance(descriptors, Mapping):
                raise ValueError("feature bundle provenance views must be a mapping")
            unknown_descriptors = set(descriptors) - set(views)
            if unknown_descriptors:
                raise ValueError(
                    "feature bundle provenance has semantics for unknown views: "
                    f"{sorted(unknown_descriptors)}")
            polarities: dict[str, str] = {}
            semantics: dict[str, str] = {}
            for view in views:
                descriptor = descriptors.get(view, {})
                if not isinstance(descriptor, Mapping):
                    raise ValueError(
                        f"feature bundle provenance view {view!r} must be a mapping")
                polarity = descriptor.get("activation_polarity", global_polarity)
                meaning = descriptor.get("code_semantics", global_semantics)
                if not isinstance(polarity, str) or not polarity.strip():
                    raise ValueError(
                        f"activation_polarity for feature view {view!r} must be non-empty")
                if not isinstance(meaning, str) or not meaning.strip():
                    raise ValueError(
                        f"code_semantics for feature view {view!r} must be non-empty")
                polarities[view] = polarity
                semantics[view] = meaning

            full_provenance = validate_portable_mapping(
                {
                    **dict(source_provenance),
                    "encoded_bundle": {
                        key: value for key, value in manifest.items()
                        if key != "provenance"
                    },
                },
                where="feature source provenance",
            )
            frozen_manifest = validate_portable_mapping(
                manifest, where="feature bundle manifest")

            for array_path, handle, initial in open_arrays:
                _verify_open_file(array_path, handle, initial)
            try:
                final_root_stat = root.lstat()
            except OSError as exc:
                raise ValueError("feature bundle root changed while opening") from exc
            if _stat_fingerprint(final_root_stat) != root_fingerprint:
                raise ValueError("feature bundle root changed while opening")
            _verify_member_inventory(root, inventory)
        finally:
            for _, handle, _ in open_arrays:
                handle.close()

        reader = cls(_OPEN_TOKEN)
        reader._root = root
        reader._row_ids = row_ids
        reader._feature_ids = feature_ids
        reader._views = views
        reader._roles = roles
        reader._orientations = orientations
        reader._metadata = restored_metadata
        reader._provenance = full_provenance
        reader._manifest = frozen_manifest
        reader._arrays = MappingProxyType(arrays)
        reader._activation_polarities = MappingProxyType(polarities)
        reader._code_semantics = MappingProxyType(semantics)
        reader._row_lookup = MappingProxyType({
            row_id: position for position, row_id in enumerate(row_ids)
        })
        return reader

    @property
    def root(self) -> Path:
        return self._root

    @property
    def row_ids(self) -> tuple[str, ...]:
        return self._row_ids

    @property
    def feature_ids(self) -> tuple[int, ...]:
        return self._feature_ids

    @property
    def view_names(self) -> tuple[str, ...]:
        """Feature view names in manifest declaration order."""
        return self._views

    @property
    def views(self) -> tuple[str, ...]:
        """Compatibility alias for :attr:`view_names`."""
        return self.view_names

    @property
    def roles(self) -> Mapping[str, str]:
        return self._roles

    @property
    def orientations(self) -> Mapping[str, str]:
        return self._orientations

    @property
    def activation_polarities(self) -> Mapping[str, str]:
        """Resolved per-view activation polarity declarations."""
        return self._activation_polarities

    @property
    def code_semantics_by_view(self) -> Mapping[str, str]:
        """Resolved per-view code-semantics declarations."""
        return self._code_semantics

    @property
    def code_semantics(self) -> Mapping[str, str]:
        """Compatibility alias for :attr:`code_semantics_by_view`."""
        return self.code_semantics_by_view

    @property
    def metadata(self) -> Mapping[str, tuple[object, ...]]:
        return self._metadata

    @property
    def provenance(self) -> Mapping[str, object]:
        return self._provenance

    @property
    def manifest(self) -> Mapping[str, object]:
        return self._manifest

    @property
    def n_rows(self) -> int:
        return len(self._row_ids)

    @property
    def n_features(self) -> int:
        return len(self._feature_ids)

    def _require_view(self, view: str) -> str:
        if not isinstance(view, str) or not view:
            raise ValueError("feature view must be a non-empty string")
        if view not in self._arrays:
            available = ", ".join(self._views)
            raise ValueError(
                f"feature source has no view {view!r}; available: {available}")
        return view

    def array(self, view: str) -> np.ndarray:
        """Return a live read-only memory map for one complete feature view.

        Read-only protects against writes through this array. It does not stop
        another process from changing the backing file after ``open`` returns.
        """
        return self._arrays[self._require_view(view)]

    def assert_row_ids(self, row_ids: Sequence[object]) -> None:
        """Require exact row identity and order, with no implicit alignment."""
        if isinstance(row_ids, (str, bytes)):
            raise ValueError("aligned row_ids must be a sequence, not a bare string")
        try:
            observed = validate_row_ids(row_ids)
        except ValueError as exc:
            raise ValueError(f"invalid aligned row_ids: {exc}") from exc
        if len(observed) != self.n_rows:
            raise ValueError(
                "row_ids are not exactly aligned: "
                f"expected {self.n_rows} rows, got {len(observed)}")
        for position, (expected, actual) in enumerate(zip(self.row_ids, observed)):
            if expected != actual:
                raise ValueError(
                    "row_ids are not exactly aligned: first mismatch at position "
                    f"{position}: expected {expected!r}, got {actual!r}")

    def row_positions(self, row_ids: Sequence[object]) -> tuple[int, ...]:
        """Resolve a unique requested row-id sequence to explicit source positions."""
        if isinstance(row_ids, (str, bytes)):
            raise ValueError("requested row_ids must be a sequence, not a bare string")
        try:
            requested = validate_row_ids(row_ids)
        except ValueError as exc:
            raise ValueError(f"invalid requested row_ids: {exc}") from exc
        missing = [row_id for row_id in requested if row_id not in self._row_lookup]
        if missing:
            raise ValueError(
                f"requested row_ids are absent from feature source: {missing}")
        return tuple(self._row_lookup[row_id] for row_id in requested)

    def _selected_views(self, views: Sequence[str] | None) -> tuple[str, ...]:
        if views is None:
            return self.view_names
        if isinstance(views, (str, bytes)):
            raise ValueError("views must be a sequence, not a bare string")
        selected = tuple(views)
        if not selected:
            raise ValueError("views must select at least one feature view")
        if any(not isinstance(view, str) or not view for view in selected):
            raise ValueError("views must contain non-empty strings")
        if len(set(selected)) != len(selected):
            raise ValueError("views must not contain duplicates")
        unknown = [view for view in selected if view not in self._arrays]
        if unknown:
            raise ValueError(
                f"unknown feature views: {unknown}; available: {list(self.view_names)}")
        return selected

    def iter_chunks(
        self,
        chunk_size: int,
        *,
        views: Sequence[str] | None = None,
        row_ids: Sequence[object] | None = None,
    ) -> Iterator[FeatureChunk]:
        """Yield bounded aligned chunks in source or explicitly selected order."""
        if type(chunk_size) is not int or chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")
        selected_views = self._selected_views(views)
        if row_ids is None:
            for start in range(0, self.n_rows, chunk_size):
                stop = min(start + chunk_size, self.n_rows)
                yield FeatureChunk(
                    row_ids=self.row_ids[start:stop],
                    positions=range(start, stop),
                    arrays=MappingProxyType({
                        view: self._arrays[view][start:stop]
                        for view in selected_views
                    }),
                    metadata=MappingProxyType({
                        name: values[start:stop]
                        for name, values in self.metadata.items()
                    }),
                )
            return

        selected_positions = self.row_positions(row_ids)
        for start in range(0, len(selected_positions), chunk_size):
            positions = selected_positions[start:start + chunk_size]
            index = np.asarray(positions, dtype=np.intp)
            chunk_arrays: dict[str, np.ndarray] = {}
            for view in selected_views:
                values = np.asarray(self._arrays[view][index])
                values.setflags(write=False)
                chunk_arrays[view] = values
            yield FeatureChunk(
                row_ids=tuple(self.row_ids[position] for position in positions),
                positions=positions,
                arrays=MappingProxyType(chunk_arrays),
                metadata=MappingProxyType({
                    name: tuple(values[position] for position in positions)
                    for name, values in self.metadata.items()
                }),
            )


__all__ = [
    "DEFAULT_MAX_ROWS",
    "DEFAULT_MAX_FEATURES",
    "DEFAULT_MAX_ELEMENTS",
    "DEFAULT_MAX_MANIFEST_BYTES",
    "DEFAULT_MAX_METADATA_FILE_BYTES",
    "DEFAULT_MAX_METADATA_COMPRESSED_BYTES",
    "DEFAULT_MAX_METADATA_COLUMNS",
    "DEFAULT_MAX_METADATA_CELLS",
    "DEFAULT_MAX_METADATA_ROW_GROUPS",
    "DEFAULT_MAX_METADATA_UNCOMPRESSED_BYTES",
    "FeatureSource",
    "FeatureChunk",
    "FeatureBundleReader",
]
