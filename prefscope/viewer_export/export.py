"""Package existing PrefScope data with an already-built static Viewer."""

from __future__ import annotations

from collections.abc import Mapping
import ctypes
from datetime import date, datetime, time, timedelta
import errno
import hashlib
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile
from typing import Any

import numpy as np
import pandas as pd

from prefscope import __version__ as _prefscope_version
from prefscope.api.feature_catalog import FeatureCatalog
from prefscope.core.features import FeatureBatch

_VIEWER_BUILD_SCHEMA = "prefscope.viewer_build"
_VIEWER_BUILD_SCHEMA_VERSION = 1
_VIEWER_DATA_SCHEMA = "prefscope.viewer_data"
_VIEWER_DATA_SCHEMA_VERSION = 2
_VIEWER_BUNDLE_SCHEMA = "prefscope.viewer_bundle"
_VIEWER_BUNDLE_SCHEMA_VERSION = 1
_VIEWER_PACKAGE = "@prefscope/viewer"
_DATA_PATH = "data/viewer-data.json"
_BUILD_PATH = "viewer-build.json"
_BUNDLE_PATH = "viewer-bundle.json"
_JS_MAX_SAFE_INTEGER = 2**53 - 1


def _jsonable(value: Any, *, where: str = "value") -> Any:
    """Normalize one value without losing JSON/JavaScript numeric identity."""
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, np.generic):
        return _jsonable(value.item(), where=where)
    if isinstance(value, (bool, str)):
        return value
    if isinstance(value, Integral):
        number = int(value)
        if abs(number) > _JS_MAX_SAFE_INTEGER:
            raise ValueError(f"{where} must be a JavaScript safe integer")
        return number
    if isinstance(value, Real):
        number = float(value)
        if math.isnan(number):
            return None
        if not math.isfinite(number):
            raise ValueError(f"{where} must be finite or missing")
        return number
    if isinstance(value, pd.Interval):
        return {
            "left": _jsonable(value.left, where=f"{where}.left"),
            "right": _jsonable(value.right, where=f"{where}.right"),
            "closed": value.closed,
        }
    if isinstance(value, (datetime, date, time, pd.Timestamp, pd.Timedelta)):
        return value.isoformat()
    if isinstance(value, timedelta):
        return str(value)
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{where} object keys must be strings")
            result[key] = _jsonable(item, where=f"{where}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [
            _jsonable(item, where=f"{where}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(f"{where} is not JSON-compatible: {type(value).__name__}")


def _table(table: pd.DataFrame) -> dict[str, Any]:
    return {
        "index": [
            _jsonable(value, where=f"table.index[{index}]")
            for index, value in enumerate(table.index.tolist())
        ],
        "index_names": _jsonable(
            list(table.index.names), where="table.index_names"
        ),
        "columns": [
            _jsonable(value, where=f"table.columns[{index}]")
            for index, value in enumerate(table.columns.tolist())
        ],
        "column_names": _jsonable(
            list(table.columns.names), where="table.column_names"
        ),
        "data": [
            [
                _jsonable(value, where=f"table.data[{row_index}][{column_index}]")
                for column_index, value in enumerate(row)
            ]
            for row_index, row in enumerate(
                table.itertuples(index=False, name=None)
            )
        ],
    }


def _feature_space(feature_space_id: str | None, status: str | None) -> dict[str, Any]:
    allowed = {
        "exact_weights",
        "declared_pinned_coordinate",
        "declared_unpinned",
        "unbound",
    }
    if feature_space_id is None:
        if status not in {None, "unbound"}:
            raise ValueError("an unbound Viewer feature space cannot declare a bound status")
        return {"feature_space_id": None, "feature_space_status": "unbound"}
    if not isinstance(feature_space_id, str) or not feature_space_id.strip():
        raise ValueError("Viewer feature_space_id must be a non-empty string or None")
    if status not in allowed - {"unbound"}:
        raise ValueError("a bound Viewer feature space needs a known bound status")
    return {"feature_space_id": feature_space_id, "feature_space_status": status}


def _viewer_data(
    features: FeatureBatch,
    catalog: FeatureCatalog | None,
    tables: Mapping[str, pd.DataFrame] | None,
) -> dict[str, Any]:
    if not isinstance(features, FeatureBatch):
        raise TypeError("features must be a FeatureBatch")
    if catalog is not None and not isinstance(catalog, FeatureCatalog):
        raise TypeError("catalog must be a FeatureCatalog or None")
    lens = features.provenance.get("lens", {})
    if not isinstance(lens, Mapping):
        raise ValueError("Viewer lens provenance must be a mapping")
    feature_space = _feature_space(
        lens.get("feature_space_id"), lens.get("feature_space_status")
    )
    unsafe_feature_ids = [
        feature_id
        for feature_id in features.feature_ids
        if abs(feature_id) > _JS_MAX_SAFE_INTEGER
    ]
    if unsafe_feature_ids:
        raise ValueError("feature_ids must be JavaScript safe integers for Viewer export")
    if catalog is not None:
        catalog.validate_for(features.matrix(next(iter(features.arrays))))
        if any(
            abs(feature_id) > _JS_MAX_SAFE_INTEGER
            for feature_id in catalog.feature_ids
        ):
            raise ValueError(
                "catalog feature_ids must be JavaScript safe integers for Viewer export"
            )
        unknown_ids = sorted(set(catalog.feature_ids) - set(features.feature_ids))
        if unknown_ids:
            raise ValueError(
                f"catalog contains feature IDs outside the exported batch: {unknown_ids[:10]}"
            )
    supplied = dict(tables or {})
    if any(not isinstance(name, str) or not name for name in supplied):
        raise ValueError("table names must be non-empty strings")
    if any(not isinstance(table, pd.DataFrame) for table in supplied.values()):
        raise TypeError("viewer tables must be pandas DataFrames")

    views = {}
    for name in features.arrays:
        matrix = features.matrix(name)
        views[name] = {
            "role": matrix.role,
            "orientation": matrix.orientation,
            "activation_polarity": matrix.activation_polarity,
            "code_semantics": matrix.code_semantics,
            "values": matrix.values.tolist(),
        }
    return {
        "schema": _VIEWER_DATA_SCHEMA,
        "schema_version": 1,  # The reusable single-space shape remains v1.
        "row_ids": list(features.row_ids),
        "feature_ids": list(features.feature_ids),
        "feature_space": feature_space,
        "views": views,
        "row_metadata": _jsonable(features.metadata),
        "provenance": _jsonable(features.provenance),
        "catalog": (
            None
            if catalog is None
            else {
                "table": _table(catalog.to_frame()),
                "feature_space": _feature_space(
                    catalog.feature_space_id, catalog.feature_space_status
                ),
                "provenance": _jsonable(catalog.provenance),
                "column_sources": _jsonable(catalog.column_sources),
            }
        ),
        "tables": {name: _table(table) for name, table in supplied.items()},
    }


def _is_reparse_point(status: os.stat_result) -> bool:
    marker = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(marker and getattr(status, "st_file_attributes", 0) & marker)


def _path_identity(path: Path) -> tuple[int, int, int]:
    status = os.lstat(path)
    return (status.st_dev, status.st_ino, status.st_mode)


def _validate_viewer_tree(viewer_dist: Path) -> None:
    """Reject links, reparse points, and special files in a static tree."""
    try:
        root_status = os.lstat(viewer_dist)
    except FileNotFoundError as exc:
        raise ValueError(f"viewer_dist must be a directory: {viewer_dist}") from exc
    if stat.S_ISLNK(root_status.st_mode) or _is_reparse_point(root_status):
        raise ValueError("viewer_dist must not be a symbolic link or reparse point")
    if not stat.S_ISDIR(root_status.st_mode):
        raise ValueError(f"viewer_dist must be a directory: {viewer_dist}")

    def visit(directory: Path) -> None:
        with os.scandir(directory) as entries:
            for entry in entries:
                path = Path(entry.path)
                relative = path.relative_to(viewer_dist).as_posix()
                status = entry.stat(follow_symlinks=False)
                if stat.S_ISLNK(status.st_mode) or _is_reparse_point(status):
                    raise ValueError(
                        "viewer_dist must not contain symbolic links or reparse "
                        f"points: {relative}"
                    )
                if stat.S_ISDIR(status.st_mode):
                    visit(path)
                elif not stat.S_ISREG(status.st_mode):
                    raise ValueError(
                        f"viewer_dist must contain only regular files and directories: "
                        f"{relative}"
                    )

    visit(viewer_dist)


def _load_viewer_build(viewer_dist: Path) -> dict[str, Any]:
    if not viewer_dist.is_dir():
        raise ValueError(f"viewer_dist must be a directory: {viewer_dist}")
    index_path = viewer_dist / "index.html"
    if not index_path.is_file():
        raise ValueError("viewer_dist must contain index.html at its root")
    build_path = viewer_dist / _BUILD_PATH
    if not build_path.is_file():
        raise ValueError(f"viewer_dist must contain {_BUILD_PATH} at its root")
    try:
        build = json.loads(build_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {_BUILD_PATH}: {exc}") from exc
    if not isinstance(build, dict):
        raise ValueError(f"{_BUILD_PATH} must contain a JSON object")
    if build.get("schema") != _VIEWER_BUILD_SCHEMA:
        raise ValueError(f"{_BUILD_PATH} schema must be {_VIEWER_BUILD_SCHEMA!r}")
    if (
        not isinstance(build.get("schema_version"), int)
        or isinstance(build.get("schema_version"), bool)
        or build["schema_version"] != _VIEWER_BUILD_SCHEMA_VERSION
    ):
        raise ValueError(f"{_BUILD_PATH} schema_version must be 1")
    if build.get("package") != _VIEWER_PACKAGE:
        raise ValueError(f"{_BUILD_PATH} package must be {_VIEWER_PACKAGE!r}")
    if not isinstance(build.get("version"), str) or not build["version"]:
        raise ValueError(f"{_BUILD_PATH} version must be a non-empty string")

    supported = build.get("supported_data_schemas")
    supports_viewer_data = False
    if isinstance(supported, list):
        for item in supported:
            if not isinstance(item, dict) or item.get("schema") != _VIEWER_DATA_SCHEMA:
                continue
            versions = item.get("versions")
            if isinstance(versions, list) and any(
                isinstance(version, int)
                and not isinstance(version, bool)
                and version == _VIEWER_DATA_SCHEMA_VERSION
                for version in versions
            ):
                supports_viewer_data = True
                break
    if not supports_viewer_data:
        raise ValueError(
            f"{_BUILD_PATH} must support {_VIEWER_DATA_SCHEMA} "
            f"v{_VIEWER_DATA_SCHEMA_VERSION}"
        )
    return build


def _exists(path: Path) -> bool:
    return os.path.lexists(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inventory(root: Path, paths: list[str]) -> list[dict[str, Any]]:
    inventory = []
    for relative in sorted(paths):
        path = root / relative
        inventory.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return inventory


def _build_identity(files: list[dict[str, Any]]) -> str:
    canonical = json.dumps(
        files,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _publish_directory_no_replace(source: Path, destination: Path) -> None:
    """Atomically publish a directory without replacing an existing path."""
    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    libc = ctypes.CDLL(None, use_errno=True)
    result: int | None = None
    if sys.platform == "darwin":
        rename_exclusive = libc.renamex_np
        rename_exclusive.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename_exclusive.restype = ctypes.c_int
        result = rename_exclusive(source_bytes, destination_bytes, 0x00000004)
    elif sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        rename_exclusive = libc.renameat2
        rename_exclusive.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename_exclusive.restype = ctypes.c_int
        result = rename_exclusive(
            -100,
            source_bytes,
            -100,
            destination_bytes,
            0x00000001,
        )
    if result is not None:
        if result == 0:
            return
        error = ctypes.get_errno()
        if error in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FileExistsError(
                error, os.strerror(error), str(destination)
            )
        raise OSError(error, os.strerror(error), str(destination))

    if sys.platform == "win32":
        # Windows rename refuses an existing destination.
        os.rename(source, destination)
        return
    raise OSError(
        errno.ENOTSUP,
        "atomic no-replace directory publication is unavailable on this platform",
        str(destination),
    )


def export_viewer_bundle(
    features: FeatureBatch,
    out: str | Path,
    *,
    viewer_dist: str | Path,
    catalog: FeatureCatalog | None = None,
    tables: Mapping[str, pd.DataFrame] | None = None,
    prompt_features: FeatureBatch | None = None,
    prompt_catalog: FeatureCatalog | None = None,
) -> Path:
    """Package supplied data with an existing static Viewer build.

    ``viewer_dist`` must be a built Viewer directory with a compatible root
    ``viewer-build.json``. The build is copied as-is; this function only serializes the
    supplied data and writes bundle metadata. It does not build the Viewer or compute
    analyses. Optional ``prompt_features`` must have exactly the same ordered row IDs
    and only prompt-role views. Its catalog and feature identity remain separate.
    """
    out = Path(out)
    viewer_dist = Path(viewer_dist)
    if _exists(out):
        raise FileExistsError(f"destination already exists: {out}")

    if out.resolve().is_relative_to(viewer_dist.resolve()):
        raise ValueError("destination must not be inside viewer_dist")
    _validate_viewer_tree(viewer_dist)
    source_identity = _path_identity(viewer_dist)
    if _exists(viewer_dist / _BUNDLE_PATH):
        raise ValueError(f"viewer_dist contains reserved path {_BUNDLE_PATH}")
    if _exists(viewer_dist / "data"):
        raise ValueError(
            "viewer_dist contains reserved path data; use a dataset-free Viewer build"
        )
    payload = _viewer_data(features, catalog, tables)
    if prompt_catalog is not None and prompt_features is None:
        raise ValueError("prompt_catalog requires prompt_features")
    prompt = None
    if prompt_features is not None:
        if not isinstance(prompt_features, FeatureBatch):
            raise TypeError("prompt_features must be a FeatureBatch or None")
        if prompt_features.row_ids != features.row_ids:
            raise ValueError(
                "prompt_features row_ids must exactly match features in order"
            )
        if any(role != "prompt" for role in prompt_features.roles.values()):
            raise ValueError("all prompt_features views must have role 'prompt'")
        prompt = _viewer_data(prompt_features, prompt_catalog, None)
    payload["schema_version"] = _VIEWER_DATA_SCHEMA_VERSION
    payload["prompt"] = prompt

    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}.staging-", dir=out.parent))
    try:
        shutil.copytree(
            viewer_dist,
            staging,
            dirs_exist_ok=True,
            symlinks=True,
        )
        staging.chmod(staging.stat().st_mode | stat.S_IWUSR | stat.S_IXUSR)
        if _path_identity(viewer_dist) != source_identity:
            raise ValueError("viewer_dist changed while it was being copied")
        _validate_viewer_tree(viewer_dist)
        _validate_viewer_tree(staging)
        if _exists(staging / _BUNDLE_PATH):
            raise ValueError(f"viewer_dist contains reserved path {_BUNDLE_PATH}")
        if _exists(staging / "data"):
            raise ValueError(
                "viewer_dist contains reserved path data; "
                "use a dataset-free Viewer build"
            )
        build = _load_viewer_build(staging)
        viewer_paths = [
            path.relative_to(staging).as_posix()
            for path in staging.rglob("*")
            if path.is_file()
        ]
        viewer_files = _inventory(staging, viewer_paths)

        data_path = staging / _DATA_PATH
        data_path.parent.mkdir(parents=True, exist_ok=True)
        data_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        files = _inventory(staging, [*viewer_paths, _DATA_PATH])
        manifest = {
            "schema": _VIEWER_BUNDLE_SCHEMA,
            "schema_version": _VIEWER_BUNDLE_SCHEMA_VERSION,
            "producer": {"package": "prefscope", "version": _prefscope_version},
            "viewer": {
                "package": build["package"],
                "version": build["version"],
                "build": {
                    "path": _BUILD_PATH,
                    "schema": build["schema"],
                    "schema_version": build["schema_version"],
                    "supported_data_schemas": build["supported_data_schemas"],
                },
                "build_sha256": _build_identity(viewer_files),
            },
            "data": {
                "path": _DATA_PATH,
                "schema": payload["schema"],
                "schema_version": payload["schema_version"],
            },
            "files": files,
        }
        (staging / _BUNDLE_PATH).write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        _publish_directory_no_replace(staging, out)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return out


__all__ = ["export_viewer_bundle"]
