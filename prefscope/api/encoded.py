"""Typed loader for reusable encoded feature bundles."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import uuid

import numpy as np
import pandas as pd

from prefscope.core.features import FeatureBatch
from prefscope.core.provenance import ordered_dataset_hash


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


def load_feature_batch(path, *, arrays=None) -> FeatureBatch:
    """Load and validate an ``encode-dataset`` directory without importing Torch."""
    root = Path(path)
    manifest_path = root / "manifest.json"
    metadata_path = root / "meta.parquet"
    battles_path = root / "battles.parquet"
    if not all(path.is_file() for path in (manifest_path, metadata_path, battles_path)):
        raise FileNotFoundError(
            "encoded bundle must contain manifest.json, meta.parquet, and "
            f"battles.parquet: {root}")
    manifest = json.loads(manifest_path.read_text())
    schema_version = manifest.get("schema_version")
    if type(schema_version) is not int or schema_version not in {1, 2}:
        raise ValueError(f"unsupported encoded bundle schema {schema_version!r}")
    declared = manifest.get("output_arrays")
    if not isinstance(declared, list) or not declared or len(set(declared)) != len(declared):
        raise ValueError("encoded bundle must declare unique output_arrays")
    expected_files = {
        "manifest.json", "meta.parquet", "battles.parquet",
        *(f"{name}.npy" for name in declared),
    }
    actual_files = {entry.name for entry in root.iterdir()}
    if actual_files != expected_files:
        raise ValueError(
            "encoded bundle contains missing or undeclared artifacts: "
            f"missing={sorted(expected_files - actual_files)}, "
            f"extra={sorted(actual_files - expected_files)}")
    selected = tuple(declared if arrays is None else arrays)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("arrays must select at least one unique feature view")
    unknown = set(selected) - set(declared)
    if unknown:
        raise ValueError(f"arrays are not declared by the bundle: {sorted(unknown)}")
    if schema_version == 1:
        unsupported = set(selected) - set(_ROLES)
        if unsupported:
            raise ValueError(f"unsupported encoded feature views: {sorted(unsupported)}")
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
        ):
            raise ValueError(
                "schema-2 encoded bundle roles/orientations must name every array")
        roles = {name: declared_roles[name] for name in selected}
        orientations = {name: declared_orientations[name] for name in selected}
    declared_dtypes = (
        {name: "float32" for name in declared}
        if schema_version == 1 else manifest.get("array_dtypes")
    )
    if (
        not isinstance(declared_dtypes, dict)
        or set(declared_dtypes) != set(declared)
        or any(value not in {"float32", "bool"} for value in declared_dtypes.values())
    ):
        raise ValueError(
            "encoded bundle array_dtypes must declare float32 or bool for every array")
    metadata = pd.read_parquet(metadata_path)
    battles = pd.read_parquet(battles_path)
    if not metadata.equals(battles):
        raise ValueError("meta.parquet and battles.parquet must be exactly identical")
    n_rows = manifest.get("n_rows")
    width = (
        manifest.get("m_total") if schema_version == 1
        else manifest.get("feature_width")
    )
    if type(n_rows) is not int or n_rows <= 0 or len(metadata) != n_rows:
        raise ValueError("encoded bundle n_rows does not match meta.parquet")
    if type(width) is not int or width <= 0:
        field = "m_total" if schema_version == 1 else "feature_width"
        raise ValueError(f"encoded bundle {field} must be a positive integer")
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
    if metadata[id_column].isna().any() or metadata[id_column].astype(str).duplicated().any():
        raise ValueError(f"encoded metadata {id_column} values must be unique and nonmissing")
    loaded = {}
    declared_shapes = manifest.get("array_shapes") or {}
    for name in declared:
        if schema_version == 1 and name not in _ROLES:
            raise ValueError(f"unsupported encoded feature view: {name}")
        if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
            raise ValueError(f"unsafe encoded feature array name: {name!r}")
        array_path = root / f"{name}.npy"
        if not array_path.is_file():
            raise FileNotFoundError(f"encoded bundle is missing {array_path.name}")
        matrix = np.load(array_path, mmap_mode="r")
        if matrix.shape != (n_rows, width):
            raise ValueError(
                f"{name} has shape {matrix.shape}; expected {(n_rows, width)}")
        if declared_shapes.get(name) != list(matrix.shape):
            raise ValueError(f"{name} shape disagrees with encoded manifest")
        expected_dtype = np.dtype(declared_dtypes[name])
        if matrix.dtype != expected_dtype or not np.isfinite(matrix).all():
            raise ValueError(
                f"{name} must contain canonical finite {expected_dtype} values")
        loaded[name] = matrix
    observed_hash = ordered_dataset_hash(metadata, loaded)
    if manifest.get("dataset_hash") != observed_hash:
        raise ValueError("encoded bundle dataset_hash does not match metadata and arrays")
    values = {name: loaded[name] for name in selected}
    raw_metadata_columns = [
        str(column) for column in metadata.columns if column != id_column]
    if schema_version == 2:
        metadata_types = manifest.get("metadata_types")
        if (
            not isinstance(metadata_types, dict)
            or set(metadata_types) != set(raw_metadata_columns)
            or any(value not in {"null", "str", "bool", "int", "float"}
                   for value in metadata_types.values())
        ):
            raise ValueError(
                "schema-2 metadata_types must declare every metadata column")
    else:
        metadata_types = {name: None for name in raw_metadata_columns}
    metadata_columns = {}
    for column in raw_metadata_columns:
        kind = metadata_types[column]
        wire_values = metadata[column].tolist()
        nonmissing = [
            value for value in wire_values
            if value is not None
            and not (not isinstance(value, str) and pd.isna(value))]
        valid_kind = (
            (kind is None)
            or (kind == "null" and not nonmissing)
            or (kind == "str" and all(isinstance(value, str) for value in nonmissing))
            or (kind == "bool" and all(isinstance(value, (bool, np.bool_))
                                       for value in nonmissing))
            or (kind == "int" and all(
                isinstance(value, (int, np.integer))
                and not isinstance(value, (bool, np.bool_))
                for value in nonmissing))
            or (kind == "float" and all(
                isinstance(value, (float, np.floating))
                for value in nonmissing))
        )
        if not valid_kind:
            raise ValueError(
                f"metadata column {column!r} disagrees with metadata_types")
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
    source_provenance = manifest.get("provenance", {}) if schema_version == 2 else {}
    if not isinstance(source_provenance, dict):
        raise ValueError("encoded bundle provenance must be a mapping")
    return FeatureBatch(
        row_ids=tuple(metadata[id_column].astype(str)),
        arrays=values,
        roles=roles,
        orientations=orientations,
        feature_ids=feature_ids,
        metadata=metadata_columns,
        activation_polarity=str(manifest.get("activation_polarity") or "unknown"),
        code_semantics=str(manifest.get("code_semantics") or "custom"),
        provenance={
            **source_provenance,
            "encoded_bundle": {
                key: value for key, value in manifest.items() if key != "provenance"
            },
        },
    )


def save_feature_batch(batch: FeatureBatch, path, *, overwrite: bool = False) -> Path:
    """Transactionally save any aligned ``FeatureBatch`` as a schema-2 bundle."""
    if not isinstance(batch, FeatureBatch):
        raise ValueError("batch must be a FeatureBatch")
    if not isinstance(overwrite, bool):
        raise ValueError("overwrite must be boolean")
    root = Path(path).expanduser().resolve()
    if root.exists() and not root.is_dir():
        raise ValueError(f"feature bundle destination is not a directory: {root}")
    if root.exists() and not overwrite:
        raise FileExistsError(f"feature bundle destination already exists: {root}")
    if "row_id" in batch.metadata or "battle_id" in batch.metadata:
        raise ValueError("feature metadata must not redefine row_id or battle_id")
    for name in batch.arrays:
        if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
            raise ValueError(f"unsafe feature array name: {name!r}")
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = root.parent / f".{root.name}.tmp-{uuid.uuid4().hex}"
    backup = root.parent / f".{root.name}.bak-{uuid.uuid4().hex}"
    try:
        staging.mkdir()
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
        metadata.to_parquet(staging / "meta.parquet", index=False)
        metadata.to_parquet(staging / "battles.parquet", index=False)
        arrays = {}
        array_dtypes = {}
        for name, values in batch.arrays.items():
            source = np.asarray(values)
            dtype = np.dtype(bool) if source.dtype == bool else np.dtype(np.float32)
            array = np.asarray(source, dtype=dtype)
            np.save(staging / f"{name}.npy", array, allow_pickle=False)
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
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2))
        load_feature_batch(staging)
        if root.exists():
            os.replace(root, backup)
        try:
            os.replace(staging, root)
        except Exception:
            if backup.exists() and not root.exists():
                os.replace(backup, root)
            raise
        shutil.rmtree(backup, ignore_errors=True)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        if backup.exists() and root.exists():
            shutil.rmtree(backup, ignore_errors=True)
    return root


__all__ = ["load_feature_batch", "save_feature_batch"]
