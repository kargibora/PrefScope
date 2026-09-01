"""Portable, order-sensitive hashes for aligned metadata and numerical arrays."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd


def canonical_metadata_value(value):
    """Return a JSON-safe deterministic value for provenance hashing."""
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return value.isoformat()
    if isinstance(value, bytes):
        return {"__bytes__": value.hex()}
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("metadata mapping keys must be strings")
        return {
            key: canonical_metadata_value(item)
            for key, item in sorted(value.items())
        }
    if isinstance(value, (list, tuple, np.ndarray)):
        return [canonical_metadata_value(item) for item in value]
    if isinstance(value, float):
        if np.isnan(value):
            return None
        if np.isposinf(value):
            return {"__float__": "inf"}
        if np.isneginf(value):
            return {"__float__": "-inf"}
    if isinstance(value, (str, int, float, bool)):
        return value
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, np.bool_)) and missing:
        return None
    raise ValueError(
        f"metadata value type {type(value).__qualname__} is not canonically supported")


def ordered_dataset_hash(
    metadata: pd.DataFrame,
    arrays: dict[str, np.ndarray],
) -> str:
    """Bind ordered metadata rows to named, ordered numeric arrays with SHA-256."""
    if not isinstance(metadata, pd.DataFrame):
        raise ValueError("metadata must be a pandas DataFrame")
    digest = hashlib.sha256()
    digest.update(b"prefscope-dataset-v1\0")
    columns = list(metadata.columns)
    header = json.dumps(columns, ensure_ascii=False, separators=(",", ":"))
    digest.update(header.encode("utf-8"))
    digest.update(b"\0")
    for row in metadata.itertuples(index=False, name=None):
        payload = [canonical_metadata_value(value) for value in row]
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)

    for name in sorted(arrays):
        values = np.asarray(arrays[name])
        if values.ndim != 2 or values.dtype not in {
            np.dtype(np.float32), np.dtype(bool)}:
            raise ValueError(
                f"{name} must be a canonical float32 or boolean 2-D array "
                "before hashing")
        if not np.isfinite(values).all():
            raise ValueError(f"{name} must contain only finite values")
        if values.dtype == np.dtype(bool):
            array = np.asarray(values, dtype=bool)
            dtype_tag = "bool"
        else:
            array = np.asarray(values, dtype="<f4")
            dtype_tag = "float32"
        digest.update(str(name).encode("utf-8"))
        digest.update(b"\0")
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
        digest.update(f"\0{dtype_tag}\0".encode())
        for start in range(0, len(array), 4096):
            chunk = np.ascontiguousarray(array[start:start + 4096])
            digest.update(memoryview(chunk).cast("B"))
    return digest.hexdigest()


__all__ = ["canonical_metadata_value", "ordered_dataset_hash"]
