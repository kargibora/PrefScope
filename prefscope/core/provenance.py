"""Portable, order-sensitive hashes for aligned metadata and numerical arrays."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd


_HASH_CHUNK_ROWS = 4096
_HASH_CHUNK_ELEMENTS = 1_048_576


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
    *,
    chunk_rows: int = _HASH_CHUNK_ROWS,
    chunk_elements: int = _HASH_CHUNK_ELEMENTS,
) -> str:
    """Bind ordered metadata rows to named arrays with bounded working memory.

    Numerical arrays are validated and hashed in one row-major pass. Both the
    row count and element count bound each canonical copy and finiteness mask,
    including arrays whose individual rows are very wide.
    """
    if not isinstance(metadata, pd.DataFrame):
        raise ValueError("metadata must be a pandas DataFrame")
    if type(chunk_rows) is not int or chunk_rows <= 0:
        raise ValueError("chunk_rows must be a positive integer")
    if type(chunk_elements) is not int or chunk_elements <= 0:
        raise ValueError("chunk_elements must be a positive integer")
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
        if values.ndim != 2 or values.shape[1] <= 0 or values.dtype not in {
            np.dtype(np.float32), np.dtype(bool)}:
            raise ValueError(
                f"{name} must be a canonical float32 or boolean 2-D array "
                "before hashing")
        is_boolean = values.dtype == np.dtype(bool)
        dtype_tag = "bool" if is_boolean else "float32"
        digest.update(str(name).encode("utf-8"))
        digest.update(b"\0")
        digest.update(json.dumps(list(values.shape), separators=(",", ":")).encode())
        digest.update(f"\0{dtype_tag}\0".encode())
        width = int(values.shape[1])
        rows_per_chunk = min(chunk_rows, max(1, chunk_elements // width))
        for row_start in range(0, len(values), rows_per_chunk):
            row_stop = min(row_start + rows_per_chunk, len(values))
            if (row_stop - row_start) * width <= chunk_elements:
                pieces = (values[row_start:row_stop],)
            else:
                # One row is wider than the element budget. Split that row by
                # columns while preserving canonical C-order byte traversal.
                pieces = (
                    values[row_start:row_stop, column_start:column_start + chunk_elements]
                    for column_start in range(0, width, chunk_elements)
                )
            for source in pieces:
                if is_boolean:
                    chunk = np.ascontiguousarray(source, dtype=bool)
                else:
                    chunk = np.ascontiguousarray(source, dtype="<f4")
                    if not np.isfinite(chunk).all():
                        raise ValueError(f"{name} must contain only finite values")
                digest.update(memoryview(chunk).cast("B"))
    return digest.hexdigest()


__all__ = ["canonical_metadata_value", "ordered_dataset_hash"]
