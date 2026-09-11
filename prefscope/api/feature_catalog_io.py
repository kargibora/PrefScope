"""Simple JSON persistence for :class:`FeatureCatalog`."""
from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import pandas as pd

from prefscope.api.feature_catalog import FeatureCatalog


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def encode_feature_catalog(catalog: FeatureCatalog) -> bytes:
    """Encode a catalog as ordinary versioned JSON bytes."""
    if not isinstance(catalog, FeatureCatalog):
        raise TypeError("catalog must be a FeatureCatalog")
    frame = catalog.to_frame()
    payload = {
        "schema_version": 1,
        "columns": list(frame.columns),
        "records": json.loads(frame.to_json(orient="records")),
        "provenance": _jsonable(catalog.provenance),
        "column_sources": _jsonable(catalog.column_sources),
    }
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def decode_feature_catalog(data: bytes | bytearray | str) -> FeatureCatalog:
    """Decode catalog JSON and validate it through ``FeatureCatalog``."""
    if isinstance(data, (bytes, bytearray)):
        data = bytes(data).decode("utf-8")
    if not isinstance(data, str):
        raise TypeError("catalog data must be bytes or a string")
    payload = json.loads(data)
    if not isinstance(payload, dict):
        raise ValueError("feature catalog payload must be a JSON object")
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported feature catalog schema_version")
    columns = payload["columns"]
    records = payload["records"]
    if not isinstance(columns, list) or not isinstance(records, list):
        raise ValueError("catalog columns and records must be lists")
    frame = pd.DataFrame.from_records(records, columns=columns)
    return FeatureCatalog(
        frame,
        provenance=payload["provenance"],
        column_sources=payload["column_sources"],
    )


def save_feature_catalog(
    catalog: FeatureCatalog,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Write a catalog JSON file."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = tempfile.NamedTemporaryFile(
        prefix=f".{path.name}-", dir=path.parent, delete=False
    )
    temporary = Path(temporary_file.name)
    try:
        with temporary_file:
            temporary_file.write(encode_feature_catalog(catalog))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def load_feature_catalog(path: str | Path) -> FeatureCatalog:
    """Load a catalog JSON file."""
    return decode_feature_catalog(Path(path).read_bytes())


__all__ = [
    "decode_feature_catalog",
    "encode_feature_catalog",
    "load_feature_catalog",
    "save_feature_catalog",
]
