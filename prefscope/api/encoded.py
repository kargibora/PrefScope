"""Simple directory persistence for backend-neutral feature batches."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np

from prefscope.core.features import FeatureBatch


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_feature_batch(
    batch: FeatureBatch,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Save arrays as ``.npy`` files plus one small JSON manifest."""
    if not isinstance(batch, FeatureBatch):
        raise TypeError("batch must be a FeatureBatch")
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{path.name}-", dir=path.parent))
    try:
        views = {}
        for index, name in enumerate(batch.arrays):
            filename = f"array_{index}.npy"
            np.save(staging / filename, batch.array(name), allow_pickle=False)
            views[name] = {
                "file": filename,
                "role": batch.roles[name],
                "orientation": batch.orientations[name],
            }
        manifest = {
            "schema_version": 1,
            "row_ids": list(batch.row_ids),
            "feature_ids": list(batch.feature_ids),
            "views": views,
            "view_order": list(batch.arrays),
            "metadata": _jsonable(batch.metadata),
            "activation_polarity": batch.activation_polarity,
            "code_semantics": batch.code_semantics,
            "provenance": _jsonable(batch.provenance),
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if path.exists():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        staging.replace(path)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return path


def load_feature_batch(
    path: str | Path,
    *,
    arrays: Sequence[str] | None = None,
) -> FeatureBatch:
    """Load a saved batch, optionally selecting named views."""
    root = Path(path)
    payload = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported FeatureBatch schema_version")
    views = payload.get("views")
    if not isinstance(views, dict) or not views:
        raise ValueError("FeatureBatch manifest needs a non-empty views object")
    view_order = payload.get("view_order", list(views))
    if not isinstance(view_order, list) or set(view_order) != set(views):
        raise ValueError("FeatureBatch manifest has invalid view_order")
    selected = view_order if arrays is None else list(arrays)
    if len(selected) != len(set(selected)):
        raise ValueError("requested array names must be unique")
    missing = [name for name in selected if name not in views]
    if missing:
        raise KeyError(f"unknown feature arrays: {missing}")
    loaded = {}
    roles = {}
    orientations = {}
    for name in selected:
        descriptor = views[name]
        filename = descriptor["file"]
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError("feature array filenames must be local names")
        loaded[name] = np.load(root / filename, allow_pickle=False)
        roles[name] = descriptor["role"]
        orientations[name] = descriptor["orientation"]
    provenance = dict(payload["provenance"])
    view_provenance = provenance.get("views")
    if isinstance(view_provenance, Mapping):
        provenance["views"] = {
            name: view_provenance[name]
            for name in selected
            if name in view_provenance
        }
    return FeatureBatch(
        row_ids=tuple(payload["row_ids"]),
        feature_ids=tuple(payload["feature_ids"]),
        arrays=loaded,
        roles=roles,
        orientations=orientations,
        metadata={key: tuple(values) for key, values in payload["metadata"].items()},
        activation_polarity=payload["activation_polarity"],
        code_semantics=payload["code_semantics"],
        provenance=provenance,
    )


__all__ = ["load_feature_batch", "save_feature_batch"]
