"""Declared, deterministic derived feature views."""
from __future__ import annotations

from pathlib import Path
from numbers import Integral
from collections.abc import Mapping

import numpy as np
import pandas as pd

from prefscope.api.feature_catalog import FeatureCatalog
from prefscope.api.feature_catalog_io import decode_feature_catalog
from prefscope.artifacts import DERIVED_FEATURE_CATALOG
from prefscope.core.features import FeatureMatrix
from prefscope.core.features import validate_feature_ids

DERIVED_VIEW_COORDINATE_SPACE = "derived_feature_view-v1"
POLE_ORDER = ("positive", "negative")
POLE_TRANSFORM = "signed_to_poles"


def _view_spec(view, *, source_view: str | None = None) -> dict:
    if isinstance(view, str):
        transform = POLE_TRANSFORM if view == "poles" else view
        return {"name": view, "transform": transform, "source_view": source_view}
    if not isinstance(view, Mapping):
        raise ValueError("derived view must be a name or mapping")
    spec = dict(view)
    if not isinstance(spec.get("name"), str) or not spec["name"].strip():
        raise ValueError("derived view needs a non-empty name")
    if not isinstance(spec.get("transform"), str) or not spec["transform"].strip():
        raise ValueError("derived view needs a transform")
    return spec


def derive_feature_matrix(
    matrix: FeatureMatrix,
    view,
    *,
    native_width: int | None = None,
) -> FeatureMatrix:
    """Apply a declared deterministic transform to a feature matrix."""
    if not isinstance(matrix, FeatureMatrix):
        raise ValueError("matrix must be a FeatureMatrix")
    spec = _view_spec(view)
    transform = spec["transform"]
    if transform != POLE_TRANSFORM:
        raise ValueError(f"unsupported derived feature transform {transform!r}")
    if matrix.activation_polarity == "nonnegative":
        raise ValueError(
            "signed_to_poles requires a native matrix with signed activations")
    width = matrix.n_features if native_width is None else native_width
    if not isinstance(width, Integral) or isinstance(width, bool) or width <= 0:
        raise ValueError("native_width must be a positive integer")
    if native_width is None and matrix.feature_ids != tuple(range(matrix.n_features)):
        raise ValueError("native_width is required for a selected feature matrix")
    native_ids = validate_feature_ids(matrix.feature_ids)
    if any(feature_id < 0 or feature_id >= int(width) for feature_id in native_ids):
        raise ValueError(f"feature IDs must be inside [0, {int(width)})")
    values = np.asarray(matrix.values, dtype=np.float32)
    output = np.empty((matrix.n_rows, 2 * len(native_ids)), dtype=np.float32)
    output[:, 0::2] = np.maximum(values, 0)
    output[:, 1::2] = np.maximum(-values, 0)
    virtual_ids = tuple(
        virtual_id
        for native_id in native_ids
        for virtual_id in (2 * native_id, 2 * native_id + 1)
    )
    provenance = dict(matrix.provenance)
    provenance.update({
        "coordinate_space": DERIVED_VIEW_COORDINATE_SPACE,
        "derived_view": spec.get("name", "derived"),
        "derived_transform": transform,
        "native_width": int(width),
        "native_feature_ids": list(native_ids),
        "pole_order": list(POLE_ORDER),
        "pole_encoding": "2 * feature_id + (pole == negative)",
    })
    return FeatureMatrix(
        output,
        matrix.row_ids,
        role=matrix.role,
        orientation="none",
        feature_ids=virtual_ids,
        metadata=matrix.metadata,
        activation_polarity="nonnegative",
        code_semantics="derived_strength",
        provenance=provenance,
    )


def build_derived_catalog(
    annotations: pd.DataFrame,
    *,
    native_width: int,
    view_name: str,
    transform: str = POLE_TRANSFORM,
    feature_space_id: str | None = None,
    source_name: str = "derived_feature_annotations.csv",
    source_sha256: str | None = None,
) -> FeatureCatalog:
    """Build a complete virtual catalog for one declared derived view."""
    if not isinstance(annotations, pd.DataFrame):
        raise ValueError("derived annotations must be a pandas DataFrame")
    if transform != POLE_TRANSFORM:
        raise ValueError(f"unsupported derived feature transform {transform!r}")
    if not isinstance(view_name, str) or not view_name.strip():
        raise ValueError("view_name must be a non-empty string")
    if not isinstance(native_width, Integral) or isinstance(native_width, bool) or native_width <= 0:
        raise ValueError("native_width must be a positive integer")
    frame = annotations.copy(deep=True)
    if "feature_space_id" in frame.columns and feature_space_id is not None:
        values = {str(value) for value in frame["feature_space_id"].dropna() if str(value)}
        if values and values != {feature_space_id}:
            raise ValueError("derived annotations do not match the feature space")
    aliases = {"concept_name": "name", "reason": "description", "naming_status": "status"}
    for old, new in aliases.items():
        if old in frame.columns and new not in frame.columns:
            frame = frame.rename(columns={old: new})
    if "pole" in frame.columns:
        if "feature_id" not in frame.columns:
            raise ValueError("derived pole annotations need feature_id")
        raw_ids = frame["feature_id"].tolist()
        if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
               for value in raw_ids):
            raise ValueError("derived feature IDs must be non-boolean integers")
        ids = tuple(int(value) for value in raw_ids)
        if any(value < 0 or value >= int(native_width) for value in ids):
            raise ValueError(f"derived feature IDs must be inside [0, {int(native_width)})")
        poles = tuple(str(value) for value in frame["pole"])
        if any(pole not in POLE_ORDER for pole in poles):
            raise ValueError("pole values must be 'positive' or 'negative'")
        if frame.duplicated(["feature_id", "pole"]).any():
            raise ValueError("derived pole annotations must have unique coordinate/pole rows")
        records = pd.DataFrame({"feature_id": np.arange(2 * int(native_width), dtype=np.int64)})
        for column in ("name", "description", "status"):
            records[column] = pd.NA
            if column not in frame:
                continue
            for source_id, pole, value in zip(ids, poles, frame[column], strict=True):
                records.loc[2 * source_id + (pole == "negative"), column] = value
    else:
        records = frame.copy()
        if "feature_id" not in records:
            raise ValueError("derived annotations need feature_id")
        if set(records["feature_id"].dropna()) != set(range(2 * int(native_width))):
            raise ValueError("derived annotations must cover every virtual feature ID")
        records["feature_id"] = np.asarray(records["feature_id"], dtype=np.int64)
    provenance = {
        "schema_version": 1,
        "source_kind": "derived_feature_annotations",
        "coordinate_space": DERIVED_VIEW_COORDINATE_SPACE,
        "view_name": view_name,
        "transform": transform,
        "native_width": int(native_width),
        "native_feature_space_id": feature_space_id,
        "feature_space_id": feature_space_id,
        "feature_space_status": "declared_pinned_coordinate" if feature_space_id else "unbound",
        "feature_id_scheme": "2 * feature_id + (pole == negative)",
        "pole_encoding": "2 * feature_id + (pole == negative)",
        "pole_order": list(POLE_ORDER),
    }
    if source_sha256 is not None:
        provenance.update({"source_artifact": source_name, "source_sha256": source_sha256})
    return FeatureCatalog(
        records,
        provenance=provenance,
        column_sources={
            column: {
                "kind": "derived_feature_annotations",
                "artifact": source_name,
                "evidence_layer": "proposed_label",
                **({"content_sha256": source_sha256} if source_sha256 else {}),
            }
            for column in ("name", "description", "status") if column in records
        },
    )


def validate_derived_catalog(catalog: FeatureCatalog, *, native_width: int,
                             view_name: str, transform: str = POLE_TRANSFORM,
                             feature_space_id: str | None = None) -> FeatureCatalog:
    if not isinstance(catalog, FeatureCatalog):
        raise ValueError("derived catalog must be a FeatureCatalog")
    provenance = catalog.provenance
    if provenance.get("coordinate_space") != DERIVED_VIEW_COORDINATE_SPACE:
        raise ValueError("catalog is not a derived feature catalog")
    if provenance.get("view_name") != view_name or provenance.get("transform") != transform:
        raise ValueError("derived catalog does not match the declared view")
    if provenance.get("native_width") != int(native_width):
        raise ValueError("derived catalog native width does not match the lens")
    native_space_id = provenance.get("native_feature_space_id")
    if provenance.get("feature_space_id") != native_space_id:
        raise ValueError("derived catalog has inconsistent feature-space identity")
    if feature_space_id is not None and native_space_id != feature_space_id:
        raise ValueError("derived catalog does not match the native feature space")
    if transform == POLE_TRANSFORM:
        if (provenance.get("feature_id_scheme") != "2 * feature_id + (pole == negative)"
                or provenance.get("pole_encoding") != "2 * feature_id + (pole == negative)"):
            raise ValueError("derived catalog has an invalid feature ID scheme")
        if tuple(provenance.get("pole_order", ())) != POLE_ORDER:
            raise ValueError("derived catalog has an invalid pole order")
    if catalog.feature_ids != tuple(range(2 * int(native_width))):
        raise ValueError("derived catalog must cover every virtual feature ID")
    return catalog


def load_derived_catalog(path, *, native_width: int, view_name: str | None = None,
                         transform: str = POLE_TRANSFORM,
                         feature_space_id: str | None = None) -> FeatureCatalog:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path, keep_default_na=False, na_values=[""])
        catalog = build_derived_catalog(
            frame, native_width=native_width, view_name=view_name, transform=transform,
            feature_space_id=feature_space_id, source_name=path.name,
        )
    else:
        catalog = decode_feature_catalog(path.read_bytes())
        if view_name is None:
            view_name = catalog.provenance.get("view_name")
        if not isinstance(view_name, str):
            raise ValueError("derived catalog does not declare a view_name")
    return validate_derived_catalog(
        catalog, native_width=native_width, view_name=view_name,
        transform=transform, feature_space_id=feature_space_id)


# Backward-compatible name for the first built-in transform.
def expand_prompt_poles(matrix: FeatureMatrix, *, native_width: int | None = None) -> FeatureMatrix:
    return derive_feature_matrix(matrix, "poles", native_width=native_width)


__all__ = [
    "DERIVED_FEATURE_CATALOG", "DERIVED_VIEW_COORDINATE_SPACE", "POLE_TRANSFORM",
    "build_derived_catalog", "derive_feature_matrix", "expand_prompt_poles",
    "load_derived_catalog", "validate_derived_catalog",
]
