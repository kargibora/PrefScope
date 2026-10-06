"""Compatibility aliases for the first prompt-pole derived view."""
from __future__ import annotations

from prefscope.api.derived_features import (
    DERIVED_VIEW_COORDINATE_SPACE,
    build_derived_catalog,
    derive_feature_matrix,
    load_derived_catalog,
    validate_derived_catalog,
)

PROMPT_POLE_CATALOG = "prompt_pole_catalog.json"
PROMPT_POLE_COORDINATE_SPACE = DERIVED_VIEW_COORDINATE_SPACE


def expand_prompt_poles(matrix, *, native_width=None):
    if matrix.role != "prompt":
        raise ValueError("prompt FeatureMatrix required for pole expansion")
    return derive_feature_matrix(matrix, "poles", native_width=native_width)


def build_prompt_pole_catalog(annotations, **kwargs):
    return build_derived_catalog(annotations, view_name="poles", **kwargs)


def load_prompt_pole_catalog(path, *, native_width, feature_space_id=None):
    return load_derived_catalog(path, native_width=native_width, view_name="poles", feature_space_id=feature_space_id)


def validate_prompt_pole_catalog(catalog, *, native_width, feature_space_id=None):
    return validate_derived_catalog(catalog, native_width=native_width, view_name="poles", feature_space_id=feature_space_id)


__all__ = [
    "PROMPT_POLE_CATALOG", "PROMPT_POLE_COORDINATE_SPACE",
    "build_prompt_pole_catalog", "expand_prompt_poles", "load_prompt_pole_catalog",
    "validate_prompt_pole_catalog",
]
