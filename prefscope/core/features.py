"""Aligned, torch-free feature matrices used by public analyses."""
from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from types import MappingProxyType
from typing import Mapping

import numpy as np

from prefscope.core.representation import (
    _aligned_metadata,
    validate_portable_mapping,
    validate_row_ids,
)


def validate_feature_ids(values, *, width: int | None = None) -> tuple[int, ...]:
    """Validate feature identities without lossy numeric coercion."""
    resolved = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError("feature_ids must contain non-boolean integers")
        resolved.append(int(value))
    ids = tuple(resolved)
    if len(set(ids)) != len(ids):
        raise ValueError("feature_ids must be unique")
    if width is not None and len(ids) != int(width):
        raise ValueError("feature_ids must match the feature width")
    return ids


def _validate_feature_batch_semantics(
    array_names,
    *,
    activation_polarity: object,
    code_semantics: object,
    provenance: Mapping[str, object],
) -> None:
    """Validate the schema-2 global and per-view semantics contract."""
    for name, value in (
        ("activation_polarity", activation_polarity),
        ("code_semantics", code_semantics),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")
    names = set(array_names)
    descriptors = provenance.get("views", {})
    if not isinstance(descriptors, Mapping):
        raise ValueError("feature batch provenance views must be a mapping")
    unknown = set(descriptors) - names
    if unknown:
        raise ValueError(
            "feature batch provenance has semantics for unknown arrays: "
            f"{sorted(unknown)}")
    for name, descriptor in descriptors.items():
        if not isinstance(descriptor, Mapping):
            raise ValueError(
                f"feature batch provenance view {name!r} must be a mapping")
        for semantic_field in ("activation_polarity", "code_semantics"):
            value = descriptor.get(semantic_field)
            if semantic_field in descriptor and (
                not isinstance(value, str) or not value.strip()
            ):
                raise ValueError(
                    f"{semantic_field} for feature array {name!r} must be a "
                    "non-empty string")


@dataclass(frozen=True, eq=False)
class FeatureMatrix:
    """One analysis-ready feature matrix with explicit semantic provenance.

    ``role`` describes the quantity, for example ``prompt``, ``response``, or
    ``response_difference``. It is metadata, not an inference claim. All rows
    have stable unique identifiers so independently produced matrices can be
    aligned safely rather than joined by position accidentally.
    """

    values: np.ndarray
    row_ids: tuple[str, ...]
    role: str = "custom"
    orientation: str = "unspecified"
    feature_ids: tuple[int, ...] | None = None
    metadata: Mapping[str, tuple[object, ...]] = field(default_factory=dict)
    activation_polarity: str = "unknown"
    code_semantics: str = "custom"
    provenance: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        source = np.asarray(self.values)
        ids = validate_row_ids(self.row_ids)
        if source.ndim != 2 or source.shape[0] != len(ids):
            raise ValueError(
                f"feature values must have shape ({len(ids)}, n_features), "
                f"got {source.shape}")
        if (
            source.shape[1] <= 0
            or not (source.dtype == bool or np.issubdtype(source.dtype, np.number))
            or np.issubdtype(source.dtype, np.complexfloating)
        ):
            raise ValueError(
                "feature values must be a non-empty real numeric/boolean matrix")
        dtype = bool if source.dtype == bool else np.float32
        with np.errstate(over="ignore", invalid="ignore"):
            matrix = np.array(source, dtype=dtype, order="C", copy=True)
        if not np.isfinite(matrix).all():
            raise ValueError(
                "feature values must be finite and representable in canonical dtype")
        matrix = np.frombuffer(
            matrix.tobytes(order="C"), dtype=matrix.dtype
        ).reshape(matrix.shape)
        if not isinstance(self.role, str) or not self.role:
            raise ValueError("feature role must be a non-empty string")
        if not isinstance(self.orientation, str) or not self.orientation:
            raise ValueError("feature orientation must be a non-empty string")
        feature_ids = (
            tuple(range(matrix.shape[1])) if self.feature_ids is None
            else validate_feature_ids(self.feature_ids, width=matrix.shape[1])
        )
        for name, value in (
            ("activation_polarity", self.activation_polarity),
            ("code_semantics", self.code_semantics),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        object.__setattr__(self, "values", matrix)
        object.__setattr__(self, "row_ids", ids)
        object.__setattr__(self, "feature_ids", feature_ids)
        object.__setattr__(self, "metadata", _aligned_metadata(self.metadata, len(ids)))
        object.__setattr__(
            self, "provenance", validate_portable_mapping(self.provenance, where="provenance"))

    @classmethod
    def from_presence(
        cls,
        presence,
        *,
        row_ids,
        role: str,
        metadata=None,
        provenance=None,
    ) -> "FeatureMatrix":
        """Build an explicitly semantic-presence matrix from ``concept_presence``.

        Raw nonzero activations must not use this constructor. ``presence`` is
        expected to expose aligned boolean ``values``, ``feature_ids``, and
        per-feature ``basis`` fields, as returned by PrefScope's PresenceMatrix.
        """
        values = np.asarray(getattr(presence, "values", None))
        raw_feature_ids = tuple(getattr(presence, "feature_ids", ()))
        basis = tuple(str(value) for value in getattr(presence, "basis", ()))
        calibrated = np.asarray(getattr(presence, "calibrated", ()), dtype=bool)
        if values.ndim != 2 or values.dtype != bool:
            raise ValueError("presence must expose a 2-D boolean values matrix")
        feature_ids = validate_feature_ids(raw_feature_ids, width=values.shape[1])
        if (
            len(feature_ids) != values.shape[1]
            or len(basis) != values.shape[1]
            or len(calibrated) != values.shape[1]
        ):
            raise ValueError(
                "presence feature_ids/basis/calibrated must match its feature width")
        if not calibrated.all() or any(value != "semantic_threshold" for value in basis):
            raise ValueError(
                "semantic FeatureMatrix conversion requires calibrated "
                "semantic_threshold presence for every feature")
        return cls(
            values=values,
            row_ids=tuple(row_ids),
            role=role,
            orientation="none",
            feature_ids=feature_ids,
            metadata=dict(metadata or {}),
            activation_polarity="nonnegative",
            code_semantics="semantic_presence",
            provenance={
                **dict(provenance or {}),
                "presence_basis": list(basis),
            },
        )

    @property
    def n_rows(self) -> int:
        return int(self.values.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.values.shape[1])


@dataclass(frozen=True, eq=False)
class FeatureBatch:
    """Several aligned feature quantities produced by one frozen lens."""

    row_ids: tuple[str, ...]
    arrays: Mapping[str, np.ndarray]
    roles: Mapping[str, str]
    orientations: Mapping[str, str] = field(default_factory=dict)
    feature_ids: tuple[int, ...] | None = None
    metadata: Mapping[str, tuple[object, ...]] = field(default_factory=dict)
    activation_polarity: str = "unknown"
    code_semantics: str = "custom"
    provenance: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ids = validate_row_ids(self.row_ids)
        arrays = dict(self.arrays)
        roles = dict(self.roles)
        orientations = dict(self.orientations)
        if not orientations:
            orientations = {name: "unspecified" for name in arrays}
        if not arrays:
            raise ValueError("feature batch must contain at least one array")
        if set(roles) != set(arrays):
            raise ValueError("feature batch roles must name every array exactly once")
        if set(orientations) != set(arrays):
            raise ValueError(
                "feature batch orientations must name every array exactly once")
        widths = set()
        checked = {}
        for name, value in arrays.items():
            if not isinstance(name, str) or not name:
                raise ValueError("feature array names must be non-empty strings")
            source = np.asarray(value)
            if source.ndim != 2 or source.shape[0] != len(ids):
                raise ValueError(
                    f"feature array {name!r} must have {len(ids)} rows, "
                    f"got {source.shape}")
            if (
                not (source.dtype == bool or np.issubdtype(source.dtype, np.number))
                or np.issubdtype(source.dtype, np.complexfloating)
                or source.shape[1] <= 0
            ):
                raise ValueError(
                    f"feature array {name!r} must be a non-empty real "
                    "numeric/boolean matrix")
            dtype = bool if source.dtype == bool else np.float32
            with np.errstate(over="ignore", invalid="ignore"):
                matrix = np.array(source, dtype=dtype, order="C", copy=True)
            if not np.isfinite(matrix).all():
                raise ValueError(
                    f"feature array {name!r} must be finite and representable in "
                    "canonical dtype")
            matrix = np.frombuffer(
                matrix.tobytes(order="C"), dtype=matrix.dtype
            ).reshape(matrix.shape)
            widths.add(matrix.shape[1])
            checked[name] = matrix
            if not isinstance(roles[name], str) or not roles[name].strip():
                raise ValueError(f"feature role for {name!r} must be non-empty")
            if (
                not isinstance(orientations[name], str)
                or not orientations[name].strip()
            ):
                raise ValueError(
                    f"feature orientation for {name!r} must be non-empty")
        if len(widths) != 1 or next(iter(widths)) <= 0:
            raise ValueError("all feature arrays must have the same positive width")
        width = next(iter(widths))
        feature_ids = (
            tuple(range(width)) if self.feature_ids is None
            else validate_feature_ids(self.feature_ids, width=width)
        )
        object.__setattr__(self, "row_ids", ids)
        object.__setattr__(self, "arrays", MappingProxyType(checked))
        object.__setattr__(self, "roles", MappingProxyType(roles))
        object.__setattr__(self, "orientations", MappingProxyType(orientations))
        provenance = validate_portable_mapping(self.provenance, where="provenance")
        _validate_feature_batch_semantics(
            checked,
            activation_polarity=self.activation_polarity,
            code_semantics=self.code_semantics,
            provenance=provenance,
        )
        object.__setattr__(self, "feature_ids", feature_ids)
        object.__setattr__(self, "metadata", _aligned_metadata(self.metadata, len(ids)))
        object.__setattr__(self, "provenance", provenance)

    def array(self, name: str) -> np.ndarray:
        try:
            return self.arrays[name]
        except KeyError:
            available = ", ".join(sorted(self.arrays))
            raise ValueError(
                f"feature batch has no array {name!r}; available: {available}") from None

    def matrix(self, name: str) -> FeatureMatrix:
        views = self.provenance.get("views", {})
        descriptor = views.get(name, {}) if isinstance(views, Mapping) else {}
        if not isinstance(descriptor, Mapping):
            raise ValueError(f"feature view provenance for {name!r} must be a mapping")
        return FeatureMatrix(
            values=self.array(name),
            row_ids=self.row_ids,
            role=self.roles[name],
            orientation=self.orientations[name],
            feature_ids=self.feature_ids,
            metadata=self.metadata,
            activation_polarity=str(
                descriptor.get("activation_polarity", self.activation_polarity)),
            code_semantics=str(descriptor.get("code_semantics", self.code_semantics)),
            provenance={
                **dict(self.provenance),
                "array_name": name,
                "view": dict(descriptor),
            },
        )


__all__ = ["FeatureMatrix", "FeatureBatch", "validate_feature_ids"]
