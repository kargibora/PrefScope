"""Typed proposed-label catalogs for sparse feature coordinates."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from types import MappingProxyType
from typing import Mapping

import numpy as np
import pandas as pd

from prefscope.api._feature_space import matrix_feature_space_identity
from prefscope.core.features import FeatureMatrix, validate_feature_ids
from prefscope.core.representation import validate_portable_mapping

CATALOG_SCHEMA_VERSION = 1
_RESERVED_COLUMNS = {
    "row_id",
    "rank",
    "feature_id",
    "activation",
    "abs_activation",
    "feature_role",
    "feature_orientation",
    "activation_polarity",
    "code_semantics",
}
_CATALOG_COLUMNS = {
    "feature_id",
    "name",
    "description",
    "source",
    "source_ref",
    "evidence_layer",
    "retrieval_status",
    "content_sha256",
}


@dataclass(frozen=True, eq=False, init=False)
class FeatureCatalog:
    """Immutable feature-ID annotations with explicit source provenance."""

    _table: pd.DataFrame = field(repr=False)
    provenance: Mapping[str, object]
    column_sources: Mapping[str, Mapping[str, object]]

    def __init__(
        self,
        table: pd.DataFrame,
        *,
        provenance: Mapping[str, object] | None = None,
        column_sources: Mapping[str, Mapping[str, object]] | None = None,
    ) -> None:
        if not isinstance(table, pd.DataFrame):
            raise ValueError("feature catalog table must be a pandas DataFrame")
        if "feature_id" not in table.columns:
            raise ValueError("feature catalog table needs a feature_id column")
        if any(not isinstance(name, str) or not name for name in table.columns):
            raise ValueError("feature catalog columns must be non-empty strings")
        if not table.columns.is_unique:
            raise ValueError("feature catalog columns must be unique")
        collisions = (set(table.columns) & _RESERVED_COLUMNS) - {"feature_id"}
        if collisions:
            raise ValueError(
                f"feature catalog columns conflict with activation fields: {sorted(collisions)}"
            )
        unknown = set(table.columns) - _CATALOG_COLUMNS
        if unknown:
            raise ValueError(
                "feature catalogs contain display labels only; unsupported columns: "
                f"{sorted(unknown)}"
            )

        frame = table.copy(deep=True).reset_index(drop=True)
        raw_ids = frame["feature_id"]
        if raw_ids.isna().any():
            raise ValueError("feature catalog feature_id values cannot be missing")
        if any(isinstance(value, (bool, np.bool_)) for value in raw_ids):
            raise ValueError("feature catalog feature_id values cannot be boolean")
        if any(not isinstance(value, Integral) for value in raw_ids):
            raise ValueError(
                "feature catalog feature_id values must be non-boolean integers"
            )
        feature_ids = validate_feature_ids(tuple(raw_ids))
        if any(feature_id < 0 for feature_id in feature_ids):
            raise ValueError("feature catalog feature_id values must be non-negative")
        frame["feature_id"] = np.asarray(feature_ids, dtype=np.int64)
        for column in set(frame.columns) - {"feature_id"}:
            present = frame[column].dropna()
            if any(not isinstance(value, str) for value in present):
                raise ValueError(
                    f"feature catalog {column} values must be strings or missing"
                )
        if "evidence_layer" in frame:
            layers = set(frame["evidence_layer"].dropna())
            if layers - {"proposed_label"}:
                raise ValueError(
                    "feature catalog evidence_layer can only be proposed_label"
                )
        if "content_sha256" in frame:
            invalid_digests = [
                value
                for value in frame["content_sha256"].dropna()
                if len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ]
            if invalid_digests:
                raise ValueError(
                    "feature catalog content_sha256 values must be lowercase SHA-256"
                )

        raw_provenance = dict(provenance or {})
        if raw_provenance.get("evidence_layer", "proposed_label") != "proposed_label":
            raise ValueError(
                "feature catalog provenance evidence_layer can only be proposed_label"
            )
        raw_provenance.setdefault("schema_version", CATALOG_SCHEMA_VERSION)
        if raw_provenance["schema_version"] != CATALOG_SCHEMA_VERSION:
            raise ValueError(
                f"feature catalog schema_version must be {CATALOG_SCHEMA_VERSION}"
            )
        feature_space_id = raw_provenance.get("feature_space_id")
        feature_space_status = raw_provenance.get("feature_space_status", "unbound")
        if feature_space_id is not None and (
            not isinstance(feature_space_id, str) or not feature_space_id.strip()
        ):
            raise ValueError("feature_space_id must be a non-empty string or None")
        if feature_space_status not in {
            "exact_weights",
            "declared_pinned_coordinate",
            "declared_unpinned",
            "unbound",
        }:
            raise ValueError("unknown feature_space_status")
        if feature_space_id is None and feature_space_status != "unbound":
            raise ValueError("a bound feature-space status needs feature_space_id")
        raw_provenance["feature_space_status"] = feature_space_status
        checked_provenance = validate_portable_mapping(
            raw_provenance, where="feature catalog provenance"
        )

        checked_sources = {}
        for column, source in dict(column_sources or {}).items():
            if column not in frame.columns or column == "feature_id":
                raise ValueError(
                    f"feature catalog source names unknown annotation column {column!r}"
                )
            if not isinstance(source, Mapping):
                raise ValueError(
                    f"feature catalog source for {column!r} must be a mapping"
                )
            if source.get("evidence_layer", "proposed_label") != "proposed_label":
                raise ValueError(
                    "feature catalog source evidence_layer can only be proposed_label"
                )
            checked_sources[column] = validate_portable_mapping(
                source, where=f"feature catalog source {column!r}"
            )

        object.__setattr__(self, "_table", frame)
        object.__setattr__(self, "provenance", MappingProxyType(checked_provenance))
        object.__setattr__(
            self,
            "column_sources",
            MappingProxyType(
                {
                    name: MappingProxyType(value)
                    for name, value in checked_sources.items()
                }
            ),
        )

    @classmethod
    def from_lens(cls, lens) -> "FeatureCatalog":
        """Build a complete ordered proposed-label catalog for one lens."""
        width = int(lens.backend.m_total)
        annotations = lens.feature_table
        if not isinstance(annotations, pd.DataFrame):
            raise ValueError("lens feature_table must be a pandas DataFrame")
        raw_ids = annotations["feature_id"]
        if raw_ids.isna().any() or any(
            isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
            for value in raw_ids
        ):
            raise ValueError(
                "lens feature_table feature IDs must be non-boolean integers"
            )
        feature_ids = validate_feature_ids(tuple(raw_ids))
        invalid = [
            feature_id
            for feature_id in feature_ids
            if feature_id < 0 or feature_id >= width
        ]
        if invalid:
            raise ValueError(
                f"lens feature_table contains IDs outside [0, {width}): {invalid[:10]}"
            )
        annotations = annotations.copy()
        annotations["feature_id"] = np.asarray(feature_ids, dtype=np.int64)
        names = pd.DataFrame({"feature_id": np.arange(width, dtype=np.int64)})
        label_column = next(
            (column for column in ("concept", "name") if column in annotations), None
        )
        if label_column is not None:
            incoming = annotations[["feature_id", label_column]].rename(
                columns={label_column: "name"}
            )
            names = names.merge(
                incoming, on="feature_id", how="left", validate="one_to_one"
            )
        identity = lens.feature_space_identity
        source = {
            "kind": "lens_annotation",
            "evidence_layer": "proposed_label",
            **identity,
        }
        return cls(
            names,
            provenance={
                "schema_version": CATALOG_SCHEMA_VERSION,
                "source_kind": "lens_feature_names",
                "input_rep": str(lens.input_rep),
                "n_features": width,
                **identity,
            },
            column_sources={"name": source} if "name" in names else {},
        )

    @classmethod
    def from_mapping(
        cls,
        labels: Mapping[int, str],
        *,
        column: str = "description",
        provenance: Mapping[str, object] | None = None,
    ) -> "FeatureCatalog":
        if column not in {"name", "description"}:
            raise ValueError(
                "feature catalog annotation column must be name or description"
            )
        return cls(
            pd.DataFrame(
                {
                    "feature_id": list(labels),
                    column: list(labels.values()),
                }
            ),
            provenance=provenance,
        )

    @property
    def feature_ids(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self._table["feature_id"])

    @property
    def labels(self) -> Mapping[int, str]:
        labels = {}
        for row in self._table.itertuples(index=False):
            values = row._asdict()
            for column in ("name", "description"):
                value = values.get(column)
                if value is not None and not pd.isna(value) and str(value).strip():
                    labels[int(values["feature_id"])] = str(value)
                    break
        return MappingProxyType(labels)

    @property
    def feature_space_id(self) -> str | None:
        value = self.provenance.get("feature_space_id")
        return str(value) if value is not None else None

    @property
    def feature_space_status(self) -> str:
        return str(self.provenance["feature_space_status"])

    def validate_for(
        self,
        matrix: FeatureMatrix,
        *,
        require_identity: bool = False,
        require_exact: bool = False,
        require_complete: bool = False,
    ) -> None:
        if not isinstance(matrix, FeatureMatrix):
            raise ValueError("matrix must be a FeatureMatrix")
        if require_complete:
            catalog_ids = set(self.feature_ids)
            missing = [
                feature_id
                for feature_id in matrix.feature_ids
                if feature_id not in catalog_ids
            ]
            if missing:
                raise ValueError(
                    f"feature catalog is missing feature IDs {missing[:10]}"
                )
        matrix_id, matrix_status = matrix_feature_space_identity(matrix)
        if self.feature_space_id is not None and matrix_id is not None:
            if self.feature_space_id != matrix_id:
                raise ValueError(
                    "feature catalog and matrix use different feature spaces"
                )
        elif require_identity or require_exact:
            raise ValueError("feature-space validation needs identity on both inputs")
        if require_exact and (
            self.feature_space_status != "exact_weights"
            or matrix_status != "exact_weights"
        ):
            raise ValueError("feature-space identity is declared but not exact weights")

    def __len__(self) -> int:
        return len(self._table)

    def to_frame(self) -> pd.DataFrame:
        return self._table.copy(deep=True)

    def select(self, feature_ids, *, strict: bool = True) -> "FeatureCatalog":
        selected = validate_feature_ids(tuple(feature_ids))
        if any(feature_id < 0 for feature_id in selected):
            raise ValueError("selected feature IDs must be non-negative")
        indexed = self._table.set_index("feature_id")
        missing = [
            feature_id for feature_id in selected if feature_id not in indexed.index
        ]
        if strict and missing:
            raise ValueError(
                f"feature catalog has no annotations for IDs {missing[:10]}"
            )
        frame = indexed.reindex(selected).reset_index()
        return FeatureCatalog(
            frame,
            provenance={
                **dict(self.provenance),
                "selection_count": len(selected),
            },
            column_sources=self.column_sources,
        )

    def merge(self, other: "FeatureCatalog") -> "FeatureCatalog":
        """Merge catalogs by ID; nonmissing values from ``other`` take precedence."""
        if not isinstance(other, FeatureCatalog):
            raise ValueError("other must be a FeatureCatalog")
        if (
            self.feature_space_id is not None
            and other.feature_space_id is not None
            and self.feature_space_id != other.feature_space_id
        ):
            raise ValueError("cannot merge catalogs from different feature spaces")
        feature_space_id = self.feature_space_id or other.feature_space_id
        statuses = {self.feature_space_status, other.feature_space_status} - {"unbound"}
        feature_space_status = (
            next(iter(statuses))
            if len(statuses) == 1
            else "declared_unpinned"
            if statuses
            else "unbound"
        )
        ordered_ids = tuple(dict.fromkeys((*self.feature_ids, *other.feature_ids)))
        left = self._table.set_index("feature_id").reindex(ordered_ids)
        right = other._table.set_index("feature_id").reindex(ordered_ids)
        columns = list(dict.fromkeys((*left.columns, *right.columns)))
        merged = pd.DataFrame(index=pd.Index(ordered_ids, name="feature_id"))
        for column in columns:
            old = (
                left[column]
                if column in left
                else pd.Series(index=merged.index, dtype=object)
            )
            new = (
                right[column]
                if column in right
                else pd.Series(index=merged.index, dtype=object)
            )
            merged[column] = new.combine_first(old)
        sources = dict(self.column_sources)
        for column, source in other.column_sources.items():
            if column in sources and dict(sources[column]) != dict(source):
                sources[column] = {
                    "kind": "merged",
                    "sources": [dict(sources[column]), dict(source)],
                }
            else:
                sources[column] = dict(source)
        return FeatureCatalog(
            merged.reset_index(),
            provenance={
                "schema_version": CATALOG_SCHEMA_VERSION,
                "source_kind": "merged_catalog",
                "merge_precedence": "right_nonmissing",
                "sources": [dict(self.provenance), dict(other.provenance)],
                "feature_space_id": feature_space_id,
                "feature_space_status": feature_space_status,
            },
            column_sources=sources,
        )


__all__ = ["CATALOG_SCHEMA_VERSION", "FeatureCatalog"]
