"""Stable contracts for task-centered, aligned dataset analyses."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import json
import re
from types import MappingProxyType
from typing import Mapping

import numpy as np
import pandas as pd

from prefscope.analysis.grouping import (
    factorize_group_ids,
    resolve_group_ids,
    validate_group_ids,
)
from prefscope.analysis.outcomes import NormalizedOutcomes, OutcomeKind, normalize_outcomes
from prefscope.core.features import FeatureMatrix
from prefscope.core.table_schema import TableContract
from prefscope.core.representation import validate_portable_mapping, validate_row_ids


@dataclass(frozen=True, eq=False)
class OutcomeSpec:
    """One aligned outcome declaration for a high-level analysis plan."""

    values: object
    row_ids: tuple[object, ...]
    kind: OutcomeKind
    names: tuple[str, ...] | None = None
    normalization: str = "auto"

    def __post_init__(self) -> None:
        ids = validate_row_ids(self.row_ids)
        normalized = self.normalize()
        if normalized.n_rows != len(ids):
            raise ValueError("outcome row_ids must have one entry per outcome row")
        object.__setattr__(self, "row_ids", ids)

    @classmethod
    def from_feature_batch(
        cls,
        batch,
        column: str = "pref",
        *,
        kind: OutcomeKind = "preference",
        names: tuple[str, ...] | None = None,
        normalization: str = "auto",
    ) -> "OutcomeSpec":
        """Create an aligned outcome from one ``FeatureBatch`` metadata column."""
        from prefscope.core.features import FeatureBatch

        if not isinstance(batch, FeatureBatch):
            raise ValueError("batch must be a FeatureBatch")
        if column not in batch.metadata:
            raise ValueError(f"feature batch metadata has no column {column!r}")
        return cls(
            values=batch.metadata[column], row_ids=batch.row_ids, kind=kind,
            names=names, normalization=normalization,
        )

    def normalize(self) -> NormalizedOutcomes:
        if isinstance(self.values, NormalizedOutcomes):
            if self.names is not None or self.normalization != "auto":
                raise ValueError(
                    "names/normalization cannot override an already normalized outcome")
            if self.values.kind != self.kind:
                raise ValueError(
                    f"OutcomeSpec kind {self.kind!r} disagrees with normalized "
                    f"kind {self.values.kind!r}")
            return self.values
        return normalize_outcomes(
            self.values,
            kind=self.kind,
            names=self.names,
            normalization=self.normalization,
        )


@dataclass(frozen=True, eq=False)
class PairedOutcomeSpec:
    """Two exactly aligned outcome views for a B-minus-A comparison."""

    values_a: object
    values_b: object
    row_ids: tuple[object, ...]
    kind: OutcomeKind
    names: tuple[str, ...] | None = None
    side_a: str = "a"
    side_b: str = "b"
    interpretation: str = "higher values mean more of the declared outcome"
    _normalized_a: NormalizedOutcomes = field(init=False, repr=False, compare=False)
    _normalized_b: NormalizedOutcomes = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.side_a, str)
            or not isinstance(self.side_b, str)
            or not self.side_a
            or not self.side_b
            or self.side_a == self.side_b
        ):
            raise ValueError("side_a and side_b must be distinct non-empty labels")
        if not isinstance(self.interpretation, str) or not self.interpretation.strip():
            raise ValueError("interpretation must describe the common outcome direction")
        ids = validate_row_ids(self.row_ids)
        side_a = normalize_outcomes(
            self.values_a, kind=self.kind, names=self.names, normalization="none")
        side_b = normalize_outcomes(
            self.values_b, kind=self.kind, names=self.names, normalization="none")
        if side_a.n_rows != len(ids) or side_b.n_rows != len(ids):
            raise ValueError("paired outcome row_ids must align to both outcome sides")
        if side_a.names != side_b.names:
            raise ValueError("paired outcome sides must share attribute names")
        object.__setattr__(self, "row_ids", ids)
        object.__setattr__(self, "names", side_a.names)
        object.__setattr__(self, "_normalized_a", side_a)
        object.__setattr__(self, "_normalized_b", side_b)

    @property
    def normalized_a(self) -> NormalizedOutcomes:
        return self._normalized_a

    @property
    def normalized_b(self) -> NormalizedOutcomes:
        return self._normalized_b


@dataclass(frozen=True, eq=False)
class AnalysisDataset:
    """Aligned feature sets, outcomes, and independent-group identifiers."""

    features: Mapping[str, FeatureMatrix]
    outcomes: Mapping[str, OutcomeSpec]
    group_ids: object | None = None
    paired_outcomes: Mapping[str, PairedOutcomeSpec] = field(default_factory=dict)
    _group_source: str = field(init=False, repr=False, compare=False)
    _row_ids: tuple[str, ...] = field(init=False, repr=False, compare=False)
    _normalized_outcomes: Mapping[str, NormalizedOutcomes] = field(
        init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        features = dict(self.features)
        outcomes = dict(self.outcomes)
        paired_outcomes = dict(self.paired_outcomes)
        if not features and not outcomes and not paired_outcomes:
            raise ValueError(
                "analysis dataset needs at least one feature or outcome set")
        for name in (*features, *outcomes, *paired_outcomes):
            if not isinstance(name, str) or not name:
                raise ValueError("feature and outcome names must be non-empty strings")
        if not all(isinstance(value, FeatureMatrix) for value in features.values()):
            raise ValueError("every feature set must be a FeatureMatrix")
        if not all(isinstance(value, OutcomeSpec) for value in outcomes.values()):
            raise ValueError("every outcome must be an OutcomeSpec")
        if not all(
            isinstance(value, PairedOutcomeSpec) for value in paired_outcomes.values()
        ):
            raise ValueError("every paired outcome must be a PairedOutcomeSpec")
        if features:
            first = next(iter(features.values()))
            rows = first.row_ids
        elif outcomes:
            rows = next(iter(outcomes.values())).row_ids
            first = None
        else:
            rows = next(iter(paired_outcomes.values())).row_ids
            first = None
        for name, matrix in features.items():
            if matrix.row_ids != rows:
                raise ValueError(
                    f"feature set {name!r} row_ids are not exactly aligned; "
                    "align explicitly before analysis")
        normalized = {}
        for name, spec in outcomes.items():
            if spec.row_ids != rows:
                raise ValueError(
                    f"outcome {name!r} row_ids are not exactly aligned to dataset")
            normalized[name] = spec.normalize()
        for name, spec in paired_outcomes.items():
            if spec.row_ids != rows:
                raise ValueError(
                    f"paired outcome {name!r} row_ids are not exactly aligned to dataset")
        if self.group_ids is not None:
            groups = validate_group_ids(self.group_ids, len(rows))
            group_source = "explicit"
        elif features:
            candidates = []
            for feature_name, matrix in features.items():
                if not matrix.metadata:
                    continue
                metadata = pd.DataFrame(dict(matrix.metadata))
                candidate = resolve_group_ids(metadata)
                if candidate is not None:
                    source = (
                        "canonical_group_id" if "group_id" in metadata.columns
                        else "normalized_prompt_hash")
                    candidates.append((feature_name, candidate, source))
            if candidates:
                candidates.sort(key=lambda value: value[2] != "canonical_group_id")
                _, groups, group_source = candidates[0]
                reference_codes, _ = factorize_group_ids(groups)
                for feature_name, candidate, _ in candidates[1:]:
                    candidate_codes, _ = factorize_group_ids(candidate)
                    if not np.array_equal(reference_codes, candidate_codes):
                        raise ValueError(
                            "feature metadata imply conflicting independent-group "
                            f"partitions; conflict at {feature_name!r}")
            else:
                groups = None
                group_source = "row"
        else:
            groups = None
            group_source = "row"
        object.__setattr__(self, "features", MappingProxyType(features))
        object.__setattr__(self, "outcomes", MappingProxyType(outcomes))
        object.__setattr__(self, "paired_outcomes", MappingProxyType(paired_outcomes))
        object.__setattr__(self, "group_ids", groups)
        object.__setattr__(self, "_group_source", group_source)
        object.__setattr__(self, "_row_ids", rows)
        object.__setattr__(self, "_normalized_outcomes", MappingProxyType(normalized))

    @property
    def row_ids(self) -> tuple[str, ...]:
        return self._row_ids

    @property
    def n_rows(self) -> int:
        return len(self.row_ids)

    @property
    def group_source(self) -> str:
        return self._group_source

    def normalized_outcome(self, name: str) -> NormalizedOutcomes:
        try:
            return self._normalized_outcomes[name]
        except KeyError:
            raise ValueError(f"unknown outcome {name!r}") from None


@dataclass(frozen=True, eq=False)
class AnalysisArtifact:
    """One named table with its estimand and multiplicity/provenance metadata."""

    name: str
    table: pd.DataFrame
    estimand: str
    metadata: Mapping[str, object] = field(default_factory=dict)
    table_contract: TableContract | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.name, str)
            or not re.fullmatch(r"[a-z][a-z0-9_]*", self.name)
        ):
            raise ValueError(
                "artifact name must use portable lower_snake_case")
        if not isinstance(self.table, pd.DataFrame):
            raise ValueError("artifact table must be a pandas DataFrame")
        columns = tuple(self.table.columns)
        if (
            len(set(columns)) != len(columns)
            or any(not isinstance(column, str) or not column for column in columns)
        ):
            raise ValueError("artifact table columns must be unique non-empty strings")
        if not isinstance(self.estimand, str) or not self.estimand:
            raise ValueError("artifact estimand must be a non-empty string")
        if self.table_contract is not None:
            if not isinstance(self.table_contract, TableContract):
                raise ValueError("artifact table_contract must be a TableContract")
            if self.table_contract.schema_name != self.name:
                raise ValueError(
                    "artifact name must match its table contract schema_name")
            self.table_contract.validate(self.table)
        object.__setattr__(self, "table", self.table.copy(deep=True))
        object.__setattr__(
            self, "metadata",
            validate_portable_mapping(self.metadata, where="artifact metadata"),
        )

    def to_manifest(self) -> dict[str, object]:
        """Return JSON-safe table metadata without serializing row-level results."""
        manifest = {
            "name": self.name,
            "estimand": self.estimand,
            "n_rows": int(len(self.table)),
            "columns": list(self.table.columns),
            "dtypes": {
                column: str(dtype)
                for column, dtype in self.table.dtypes.items()
            },
            "metadata": dict(self.metadata),
        }
        if self.table_contract is not None:
            manifest["table_schema"] = self.table_contract.to_manifest()
        json.dumps(manifest, sort_keys=True, allow_nan=False)
        return manifest


class AnalysisComponent(ABC):
    """Extension point for one reusable analysis over an AnalysisDataset."""

    name: str
    table_contract: TableContract | None = None

    @abstractmethod
    def run(self, dataset: AnalysisDataset) -> AnalysisArtifact:
        """Run the component without mutating the aligned input dataset."""
