"""Plan execution and typed results for task-centered analyses."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from types import MappingProxyType
from typing import Mapping

import pandas as pd

from prefscope.api.analysis_components import OutcomeAssociations, PairedOutcomeShifts
from prefscope.api.analysis_contracts import (
    AnalysisArtifact,
    AnalysisComponent,
    AnalysisDataset,
)
from prefscope.core import registry
from prefscope.core.features import FeatureBatch, FeatureMatrix


@dataclass(frozen=True, eq=False)
class AnalysisPlan:
    """Ordered collection of reusable analysis components."""

    components: tuple[AnalysisComponent, ...] = field(
        default_factory=lambda: (OutcomeAssociations(),))

    def __post_init__(self) -> None:
        components = tuple(self.components)
        if not components:
            raise ValueError("analysis plan needs at least one component")
        if not all(isinstance(value, AnalysisComponent) for value in components):
            raise ValueError("analysis plan components must implement AnalysisComponent")
        names = [value.name for value in components]
        if len(set(names)) != len(names):
            raise ValueError("analysis component names must be unique within a plan")
        object.__setattr__(self, "components", components)

    @classmethod
    def from_names(cls, names, **component_options) -> "AnalysisPlan":
        resolved_names = tuple(str(name) for name in names)
        extra = set(component_options) - set(resolved_names)
        if extra:
            raise ValueError(
                f"component options were supplied for unselected components: "
                f"{sorted(extra)}")
        components = []
        for name in resolved_names:
            options = dict(component_options.get(name, {}))
            components.append(registry.make("analysis_component", name, **options))
        return cls(tuple(components))


@dataclass(frozen=True, eq=False)
class DatasetAnalysisResult:
    """Typed artifacts returned by :func:`analyze_dataset`."""

    dataset: AnalysisDataset
    artifacts: Mapping[str, AnalysisArtifact]

    def __post_init__(self) -> None:
        artifacts = dict(self.artifacts)
        if not artifacts or any(name != value.name for name, value in artifacts.items()):
            raise ValueError("result artifacts must be a non-empty name-aligned mapping")
        object.__setattr__(self, "artifacts", MappingProxyType(artifacts))

    def artifact(self, name: str) -> AnalysisArtifact:
        try:
            return self.artifacts[name]
        except KeyError:
            raise ValueError(f"unknown analysis artifact {name!r}") from None

    @property
    def outcome_associations(self) -> pd.DataFrame | None:
        artifact = self.artifacts.get("outcome_associations")
        return None if artifact is None else artifact.table

    def to_manifest(self) -> dict[str, object]:
        """Return a portable summary that can accompany separately saved tables."""
        digest = hashlib.sha256()
        digest.update(b"prefscope-analysis-rows-v1\0")
        for row_id in self.dataset.row_ids:
            encoded = row_id.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
        manifest = {
            "schema_version": 1,
            "n_rows": self.dataset.n_rows,
            "row_ids_sha256": digest.hexdigest(),
            "group_source": self.dataset.group_source,
            "artifacts": [
                artifact.to_manifest() for artifact in self.artifacts.values()
            ],
        }
        json.dumps(manifest, sort_keys=True, allow_nan=False)
        return manifest


def analyze_dataset(
    features=None,
    outcomes=None,
    *,
    paired_outcomes=None,
    group_ids=None,
    plan: AnalysisPlan | None = None,
) -> DatasetAnalysisResult:
    """Run a reusable analysis plan over already computed feature matrices.

    ``features`` may be one :class:`FeatureMatrix` or a named mapping. ``outcomes``
    is a named mapping of :class:`OutcomeSpec`. Custom analyses subclass
    :class:`AnalysisComponent` and can be passed alongside the built-ins.
    """
    if features is None:
        feature_mapping = {}
    elif isinstance(features, FeatureMatrix):
        feature_mapping = {"features": features}
    elif isinstance(features, FeatureBatch):
        feature_mapping = {
            features.roles[name]: features.matrix(name)
            for name in features.arrays
        }
        if len(feature_mapping) != len(features.arrays):
            raise ValueError("FeatureBatch roles must be unique for direct analysis")
    else:
        feature_mapping = dict(features)
    outcome_mapping = {} if outcomes is None else dict(outcomes)
    paired_mapping = {} if paired_outcomes is None else dict(paired_outcomes)
    dataset = AnalysisDataset(
        features=feature_mapping,
        outcomes=outcome_mapping,
        group_ids=group_ids,
        paired_outcomes=paired_mapping,
    )
    if plan is None:
        components = []
        if dataset.outcomes:
            components.append(OutcomeAssociations())
        if dataset.paired_outcomes:
            components.append(PairedOutcomeShifts())
        resolved_plan = AnalysisPlan(tuple(components))
    else:
        resolved_plan = plan
    artifacts = {}
    for component in resolved_plan.components:
        artifact = component.run(dataset)
        if artifact.name in artifacts:
            raise ValueError(f"analysis component produced duplicate {artifact.name!r}")
        artifacts[artifact.name] = artifact
    return DatasetAnalysisResult(dataset=dataset, artifacts=artifacts)
