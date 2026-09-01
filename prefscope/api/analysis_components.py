"""Built-in components for the task-centered analysis API."""
from __future__ import annotations

import numpy as np
import pandas as pd

from prefscope.analysis.outcomes import associate_outcomes
from prefscope.analysis.paired import paired_concept_shift
from prefscope.analysis.paired_outcomes import (
    paired_outcome_shift,
    paired_outcome_shift_by_concept,
)
from prefscope.api.analysis_contracts import (
    AnalysisArtifact,
    AnalysisComponent,
    AnalysisDataset,
)
from prefscope.api.analysis_schemas import (
    FEATURE_ARTIFACT_DIAGNOSTICS,
    OUTCOME_ASSOCIATIONS,
    PAIRED_CONCEPT_SHIFT,
    PAIRED_OUTCOME_SHIFTS,
    PREFERENCE_LENGTH_CONFOUNDS,
    PROMPT_CONDITIONED_OUTCOME_SHIFTS,
)
from prefscope.core import registry
from prefscope.pipeline.confounds import screen_length_confound


@registry.register("analysis_component", "outcome-associations")
class OutcomeAssociations(AnalysisComponent):
    """Associate every selected feature set with every declared outcome."""

    name = "outcome_associations"
    table_contract = OUTCOME_ASSOCIATIONS

    def __init__(self, *, feature_sets=None, min_units: int = 3) -> None:
        if (
            not isinstance(min_units, int)
            or isinstance(min_units, bool)
            or min_units < 3
        ):
            raise ValueError("min_units must be an integer >= 3")
        if feature_sets is not None:
            feature_sets = tuple(feature_sets)
            if (
                not feature_sets
                or len(set(feature_sets)) != len(feature_sets)
                or any(not isinstance(value, str) or not value for value in feature_sets)
            ):
                raise ValueError("feature_sets must contain unique non-empty names")
        self.feature_sets = feature_sets
        self.min_units = min_units

    def run(self, dataset: AnalysisDataset) -> AnalysisArtifact:
        if not dataset.outcomes:
            raise ValueError("outcome-associations needs at least one declared outcome")
        selected = (
            tuple(dataset.features) if self.feature_sets is None else self.feature_sets)
        unknown = set(selected) - set(dataset.features)
        if unknown:
            raise ValueError(f"unknown feature sets: {sorted(unknown)}")
        tables = []
        normalizations = {}
        for feature_name in selected:
            features = dataset.features[feature_name]
            for outcome_name in dataset.outcomes:
                normalized = dataset.normalized_outcome(outcome_name)
                result = associate_outcomes(
                    features.values,
                    normalized,
                    feature_ids=features.feature_ids,
                    group_ids=dataset.group_ids,
                    min_units=self.min_units,
                )
                table = result.table.copy()
                table.insert(0, "feature_set", feature_name)
                table.insert(1, "outcome_set", outcome_name)
                table["feature_role"] = features.role
                table["multiplicity_family"] = f"{feature_name}:{outcome_name}"
                tables.append(table)
                normalizations[f"{feature_name}:{outcome_name}"] = {
                    "kind": normalized.kind,
                    "names": list(normalized.names),
                    "normalization": normalized.normalization,
                    "center": normalized.center.tolist(),
                    "scale": normalized.scale.tolist(),
                }
        table = (
            pd.concat(tables, ignore_index=True)
            if tables else self.table_contract.empty_frame()
        )
        return AnalysisArtifact(
            name=self.name,
            table_contract=self.table_contract,
            table=table,
            estimand=(
                "descriptive feature-outcome association at the declared row or "
                "equal-weight independent-group unit; not a causal effect"
            ),
            metadata={
                "multiplicity": "BH within each feature_set:outcome_set family",
                "group_source": dataset.group_source,
                "inference_note": (
                    "p/q values test a range-midpoint Fisher table when supported; "
                    "they are not p-values for the descriptive Pearson correlation "
                    "or OLS slope"
                ),
                "normalizations": normalizations,
            },
        )


@registry.register("analysis_component", "feature-artifact-diagnostics")
class FeatureArtifactDiagnostics(AnalysisComponent):
    """Summarize numerical activity and provenance of aligned feature artifacts."""

    name = "feature_artifact_diagnostics"
    table_contract = FEATURE_ARTIFACT_DIAGNOSTICS

    def __init__(
        self,
        *,
        feature_sets: tuple[str, ...] | None = None,
        zero_tolerance: float = 0.0,
    ) -> None:
        if feature_sets is not None:
            feature_sets = tuple(feature_sets)
            if (
                not feature_sets
                or len(set(feature_sets)) != len(feature_sets)
                or any(not isinstance(value, str) or not value for value in feature_sets)
            ):
                raise ValueError("feature_sets must contain unique non-empty names")
        if not np.isfinite(float(zero_tolerance)) or float(zero_tolerance) < 0:
            raise ValueError("zero_tolerance must be a finite non-negative number")
        self.feature_sets = feature_sets
        self.zero_tolerance = float(zero_tolerance)

    def run(self, dataset: AnalysisDataset) -> AnalysisArtifact:
        selected = (
            tuple(dataset.features) if self.feature_sets is None else self.feature_sets)
        if not selected:
            raise ValueError("feature artifact diagnostics need at least one feature set")
        unknown = set(selected) - set(dataset.features)
        if unknown:
            raise ValueError(f"unknown feature sets: {sorted(unknown)}")
        rows = []
        for name in selected:
            matrix = dataset.features[name]
            active = np.abs(matrix.values) > self.zero_tolerance
            l0 = active.sum(axis=1)
            feature_support = active.sum(axis=0)
            rows.append({
                "feature_set": name,
                "role": matrix.role,
                "orientation": matrix.orientation,
                "activation_polarity": matrix.activation_polarity,
                "code_semantics": matrix.code_semantics,
                "n_rows": matrix.n_rows,
                "n_features": matrix.n_features,
                "zero_tolerance": self.zero_tolerance,
                "nonzero_density": float(active.mean()),
                "mean_l0": float(l0.mean()),
                "min_l0": int(l0.min()),
                "max_l0": int(l0.max()),
                "zero_row_fraction": float((l0 == 0).mean()),
                "n_never_active_features": int((feature_support == 0).sum()),
                "n_always_active_features": int(
                    (feature_support == matrix.n_rows).sum()),
                "mean_abs_value": float(np.abs(matrix.values).mean()),
                "max_abs_value": float(np.abs(matrix.values).max()),
                "provenance_declared": bool(matrix.provenance),
            })
        return AnalysisArtifact(
            name=self.name,
            table_contract=self.table_contract,
            table=pd.DataFrame(rows),
            estimand=(
                "deterministic numerical feature-artifact health summary; nonzero "
                "activity is not a semantic-presence claim"
            ),
            metadata={
                "inference": "none",
                "semantic_presence_claim": "none",
                "zero_tolerance": self.zero_tolerance,
            },
        )


@registry.register("analysis_component", "preference-length-confounds")
class PreferenceLengthConfounds(AnalysisComponent):
    """Screen A-minus-B preference features for response-length entanglement."""

    name = "preference_length_confounds"
    table_contract = PREFERENCE_LENGTH_CONFOUNDS

    def __init__(
        self,
        *,
        feature_set: str,
        outcome: str,
        length_column: str,
        length_orientation: str,
        confound_threshold: float = 0.3,
        collapse_fraction: float = 0.5,
        permutations: int = 0,
        seed: int = 0,
    ) -> None:
        for name, value in {
            "feature_set": feature_set,
            "outcome": outcome,
            "length_column": length_column,
        }.items():
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if length_orientation != "a_minus_b":
            raise ValueError("length_orientation must explicitly be 'a_minus_b'")
        if not 0 <= float(confound_threshold) <= 1:
            raise ValueError("confound_threshold must be in [0, 1]")
        if not 0 <= float(collapse_fraction) <= 1:
            raise ValueError("collapse_fraction must be in [0, 1]")
        if (
            not isinstance(permutations, int)
            or isinstance(permutations, bool)
            or permutations < 0
        ):
            raise ValueError("permutations must be a non-negative integer")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("seed must be an integer")
        self.feature_set = feature_set
        self.outcome = outcome
        self.length_column = length_column
        self.length_orientation = length_orientation
        self.confound_threshold = float(confound_threshold)
        self.collapse_fraction = float(collapse_fraction)
        self.permutations = permutations
        self.seed = seed

    def run(self, dataset: AnalysisDataset) -> AnalysisArtifact:
        if self.feature_set not in dataset.features:
            raise ValueError(f"unknown feature set {self.feature_set!r}")
        if self.outcome not in dataset.outcomes:
            raise ValueError(f"unknown preference outcome {self.outcome!r}")
        features = dataset.features[self.feature_set]
        if features.orientation != "a_minus_b":
            raise ValueError(
                "preference length confounds require feature orientation='a_minus_b'")
        if self.length_column not in features.metadata:
            raise ValueError(
                f"feature metadata is missing length column {self.length_column!r}")
        outcome = dataset.normalized_outcome(self.outcome)
        if outcome.kind not in {"binary", "probability", "preference"}:
            raise ValueError(
                "preference length confounds require binary/probability/preference "
                "P(A preferred) values")
        if outcome.n_attributes != 1:
            raise ValueError("preference outcome must have exactly one attribute")
        labels = outcome.raw_values[:, 0]
        lengths = pd.to_numeric(
            pd.Series(features.metadata[self.length_column]), errors="raise"
        ).to_numpy(dtype=float)
        table, summary = screen_length_confound(
            features.values,
            labels,
            lengths,
            confound_threshold=self.confound_threshold,
            collapse_fraction=self.collapse_fraction,
            permutations=self.permutations,
            seed=self.seed,
            group_ids=dataset.group_ids,
        )
        feature_lookup = dict(enumerate(features.feature_ids))
        table["feature_id"] = table["feature_id"].map(feature_lookup)
        table.insert(0, "feature_set", self.feature_set)
        table.insert(1, "outcome_set", self.outcome)
        table["feature_orientation"] = "a_minus_b"
        table["length_orientation"] = self.length_orientation
        table["outcome_orientation"] = "p_a_preferred"
        table["tie_policy"] = "retained_as_0.5_neutral"
        table["multiplicity_family"] = f"{self.feature_set}:{self.outcome}:preference"
        return AnalysisArtifact(
            name=self.name,
            table_contract=self.table_contract,
            table=table,
            estimand=(
                "descriptive preference/length entanglement screen on A-minus-B "
                "feature and length coordinates; not a bias or causal classifier"
            ),
            metadata={
                "group_source": dataset.group_source,
                "feature_orientation": "a_minus_b",
                "length_orientation": self.length_orientation,
                "outcome_orientation": "p_a_preferred",
                "screen_summary": summary,
                "causal_claim": "none; sensitivity screen only",
            },
        )


@registry.register("analysis_component", "paired-outcome-shifts")
class PairedOutcomeShifts(AnalysisComponent):
    """Compare paired outcome sets with B-minus-A orientation."""

    name = "paired_outcome_shifts"
    table_contract = PAIRED_OUTCOME_SHIFTS

    def __init__(
        self,
        *,
        outcome_sets: tuple[str, ...] | None = None,
        confidence: float = 0.95,
        min_units: int = 10,
    ) -> None:
        if outcome_sets is not None:
            outcome_sets = tuple(outcome_sets)
            if (
                not outcome_sets
                or len(set(outcome_sets)) != len(outcome_sets)
                or any(not isinstance(value, str) or not value for value in outcome_sets)
            ):
                raise ValueError("outcome_sets must contain unique non-empty names")
        if not 0 < float(confidence) < 1:
            raise ValueError("confidence must be in (0, 1)")
        if (
            not isinstance(min_units, int)
            or isinstance(min_units, bool)
            or min_units < 2
        ):
            raise ValueError("min_units must be an integer >= 2")
        self.outcome_sets = outcome_sets
        self.confidence = float(confidence)
        self.min_units = min_units

    def run(self, dataset: AnalysisDataset) -> AnalysisArtifact:
        if not dataset.paired_outcomes:
            raise ValueError("paired-outcome-shifts needs at least one paired outcome")
        selected = (
            tuple(dataset.paired_outcomes) if self.outcome_sets is None
            else tuple(self.outcome_sets)
        )
        unknown = set(selected) - set(dataset.paired_outcomes)
        if unknown:
            raise ValueError(f"unknown paired outcome sets: {sorted(unknown)}")
        tables = []
        for name in selected:
            spec = dataset.paired_outcomes[name]
            table = paired_outcome_shift(
                spec.normalized_a,
                spec.normalized_b,
                group_ids=dataset.group_ids,
                confidence=self.confidence,
                min_units=self.min_units,
            )
            table.insert(0, "outcome_set", name)
            table.insert(1, "side_a", spec.side_a)
            table.insert(2, "side_b", spec.side_b)
            table.insert(3, "outcome_interpretation", spec.interpretation)
            table["multiplicity_family"] = f"{name}:paired_outcome_shift"
            tables.append(table)
        return AnalysisArtifact(
            name=self.name,
            table_contract=self.table_contract,
            table=pd.concat(tables, ignore_index=True),
            estimand=(
                "B-minus-A paired outcome change; row-weighted for unique rows or "
                "equal-weight over declared independent groups"
            ),
            metadata={
                "orientation": "delta_b_minus_a",
                "group_source": dataset.group_source,
                "missingness": "pairwise complete per outcome attribute",
                "multiplicity": "BH across attributes within each paired outcome set",
                "outcome_scale": "raw",
                "causal_claim": "none; descriptive paired change",
                "side_labels": {
                    name: [dataset.paired_outcomes[name].side_a,
                           dataset.paired_outcomes[name].side_b]
                    for name in selected
                },
            },
        )


@registry.register("analysis_component", "prompt-conditioned-outcome-shifts")
class PromptConditionedOutcomeShifts(AnalysisComponent):
    """Estimate heterogeneity of paired outcome shifts by calibrated prompt concept."""

    name = "prompt_conditioned_outcome_shifts"
    table_contract = PROMPT_CONDITIONED_OUTCOME_SHIFTS

    def __init__(
        self,
        *,
        prompt_features: str,
        outcome_sets: tuple[str, ...] | None = None,
        confidence: float = 0.95,
        min_units_per_arm: int = 5,
    ) -> None:
        if not isinstance(prompt_features, str) or not prompt_features:
            raise ValueError("prompt_features must name one feature set")
        if not 0 < float(confidence) < 1:
            raise ValueError("confidence must be in (0, 1)")
        if (
            not isinstance(min_units_per_arm, int)
            or isinstance(min_units_per_arm, bool)
            or min_units_per_arm < 2
        ):
            raise ValueError("min_units_per_arm must be an integer >= 2")
        if outcome_sets is not None:
            outcome_sets = tuple(outcome_sets)
            if (
                not outcome_sets
                or len(set(outcome_sets)) != len(outcome_sets)
                or any(not isinstance(value, str) or not value for value in outcome_sets)
            ):
                raise ValueError("outcome_sets must contain unique non-empty names")
        self.prompt_features = prompt_features
        self.outcome_sets = outcome_sets
        self.confidence = float(confidence)
        self.min_units_per_arm = min_units_per_arm

    def run(self, dataset: AnalysisDataset) -> AnalysisArtifact:
        if self.prompt_features not in dataset.features:
            raise ValueError(f"unknown prompt feature set {self.prompt_features!r}")
        prompt = dataset.features[self.prompt_features]
        if prompt.role != "prompt":
            raise ValueError("prompt-conditioned outcome shifts require role='prompt'")
        if prompt.code_semantics != "semantic_presence":
            raise ValueError(
                "prompt-conditioned outcome shifts require calibrated semantic-presence "
                "features created with FeatureMatrix.from_presence")
        bases = tuple(prompt.provenance.get("presence_basis", ()))
        if len(bases) != prompt.n_features:
            raise ValueError("prompt features must carry explicit presence_basis")
        selected = (
            tuple(dataset.paired_outcomes) if self.outcome_sets is None
            else tuple(self.outcome_sets)
        )
        if not selected:
            raise ValueError(
                "prompt-conditioned outcome shifts need at least one paired outcome")
        unknown = set(selected) - set(dataset.paired_outcomes)
        if unknown:
            raise ValueError(f"unknown paired outcome sets: {sorted(unknown)}")
        tables = []
        for name in selected:
            spec = dataset.paired_outcomes[name]
            table = paired_outcome_shift_by_concept(
                prompt.values,
                spec.normalized_a,
                spec.normalized_b,
                feature_ids=prompt.feature_ids,
                basis=bases,
                group_ids=dataset.group_ids,
                confidence=self.confidence,
                min_units_per_arm=self.min_units_per_arm,
            )
            table.insert(0, "prompt_feature_set", self.prompt_features)
            table.insert(1, "outcome_set", name)
            table.insert(2, "side_a", spec.side_a)
            table.insert(3, "side_b", spec.side_b)
            table.insert(4, "outcome_interpretation", spec.interpretation)
            table["multiplicity_family"] = (
                f"{self.prompt_features}:{name}:paired_shift_heterogeneity")
            tables.append(table)
        return AnalysisArtifact(
            name=self.name,
            table_contract=self.table_contract,
            table=pd.concat(tables, ignore_index=True),
            estimand=(
                "difference in paired B-minus-A outcome change between calibrated "
                "prompt-concept-present and prompt-concept-absent independent units"
            ),
            metadata={
                "orientation": "delta_b_minus_a",
                "group_source": dataset.group_source,
                "presence_claim": "calibrated semantic presence",
                "multiplicity": (
                    "BH across all prompt-feature × outcome-attribute heterogeneity "
                    "tests within each paired outcome set"),
                "causal_claim": "none; descriptive paired-change heterogeneity",
                "outcome_scale": "raw",
                "side_labels": {
                    name: [dataset.paired_outcomes[name].side_a,
                           dataset.paired_outcomes[name].side_b]
                    for name in selected
                },
            },
        )


@registry.register("analysis_component", "paired-concept-shift")
class PairedConceptShift(AnalysisComponent):
    """Compare calibrated concept prevalence for two prompt-aligned response sets."""

    name = "paired_concept_shift"
    table_contract = PAIRED_CONCEPT_SHIFT

    def __init__(
        self,
        *,
        side_a: str,
        side_b: str,
        confidence: float = 0.95,
    ) -> None:
        if not side_a or not side_b or side_a == side_b:
            raise ValueError("side_a and side_b must be distinct non-empty feature sets")
        if not 0 < float(confidence) < 1:
            raise ValueError("confidence must be in (0, 1)")
        self.side_a = str(side_a)
        self.side_b = str(side_b)
        self.confidence = float(confidence)

    def run(self, dataset: AnalysisDataset) -> AnalysisArtifact:
        missing = {self.side_a, self.side_b} - set(dataset.features)
        if missing:
            raise ValueError(f"paired concept shift is missing feature sets: {sorted(missing)}")
        side_a = dataset.features[self.side_a]
        side_b = dataset.features[self.side_b]
        if side_a.feature_ids != side_b.feature_ids:
            raise ValueError("paired concept feature IDs must be exactly aligned")
        invalid_roles = {"prompt", "response_difference"}
        if side_a.role in invalid_roles or side_b.role in invalid_roles:
            raise ValueError(
                "paired concept shift requires absolute response feature matrices")
        contrast_orientations = {"a_minus_b", "b_minus_a", "delta_b_minus_a"}
        if (
            side_a.orientation in contrast_orientations
            or side_b.orientation in contrast_orientations
        ):
            raise ValueError(
                "paired concept shift cannot consume contrast-oriented feature matrices")
        if (
            side_a.code_semantics != "semantic_presence"
            or side_b.code_semantics != "semantic_presence"
        ):
            raise ValueError(
                "paired concept shift requires calibrated semantic-presence matrices; "
                "convert concept_presence(...) with FeatureMatrix.from_presence"
            )
        basis_a = tuple(side_a.provenance.get("presence_basis", ()))
        basis_b = tuple(side_b.provenance.get("presence_basis", ()))
        if basis_a != basis_b or len(basis_a) != side_a.n_features:
            raise ValueError("paired concept matrices must share explicit presence_basis")
        table = paired_concept_shift(
            side_a.values,
            side_b.values,
            feature_ids=side_a.feature_ids,
            basis=basis_a,
            group_ids=dataset.group_ids,
            confidence=self.confidence,
        )
        table.insert(0, "side_a", self.side_a)
        table.insert(1, "side_b", self.side_b)
        table["orientation"] = "delta_b_minus_a"
        table["multiplicity_family"] = f"{self.side_a}:{self.side_b}:paired_shift"
        return AnalysisArtifact(
            name=self.name,
            table_contract=self.table_contract,
            table=table,
            estimand=(
                "B-minus-A calibrated concept-prevalence shift over aligned responses; "
                "row-weighted for unique rows or equal-weight over repeated groups"
            ),
            metadata={
                "side_a": self.side_a,
                "side_b": self.side_b,
                "orientation": "delta_b_minus_a",
                "group_source": dataset.group_source,
                "presence_claim": "calibrated semantic presence",
                "multiplicity": "BH across features in this paired comparison",
            },
        )
