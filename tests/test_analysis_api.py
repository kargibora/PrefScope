from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from prefscope.analysis.presence import PresenceMatrix
from prefscope import (
    AnalysisArtifact,
    AnalysisComponent,
    AnalysisPlan,
    FeatureMatrix,
    FeatureArtifactDiagnostics,
    OutcomeAssociations,
    OutcomeSpec,
    PairedConceptShift,
    PairedOutcomeShifts,
    PairedOutcomeSpec,
    PromptConditionedOutcomeShifts,
    PreferenceLengthConfounds,
    analyze_dataset,
)


def _features(values, ids=("a", "b", "c", "d", "e", "f", "g", "h", "i", "j")):
    return FeatureMatrix(
        values=np.asarray(values, dtype=float)[:, None],
        row_ids=ids,
        role="response",
        orientation="none",
        provenance={"source": "test"},
    )


def test_analyze_dataset_returns_typed_outcome_artifact():
    values = np.array([0.0] * 5 + [1.0] * 5)
    result = analyze_dataset(
        {"response": _features(values)},
        {"correct": OutcomeSpec(values, row_ids=tuple("abcdefghij"), kind="binary")},
    )

    table = result.outcome_associations
    assert table is not None and len(table) == 1
    row = table.iloc[0]
    assert row["feature_set"] == "response"
    assert row["outcome_set"] == "correct"
    assert row["feature_role"] == "response"
    assert row["multiplicity_family"] == "response:correct"
    assert row["inference_test"] == "fisher_exact_range_midpoint_split"
    assert np.isclose(row["p_value"], 2.0 / 252.0)
    artifact = result.artifact("outcome_associations")
    assert "not a causal effect" in artifact.estimand
    assert artifact.metadata["multiplicity"].startswith("BH within")


def test_grouped_high_level_analysis_is_invariant_to_row_duplication():
    base = FeatureMatrix(
        np.array([[0.0], [1.0], [3.0]]),
        row_ids=("a", "b", "c"), role="response")
    base_result = analyze_dataset(
        base,
        {"reward": OutcomeSpec([0.0, 1.0, 2.0], row_ids=("a", "b", "c"), kind="continuous")},
        group_ids=["a", "b", "c"],
    ).outcome_associations.iloc[0]

    repeated = FeatureMatrix(
        np.array([[0.0]] * 20 + [[1.0], [3.0]]),
        row_ids=tuple(f"r{i}" for i in range(22)), role="response")
    repeated_result = analyze_dataset(
        repeated,
        {"reward": OutcomeSpec(
            [0.0] * 20 + [1.0, 2.0],
            row_ids=tuple(f"r{i}" for i in range(22)), kind="continuous")},
        group_ids=["a"] * 20 + ["b", "c"],
    ).outcome_associations.iloc[0]

    assert base_result["analysis_unit"] == "group"
    assert np.isclose(base_result["slope"], repeated_result["slope"])
    assert np.isclose(
        base_result["association_outcome_scale"],
        repeated_result["association_outcome_scale"],
    )


def test_analysis_rejects_positional_misalignment_and_missing_groups():
    first = _features(np.arange(10))
    second = _features(np.arange(10), ids=tuple(reversed(first.row_ids)))
    outcome = {"reward": OutcomeSpec(
        np.arange(10), row_ids=tuple("abcdefghij"), kind="continuous")}
    with pytest.raises(ValueError, match="row_ids are not exactly aligned"):
        analyze_dataset({"a": first, "b": second}, outcome)
    groups = np.array(["g"] * 10, dtype=object)
    groups[3] = None
    with pytest.raises(ValueError, match="must not contain missing"):
        analyze_dataset(first, outcome, group_ids=groups)


class MeanFeature(AnalysisComponent):
    name = "mean_feature"

    def run(self, dataset):
        rows = [
            {"feature_set": name, "mean": float(matrix.values.mean())}
            for name, matrix in dataset.features.items()
        ]
        return AnalysisArtifact(
            name=self.name,
            table=pd.DataFrame(rows),
            estimand="row-weighted descriptive mean feature activation",
        )


def test_custom_component_composes_with_builtin_plan():
    plan = AnalysisPlan((OutcomeAssociations(), MeanFeature()))
    result = analyze_dataset(
        _features(np.arange(10)),
        {"reward": OutcomeSpec(
        np.arange(10), row_ids=tuple("abcdefghij"), kind="continuous")},
        plan=plan,
    )
    assert set(result.artifacts) == {"outcome_associations", "mean_feature"}
    assert result.artifact("mean_feature").table.loc[0, "mean"] == 4.5


def test_plan_can_resolve_registered_components_by_name():
    plan = AnalysisPlan.from_names(
        ["outcome-associations"],
        **{"outcome-associations": {"min_units": 5}},
    )
    assert isinstance(plan.components[0], OutcomeAssociations)
    conditioned = AnalysisPlan.from_names(
        ["prompt-conditioned-outcome-shifts"],
        **{"prompt-conditioned-outcome-shifts": {"prompt_features": "prompt"}},
    )
    assert isinstance(conditioned.components[0], PromptConditionedOutcomeShifts)
    with pytest.raises(ValueError, match="unselected components"):
        AnalysisPlan.from_names(
            ["outcome-associations"],
            **{"paired-outcome-shifts": {}},
        )


def test_analysis_automatically_groups_identical_prompts_from_feature_metadata():
    matrix = FeatureMatrix(
        np.array([[0.0], [2.0], [1.0]]),
        row_ids=("a", "b", "c"),
        role="response",
        metadata={"prompt": ("same", "same", "other")},
    )
    result = analyze_dataset(
        matrix,
        {"reward": OutcomeSpec(
            [0.0, 1.0, 1.0], row_ids=("a", "b", "c"), kind="continuous")},
    )
    assert result.dataset.group_source == "normalized_prompt_hash"
    row = result.outcome_associations.iloc[0]
    assert row["analysis_unit"] == "group"
    assert row["n_units"] == 2


def _presence(values):
    width = np.asarray(values).shape[1]
    return PresenceMatrix(
        values=np.asarray(values, dtype=bool),
        feature_ids=np.arange(width),
        basis=np.array(["semantic_threshold"] * width, dtype=object),
        thresholds=np.ones(width),
        calibrated=np.ones(width, dtype=bool),
    )


def test_paired_concept_component_requires_and_reports_semantic_presence():
    row_ids = tuple(f"p{i}" for i in range(20))
    side_a = FeatureMatrix.from_presence(
        _presence(np.zeros((20, 1), dtype=bool)),
        row_ids=row_ids, role="response_a")
    side_b = FeatureMatrix.from_presence(
        _presence(np.ones((20, 1), dtype=bool)),
        row_ids=row_ids, role="response_b")
    plan = AnalysisPlan((PairedConceptShift(side_a="before", side_b="after"),))

    result = analyze_dataset(
        {"before": side_a, "after": side_b}, plan=plan)

    row = result.artifact("paired_concept_shift").table.iloc[0]
    assert row["delta_b_minus_a"] == 1.0
    assert row["orientation"] == "delta_b_minus_a"
    assert row["presence_basis"] == "semantic_threshold"
    assert result.artifact("paired_concept_shift").metadata["presence_claim"] == (
        "calibrated semantic presence")


def test_paired_concept_component_rejects_raw_presence_codes():
    rows = tuple(f"p{i}" for i in range(10))
    raw = FeatureMatrix(
        np.ones((10, 1)), row_ids=rows, role="response",
        code_semantics="presence")
    plan = AnalysisPlan((PairedConceptShift(side_a="a", side_b="b"),))
    with pytest.raises(ValueError, match="calibrated semantic-presence"):
        analyze_dataset({"a": raw, "b": raw}, plan=plan)


def test_paired_outcome_component_runs_by_default_and_is_typed():
    rows = tuple(f"p{i}" for i in range(10))
    features = FeatureMatrix(
        np.arange(10, dtype=float)[:, None], row_ids=rows, role="prompt")
    paired = PairedOutcomeSpec(
        np.zeros(10), np.ones(10), row_ids=rows, kind="probability")

    result = analyze_dataset(
        features, paired_outcomes={"quality": paired})

    artifact = result.artifact("paired_outcome_shifts")
    row = artifact.table.iloc[0]
    assert row["outcome_set"] == "quality"
    assert row["delta_b_minus_a"] == 1.0
    assert row["orientation"] == "delta_b_minus_a"
    assert row["multiplicity_family"] == "quality:paired_outcome_shift"
    assert isinstance(PairedOutcomeShifts(), AnalysisComponent)


def test_paired_outcome_component_rejects_misaligned_row_ids():
    rows = tuple(f"p{i}" for i in range(10))
    features = FeatureMatrix(
        np.arange(10, dtype=float)[:, None], row_ids=rows, role="prompt")
    paired = PairedOutcomeSpec(
        np.zeros(10), np.ones(10), row_ids=tuple(reversed(rows)),
        kind="probability")
    with pytest.raises(ValueError, match="row_ids are not exactly aligned"):
        analyze_dataset(features, paired_outcomes={"quality": paired})


def test_prompt_conditioned_outcome_component_uses_calibrated_prompt_presence():
    rows = tuple(f"p{i}" for i in range(20))
    prompt = FeatureMatrix.from_presence(
        _presence(np.array([[1]] * 10 + [[0]] * 10, dtype=bool)),
        row_ids=rows,
        role="prompt",
    )
    paired = PairedOutcomeSpec(
        np.zeros(20), np.array([1.0] * 10 + [0.0] * 10),
        row_ids=rows, kind="probability")
    plan = AnalysisPlan((
        PromptConditionedOutcomeShifts(prompt_features="prompt_concepts"),
    ))
    result = analyze_dataset(
        {"prompt_concepts": prompt},
        paired_outcomes={"quality": paired},
        plan=plan,
    )
    artifact = result.artifact("prompt_conditioned_outcome_shifts")
    row = artifact.table.iloc[0]
    assert row["heterogeneity_present_minus_absent"] == 1.0
    assert row["multiplicity_family"] == (
        "prompt_concepts:quality:paired_shift_heterogeneity")
    assert "difference in paired B-minus-A" in artifact.estimand


def test_outcome_only_paired_analysis_needs_no_dummy_feature():
    rows = tuple(f"p{i}" for i in range(10))
    paired = PairedOutcomeSpec(
        np.zeros(10), np.ones(10), row_ids=rows, kind="probability",
        side_a="base", side_b="candidate",
        interpretation="higher quality is better",
    )
    result = analyze_dataset(paired_outcomes={"quality": paired})
    row = result.artifact("paired_outcome_shifts").table.iloc[0]
    assert row["side_a"] == "base" and row["side_b"] == "candidate"
    assert row["outcome_scale"] == "raw"
    assert result.dataset.row_ids == rows


def test_paired_component_options_fail_early():
    with pytest.raises(ValueError, match="outcome_sets"):
        PairedOutcomeShifts(outcome_sets=())
    with pytest.raises(ValueError, match="min_units"):
        PairedOutcomeShifts(min_units=True)
    with pytest.raises(ValueError, match="confidence"):
        PairedOutcomeShifts(confidence=1.0)
    with pytest.raises(ValueError, match="distinct"):
        PairedOutcomeSpec(
            [0.0, 1.0], [1.0, 0.0], row_ids=("a", "b"),
            kind="probability", side_a="same", side_b="same")


def test_semantic_feature_conversion_rejects_positive_nonzero_fallback():
    presence = PresenceMatrix(
        values=np.ones((2, 1), dtype=bool),
        feature_ids=np.array([0]),
        basis=np.array(["positive_nonzero"], dtype=object),
        thresholds=np.array([0.0]),
        calibrated=np.array([False]),
    )
    with pytest.raises(ValueError, match="semantic_threshold"):
        FeatureMatrix.from_presence(
            presence, row_ids=("a", "b"), role="prompt")


def test_analysis_artifact_metadata_must_be_portable():
    with pytest.raises(ValueError, match="absolute local paths"):
        AnalysisArtifact(
            name="bad", table=pd.DataFrame({"x": [1]}), estimand="test",
            metadata={"artifact_path": "/tmp/result.csv"},
        )


def test_preference_length_confound_component_requires_explicit_orientation():
    rows = tuple(f"r{i}" for i in range(40))
    signs = np.array([-1.0, 1.0] * 20)
    features = FeatureMatrix(
        signs[:, None],
        row_ids=rows,
        role="response_difference",
        orientation="a_minus_b",
        feature_ids=(7,),
        metadata={"length_a_minus_b": tuple(signs)},
        code_semantics="axis",
    )
    preference = OutcomeSpec(
        (signs > 0).astype(float), row_ids=rows, kind="preference")
    plan = AnalysisPlan((PreferenceLengthConfounds(
        feature_set="difference",
        outcome="preference",
        length_column="length_a_minus_b",
        length_orientation="a_minus_b",
    ),))
    result = analyze_dataset(
        {"difference": features}, {"preference": preference}, plan=plan)
    artifact = result.artifact("preference_length_confounds")
    assert artifact.table.loc[0, "feature_id"] == 7
    assert artifact.table.loc[0, "length_orientation"] == "a_minus_b"
    assert artifact.table.loc[0, "outcome_orientation"] == "p_a_preferred"
    assert artifact.table.loc[0, "confound_entangled"]
    assert artifact.metadata["causal_claim"] == "none; sensitivity screen only"

    wrong = FeatureMatrix(
        signs[:, None], row_ids=rows, role="response_difference",
        orientation="b_minus_a",
        metadata={"length_a_minus_b": tuple(signs)},
    )
    with pytest.raises(ValueError, match="orientation='a_minus_b'"):
        analyze_dataset(
            {"difference": wrong}, {"preference": preference}, plan=plan)


def test_feature_artifact_diagnostics_do_not_claim_semantic_presence():
    matrix = FeatureMatrix(
        np.array([[0.0, 1.0], [0.0, 0.0]]),
        row_ids=("a", "b"), role="response", orientation="none",
        code_semantics="axis")
    plan = AnalysisPlan((FeatureArtifactDiagnostics(),))
    artifact = analyze_dataset(matrix, plan=plan).artifact(
        "feature_artifact_diagnostics")
    row = artifact.table.iloc[0]
    assert row["mean_l0"] == 0.5
    assert row["n_never_active_features"] == 1
    assert row["zero_row_fraction"] == 0.5
    assert artifact.metadata["semantic_presence_claim"] == "none"
    assert "not a semantic-presence claim" in artifact.estimand


def test_analysis_result_manifest_is_json_safe_and_hashes_row_ids():
    rows = tuple("abcdefghij")
    result = analyze_dataset(
        _features(np.arange(10)),
        {"reward": OutcomeSpec(
            np.arange(10), row_ids=rows, kind="continuous")},
    )
    manifest = result.to_manifest()
    assert manifest["schema_version"] == 1
    assert manifest["n_rows"] == 10
    assert len(manifest["row_ids_sha256"]) == 64
    assert manifest["artifacts"][0]["name"] == "outcome_associations"
    json.dumps(manifest, allow_nan=False)


def test_artifact_names_and_columns_are_portable():
    with pytest.raises(ValueError, match="lower_snake_case"):
        AnalysisArtifact(
            name="../bad", table=pd.DataFrame({"x": [1]}), estimand="test")
    with pytest.raises(ValueError, match="columns"):
        AnalysisArtifact(
            name="bad_columns", table=pd.DataFrame([[1]], columns=[1]),
            estimand="test")


def test_analysis_rejects_conflicting_feature_metadata_group_partitions():
    rows = ("a", "b", "c", "d")
    first = FeatureMatrix(
        np.ones((4, 1)), row_ids=rows, role="response",
        metadata={"group_id": ("x", "x", "y", "y")})
    conflicting = FeatureMatrix(
        np.ones((4, 1)), row_ids=rows, role="response",
        metadata={"group_id": ("x", "y", "x", "y")})
    outcome = OutcomeSpec(
        [0.0, 1.0, 0.0, 1.0], row_ids=rows, kind="binary")
    with pytest.raises(ValueError, match="conflicting independent-group"):
        analyze_dataset({"first": first, "second": conflicting}, {"y": outcome})


def test_analysis_accepts_different_labels_for_the_same_group_partition():
    rows = ("a", "b", "c", "d")
    first = FeatureMatrix(
        np.ones((4, 1)), row_ids=rows, role="response",
        metadata={"group_id": (1, 1, 2, 2)})
    same_partition = FeatureMatrix(
        np.ones((4, 1)), row_ids=rows, role="response",
        metadata={"group_id": ("left", "left", "right", "right")})
    plan = AnalysisPlan((FeatureArtifactDiagnostics(),))
    result = analyze_dataset(
        {"first": first, "second": same_partition}, plan=plan)
    assert result.dataset.group_source == "canonical_group_id"


def test_paired_concept_component_rejects_prompt_or_contrast_roles():
    rows = tuple(f"p{i}" for i in range(10))
    prompt = FeatureMatrix.from_presence(
        _presence(np.ones((10, 1), dtype=bool)),
        row_ids=rows, role="prompt")
    plan = AnalysisPlan((PairedConceptShift(side_a="a", side_b="b"),))
    with pytest.raises(ValueError, match="absolute response"):
        analyze_dataset({"a": prompt, "b": prompt}, plan=plan)


def test_analysis_artifact_detaches_caller_owned_dataframe():
    source = pd.DataFrame({"value": [1.0]})
    artifact = AnalysisArtifact(name="snapshot", table=source, estimand="test")
    source.loc[0, "value"] = 9.0
    assert artifact.table.loc[0, "value"] == 1.0
