"""Versioned table contracts for built-in task-centered analyses."""
from __future__ import annotations

from prefscope.core.table_schema import TableContract


def _contract(
    name,
    columns,
    *,
    key,
    orientation,
    integer=(),
    nullable_integer=(),
    float_=(),
    boolean=(),
    units=None,
):
    kinds = {column: "string" for column in columns}
    for kind, selected in (
        ("integer", integer),
        ("nullable_integer", nullable_integer),
        ("float", float_),
        ("boolean", boolean),
    ):
        for column in selected:
            kinds[column] = kind
    return TableContract(
        schema_name=name,
        schema_version=1,
        required_columns=tuple(columns),
        dtypes=kinds,
        unique_key=tuple(key),
        orientation=orientation,
        units=units or {},
    )


OUTCOME_ASSOCIATIONS = _contract(
    "outcome_associations",
    (
        "feature_set", "outcome_set", "outcome", "outcome_kind", "feature_id",
        "n_rows", "n_units", "analysis_unit", "feature_mean", "outcome_mean",
        "correlation", "slope", "p_value", "q_value", "feature_low_units",
        "feature_high_units", "outcome_low_units", "outcome_high_units",
        "association_outcome_center", "association_outcome_scale",
        "inference_supported", "missingness_policy", "tie_policy", "estimand",
        "inference_test", "feature_role", "multiplicity_family",
    ),
    key=("feature_set", "outcome_set", "outcome", "feature_id"),
    orientation="feature_activation_to_declared_outcome",
    integer=(
        "feature_id", "n_rows", "n_units", "feature_low_units",
        "feature_high_units", "outcome_low_units", "outcome_high_units",
    ),
    float_=(
        "feature_mean", "outcome_mean", "correlation", "slope", "p_value",
        "q_value", "association_outcome_center", "association_outcome_scale",
    ),
    boolean=("inference_supported",),
    units={
        "correlation": "unitless", "p_value": "unitless", "q_value": "unitless",
        "slope": "normalized outcome units per feature-code unit",
    },
)

FEATURE_ARTIFACT_DIAGNOSTICS = _contract(
    "feature_artifact_diagnostics",
    (
        "feature_set", "role", "orientation", "activation_polarity",
        "code_semantics", "n_rows", "n_features", "zero_tolerance",
        "nonzero_density", "mean_l0", "min_l0", "max_l0", "zero_row_fraction",
        "n_never_active_features", "n_always_active_features", "mean_abs_value",
        "max_abs_value", "provenance_declared",
    ),
    key=("feature_set",),
    orientation="per_feature_set_as_declared",
    integer=(
        "n_rows", "n_features", "min_l0", "max_l0", "n_never_active_features",
        "n_always_active_features",
    ),
    float_=(
        "zero_tolerance", "nonzero_density", "mean_l0", "zero_row_fraction",
        "mean_abs_value", "max_abs_value",
    ),
    boolean=("provenance_declared",),
    units={"nonzero_density": "proportion", "zero_row_fraction": "proportion"},
)

PREFERENCE_LENGTH_CONFOUNDS = _contract(
    "preference_length_confounds",
    (
        "feature_set", "outcome_set", "feature_id", "win_assoc", "correlation",
        "n_fire", "n_groups", "n_independent_groups", "estimand",
        "correlation_test", "significant", "corr_confound_len",
        "correlation_resid_len", "n_confound_groups", "confound_estimand",
        "confound_entangled", "feature_orientation", "length_orientation",
        "outcome_orientation", "tie_policy", "multiplicity_family",
    ),
    key=("feature_set", "outcome_set", "feature_id"),
    orientation="a_minus_b_features_and_length__p_a_preferred",
    integer=(
        "feature_id", "n_fire", "n_groups", "n_independent_groups",
        "n_confound_groups",
    ),
    float_=("win_assoc", "correlation", "corr_confound_len", "correlation_resid_len"),
    boolean=("significant", "confound_entangled"),
    units={
        "win_assoc": "probability-point difference", "correlation": "unitless",
        "corr_confound_len": "unitless", "correlation_resid_len": "unitless",
    },
)

PAIRED_OUTCOME_SHIFTS = _contract(
    "paired_outcome_shifts",
    (
        "outcome_set", "side_a", "side_b", "outcome_interpretation", "outcome",
        "outcome_kind", "contrast_type", "n_rows_total", "n_observed_a",
        "n_observed_b", "n_paired_rows", "n_missing_a_only", "n_missing_b_only",
        "n_missing_both", "n_rows", "n_units", "analysis_unit", "mean_a",
        "mean_b", "delta_b_minus_a", "estimate", "std_paired_unit_delta",
        "ci_low", "ci_high", "confidence", "ci_method", "p_value",
        "inference_supported", "inference_test", "support_reason", "n_a1_b0",
        "n_a0_b1", "discordant_pairs", "side_orientation", "contrast_orientation",
        "orientation", "outcome_scale", "missingness_policy", "tie_policy",
        "estimand", "q_value", "multiplicity_family",
    ),
    key=("outcome_set", "outcome"),
    orientation="b_minus_a",
    integer=(
        "n_rows_total", "n_observed_a", "n_observed_b", "n_paired_rows",
        "n_missing_a_only", "n_missing_b_only", "n_missing_both", "n_rows", "n_units",
    ),
    nullable_integer=("n_a1_b0", "n_a0_b1", "discordant_pairs"),
    float_=(
        "mean_a", "mean_b", "delta_b_minus_a", "estimate",
        "std_paired_unit_delta", "ci_low", "ci_high", "confidence", "p_value",
        "q_value",
    ),
    boolean=("inference_supported",),
    units={
        "delta_b_minus_a": "raw declared outcome units", "estimate": "raw declared outcome units",
        "ci_low": "raw declared outcome units", "ci_high": "raw declared outcome units",
        "p_value": "unitless", "q_value": "unitless",
    },
)

PROMPT_CONDITIONED_OUTCOME_SHIFTS = _contract(
    "prompt_conditioned_outcome_shifts",
    (
        "prompt_feature_set", "outcome_set", "side_a", "side_b",
        "outcome_interpretation", "outcome", "outcome_kind", "contrast_type",
        "feature_id", "presence_basis", "n_rows_total", "n_observed_a",
        "n_observed_b", "n_paired_rows", "n_missing_a_only", "n_missing_b_only",
        "n_missing_both", "n_rows", "n_units", "n_present_units", "n_absent_units",
        "n_paired_rows_present", "n_paired_rows_absent", "mean_a_present",
        "mean_b_present", "shift_present_b_minus_a", "mean_a_absent",
        "mean_b_absent", "shift_absent_b_minus_a",
        "heterogeneity_present_minus_absent", "estimate", "ci_low", "ci_high",
        "confidence", "ci_method", "p_value", "inference_supported",
        "inference_test", "support_reason", "analysis_unit", "side_orientation",
        "contrast_orientation", "orientation", "outcome_scale", "missingness_policy",
        "tie_policy", "estimand", "q_value", "multiplicity_family",
    ),
    key=("prompt_feature_set", "outcome_set", "outcome", "feature_id"),
    orientation="present_minus_absent_of_b_minus_a",
    integer=(
        "feature_id", "n_rows_total", "n_observed_a", "n_observed_b",
        "n_paired_rows", "n_missing_a_only", "n_missing_b_only", "n_missing_both",
        "n_rows", "n_units", "n_present_units", "n_absent_units",
        "n_paired_rows_present", "n_paired_rows_absent",
    ),
    float_=(
        "mean_a_present", "mean_b_present", "shift_present_b_minus_a",
        "mean_a_absent", "mean_b_absent", "shift_absent_b_minus_a",
        "heterogeneity_present_minus_absent", "estimate", "ci_low", "ci_high",
        "confidence", "p_value", "q_value",
    ),
    boolean=("inference_supported",),
    units={
        "heterogeneity_present_minus_absent": "raw declared outcome units",
        "estimate": "raw declared outcome units", "ci_low": "raw declared outcome units",
        "ci_high": "raw declared outcome units", "p_value": "unitless",
        "q_value": "unitless",
    },
)

PAIRED_CONCEPT_SHIFT = _contract(
    "paired_concept_shift",
    (
        "side_a", "side_b", "feature_id", "n_pairs", "n_groups", "prevalence_a",
        "prevalence_b", "delta_b_minus_a", "ci_low", "ci_high", "ci_method",
        "a_only", "b_only", "n_discordant", "n_nonzero_groups", "p_value",
        "q_value", "test", "inference_test", "orientation", "estimand",
        "presence_basis", "multiplicity_family",
    ),
    key=("side_a", "side_b", "feature_id"),
    orientation="b_minus_a",
    integer=(
        "feature_id", "n_pairs", "n_groups", "a_only", "b_only", "n_discordant",
        "n_nonzero_groups",
    ),
    float_=(
        "prevalence_a", "prevalence_b", "delta_b_minus_a", "ci_low", "ci_high",
        "p_value", "q_value",
    ),
    units={
        "prevalence_a": "proportion", "prevalence_b": "proportion",
        "delta_b_minus_a": "proportion", "ci_low": "proportion",
        "ci_high": "proportion", "p_value": "unitless", "q_value": "unitless",
    },
)

BUILTIN_ANALYSIS_TABLE_CONTRACTS = {
    contract.schema_name: contract for contract in (
        OUTCOME_ASSOCIATIONS,
        FEATURE_ARTIFACT_DIAGNOSTICS,
        PREFERENCE_LENGTH_CONFOUNDS,
        PAIRED_OUTCOME_SHIFTS,
        PROMPT_CONDITIONED_OUTCOME_SHIFTS,
        PAIRED_CONCEPT_SHIFT,
    )
}

__all__ = [
    "BUILTIN_ANALYSIS_TABLE_CONTRACTS", "OUTCOME_ASSOCIATIONS",
    "FEATURE_ARTIFACT_DIAGNOSTICS", "PREFERENCE_LENGTH_CONFOUNDS",
    "PAIRED_OUTCOME_SHIFTS", "PROMPT_CONDITIONED_OUTCOME_SHIFTS",
    "PAIRED_CONCEPT_SHIFT",
]
