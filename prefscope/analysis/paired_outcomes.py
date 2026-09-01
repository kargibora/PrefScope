"""Paired response-set outcome shifts with explicit independent-unit inference."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from prefscope.analysis.grouping import factorize_group_ids, validate_group_ids
from prefscope.analysis.outcomes import NormalizedOutcomes
from prefscope.analysis.stats import (
    benjamini_hochberg,
    bounded_mean_difference_hoeffding,
    bounded_mean_hoeffding,
)
from prefscope.core.features import validate_feature_ids


_BOUNDED_KINDS = {"binary", "probability", "preference"}


def _group_complete_pairs(a, b, group_ids):
    complete = np.isfinite(a) & np.isfinite(b)
    a = np.asarray(a[complete], dtype=float)
    b = np.asarray(b[complete], dtype=float)
    if group_ids is None:
        return a, b, int(complete.sum()), int(complete.sum()), True
    groups = validate_group_ids(group_ids, len(complete))[complete]
    codes, n_groups = factorize_group_ids(groups)
    counts = np.bincount(codes, minlength=n_groups).astype(float)
    kept = counts > 0
    group_a = np.zeros(n_groups, dtype=float)
    group_b = np.zeros(n_groups, dtype=float)
    np.add.at(group_a, codes, a)
    np.add.at(group_b, codes, b)
    group_a[kept] /= counts[kept]
    group_b[kept] /= counts[kept]
    singleton_groups = bool(kept.any() and np.all(counts[kept] == 1))
    return (
        group_a[kept], group_b[kept], int(complete.sum()), int(kept.sum()),
        singleton_groups,
    )


def paired_outcome_shift(
    outcomes_a: NormalizedOutcomes,
    outcomes_b: NormalizedOutcomes,
    *,
    group_ids=None,
    confidence: float = 0.95,
    min_units: int = 10,
) -> pd.DataFrame:
    """Estimate B-minus-A outcome changes over aligned rows or equal-weight groups.

    Missingness is pairwise per attribute. Binary singleton-row inference uses the
    exact McNemar/binomial test. Other bounded outcomes, and all grouped bounded
    outcomes, use finite-sample Hoeffding inference on independent B-minus-A units.
    Continuous outcomes remain descriptive because no outcome range is declared.
    """
    if not isinstance(outcomes_a, NormalizedOutcomes) or not isinstance(
        outcomes_b, NormalizedOutcomes
    ):
        raise ValueError("outcomes_a and outcomes_b must be NormalizedOutcomes")
    if outcomes_a.kind != outcomes_b.kind or outcomes_a.names != outcomes_b.names:
        raise ValueError("paired outcomes must have identical kind and attribute names")
    if outcomes_a.raw_values.shape != outcomes_b.raw_values.shape:
        raise ValueError("paired outcome matrices must be exactly aligned")
    if outcomes_a.normalization != "none" or outcomes_b.normalization != "none":
        raise ValueError("paired outcome shifts require unnormalized outcome values")
    if not isinstance(min_units, int) or isinstance(min_units, bool) or min_units < 2:
        raise ValueError("min_units must be an integer >= 2")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    if group_ids is not None:
        validate_group_ids(group_ids, outcomes_a.n_rows)

    rows = []
    for column, outcome_name in enumerate(outcomes_a.names):
        raw_a = outcomes_a.raw_values[:, column]
        raw_b = outcomes_b.raw_values[:, column]
        observed_a = np.isfinite(raw_a)
        observed_b = np.isfinite(raw_b)
        paired = observed_a & observed_b
        units_a, units_b, n_rows, n_units, singleton_rows = _group_complete_pairs(
            raw_a,
            raw_b,
            group_ids,
        )
        delta = units_b - units_a
        mean_a = float(units_a.mean()) if n_units else np.nan
        mean_b = float(units_b.mean()) if n_units else np.nan
        effect = float(delta.mean()) if n_units else np.nan
        supported = outcomes_a.kind in _BOUNDED_KINDS and n_units >= min_units
        ci_low = ci_high = p_value = np.nan
        test = "not_run_unbounded_continuous"
        discordant = np.nan
        a_only = b_only = np.nan
        ci_method = "not_run"
        support_reason = (
            "supported" if supported
            else "unbounded_outcome" if outcomes_a.kind not in _BOUNDED_KINDS
            else "fewer_than_min_independent_units"
        )
        if supported:
            bounded = bounded_mean_hoeffding(
                delta, lower=-1.0, upper=1.0, confidence=confidence)
            ci_low, ci_high = bounded["ci_low"], bounded["ci_high"]
            p_value = bounded["p_value"]
            test = "hoeffding_bounded_paired_mean"
            ci_method = "hoeffding_bounded_mean"
            if outcomes_a.kind == "binary" and singleton_rows:
                a_bool = units_a.astype(bool)
                b_bool = units_b.astype(bool)
                a_only = int(np.sum(a_bool & ~b_bool))
                b_only = int(np.sum(~a_bool & b_bool))
                discordant = a_only + b_only
                p_value = (
                    float(binomtest(b_only, int(discordant), 0.5).pvalue)
                    if discordant else 1.0
                )
                test = "exact_mcnemar_binomial"
        elif outcomes_a.kind in _BOUNDED_KINDS:
            test = "insufficient_independent_units"
        rows.append({
            "outcome": outcome_name,
            "outcome_kind": outcomes_a.kind,
            "contrast_type": "overall",
            "n_rows_total": outcomes_a.n_rows,
            "n_observed_a": int(observed_a.sum()),
            "n_observed_b": int(observed_b.sum()),
            "n_paired_rows": int(paired.sum()),
            "n_missing_a_only": int((~observed_a & observed_b).sum()),
            "n_missing_b_only": int((observed_a & ~observed_b).sum()),
            "n_missing_both": int((~observed_a & ~observed_b).sum()),
            "n_rows": n_rows,
            "n_units": n_units,
            "analysis_unit": "row" if group_ids is None else "group",
            "mean_a": mean_a,
            "mean_b": mean_b,
            "delta_b_minus_a": effect,
            "estimate": effect,
            "std_paired_unit_delta": (
                float(delta.std(ddof=1)) if n_units >= 2 else np.nan),
            "ci_low": ci_low,
            "ci_high": ci_high,
            "confidence": confidence,
            "ci_method": ci_method,
            "p_value": p_value,
            "inference_supported": bool(supported),
            "inference_test": test,
            "support_reason": support_reason,
            "n_a1_b0": a_only,
            "n_a0_b1": b_only,
            "discordant_pairs": discordant,
            "side_orientation": "b_minus_a",
            "contrast_orientation": "b_minus_a",
            "orientation": "delta_b_minus_a",
            "outcome_scale": "raw",
            "missingness_policy": "pairwise_complete_per_outcome",
            "tie_policy": (
                "retained_as_0.5_neutral"
                if outcomes_a.kind == "preference" else "not_applicable"),
            "estimand": (
                "mean B-minus-A outcome change across aligned rows"
                if group_ids is None else
                "equal-independent-group mean of within-group B-minus-A outcome change"
            ),
        })
    table = pd.DataFrame(rows)
    table["q_value"] = benjamini_hochberg(table["p_value"])
    table["multiplicity_family"] = (
        "all outcome-attribute shift tests in this paired outcome set")
    return table


def _strict_presence(values) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 2:
        raise ValueError("prompt_presence must be a 2-D matrix")
    if raw.dtype == bool:
        return raw
    if not np.issubdtype(raw.dtype, np.number):
        raise ValueError("prompt_presence must contain boolean or numeric 0/1 values")
    numeric = np.asarray(raw, dtype=float)
    if not np.isfinite(numeric).all() or not np.isin(numeric, [0.0, 1.0]).all():
        raise ValueError("prompt_presence must contain finite boolean or numeric 0/1 values")
    return numeric.astype(bool)


def _concept_units(a, b, presence, complete, group_ids):
    if group_ids is None:
        return a[complete], b[complete], presence[complete], int(complete.sum())
    groups = validate_group_ids(group_ids, len(complete))
    codes, n_groups = factorize_group_ids(groups)
    group_presence = np.zeros((n_groups, presence.shape[1]), dtype=bool)
    for group in range(n_groups):
        rows = presence[codes == group]
        if np.any(rows != rows[0]):
            raise ValueError(
                "prompt concept presence must be constant within each independent group")
        group_presence[group] = rows[0]
    complete_codes = codes[complete]
    counts = np.bincount(complete_codes, minlength=n_groups).astype(float)
    kept = counts > 0
    group_a = np.zeros(n_groups, dtype=float)
    group_b = np.zeros(n_groups, dtype=float)
    np.add.at(group_a, complete_codes, a[complete])
    np.add.at(group_b, complete_codes, b[complete])
    group_a[kept] /= counts[kept]
    group_b[kept] /= counts[kept]
    return group_a[kept], group_b[kept], group_presence[kept], int(complete.sum())


def paired_outcome_shift_by_concept(
    prompt_presence,
    outcomes_a: NormalizedOutcomes,
    outcomes_b: NormalizedOutcomes,
    *,
    feature_ids=None,
    basis=None,
    group_ids=None,
    confidence: float = 0.95,
    min_units_per_arm: int = 5,
) -> pd.DataFrame:
    """Test whether paired B-minus-A outcome changes differ by prompt concept.

    This is a heterogeneity/interaction estimand: the mean paired outcome shift among
    concept-present independent units minus the mean shift among concept-absent units.
    It is not two unrelated stratum-specific association tests.
    """
    presence = _strict_presence(prompt_presence)
    if not isinstance(outcomes_a, NormalizedOutcomes) or not isinstance(
        outcomes_b, NormalizedOutcomes
    ):
        raise ValueError("outcomes_a and outcomes_b must be NormalizedOutcomes")
    if outcomes_a.kind != outcomes_b.kind or outcomes_a.names != outcomes_b.names:
        raise ValueError("paired outcomes must have identical kind and attribute names")
    if outcomes_a.raw_values.shape != outcomes_b.raw_values.shape:
        raise ValueError("paired outcome matrices must be exactly aligned")
    if outcomes_a.normalization != "none" or outcomes_b.normalization != "none":
        raise ValueError("paired outcome shifts require unnormalized outcome values")
    if presence.shape[0] != outcomes_a.n_rows:
        raise ValueError("prompt presence and outcomes must have exactly aligned rows")
    if not isinstance(min_units_per_arm, int) or isinstance(min_units_per_arm, bool):
        raise ValueError("min_units_per_arm must be an integer")
    if min_units_per_arm < 2:
        raise ValueError("min_units_per_arm must be >= 2")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    ids = (
        tuple(range(presence.shape[1])) if feature_ids is None
        else validate_feature_ids(feature_ids, width=presence.shape[1])
    )
    bases = (
        tuple("unspecified" for _ in ids) if basis is None
        else tuple(str(value) for value in basis)
    )
    if len(bases) != len(ids) or any(not value for value in bases):
        raise ValueError("basis must have one non-empty entry per presence column")
    if group_ids is not None:
        validate_group_ids(group_ids, presence.shape[0])

    rows = []
    for outcome_column, outcome_name in enumerate(outcomes_a.names):
        raw_a = outcomes_a.raw_values[:, outcome_column]
        raw_b = outcomes_b.raw_values[:, outcome_column]
        observed_a = np.isfinite(raw_a)
        observed_b = np.isfinite(raw_b)
        complete = observed_a & observed_b
        units_a, units_b, unit_presence, n_rows = _concept_units(
            raw_a, raw_b, presence, complete, group_ids)
        shifts = units_b - units_a
        for feature_column, feature_id in enumerate(ids):
            inside = unit_presence[:, feature_column]
            outside = ~inside
            n_inside = int(inside.sum())
            n_outside = int(outside.sum())
            shift_inside = float(shifts[inside].mean()) if n_inside else np.nan
            shift_outside = float(shifts[outside].mean()) if n_outside else np.nan
            heterogeneity = shift_inside - shift_outside
            supported = (
                outcomes_a.kind in _BOUNDED_KINDS
                and n_inside >= min_units_per_arm
                and n_outside >= min_units_per_arm
            )
            ci_low = ci_high = p_value = np.nan
            inference_test = "not_run_unbounded_continuous"
            ci_method = "not_run"
            support_reason = (
                "supported" if supported
                else "unbounded_outcome" if outcomes_a.kind not in _BOUNDED_KINDS
                else "fewer_than_min_units_in_a_heterogeneity_stratum"
            )
            if supported:
                inference = bounded_mean_difference_hoeffding(
                    shifts[inside],
                    shifts[outside],
                    lower=-1.0,
                    upper=1.0,
                    confidence=confidence,
                )
                ci_low = inference["ci_low"]
                ci_high = inference["ci_high"]
                p_value = inference["p_value"]
                inference_test = "hoeffding_bounded_difference_in_paired_shifts"
                ci_method = "hoeffding_bounded_mean_difference"
            elif outcomes_a.kind in _BOUNDED_KINDS:
                inference_test = "insufficient_independent_units_per_arm"
            paired_present_rows = int(
                (complete & presence[:, feature_column]).sum())
            paired_absent_rows = int(
                (complete & ~presence[:, feature_column]).sum())
            rows.append({
                "outcome": outcome_name,
                "outcome_kind": outcomes_a.kind,
                "contrast_type": "prompt_concept_heterogeneity",
                "feature_id": feature_id,
                "presence_basis": bases[feature_column],
                "n_rows_total": outcomes_a.n_rows,
                "n_observed_a": int(observed_a.sum()),
                "n_observed_b": int(observed_b.sum()),
                "n_paired_rows": int(complete.sum()),
                "n_missing_a_only": int((~observed_a & observed_b).sum()),
                "n_missing_b_only": int((observed_a & ~observed_b).sum()),
                "n_missing_both": int((~observed_a & ~observed_b).sum()),
                "n_rows": n_rows,
                "n_units": len(shifts),
                "n_present_units": n_inside,
                "n_absent_units": n_outside,
                "n_paired_rows_present": paired_present_rows,
                "n_paired_rows_absent": paired_absent_rows,
                "mean_a_present": (
                    float(units_a[inside].mean()) if n_inside else np.nan),
                "mean_b_present": (
                    float(units_b[inside].mean()) if n_inside else np.nan),
                "shift_present_b_minus_a": shift_inside,
                "mean_a_absent": (
                    float(units_a[outside].mean()) if n_outside else np.nan),
                "mean_b_absent": (
                    float(units_b[outside].mean()) if n_outside else np.nan),
                "shift_absent_b_minus_a": shift_outside,
                "heterogeneity_present_minus_absent": heterogeneity,
                "estimate": heterogeneity,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "confidence": confidence,
                "ci_method": ci_method,
                "p_value": p_value,
                "inference_supported": bool(supported),
                "inference_test": inference_test,
                "support_reason": support_reason,
                "analysis_unit": "row" if group_ids is None else "group",
                "side_orientation": "b_minus_a",
                "contrast_orientation": "concept_present_minus_absent",
                "orientation": "delta_b_minus_a",
                "outcome_scale": "raw",
                "missingness_policy": "pairwise_complete_per_outcome",
                "tie_policy": (
                    "retained_as_0.5_neutral"
                    if outcomes_a.kind == "preference" else "not_applicable"),
                "estimand": (
                    "difference in paired B-minus-A outcome shift between "
                    "prompt-concept-present and prompt-concept-absent independent units"
                ),
            })
    table = pd.DataFrame(rows)
    table["q_value"] = benjamini_hochberg(table["p_value"])
    table["multiplicity_family"] = (
        "all prompt-feature × outcome-attribute heterogeneity tests in this "
        "paired outcome set")
    return table


__all__ = ["paired_outcome_shift", "paired_outcome_shift_by_concept"]
