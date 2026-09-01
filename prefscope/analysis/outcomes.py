"""Typed outcome normalization and descriptive feature associations.

Outcomes describe labels or measurements attached to post-training examples.  The
associations in this module are observational summaries of a dataset.  They do not
identify causal effects or say that an outcome is objectively good or bad.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, pearsonr

from prefscope.analysis.grouping import factorize_group_ids, validate_group_ids
from prefscope.analysis.paired import bh_adjust
from prefscope.core.features import validate_feature_ids


OutcomeKind = Literal[
    "binary", "probability", "preference", "continuous", "multi_continuous",
]
Normalization = Literal["auto", "none", "zscore"]

OUTCOME_KINDS = (
    "binary", "probability", "preference", "continuous", "multi_continuous",
)
OUTCOME_NORMALIZATIONS = ("auto", "none", "zscore")


@dataclass(frozen=True, eq=False)
class NormalizedOutcomes:
    """Validated outcome matrix with explicit missingness and scaling provenance.

    All arrays have shape ``(n_rows, n_attributes)``. ``raw_values`` and ``values``
    retain missing entries as ``NaN``; ``observed`` is their aligned validity mask.
    ``center`` and ``scale`` describe ``values = (raw_values - center) / scale`` for
    z-scored attributes.  With ``normalization='none'``, center is zero and scale is one.
    Binary, probability, preference, and continuous outcomes have one attribute;
    ``multi_continuous`` can have several independently normalized attributes.
    """

    kind: OutcomeKind
    names: tuple[str, ...]
    raw_values: np.ndarray
    values: np.ndarray
    observed: np.ndarray
    normalization: Literal["none", "zscore"]
    center: np.ndarray
    scale: np.ndarray

    def __post_init__(self) -> None:
        if self.kind not in OUTCOME_KINDS:
            raise ValueError(f"unknown outcome kind {self.kind!r}")
        raw = np.asarray(self.raw_values)
        values = np.asarray(self.values)
        observed = np.asarray(self.observed)
        if raw.ndim != 2 or values.shape != raw.shape or observed.shape != raw.shape:
            raise ValueError(
                "raw_values, values, and observed must be aligned 2-D arrays")
        for name, array in {"raw_values": raw, "values": values}.items():
            if (
                not np.issubdtype(array.dtype, np.number)
                or np.issubdtype(array.dtype, np.complexfloating)
            ):
                raise ValueError(f"{name} must be a real numeric matrix")
        if observed.dtype != bool:
            raise ValueError("observed must be a boolean matrix")
        names = tuple(self.names)
        if len(names) != raw.shape[1]:
            raise ValueError("names must have one entry per outcome attribute")
        if (
            len(set(names)) != len(names)
            or any(not isinstance(name, str) or not name for name in names)
        ):
            raise ValueError("names must be unique non-empty strings")
        if self.kind != "multi_continuous" and raw.shape[1] != 1:
            raise ValueError(f"{self.kind} outcomes must contain exactly one column")
        if self.normalization not in {"none", "zscore"}:
            raise ValueError("normalization must be 'none' or 'zscore'")
        center = np.asarray(self.center)
        scale = np.asarray(self.scale)
        for name, array in {"center": center, "scale": scale}.items():
            if (
                array.ndim != 1
                or len(array) != raw.shape[1]
                or not np.issubdtype(array.dtype, np.number)
                or np.issubdtype(array.dtype, np.complexfloating)
            ):
                raise ValueError(
                    f"{name} must be a real vector with one entry per outcome attribute")
        center = np.asarray(center, dtype=float)
        scale = np.asarray(scale, dtype=float)
        if not np.isfinite(center).all():
            raise ValueError("center must contain only finite values")
        if not np.all(np.isfinite(scale) & (scale > 0)):
            raise ValueError("scale must contain one positive finite value per attribute")
        expected_observed = np.isfinite(raw)
        if not np.array_equal(observed, expected_observed):
            raise ValueError("observed must exactly mark finite raw_values")
        if np.isinf(values).any() or not np.isnan(values[~expected_observed]).all():
            raise ValueError("normalized values must preserve missing entries as NaN")
        raw_float = np.asarray(raw, dtype=float)
        expected_values = (raw_float - center) / scale
        if not np.allclose(
            values[expected_observed], expected_values[expected_observed],
            rtol=1e-12, atol=1e-12,
        ):
            raise ValueError("values must equal (raw_values - center) / scale")
        observed_raw = raw_float[expected_observed]
        if self.kind == "binary" and not np.isin(observed_raw, [0.0, 1.0]).all():
            raise ValueError("binary outcomes accept only 0 and 1")
        if self.kind in {"probability", "preference"} and (
            ((observed_raw < 0.0) | (observed_raw > 1.0)).any()
        ):
            raise ValueError(f"{self.kind} outcomes must lie in [0, 1]")
        if self.normalization == "none" and (
            not np.array_equal(center, np.zeros_like(center))
            or not np.array_equal(scale, np.ones_like(scale))
        ):
            raise ValueError("normalization='none' requires zero center and unit scale")
        detached = {
            "raw_values": np.array(raw_float, dtype=float, copy=True),
            "values": np.array(values, dtype=float, copy=True),
            "observed": np.array(observed, dtype=bool, copy=True),
            "center": np.array(center, dtype=float, copy=True),
            "scale": np.array(scale, dtype=float, copy=True),
        }
        object.__setattr__(self, "names", names)
        for name, array in detached.items():
            immutable = np.frombuffer(
                array.tobytes(order="C"), dtype=array.dtype
            ).reshape(array.shape)
            object.__setattr__(self, name, immutable)

    @property
    def n_rows(self) -> int:
        return int(self.values.shape[0])

    @property
    def n_attributes(self) -> int:
        return int(self.values.shape[1])


@dataclass(frozen=True, eq=False)
class OutcomeAssociationResult:
    """Long-form descriptive associations and their analysis-unit contract."""

    table: pd.DataFrame
    outcomes: NormalizedOutcomes
    grouped: bool
    method: str = "pearson_ols_descriptive_and_fisher_exact_range_midpoint_split"
    estimand: str = "descriptive association; not a causal effect"

    def __post_init__(self) -> None:
        required = {
            "outcome", "outcome_kind", "feature_id", "n_rows", "n_units",
            "analysis_unit", "feature_mean", "outcome_mean", "correlation",
            "slope", "p_value", "q_value", "estimand", "inference_test",
        }
        missing = required - set(self.table.columns)
        if missing:
            raise ValueError(f"association table is missing {sorted(missing)}")
        object.__setattr__(self, "table", self.table.copy(deep=True))


def _numeric_matrix(values) -> tuple[np.ndarray, tuple[str, ...] | None, int]:
    inferred_names = None
    if isinstance(values, pd.DataFrame):
        inferred_names = tuple(str(column) for column in values.columns)
        array = values.to_numpy(dtype=object)
        original_ndim = 2
    elif isinstance(values, pd.Series):
        inferred_names = ((str(values.name),) if values.name is not None else None)
        array = values.to_numpy(dtype=object)
        original_ndim = 1
    else:
        array = np.asarray(values, dtype=object)
        original_ndim = array.ndim
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError(f"outcome values must be 1-D or 2-D, got shape {array.shape}")
    try:
        numeric = pd.to_numeric(
            pd.Series(array.reshape(-1), dtype=object), errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("outcome values must be numeric or missing") from exc
    matrix = numeric.to_numpy(dtype=float, na_value=np.nan).reshape(array.shape)
    if np.isinf(matrix).any():
        raise ValueError("outcome values must be finite or missing, not infinite")
    return matrix, inferred_names, original_ndim


def normalize_outcomes(
    values,
    *,
    kind: OutcomeKind,
    names=None,
    normalization: Normalization = "auto",
) -> NormalizedOutcomes:
    """Validate and normalize one or more aligned outcomes without imputing missing data.

    ``binary`` accepts only 0/1, while ``probability`` and ``preference`` accept the
    closed interval [0, 1]. ``continuous`` is a single real-valued rating and
    ``multi_continuous`` is a 2-D matrix of real-valued attributes. ``auto`` leaves
    bounded outcomes on their natural scale and z-scores continuous attributes using
    observed rows only (population standard deviation). Constant or all-missing
    attributes use scale 1, so observed constant values normalize to zero and missing
    values remain missing.
    """
    if kind not in OUTCOME_KINDS:
        raise ValueError(f"kind must be one of {list(OUTCOME_KINDS)}")
    if normalization not in OUTCOME_NORMALIZATIONS:
        raise ValueError(
            f"normalization must be one of {list(OUTCOME_NORMALIZATIONS)}")
    raw, inferred_names, original_ndim = _numeric_matrix(values)
    width = raw.shape[1]
    if kind == "multi_continuous":
        if original_ndim != 2:
            raise ValueError("multi_continuous outcomes must be a 2-D matrix")
    elif width != 1:
        raise ValueError(f"{kind} outcomes must contain exactly one column")

    observed = np.isfinite(raw)
    present = raw[observed]
    if kind == "binary" and present.size and not np.isin(present, [0.0, 1.0]).all():
        raise ValueError("binary outcomes must contain only 0, 1, or missing values")
    if kind in {"probability", "preference"} and present.size:
        if ((present < 0.0) | (present > 1.0)).any():
            raise ValueError(
                f"{kind} outcomes must be inside [0, 1] or missing")

    if names is None:
        if inferred_names is not None:
            resolved_names = inferred_names
        elif width == 1:
            resolved_names = ("outcome",)
        else:
            resolved_names = tuple(f"outcome_{j}" for j in range(width))
    else:
        resolved_names = tuple(str(name) for name in names)
    if len(resolved_names) != width:
        raise ValueError("names must have one entry per outcome attribute")
    if len(set(resolved_names)) != width or any(not name for name in resolved_names):
        raise ValueError("outcome names must be unique non-empty strings")

    resolved_normalization = (
        "zscore" if normalization == "auto" and kind in {
            "continuous", "multi_continuous",
        } else "none" if normalization == "auto" else normalization
    )
    center = np.zeros(width, dtype=float)
    scale = np.ones(width, dtype=float)
    normalized = raw.copy()
    if resolved_normalization == "zscore":
        for j in range(width):
            column = raw[:, j]
            mask = observed[:, j]
            if not mask.any():
                continue
            center[j] = float(column[mask].mean())
            std = float(column[mask].std(ddof=0))
            scale[j] = std if np.isfinite(std) and std > 0 else 1.0
            normalized[mask, j] = (column[mask] - center[j]) / scale[j]
    normalized[~observed] = np.nan
    return NormalizedOutcomes(
        kind=kind,
        names=resolved_names,
        raw_values=np.asarray(raw, dtype=float),
        values=np.asarray(normalized, dtype=float),
        observed=np.asarray(observed, dtype=bool),
        normalization=resolved_normalization,
        center=center,
        scale=scale,
    )


def _validate_features(features, feature_ids, n_rows: int) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(features)
    if matrix.ndim != 2 or matrix.shape[0] != n_rows:
        raise ValueError(
            f"features must be a 2-D matrix with {n_rows} rows, got {matrix.shape}")
    if not np.issubdtype(matrix.dtype, np.number):
        raise ValueError("features must be numeric")
    if not np.isfinite(matrix).all():
        raise ValueError("features must contain only finite values")
    ids = (
        np.arange(matrix.shape[1], dtype=int)
        if feature_ids is None
        else np.asarray(
            validate_feature_ids(feature_ids, width=matrix.shape[1]), dtype=int)
    )
    return matrix, ids


def _group_structure(outcome: np.ndarray, groups: np.ndarray):
    """Return factor codes/counts/outcome means without allocating groups × features."""
    codes, n_groups = factorize_group_ids(groups)
    counts = np.bincount(codes, minlength=n_groups).astype(float)
    unit_outcome = np.bincount(
        codes, weights=outcome, minlength=n_groups).astype(float)
    if n_groups:
        unit_outcome /= counts
    return codes, counts, unit_outcome


def associate_outcomes(
    features,
    outcomes: NormalizedOutcomes,
    *,
    feature_ids=None,
    group_ids=None,
    min_units: int = 3,
) -> OutcomeAssociationResult:
    """Describe feature/outcome associations at row or equal-weight group level.

    For each feature and outcome attribute, this reports Pearson correlation and the
    simple OLS slope of the normalized outcome on the feature.  When ``group_ids`` is
    provided, feature and outcome values are first averaged within each observed group;
    every group then receives equal weight. Missing outcomes are omitted separately per
    attribute and are never imputed. P-values treat rows (or aggregated groups) as the
    independent analysis units and BH ``q_value`` adjusts the full returned family.

    These are dataset-specific descriptive associations, not causal effects and not
    judgments that a feature is good or bad.
    """
    if not isinstance(outcomes, NormalizedOutcomes):
        raise ValueError("outcomes must be a NormalizedOutcomes result")
    if int(min_units) < 3:
        raise ValueError("min_units must be at least 3")
    matrix, ids = _validate_features(features, feature_ids, outcomes.n_rows)
    groups = None
    if group_ids is not None:
        candidate_groups = np.asarray(group_ids, dtype=object)
        if candidate_groups.ndim != 1 or len(candidate_groups) != outcomes.n_rows:
            raise ValueError("group_ids must have one entry per outcome row")
        groups = validate_group_ids(candidate_groups, outcomes.n_rows)

    rows = []
    for outcome_index, outcome_name in enumerate(outcomes.names):
        mask = outcomes.observed[:, outcome_index]
        row_features = matrix[mask]
        row_outcome = outcomes.values[mask, outcome_index]
        row_raw_outcome = outcomes.raw_values[mask, outcome_index]
        group_codes = group_counts = None
        association_center = float(outcomes.center[outcome_index])
        association_scale = float(outcomes.scale[outcome_index])
        if groups is None:
            unit_outcome = row_outcome
            analysis_unit = "row"
            estimand = "row_weighted_feature_outcome_association"
        else:
            group_codes, group_counts, unit_raw_outcome = _group_structure(
                row_raw_outcome, groups[mask])
            if outcomes.normalization == "zscore" and len(unit_raw_outcome):
                association_center = float(unit_raw_outcome.mean())
                candidate_scale = float(unit_raw_outcome.std(ddof=0))
                association_scale = (
                    candidate_scale if np.isfinite(candidate_scale)
                    and candidate_scale > 0 else 1.0)
                unit_outcome = (
                    unit_raw_outcome - association_center) / association_scale
            else:
                unit_outcome = unit_raw_outcome
            analysis_unit = "group"
            estimand = "equal_group_weight_group_mean_association"
        n_rows = int(mask.sum())
        n_units = int(len(unit_outcome))
        outcome_mean = (
            float(unit_outcome.mean()) if n_units else float("nan"))
        for column, feature_id in enumerate(ids):
            if group_codes is None:
                x = row_features[:, column]
            else:
                x = np.bincount(
                    group_codes, weights=row_features[:, column], minlength=n_units,
                ).astype(float)
                if n_units:
                    x /= group_counts
            feature_mean = float(x.mean()) if n_units else float("nan")
            correlation = p_value = slope = float("nan")
            feature_high = outcome_high = 0
            feature_low = outcome_low = n_units
            inference_supported = False
            varying = (
                n_units >= int(min_units) and np.ptp(x) > 0
                and np.ptp(unit_outcome) > 0)
            if varying:
                correlation = float(pearsonr(x, unit_outcome).statistic)
                centered = x - feature_mean
                slope = float(
                    np.dot(centered, unit_outcome - outcome_mean)
                    / np.dot(centered, centered))
                feature_threshold = (float(x.min()) + float(x.max())) / 2.0
                outcome_threshold = (
                    float(unit_outcome.min()) + float(unit_outcome.max())) / 2.0
                feature_high = int((x > feature_threshold).sum())
                feature_low = int(n_units - feature_high)
                outcome_high = int((unit_outcome > outcome_threshold).sum())
                outcome_low = int(n_units - outcome_high)
                inference_supported = bool(
                    n_units >= max(int(min_units), 10)
                    and min(feature_low, feature_high, outcome_low, outcome_high) >= 5)
                if inference_supported:
                    feature_is_high = x > feature_threshold
                    outcome_is_high = unit_outcome > outcome_threshold
                    contingency = np.array([
                        [np.sum(feature_is_high & outcome_is_high),
                         np.sum(feature_is_high & ~outcome_is_high)],
                        [np.sum(~feature_is_high & outcome_is_high),
                         np.sum(~feature_is_high & ~outcome_is_high)],
                    ])
                    p_value = float(fisher_exact(contingency).pvalue)
            rows.append({
                "outcome": outcome_name,
                "outcome_kind": outcomes.kind,
                "feature_id": int(feature_id),
                "n_rows": n_rows,
                "n_units": n_units,
                "analysis_unit": analysis_unit,
                "feature_mean": feature_mean,
                "outcome_mean": outcome_mean,
                "correlation": float(correlation),
                "slope": slope,
                "p_value": float(p_value),
                "feature_low_units": feature_low,
                "feature_high_units": feature_high,
                "outcome_low_units": outcome_low,
                "outcome_high_units": outcome_high,
                "association_outcome_center": association_center,
                "association_outcome_scale": association_scale,
                "inference_supported": inference_supported,
                "missingness_policy": "per_outcome_attribute_complete_cases",
                "tie_policy": (
                    "retained_as_0.5_neutral"
                    if outcomes.kind == "preference" else "not_applicable"),
                "estimand": estimand,
                "inference_test": (
                    "fisher_exact_range_midpoint_split" if inference_supported
                    else "not_run_thin_independent_support"),
            })
    columns = [
        "outcome", "outcome_kind", "feature_id", "n_rows", "n_units",
        "analysis_unit", "feature_mean", "outcome_mean", "correlation", "slope",
        "p_value", "q_value", "feature_low_units", "feature_high_units",
        "outcome_low_units", "outcome_high_units", "association_outcome_center",
        "association_outcome_scale", "inference_supported",
        "missingness_policy", "tie_policy", "estimand", "inference_test",
    ]
    table = pd.DataFrame(rows)
    if table.empty:
        table = pd.DataFrame(columns=columns)
    else:
        table["q_value"] = bh_adjust(table["p_value"].to_numpy(dtype=float))
        table = table[columns]
    result_estimand = (
        "equal group weight association between group-mean feature activation and "
        "group-mean outcome; descriptive, not a causal effect" if groups is not None else
        "row-weighted association between feature activation and outcome; descriptive, "
        "not a causal effect"
    )
    return OutcomeAssociationResult(
        table=table, outcomes=outcomes, grouped=groups is not None,
        method="pearson_ols_descriptive_and_fisher_exact_range_midpoint_split",
        estimand=result_estimand)


def associate_outcomes_by_group(
    features,
    outcomes: NormalizedOutcomes,
    group_ids,
    *,
    feature_ids=None,
    min_groups: int = 3,
) -> OutcomeAssociationResult:
    """Explicit equal-group-weight wrapper around :func:`associate_outcomes`."""
    return associate_outcomes(
        features, outcomes, feature_ids=feature_ids, group_ids=group_ids,
        min_units=min_groups,
    )
