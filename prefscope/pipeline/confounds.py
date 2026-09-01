"""Screen preference-associated response concepts for a length confound.

This module is intentionally a screening analysis, not a bias classifier.  A feature
that co-varies with response length may still represent real quality; the result says
that the observational data do not cleanly separate those explanations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from prefscope.analysis.dataset import feature_confound_correlation
from prefscope.analysis.grouping import factorize_group_ids, validate_group_ids
from prefscope.pipeline.winrelevance import win_relevance


def partial_correlation(x, y, control) -> float:
    """Pearson correlation of ``x`` and ``y`` after one scalar control."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    control = np.asarray(control, dtype=float)
    if not (len(x) == len(y) == len(control)):
        raise ValueError("partial-correlation inputs must have equal length")
    if not len(x) or min(np.std(x), np.std(y), np.std(control)) == 0:
        return float("nan")
    rxy = np.corrcoef(x, y)[0, 1]
    rxz = np.corrcoef(x, control)[0, 1]
    ryz = np.corrcoef(y, control)[0, 1]
    denominator = np.sqrt(max(0.0, (1.0 - rxz**2) * (1.0 - ryz**2)))
    # At perfect/nearly-perfect collinearity the controlled residual has no usable
    # variance. Floating-point roundoff can leave a tiny positive denominator and a
    # plausible-looking arbitrary ratio, so fail closed instead of reporting it.
    return (
        float((rxy - rxz * ryz) / denominator)
        if denominator > 1e-12 else float("nan")
    )


def screen_length_confound(
    z_diff,
    human_pref,
    length_difference,
    *,
    annotations: pd.DataFrame | None = None,
    confound_threshold: float = 0.3,
    collapse_fraction: float = 0.5,
    permutations: int = 0,
    seed: int = 0,
    group_ids=None,
) -> tuple[pd.DataFrame, dict]:
    """Return a per-feature length-confound screen and permutation summary.

    All correlations for a feature use the same rows: battles on which its signed
    A-minus-B code is nonzero. ``human_pref`` is P(A preferred), including 0.5 ties.
    ``confound_entangled`` means strong length covariance and either at least a
    ``collapse_fraction`` reduction in absolute outcome correlation after controlling
    for length or an unidentified residual under collinearity. It is evidence of
    entanglement, not evidence that a concept is bad.
    """
    z = np.asarray(z_diff)
    y = np.asarray(human_pref, dtype=float)
    length = np.asarray(length_difference, dtype=float)
    if z.ndim != 2:
        raise ValueError(f"z_diff must be 2-D, got shape {z.shape}")
    if not (len(z) == len(y) == len(length)):
        raise ValueError("z_diff, human_pref, and length_difference must align by row")
    if not np.isfinite(z).all() or not np.isfinite(y).all() or not np.isfinite(length).all():
        raise ValueError("confound-screen inputs must be finite")
    if ((y < 0.0) | (y > 1.0)).any():
        raise ValueError("human_pref must contain probabilities in [0, 1]")
    if float(confound_threshold) < 0 or float(confound_threshold) > 1:
        raise ValueError("confound_threshold must be in [0, 1]")
    if float(collapse_fraction) < 0 or float(collapse_fraction) > 1:
        raise ValueError("collapse_fraction must be in [0, 1]")
    if int(permutations) < 0:
        raise ValueError("permutations must be non-negative")
    groups = None
    grouped = False
    if group_ids is not None:
        groups = validate_group_ids(group_ids, len(z))
        _, total_groups = factorize_group_ids(groups)
        grouped = total_groups < len(z)

    outcome = 2.0 * y - 1.0
    relevance = win_relevance(z, y, group_ids=group_ids)[
        [
            "feature_id", "win_assoc", "correlation", "n_fire", "n_groups",
            "n_independent_groups", "estimand", "correlation_test", "significant",
        ]
    ]
    residual = []
    if not grouped:
        confound = feature_confound_correlation(
            z, length, nonzero_only=True
        ).rename(columns={"corr": "corr_confound_len"})
        for feature_id in range(z.shape[1]):
            column = z[:, feature_id]
            firing = column != 0
            residual.append({
                "feature_id": feature_id,
                "correlation_resid_len": partial_correlation(
                    np.sign(column[firing]), outcome[firing], length[firing]
                ),
                "n_confound_groups": int(firing.sum()),
                "confound_estimand": "row_weighted_firing_rows",
            })
    else:
        confound_rows = []
        for feature_id in range(z.shape[1]):
            column = z[:, feature_id]
            firing = column != 0
            codes, n_feature_groups = factorize_group_ids(groups[firing])
            counts = np.bincount(codes, minlength=n_feature_groups).astype(float)
            sign_mean = np.bincount(
                codes, weights=np.sign(column[firing]), minlength=n_feature_groups
            ) / counts
            outcome_mean = np.bincount(
                codes, weights=outcome[firing], minlength=n_feature_groups
            ) / counts
            length_mean = np.bincount(
                codes, weights=length[firing], minlength=n_feature_groups
            ) / counts
            corr_length = (
                float(np.corrcoef(sign_mean, length_mean)[0, 1])
                if n_feature_groups > 1
                and np.ptp(sign_mean) > 0 and np.ptp(length_mean) > 0
                else float("nan")
            )
            confound_rows.append({
                "feature_id": feature_id,
                "corr_confound_len": corr_length,
            })
            residual.append({
                "feature_id": feature_id,
                "correlation_resid_len": partial_correlation(
                    sign_mean, outcome_mean, length_mean),
                "n_confound_groups": n_feature_groups,
                "confound_estimand": "equal_group_weight_firing_group_means",
            })
        confound = pd.DataFrame(confound_rows)
    result = relevance.merge(confound, on="feature_id").merge(
        pd.DataFrame(residual), on="feature_id"
    )
    result["confound_entangled"] = (
        (result["corr_confound_len"].abs() >= float(confound_threshold))
        & (result["correlation"].abs() > 0)
        & (
            result["correlation_resid_len"].isna()
            | (
                result["correlation_resid_len"].abs()
                < float(collapse_fraction) * result["correlation"].abs()
            )
        )
    )

    if annotations is not None:
        if "feature_id" not in annotations.columns:
            raise ValueError("annotations need a feature_id column")
        table = annotations.copy()
        table["feature_id"] = pd.to_numeric(
            table["feature_id"], errors="raise"
        ).astype(int)
        table = table.drop_duplicates("feature_id", keep="last")
        columns = [
            column for column in ("feature_id", "concept", "fidelity_pass")
            if column in table.columns
        ]
        result = result.merge(table[columns], on="feature_id", how="left")
        front = [column for column in columns if column in result.columns]
        result = result[front + [column for column in result.columns if column not in front]]

    n_observed = int(relevance["significant"].sum())
    summary = {
        "n_rows": int(len(z)),
        "n_features": int(z.shape[1]),
        "analysis_unit": "group" if grouped else "row",
        "confound_estimand": (
            "equal_group_weight_firing_group_means" if grouped
            else "row_weighted_firing_rows"),
        "n_significant_observed": n_observed,
        "n_confound_entangled": int(result["confound_entangled"].sum()),
        "confound_threshold": float(confound_threshold),
        "collapse_fraction": float(collapse_fraction),
        "permutations": int(permutations),
        "permutation_significant_mean": None,
        "permutation_significant_p95": None,
        "permutation_significant_max": None,
        "permutation_empirical_p": None,
    }
    if int(permutations):
        if grouped:
            raise ValueError(
                "permutation null with repeated groups is not supported; use "
                "group-aware win-relevance inference without --permute")
        rng = np.random.default_rng(seed)
        null = np.asarray([
            int(win_relevance(z, rng.permutation(y))["significant"].sum())
            for _ in range(int(permutations))
        ])
        summary.update({
            "permutation_significant_mean": float(null.mean()),
            "permutation_significant_p95": float(np.percentile(null, 95)),
            "permutation_significant_max": int(null.max()),
            "permutation_empirical_p": float(
                ((null >= n_observed).sum() + 1) / (len(null) + 1)
            ),
        })
    return result, summary


__all__ = ["partial_correlation", "screen_length_confound"]
