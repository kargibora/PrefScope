"""Reusable statistical primitives shared across feature analyses."""
from __future__ import annotations

import numpy as np
from scipy.stats import ttest_ind


def inside_outside_contrast(inside, outside) -> dict:
    """Welch two-sample contrast of an ``inside`` group against an ``outside`` group.

    The building block behind "does this group express something more than the
    rest?" — e.g. a model vs the pool in model diagnosis, or the items where one
    feature fires vs where it is silent when relating two feature sets.

    Returns ``mean_inside``, ``mean_outside``, ``delta`` (= mean_inside −
    mean_outside), Welch's ``welch_t`` / ``welch_p`` (unequal variances), and
    ``cohens_d`` (delta standardized by the pooled standard deviation). Degenerate
    inputs (fewer than two samples on a side, or both sides constant) still report
    the means and delta but leave the test statistics as NaN.
    """
    inside = np.asarray(inside, dtype=np.float64)
    outside = np.asarray(outside, dtype=np.float64)
    mean_in = float(inside.mean()) if inside.size else float("nan")
    mean_out = float(outside.mean()) if outside.size else float("nan")
    out = {"mean_inside": mean_in, "mean_outside": mean_out,
           "delta": mean_in - mean_out, "welch_t": float("nan"),
           "welch_p": float("nan"), "cohens_d": float("nan")}
    if inside.size < 2 or outside.size < 2 or (inside.var() == 0 and outside.var() == 0):
        return out
    t, p = ttest_ind(inside, outside, equal_var=False)
    s_pool = np.sqrt((inside.var(ddof=1) + outside.var(ddof=1)) / 2.0)
    out["welch_t"] = float(t)
    out["welch_p"] = float(p)
    out["cohens_d"] = float(out["delta"] / s_pool) if s_pool > 0 else float("nan")
    return out


def benjamini_hochberg(p_values) -> np.ndarray:
    """Adjust finite p-values with BH while preserving missing entries."""
    p = np.asarray(p_values, dtype=float)
    q = np.full(p.shape, np.nan, dtype=float)
    valid = np.flatnonzero(np.isfinite(p))
    if not len(valid):
        return q
    order = valid[np.argsort(p[valid])]
    ranked = p[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    q[order] = np.clip(adjusted, 0.0, 1.0)
    return q


def bounded_mean_hoeffding(
    values,
    *,
    lower: float,
    upper: float,
    confidence: float = 0.95,
) -> dict[str, float]:
    """Distribution-free mean CI and two-sided test of a zero mean.

    Observations must be independent and lie in the declared closed interval.
    """
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError("values must be a non-empty finite 1-D array")
    if not np.isfinite([lower, upper]).all() or not lower < upper:
        raise ValueError("lower and upper must define a finite increasing interval")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    tolerance = np.finfo(float).eps * 16
    if np.any(x < lower - tolerance) or np.any(x > upper + tolerance):
        raise ValueError("values fall outside the declared bounded interval")
    mean = float(x.mean())
    width = float(upper - lower)
    alpha = 1.0 - confidence
    radius = width * np.sqrt(np.log(2.0 / alpha) / (2.0 * len(x)))
    p_value = min(1.0, 2.0 * np.exp(-2.0 * len(x) * mean * mean / (width * width)))
    return {
        "mean": mean,
        "ci_low": max(lower, mean - radius),
        "ci_high": min(upper, mean + radius),
        "p_value": float(p_value),
    }


def bounded_mean_difference_hoeffding(
    inside,
    outside,
    *,
    lower: float,
    upper: float,
    confidence: float = 0.95,
) -> dict[str, float]:
    """Distribution-free two-sample difference-in-means inference."""
    left = np.asarray(inside, dtype=float)
    right = np.asarray(outside, dtype=float)
    if left.ndim != 1 or right.ndim != 1 or not len(left) or not len(right):
        raise ValueError("inside and outside must be non-empty 1-D arrays")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("inside and outside must contain finite values")
    if not np.isfinite([lower, upper]).all() or not lower < upper:
        raise ValueError("lower and upper must define a finite increasing interval")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    tolerance = np.finfo(float).eps * 16
    if (
        np.any(left < lower - tolerance)
        or np.any(left > upper + tolerance)
        or np.any(right < lower - tolerance)
        or np.any(right > upper + tolerance)
    ):
        raise ValueError("values fall outside the declared bounded interval")
    effect = float(left.mean() - right.mean())
    width = float(upper - lower)
    inverse_support = 1.0 / len(left) + 1.0 / len(right)
    alpha = 1.0 - confidence
    radius = width * np.sqrt(
        0.5 * inverse_support * np.log(2.0 / alpha))
    p_value = min(
        1.0,
        2.0 * np.exp(
            -2.0 * effect * effect / (width * width * inverse_support)),
    )
    return {
        "difference": effect,
        "ci_low": max(-width, effect - radius),
        "ci_high": min(width, effect + radius),
        "p_value": float(p_value),
    }
