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
