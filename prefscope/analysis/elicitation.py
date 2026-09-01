"""Prompt-concept ↔ response-concept co-activation analysis.

The descriptive statistic uses response rows:

    lift = P(Y fires | X fires) / P(Y fires)

When response rows repeat an independent prompt group, significance instead uses groups
as the sampling unit. For each group, response prevalence is the fraction of its response
rows where Y fires. The group-level estimand is the difference in mean per-group response
prevalence between groups where X fires and groups where X does not fire. A two-sided distribution-free Hoeffding bound compares those independent group means. Prompt membership must be constant
within every group; ambiguous groups fail closed.

Bonferroni correction covers the full family of statistically testable feature pairs,
including pairs omitted from the returned table by ``min_cooccur``. Lift and the support
counts remain row-level descriptive quantities. They are not causal effects.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from prefscope.analysis.grouping import factorize_group_ids, validate_group_ids
from prefscope.core.features import validate_feature_ids


def _feature_ids(width: int, selected, *, name: str) -> list[int]:
    ids = (
        list(range(width)) if selected is None
        else list(validate_feature_ids(selected))
    )
    if ids and (min(ids) < 0 or max(ids) >= int(width)):
        raise ValueError(f"{name} feature ids must be inside [0, {width})")
    if len(set(ids)) != len(ids):
        raise ValueError(f"{name} feature ids must be unique")
    return ids


def _group_codes(group_ids, n_rows: int) -> tuple[np.ndarray, np.ndarray]:
    values = validate_group_ids(group_ids, n_rows)
    codes, n_groups = factorize_group_ids(values)
    labels = np.empty(n_groups, dtype=object)
    for group in range(n_groups):
        label = values[np.flatnonzero(codes == group)[0]]
        labels[group] = label.item() if isinstance(label, np.generic) else label
    return codes, labels


def _check_group_prompt_membership(
    group_min: np.ndarray,
    group_max: np.ndarray,
    group_labels: np.ndarray,
    pcols: list[int],
) -> np.ndarray:
    mismatch = np.argwhere(group_min != group_max)
    if len(mismatch):
        group_index, feature_index = mismatch[0]
        raise ValueError(
            "prompt membership must be constant within each group; "
            f"group {group_labels[group_index]!r} varies for prompt feature "
            f"{pcols[feature_index]}"
        )
    return group_max.astype(bool, copy=False)


def _bounded_group_test(
    group_prompt: np.ndarray,
    group_response: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compare mean per-group response prevalence for X-present vs X-absent groups."""
    n_groups, n_response_features = group_response.shape
    n_prompt_features = group_prompt.shape[1]
    shape = (n_prompt_features, n_response_features)
    statistic = np.full(shape, np.nan, dtype=np.float64)
    p_value = np.full(shape, np.nan, dtype=np.float64)
    mean_x = np.full(shape, np.nan, dtype=np.float64)
    mean_not_x = np.full(shape, np.nan, dtype=np.float64)
    n_x = group_prompt.sum(axis=0).astype(np.int64)
    n_not_x = n_groups - n_x

    for prompt_index in range(n_prompt_features):
        present = group_prompt[:, prompt_index]
        nx = int(n_x[prompt_index])
        n0 = int(n_not_x[prompt_index])
        if nx == 0 or n0 == 0:
            continue
        values_x = group_response[present]
        values_0 = group_response[~present]
        mx = values_x.mean(axis=0)
        m0 = values_0.mean(axis=0)
        mean_x[prompt_index] = mx
        mean_not_x[prompt_index] = m0
        if nx < 2 or n0 < 2:
            continue

        difference = mx - m0
        # Group response prevalence is bounded in [0, 1]. This two-sample
        # Hoeffding bound remains valid when an arm has zero empirical variance,
        # unlike a degenerate Welch denominator.
        weight_width = 1.0 / nx + 1.0 / n0
        p = np.minimum(
            1.0, 2.0 * np.exp(-2.0 * np.square(difference) / weight_width))
        statistic[prompt_index] = difference
        p_value[prompt_index] = p

    return statistic, p_value, mean_x, mean_not_x, n_x, n_not_x


def _association_from_counts(
    n_x,
    n_y,
    cooc,
    *,
    R: int,
    pcols,
    rcols,
    min_support: int,
    min_cooccur: int,
    group_prompt: np.ndarray | None = None,
    group_response: np.ndarray | None = None,
) -> pd.DataFrame:
    """Finish row-level lift and sampling-unit-aware significance calculations."""
    from scipy.stats import chi2

    p_y = n_y / R
    with np.errstate(divide="ignore", invalid="ignore"):
        p_y_given_x = cooc / n_x[:, None]
        lift = p_y_given_x / p_y[None, :]

    grouped = group_prompt is not None
    shape = cooc.shape
    chi2_stat = np.full(shape, np.nan, dtype=np.float64)
    # ``welch_t`` is retained as a legacy output column but no Welch test is run.
    welch_t = np.full(shape, np.nan, dtype=np.float64)
    group_difference_statistic = np.full(shape, np.nan, dtype=np.float64)
    group_mean_x = np.full(shape, np.nan, dtype=np.float64)
    group_mean_not_x = np.full(shape, np.nan, dtype=np.float64)
    n_groups_x = np.full(len(pcols), np.nan, dtype=np.float64)
    n_groups_not_x = np.full(len(pcols), np.nan, dtype=np.float64)

    row_support = n_x[:, None] >= min_support
    if grouped:
        if group_response is None:
            raise ValueError("group_response is required with group_prompt")
        (
            group_difference_statistic,
            pval,
            group_mean_x,
            group_mean_not_x,
            n_groups_x,
            n_groups_not_x,
        ) = _bounded_group_test(group_prompt, group_response)
        valid_split = (
            (n_groups_x[:, None] >= int(min_support))
            & (n_groups_not_x[:, None] >= 2)
        )
        testable = row_support & valid_split & np.isfinite(pval)
        method = "two_sample_hoeffding_group_prevalence"
        estimand = (
            "difference in mean per-group response prevalence between prompt-feature "
            "present and absent groups"
        )
        n_groups = int(group_prompt.shape[0])
    else:
        # Vectorized 2x2 chi-square with Yates correction for independent response rows.
        a = cooc.astype(np.float64)
        b = n_x[:, None].astype(np.float64) - a
        c = n_y[None, :].astype(np.float64) - a
        R64 = float(R)
        d = R64 - a - b - c
        numerator = R64 * np.clip(np.abs(a * d - b * c) - R64 / 2.0, 0, None) ** 2
        denominator = (a + b) * (c + d) * (a + c) * (b + d)
        with np.errstate(divide="ignore", invalid="ignore"):
            chi2_stat = np.where(denominator > 0, numerator / denominator, np.nan)
        pval = chi2.sf(chi2_stat, 1)
        testable = (
            row_support
            & (n_x[:, None] < R)
            & (n_y[None, :] > 0)
            & (n_y[None, :] < R)
            & np.isfinite(pval)
        )
        method = "chi2_yates_independent_rows"
        estimand = "row-level independence of prompt and response feature firing"
        n_groups = R

    n_tested = int(testable.sum())
    adjusted = np.full(shape, np.nan, dtype=np.float64)
    adjusted[testable] = np.minimum(pval[testable] * max(1, n_tested), 1.0)

    keep = row_support & (cooc >= min_cooccur)
    pi, ci = np.where(keep)
    rows = pd.DataFrame(
        {
            "prompt_feature": np.asarray(pcols)[pi],
            "completion_feature": np.asarray(rcols)[ci],
            "n_x": n_x[pi].astype(int),
            "n_y": n_y[ci].astype(int),
            "n_cooccur": cooc[pi, ci].astype(int),
            "p_y": p_y[ci],
            "p_y_given_x": p_y_given_x[pi, ci],
            "lift": lift[pi, ci],
            "chi2": chi2_stat[pi, ci],
            "welch_t": welch_t[pi, ci],
            "group_difference_statistic": group_difference_statistic[pi, ci],
            "p_value": pval[pi, ci],
            "p_bonferroni": adjusted[pi, ci],
            "n_groups_x": n_groups_x[pi],
            "n_groups_not_x": n_groups_not_x[pi],
            "mean_group_response_x": group_mean_x[pi, ci],
            "mean_group_response_not_x": group_mean_not_x[pi, ci],
        }
    )
    rows["group_prevalence_difference"] = (
        rows["mean_group_response_x"] - rows["mean_group_response_not_x"]
    )
    rows["log2_lift"] = np.log2(rows["lift"].clip(lower=1e-6))
    rows["significant"] = rows["p_bonferroni"].fillna(1.0) < 0.05
    rows["inference_method"] = method
    rows.attrs.update(
        n_tested=n_tested,
        n_groups=n_groups,
        inference_method=method,
        estimand=estimand,
    )
    order = rows.assign(_abs=rows["log2_lift"].abs()).sort_values(
        ["significant", "_abs"], ascending=[False, False]
    ).index
    return rows.reindex(order).reset_index(drop=True)


def _positive_int(value, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_numeric_matrix(value, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} must be a 2-D numeric matrix")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def prompt_response_association(
    prompt_fire: np.ndarray,
    resp_fire: np.ndarray,
    *,
    prompt_features=None,
    resp_features=None,
    min_support: int = 30,
    min_cooccur: int = 5,
    group_ids=None,
    chunk_size: int = 8192,
) -> pd.DataFrame:
    """Return row-level co-activation lift with group-aware inference.

    ``group_ids`` optionally identifies independent prompt groups for response rows. If
    an ID repeats, prompt-feature membership must be constant within that group. The
    returned counts and lift still describe rows, while p-values use a two-sided distribution-free Hoeffding comparison
    of per-group response prevalence. Unique IDs leave the independent-row chi-square
    test in use. Bonferroni correction covers every testable selected feature pair.
    """
    prompt_fire = _finite_numeric_matrix(prompt_fire, name="prompt_fire")
    resp_fire = _finite_numeric_matrix(resp_fire, name="resp_fire")
    if prompt_fire.shape[0] != resp_fire.shape[0]:
        raise ValueError(
            f"row mismatch: prompt {prompt_fire.shape} vs response {resp_fire.shape}"
        )
    chunk_size = _positive_int(chunk_size, name="chunk_size")
    min_support = _positive_int(min_support, name="min_support")
    min_cooccur = _positive_int(min_cooccur, name="min_cooccur")
    R = int(prompt_fire.shape[0])
    pcols = _feature_ids(prompt_fire.shape[1], prompt_features, name="prompt")
    rcols = _feature_ids(resp_fire.shape[1], resp_features, name="response")

    codes = labels = None
    grouped = False
    if group_ids is not None:
        codes, labels = _group_codes(group_ids, R)
        grouped = len(labels) < R
    n_x = np.zeros(len(pcols), dtype=np.float64)
    n_y = np.zeros(len(rcols), dtype=np.float64)
    cooc = np.zeros((len(pcols), len(rcols)), dtype=np.float64)
    if grouped:
        n_groups = len(labels)
        group_min = np.ones((n_groups, len(pcols)), dtype=np.int8)
        group_max = np.zeros((n_groups, len(pcols)), dtype=np.int8)
        group_response_sum = np.zeros((n_groups, len(rcols)), dtype=np.float64)
        group_row_count = np.zeros(n_groups, dtype=np.int64)

    for start in range(0, R, chunk_size):
        stop = min(start + chunk_size, R)
        Pf = np.ascontiguousarray(prompt_fire[start:stop, pcols] > 0, dtype=np.float32)
        Rf = np.ascontiguousarray(resp_fire[start:stop, rcols] > 0, dtype=np.float32)
        n_x += Pf.sum(0)
        n_y += Rf.sum(0)
        with np.errstate(all="ignore"):
            cooc += Pf.T @ Rf
        if grouped:
            chunk_codes = codes[start:stop]
            np.minimum.at(group_min, chunk_codes, Pf.astype(np.int8, copy=False))
            np.maximum.at(group_max, chunk_codes, Pf.astype(np.int8, copy=False))
            np.add.at(group_response_sum, chunk_codes, Rf)
            np.add.at(group_row_count, chunk_codes, 1)

    group_prompt = group_response = None
    if grouped:
        group_prompt = _check_group_prompt_membership(
            group_min, group_max, labels, pcols
        )
        group_response = group_response_sum / group_row_count[:, None]
    return _association_from_counts(
        n_x,
        n_y,
        cooc,
        R=R,
        pcols=pcols,
        rcols=rcols,
        min_support=min_support,
        min_cooccur=min_cooccur,
        group_prompt=group_prompt,
        group_response=group_response,
    )


def prompt_response_association_paired(
    prompt_fire: np.ndarray,
    resp_a: np.ndarray,
    resp_b: np.ndarray,
    *,
    prompt_features=None,
    resp_features=None,
    min_support: int = 30,
    min_cooccur: int = 5,
    group_ids=None,
    chunk_size: int = 8192,
) -> pd.DataFrame:
    """Analyze two responses per prompt without materializing a stacked matrix.

    Each prompt row is an independent group by default. Repeated ``group_ids`` may join
    several prompt rows, but selected prompt-feature membership must then be constant.
    Counts and lift use both response rows; inference compares per-group response
    prevalence with a two-sided distribution-free Hoeffding bound.
    """
    prompt_fire = _finite_numeric_matrix(prompt_fire, name="prompt_fire")
    resp_a = _finite_numeric_matrix(resp_a, name="resp_a")
    resp_b = _finite_numeric_matrix(resp_b, name="resp_b")
    if resp_a.shape != resp_b.shape or len(prompt_fire) != len(resp_a):
        raise ValueError(
            f"row/shape mismatch: prompt {prompt_fire.shape}, A {resp_a.shape}, "
            f"B {resp_b.shape}"
        )
    chunk_size = _positive_int(chunk_size, name="chunk_size")
    min_support = _positive_int(min_support, name="min_support")
    min_cooccur = _positive_int(min_cooccur, name="min_cooccur")
    n_items = int(len(prompt_fire))
    pcols = _feature_ids(prompt_fire.shape[1], prompt_features, name="prompt")
    rcols = _feature_ids(resp_a.shape[1], resp_features, name="response")
    if group_ids is None:
        codes = np.arange(n_items, dtype=np.int64)
        labels = codes.astype(object)
    else:
        codes, labels = _group_codes(group_ids, n_items)
    n_groups = len(labels)

    n_x = np.zeros(len(pcols), dtype=np.float64)
    n_y = np.zeros(len(rcols), dtype=np.float64)
    cooc = np.zeros((len(pcols), len(rcols)), dtype=np.float64)
    group_min = np.ones((n_groups, len(pcols)), dtype=np.int8)
    group_max = np.zeros((n_groups, len(pcols)), dtype=np.int8)
    group_response_sum = np.zeros((n_groups, len(rcols)), dtype=np.float64)
    group_response_count = np.zeros(n_groups, dtype=np.int64)

    for start in range(0, n_items, chunk_size):
        stop = min(start + chunk_size, n_items)
        Pf = np.ascontiguousarray(prompt_fire[start:stop, pcols] > 0, dtype=np.float32)
        Ra = np.ascontiguousarray(resp_a[start:stop, rcols] > 0, dtype=np.float32)
        Rb = np.ascontiguousarray(resp_b[start:stop, rcols] > 0, dtype=np.float32)
        n_x += 2.0 * Pf.sum(0)
        n_y += Ra.sum(0) + Rb.sum(0)
        with np.errstate(all="ignore"):
            cooc += Pf.T @ Ra
            cooc += Pf.T @ Rb
        chunk_codes = codes[start:stop]
        pf_int = Pf.astype(np.int8, copy=False)
        np.minimum.at(group_min, chunk_codes, pf_int)
        np.maximum.at(group_max, chunk_codes, pf_int)
        np.add.at(group_response_sum, chunk_codes, Ra + Rb)
        np.add.at(group_response_count, chunk_codes, 2)

    group_prompt = _check_group_prompt_membership(group_min, group_max, labels, pcols)
    group_response = group_response_sum / group_response_count[:, None]
    return _association_from_counts(
        n_x,
        n_y,
        cooc,
        R=2 * n_items,
        pcols=pcols,
        rcols=rcols,
        min_support=min_support,
        min_cooccur=min_cooccur,
        group_prompt=group_prompt,
        group_response=group_response,
    )
