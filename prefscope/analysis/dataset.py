"""Diagnose a pairwise preference dataset from its difference-lens codes.

Operates on ``z`` (N, M): signed SAE codes of the chosen-minus-rejected contrast,
with the fixed orientation ``chosen = A`` so ``z[i, f] > 0`` means feature ``f``
is expressed more strongly in the chosen response of example ``i``. Implements the
math of docs/dataset-diagnosis-design.md §3: per-feature reward direction ``r_f``,
split-half sign stability, and the per-example ``spurious_share`` and
``label_inconsistency`` scores. Pure numpy/pandas — independent of how ``z`` was
produced.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from prefscope.analysis.grouping import factorize_group_ids, validate_group_ids
from prefscope.analysis.stats import inside_outside_contrast


def dataset_reward(z) -> np.ndarray:
    """r_f = mean_i sign(z[i, f]) in [-1, 1] (design §3.1).

    +1 ⇒ the dataset systematically prefers responses expressing feature f;
    -1 ⇒ systematically penalizes it; ~0 ⇒ label-irrelevant.

    Examples where f does not fire (z[i,f] == 0) contribute sign(0) == 0, exactly
    as design §3.1 averages over all N rows — so a very sparse feature is pulled
    toward 0 (and then dropped by split-half stability). This intentionally
    differs from ``winrelevance.win_relevance``, which conditions on firing rows.
    """
    z = np.asarray(z, dtype=np.float64)
    if z.shape[0] == 0:
        return np.zeros(z.shape[1])
    return np.sign(z).mean(axis=0)


def split_half_stable(z, effect_fn, *, seed: int = 0) -> pd.DataFrame:
    """Recompute a per-feature effect on two disjoint random halves; flag features
    whose effect has the same (nonzero) sign on both halves (design §3.1/§5).

    ``effect_fn``: (n, M) -> (M,) per-feature statistic (e.g. ``dataset_reward``).
    Returns columns: feature_id, effect (full), effect_a, effect_b, stable.
    """
    z = np.asarray(z, dtype=np.float64)
    n, m = z.shape
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    a_idx, b_idx = perm[: n // 2], perm[n // 2:]
    eff = np.asarray(effect_fn(z), dtype=np.float64)
    eff_a = np.asarray(effect_fn(z[a_idx]), dtype=np.float64)
    eff_b = np.asarray(effect_fn(z[b_idx]), dtype=np.float64)
    stable = (np.sign(eff_a) == np.sign(eff_b)) & (eff_a != 0) & (eff_b != 0)
    return pd.DataFrame({"feature_id": np.arange(m), "effect": eff,
                         "effect_a": eff_a, "effect_b": eff_b, "stable": stable})


def spurious_share(z, undesirable, *, eps: float = 1e-9) -> np.ndarray:
    """Per-example share of the chosen-vs-rejected difference carried by the
    undesirable features U (design §3.2A):

        spurious_share(i) = sum_{f in U} |z[i,f]| / max(sum_f |z[i,f]|, eps)

    High ⇒ the preference is mostly explained by a confound (length, format, …).
    ``eps`` is a zero-floor for all-silent rows, not an additive offset, so a pair
    whose entire difference is on undesirable features scores exactly 1.0.
    """
    z = np.abs(np.asarray(z, dtype=np.float64))
    cols = sorted({int(f) for f in undesirable})
    denom = np.maximum(z.sum(axis=1), eps)
    if not cols:
        return np.zeros(z.shape[0])
    return z[:, cols].sum(axis=1) / denom


def label_inconsistency(z, reward, undesirable) -> np.ndarray:
    """Per-example agreement with the dataset's reward pattern on the NON-spurious
    (quality) features (design §3.2B):

        a_i = sum_{f not in U} sign(z[i,f]) * r_f

    a_i < 0 ⇒ on the genuine-quality axes the chosen response is the weaker one,
    yet it is labeled preferred → candidate mislabel / confounded pair.
    """
    z = np.asarray(z, dtype=np.float64)
    reward = np.asarray(reward, dtype=np.float64)
    drop = {int(f) for f in undesirable}
    keep = [f for f in range(z.shape[1]) if f not in drop]
    return np.sign(z[:, keep]) @ reward[keep]


def diagnose_dataset(z, undesirable, *, ids=None, names=None, seed: int = 0):
    """Compose the dataset diagnosis. Returns (per_feature_df, per_sample_df).

    per_feature: dataset reward r_f + split-half stability (+ concept name if
    ``names`` has feature_id/concept). per_sample: spurious_share + label
    inconsistency, one row per example (id from ``ids`` or the row index).
    """
    z = np.asarray(z, dtype=np.float64)
    n = z.shape[0]
    per_feature = split_half_stable(z, dataset_reward, seed=seed)
    if names is not None and "concept" in getattr(names, "columns", []):
        per_feature = per_feature.merge(names[["feature_id", "concept"]],
                                        on="feature_id", how="left")
    reward = per_feature.sort_values("feature_id")["effect"].to_numpy()
    per_sample = pd.DataFrame({
        "id": list(ids) if ids is not None else list(range(n)),
        "spurious_share": spurious_share(z, undesirable),
        "label_inconsistency": label_inconsistency(z, reward, undesirable),
    })
    return per_feature, per_sample


def region_behavior_contrast(z, cluster_ids, *, seed: int = 0) -> pd.DataFrame:
    """For each example-cluster (region B_k) and feature m, the feature-conditioned
    contrast Δ_{k,m} = net_direction(sign(z[:,m])) inside B_k minus outside (design
    §1 feature-conditioned). Reported with a Welch p (Bonferroni over all tested
    pairs) and split-half sign stability — a (region, behavior) is trustworthy only
    if both halves agree and it survives correction.

    Returns long-format [cluster_id, feature_id, delta, welch_p, p_bonferroni, stable].
    """
    s = np.sign(np.asarray(z, dtype=np.float64))
    cluster_ids = np.asarray(cluster_ids)
    n, m = s.shape
    perm = np.random.default_rng(seed).permutation(n)
    in_a = np.zeros(n, dtype=bool)
    in_a[perm[: n // 2]] = True

    rows = []
    for k in np.unique(cluster_ids):
        inside = cluster_ids == k
        outside = ~inside
        for f in range(m):
            c = inside_outside_contrast(s[inside, f], s[outside, f])
            if not np.isfinite(c["welch_p"]):
                continue
            da = inside_outside_contrast(s[inside & in_a, f], s[outside & in_a, f])["delta"]
            db = inside_outside_contrast(s[inside & ~in_a, f], s[outside & ~in_a, f])["delta"]
            stable = bool(np.isfinite(da) and np.isfinite(db) and da != 0 and db != 0
                          and np.sign(da) == np.sign(db))
            rows.append({"cluster_id": int(k), "feature_id": int(f),
                         "delta": c["delta"], "welch_p": c["welch_p"], "stable": stable})

    df = pd.DataFrame(rows, columns=["cluster_id", "feature_id", "delta",
                                     "welch_p", "stable"])
    # Bonferroni over ALL attempted (cluster, feature) tests — including degenerate
    # ones dropped above. Using len(df) would shrink the denominator and inflate
    # significance whenever a cluster is too small to contrast.
    n_tests = max(1, int(len(np.unique(cluster_ids)) * m))
    df["p_bonferroni"] = (df["welch_p"] * n_tests).clip(upper=1.0)
    return df[["cluster_id", "feature_id", "delta", "welch_p", "p_bonferroni",
               "stable"]]


def _value_moments(
    z, indices, *, chunk_rows: int = 10_000, take_sign: bool = True,
):
    """Count/sum/squared-sum without materializing a dense float64 copy."""
    indices = np.asarray(indices, dtype=int)
    m = z.shape[1]
    total = np.zeros(m, dtype=np.float64)
    square = np.zeros(m, dtype=np.float64)
    for start in range(0, len(indices), int(chunk_rows)):
        block = np.asarray(z[indices[start:start + int(chunk_rows)]])
        if take_sign:
            block = np.sign(block)
        total += block.sum(axis=0, dtype=np.float64)
        square += np.square(block).sum(axis=0, dtype=np.float64)
    return int(len(indices)), total, square


def _sign_moments(z, indices, *, chunk_rows: int = 10_000):
    return _value_moments(z, indices, chunk_rows=chunk_rows, take_sign=True)


def _moments_mean_var(n, total, square):
    if n <= 0:
        return np.full_like(total, np.nan), np.full_like(total, np.nan)
    mean = total / float(n)
    if n <= 1:
        return mean, np.full_like(total, np.nan)
    var = np.maximum((square - (total * total) / float(n)) / float(n - 1), 0.0)
    return mean, var


def _welch_from_moments(inside, outside):
    """Vectorized Welch test/effect size from (n, sum, square-sum) tuples."""
    from scipy.stats import t as student_t

    ni, si, qi = inside
    no, so, qo = outside
    mi, vi = _moments_mean_var(ni, si, qi)
    mo, vo = _moments_mean_var(no, so, qo)
    delta = mi - mo
    se2 = vi / max(ni, 1) + vo / max(no, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        stat = np.abs(delta) / np.sqrt(se2)
        denom = ((vi / max(ni, 1)) ** 2 / max(ni - 1, 1)
                 + (vo / max(no, 1)) ** 2 / max(no - 1, 1))
        dof = np.square(se2) / denom
        p = 2.0 * student_t.sf(stat, dof)
    zero_se = se2 == 0
    p = np.where(zero_se & (delta != 0), 0.0, p)
    p = np.where(zero_se & (delta == 0), np.nan, p)
    pooled_num = max(ni - 1, 0) * vi + max(no - 1, 0) * vo
    pooled_den = max(ni + no - 2, 0)
    pooled = np.sqrt(pooled_num / pooled_den) if pooled_den else np.full_like(delta, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        effect = delta / pooled
    effect = np.where(pooled > 0, effect, np.nan)
    return delta, p, effect


def _strict_membership_matrix(values) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 2:
        raise ValueError("membership must be a 2-D matrix")
    if raw.dtype == bool:
        return raw
    if not np.issubdtype(raw.dtype, np.number):
        raise ValueError("membership must contain boolean or numeric 0/1 values")
    numeric = np.asarray(raw, dtype=float)
    if not np.isfinite(numeric).all() or not np.isin(numeric, [0.0, 1.0]).all():
        raise ValueError("membership must contain finite boolean or numeric 0/1 values")
    return numeric.astype(bool)


def region_membership_contrast(
    z,
    membership,
    *,
    region_ids=None,
    seed: int = 0,
    min_inside: int = 2,
    min_outside: int = 2,
    chunk_rows: int = 10_000,
    group_ids=None,
) -> pd.DataFrame:
    """Inside-vs-outside behavior contrast for overlapping example regions.

    ``membership`` is ``(N, K)`` boolean and may contain several true regions per
    example. This is the multi-label counterpart of :func:`region_behavior_contrast`;
    it reports support, a bounded two-sample Hoeffding p-value, descriptive Cohen's d,
    Bonferroni correction over every attempted ``K × M`` cell, and random split-half sign stability.

    Moments are accumulated in row chunks, so a memory-mapped ``N × M`` lens is not
    converted into the multi-gigabyte float64 array that a direct ``np.sign`` would
    allocate.
    """
    z = np.asarray(z)
    regions = _strict_membership_matrix(membership)
    if (
        z.ndim != 2
        or not np.issubdtype(z.dtype, np.number)
        or z.shape[0] != regions.shape[0]
    ):
        raise ValueError(
            f"z and membership must be 2-D with equal rows, got {z.shape} and "
            f"{regions.shape}")
    if not np.isfinite(z).all():
        raise ValueError("z must contain only finite values")
    for name, value in {
        "min_inside": min_inside,
        "min_outside": min_outside,
        "chunk_rows": chunk_rows,
    }.items():
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    n, m = z.shape
    if region_ids is None:
        region_ids = np.arange(regions.shape[1], dtype=int)
    region_ids = np.asarray(region_ids)
    if len(region_ids) != regions.shape[1]:
        raise ValueError("region_ids length must equal membership columns")
    try:
        duplicate_regions = pd.Index(region_ids).has_duplicates
    except TypeError as exc:
        raise ValueError("region_ids must contain hashable scalar values") from exc
    if duplicate_regions or pd.isna(region_ids).any():
        raise ValueError("region_ids must contain unique nonmissing values")

    grouped = False
    row_inside = regions.sum(axis=0).astype(int)
    row_outside = (n - row_inside).astype(int)
    if group_ids is not None:
        group_values = validate_group_ids(group_ids, n)
        group_codes, n_groups = factorize_group_ids(group_values)
        if n_groups < n:
            grouped = True
            grouped_values = []
            grouped_regions = []
            for group in range(n_groups):
                index = np.flatnonzero(group_codes == group)
                membership_rows = regions[index]
                if not np.all(membership_rows == membership_rows[0]):
                    raise ValueError(
                        "prompt-region membership must be constant within each group")
                grouped_regions.append(membership_rows[0])
                grouped_values.append(
                    np.sign(np.asarray(z[index])).mean(axis=0, dtype=np.float64))
            z = np.asarray(grouped_values, dtype=np.float64)
            regions = np.asarray(grouped_regions, dtype=bool)
            n = n_groups

    perm = np.random.default_rng(seed).permutation(n)
    half_a = np.zeros(n, dtype=bool)
    half_a[perm[:n // 2]] = True
    all_a = np.flatnonzero(half_a)
    all_b = np.flatnonzero(~half_a)
    global_a = _value_moments(
        z, all_a, chunk_rows=chunk_rows, take_sign=not grouped)
    global_b = _value_moments(
        z, all_b, chunk_rows=chunk_rows, take_sign=not grouped)
    global_full = (
        global_a[0] + global_b[0],
        global_a[1] + global_b[1],
        global_a[2] + global_b[2],
    )

    def subtract(full, part):
        return full[0] - part[0], full[1] - part[1], full[2] - part[2]

    rows = []
    for column, region_id in enumerate(region_ids):
        inside = regions[:, column]
        idx_a = np.flatnonzero(inside & half_a)
        idx_b = np.flatnonzero(inside & ~half_a)
        in_a = _value_moments(
            z, idx_a, chunk_rows=chunk_rows, take_sign=not grouped)
        in_b = _value_moments(
            z, idx_b, chunk_rows=chunk_rows, take_sign=not grouped)
        in_full = (
            in_a[0] + in_b[0],
            in_a[1] + in_b[1],
            in_a[2] + in_b[2],
        )
        out_full = subtract(global_full, in_full)
        if in_full[0] < int(min_inside) or out_full[0] < int(min_outside):
            continue
        delta, _, effect = _welch_from_moments(in_full, out_full)
        # Signed feature values are bounded in [-1, 1]. Use the same
        # distribution-free two-sample Hoeffding bound for rows and groups so a
        # zero empirical variance cannot create false p=0 certainty.
        width = 4.0 / in_full[0] + 4.0 / out_full[0]
        p_value = np.minimum(
            1.0, 2.0 * np.exp(-2.0 * np.square(delta) / width))
        delta_a, _, _ = _welch_from_moments(in_a, subtract(global_a, in_a))
        delta_b, _, _ = _welch_from_moments(in_b, subtract(global_b, in_b))
        stable = (
            np.isfinite(delta_a) & np.isfinite(delta_b)
            & (delta_a != 0) & (delta_b != 0)
            & (np.sign(delta_a) == np.sign(delta_b))
        )
        valid = np.flatnonzero(np.isfinite(p_value))
        for feature_id in valid:
            row = {
                "region_id": (
                    region_id.item() if isinstance(region_id, np.generic) else region_id),
                "feature_id": int(feature_id),
                "n_inside": int(row_inside[column] if grouped else in_full[0]),
                "n_outside": int(row_outside[column] if grouped else out_full[0]),
                "delta": float(delta[feature_id]),
                "cohens_d": float(effect[feature_id]),
                "welch_p": float(p_value[feature_id]),
                "p_value": float(p_value[feature_id]),
                "stable": bool(stable[feature_id]),
                "inference_test": "two_sample_hoeffding_bounded_signed_means",
                "estimand": "row_weighted_mean_signed_prevalence_difference",
                "analysis_unit": "row",
            }
            if grouped:
                row.update({
                    "n_inside_groups": int(in_full[0]),
                    "n_outside_groups": int(out_full[0]),
                    "n_independent_groups": int(n),
                    "estimand": "equal_prompt_group_mean_signed_prevalence",
                    "inference_test": "two_sample_hoeffding_bounded_signed_means",
                    "analysis_unit": "prompt_group",
                })
            rows.append(row)

    columns = [
        "region_id", "feature_id", "n_inside", "n_outside", "delta",
        "cohens_d", "welch_p", "p_value", "p_bonferroni", "stable",
        "inference_test", "estimand", "analysis_unit",
    ]
    if grouped:
        columns += [
            "n_inside_groups", "n_outside_groups", "n_independent_groups",

        ]
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=columns)
    n_tests = max(1, int(regions.shape[1] * m))
    frame["p_bonferroni"] = (frame["welch_p"] * n_tests).clip(upper=1.0)
    return frame[columns]


def feature_confound_correlation(z, surrogate, *, nonzero_only: bool = False) -> pd.DataFrame:
    """Per-feature correlation of the reward direction sign(z[:,f]) with a
    per-example surrogate confound (e.g. length(chosen) - length(rejected)).
    Returns [feature_id, corr] sorted by |corr| (NaN last). High |corr| ⇒ the
    feature's preference direction tracks the confound (design §3.4 option 2).
    With ``nonzero_only=True``, compute each correlation only on rows where that feature
    fires. This matches firing-conditioned reward correlations and is appropriate when
    comparing the raw and length-residualized reward association for a bias screen.
    """
    # Do not cast/sign the full dense matrix. For a 241k × 2048 lens that old path
    # allocated two ~3.7 GiB float64 arrays even when ``z`` was memory-mapped. Read one
    # feature at a time so the confound screen stays usable on ordinary CPU nodes.
    z = np.asarray(z)
    y = np.asarray(surrogate, dtype=np.float64)
    n, m = z.shape
    corr = np.full(m, np.nan)
    if n > 1 and np.std(y) > 0:
        for f in range(m):
            raw = np.asarray(z[:, f], dtype=np.float64)
            mask = raw != 0 if nonzero_only else slice(None)
            col = np.sign(raw[mask])
            yf = y[mask]
            if len(col) > 1 and np.std(col) > 0 and np.std(yf) > 0:
                corr[f] = float(np.corrcoef(col, yf)[0, 1])
    df = pd.DataFrame({"feature_id": np.arange(m), "corr": corr})
    order = df["corr"].abs().sort_values(ascending=False, na_position="last").index
    return df.reindex(order).reset_index(drop=True)


def auto_undesirable(z, surrogate, *, threshold: float = 0.3) -> list:
    """Feature ids whose reward direction correlates with the surrogate beyond
    |corr| >= threshold — auto-tagged spurious (design §3.4 option 2). Use as the
    ``undesirable`` set for ``diagnose_dataset`` without manual labeling.
    """
    df = feature_confound_correlation(z, surrogate)
    return df.loc[df["corr"].abs() >= threshold, "feature_id"].astype(int).tolist()


def symmetric_activity(z_a, z_b) -> np.ndarray:
    """s = (|z_a| + |z_b|) / 2 — the per-example, per-concept activity magnitude,
    independent of which side (chosen/rejected) expressed it. This is the profile
    *Anatomy of Post-Training* (App. B.1) clusters examples on, so that regions
    group by which behaviors an example *involves*, not by preference direction.
    """
    return (np.abs(np.asarray(z_a, dtype=np.float64))
            + np.abs(np.asarray(z_b, dtype=np.float64))) / 2.0
