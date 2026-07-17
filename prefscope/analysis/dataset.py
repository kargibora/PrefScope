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


def feature_confound_correlation(z, surrogate) -> pd.DataFrame:
    """Per-feature correlation of the reward direction sign(z[:,f]) with a
    per-example surrogate confound (e.g. length(chosen) - length(rejected)).
    Returns [feature_id, corr] sorted by |corr| (NaN last). High |corr| ⇒ the
    feature's preference direction tracks the confound (design §3.4 option 2).
    """
    s = np.sign(np.asarray(z, dtype=np.float64))
    y = np.asarray(surrogate, dtype=np.float64)
    n, m = s.shape
    corr = np.full(m, np.nan)
    if n > 1 and np.std(y) > 0:
        for f in range(m):
            col = s[:, f]
            if np.std(col) > 0:
                corr[f] = float(np.corrcoef(col, y)[0, 1])
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
