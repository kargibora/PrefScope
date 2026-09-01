"""Which SAE directions do *humans* reward? (the WIMHF reward question).

Given the lens's contrast codes ``z_diff`` (A-minus-B activations) and the human
preference ``human_pref`` = P(A preferred) per battle, measure, per feature,
whether the A-side expressing the concept more goes with humans preferring A.

This is model-independent — it characterises the *features* against human
feedback. Crossed with a model's diagnosis (``net_direction``), it answers the
actionable question: does the model under-express a behaviour humans reward?
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest, fisher_exact, pearsonr, t as student_t


def _validated_group_codes(group_ids, n_rows: int) -> np.ndarray | None:
    """Validate row-aligned, non-missing, hashable group ids and factorize them."""
    if group_ids is None:
        return None
    groups = np.asarray(group_ids, dtype=object)
    if groups.ndim != 1 or len(groups) != n_rows:
        raise ValueError(
            f"group_ids must be one-dimensional with {n_rows} rows; got "
            f"shape {groups.shape}")
    if bool(np.asarray(pd.isna(groups)).any()):
        raise ValueError("group_ids must not contain missing values")
    try:
        codes, _ = pd.factorize(groups, sort=False)
    except TypeError as exc:
        raise ValueError("group_ids must contain hashable scalar values") from exc
    if bool((codes < 0).any()):
        raise ValueError("group_ids must not contain missing values")
    return np.asarray(codes, dtype=int)


def _bounded_mean_p(value: float, null: float, n_groups: int, width: float) -> float:
    """Two-sided Hoeffding bound for an equal-weight mean of bounded groups."""
    if n_groups < 1 or not np.isfinite(value):
        return float("nan")
    delta = abs(float(value) - float(null))
    return float(min(1.0, 2.0 * np.exp(-2.0 * n_groups * (delta / width) ** 2)))


def win_relevance(z_diff: np.ndarray, human_pref, *, features=None,
                  group_ids=None) -> pd.DataFrame:
    """Per-feature human-win relevance.

    z_diff: (N, M) contrast codes (z>0 = A expresses the concept more).
    human_pref: (N,) y = P(A preferred) in {0.0, 0.5, 1.0}.
    group_ids: optional prompt/group ids. If any id repeats, descriptive effects
    weight each group equally and sign inference treats groups as the independent
    units rather than treating repeated rows as independent battles.
    """
    z = np.asarray(z_diff, dtype=np.float32)
    y = np.asarray(human_pref, dtype=float)          # P(A preferred)
    if z.ndim != 2:
        raise ValueError("z_diff must be a 2D array")
    if y.ndim != 1 or len(y) != len(z):
        raise ValueError("z_diff and human_pref must have the same rows")
    groups_all = _validated_group_codes(group_ids, len(z))
    valid = np.isfinite(y)
    z, y = z[valid], y[valid]
    groups = groups_all[valid] if groups_all is not None else None
    grouped = groups is not None and len(np.unique(groups)) < len(groups)
    yc = 2.0 * y - 1.0                                # +1 A, -1 B, 0 tie
    n, m = z.shape
    n_groups = int(len(np.unique(groups))) if groups is not None else int(n)
    feats = list(range(m)) if features is None else list(features)
    columns = ["feature_id", "n_fire", "fire_rate", "win_rate_a_more",
               "win_rate_a_less", "win_assoc", "n_decisive_fire",
               "preferred_side_rate", "preferred_minus_rejected_mean",
               "preference_sign_p", "preference_sign_p_bonferroni",
               "preference_sign_significant", "correlation", "p_value",
               "p_bonferroni", "sign", "significant", "n_groups",
               "n_independent_groups", "n_fire_groups", "n_decisive_fire_groups",
               "n_correlation_groups", "estimand", "preference_sign_test",
               "correlation_test", "tie_policy"]
    if not feats:
        return pd.DataFrame(columns=columns)

    rows = []
    decisive = y != 0.5
    winner_sign = np.sign(y[decisive] - 0.5)
    decisive_groups = groups[decisive] if groups is not None else None
    for f in feats:
        col = z[:, f]
        fire = col != 0
        more, less = col > 0, col < 0
        oriented = col[decisive] * winner_sign
        oriented_fire = oriented != 0
        n_oriented = int(oriented_fire.sum())
        n_preferred = int((oriented[oriented_fire] > 0).sum())

        if grouped:
            group_values = np.unique(groups)
            fire_rate = float(np.mean([
                fire[groups == group].mean() for group in group_values
            ]))
            more_means = [
                float(y[(groups == group) & more].mean())
                for group in group_values if bool(((groups == group) & more).any())
            ]
            less_means = [
                float(y[(groups == group) & less].mean())
                for group in group_values if bool(((groups == group) & less).any())
            ]
            a_more = float(np.mean(more_means)) if more_means else float("nan")
            a_less = float(np.mean(less_means)) if less_means else float("nan")

            decisive_group_values = np.unique(decisive_groups)
            preferred_by_group = []
            oriented_mean_by_group = []
            for group in decisive_group_values:
                in_group = decisive_groups == group
                group_oriented = oriented[in_group]
                group_fire = group_oriented != 0
                if bool(group_fire.any()):
                    preferred_by_group.append(float((group_oriented[group_fire] > 0).mean()))
                oriented_mean_by_group.append(float(group_oriented.mean()))
            preferred_rate = (float(np.mean(preferred_by_group))
                              if preferred_by_group else float("nan"))
            oriented_mean = (float(np.mean(oriented_mean_by_group))
                             if oriented_mean_by_group else float("nan"))
            n_decisive_fire_groups = len(preferred_by_group)
            sign_p = _bounded_mean_p(
                preferred_rate, 0.5, n_decisive_fire_groups, 1.0)

            group_sign_mean = []
            group_outcome_mean = []
            for group in group_values:
                use = (groups == group) & fire
                if bool(use.any()):
                    group_sign_mean.append(float(np.sign(col[use]).mean()))
                    group_outcome_mean.append(float(yc[use].mean()))
            n_fire_groups = len(group_sign_mean)
            sign_values = np.asarray(group_sign_mean)
            outcome_values = np.asarray(group_outcome_mean)
            sign_threshold = (
                (float(sign_values.min()) + float(sign_values.max())) / 2.0
                if n_fire_groups else float("nan"))
            outcome_threshold = (
                (float(outcome_values.min()) + float(outcome_values.max())) / 2.0
                if n_fire_groups else float("nan"))
            sign_high = int((sign_values > sign_threshold).sum())
            outcome_high = int((outcome_values > outcome_threshold).sum())
            if (n_fire_groups > 1 and np.ptp(sign_values) > 0
                    and np.ptp(outcome_values) > 0):
                corr = float(pearsonr(sign_values, outcome_values).statistic)
                n_correlation_groups = n_fire_groups
                if (
                    n_fire_groups >= 10
                    and min(
                        sign_high, n_fire_groups - sign_high,
                        outcome_high, n_fire_groups - outcome_high,
                    ) >= 5
                ):
                    sign_is_high = sign_values > sign_threshold
                    outcome_is_high = outcome_values > outcome_threshold
                    contingency = np.array([
                        [np.sum(sign_is_high & outcome_is_high),
                         np.sum(sign_is_high & ~outcome_is_high)],
                        [np.sum(~sign_is_high & outcome_is_high),
                         np.sum(~sign_is_high & ~outcome_is_high)],
                    ])
                    pval = float(fisher_exact(contingency).pvalue)
                else:
                    pval = float("nan")
            else:
                corr, pval = float("nan"), float("nan")
                n_correlation_groups = 0
            estimand = "equal_group_weight"
            sign_test = "two_sided_hoeffding_bounded_group_mean"
            corr_test = "fisher_exact_range_midpoint_split_across_group_means"
        else:
            a_more = float(y[more].mean()) if more.any() else float("nan")
            a_less = float(y[less].mean()) if less.any() else float("nan")
            fire_rate = float(fire.mean()) if n else float("nan")
            preferred_rate = (
                float(n_preferred / n_oriented) if n_oriented else float("nan"))
            oriented_mean = (
                float(oriented.mean()) if len(oriented) else float("nan"))
            sign_p = (
                float(binomtest(n_preferred, n_oriented, p=0.5).pvalue)
                if n_oriented else float("nan"))
            sign_values = np.sign(col[fire]).astype(float)
            outcome_values = yc[fire].astype(float)
            n_fire_rows = int(fire.sum())
            if n_fire_rows:
                sign_threshold = (
                    float(sign_values.min()) + float(sign_values.max())) / 2.0
                outcome_threshold = (
                    float(outcome_values.min()) + float(outcome_values.max())) / 2.0
                sign_is_high = sign_values > sign_threshold
                outcome_is_high = outcome_values > outcome_threshold
                arm_support = min(
                    int(sign_is_high.sum()), int((~sign_is_high).sum()),
                    int(outcome_is_high.sum()), int((~outcome_is_high).sum()))
            else:
                arm_support = 0
            if n_fire_rows > 1 and np.ptp(sign_values) > 0                     and np.ptp(outcome_values) > 0:
                corr = float(pearsonr(sign_values, outcome_values).statistic)
                n_correlation_groups = n_fire_rows
                if n_fire_rows >= 10 and arm_support >= 5:
                    contingency = np.array([
                        [np.sum(sign_is_high & outcome_is_high),
                         np.sum(sign_is_high & ~outcome_is_high)],
                        [np.sum(~sign_is_high & outcome_is_high),
                         np.sum(~sign_is_high & ~outcome_is_high)],
                    ])
                    pval = float(fisher_exact(contingency).pvalue)
                else:
                    pval = float("nan")
            else:
                corr, pval = float("nan"), float("nan")
                n_correlation_groups = 0
            n_fire_groups = int(fire.sum())
            n_decisive_fire_groups = n_oriented
            estimand = "battle_weighted"
            sign_test = "two_sided_exact_binomial"
            corr_test = "fisher_exact_range_midpoint_split_across_rows"

        rows.append({
            "feature_id": int(f), "n_fire": int(fire.sum()),
            "fire_rate": fire_rate,
            "win_rate_a_more": a_more, "win_rate_a_less": a_less,
            "win_assoc": a_more - a_less,
            "n_decisive_fire": n_oriented,
            "preferred_side_rate": preferred_rate,
            "preferred_minus_rejected_mean": oriented_mean,
            "preference_sign_p": sign_p,
            "correlation": corr, "p_value": pval,
            "n_groups": n_groups,
            "n_independent_groups": n_groups,
            "n_fire_groups": n_fire_groups,
            "n_decisive_fire_groups": n_decisive_fire_groups,
            "n_correlation_groups": n_correlation_groups,
            "estimand": estimand,
            "preference_sign_test": sign_test,
            "correlation_test": corr_test,
            "tie_policy": "retained_as_0.5_neutral",
        })
    df = pd.DataFrame(rows)
    df["preference_sign_p_bonferroni"] = (
        df["preference_sign_p"] * len(feats)).clip(upper=1.0)
    df["preference_sign_significant"] = (
        df["preference_sign_p_bonferroni"] < 0.05)
    df["p_bonferroni"] = (df["p_value"] * len(feats)).clip(upper=1.0)
    df["sign"] = np.sign(df["correlation"]).astype("Int64")
    df["significant"] = df["p_bonferroni"] < 0.05
    return df


def _standardize(x: np.ndarray) -> np.ndarray:
    s = x.std()
    return (x - x.mean()) / s if s > 0 else np.zeros_like(x)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _weighted_standardize(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    total = float(weights.sum())
    mean = float(np.sum(weights * x) / total)
    variance = float(np.sum(weights * (x - mean) ** 2) / total)
    return (x - mean) / np.sqrt(variance) if variance > 0 else np.zeros_like(x)


def win_relevance_logistic(z_diff: np.ndarray, human_pref, length, *,
                           features=None, group_ids=None) -> pd.DataFrame:
    """WIMHF length-controlled per-feature logistic average marginal effects.

    With repeated ``group_ids``, the point estimand gives every group equal total
    weight. Inference uses a group-clustered sandwich Wald test, so repeated rows
    from one prompt do not masquerade as independent evidence. Without repeated
    groups, the historical battle-weighted MLE and likelihood-ratio test are kept.
    """
    import warnings

    from scipy.stats import chi2
    from sklearn.linear_model import LogisticRegression

    z = np.asarray(z_diff, dtype=np.float64)
    y_all = np.asarray(human_pref, dtype=float)
    length = np.asarray(length, dtype=float)
    if z.ndim != 2:
        raise ValueError("z_diff must be a 2D array")
    if y_all.ndim != 1 or length.ndim != 1 \
            or len(y_all) != len(z) or len(length) != len(z):
        raise ValueError("z_diff, human_pref, and length must have the same rows")
    groups_all = _validated_group_codes(group_ids, len(z))
    dec = np.isfinite(y_all) & np.isfinite(length) & (y_all != 0.5)
    y = (y_all[dec] > 0.5).astype(int)
    groups = groups_all[dec] if groups_all is not None else None
    grouped = groups is not None and len(np.unique(groups)) < len(groups)
    n, m = z.shape
    feats = list(range(m)) if features is None else [int(f) for f in features]
    n_independent = (int(len(np.unique(groups))) if groups is not None
                     else int(dec.sum()))
    estimand = "equal_group_weight" if grouped else "battle_weighted"
    inference_test = ("cluster_robust_wald_t_g_minus_1_hc1" if grouped
                      else "likelihood_ratio_chi_square_1df")
    outcome_high_groups = outcome_low_groups = 0
    columns = ["feature_id", "beta", "delta_win_rate", "lr_p", "separable",
               "delta_win_p_bonferroni", "delta_win_significant", "n_groups",
               "n_independent_groups", "feature_low_groups", "feature_high_groups",
               "outcome_low_groups", "outcome_high_groups", "inference_supported",
               "estimand", "inference_test", "tie_policy"]
    if not feats:
        return pd.DataFrame(columns=columns)

    def _nan_row(f):
        return {"feature_id": f, "beta": float("nan"),
                "delta_win_rate": float("nan"), "lr_p": float("nan"),
                "separable": False, "n_groups": n_independent,
                "n_independent_groups": n_independent,
                "feature_low_groups": float("nan"),
                "feature_high_groups": float("nan"),
                "outcome_low_groups": outcome_low_groups if grouped else float("nan"),
                "outcome_high_groups": outcome_high_groups if grouped else float("nan"),
                "inference_supported": False,
                "estimand": estimand, "inference_test": inference_test}

    if dec.sum() < 2 or len(set(y.tolist())) < 2:
        df = pd.DataFrame([_nan_row(f) for f in feats])
    else:
        z_dec = z[dec]
        if grouped:
            _, group_index = np.unique(groups, return_inverse=True)
            counts = np.bincount(group_index)
            # The constant factor makes weights sum to N; it changes neither the MLE
            # nor the equal-group AME, but keeps optimizer tolerances on their usual scale.
            sample_weight = len(y) / (len(counts) * counts[group_index])
            len_std = _weighted_standardize(length[dec], sample_weight)
        else:
            group_index = np.arange(len(y), dtype=int)
            sample_weight = None
            len_std = _standardize(length[dec])
        len_col = len_std.reshape(-1, 1)
        if grouped:
            group_y_mean = np.bincount(
                group_index, weights=y, minlength=n_independent) / counts
            outcome_threshold = (
                float(group_y_mean.min()) + float(group_y_mean.max())) / 2.0
            outcome_high_groups = int((group_y_mean > outcome_threshold).sum())
            outcome_low_groups = n_independent - outcome_high_groups
        else:
            outcome_high_groups = outcome_low_groups = 0

        _MAXIT = 2000

        def _fit_mle(X):
            with np.errstate(all="ignore"):
                model = LogisticRegression(C=np.inf, max_iter=_MAXIT)
                return model.fit(X, y, sample_weight=sample_weight)

        def _fit_pen(X):
            model = LogisticRegression(C=1.0, max_iter=500)
            return model.fit(X, y, sample_weight=sample_weight)

        _SEP = 10.0

        def _ll(model, X):
            with np.errstate(all="ignore"):
                probability = np.clip(
                    model.predict_proba(X)[:, 1], 1e-12, 1 - 1e-12)
            contribution = y * np.log(probability) + (1 - y) * np.log(1 - probability)
            if sample_weight is not None:
                contribution = sample_weight * contribution
            return float(np.sum(contribution))

        if not grouped:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                red = _fit_mle(len_col)
                ll_red = _ll(red, len_col)
        else:
            ll_red = float("nan")

        def _ame(a, b_z, g):
            effect = (_sigmoid(a + b_z + g * len_std)
                      - _sigmoid(a + g * len_std))
            if sample_weight is None:
                return float(effect.mean())
            return float(np.average(effect, weights=sample_weight))

        def _cluster_wald_p(model, X) -> float:
            """HC1 cluster-sandwich Wald p-value for the standardized feature."""
            if (
                n_independent < 10
                or min(outcome_high_groups, outcome_low_groups) < 5
            ):
                return float("nan")
            design = np.column_stack([np.ones(len(X)), X])
            probability = model.predict_proba(X)[:, 1]
            weights = np.asarray(sample_weight, dtype=float)
            hessian = design.T @ (
                (weights * probability * (1.0 - probability))[:, None] * design)
            bread = np.linalg.pinv(hessian)
            row_scores = design * (weights * (y - probability))[:, None]
            cluster_scores = np.zeros((n_independent, design.shape[1]), dtype=float)
            np.add.at(cluster_scores, group_index, row_scores)
            covariance = bread @ (cluster_scores.T @ cluster_scores) @ bread
            k_params = design.shape[1]
            if n_independent > 1 and len(y) > k_params:
                covariance *= (n_independent / (n_independent - 1.0)
                               * (len(y) - 1.0) / (len(y) - k_params))
            variance = float(covariance[1, 1])
            if not np.isfinite(variance) or variance <= 0:
                return float("nan")
            z_score = float(model.coef_[0, 0]) / np.sqrt(variance)
            # A t reference with G-1 df is safer than normal asymptotics when the
            # number of independent prompt groups is modest.
            return float(2.0 * student_t.sf(abs(z_score), df=n_independent - 1))

        def _one(f):
            zf = z_dec[:, f]
            z_std = (_weighted_standardize(zf, sample_weight) if grouped
                     else _standardize(zf))
            if not bool(np.any(z_std != 0)):
                return _nan_row(f)
            if grouped:
                group_z_mean = np.bincount(
                    group_index, weights=z_std, minlength=n_independent) / counts
                feature_threshold = (
                    float(group_z_mean.min()) + float(group_z_mean.max())) / 2.0
                feature_high_groups = int(
                    (group_z_mean > feature_threshold).sum())
                feature_low_groups = n_independent - feature_high_groups
                inference_supported = bool(
                    n_independent >= 10
                    and min(
                        feature_low_groups, feature_high_groups,
                        outcome_low_groups, outcome_high_groups,
                    ) >= 5)
            else:
                feature_low_groups = feature_high_groups = float("nan")
                inference_supported = True
            X = np.column_stack([z_std, len_std])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                full = _fit_mle(X)
            a = float(full.intercept_[0])
            b_z, g = map(float, full.coef_[0])
            if abs(b_z) > _SEP:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pen = _fit_pen(X)
                a = float(pen.intercept_[0])
                b_z, g = map(float, pen.coef_[0])
                return {"feature_id": f, "beta": b_z,
                        "delta_win_rate": _ame(a, b_z, g),
                        "lr_p": float("nan"), "separable": True,
                        "n_groups": n_independent,
                        "n_independent_groups": n_independent,
                        "feature_low_groups": feature_low_groups,
                        "feature_high_groups": feature_high_groups,
                        "outcome_low_groups": (
                            outcome_low_groups if grouped else float("nan")),
                        "outcome_high_groups": (
                            outcome_high_groups if grouped else float("nan")),
                        "inference_supported": False,
                        "estimand": estimand, "inference_test": inference_test}
            if grouped:
                p_value = (
                    _cluster_wald_p(full, X)
                    if inference_supported else float("nan"))
            else:
                stat = max(0.0, 2.0 * (_ll(full, X) - ll_red))
                p_value = float(chi2.sf(stat, 1))
            return {"feature_id": f, "beta": b_z,
                    "delta_win_rate": _ame(a, b_z, g),
                    "lr_p": p_value, "separable": False,
                    "n_groups": n_independent,
                    "n_independent_groups": n_independent,
                    "feature_low_groups": feature_low_groups,
                    "feature_high_groups": feature_high_groups,
                    "outcome_low_groups": (
                        outcome_low_groups if grouped else float("nan")),
                    "outcome_high_groups": (
                        outcome_high_groups if grouped else float("nan")),
                    "inference_supported": bool(inference_supported),
                    "estimand": estimand, "inference_test": inference_test}

        if len(feats) > 256:
            import os
            from joblib import Parallel, delayed
            njobs = min(8, os.cpu_count() or 1)
            print(f"  win-relevance: fitting {len(feats)} per-feature logistic models "
                  f"(n_jobs={njobs})…", flush=True)
            rows = Parallel(n_jobs=njobs, prefer="threads")(
                delayed(_one)(f) for f in feats)
        else:
            rows = [_one(f) for f in feats]
        df = pd.DataFrame(rows)

    df["delta_win_p_bonferroni"] = (df["lr_p"] * len(feats)).clip(upper=1.0)
    df["delta_win_significant"] = df["delta_win_p_bonferroni"] < 0.05
    df["tie_policy"] = "dropped_from_binary_logistic"
    return df


def cluster_win_relevance(z_diff: np.ndarray, human_pref, length, clusters: pd.DataFrame, *,
                          aggregate: str = "mean", group_ids=None) -> pd.DataFrame:
    """Anatomy-style cluster-level win-relevance.

    Aggregate each behavior cluster's member features into ONE signed activation per
    battle (the cluster's net chosen-vs-rejected contrast), then run the *same*
    length-controlled logistic (``win_relevance_logistic``) on the cluster-score matrix.
    The unit of analysis (cluster vs feature) is just a data transform — the statistics
    are single-sourced, so length-control / LR-test / Bonferroni are identical to the
    per-feature path (Bonferroni now scales with #clusters, not #features).

    ``clusters``: DataFrame with ``feature_id``, ``cluster_id`` (+ optional ``behavior``) —
    e.g. ``feature_clusters.csv`` from ``cluster-features --fidelity-only`` (members are the
    verified features). ``aggregate``: ``mean`` (default, signed) or ``sum``. A cluster whose
    members cancel under signed mean is an *incoherent* cluster — informative, not a bug.
    """
    z = np.asarray(z_diff, dtype=np.float64)
    cl = clusters.dropna(subset=["cluster_id"]).copy()
    cl["cluster_id"] = cl["cluster_id"].astype(int)
    cl["feature_id"] = cl["feature_id"].astype(int)

    members = {int(c): [f for f in g["feature_id"].tolist() if 0 <= f < z.shape[1]]
               for c, g in cl.groupby("cluster_id")}
    cids = [c for c in sorted(members) if members[c]]
    if not cids:
        return pd.DataFrame(columns=["cluster_id", "n_features", "beta", "delta_win_rate",
                                     "lr_p", "delta_win_p_bonferroni", "delta_win_significant"])

    agg = np.mean if aggregate == "mean" else np.sum
    Zc = np.column_stack([agg(z[:, members[c]], axis=1) for c in cids])

    dwr = win_relevance_logistic(
        Zc, human_pref, length, features=list(range(len(cids))), group_ids=group_ids)
    dwr["cluster_id"] = dwr["feature_id"].map(lambda i: cids[int(i)])
    dwr["n_features"] = dwr["cluster_id"].map(lambda c: len(members[c]))
    dwr = dwr.drop(columns=["feature_id"])

    beh = None
    if "behavior" in cl.columns:
        beh = cl.dropna(subset=["behavior"]).groupby("cluster_id")["behavior"].first()
        dwr["behavior"] = dwr["cluster_id"].map(beh)

    lead = ["cluster_id"] + (["behavior"] if beh is not None else []) + ["n_features"]
    cols = lead + [c for c in dwr.columns if c not in lead]
    return (dwr[cols]
            .reindex(dwr["delta_win_rate"].abs().sort_values(ascending=False).index)
            .reset_index(drop=True))


def conditional_win_relevance(z_diff: np.ndarray, human_pref, length, prompt_concept, *,
                              prompt_region_ids=None, features=None, group_ids=None,
                              min_battles: int = 300, min_fire: int = 20) -> pd.DataFrame:
    """Conditional (prompt-type × behavior) win-rate — the interaction δ_{f,k}.

    For each prompt type k, the per-feature length-controlled Δwin-rate **among battles
    of that type** — i.e. how much behavior f wins (+) or loses (−) when the prompt is
    type k. This makes the "criterion is conditional" claim statistical: a feature can
    have δ_{f,0} > 0 (detail wins for guidance prompts) and δ_{f,1} < 0 (detail loses for
    clarification prompts). Reuses ``win_relevance_logistic`` per prompt-type subset, so
    length-control / LR-test are identical; Bonferroni is applied over all (f, k) cells.

    ``z_diff`` is the **unoriented** difference code (the logistic learns the sign from
    ``human_pref``). ``prompt_concept`` may be either legacy categorical ids ``(N,)``
    or overlapping presence ``(N, K)``; pass ``prompt_region_ids`` for the latter.
    """
    pc = np.asarray(prompt_concept)
    y = np.asarray(human_pref, dtype=float)
    length = np.asarray(length, dtype=float)
    groups = _validated_group_codes(group_ids, len(y))
    support_groups = groups if groups is not None else np.arange(len(y), dtype=object)
    if pc.ndim == 1:
        region_masks = [
            (int(k), pc == k) for k in sorted({int(x) for x in pc if x >= 0})]
    elif pc.ndim == 2:
        if pc.shape[0] != len(y):
            raise ValueError("prompt membership rows must match human_pref")
        ids = (np.arange(pc.shape[1], dtype=int) if prompt_region_ids is None
               else np.asarray(prompt_region_ids))
        if len(ids) != pc.shape[1]:
            raise ValueError("prompt_region_ids length must match membership columns")
        region_masks = [
            (int(region_id), np.asarray(pc[:, j], dtype=bool))
            for j, region_id in enumerate(ids)]
    else:
        raise ValueError("prompt_concept must be categorical (N,) or membership (N,K)")
    if groups is not None:
        for group in np.unique(groups):
            rows = groups == group
            for _, region_mask in region_masks:
                values = region_mask[rows]
                if not np.all(values == values[0]):
                    raise ValueError(
                        "prompt membership must be constant within each group")
    parts = []
    fit_eligible = np.isfinite(y) & np.isfinite(length) & (y != 0.5)
    for k, mask in region_masks:
        fit_mask = mask & fit_eligible
        if int(fit_mask.sum()) < min_battles:
            continue
        region_groups = support_groups[fit_mask]
        sub = win_relevance_logistic(
            z_diff[fit_mask], y[fit_mask], length[fit_mask], features=features,
            group_ids=region_groups)
        keep_columns = [
            "feature_id", "beta", "delta_win_rate", "lr_p", "estimand",
            "inference_test", "n_independent_groups", "feature_low_groups",
            "feature_high_groups", "outcome_low_groups", "outcome_high_groups",
            "inference_supported",
        ]
        sub = sub[[column for column in keep_columns if column in sub.columns]].copy()
        sub.insert(0, "prompt_concept", k)
        sub["n_battles"] = int(mask.sum())
        sub["n_fit_battles"] = int(fit_mask.sum())
        sub["n_prompt_groups"] = int(len(set(region_groups.tolist())))
        # Report support on the exact decisive, finite rows used by the fit.
        zk = z_diff[fit_mask]
        sub["n_fire"] = [int((zk[:, int(f)] != 0).sum()) for f in sub["feature_id"]]
        sub["n_fire_groups"] = [
            int(len(set(region_groups[zk[:, int(f)] != 0].tolist())))
            for f in sub["feature_id"]
        ]
        parts.append(sub)
    if not parts:
        return pd.DataFrame(columns=[
            "prompt_concept", "feature_id", "beta", "delta_win_rate", "lr_p",
            "estimand", "inference_test", "n_independent_groups", "n_battles",
            "n_prompt_groups", "n_fit_battles", "n_fire", "n_fire_groups", "cond_p_bonferroni",
            "cond_significant",
        ])
    out = pd.concat(parts, ignore_index=True)
    out["cond_p_bonferroni"] = (out["lr_p"] * len(out)).clip(upper=1.0)   # over all (f,k) cells
    # gate significance on feature-specific support too: a δ_{f,k} cell needs the feature to
    # actually fire in >= min_fire battles OF THAT TYPE, not just clear the Bonferroni p on
    # thin support (n_battles overstates the effective sample) (#5).
    out["cond_significant"] = (
        (out["cond_p_bonferroni"] < 0.05) & (out["n_fire_groups"] >= min_fire)
    )
    return out.reindex(out["delta_win_rate"].abs().sort_values(ascending=False).index
                       ).reset_index(drop=True)
