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
from scipy.stats import pearsonr


def win_relevance(z_diff: np.ndarray, human_pref, *, features=None) -> pd.DataFrame:
    """Per-feature human-win relevance.

    z_diff: (N, M) contrast codes (z>0 = A expresses the concept more).
    human_pref: (N,) y = P(A preferred) in {0.0, 0.5, 1.0}.
    """
    z = np.asarray(z_diff, dtype=np.float32)
    y = np.asarray(human_pref, dtype=float)          # P(A preferred)
    yc = 2.0 * y - 1.0                                # +1 A, -1 B, 0 tie
    n, m = z.shape
    feats = list(range(m)) if features is None else list(features)
    columns = ["feature_id", "n_fire", "fire_rate", "win_rate_a_more",
               "win_rate_a_less", "win_assoc", "correlation", "p_value",
               "p_bonferroni", "sign", "significant"]
    if not feats:
        return pd.DataFrame(columns=columns)

    rows = []
    for f in feats:
        col = z[:, f]
        fire = col != 0
        more, less = col > 0, col < 0
        a_more = float(y[more].mean()) if more.any() else float("nan")
        a_less = float(y[less].mean()) if less.any() else float("nan")
        # correlate activation sign with human preference over firing battles
        if int(fire.sum()) > 1 and len(set(np.sign(col[fire]).tolist())) > 1 \
                and len(set(yc[fire].tolist())) > 1:
            r, p = pearsonr(np.sign(col[fire]).astype(float), yc[fire])
            corr, pval = float(r), float(p)
        else:
            corr, pval = float("nan"), float("nan")
        rows.append({
            "feature_id": int(f), "n_fire": int(fire.sum()),
            "fire_rate": float(fire.mean()) if n else float("nan"),
            "win_rate_a_more": a_more, "win_rate_a_less": a_less,
            "win_assoc": a_more - a_less,
            "correlation": corr, "p_value": pval,
        })
    df = pd.DataFrame(rows)
    df["p_bonferroni"] = (df["p_value"] * len(feats)).clip(upper=1.0)
    df["sign"] = np.sign(df["correlation"]).astype("Int64")
    df["significant"] = df["p_bonferroni"] < 0.05
    return df


def _standardize(x: np.ndarray) -> np.ndarray:
    s = x.std()
    return (x - x.mean()) / s if s > 0 else np.zeros_like(x)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def win_relevance_logistic(z_diff: np.ndarray, human_pref, length, *,
                           features=None) -> pd.DataFrame:
    """WIMHF Δwin-rate (App. A.2): per-feature length-controlled logistic AME.

    Per feature, fit a univariate logistic ``P(win)=σ(α + β·z_f + γ·ℓΔ)`` over the
    DECISIVE pairs (human_pref != 0.5), with z_f and ℓΔ each standardized. The
    reported ``delta_win_rate`` is the average marginal effect — the mean over pairs
    of ``σ(α + β + γ·ℓΔ_i) − σ(α + γ·ℓΔ_i)`` (predicted win-rate at z_f = +1 sd vs 0).
    Significance is a likelihood-ratio test vs the length-only model (χ²₁), Bonferroni
    over features. Both models are fit by *unpenalized* MLE so the χ² LRT is valid;
    (quasi-)separable features (``separable=True``) get ``lr_p=NaN`` — never a spurious
    tiny p — with a stable penalized point estimate. ``length`` = ℓΔ per battle
    (e.g. word-count difference A−B).
    """
    import warnings

    from scipy.stats import chi2
    from sklearn.linear_model import LogisticRegression

    z = np.asarray(z_diff, dtype=np.float64)
    y_all = np.asarray(human_pref, dtype=float)
    length = np.asarray(length, dtype=float)
    dec = y_all != 0.5
    y = (y_all[dec] > 0.5).astype(int)
    n, m = z.shape
    feats = list(range(m)) if features is None else [int(f) for f in features]
    columns = ["feature_id", "beta", "delta_win_rate", "lr_p", "separable",
               "delta_win_p_bonferroni", "delta_win_significant"]
    if not feats:
        return pd.DataFrame(columns=columns)

    def _nan_row(f):
        return {"feature_id": f, "beta": float("nan"),
                "delta_win_rate": float("nan"), "lr_p": float("nan"),
                "separable": False}

    if dec.sum() < 2 or len(set(y.tolist())) < 2:
        df = pd.DataFrame([_nan_row(f) for f in feats])
    else:
        len_std = _standardize(length[dec])
        len_col = len_std.reshape(-1, 1)
        z_dec = z[dec]

        # Unpenalized MLE (C=inf): the χ² likelihood-ratio test below is only valid for
        # the *maximum-likelihood* fit. L2 shrinkage biases the LR statistic away from its
        # χ²₁ null, so a penalized fit would emit invalid p-values (GPT review). lbfgs takes
        # C=inf for the unpenalized fit (penalty=None is deprecated in sklearn ≥1.8).
        _MAXIT = 2000

        def _fit_mle(X):
            # separable columns make the unpenalized fit's coefficients run away, which
            # trips benign numpy overflow warnings deep in lbfgs — silence them (we detect
            # and handle separation explicitly via the coefficient magnitude below).
            with np.errstate(all="ignore"):
                return LogisticRegression(C=np.inf, max_iter=_MAXIT).fit(X, y)

        # Fallback for (quasi-)separable features, where the MLE diverges and the LRT is
        # undefined: a lightly-penalized fit still gives a *stable* point estimate (we keep
        # reporting beta / Δwin-rate) but we NaN the p-value instead of a bogus tiny one.
        def _fit_pen(X):
            return LogisticRegression(C=1.0, max_iter=500).fit(X, y)

        # |standardized coef| this large ⇒ (near-)perfect separation (odds ratio > e¹⁰ per
        # SD): the unpenalized fit has run away and the LR test is not trustworthy.
        _SEP = 10.0

        def _ll(model, X):
            with np.errstate(all="ignore"):
                p = np.clip(model.predict_proba(X)[:, 1], 1e-12, 1 - 1e-12)
            return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            red = _fit_mle(len_col)         # length-only null — fit once, shared by all
            ll_red = _ll(red, len_col)

        def _ame(a, b_z, g):
            return float(np.mean(_sigmoid(a + b_z + g * len_std)
                                 - _sigmoid(a + g * len_std)))

        def _one(f):
            zf = z_dec[:, f]
            if zf.std() == 0:
                return _nan_row(f)
            X = np.column_stack([_standardize(zf), len_std])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                full = _fit_mle(X)
            a = float(full.intercept_[0]); b_z, g = map(float, full.coef_[0])
            if abs(b_z) > _SEP:             # separable: MLE diverges, LRT undefined
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pen = _fit_pen(X)
                a = float(pen.intercept_[0]); b_z, g = map(float, pen.coef_[0])
                return {"feature_id": f, "beta": b_z, "delta_win_rate": _ame(a, b_z, g),
                        "lr_p": float("nan"), "separable": True}
            stat = max(0.0, 2.0 * (_ll(full, X) - ll_red))
            return {"feature_id": f, "beta": b_z, "delta_win_rate": _ame(a, b_z, g),
                    "lr_p": float(chi2.sf(stat, 1)), "separable": False}

        # one logistic per feature — overcomplete lenses have thousands, so parallelize
        # (and announce it) instead of a silent multi-minute serial loop that looks hung.
        if len(feats) > 256:
            import os
            from joblib import Parallel, delayed
            njobs = min(8, os.cpu_count() or 1)
            print(f"  win-relevance: fitting {len(feats)} per-feature logistic models "
                  f"(n_jobs={njobs})…", flush=True)
            rows = Parallel(n_jobs=njobs, prefer="threads")(delayed(_one)(f) for f in feats)
        else:
            rows = [_one(f) for f in feats]
        df = pd.DataFrame(rows)

    df["delta_win_p_bonferroni"] = (df["lr_p"] * len(feats)).clip(upper=1.0)
    df["delta_win_significant"] = df["delta_win_p_bonferroni"] < 0.05
    return df


def cluster_win_relevance(z_diff: np.ndarray, human_pref, length, clusters: pd.DataFrame, *,
                          aggregate: str = "mean") -> pd.DataFrame:
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

    dwr = win_relevance_logistic(Zc, human_pref, length, features=list(range(len(cids))))
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
                              features=None, min_battles: int = 300,
                              min_fire: int = 20) -> pd.DataFrame:
    """Conditional (prompt-type × behavior) win-rate — the interaction δ_{f,k}.

    For each prompt type k, the per-feature length-controlled Δwin-rate **among battles
    of that type** — i.e. how much behavior f wins (+) or loses (−) when the prompt is
    type k. This makes the "criterion is conditional" claim statistical: a feature can
    have δ_{f,0} > 0 (detail wins for guidance prompts) and δ_{f,1} < 0 (detail loses for
    clarification prompts). Reuses ``win_relevance_logistic`` per prompt-type subset, so
    length-control / LR-test are identical; Bonferroni is applied over all (f, k) cells.

    ``z_diff`` is the **unoriented** difference code (the logistic learns the sign from
    ``human_pref``); ``prompt_concept`` is the per-battle prompt cluster id.
    """
    pc = np.asarray(prompt_concept)
    y = np.asarray(human_pref, dtype=float)
    length = np.asarray(length, dtype=float)
    parts = []
    for k in sorted({int(x) for x in pc if x >= 0}):
        mask = pc == k
        if int(mask.sum()) < min_battles:
            continue
        sub = win_relevance_logistic(z_diff[mask], y[mask], length[mask], features=features)
        sub = sub[["feature_id", "beta", "delta_win_rate", "lr_p"]].copy()
        sub.insert(0, "prompt_concept", k)
        sub["n_battles"] = int(mask.sum())
        # effective per-cell support: battles of this type where the feature actually
        # fires — the honest n for a δ_{f,k} cell (n_battles alone overstates it).
        zk = z_diff[mask]
        sub["n_fire"] = [int((zk[:, int(f)] != 0).sum()) for f in sub["feature_id"]]
        parts.append(sub)
    if not parts:
        return pd.DataFrame(columns=["prompt_concept", "feature_id", "beta", "delta_win_rate",
                                     "lr_p", "n_battles", "n_fire",
                                     "cond_p_bonferroni", "cond_significant"])
    out = pd.concat(parts, ignore_index=True)
    out["cond_p_bonferroni"] = (out["lr_p"] * len(out)).clip(upper=1.0)   # over all (f,k) cells
    # gate significance on feature-specific support too: a δ_{f,k} cell needs the feature to
    # actually fire in >= min_fire battles OF THAT TYPE, not just clear the Bonferroni p on
    # thin support (n_battles overstates the effective sample) (#5).
    out["cond_significant"] = (out["cond_p_bonferroni"] < 0.05) & (out["n_fire"] >= min_fire)
    return out.reindex(out["delta_win_rate"].abs().sort_values(ascending=False).index
                       ).reset_index(drop=True)
