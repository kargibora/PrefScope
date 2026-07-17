"""Predictive validation: does the diagnosed deficit predict actual win rate?

This is the end-to-end check that the lens's per-model diagnosis means something:
a model that under-expresses the features humans reward should actually lose more.

For each model m we form a *predicted advantage*

    s_hat(m) = sum_f net_direction_f(m) * w_f

where ``net_direction_f(m) = P(z_f>0 | m) - P(z_f<0 | m)`` is how much m
over/under-expresses feature f (from the oriented-code bank), and ``w_f`` is the
human-reward weight of feature f. By default ``w_f`` is the *length-controlled*
average marginal effect ``delta_win_rate`` (from ``win_relevance_logistic``): the
predicted change in win rate when the A-side expresses f more, holding the
word-count gap fixed. Arena has a strong length bias (longer answers win), so the
raw ``win_assoc`` partly reflects verbosity; weighting by ``delta_win_rate`` makes
the prediction length-controlled. We correlate ``s_hat(m)`` with the model's
actual win rate ``a(m)`` across models and report R^2 and Spearman rho.

This is an **exploratory, associational** check across a *small* number of models
(n ~ 10), not a high-powered estimate. Read it through the bootstrap CI (resampling
the model rows) and the permutation p-value (shuffling actual win rate across
models): a single point estimate of R^2 over ~10 points is fragile, so the CI and
the null tail are how to judge whether the association is real.

``loo=True`` recomputes the weights for each model from battles *not involving it*
(using the bank's natural ``orientation == "a"`` rows, which reproduce the lens
``z_diff`` with ``win`` = P(A preferred)), so the prediction is honestly held out.
When a per-battle ``length`` is available in the bank meta, the LOO refit is also
length-controlled (logistic AME); otherwise it falls back to the raw association
and ``loo_length_controlled`` is set False.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from prefscope.pipeline.winrelevance import win_relevance, win_relevance_logistic

_N_BOOT = 2000


def _net_direction(z: np.ndarray) -> np.ndarray:
    """Per-feature net_direction over a set of oriented rows."""
    return (z > 0).mean(axis=0) - (z < 0).mean(axis=0)


def _weights_from_winrel(wr: pd.DataFrame, feats, weight_col, significant_only) -> np.ndarray:
    if weight_col not in wr.columns:
        raise ValueError(
            f"weight_col {weight_col!r} not in win-relevance columns {list(wr.columns)}; "
            f"pass a CSV carrying it (the `win-relevance` CLI emits both `win_assoc` and "
            f"`delta_win_rate`) or set weight_col explicitly")
    sub = wr.set_index("feature_id")
    if significant_only:
        # length-controlled weights carry their own significance flag; raw win_assoc
        # uses the correlation-based `significant`.
        sig_col = "delta_win_significant" if weight_col == "delta_win_rate" else "significant"
        if sig_col not in sub.columns and "significant" in sub.columns:
            sig_col = "significant"
        if sig_col in sub.columns:
            sub = sub.where(sub[sig_col].astype(bool), other=np.nan)
    w = sub.reindex(feats)[weight_col].to_numpy(dtype=float)
    return np.nan_to_num(w, nan=0.0)


def validate_diagnosis(bank_Z: np.ndarray, bank_meta: pd.DataFrame,
                       win_relevance_df: pd.DataFrame, *,
                       weight_col: str = "delta_win_rate", significant_only: bool = True,
                       min_battles: int = 20, loo: bool = False, seed: int = 0):
    """Correlate predicted deficit score with actual win rate across models.

    By default features are weighted by the length-controlled ``delta_win_rate``
    (the logistic average marginal effect). Raises ``ValueError`` if ``weight_col``
    is absent from ``win_relevance_df``.

    Returns ``(per_model_df, summary)``. ``per_model_df`` has one row per model
    (``model``, ``n_battles``, ``predicted_score``, ``actual_win_rate`` and, when
    ``loo``, ``predicted_score_loo``). ``summary`` holds the in-sample R^2 /
    Spearman, their bootstrap CIs and a permutation p-value and, when ``loo``, the
    leave-one-model-out counterparts plus ``loo_length_controlled``.

    This is exploratory across a small n (~10 models); read the CI and permutation
    p, not the bare point estimate.

    NOTE (#7): ``loo`` is **leave-one-model-out feature WEIGHTING**, not fully held-out
    prediction. Only the reward weights are refit excluding the target; the target's
    concept profile and ``actual_win_rate`` still come from its own battles, and models
    are coupled (one model's win is another's loss). ``summary["loo_semantics"]`` records
    this. Treat ``loo_*`` as a weight-overfitting check, not out-of-sample generalization.
    """
    bank_Z = np.asarray(bank_Z, dtype=np.float32)
    feats = win_relevance_df["feature_id"].astype(int).tolist()
    cols = bank_Z[:, feats]                                   # (2N, F)
    w = _weights_from_winrel(win_relevance_df, feats, weight_col, significant_only)

    self_model = bank_meta["self_model"].to_numpy()
    win = bank_meta["win"].to_numpy(dtype=float)

    # per-battle length (word-count gap), oriented to match each row's self-minus-other
    # codes; absent on legacy banks -> raw (uncontrolled) LOO refit. The bank emits an
    # all-zero length SENTINEL when completion text is unavailable, so require a
    # NON-DEGENERATE column (std > 0) — otherwise we'd report loo_length_controlled=True
    # while "controlling" for a constant, which is no control at all (#4).
    _len_col = bank_meta["length"].to_numpy(dtype=float) if "length" in bank_meta.columns else None
    # require SOME finite value before np.nanstd (all-NaN / empty would warn), then non-zero
    # spread — the all-zero sentinel and a constant both fail this, so they can't be "controlled".
    has_length = (_len_col is not None and _len_col.size > 0
                  and bool(np.isfinite(_len_col).any())
                  and float(np.nanstd(_len_col)) > 0.0)
    length = _len_col if has_length else None

    # rows reproducing the natural lens z_diff (A-as-self) — used for honest LOO
    a_mask = (bank_meta["orientation"] == "a").to_numpy()
    z_ab = cols[a_mask]
    y_ab = win[a_mask]                                        # P(A preferred)
    len_ab = length[a_mask] if has_length else None
    sm_a = self_model[a_mask]                                 # model_a
    om_a = bank_meta.loc[a_mask, "other_model"].to_numpy()    # model_b

    counts = pd.Series(self_model).value_counts()
    models = [m for m in counts.index if counts[m] >= min_battles]

    rows = []
    for m in models:
        mask = self_model == m
        nd = _net_direction(cols[mask])
        row = {"model": m, "n_battles": int(mask.sum()),
               "predicted_score": float(nd @ w),
               "actual_win_rate": float(win[mask].mean())}
        if loo:
            keep = (sm_a != m) & (om_a != m)
            if has_length:
                wr_loo = win_relevance_logistic(z_ab[keep], y_ab[keep], len_ab[keep],
                                                features=list(range(len(feats))))
                w_loo = _weights_from_winrel(
                    wr_loo.assign(feature_id=feats), feats, "delta_win_rate",
                    significant_only)
            else:
                # raw fallback: win_relevance has no delta_win_rate, so weight by
                # win_assoc (the raw association) regardless of the in-sample weight_col.
                wr_loo = win_relevance(z_ab[keep], y_ab[keep],
                                       features=list(range(len(feats))))
                loo_col = weight_col if weight_col in wr_loo.columns else "win_assoc"
                w_loo = _weights_from_winrel(
                    wr_loo.assign(feature_id=feats), feats, loo_col, significant_only)
            row["predicted_score_loo"] = float(nd @ w_loo)
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("predicted_score", ascending=False).reset_index(drop=True)
    summary = {"n_models": int(len(df)), "weight_col": weight_col,
               "significant_only": bool(significant_only), "min_battles": int(min_battles)}
    rng = np.random.default_rng(seed)
    summary.update(_fit(df["predicted_score"], df["actual_win_rate"],
                        prefix="insample", rng=rng))
    if loo:
        summary["loo_length_controlled"] = bool(has_length)
        # LOO here = weights held out per model only; profile + actual_win_rate come from
        # the target's own battles. Name it honestly so consumers don't render it as
        # out-of-sample prediction (#7).
        summary["loo_semantics"] = "feature_weighting"
        summary.update(_fit(df["predicted_score_loo"], df["actual_win_rate"],
                            prefix="loo", rng=rng))
    return df, summary


def _fit(pred, actual, *, prefix: str, rng: np.random.Generator | None = None) -> dict:
    pred = np.asarray(pred, dtype=float)
    actual = np.asarray(actual, dtype=float)
    ok = np.isfinite(pred) & np.isfinite(actual)

    def _nan():
        keys = ["r", "r2", "spearman", "p", "r2_ci_lo", "r2_ci_hi",
                "spearman_ci_lo", "spearman_ci_hi", "r2_perm_p"]
        return {f"{prefix}_{k}": float("nan") for k in keys}

    if ok.sum() < 3 or np.var(pred[ok]) == 0 or np.var(actual[ok]) == 0:
        return _nan()
    p_ok = pred[ok]
    a_ok = actual[ok]
    r, p = pearsonr(p_ok, a_ok)
    rho, _ = spearmanr(p_ok, a_ok)
    out = {f"{prefix}_r": float(r), f"{prefix}_r2": float(r * r),
           f"{prefix}_spearman": float(rho), f"{prefix}_p": float(p)}

    if rng is None:
        rng = np.random.default_rng(0)
    n = p_ok.shape[0]
    obs_r2 = float(r * r)

    # bootstrap CI: resample the model rows with replacement
    boot_r2, boot_rho = [], []
    for _ in range(_N_BOOT):
        idx = rng.integers(0, n, size=n)
        pb, ab = p_ok[idx], a_ok[idx]
        if np.var(pb) == 0 or np.var(ab) == 0:
            continue
        rb, _ = pearsonr(pb, ab)
        rhob, _ = spearmanr(pb, ab)
        boot_r2.append(rb * rb)
        if np.isfinite(rhob):
            boot_rho.append(rhob)

    def _ci(vals):
        if len(vals) < 2:
            return float("nan"), float("nan")
        lo, hi = np.percentile(vals, [2.5, 97.5])
        return float(lo), float(hi)

    r2_lo, r2_hi = _ci(boot_r2)
    rho_lo, rho_hi = _ci(boot_rho)

    # permutation null: shuffle actual across models, recompute r^2
    perm_ge = 0
    for _ in range(_N_BOOT):
        ap = rng.permutation(a_ok)
        if np.var(ap) == 0:
            continue
        rp, _ = pearsonr(p_ok, ap)
        if rp * rp >= obs_r2:
            perm_ge += 1
    r2_perm_p = (perm_ge + 1) / (_N_BOOT + 1)

    out.update({
        f"{prefix}_r2_ci_lo": r2_lo, f"{prefix}_r2_ci_hi": r2_hi,
        f"{prefix}_spearman_ci_lo": rho_lo, f"{prefix}_spearman_ci_hi": rho_hi,
        f"{prefix}_r2_perm_p": float(r2_perm_p),
    })
    return out
