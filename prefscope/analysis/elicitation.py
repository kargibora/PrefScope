"""Prompt-concept ↔ response-concept *co-activation* (conditional co-occurrence lift).

Descriptive complement to the preference analysis: independent of who *wins*, which
response concepts Y co-occur with a prompt concept above their base rate? E.g.
"finance question ↔ descriptive answer", "written in French ↔ response in French",
"translation request ↔ bilingual / quoted source".

The unit of observation is a single response (stack A and B), each paired with the
prompt-concept activations of *its* prompt. A concept "fires" when its SAE code is
positive (top-k codes are non-negative). For each (prompt feature X, response feature Y):

    lift = P(Y fires | X fires) / P(Y fires)

with a 2×2 χ² test of independence (Yates-corrected) and Bonferroni over all *testable*
cells. lift > 1 = Y co-occurs with X above base rate; lift < 1 = below.

CAVEATS (read before claiming anything):
  * lift is SYMMETRIC: lift(X,Y) = P(X,Y)/(P(X)P(Y)) = lift(Y,X). The only thing that
    makes this "prompt → response" is structural — the prompt is observed before the
    response — not the statistic. Do not read it as causal "elicitation"; topic, model
    identity, and length are NOT controlled.
  * Stacking A and B gives 2N rows but only N independent prompts (both responses share
    a prompt). Pass ``n_clusters=N`` so the χ² is scaled by N/(2N) to correct for this
    within-battle dependence (otherwise p-values are anti-conservative, worst for the
    strongest signals). It runs directly on (verified) feature axes — no feature clusters.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def prompt_response_association(prompt_fire: np.ndarray, resp_fire: np.ndarray, *,
                               prompt_features=None, resp_features=None,
                               min_support: int = 30, min_cooccur: int = 5,
                               n_clusters: int | None = None) -> pd.DataFrame:
    """Prompt↔response co-occurrence table via co-activation lift.

    ``prompt_fire`` (R, Mp) and ``resp_fire`` (R, Mc) are boolean/0-1 firing matrices
    over the SAME R rows (R = #responses = 2·#battles when A and B are stacked; each row
    pairs one response with its prompt). ``prompt_features`` / ``resp_features`` optionally
    restrict the tested axes to a subset (e.g. the verified features) — column indices.

    ``n_clusters``: number of INDEPENDENT units (e.g. #battles). When A/B are stacked,
    R = 2·n_clusters but the responses within a battle share a prompt, so the χ² is
    scaled by ``n_clusters / R`` (design-effect correction, conservative at intra-cluster
    correlation 1). Omit for genuinely independent rows.

    Returns one row per *reported* (prompt_feature, completion_feature) cell with support
    ``n_x >= min_support`` and ``n_cooccur >= min_cooccur``, sorted significant-first then
    by |log2 lift|. Bonferroni divides by all *testable* cells (``n_x >= min_support`` and
    ``n_y > 0``), not just the reported ones.
    """
    from scipy.stats import chi2

    # guard the statistic structurally, not just via the default thresholds: a min of 1
    # keeps the 0/0 (n_x=0) and x/0 (n_y=0) cells out of the kept set for any caller.
    min_support = max(1, int(min_support))
    min_cooccur = max(1, int(min_cooccur))

    # contiguous float32 — these are the big (2N, M) arrays, so keep them small;
    # counts stay exact (<2^24) and the χ² products below upcast to float64.
    Pf = np.ascontiguousarray(np.asarray(prompt_fire) > 0, dtype=np.float32)
    Rf = np.ascontiguousarray(np.asarray(resp_fire) > 0, dtype=np.float32)
    R = Pf.shape[0]
    if Rf.shape[0] != R:
        raise ValueError(f"row mismatch: prompt {Pf.shape} vs response {Rf.shape}")

    pcols = list(range(Pf.shape[1])) if prompt_features is None else [int(c) for c in prompt_features]
    rcols = list(range(Rf.shape[1])) if resp_features is None else [int(c) for c in resp_features]
    Pf, Rf = Pf[:, pcols], Rf[:, rcols]

    n_x = Pf.sum(0)                         # (Mp,)  X fires
    n_y = Rf.sum(0)                         # (Mc,)  Y fires
    with np.errstate(all="ignore"):
        cooc = Pf.T @ Rf                    # (Mp, Mc)  X&Y
    p_y = n_y / R                           # base rate of each Y

    with np.errstate(divide="ignore", invalid="ignore"):
        p_y_given_x = cooc / n_x[:, None]
        lift = p_y_given_x / p_y[None, :]

    # vectorized 2×2 χ² with Yates continuity correction. Upcast the small (Mp, Mc)
    # tables to float64 — the products a·d / b·c overflow float32's exact-integer range.
    a = cooc.astype(np.float64)
    b = n_x[:, None].astype(np.float64) - a
    c = n_y[None, :].astype(np.float64) - a
    R64 = float(R)
    d = R64 - a - b - c
    num = R64 * np.clip(np.abs(a * d - b * c) - R64 / 2.0, 0, None) ** 2
    den = (a + b) * (c + d) * (a + c) * (b + d)
    with np.errstate(divide="ignore", invalid="ignore"):
        stat = np.where(den > 0, num / den, 0.0)
    # design-effect correction for stacked A/B (within-battle dependence): χ² scales
    # linearly with N, so multiplying by n_clusters/R rescales to the independent-unit N.
    scale = float(np.clip(n_clusters / R, 0.0, 1.0)) if n_clusters else 1.0
    stat = stat * scale
    pval = chi2.sf(stat, 1)

    # Bonferroni denominator = every cell we COULD test (enough prompt support + Y ever
    # fires), not only the ones that happened to co-occur ≥ min_cooccur (which would bias
    # the correction toward the positive-lift cells we kept).
    testable = (n_x[:, None] >= min_support) & (n_y[None, :] > 0)
    n_tested = int(testable.sum())

    keep = (n_x[:, None] >= min_support) & (cooc >= min_cooccur)
    pi, ci = np.where(keep)
    rows = pd.DataFrame({
        "prompt_feature": np.asarray(pcols)[pi],
        "completion_feature": np.asarray(rcols)[ci],
        "n_x": n_x[pi].astype(int),
        "n_y": n_y[ci].astype(int),
        "n_cooccur": cooc[pi, ci].astype(int),
        "p_y": p_y[ci],
        "p_y_given_x": p_y_given_x[pi, ci],
        "lift": lift[pi, ci],
        "chi2": stat[pi, ci],
        "p_value": pval[pi, ci],
    })
    rows["log2_lift"] = np.log2(rows["lift"].clip(lower=1e-6))
    rows["p_bonferroni"] = (rows["p_value"] * max(1, n_tested)).clip(upper=1.0)
    rows["significant"] = rows["p_bonferroni"] < 0.05
    rows.attrs["n_tested"] = n_tested
    # significant-first, then by |log2 lift| — so a rare, noisy high-lift cell that fails
    # the (corrected) significance test can't outrank a reliable association.
    order = rows.assign(_abs=rows["log2_lift"].abs()).sort_values(
        ["significant", "_abs"], ascending=[False, False]).index
    return rows.reindex(order).reset_index(drop=True)
