"""Conditional concept-omission diagnosis — "what a model fails to do for a prompt type".

For a model M and a prompt concept X, flag a response concept Y when Y is
(1) *characteristic* of X — normally elicited (from ``prompt_response_elicitation``),
(2) *rewarded* within X — positive conditional Δwin (``conditional_win_relevance``), and
(3) *under-produced* by M relative to the models it actually faced on those X-prompts.

The three legs are composed, not re-estimated: (1)+(2) are read from the existing
gate tables and only define the *candidate* (X, Y) cells; this module computes leg (3),
the per-(model, prompt-type, concept) production shortfall, against a **paired
opponent baseline** — within each X-battle M played, M's fire rate of Y vs its
opponent's. Same battle ⇒ same prompt/topic/length context, so the shortfall is not a
marginal-distribution artifact.

This surfaces *defect hypotheses*, not proven defects: the shortfall is correlational
(Y may be substitutable), so a flag says "Y is a rewarded, characteristic response
concept for X that M under-produces", corroborated — never "M would win +x% with Y".
Gates: base-rate floor (can't omit a rare concept), min battles, paired McNemar test
with Bonferroni, split-half sign stability, and a within-model win-when-fired check.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2


def _mcnemar_p(b01: int, b10: int) -> float:
    """Two-sided McNemar p (opponent-fired-not-M vs M-fired-not-opponent), χ² w/ continuity."""
    n = b01 + b10
    if n == 0:
        return 1.0
    # clamp at 0: equal discordant counts (b01 == b10) must give stat 0, not (0-1)**2/n (#3)
    stat = max(0.0, abs(b01 - b10) - 1.0) ** 2 / n
    return float(chi2.sf(stat, 1))


def _wilson_lb(k: int, n: int, z: float = 1.96) -> float:
    """Lower bound of the Wilson score interval for a proportion k/n (0 if n==0)."""
    if n <= 0:
        return 0.0
    p = k / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    margin = (z / d) * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return float(centre - margin)


def _winning_prevalence(z_a, z_b, X, y, by_x, *, z=1.96):
    """Per (X, Y): P(Y fires | X, winner) and P(Y | X, loser) over decisive battles.

    The "expectation" leg (method review): a lack is only legible when the *winning*
    answers near-universally express Y (high Wilson lower bound) AND winners exceed
    losers on it (gap) — otherwise a bare Δwin>0 can flag a rare-among-winners concept.
    Winner is A when ``y==1``, B when ``y==0``; ties dropped.
    """
    dec = y != 0.5
    a_won = y == 1.0
    out: dict[tuple[int, int], dict] = {}
    for x, ys in by_x.items():
        xm = dec & (X == x)
        aw = a_won[xm]
        za, zb = z_a[xm], z_b[xm]
        n = int(xm.sum())
        for yy in ys:
            wf = np.where(aw, za[:, yy], zb[:, yy])          # winner's fire of Y
            lf = np.where(aw, zb[:, yy], za[:, yy])          # loser's fire of Y
            k = int(wf.sum())
            out[(x, yy)] = {"p_win": float(wf.mean()) if n else float("nan"),
                            "p_lose": float(lf.mean()) if n else float("nan"),
                            "wilson_lb": _wilson_lb(k, n, z), "n_win": n}
    return out


def gate_candidates(elicitation: pd.DataFrame, conditional: pd.DataFrame) -> set[tuple[int, int]]:
    """Candidate (prompt X, response Y) cells: elicited (lift sig & >1) AND rewarded (δ sig & >0).

    ``elicitation``: ``prompt_response_elicitation.csv`` (prompt_feature, completion_feature,
    lift, significant). ``conditional``: ``conditional_win_relevance.csv`` (prompt_concept,
    feature_id, delta_win_rate, cond_significant).
    """
    el = elicitation
    elicited = {
        (int(x), int(y))
        for x, y, lift, sig in zip(el["prompt_feature"], el["completion_feature"],
                                   el["lift"], el["significant"])
        if bool(sig) and float(lift) > 1.0
    }
    cd = conditional
    rewarded = {
        (int(x), int(y))
        for x, y, d, sig in zip(cd["prompt_concept"], cd["feature_id"],
                                cd["delta_win_rate"], cd["cond_significant"])
        if bool(sig) and float(d) > 0.0
    }
    return elicited & rewarded


def conditional_omissions(
    z_a: np.ndarray, z_b: np.ndarray, model_a, model_b, prompt_type, human_pref,
    candidates: set[tuple[int, int]], *,
    min_battles: int = 300, base_floor: float = 0.15, min_shortfall: float = 0.05,
    expect_theta: float = 0.8, expect_gap: float = 0.1, min_win: int = 200,
    seed: int = 0,
) -> pd.DataFrame:
    """Per (model, prompt_type X, response Y) production shortfall vs the paired opponent.

    ``z_a``/``z_b``: (N, M) completion codes for side A/B (fire ⇔ z>0). ``model_a``/
    ``model_b``: (N,) model on each side. ``prompt_type``: (N,) X per battle (argmax of
    the prompt code). ``human_pref``: (N,) P(A preferred) in {0,0.5,1}. ``candidates``:
    gated (X, Y) cells from :func:`gate_candidates`.

    ``flagged`` requires, conjunctively: a paired-opponent shortfall (McNemar, Bonferroni,
    split-half stable) AND an **expectation gate** — winners near-universally express Y for
    this prompt type (``win_wilson_lb >= expect_theta``) and beat losers on it
    (``p_win - p_lose >= expect_gap``) with ``>= min_win`` winners. The expectation gate is
    what makes "lacking it is bad" legible: it keeps a concept that winners only *sometimes*
    produce (a style choice) from being flagged as a defect, even if the model under-produces it.

    Returns one row per evaluated (model, X, Y) with ``expected`` (opponent fire),
    ``produced`` (M fire), ``shortfall``, ``n`` battles, ``mcnemar_p``, ``stable``
    (split-half sign), ``won_when_fired``/``won_when_not`` (corroboration), ``p_win``/
    ``p_lose``/``win_wilson_lb`` (winners' prevalence of Y), ``expected_norm``, and
    ``flagged``. ``mcnemar_p_bonferroni`` is added over evaluated cells.
    """
    z_a = np.asarray(z_a) > 0
    z_b = np.asarray(z_b) > 0
    model_a = np.asarray(model_a).astype(str)
    model_b = np.asarray(model_b).astype(str)
    X = np.asarray(prompt_type)
    y = np.asarray(human_pref, dtype=float)
    rng = np.random.default_rng(seed)

    by_x: dict[int, list[int]] = {}
    for x, yy in candidates:
        by_x.setdefault(int(x), []).append(int(yy))

    # expectation leg: how universally winners express each candidate concept (model-indep)
    prev = _winning_prevalence(z_a, z_b, X, y, by_x)

    models = sorted(set(model_a.tolist()) | set(model_b.tolist()))
    rows = []
    for m in models:
        involved = (model_a == m) | (model_b == m)
        for x, ys in by_x.items():
            mask = involved & (X == x)
            n = int(mask.sum())
            if n < min_battles:
                continue
            m_is_a = model_a[mask] == m
            # M won this battle? (decisive only) — M as A wins iff A preferred, etc.
            dec = y[mask] != 0.5
            m_won = np.where(m_is_a, y[mask] == 1.0, y[mask] == 0.0)
            half = rng.permutation(n)
            h1, h2 = half[: n // 2], half[n // 2:]
            for yy in ys:
                m_fire = np.where(m_is_a, z_a[mask, yy], z_b[mask, yy])
                o_fire = np.where(m_is_a, z_b[mask, yy], z_a[mask, yy])
                expected, produced = float(o_fire.mean()), float(m_fire.mean())
                if expected < base_floor:
                    continue
                shortfall = expected - produced
                b01 = int((o_fire & ~m_fire).sum())     # opponent fired, M didn't
                b10 = int((~o_fire & m_fire).sum())      # M fired, opponent didn't
                p = _mcnemar_p(b01, b10)
                s1 = o_fire[h1].mean() - m_fire[h1].mean()
                s2 = o_fire[h2].mean() - m_fire[h2].mean()
                stable = bool(np.sign(s1) == np.sign(s2) and shortfall > 0)
                wf = m_won[dec & m_fire]
                wn = m_won[dec & ~m_fire]
                pv = prev[(x, yy)]
                rows.append({
                    "model": m, "prompt_concept": int(x), "feature_id": int(yy),
                    "n": n, "expected": round(expected, 4), "produced": round(produced, 4),
                    "shortfall": round(shortfall, 4), "mcnemar_p": p,
                    "stable": stable,
                    "won_when_fired": round(float(wf.mean()), 4) if len(wf) else None,
                    "won_when_not": round(float(wn.mean()), 4) if len(wn) else None,
                    # expectation leg — winners' prevalence of Y for this prompt type
                    "p_win": round(pv["p_win"], 4), "p_lose": round(pv["p_lose"], 4),
                    "win_wilson_lb": round(pv["wilson_lb"], 4), "n_win": pv["n_win"],
                })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["mcnemar_p_bonferroni"] = (df["mcnemar_p"] * len(df)).clip(upper=1.0)
    # expectation gate: winners near-universally produce Y (Wilson LB ≥ θ) AND beat losers
    # on it (gap ≥ ε) with enough winners — makes "lacking it is bad" legible, not just Δwin>0.
    df["expected_norm"] = (
        (df["win_wilson_lb"] >= expect_theta)
        & ((df["p_win"] - df["p_lose"]) >= expect_gap)
        & (df["n_win"] >= min_win)
    )
    df["flagged"] = (
        (df["shortfall"] >= min_shortfall)
        & (df["mcnemar_p_bonferroni"] < 0.05)
        & df["stable"]
        & df["expected_norm"]
    )
    return df.sort_values(["model", "shortfall"], ascending=[True, False]).reset_index(drop=True)
