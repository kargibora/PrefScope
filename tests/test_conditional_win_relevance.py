"""conditional_win_relevance: the prompt-type × behavior interaction δ_{f,k} (sign-flips)."""
import numpy as np

from prefscope.pipeline.winrelevance import conditional_win_relevance


def test_conditional_win_relevance_captures_sign_flip():
    rng = np.random.default_rng(0)
    n = 1600
    z = rng.normal(0, 1, (n, 1))
    pc = np.array([0] * (n // 2) + [1] * (n // 2))   # two prompt types
    y = np.zeros(n)
    p0 = pc == 0
    # type 0: A wins when the feature is HIGH; type 1: A wins when it is LOW (flip)
    y[p0] = (z[p0, 0] + rng.normal(0, 0.3, int(p0.sum())) > 0).astype(float)
    y[~p0] = (-z[~p0, 0] + rng.normal(0, 0.3, int((~p0).sum())) > 0).astype(float)

    out = conditional_win_relevance(z, y, np.zeros(n), pc, min_battles=100)

    assert set(out["prompt_concept"]) == {0, 1}
    d0 = out[out["prompt_concept"] == 0].iloc[0]["delta_win_rate"]
    d1 = out[out["prompt_concept"] == 1].iloc[0]["delta_win_rate"]
    assert d0 > 0.1 and d1 < -0.1                      # the conditional sign-flip
    assert bool(out[out["prompt_concept"] == 0].iloc[0]["cond_significant"])
    assert {"cond_p_bonferroni", "cond_significant", "n_battles"} <= set(out.columns)


def test_conditional_win_relevance_skips_thin_prompt_types():
    z = np.random.default_rng(1).normal(0, 1, (500, 2))
    y = (z[:, 0] > 0).astype(float)
    pc = np.array([0] * 480 + [1] * 20)               # type 1 too small
    out = conditional_win_relevance(z, y, np.zeros(500), pc, min_battles=100)
    assert set(out["prompt_concept"]) == {0}          # thin type dropped
