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


def test_conditional_win_relevance_accepts_overlapping_prompt_membership():
    rng = np.random.default_rng(5)
    n = 1200
    z = rng.normal(0, 1, (n, 1))
    membership = np.zeros((n, 2), dtype=bool)
    membership[:800, 0] = True
    membership[400:, 1] = True  # 400 rows belong to both prompt concepts
    y = np.zeros(n)
    y[membership[:, 0] & ~membership[:, 1]] = (
        z[membership[:, 0] & ~membership[:, 1], 0] > 0).astype(float)
    y[membership[:, 1]] = (
        z[membership[:, 1], 0] < 0).astype(float)

    out = conditional_win_relevance(
        z, y, np.zeros(n), membership, prompt_region_ids=[10, 20],
        min_battles=300)

    assert set(out["prompt_concept"]) == {10, 20}


def test_conditional_support_uses_exact_fitting_rows():
    rng = np.random.default_rng(7)
    z = rng.normal(size=(400, 1))
    y = np.full(400, 0.5)
    y[:50] = (z[:50, 0] > 0).astype(float)
    membership = np.ones((400, 1), dtype=bool)

    out = conditional_win_relevance(
        z, y, np.zeros(400), membership, min_battles=100)
    assert out.empty


def test_conditional_rejects_varying_membership_within_group():
    z = np.ones((4, 1))
    y = np.array([1.0, 0.0, 1.0, 0.0])
    membership = np.array([[True], [False], [True], [False]])
    with np.testing.assert_raises_regex(ValueError, "constant within each group"):
        conditional_win_relevance(
            z, y, np.zeros(4), membership,
            group_ids=["same", "same", "a", "b"], min_battles=1)
