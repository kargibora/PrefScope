from prefscope.analysis.omission import _mcnemar_p


def test_mcnemar_equal_discordant_is_not_significant():
    """Equal discordant counts = no asymmetry: the continuity-corrected stat must clamp to
    0 (p == 1.0), not (0-1)**2/n which inflated significance (#3)."""
    assert _mcnemar_p(5, 5) == 1.0
    assert _mcnemar_p(1, 1) == 1.0
    assert _mcnemar_p(0, 0) == 1.0


def test_mcnemar_strong_asymmetry_is_significant():
    assert _mcnemar_p(20, 0) < 0.05
