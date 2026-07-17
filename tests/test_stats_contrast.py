import numpy as np

from prefscope.analysis import inside_outside_contrast


def test_contrast_detects_separated_groups():
    rng = np.random.default_rng(0)
    inside = rng.normal(1.0, 1.0, 200)
    outside = rng.normal(0.0, 1.0, 200)
    c = inside_outside_contrast(inside, outside)
    assert c["delta"] > 0.5                 # inside mean clearly above outside
    assert c["welch_p"] < 0.05              # and significantly so
    assert c["cohens_d"] > 0


def test_contrast_degenerate_inputs_are_nan():
    one = inside_outside_contrast([1.0], [0.0])          # < 2 per side
    assert one["delta"] == 1.0 and np.isnan(one["welch_p"])
    const = inside_outside_contrast([1, 1, 1], [0, 0, 0])  # both constant
    assert const["delta"] == 1.0 and np.isnan(const["welch_t"])


def test_contrast_reports_means():
    c = inside_outside_contrast([2.0, 4.0], [1.0, 1.0])
    assert c["mean_inside"] == 3.0 and c["mean_outside"] == 1.0
