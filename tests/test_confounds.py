import numpy as np
import pandas as pd
import pytest

from prefscope.pipeline.confounds import partial_correlation, screen_length_confound


def test_length_confound_screen_handles_perfect_collinearity_without_false_verdict():
    direction = np.tile(np.array([-1.0, 1.0]), 100)
    z = np.column_stack([direction, np.tile([-1.0, -1.0, 1.0, 1.0], 50)])
    y = (direction > 0).astype(float)
    length = direction.copy()
    annotations = pd.DataFrame({
        "feature_id": [0, 1],
        "concept": ["tracks length", "other"],
        "fidelity_pass": ["True", "False"],
    })

    result, summary = screen_length_confound(
        z, y, length, annotations=annotations
    )
    by_id = result.set_index("feature_id")

    assert by_id.loc[0, "corr_confound_len"] == pytest.approx(1.0)
    # Perfect collinearity makes the partial correlation unidentified, which is itself
    # maximum entanglement: the data cannot separate reward from response length.
    assert np.isnan(by_id.loc[0, "correlation_resid_len"])
    assert bool(by_id.loc[0, "confound_entangled"])
    assert summary["n_features"] == 2
    assert summary["n_rows"] == 200


def test_length_confound_screen_flags_large_finite_collapse():
    rng = np.random.default_rng(3)
    n = 4000
    length = rng.normal(size=n)
    feature = length + rng.normal(scale=0.5, size=n)
    outcome_score = length + rng.normal(scale=0.8, size=n)
    y = (outcome_score > 0).astype(float)
    z = np.sign(feature)[:, None]

    result, _ = screen_length_confound(z, y, length)
    row = result.iloc[0]

    assert abs(row["corr_confound_len"]) >= 0.3
    assert abs(row["correlation_resid_len"]) < 0.5 * abs(row["correlation"])
    assert bool(row["confound_entangled"])


def test_confound_screen_validates_inputs_and_reports_permutation_null():
    z = np.tile([[-1.0], [1.0]], (20, 1))
    y = np.tile([0.0, 1.0], 20)
    length = np.linspace(-1.0, 1.0, 40)

    _, summary = screen_length_confound(
        z, y, length, permutations=3, seed=7
    )
    assert summary["permutations"] == 3
    assert 0.0 <= summary["permutation_empirical_p"] <= 1.0
    with pytest.raises(ValueError, match="align"):
        screen_length_confound(z, y[:-1], length)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        screen_length_confound(z, y, length, collapse_fraction=2.0)


def test_partial_correlation_is_nan_for_degenerate_inputs():
    assert np.isnan(partial_correlation([1, 1], [0, 1], [0, 1]))


def test_grouped_confound_estimand_is_invariant_to_unequal_row_duplication():
    rng = np.random.default_rng(11)
    n_groups = 30
    sign = np.where(np.arange(n_groups) % 2, 1.0, -1.0)
    outcome = (sign + rng.normal(scale=1.0, size=n_groups) > 0).astype(float)
    length = sign + rng.normal(scale=0.8, size=n_groups)

    def expanded(counts):
        index = np.repeat(np.arange(n_groups), counts)
        return (
            sign[index, None], outcome[index], length[index],
            np.array([f"g{value}" for value in index]),
        )

    equal = expanded(np.full(n_groups, 2))
    unequal_counts = np.full(n_groups, 2)
    unequal_counts[0] = 50
    unequal = expanded(unequal_counts)
    first, first_summary = screen_length_confound(
        equal[0], equal[1], equal[2], group_ids=equal[3])
    second, second_summary = screen_length_confound(
        unequal[0], unequal[1], unequal[2], group_ids=unequal[3])

    assert first_summary["analysis_unit"] == "group"
    assert second_summary["confound_estimand"] == (
        "equal_group_weight_firing_group_means")
    assert first.loc[0, "corr_confound_len"] == pytest.approx(
        second.loc[0, "corr_confound_len"])
    assert first.loc[0, "correlation_resid_len"] == pytest.approx(
        second.loc[0, "correlation_resid_len"])
