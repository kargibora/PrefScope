"""Unit tests for the canonical pair schema (`prefscope.data.pair_schema`).

The schema is the single point of truth for pair column names and label
orientation; these tests pin (a) the encode-dataset -> canonical renames,
(b) has_preference detection, and (c) that ``orient_by_label`` reproduces the
exact ``keep``/``sign(y - 0.5)`` semantics the pipeline call sites used inline.
"""
import numpy as np
import pandas as pd
import pytest

from prefscope.data.pair_schema import (
    LABEL, MODEL_A, MODEL_B, PROMPT, RESPONSE_A, RESPONSE_B,
    normalize_pair_columns, orient_by_label,
)


def _canonical_df():
    return pd.DataFrame({
        PROMPT: ["p1", "p2"], RESPONSE_A: ["a1", "a2"], RESPONSE_B: ["b1", "b2"],
        MODEL_A: ["X", "Y"], MODEL_B: ["Y", "X"], LABEL: [1.0, 0.0]})


def test_normalize_canonical_passthrough_is_idempotent():
    df = _canonical_df()
    out, has_pref = normalize_pair_columns(df)
    assert list(out.columns) == list(df.columns)
    assert has_pref is True
    assert out is not df                      # copy, never a view of the input
    out2, has_pref2 = normalize_pair_columns(out)
    assert list(out2.columns) == list(out.columns) and has_pref2 is True


def test_normalize_renames_encode_dataset_columns():
    df = pd.DataFrame({"prompt": ["p"], "response": ["a"], "response_2": ["b"],
                       "model": ["X"], "model_2": ["Y"], "label": [0.5]})
    out, has_pref = normalize_pair_columns(df)
    assert set(out.columns) == {PROMPT, RESPONSE_A, RESPONSE_B, MODEL_A, MODEL_B, LABEL}
    assert has_pref is True
    assert list(df.columns)[1] == "response"  # input frame untouched


def test_normalize_never_clobbers_an_existing_canonical_column():
    # both the alias and its canonical twin present -> the alias is left alone
    df = pd.DataFrame({"model": ["alias"], "model_a": ["canon"]})
    out, _ = normalize_pair_columns(df)
    assert list(out["model_a"]) == ["canon"] and list(out["model"]) == ["alias"]


def test_has_preference_detection():
    no_label = pd.DataFrame({PROMPT: ["p"], RESPONSE_A: ["a"]})
    assert normalize_pair_columns(no_label)[1] is False
    all_nan = no_label.assign(**{LABEL: [np.nan]})
    assert normalize_pair_columns(all_nan)[1] is False
    one_value = pd.DataFrame({LABEL: [np.nan, 1.0]})
    assert normalize_pair_columns(one_value)[1] is True


def test_orient_by_label_flips_toward_winner_and_drops_ties():
    y = np.array([1.0, 0.0, 0.5, np.nan])
    diff = np.array([[1.0, -2.0], [3.0, -4.0], [5.0, 6.0], [7.0, 8.0]])
    oriented, keep = orient_by_label(y, diff)
    np.testing.assert_array_equal(keep, [True, True, False, False])
    # y=1 -> A won -> unchanged (+ = winner-more); y=0 -> B won -> sign-flipped
    np.testing.assert_allclose(oriented, [[1.0, -2.0], [-3.0, 4.0]])
    np.testing.assert_allclose(diff[0], [1.0, -2.0])   # input not modified in place


def test_orient_by_label_matches_inline_pipeline_semantics():
    # the exact expression prompt_delta / export_prompt_map used inline
    rng = np.random.default_rng(0)
    y = rng.choice([0.0, 0.5, 1.0, np.nan], size=40)
    diff = rng.normal(size=(40, 3))
    oriented, keep = orient_by_label(y, diff)
    expect_keep = ~np.isnan(y) & (y != 0.5)
    np.testing.assert_array_equal(keep, expect_keep)
    np.testing.assert_allclose(
        oriented, diff[expect_keep] * np.sign(y[expect_keep] - 0.5)[:, None])


def test_orient_by_label_keep_ties_zeroes_them():
    y = np.array([1.0, 0.5])
    diff = np.array([[2.0], [3.0]])
    oriented, keep = orient_by_label(y, diff, drop_ties=False)
    np.testing.assert_array_equal(keep, [True, True])
    np.testing.assert_allclose(oriented, [[2.0], [0.0]])  # tie has no winner -> zeroed


def test_orient_by_label_row_mismatch_raises():
    with pytest.raises(ValueError, match="row mismatch"):
        orient_by_label(np.array([1.0]), np.zeros((2, 2)))
