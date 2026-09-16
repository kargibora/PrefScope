import json
from math import comb

import numpy as np
import pandas as pd
import pytest

from prefscope.recipes.analysis.context import (
    _context_membership,
    classify_feature,
    profile_feature_context,
    profile_prompt_linkage,
)


def test_ambiguous_semantic_role_is_not_promoted_to_general():
    category = classify_feature(
        semantic_role="mixed_or_unclear", requested_share=0.0,
        choice_ratio=0.8, prompt_dependence=0.1,
        n_contexts=8, max_context_share=0.2)

    assert category == "context_specific"


def test_context_profile_separates_general_tendency_from_prompt_content():
    # Six prompt contexts, twenty A-vs-B battles each.
    contexts = np.repeat(np.arange(6), 20)
    n = len(contexts)
    z_a = np.zeros((n, 2), dtype=np.float32)
    z_b = np.zeros((n, 2), dtype=np.float32)
    # f0: model A consistently chooses the response policy in every context; B does not.
    z_a[:, 0] = 2.0
    # f1: both models produce the requested output type only on context 0; prompt-forced.
    z_a[contexts == 0, 1] = 2.0
    z_b[contexts == 0, 1] = 2.0
    calibration = pd.DataFrame({
        "feature_id": [0, 1],
        "concept": ["declines unsafe requests", "predicts sports scores in a table"],
        "semantic_threshold": [1.0, 1.0],
        "presence_pass": [True, True],
        "semantic_role": ["response_policy", "requested_task"],
        "requested_share": [0.0, 1.0],
    })
    feature, model = profile_feature_context(
        z_a, z_b, calibration, contexts,
        np.array(["A"] * n), np.array(["B"] * n),
        min_context_occurrences=10, min_model_context_battles=20,
        min_model_context_discordant=3)

    indexed = feature.set_index("feature_id")
    assert indexed.loc[0, "behavior_category"] == "general"
    assert indexed.loc[0, "paired_choice_ratio"] == 1.0
    assert indexed.loc[1, "behavior_category"] == "prompt_content"
    assert indexed.loc[1, "paired_choice_ratio"] == 0.0

    a0 = model[(model["model"] == "A") & (model["feature_id"] == 0)].iloc[0]
    assert bool(a0["cross_context_stable"]) is True
    assert a0["behavior_category"] == "general"
    # Prompt-forced f1 has no discordant A/B evidence, so it cannot become a model tendency.
    assert not ((model["model"] == "A") & (model["feature_id"] == 1)).any()


def test_context_profile_accepts_overlapping_prompt_membership():
    base = np.repeat(np.arange(6), 20)
    n = len(base)
    membership = np.column_stack(
        [base == context for context in range(6)] + [np.ones(n, dtype=bool)])
    context_ids = [10, 11, 12, 13, 14, 15, 99]
    z_a = np.full((n, 1), 2.0, dtype=np.float32)
    z_b = np.zeros((n, 1), dtype=np.float32)
    calibration = pd.DataFrame({
        "feature_id": [0], "concept": ["uses a structured answer"],
        "semantic_threshold": [1.0], "presence_pass": [True],
        "semantic_role": ["presentation"], "requested_share": [0.0],
    })

    feature, model = profile_feature_context(
        z_a, z_b, calibration, membership,
        np.array(["A"] * n), np.array(["B"] * n),
        prompt_context_ids=context_ids,
        min_context_occurrences=10, min_model_context_battles=20,
        min_model_context_discordant=3)

    assert feature.loc[0, "n_supported_prompt_contexts"] == 7
    a = model[(model["model"] == "A") & (model["feature_id"] == 0)].iloc[0]
    assert a["n_supported_contexts"] == 7
    assert bool(a["cross_context_stable"]) is True


def test_prompt_linkage_separates_linked_unlinked_and_sparse_features():
    contexts = np.repeat(np.arange(4), 100)
    prompt_scores = np.zeros((len(contexts), 4), dtype=np.float32)
    for context in range(4):
        rows = np.flatnonzero(contexts == context)
        prompt_scores[rows, context] = np.linspace(2.0, 1.0, len(rows))
    z_a = np.zeros((len(contexts), 4), dtype=np.float32)
    z_b = np.zeros_like(z_a)

    # f0 is strong only outside every prompt feature's high-activation tails.
    for context in range(4):
        rows = np.flatnonzero(contexts == context)[-20:]
        z_a[rows, 0] = 2.0 + np.arange(20, dtype=np.float32) / 1000
    # f1 is aligned with the strongest prompt activations in context 0.
    z_a[np.flatnonzero(contexts == 0)[:80], 1] = 3.0
    # f2 does not have enough positive prompts to classify.
    z_a[:10, 2] = 2.0
    # f3 would have sufficient evidence but is excluded by fidelity.
    z_a[:, 3] = 2.0

    features = pd.DataFrame({
        "feature_id": [0, 1, 2, 3],
        "concept": ["refuses", "writes Python code", "rare phrase", "unverified"],
        "fidelity_pass": [True, True, True, False],
        "semantic_role": [
            "response_policy", "requested_task", "mixed_or_unclear", "presentation",
        ],
        "requested_share": [0.0, 1.0, np.nan, 0.0],
    })
    prompt_names = pd.DataFrame({
        "feature_id": [10, 11, 12, 13],
        "concept": ["safety", "coding", "math", "writing"],
    })

    result = profile_prompt_linkage(
        z_a,
        z_b,
        prompt_scores,
        features=features,
        prompt_names=prompt_names,
        prompt_context_ids=[10, 11, 12, 13],
        top_n=80,
        min_top_examples=30,
        prompt_tail_fractions=(0.1, 0.15, 0.2),
        min_tail_overlap=5,
        min_context_lift=2.0,
        min_stable_scales=2,
    ).set_index("feature_id")

    assert list(result.index) == [0, 1, 2]
    assert result.loc[0, "prompt_scope"] == "no_detected_prompt_link"
    assert result.loc[0, "feature_type"] == "response_behavior"
    assert result.loc[1, "prompt_scope"] == "prompt_linked"
    assert result.loc[1, "feature_type"] == "requested_or_content"
    assert result.loc[2, "prompt_scope"] == "insufficient_evidence"
    assert result.loc[0, "paired_choice_ratio"] == 1.0
    assert result.loc[1, "n_linked_prompt_contexts"] == 1


def test_context_membership_rejects_nan_and_preserves_mixed_label_types():
    with np.testing.assert_raises_regex(ValueError, "finite boolean or numeric 0/1"):
        _context_membership(np.array([[1.0], [np.nan]]))
    ids, membership = _context_membership([1, True, 1.0, "1"])
    assert len(ids) == 4
    assert membership.shape == (4, 4)
    assert np.array_equal(membership.sum(axis=1), np.ones(4))


def test_constant_activation_tails_do_not_fabricate_prompt_links():
    scores = np.ones((10000, 1))
    row = profile_prompt_linkage(scores, scores, scores).iloc[0]

    assert row["prompt_scope"] == "no_detected_prompt_link"
    assert row["n_top_prompts"] == len(scores)
    assert row["n_linked_prompt_contexts"] == 0
    context = json.loads(row["top_prompt_contexts_json"])[0]
    for scale in context["scales"]:
        assert scale["n_overlap"] == len(scores)
        assert scale["top_share"] == scale["corpus_share"] == 1.0
        assert scale["lift"] == scale["q_value"] == 1.0


def test_partial_cutoff_ties_use_actual_sizes_and_are_row_order_invariant():
    response = np.array([5, 4, 4, 4, 4, 0, 0, 0, 0, 0])[:, None]
    prompt = np.array([4, 3, 3, 3, 0, 0, 0, 0, 0, 0])[:, None]
    options = dict(top_n=2, min_top_examples=2, prompt_tail_fractions=(0.2, 0.3),
                   min_tail_overlap=1, min_context_lift=1.0, min_stable_scales=2)
    result = profile_prompt_linkage(response, np.zeros_like(response), prompt, **options)
    permutation = np.array([9, 3, 5, 1, 8, 4, 0, 7, 2, 6])
    permuted = profile_prompt_linkage(
        response[permutation], np.zeros_like(response), prompt[permutation], **options)

    pd.testing.assert_frame_equal(result, permuted)
    row = result.iloc[0]
    assert row["n_top_prompts"] == 5
    assert row["strong_activation_threshold"] == 4
    assert row["prompt_scope"] == "prompt_linked"
    context = json.loads(row["top_prompt_contexts_json"])[0]
    # Both requested scales still count, even though their realized sets coincide.
    assert context["n_scales_passed"] == 2
    for scale in context["scales"]:
        assert scale["n_overlap"] == 4
        assert scale["top_share"] == 0.8
        assert scale["corpus_share"] == 0.4
        assert scale["lift"] == 2.0
        assert scale["q_value"] == pytest.approx(comb(5, 4) / comb(10, 4))


def test_linkage_does_not_create_ties_by_rounding_float64_scores():
    scores = np.array([1 + 3e-9, 1 + 2e-9, 1 + 1e-9, 1])[:, None]
    row = profile_prompt_linkage(
        scores, np.zeros_like(scores), scores, top_n=2, min_top_examples=1,
        prompt_tail_fractions=(0.5,), min_stable_scales=1).iloc[0]

    assert row["n_top_prompts"] == 2
    assert row["strong_activation_threshold"] == scores[1, 0]
    scale = json.loads(row["top_prompt_contexts_json"])[0]["scales"][0]
    assert scale["n_overlap"] == 2
    assert scale["corpus_share"] == 0.5


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, 1 + 1j, "missing", None])
@pytest.mark.parametrize("column", ["z_a", "z_b", "prompt_scores"])
def test_linkage_rejects_nonfinite_or_nonreal_scores(bad, column):
    inputs = {name: np.ones((2, 1)) for name in ("z_a", "z_b", "prompt_scores")}
    inputs[column] = np.array([[bad], [1]])
    with pytest.raises(ValueError, match="finite real scores"):
        profile_prompt_linkage(**inputs)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, 1 + 1j, "missing", None])
@pytest.mark.parametrize("column", ["z_a", "z_b"])
def test_context_profile_rejects_nonfinite_or_nonreal_scores(bad, column):
    inputs = {name: np.ones((2, 1)) for name in ("z_a", "z_b")}
    inputs[column] = np.array([[bad], [1]])
    calibration = pd.DataFrame({"feature_id": [0], "semantic_threshold": [0.5]})
    with pytest.raises(ValueError, match="finite real scores"):
        profile_feature_context(
            **inputs, calibration=calibration, prompt_context=[0, 0],
            model_a=["A", "A"], model_b=["B", "B"])


def test_context_membership_rejects_complex_boolean_values():
    with pytest.raises(ValueError, match="real numeric 0/1"):
        _context_membership(np.array([[1 + 1j], [0]]))


@pytest.mark.parametrize("empty_case", ["concordant", "failed_calibration", "empty_calibration", "no_rows"])
def test_context_profile_empty_results_keep_full_columns(empty_case):
    response = np.ones((2, 1))
    calibration = pd.DataFrame({"feature_id": [0], "semantic_threshold": [0.5],
                                "presence_pass": [True]})
    full_features, full_models = profile_feature_context(
        response, np.zeros_like(response), calibration, [0, 0], ["A", "A"], ["B", "B"])
    other = response.copy()
    contexts, model_a, model_b = [0, 0], ["A", "A"], ["B", "B"]
    if empty_case == "failed_calibration":
        calibration["presence_pass"] = False
    elif empty_case == "empty_calibration":
        calibration = calibration.iloc[:0]
    elif empty_case == "no_rows":
        response, other = response[:0], other[:0]
        contexts, model_a, model_b = [], [], []
    features, models = profile_feature_context(
        response, other, calibration, contexts, model_a, model_b)

    assert list(features.columns) == list(full_features.columns)
    assert list(models.columns) == list(full_models.columns)
    assert models.empty
    assert features.empty == (empty_case in {"failed_calibration", "empty_calibration"})


def test_linkage_empty_results_keep_full_columns():
    scores = np.ones((2, 1))
    full = profile_prompt_linkage(scores, scores, scores)
    empty = profile_prompt_linkage(
        scores, scores, scores,
        features=pd.DataFrame({"feature_id": [0], "fidelity_pass": [False]}))
    assert empty.empty
    assert list(empty.columns) == list(full.columns)


def test_context_profile_counts_typed_labels_separately_and_retains_json_ids():
    labels = [1, 1, True, True, True, 1.0, "1", "1", "coding"]
    n = len(labels)
    response = np.ones((n, 1))
    calibration = pd.DataFrame({"feature_id": [0], "semantic_threshold": [0.5]})
    prompt_names = pd.DataFrame({
        "feature_id": pd.Series([1, True, 1.0, "1", "coding"], dtype=object),
        "concept": ["integer", "boolean", "float", "string", "topic"],
    })
    features, models = profile_feature_context(
        response, np.zeros_like(response), calibration, labels, ["A"] * n, ["B"] * n,
        prompt_names=prompt_names, min_context_occurrences=1,
        min_model_context_battles=1, min_model_context_discordant=1)

    assert features.loc[0, "n_supported_prompt_contexts"] == 5
    assert features.loc[0, "max_prompt_context_share"] == 3 / n
    contexts = json.loads(features.loc[0, "top_prompt_contexts_json"])
    actual = {(type(row["prompt_feature_id"]), row["prompt_feature_id"]):
              (row["n_present"], row["concept"]) for row in contexts}
    expected = {(int, 1): (2, "integer"), (bool, True): (3, "boolean"),
                (float, 1.0): (1, "float"), (str, "1"): (2, "string"),
                (str, "coding"): (1, "topic")}
    assert actual == expected
    for model in models.itertuples():
        assert model.n_supported_contexts == 5
        effects = json.loads(model.top_contexts_json)
        assert {(type(row["prompt_feature_id"]), row["prompt_feature_id"]):
                (row["n_battles"], row["concept"]) for row in effects} == expected


@pytest.mark.parametrize("bad_id", [0.9, True, 0.0, "0"])
@pytest.mark.parametrize("source", ["features", "prompt_names", "prompt_context_ids"])
def test_linkage_rejects_noninteger_ids(bad_id, source):
    options = {source: [bad_id] if source == "prompt_context_ids" else pd.DataFrame({
        "feature_id": pd.Series([bad_id], dtype=object), "concept": ["test"]})}
    scores = np.ones((2, 1))
    with pytest.raises(ValueError, match="non-boolean integers"):
        profile_prompt_linkage(scores, scores, scores, **options)


@pytest.mark.parametrize("bad_id", [-1, 1])
def test_linkage_rejects_response_ids_outside_score_columns(bad_id):
    scores = np.ones((2, 1))
    with pytest.raises(ValueError, match="inside"):
        profile_prompt_linkage(
            scores, scores, scores, features=pd.DataFrame({"feature_id": [bad_id]}))


def test_linkage_rejects_duplicate_context_ids_and_keeps_last_annotation():
    scores = np.ones((2, 1))
    with pytest.raises(ValueError, match="unique"):
        profile_prompt_linkage(scores, scores, np.ones((2, 2)), prompt_context_ids=[2, 2])
    result = profile_prompt_linkage(
        scores, scores, scores, prompt_context_ids=[np.int64(2)],
        features=pd.DataFrame({"feature_id": [0, 0], "concept": ["old", "new"]}),
        prompt_names=pd.DataFrame({"feature_id": [2, 2], "concept": ["old", "new"]}))
    assert result.loc[0, "concept"] == "new"
    context = json.loads(result.loc[0, "top_prompt_contexts_json"])[0]
    assert context["prompt_feature_id"] == 2
    assert context["concept"] == "new"


@pytest.mark.parametrize("bad_id", [0.9, True, 0.0, "0", -1, 1])
def test_context_profile_rejects_invalid_calibration_ids(bad_id):
    scores = np.ones((2, 1))
    calibration = pd.DataFrame({"feature_id": pd.Series([bad_id], dtype=object),
                                "semantic_threshold": [0.5]})
    with pytest.raises(ValueError, match="non-boolean integers|inside"):
        profile_feature_context(scores, scores, calibration, [0, 0], ["A", "A"], ["B", "B"])


@pytest.mark.parametrize("dtype", [bool, np.int64, np.float32, np.float64])
def test_context_profilers_accept_real_numeric_and_boolean_scores(dtype):
    present = np.ones((2, 1), dtype=dtype)
    absent = np.zeros_like(present)
    linkage = profile_prompt_linkage(present, absent, present)
    calibration = pd.DataFrame({"feature_id": [0], "semantic_threshold": [0.5]})
    features, models = profile_feature_context(
        present, absent, calibration, [0, 0], ["A", "A"], ["B", "B"])

    assert linkage.loc[0, "n_top_prompts"] == 2
    assert features.loc[0, "paired_choice_ratio"] == 1.0
    assert len(models) == 2
