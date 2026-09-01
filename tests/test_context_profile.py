import numpy as np
import pandas as pd

from prefscope.analysis.context import (
    _context_membership,
    classify_feature,
    profile_feature_context,
    profile_prompt_linkage,
)
from prefscope.cli import main


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


def test_context_profile_cli_runs_llm_free_without_calibration(tmp_path):
    completion_lens = tmp_path / "completion"
    prompt_lens = tmp_path / "prompt"
    completion_lens.mkdir()
    prompt_lens.mkdir()
    contexts = np.repeat(np.arange(4), 100)
    ids = np.asarray([f"battle-{i}" for i in range(len(contexts))])
    meta = pd.DataFrame({
        "instruction_id": ids,
        "model_a": "model-a",
        "model_b": "model-b",
    })
    meta.to_parquet(completion_lens / "battles.parquet", index=False)
    meta.assign(battle_id=ids).to_parquet(
        prompt_lens / "battles.parquet", index=False
    )
    z_a = np.zeros((len(contexts), 1), dtype=np.float32)
    z_a[np.flatnonzero(contexts == 0)[:80], 0] = 2.0
    np.save(completion_lens / "z_a.npy", z_a)
    np.save(completion_lens / "z_b.npy", np.zeros_like(z_a))
    z_prompt = np.zeros((len(contexts), 4), dtype=np.float32)
    for context in range(4):
        rows = np.flatnonzero(contexts == context)
        z_prompt[rows, context] = np.linspace(2.0, 1.0, len(rows))
    np.save(prompt_lens / "z_prompt.npy", z_prompt)

    names = tmp_path / "features.csv"
    pd.DataFrame({
        "feature_id": [0],
        "concept": ["uses a response policy"],
        "fidelity_pass": [True],
    }).to_csv(names, index=False)
    prompt_names = tmp_path / "prompt_names.csv"
    prompt_fidelity = tmp_path / "prompt_fidelity.csv"
    pd.DataFrame({
        "feature_id": range(4),
        "concept": ["safety", "coding", "math", "writing"],
    }).to_csv(prompt_names, index=False)
    pd.DataFrame({
        "feature_id": range(4),
        "concept": ["safety", "coding", "math", "writing"],
        "fidelity_pass": True,
    }).to_csv(prompt_fidelity, index=False)
    out = tmp_path / "scope.csv"

    code = main([
        "context-profile",
        "--completion-lens", str(completion_lens),
        "--prompt-lens", str(prompt_lens),
        "--names", str(names),
        "--prompt-names", str(prompt_names),
        "--prompt-fidelity", str(prompt_fidelity),
        "--out", str(out),
        "--top-n", "80",
        "--prompt-tail-fractions", "0.1", "0.15", "0.2",
    ])

    assert code == 0
    result = pd.read_csv(out)
    assert result.loc[0, "prompt_scope"] == "prompt_linked"
    assert result.loc[0, "scope_method"] == "stable_prompt_tail_enrichment"


def test_context_membership_rejects_nan_and_preserves_mixed_label_types():
    with np.testing.assert_raises_regex(ValueError, "finite boolean or numeric 0/1"):
        _context_membership(np.array([[1.0], [np.nan]]))
    ids, membership = _context_membership([1, True, 1.0, "1"])
    assert len(ids) == 4
    assert membership.shape == (4, 4)
    assert np.array_equal(membership.sum(axis=1), np.ones(4))
