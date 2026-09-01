"""run_elicitation (the `prefscope elicit` core): loads two lenses, aligns, restricts
to verified axes, returns the co-activation lift edge table."""
import numpy as np
import pandas as pd
import pytest

from prefscope.artifacts import BATTLES, Z_A, Z_B, Z_PROMPT
from prefscope.pipeline.elicit import run_elicitation


def _lens(d, arrays, ids):
    d.mkdir(parents=True, exist_ok=True)
    for name, arr in arrays.items():
        np.save(d / name, arr)
    pd.DataFrame({"battle_id": [str(i) for i in ids]}).to_parquet(d / BATTLES)


def test_run_elicitation_aligns_and_returns_edges(tmp_path):
    rng = np.random.default_rng(0)
    N, Mc, Mp = 300, 10, 6
    za = (rng.random((N, Mc)) * (rng.random((N, Mc)) < 0.2)).astype(np.float32)
    zb = (rng.random((N, Mc)) * (rng.random((N, Mc)) < 0.2)).astype(np.float32)
    zp = (rng.random((N, Mp)) * (rng.random((N, Mp)) < 0.3)).astype(np.float32)
    # inject prompt-feature 1 -> response-feature 2 co-occurrence
    zb[:, 2] = np.where((zp[:, 1] > 0) & (rng.random(N) < 0.7), 1.0, zb[:, 2]).astype(np.float32)

    clens, plens = tmp_path / "c", tmp_path / "p"
    _lens(clens, {Z_A: za, Z_B: zb}, range(N))
    _lens(plens, {Z_PROMPT: zp}, range(N))

    edges = run_elicitation(clens, plens, min_support=20, min_cooccur=5, log=lambda *_: None)
    assert {"prompt_feature", "completion_feature", "lift", "significant"} <= set(edges.columns)
    hit = edges[(edges.prompt_feature == 1) & (edges.completion_feature == 2)]
    assert len(hit) == 1 and hit["lift"].iloc[0] > 1.5   # the injected elicitation surfaces


def test_run_elicitation_accepts_single_response_lens(tmp_path):
    rng = np.random.default_rng(4)
    n, mc, mp = 300, 8, 5
    zp = (rng.random((n, mp)) < 0.25).astype(np.float32)
    za = (rng.random((n, mc)) < 0.15).astype(np.float32)
    # prompt feature 1 reliably co-occurs with completion feature 3
    za[:, 3] = np.where(zp[:, 1] > 0, 1.0, za[:, 3])

    clens, plens = tmp_path / "single", tmp_path / "prompt"
    _lens(clens, {Z_A: za}, range(n))
    _lens(plens, {Z_PROMPT: zp}, range(n))

    messages = []
    edges = run_elicitation(
        clens, plens, min_support=20, min_cooccur=5, log=messages.append)

    hit = edges[(edges.prompt_feature == 1) & (edges.completion_feature == 3)]
    assert len(hit) == 1 and hit.iloc[0]["lift"] > 2.0
    assert any("prompt/completion items" in message for message in messages)


def test_run_elicitation_accepts_merged_annotation_tables(tmp_path):
    n = 80
    zp = np.zeros((n, 2), np.float32)
    za = np.zeros((n, 2), np.float32)
    zp[:40, 1] = 1
    za[:40, 0] = 1
    clens, plens = tmp_path / "completion", tmp_path / "prompt"
    _lens(clens, {Z_A: za}, range(n))
    _lens(plens, {Z_PROMPT: zp}, range(n))
    completion_annotations = pd.DataFrame({
        "feature_id": [0, 1], "concept": ["response", "unused"],
        "fidelity_pass": [True, False],
    })
    prompt_annotations = pd.DataFrame({
        "feature_id": [0, 1], "concept": ["unused", "question"],
        "fidelity_pass": [False, True],
    })

    edges = run_elicitation(
        clens, plens,
        completion_names=completion_annotations,
        completion_fidelity=completion_annotations,
        prompt_names=prompt_annotations,
        prompt_fidelity=prompt_annotations,
        min_support=10, min_cooccur=5, log=lambda *_: None)

    assert set(edges["completion_feature"]) == {0}
    assert set(edges["prompt_feature"]) == {1}
    assert edges.iloc[0]["completion_feature_name"] == "response"
    assert edges.iloc[0]["prompt_feature_name"] == "question"


def test_run_elicitation_rejects_difference_lens(tmp_path):
    # a difference lens has z_diff but no z_a/z_b
    d = tmp_path / "diff"
    _lens(d, {"z_diff.npy": np.zeros((5, 4), np.float32)}, range(5))
    p = tmp_path / "p"
    _lens(p, {Z_PROMPT: np.zeros((5, 4), np.float32)}, range(5))
    with pytest.raises(ValueError, match="INDIVIDUAL"):
        run_elicitation(d, p)


def test_run_elicitation_aligns_optional_prompt_groups(tmp_path):
    ids = np.arange(8)
    group_ids = np.array(["a", "a", "b", "b", "c", "c", "d", "d"])
    prompt_by_id = (ids >= 4).astype(np.float32)[:, None]
    response_by_id = np.array([0, 0, 1, 0, 1, 1, 1, 0], dtype=np.float32)[:, None]

    clens, plens = tmp_path / "grouped_completion", tmp_path / "grouped_prompt"
    _lens(clens, {Z_A: response_by_id}, ids)
    reverse = ids[::-1]
    _lens(plens, {Z_PROMPT: prompt_by_id[reverse]}, reverse)

    result = run_elicitation(
        clens,
        plens,
        group_ids=group_ids,
        min_support=1,
        min_cooccur=1,
        log=lambda *_: None,
    )

    assert result.attrs["n_groups"] == 4
    assert result.attrs["inference_method"] == "two_sample_hoeffding_group_prevalence"
    assert result.iloc[0]["n_x"] == 4


def test_run_elicitation_validates_group_ids_in_completion_order(tmp_path):
    clens, plens = tmp_path / "completion_groups", tmp_path / "prompt_groups"
    _lens(clens, {Z_A: np.ones((4, 1), np.float32)}, range(4))
    _lens(plens, {Z_PROMPT: np.ones((4, 1), np.float32)}, range(4))

    with pytest.raises(ValueError, match="length 4.*completion-lens order"):
        run_elicitation(clens, plens, group_ids=["a", "b"], log=lambda *_: None)
