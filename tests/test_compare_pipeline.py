import json

import numpy as np
import pandas as pd
import pytest

from prefscope.__main__ import main
from prefscope.pipeline.compare import compare_encoded_responses
from prefscope.viewer_export.comparison import export_paired_comparison


def _write_bundle(tmp_path):
    response = tmp_path / "responses"
    prompt = tmp_path / "prompts"
    features = tmp_path / "response_features"
    prompt_features = tmp_path / "prompt_features"
    for path in (response, prompt, features, prompt_features):
        path.mkdir()

    n = 40
    meta = pd.DataFrame({
        "row_id": np.arange(n),
        "battle_id": [f"p{i}" for i in range(n)],
        "prompt": [f"question {i}" for i in range(n)],
        "completion_a": [f"base {i}" for i in range(n)],
        "completion_b": [f"adapted {i}" for i in range(n)],
        "model_a": ["base"] * n,
        "model_b": ["adapted"] * n,
        # This must not enter the descriptive paired comparison.
        "human_pref": np.where(np.arange(n) % 2, 1.0, 0.0),
    })
    meta.to_parquet(response / "meta.parquet", index=False)
    za = np.zeros((n, 2), dtype=np.float32)
    zb = np.zeros_like(za)
    zb[:, 0] = 2.0
    za[:20, 1] = zb[:20, 1] = 2.0
    np.save(response / "z_a.npy", za)
    np.save(response / "z_b.npy", zb)

    names = pd.DataFrame({
        "feature_id": [0, 1],
        "concept": ["uses a structured answer", "discusses mathematics"],
    })
    names.to_csv(features / "feature_names.csv", index=False)
    pd.DataFrame({"feature_id": [0, 1], "fidelity_pass": [True, True]}) \
        .to_csv(features / "feature_fidelity.csv", index=False)
    pd.DataFrame({
        "feature_id": [0, 1], "semantic_threshold": [1.0, 1.0],
        "presence_pass": [True, True],
        "semantic_role": ["presentation", "topic_content"],
        "requested_share": [0.0, 1.0],
    }).to_csv(features / "feature_calibration.csv", index=False)

    # Reverse the prompt row order to exercise ID-based alignment.
    pmeta = meta[["row_id", "battle_id", "prompt"]].iloc[::-1].reset_index(drop=True)
    pmeta.to_parquet(prompt / "meta.parquet", index=False)
    zp = np.zeros((n, 2), dtype=np.float32)
    original_context = np.column_stack([
        np.arange(n) < 20, np.arange(n) >= 20]).astype(np.float32) * 2
    zp[:] = original_context[::-1]
    np.save(prompt / "z_prompt.npy", zp)
    pd.DataFrame({
        "feature_id": [0, 1], "concept": ["math prompt", "other prompt"],
    }).to_csv(prompt_features / "prompt_feature_names.csv", index=False)
    pd.DataFrame({"feature_id": [0, 1], "fidelity_pass": [True, True]}) \
        .to_csv(prompt_features / "prompt_feature_fidelity.csv", index=False)
    pd.DataFrame({
        "feature_id": [0, 1], "semantic_threshold": [1.0, 1.0],
        "presence_pass": [True, True],
    }).to_csv(prompt_features / "feature_calibration.csv", index=False)
    return response, prompt, features, prompt_features


def test_compare_encoded_responses_writes_stable_artifacts(tmp_path):
    response, prompt, features, prompt_features = _write_bundle(tmp_path)
    result = compare_encoded_responses(
        response, features=features, prompt_dir=prompt,
        prompt_features=prompt_features, side_a_name="base",
        side_b_name="adapted", min_context_pairs=10)

    assert result.overall["feature_id"].tolist() == [0, 1]
    assert result.overall.set_index("feature_id").loc[0, "delta_b_minus_a"] == 1.0
    assert len(result.conditional) == 4
    assert set(result.conditional["region_concept"]) == {"math prompt", "other prompt"}
    assert set(result.examples["side_a_name"]) == {"base"}
    assert result.manifest["preference_labels_used"] is False

    out = result.save(tmp_path / "comparison")
    for filename in (
        "comparison.json", "concept_shift.parquet",
        "concept_shift_by_context.parquet", "response_scope.parquet",
        "paired_examples.parquet",
    ):
        assert (out / filename).exists()
    assert json.loads((out / "comparison.json").read_text())["side_b_name"] == "adapted"

    viewer = export_paired_comparison(out)
    assert viewer["meta"]["preference_labels_used"] is False
    assert len(viewer["concepts"]) == 2
    assert len(viewer["contexts"]) == 4
    assert viewer["examples"][0]["side_b_name"] == "adapted"


def test_compare_responses_cli_matches_pipeline_contract(tmp_path):
    response, prompt, features, prompt_features = _write_bundle(tmp_path)
    out = tmp_path / "cli_comparison"
    code = main([
        "compare-responses", "--encoded-dir", str(response),
        "--features", str(features), "--prompt-encoded-dir", str(prompt),
        "--prompt-features", str(prompt_features), "--side-a-name", "base",
        "--side-b-name", "adapted", "--min-context-pairs", "10",
        "--out", str(out),
    ])
    assert code == 0
    assert (out / "concept_shift.parquet").exists()
    assert pd.read_parquet(out / "concept_shift.parquet")["feature_id"].tolist() == [0, 1]


def test_compare_without_prompt_bundle_has_typed_empty_context_artifact(tmp_path):
    response, _, features, _ = _write_bundle(tmp_path)
    result = compare_encoded_responses(response, features=features)
    out = result.save(tmp_path / "no_prompt")

    conditional = pd.read_parquet(out / "concept_shift_by_context.parquet")
    assert conditional.empty
    assert {"region_id", "feature_id", "delta_b_minus_a"} <= set(conditional.columns)
    assert result.manifest["region_kind"] is None


def test_compare_groups_repeated_generations_and_aligns_by_unique_fallback(tmp_path):
    response, prompt, features, prompt_features = _write_bundle(tmp_path)
    response_meta = pd.read_parquet(response / "meta.parquet")
    prompt_meta = pd.read_parquet(prompt / "meta.parquet")
    # Duplicate battle IDs are legitimate when several generations share a prompt.
    # Alignment must fall back to unique row_id while the explicit group column controls
    # the independent unit for effects and uncertainty.
    response_meta["battle_id"] = [f"p{i // 2}" for i in range(len(response_meta))]
    prompt_meta["battle_id"] = [f"p{int(i) // 2}" for i in prompt_meta["row_id"]]
    response_meta["prompt_group"] = response_meta["battle_id"]
    response_meta.to_parquet(response / "meta.parquet", index=False)
    prompt_meta.to_parquet(prompt / "meta.parquet", index=False)

    result = compare_encoded_responses(
        response, features=features, prompt_dir=prompt,
        prompt_features=prompt_features, group_col="prompt_group",
        min_context_pairs=10)

    assert set(result.overall["n_groups"]) == {20}
    assert set(result.overall["test"]) == {"cluster_hoeffding"}
    assert result.overall.set_index("feature_id").loc[0, "n_nonzero_groups"] == 20
    assert len(result.conditional) == 4


def test_compare_group_column_error_explains_how_to_preserve_it(tmp_path):
    response, _, features, _ = _write_bundle(tmp_path)
    with pytest.raises(ValueError, match="--metadata-col prompt_group"):
        compare_encoded_responses(
            response, features=features, group_col="prompt_group")


def test_compare_prompt_clusters_carries_human_readable_region_labels(tmp_path):
    response, prompt, features, prompt_features = _write_bundle(tmp_path)
    clusters = pd.DataFrame({
        "feature_id": [0, 1], "cluster_id": [10, 20],
        "behavior": ["mathematics requests", "other requests"],
    })

    result = compare_encoded_responses(
        response, features=features, prompt_dir=prompt,
        prompt_features=prompt_features, prompt_clusters=clusters,
        min_context_pairs=10)

    assert set(result.conditional["region_concept"]) == {
        "mathematics requests", "other requests"}
    assert set(result.conditional["region_kind"]) == {"prompt_cluster"}
