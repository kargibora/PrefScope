import json
import re

import numpy as np
import pandas as pd

from prefscope.interpret.role import (
    classify_response_roles,
    combine_behavior_scope,
    semantic_family,
)


class _RoleClient:
    def __init__(self):
        self.prompts = []

    def raw(self, messages, **kwargs):
        prompt = messages[-1]["content"]
        self.prompts.append(prompt)
        behavioral = "structured headings" in prompt
        labels = []
        for sample_id in re.findall(r"<sample id=(\d+)", prompt):
            labels.append({
                "id": int(sample_id),
                "concept_present": True,
                "role": "presentation" if behavioral else "topic_content",
                "prompt_relation": (
                    "independently_chosen" if behavioral else "explicitly_requested"
                ),
            })
        return json.dumps({
            "feature_summary": (
                "Organizes answers with headings."
                if behavioral
                else "Discusses the requested medical subject."
            ),
            "labels": labels,
        })


class _IncompleteThenCompleteClient:
    def __init__(self):
        self.calls = 0

    def raw(self, messages, **kwargs):
        self.calls += 1
        sample_ids = [
            int(sample_id)
            for sample_id in re.findall(r"<sample id=(\d+)", messages[-1]["content"])
        ]
        if self.calls == 1:
            sample_ids = sample_ids[:1]
        return json.dumps({
            "feature_summary": "Organizes answers with headings.",
            "labels": [
                {
                    "id": sample_id,
                    "concept_present": True,
                    "role": "presentation",
                    "prompt_relation": "independently_chosen",
                }
                for sample_id in sample_ids
            ],
        })


class _PersistentlyIncompleteBatchClient:
    def __init__(self):
        self.calls = 0

    def raw(self, messages, **kwargs):
        self.calls += 1
        sample_ids = [
            int(sample_id)
            for sample_id in re.findall(r"<sample id=(\d+)", messages[-1]["content"])
        ]
        if len(sample_ids) > 1:
            sample_ids = sample_ids[:1]
        return json.dumps({
            "feature_summary": "Organizes answers with headings.",
            "labels": [
                {
                    "id": sample_id,
                    "concept_present": True,
                    "role": "presentation",
                    "prompt_relation": "independently_chosen",
                }
                for sample_id in sample_ids
            ],
        })


def test_role_classifier_separates_behavioral_and_prompt_specific_properties():
    n = 20
    battles = pd.DataFrame({
        "prompt": [f"Explain medical topic {i}" for i in range(n)],
        "completion_a": [f"## Finding {i}\nMedical response" for i in range(n)],
        "completion_b": [f"Plain counterpart {i}" for i in range(n)],
    })
    z_a = np.column_stack([
        np.linspace(1.0, 2.0, n),
        np.linspace(2.0, 1.0, n),
    ]).astype(np.float32)
    z_b = np.zeros_like(z_a)
    names = pd.DataFrame({
        "feature_id": [0, 1],
        "concept": ["uses structured headings", "discusses medicine"],
    })
    linkage = pd.DataFrame({
        "feature_id": [0, 1],
        "prompt_scope": ["no_detected_prompt_link", "prompt_linked"],
    })
    client = _RoleClient()

    result = classify_response_roles(
        battles,
        z_a,
        z_b,
        names,
        client,
        instruction_ids=[f"p{i}" for i in range(n)],
        linkage=linkage,
        n_top=4,
        n_random=2,
        min_valid_examples=4,
    ).set_index("feature_id")

    assert result.loc[0, "semantic_family"] == "behavioral"
    assert result.loc[0, "semantic_role"] == "presentation"
    assert result.loc[0, "behavior_scope"] == "candidate_cross_prompt_behavior"
    assert result.loc[0, "independent_share"] == 1.0
    assert result.loc[1, "semantic_family"] == "prompt_specific"
    assert result.loc[1, "behavior_scope"] == "prompt_content"
    assert result.loc[1, "requested_share"] == 1.0
    assert result.loc[1, "prompt_driven_share"] == 1.0
    assert "PAIRED RESPONSE TO THE SAME REQUEST" in client.prompts[0]
    assert len(json.loads(result.loc[0, "samples_json"])) == 6


def test_role_scope_helpers_are_conservative():
    assert semantic_family("reasoning_strategy") == "behavioral"
    assert semantic_family("topic_content") == "prompt_specific"
    assert combine_behavior_scope(
        "behavioral",
        "no_detected_prompt_link",
        prompt_driven_share=0.25,
        independent_share=0.75,
    ) == "candidate_cross_prompt_behavior"
    assert combine_behavior_scope(
        "behavioral",
        "no_detected_prompt_link",
        prompt_driven_share=0.75,
        independent_share=0.25,
    ) == "context_conditional_behavior"
    assert combine_behavior_scope(
        "prompt_specific",
        "no_detected_prompt_link",
        prompt_driven_share=0.0,
        independent_share=1.0,
    ) == "prompt_content"


def test_role_classifier_retries_an_incomplete_batch():
    n = 4
    battles = pd.DataFrame({
        "prompt": [f"Explain topic {i}" for i in range(n)],
        "completion_a": [f"## Finding {i}\nResponse" for i in range(n)],
        "completion_b": [f"Plain counterpart {i}" for i in range(n)],
    })
    z_a = np.ones((n, 1), dtype=np.float32)
    z_b = np.zeros_like(z_a)
    names = pd.DataFrame({
        "feature_id": [0],
        "concept": ["uses structured headings"],
    })
    client = _IncompleteThenCompleteClient()

    result = classify_response_roles(
        battles,
        z_a,
        z_b,
        names,
        client,
        n_top=4,
        n_random=0,
        batch_size=4,
        min_valid_examples=4,
    ).iloc[0]

    assert client.calls == 2
    assert result["classification_status"] == "ok"
    assert result["n_labelled"] == 4
    assert result["n_present"] == 4
    assert result["n_valid"] == 4
    assert result["label_coverage"] == 1.0
    assert json.loads(result["batch_summaries_json"])[0]["n_labelled"] == 4


def test_role_classifier_falls_back_to_missing_examples_individually():
    n = 4
    battles = pd.DataFrame({
        "prompt": [f"Explain topic {i}" for i in range(n)],
        "completion_a": [f"## Finding {i}\nResponse" for i in range(n)],
        "completion_b": [f"Plain counterpart {i}" for i in range(n)],
    })
    z_a = np.ones((n, 1), dtype=np.float32)
    z_b = np.zeros_like(z_a)
    names = pd.DataFrame({
        "feature_id": [0],
        "concept": ["uses structured headings"],
    })
    client = _PersistentlyIncompleteBatchClient()

    result = classify_response_roles(
        battles,
        z_a,
        z_b,
        names,
        client,
        n_top=4,
        n_random=0,
        batch_size=4,
        min_valid_examples=4,
    ).iloc[0]

    assert client.calls == 5
    assert result["classification_status"] == "ok"
    assert result["n_labelled"] == 4
    assert result["n_present"] == 4
    assert result["n_valid"] == 4
    assert result["label_coverage"] == 1.0
    assert json.loads(result["batch_summaries_json"])[0]["n_labelled"] == 4
