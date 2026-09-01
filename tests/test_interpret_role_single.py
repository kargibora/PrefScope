import json
import re

import numpy as np
import pandas as pd

from prefscope.interpret.role import _classify_batch, _select_evidence, classify_response_roles


def _battles(n=6, paired=True):
    data = {
        "prompt": [f"p{i}" for i in range(n)],
        "completion_a": [f"a{i}" for i in range(n)],
    }
    if paired:
        data["completion_b"] = [f"b{i}" for i in range(n)]
    return pd.DataFrame(data)


class _Client:
    def __init__(self):
        self.prompts = []

    def raw(self, messages, **kwargs):
        system = messages[0]["content"]
        prompt = messages[-1]["content"]
        self.prompts.append((system, prompt))
        labels = [
            {"id": int(sid), "concept_present": True, "role": "presentation",
             "prompt_relation": "independently_chosen"}
            for sid in re.findall(r"<sample id=(\d+)", prompt)
        ]
        return json.dumps({"feature_summary": "s", "labels": labels})


def test_select_evidence_without_z_b_scores_on_z_a():
    z_a = np.zeros((6, 2), dtype=np.float32)
    z_a[:, 0] = [0.0, 5.0, 1.0, 0.0, 3.0, 2.0]
    ev = _select_evidence(z_a, None, list(range(6)), 0, n_top=3, n_random=0, seed=0)
    assert [e["row_index"] for e in ev] == [1, 4, 5]
    assert {e["side"] for e in ev} == {"a"}
    assert all(e["counterpart_activation"] is None for e in ev)


def test_classify_batch_omits_paired_block_for_single_response():
    client = _Client()
    ev = [{"row_index": 0, "instruction_id": "0", "side": "a", "activation": 1.0,
           "counterpart_activation": None, "evidence_kind": "top"}]
    _, rendered = _classify_batch(client, "concept", ev, _battles(paired=False))
    system, prompt = client.prompts[0]
    assert "PAIRED RESPONSE" not in prompt
    assert "paired response is contrastive context" not in system
    assert "FEATURE RESPONSE" in prompt
    assert rendered[0]["counterpart_excerpt"] is None


def test_classify_batch_keeps_paired_block_when_available():
    client = _Client()
    ev = [{"row_index": 0, "instruction_id": "0", "side": "a", "activation": 1.0,
           "counterpart_activation": 0.5, "evidence_kind": "top"}]
    _classify_batch(client, "concept", ev, _battles(paired=True))
    system, prompt = client.prompts[0]
    assert "PAIRED RESPONSE" in prompt
    assert "paired response is contrastive context" in system


def test_classify_response_roles_runs_on_single_response_data():
    z_a = np.zeros((6, 1), dtype=np.float32)
    z_a[:, 0] = [0.0, 5.0, 1.0, 0.0, 3.0, 2.0]
    names = pd.DataFrame({"feature_id": [0], "concept": ["uses headings"]})
    out = classify_response_roles(
        _battles(paired=False), z_a, None, names, _Client(),
        n_top=3, n_random=0, min_valid_examples=1,
    )
    assert len(out) == 1
    assert out.iloc[0]["semantic_role"] == "presentation"
    assert out.iloc[0]["classification_status"] == "ok"


def test_paired_path_still_requires_matching_shapes():
    import pytest
    z = np.zeros((6, 1), dtype=np.float32)
    names = pd.DataFrame({"feature_id": [0], "concept": ["c"]})
    with pytest.raises(ValueError):
        classify_response_roles(_battles(), z, np.zeros((5, 1), np.float32), names, _Client())
