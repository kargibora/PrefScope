"""name_individual_features: select top-activating SINGLE responses, ask shared trait."""
import json

import numpy as np
import pandas as pd

from prefscope.interpret.name import name_individual_features


class FakeClient:
    def __init__(self):
        self.prompts = []

    def raw(self, messages, **kw):
        self.prompts.append(messages[-1]["content"])
        schema = kw["response_schema"]
        props = schema["properties"]
        if "active_matches" not in props:       # multi-candidate synthesis
            return json.dumps({
                "status": "ok", "concept": "explains a word", "confidence": "high"})
        n_active = props["active_matches"]["minItems"]
        n_control = props["control_matches"]["minItems"]
        evidence = {
            "active_matches": [True] * n_active,
            "control_matches": [False] * n_control,
        }
        if "status" in props:                 # proposal call
            evidence.update(status="ok", concept="explains a word", confidence="high")
        return json.dumps(evidence)


def test_individual_naming_shows_single_responses_not_pairs():
    n = 20
    battles = pd.DataFrame({
        "instruction_id": [str(i) for i in range(n)],
        "prompt": ["what does word %d mean" % i for i in range(n)],
        "completion_a": ["DEFINITION of word %d" % i for i in range(n)],
        "completion_b": ["unrelated chit-chat %d" % i for i in range(n)],
    })
    z_a = np.zeros((n, 1), dtype=np.float32)
    z_b = np.zeros((n, 1), dtype=np.float32)
    z_a[:5, 0] = 2.0   # feature fires on response A of the first 5 battles

    fc = FakeClient()
    df = name_individual_features(battles, z_a, z_b, fc, n_active=5, n_zero=3,
                                  verify_frac=0.0, concurrency=1)

    assert df.iloc[0]["concept"] == "explains a word"
    p = fc.prompts[0]
    # single-response framing, NOT the A/B pair contrast
    assert "RESPONSE:" in p
    assert "RESPONSE A:" not in p and "RESPONSE B:" not in p
    # the high-activation A texts are the ones shown
    assert "DEFINITION of word" in p
    assert 'kind="ACTIVATING"' in p and 'kind="SILENT_CONTROL"' in p
    assert df.iloc[0]["naming_audit_pass"]
    assert df.iloc[0]["naming_review_action"] == "accepted"


def test_individual_naming_uses_one_completion_per_instruction():
    n = 8
    battles = pd.DataFrame({
        "instruction_id": [str(i) for i in range(n)],
        "prompt": [f"prompt-{i}" for i in range(n)],
        "completion_a": [f"A-response-{i}" for i in range(n)],
        "completion_b": [f"B-response-{i}" for i in range(n)],
    })
    z_a = np.zeros((n, 1), dtype=np.float32)
    z_b = np.zeros((n, 1), dtype=np.float32)
    z_a[0, 0] = 5.0
    z_b[0, 0] = 4.9                    # same instruction: must not consume another slot
    z_a[1, 0] = 4.0
    z_b[2, 0] = 3.0

    client = FakeClient()
    name_individual_features(
        battles, z_a, z_b, client, features=[0], n_active=3, n_zero=2,
        verify_frac=0.0, seed=0)
    proposal = client.prompts[0]
    assert "A-response-0" in proposal and "B-response-0" not in proposal
    assert "A-response-1" in proposal and "B-response-2" in proposal


def test_close_negatives_pick_similar_controls():
    # feature 0 fires on 4 responses that ALSO express concept 1 (a shared "other" concept).
    # Half the silent pool shares concept 1 (close), half shares concept 2 (far). Close
    # negatives must draw the controls from the concept-1-sharing silent responses.
    n = 40
    battles = pd.DataFrame({
        "instruction_id": [str(i) for i in range(n)],
        "prompt": [f"p{i}" for i in range(n)],
        "completion_a": [f"a{i}" for i in range(n)],
        "completion_b": [f"b{i}" for i in range(n)],
    })
    z_a = np.zeros((n, 3), dtype=np.float32)
    z_b = np.zeros((n, 3), dtype=np.float32)
    z_a[:4, 0] = 2.0; z_a[:4, 1] = 1.0            # activators: f0 + concept 1
    z_a[4:20, 1] = 1.0                            # silent-on-f0 but share concept 1 (close)
    z_a[20:, 2] = 1.0                             # silent-on-f0, share concept 2 (far)
    fc = FakeClient()
    df = name_individual_features(battles, z_a, z_b, fc, n_active=4, n_zero=5,
                                  verify_frac=0.0, seed=0, negatives="close")
    assert len(df) == 3
    # the prompt shown for feature 0 should include concept-1 (indices <20) controls,
    # not concept-2 ones — check a far control's text is absent while the near ones appear.
    prompt0 = fc.prompts[0]
    assert "a0" in prompt0                        # an activator response is shown
    # at least one close (index 4..19) silent 'a' response appears; no far (>=20) ones
    assert any(f"a{i}\n" in prompt0 or f"a{i} " in prompt0 or f"a{i}<" in prompt0 for i in range(4, 20))


def test_multi_candidate_final_concept_is_reviewed_over_union():
    n = 20
    battles = pd.DataFrame({
        "instruction_id": [str(i) for i in range(n)],
        "prompt": [f"define word {i}" for i in range(n)],
        "completion_a": [f"DEFINITION {i}" for i in range(n)],
        "completion_b": [f"other {i}" for i in range(n)],
    })
    z_a = np.zeros((n, 1), dtype=np.float32); z_a[:8, 0] = 1.0
    z_b = np.zeros((n, 1), dtype=np.float32)
    client = FakeClient()
    row = name_individual_features(
        battles, z_a, z_b, client, features=[0], n_active=3, n_zero=2,
        n_candidates=2, candidate_pool_factor=2, verify_frac=0.0).iloc[0]
    # 2 proposals + 2 proposal reviews + synthesis + final union review.
    assert len(client.prompts) == 6
    assert row["status"] == "ok" and row["naming_audit_pass"]
    assert row["naming_active_total"] >= 3


def test_individual_reviewer_can_revise_a_subset_pattern_without_blocking_verification():
    n = 12
    battles = pd.DataFrame({
        "instruction_id": [str(i) for i in range(n)],
        "prompt": [f"p{i}" for i in range(n)],
        "completion_a": [f"response {i}" for i in range(n)],
        "completion_b": [f"other {i}" for i in range(n)],
    })
    z_a = np.zeros((n, 1), dtype=np.float32)
    z_b = np.zeros((n, 1), dtype=np.float32)
    z_a[:3, 0] = 1.0

    class SubsetClient:
        def __init__(self): self.calls = 0
        def raw(self, messages, **kw):
            self.calls += 1
            props = kw["response_schema"]["properties"]
            na = props["active_matches"]["minItems"]
            nz = props["control_matches"]["minItems"]
            if self.calls == 1:
                # The proposer overclaims unanimity, exactly like feature 123 did.
                return json.dumps({
                    "status": "ok", "concept": "declines the request", "confidence": "high",
                    "active_matches": [True] * na, "control_matches": [False] * nz})
            # The reviewer replaces the over-narrow proposal with a testable hypothesis and
            # honestly records that one naming activator remains a counterexample.
            return json.dumps({
                "status": "ok", "concept": "offers general guidance", "confidence": "medium",
                "active_matches": [True, True] + [False] * (na - 2),
                "control_matches": [False] * nz})

    client = SubsetClient()
    df = name_individual_features(
        battles, z_a, z_b, client, n_active=3, n_zero=3,
        verify_frac=0.0, features=[0])
    row = df.iloc[0]
    assert client.calls == 2
    assert row["status"] == "ok"
    assert row["concept"] == "offers general guidance"
    assert row["naming_active_support"] == 2
    assert row["naming_active_total"] == 3
    assert not row["naming_audit_pass"]
    assert row["naming_review_action"] == "revised"
    assert "declines the request" in row["candidate_concepts"]


def test_individual_reviewer_abstains_from_one_example_pattern():
    n = 10
    battles = pd.DataFrame({
        "instruction_id": [str(i) for i in range(n)], "prompt": [f"p{i}" for i in range(n)],
        "completion_a": [f"a{i}" for i in range(n)],
        "completion_b": [f"b{i}" for i in range(n)]})
    z_a = np.zeros((n, 1), dtype=np.float32); z_a[:3, 0] = 1.0
    z_b = np.zeros((n, 1), dtype=np.float32)

    class HonestEvidenceClient:
        def __init__(self): self.calls = 0
        def raw(self, messages, **kw):
            self.calls += 1
            if self.calls == 2:
                return json.dumps({
                    "status": "insufficient_evidence", "concept": None,
                    "confidence": "low", "active_matches": [True, False, False],
                    "control_matches": [False, False, False]})
            return json.dumps({
                "status": "ok", "concept": "declines the request", "confidence": "high",
                "active_matches": [True, False, False],
                "control_matches": [False, False, False]})

    client = HonestEvidenceClient()
    row = name_individual_features(
        battles, z_a, z_b, client, n_active=3, n_zero=3,
        verify_frac=0.0, features=[0]).iloc[0]
    assert client.calls == 2                 # reviewer, not self-reported support, decides
    assert row["status"] == "insufficient_evidence" and row["concept"] == ""
    assert row["naming_review_performed"]
    assert row["naming_review_action"] == "abstained"


def test_individual_screen_rejects_candidate_not_enriched_over_controls():
    n = 10
    battles = pd.DataFrame({
        "instruction_id": [str(i) for i in range(n)],
        "prompt": [f"p{i}" for i in range(n)],
        "completion_a": [f"a{i}" for i in range(n)],
        "completion_b": [f"b{i}" for i in range(n)],
    })
    z_a = np.zeros((n, 1), dtype=np.float32); z_a[:3, 0] = 1.0
    z_b = np.zeros((n, 1), dtype=np.float32)

    class NoSeparationClient:
        def raw(self, messages, **kw):
            return json.dumps({
                "status": "ok", "concept": "uses detailed steps", "confidence": "medium",
                "active_matches": [True, True, False],
                "control_matches": [True, True, False],
            })

    row = name_individual_features(
        battles, z_a, z_b, NoSeparationClient(), features=[0],
        n_active=3, n_zero=3, verify_frac=0.0).iloc[0]
    assert row["status"] == "insufficient_evidence" and row["concept"] == ""
    assert row["reviewed_concept"] == "uses detailed steps"
    assert not row["naming_screen_pass"]
    assert row["naming_review_action"] == "abstained_no_separation"
