import numpy as np
import pandas as pd

from prefscope.interpret.name import name_features
from prefscope.interpret.prompt_name import (
    _sample_group_representatives, name_prompt_features,
)


class FakeClient:
    """Returns a WIMHF-style quoted concept; echoes 'short concept' if asked to abbreviate."""
    def __init__(self): self.calls = 0
    def raw(self, messages, **kw):
        self.calls += 1
        text = messages[-1]["content"]
        if "abbreviate" in text.lower():
            return '"short concept"'
        return '- "uses code blocks"'


def _battles(n=60):
    return pd.DataFrame({
        "instruction_id": [str(i) for i in range(n)],
        "prompt": [f"p{i}" for i in range(n)],
        "completion_a": [f"a{i}" for i in range(n)],
        "completion_b": [f"b{i}" for i in range(n)],
    })


def _zdiff(n=60, m=3, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, m)).astype(np.float32)
    z[np.abs(z) < 0.3] = 0.0
    return z


def test_name_features_returns_plain_concept_per_feature():
    df = name_features(_battles(), _zdiff(), FakeClient(),
                       n_active=5, n_zero=5, verify_frac=0.2, seed=0)
    assert list(df["feature_id"]) == [0, 1, 2]
    assert (df["concept"] == "uses code blocks").all()
    assert "concept_abbrev" in df.columns
    # abstention support: a plain {"concept": ...} response parses as status "ok"
    assert (df["status"] == "ok").all()
    assert "confidence" in df.columns


def test_name_features_abbreviates_when_enabled():
    client = FakeClient()
    df = name_features(_battles(), _zdiff(), client, abbreviate=True,
                       n_active=5, n_zero=5, seed=0)
    assert (df["concept_abbrev"] == "short concept").all()


def test_name_features_respects_feature_subset():
    df = name_features(_battles(), _zdiff(m=4), FakeClient(), features=[1, 3],
                       n_active=5, n_zero=5, seed=0)
    assert list(df["feature_id"]) == [1, 3]


def test_name_features_multi_candidate_synthesizes_one_row_per_feature():
    client = FakeClient()
    df = name_features(_battles(), _zdiff(m=1), client, features=[0],
                       n_active=4, n_zero=4, n_candidates=3,
                       candidate_pool_factor=2, seed=0)
    assert len(df) == 1 and df.iloc[0]["n_candidates"] == 3
    assert df.iloc[0]["concept"] == "uses code blocks"
    assert "uses code blocks" in df.iloc[0]["candidate_concepts"]
    assert client.calls == 4  # three independent proposals + one synthesis


def test_name_features_concurrency_matches_sequential():
    """Concurrency must not change results or order (deterministic per-feature rng)."""
    import numpy as np
    import pandas as pd
    from prefscope.interpret.name import name_features

    rng = np.random.default_rng(0)
    z = rng.standard_normal((40, 6)).astype("float32")
    battles = pd.DataFrame({
        "instruction_id": [str(i) for i in range(40)],
        "prompt": [f"p{i}" for i in range(40)],
        "completion_a": [f"a{i}" for i in range(40)],
        "completion_b": [f"b{i}" for i in range(40)],
    })

    class Client:
        def raw(self, messages, **kw):
            return '- "concept"'

    seq = name_features(battles, z, Client(), concurrency=1)
    par = name_features(battles, z, Client(), concurrency=4)
    pd.testing.assert_frame_equal(seq, par)
    assert list(seq["feature_id"]) == list(range(6))


def test_name_features_keeps_descriptive_polysemantic_label_without_abbreviating():
    class MixedClient:
        def __init__(self): self.calls = 0
        def raw(self, messages, **kw):
            self.calls += 1
            return ('{"status":"polysemantic",'
                    '"concept":"database queries; numbered formatting",'
                    '"confidence":"medium"}')

    client = MixedClient()
    df = name_features(_battles(), _zdiff(m=1), client, features=[0],
                       n_active=5, n_zero=5, seed=0, abbreviate=True)
    row = df.iloc[0]
    assert row["status"] == "polysemantic"
    assert row["concept"] == "database queries; numbered formatting"
    assert row["concept_abbrev"] == ""
    assert client.calls == 1


def test_name_features_synthesis_failure_prefers_mixed_over_ok_candidate():
    class FailingSynthesisClient:
        def __init__(self): self.calls = 0
        def raw(self, messages, **kw):
            self.calls += 1
            if self.calls == 1:
                return ('{"status":"ok","concept":"uses numbered formatting",'
                        '"confidence":"medium"}')
            if self.calls == 2:
                return ('{"status":"polysemantic",'
                        '"concept":"database queries; numbered formatting",'
                        '"confidence":"medium"}')
            raise RuntimeError("synthesis failed")

    row = name_features(
        _battles(), _zdiff(m=1), FailingSynthesisClient(), features=[0],
        n_active=4, n_zero=4, n_candidates=2, seed=0).iloc[0]
    assert row["status"] == "polysemantic"
    assert row["concept"] == "database queries; numbered formatting"


def test_prompt_namer_uses_broad_semantics_and_keeps_mixed_fallback():
    prompts = [f"prompt {i}" for i in range(12)]
    z = np.zeros((12, 1), dtype=np.float32)
    z[:4, 0] = 1.0

    class FailingSynthesisClient:
        def __init__(self):
            self.calls = 0
            self.messages = []
        def raw(self, messages, **kw):
            self.calls += 1
            self.messages.extend(message["content"] for message in messages)
            if self.calls == 3:
                return "{malformed synthesis"
            return ('{"status":"polysemantic",'
                    '"concept":"translation requests; legal questions",'
                    '"confidence":"medium"}')

    client = FailingSynthesisClient()
    row = name_prompt_features(
        prompts, z, client, features=[0], n_active=3, n_zero=3,
        verify_frac=0.0, n_candidates=2, seed=0).iloc[0]
    instructions = " ".join(client.messages).lower()
    assert "audience" in instructions and "constraint" in instructions
    assert "interaction" in instructions and "language" in instructions
    assert row["status"] == "polysemantic"
    assert row["concept"] == "translation requests; legal questions"


def test_prompt_namer_emits_visible_evidence_summary():
    prompts = ["write Python code", "implement a Python function", "explain Python", "history"]
    z = np.asarray([[2.0], [1.5], [0.0], [0.0]], dtype=np.float32)

    class EvidenceClient:
        def __init__(self):
            self.prompt = ""
        def raw(self, messages, **kw):
            self.prompt = messages[-1]["content"]
            return ('{"status":"ok","concept":"asks for Python code",'
                    '"confidence":"high","evidence_summary":'
                    '"Both activators request Python implementations; controls do not."}')

    client = EvidenceClient()
    row = name_prompt_features(
        prompts, z, client, features=[0], n_active=2, n_zero=2,
        verify_frac=0.0, seed=0).iloc[0]
    assert row["evidence_summary"] == (
        "Both activators request Python implementations; controls do not."
    )
    assert "evidence_summary" in client.prompt
    assert "topic/domain" in client.prompt


def test_prompt_naming_uses_unique_source_groups_for_evidence():
    prompts = [
        "active source first", "active source duplicate one", "active source duplicate two",
        "active source second", "active source third",
        "active source silent sibling one", "active source silent sibling two",
        "mixed source active", "mixed source zero one", "mixed source zero two",
        "zero source first", "zero source duplicate", "zero source second", "zero source third",
    ]
    groups = ["a", "a", "a", "b", "c", "a", "a", "m", "m", "m", "z", "z", "y", "x"]
    z = np.asarray(
        [[9], [8], [7], [6], [5], [0], [0], [1], [0], [0], [0], [0], [0], [0]],
        dtype=np.float32,
    )

    class CaptureClient:
        def __init__(self): self.body = ""
        def raw(self, messages, **kw):
            self.body = messages[-1]["content"]
            return '{"status":"ok","concept":"tests source grouping","confidence":"high"}'

    for negatives in ("random", "close"):
        client = CaptureClient()
        name_prompt_features(
            prompts, z, client, features=[0], n_active=3, n_zero=3,
            verify_frac=0.0, instruction_ids=groups, negatives=negatives,
            cand_cap=3, seed=0)
        body = client.body
        assert "active source first" in body
        assert "active source second" in body and "active source third" in body
        assert "active source duplicate" not in body
        assert "active source silent sibling" not in body
        assert "mixed source zero" not in body
        assert sum(text in body for text in ("zero source first", "zero source duplicate")) == 1
        assert "zero source second" in body and "zero source third" in body


def test_prompt_control_sampling_is_uniform_over_groups_not_rows():
    groups = ["large"] * 100 + ["small-a", "small-b"]
    indices = np.arange(len(groups))
    counts = {group: 0 for group in set(groups)}
    for seed in range(300):
        selected = _sample_group_representatives(
            indices, groups, 1, np.random.default_rng(seed))
        counts[groups[int(selected[0])]] += 1

    # Row-weighted sampling would choose "large" about 294/300 times.
    assert counts["large"] < 150
    assert counts["small-a"] > 50 and counts["small-b"] > 50
