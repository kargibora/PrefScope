import numpy as np
import pandas as pd

from prefscope.interpret.name import name_features


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
