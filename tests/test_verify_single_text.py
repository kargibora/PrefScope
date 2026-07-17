import numpy as np
import pandas as pd
import pytest

from prefscope.interpret.prompts import parse_presence
from prefscope.interpret.verify import verify_single_text_features


def test_parse_presence():
    assert parse_presence("Yes") == 1
    assert parse_presence("yes, clearly") == 1
    assert parse_presence("Present") == 1
    assert parse_presence("No") == 0
    assert parse_presence("absent") == 0
    assert parse_presence("") is None           # empty -> MISSING, not a "No" vote
    assert parse_presence("hmmmm") is None      # unparseable -> MISSING
    assert parse_presence("I think yes") == 1   # falls back to the yes token


class _FakeClient:
    """Says 'Yes' iff the text shown in the prompt contains the marker."""
    def raw(self, messages, **kwargs):
        return "Yes" if "ZQX" in " ".join(m["content"] for m in messages) else "No"


def test_verify_single_text_passes_faithful_feature():
    # feature 0 fires (z=2) exactly on the 5 texts containing the marker.
    texts = [f"ZQX sample {i}" for i in range(5)] + \
            [f"absent sample {i}" for i in range(5, 20)]
    z = np.zeros((20, 2), dtype=np.float32)
    z[:5, 0] = 2.0                       # feature 0: faithful (fires on marked texts)
    # feature 1 never fires -> a degenerate, non-passing axis
    names = pd.DataFrame({"feature_id": [0, 1], "concept": ["uses a special marker"] * 2})

    out = verify_single_text_features(
        texts, z, names, _FakeClient(),
        n_active=5, n_zero=5, verify_frac=1.0, seed=0)

    f0 = out.set_index("feature_id").loc[0]
    assert f0["agreement"] == 1.0
    assert f0["correlation"] == 1.0
    assert bool(f0["fidelity_pass"]) is True

    f1 = out.set_index("feature_id").loc[1]
    assert bool(f1["fidelity_pass"]) is False   # never fires -> can't be verified


def test_single_text_stratified_random_respects_total_example_budget():
    texts = [f"ZQX active {i}" for i in range(30)] + [f"silent {i}" for i in range(30)]
    z = np.zeros((60, 1), dtype=np.float32)
    z[:30, 0] = np.linspace(0.1, 3.0, 30)
    names = pd.DataFrame({"feature_id": [0], "concept": ["uses a marker"]})
    out = verify_single_text_features(
        texts, z, names, _FakeClient(), verify_frac=1.0, seed=0,
        sampling="stratified-random", n_examples=20)
    row = out.iloc[0]
    assert row["n_attempted"] == 20
    assert row["n_pos_ok"] == 10 and row["n_neg_ok"] == 10
    assert bool(row["fidelity_pass"]) is True


class _FlakyClient:
    """Raises on every 3rd call; otherwise says 'Yes' iff the marker is present."""
    def __init__(self):
        self.n = 0
    def raw(self, messages, **kwargs):
        self.n += 1
        if self.n % 3 == 0:
            raise RuntimeError("simulated API failure")
        return "Yes" if "ZQX" in " ".join(m["content"] for m in messages) else "No"


def test_verify_treats_failures_as_missing_not_negative():
    texts = [f"ZQX sample {i}" for i in range(5)] + [f"absent {i}" for i in range(5, 15)]
    z = np.zeros((15, 1), dtype=np.float32); z[:5, 0] = 2.0
    names = pd.DataFrame({"feature_id": [0], "concept": ["uses a special marker"]})
    out = verify_single_text_features(texts, z, names, _FlakyClient(),
                                      n_active=5, n_zero=5, verify_frac=1.0, seed=0)
    row = out.iloc[0]
    # failures are dropped, not scored: n (used) < n_attempted, and they're reported
    assert row["n_failed"] > 0
    assert row["n"] == row["n_attempted"] - row["n_failed"]
    assert 0.0 < row["success_rate"] < 1.0
    # the surviving observations are still perfectly faithful (failures didn't add noise)
    assert row["correlation"] == pytest.approx(1.0)


def test_verify_skips_abstained_names_and_excludes_from_bonferroni():
    texts = [f"ZQX sample {i}" for i in range(5)] + [f"absent {i}" for i in range(5, 15)]
    z = np.zeros((15, 2), dtype=np.float32); z[:5, 0] = 2.0; z[:5, 1] = 2.0
    names = pd.DataFrame({
        "feature_id": [0, 1],
        "concept": ["uses a special marker", ""],      # feature 1 abstained
        "status": ["ok", "polysemantic"],
    })
    out = verify_single_text_features(texts, z, names, _FakeClient(),
                                      n_active=5, n_zero=5, verify_frac=1.0, seed=0)
    o = out.set_index("feature_id")
    # tested feature passes; abstained feature is re-attached as non-passing, never verified
    assert bool(o.loc[0]["fidelity_pass"]) is True
    assert bool(o.loc[1]["fidelity_pass"]) is False
    assert o.loc[1]["skipped_reason"] == "polysemantic"
    assert np.isnan(o.loc[1]["correlation"])           # LLM never asked about it
    # Bonferroni over TESTED concepts only (m=1), so the tested feature still reaches p<0.05
    assert o.loc[0]["p_bonferroni"] < 0.05


def test_low_success_rate_fails_the_fidelity_gate():
    texts = [f"ZQX {i}" for i in range(8)] + [f"absent {i}" for i in range(8, 24)]
    z = np.zeros((24, 1), dtype=np.float32); z[:8, 0] = 2.0
    names = pd.DataFrame({"feature_id": [0], "concept": ["uses a special marker"]})
    out = verify_single_text_features(texts, z, names, _FlakyClient(),
                                      n_active=8, n_zero=8, verify_frac=1.0, seed=0)
    row = out.iloc[0]
    assert row["success_rate"] < 0.8
    # a few surviving aligned judgments must NOT pass when most annotations failed
    assert bool(row["fidelity_pass"]) is False


def test_verify_single_text_flags_nondiscriminating_feature():
    # every text is marked, so the LLM always says 'Yes' regardless of the SAE:
    # no correlation between activation and presence -> not faithful.
    texts = [f"ZQX sample {i}" for i in range(20)]
    z = np.zeros((20, 1), dtype=np.float32)
    z[:5, 0] = 2.0                       # feature fires on 5 of the 20
    names = pd.DataFrame({"feature_id": [0], "concept": ["uses a special marker"]})

    out = verify_single_text_features(
        texts, z, names, _FakeClient(),
        n_active=5, n_zero=5, verify_frac=1.0, seed=0)
    assert bool(out.iloc[0]["fidelity_pass"]) is False


def test_similar_negatives_require_embeddings():
    texts = ["ZQX a", "ZQX b", "plain c", "plain d"]
    z = np.zeros((4, 1), dtype=np.float32); z[:2, 0] = 1.0
    names = pd.DataFrame({"feature_id": [0], "concept": ["uses a marker"]})
    with pytest.raises(ValueError):
        verify_single_text_features(texts, z, names, _FakeClient(),
                                    negatives="similar", n_active=2, n_zero=2,
                                    verify_frac=1.0, seed=0)


def test_close_negatives_expose_overbroad_feature():
    # feature 0 fires on 5 "ZQX" texts; its (over-broad) concept also matches 5
    # SILENT "ZQX" texts (close negatives) + 20 plain silent texts.
    texts = ([f"ZQX active {i}" for i in range(5)]
             + [f"ZQX silent {i}" for i in range(5)]
             + [f"plain silent {i}" for i in range(20)])
    z = np.zeros((30, 1), dtype=np.float32); z[:5, 0] = 2.0
    E = np.zeros((30, 2), dtype=np.float32)
    E[:5] = [1.0, 0.0]; E[5:10] = [1.0, 0.05]      # ZQX-silent near the active set
    E[10:] = [0.0, 1.0]                            # plain-silent far away
    names = pd.DataFrame({"feature_id": [0], "concept": ["uses a special marker"]})

    sim = verify_single_text_features(texts, z, names, _FakeClient(), embeddings=E,
                                      negatives="similar", n_active=5, n_zero=5,
                                      verify_frac=1.0, seed=0)
    rnd = verify_single_text_features(texts, z, names, _FakeClient(),
                                      negatives="random", n_active=5, n_zero=5,
                                      verify_frac=1.0, seed=0)
    assert sim.iloc[0]["fp_rate"] == 1.0                  # all close negatives say "present"
    assert rnd.iloc[0]["fp_rate"] < 1.0                   # random negatives mostly don't
    assert bool(sim.iloc[0]["fidelity_pass"]) is False    # over-broad feature fails under close negatives
