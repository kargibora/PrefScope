import numpy as np
import pandas as pd

from prefscope.interpret.verify import compute_metrics, verify_features


def test_compute_metrics_perfect_correlation():
    sae = np.array([1, 1, -1, -1, 0, 0])
    llm = np.array([1, 1, -1, -1, 0, 0])
    m = compute_metrics(sae, llm)
    assert m["agreement"] == 1.0
    assert m["correlation"] > 0.99


def _battles(n=80):
    return pd.DataFrame({
        "instruction_id": [str(i) for i in range(n)],
        "prompt": [f"p{i}" for i in range(n)],
        "completion_a": [f"a{i}" for i in range(n)],
        "completion_b": [f"b{i}" for i in range(n)],
    })


class SignClient:
    """Faithful mock: reads ONLY the prompt text, parses the battle id from
    'RESPONSE A:\\na{i}' and the feature id from the concept 'concept{f}', and
    returns the sign of z_diff[i, f] as A/B/Tie. No answer is injected."""
    def __init__(self, z_diff): self.z = z_diff
    def raw(self, messages, **kw):
        import re
        text = messages[-1]["content"]
        i = int(re.search(r"RESPONSE A:\s*a(\d+)", text).group(1))
        f = int(re.search(r"concept(\d+)", text).group(1))
        s = float(np.sign(self.z[i, f]))
        return "A" if s > 0 else ("B" if s < 0 else "Tie")


def test_verify_features_passes_when_llm_matches_sign():
    n, m = 80, 2
    rng = np.random.default_rng(0)
    z_diff = rng.standard_normal((n, m)).astype(np.float32)
    z_diff[np.abs(z_diff) < 0.3] = 0.0
    names = pd.DataFrame({"feature_id": [0, 1],
                          "concept": ["concept0", "concept1"]})
    df = verify_features(_battles(n), z_diff, names, SignClient(z_diff),
                         n_per_bucket=8, verify_frac=0.5, seed=0,
                         fidelity_threshold=0.5)
    assert set(df["feature_id"]) == {0, 1}
    assert df["fidelity_pass"].all()
    assert (df["p_bonferroni"] <= df["p_value"] * 2 + 1e-9).all()


class FlippedSignClient(SignClient):
    """Always answers the OPPOSITE side -> strong NEGATIVE correlation."""
    def raw(self, messages, **kw):
        out = super().raw(messages, **kw)
        return {"A": "B", "B": "A"}.get(out, "Tie")


def test_verify_fails_flipped_polarity():
    """A flipped-polarity name describes the OPPOSITE pole: strong NEGATIVE correlation.
    Fidelity requires a POSITIVE correlation, so such a name must FAIL — otherwise
    downstream 'more of concept X' silently means LESS of X."""
    n, m = 80, 2
    rng = np.random.default_rng(1)
    z_diff = rng.standard_normal((n, m)).astype(np.float32)
    z_diff[np.abs(z_diff) < 0.3] = 0.0
    names = pd.DataFrame({"feature_id": [0, 1], "concept": ["concept0", "concept1"]})
    df = verify_features(_battles(n), z_diff, names, FlippedSignClient(z_diff),
                         n_per_bucket=8, verify_frac=0.5, seed=0)
    # strong but NEGATIVE correlation -> sign recorded as -1, but it does NOT pass
    assert (df["correlation"] < 0).all()
    assert (df["sign"] == -1).all()
    assert not df["fidelity_pass"].any()


def test_verify_stratified_random_respects_total_example_budget():
    n = 300
    z = np.tile(np.array([[1.0], [-1.0], [0.0]], np.float32), (n // 3, 1))
    names = pd.DataFrame({"feature_id": [0], "concept": ["concept0"]})
    out = verify_features(
        _battles(n), z, names, SignClient(z), verify_frac=1.0, seed=0,
        sampling="stratified-random", n_examples=31)
    row = out.iloc[0]
    assert row["n_attempted"] == 31
    assert row["n_pos_ok"] == 11 and row["n_neg_ok"] == 10
    assert bool(row["fidelity_pass"]) is True
