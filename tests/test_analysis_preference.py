# tests/test_analysis_preference.py
import numpy as np
import pandas as pd
import pytest

from prefscope.analysis import evaluate_preference


def _signal_dataset(n=240, m=5, seed=0):
    """Feature 0's sign determines the preference; others are noise."""
    rng = np.random.default_rng(seed)
    codes = rng.normal(size=(n, m)).astype(np.float32)
    pref = (codes[:, 0] > 0).astype(float)        # P(A preferred) in {0,1}
    return codes, pd.DataFrame({"pref": pref})


def test_evaluate_recovers_signal_above_baseline():
    codes, meta = _signal_dataset()
    out = evaluate_preference(codes, meta, seed=0)
    assert out["accuracy"] > 0.85
    assert out["accuracy"] > out["baseline_accuracy"]
    assert 0.0 <= out["auc"] <= 1.0 and out["auc"] > 0.85
    assert int(out["top_features"].iloc[0]["feature_id"]) == 0   # signal feature ranked first


def test_evaluate_drops_ties():
    codes, meta = _signal_dataset(n=240)
    meta = meta.copy()
    meta.loc[:19, "pref"] = 0.5                  # 20 ties
    out = evaluate_preference(codes, meta, seed=0)
    assert out["n"] == 220                        # ties excluded


def test_evaluate_single_class_raises():
    codes = np.random.default_rng(0).normal(size=(30, 4)).astype(np.float32)
    meta = pd.DataFrame({"pref": [1.0] * 30})    # only one class
    with pytest.raises(ValueError, match="both"):
        evaluate_preference(codes, meta)


def test_evaluate_requires_pref():
    codes = np.zeros((10, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="pref"):
        evaluate_preference(codes, pd.DataFrame({"x": range(10)}))
