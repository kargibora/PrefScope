# tests/test_analysis_run.py
import numpy as np
import pandas as pd
import pytest

from prefscope.analysis import diagnose, feature_preference_relevance


def _codes_meta():
    # 4 battles, 2 features. Feature 0: self over-expresses (z>0) on the 2 wins.
    codes = np.array([[ 1.0, 0.0],
                      [ 1.0, 0.0],
                      [-1.0, 0.0],
                      [-1.0, 0.0]], dtype=np.float32)
    meta = pd.DataFrame({"pref": [1.0, 1.0, 0.0, 0.0]})   # P(self preferred)
    return codes, meta


def test_diagnose_basic_columns_and_direction():
    codes, meta = _codes_meta()
    df = diagnose(codes, meta)
    assert {"feature_id", "net_direction", "outcome_assoc", "win_rate"} <= set(df.columns)
    f0 = df.set_index("feature_id").loc[0]
    # feature 0: +1 on two rows, -1 on two -> net_direction = 0.5 - 0.5 = 0.0
    assert f0["net_direction"] == 0.0
    # self over-expresses (z>0) exactly on the wins -> outcome_assoc = 1.0 - 0.0
    assert f0["outcome_assoc"] == 1.0
    assert df.iloc[0]["net_direction"] >= df.iloc[-1]["net_direction"]   # sorted desc


def test_diagnose_length_mismatch_raises():
    codes, meta = _codes_meta()
    with pytest.raises(ValueError, match="length mismatch"):
        diagnose(codes, meta.iloc[:3])


def test_diagnose_requires_pref():
    codes, _ = _codes_meta()
    with pytest.raises(ValueError, match="pref"):
        diagnose(codes, pd.DataFrame({"x": [1, 2, 3, 4]}))


def test_diagnose_attaches_names_and_fidelity_filters():
    codes, meta = _codes_meta()
    names = pd.DataFrame({"feature_id": [0, 1], "concept": ["clarity", "verbosity"],
                          "fidelity_pass": [True, False]})
    df = diagnose(codes, meta, names=names, fidelity_only=True)
    assert list(df["feature_id"]) == [0]            # only the fidelity-passing feature
    assert df.iloc[0]["concept"] == "clarity"


def test_feature_preference_relevance_runs():
    codes, meta = _codes_meta()
    df = feature_preference_relevance(codes, meta)
    assert "win_assoc" in df.columns and "feature_id" in df.columns
    assert len(df) == 2
