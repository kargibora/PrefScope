import numpy as np
import pandas as pd

from prefscope.pipeline.diagnose import diagnose_features, run_diagnose


def test_diagnose_features_basic_stats():
    # 4 battles, 2 features. z>0 = target over-expresses the concept.
    z = np.array([[2.0, 0.0],
                  [1.0, 0.0],
                  [-3.0, 0.0],
                  [0.0, 5.0]], dtype=np.float32)
    win = np.array([1.0, 0.5, 0.0, 1.0])
    df = diagnose_features(z, win).set_index("feature_id")

    f0 = df.loc[0]
    assert f0["n"] == 4
    assert f0["fire_rate"] == 0.75
    assert f0["self_more_rate"] == 0.5
    assert f0["self_less_rate"] == 0.25
    assert f0["net_direction"] == 0.25
    np.testing.assert_allclose(f0["mean_abs_z"], 1.5)
    np.testing.assert_allclose(f0["win_rate"], 0.625)
    np.testing.assert_allclose(f0["win_rate_self_more"], 0.75)
    np.testing.assert_allclose(f0["win_rate_self_less"], 0.0)
    np.testing.assert_allclose(f0["outcome_assoc"], 0.75)

    f1 = df.loc[1]
    assert f1["fire_rate"] == 0.25
    np.testing.assert_allclose(f1["win_rate_self_more"], 1.0)
    # only one side ever fires -> association undefined
    assert np.isnan(f1["win_rate_self_less"])
    assert np.isnan(f1["outcome_assoc"])


def test_diagnose_features_subset():
    z = np.zeros((3, 4), dtype=np.float32)
    z[:, 2] = [1.0, -1.0, 1.0]
    win = np.array([1.0, 0.0, 1.0])
    df = diagnose_features(z, win, features=[2])
    assert list(df["feature_id"]) == [2]
    assert df.iloc[0]["fire_rate"] == 1.0


class _FakeEmbedder:
    # first char ordinal -> 3-dim vector
    def encode(self, prompts, completions):
        return np.array([[float(ord(c[0])), 0.0, 0.0] for c in completions],
                        dtype=np.float32)


class _DiffProjector:
    """project(x) = x*2 ; difference lens projects the contrast vector itself."""
    def project(self, x):
        return np.asarray(x, dtype=np.float32) * 2.0


def _battles():
    return pd.DataFrame([
        {"instruction_id": "1", "model_a": "M", "model_b": "X",
         "prompt": "p", "completion_a": "m...", "completion_b": "x...",
         "y_judge": 1.0},
        {"instruction_id": "2", "model_a": "Z", "model_b": "M",
         "prompt": "p", "completion_a": "z...", "completion_b": "n...",
         "y_judge": 0.0},
    ])


def test_run_diagnose_difference_orientation():
    # battle 2: M is side B and y_judge=0 -> M preferred -> win
    df, summary = run_diagnose(_battles(), "M", _FakeEmbedder(), _DiffProjector(),
                               input_rep="difference")
    assert summary["model"] == "M"
    assert summary["n_battles"] == 2
    # both battles M wins -> win_rate 1.0
    np.testing.assert_allclose(summary["win_rate"], 1.0)
    assert "feature_id" in df.columns


def test_run_diagnose_names_filter_to_passing():
    names = pd.DataFrame({"feature_id": [0, 1, 2],
                          "concept": ["a", "b", "c"],
                          "concept_abbrev": ["", "", ""],
                          "fidelity_pass": [True, False, True]})
    df, _ = run_diagnose(_battles(), "M", _FakeEmbedder(), _DiffProjector(),
                         input_rep="difference", names=names)
    # only passing features kept, concept attached
    assert set(df["feature_id"]) == {0, 2}
    assert "concept" in df.columns


def test_run_diagnose_raises_when_model_absent():
    import pytest
    with pytest.raises(ValueError):
        run_diagnose(_battles(), "NOPE", _FakeEmbedder(), _DiffProjector(),
                     input_rep="difference")


def test_run_diagnose_returns_per_battle_evidence():
    df, summary, pb = run_diagnose(_battles(), "M", _FakeEmbedder(), _DiffProjector(),
                                   input_rep="difference", return_battles=True)
    assert len(pb) == 2
    # text + outcome + per-axis z columns present
    assert {"self_completion", "other_completion", "outcome", "win"} <= set(pb.columns)
    assert any(c.startswith("z") for c in pb.columns)
    assert "self_model" in pb.columns and (pb["self_model"] == "M").all()
