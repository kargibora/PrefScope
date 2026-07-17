import pandas as pd

from prefscope.data.orient import model_counts, orient_to_model


def _battles() -> pd.DataFrame:
    return pd.DataFrame([
        {"instruction_id": "1", "model_a": "M", "model_b": "X",
         "completion_a": "m1", "completion_b": "x1", "y_judge": 1.0,
         "len_a": 10, "len_b": 20},
        {"instruction_id": "2", "model_a": "M", "model_b": "Y",
         "completion_a": "m2", "completion_b": "y2", "y_judge": 0.0,
         "len_a": 11, "len_b": 21},
        {"instruction_id": "3", "model_a": "Z", "model_b": "M",
         "completion_a": "z3", "completion_b": "m3", "y_judge": 0.0,
         "len_a": 12, "len_b": 22},
        {"instruction_id": "4", "model_a": "Z", "model_b": "M",
         "completion_a": "z4", "completion_b": "m4", "y_judge": 0.5,
         "len_a": 13, "len_b": 23},
        {"instruction_id": "5", "model_a": "X", "model_b": "Y",
         "completion_a": "x5", "completion_b": "y5", "y_judge": 1.0,
         "len_a": 14, "len_b": 24},
    ])


def test_model_counts():
    counts = model_counts(_battles())
    assert counts["M"] == 4
    assert counts["X"] == 2


def test_orient_sign_and_outcome():
    out = orient_to_model(_battles(), "M").set_index("instruction_id")
    assert len(out) == 4  # battle 5 excluded
    assert out.loc["1", "sign"] == 1
    assert out.loc["1", "self_completion"] == "m1"
    assert out.loc["1", "other_completion"] == "x1"
    assert out.loc["1", "outcome"] == "win"
    assert out.loc["1", "self_len"] == 10
    assert out.loc["2", "outcome"] == "loss"
    assert out.loc["3", "sign"] == -1
    assert out.loc["3", "self_completion"] == "m3"
    assert out.loc["3", "other_completion"] == "z3"
    assert out.loc["3", "outcome"] == "win"
    assert out.loc["3", "self_len"] == 22
    assert out.loc["4", "outcome"] == "tie"


def test_orient_missing_model_returns_empty():
    out = orient_to_model(_battles(), "NOPE")
    assert out.empty


def test_orient_rejects_bad_y_judge():
    import pytest
    df = pd.DataFrame([
        {"instruction_id": "1", "model_a": "M", "model_b": "X",
         "completion_a": "m", "completion_b": "x", "y_judge": 0.3,
         "len_a": 1, "len_b": 1},
    ])
    with pytest.raises(ValueError):
        orient_to_model(df, "M")
