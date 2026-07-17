import pandas as pd

from prefscope.pipeline.inspect import summarize


def _battles():
    return pd.DataFrame([
        {"instruction_id": "1", "model_a": "X", "model_b": "Y", "y_judge": 1.0, "lang": "en"},
        {"instruction_id": "2", "model_a": "Y", "model_b": "X", "y_judge": 0.0, "lang": "en"},
        {"instruction_id": "3", "model_a": "X", "model_b": "Z", "y_judge": 0.5, "lang": "de"},
    ])


def test_summarize_counts():
    s = summarize(_battles())
    assert s["n_battles"] == 3
    assert s["n_models"] == 3
    assert s["model_counts"]["X"] == 3
    assert s["model_counts"]["Y"] == 2
    assert s["y_judge_dist"][0.5] == 1
    assert s["langs"]["en"] == 2


def test_summarize_label_free_corpus():
    # a build-corpus table: no y_judge, language column named "language"
    corpus = pd.DataFrame([
        {"battle_id": "a", "model_a": "X", "model_b": "Y", "language": "en"},
        {"battle_id": "b", "model_a": "Y", "model_b": "Z", "language": "de"},
    ])
    s = summarize(corpus)
    assert s["n_battles"] == 2 and s["n_models"] == 3
    assert "y_judge_dist" not in s          # no preference column -> omitted
    assert s["langs"]["en"] == 1
