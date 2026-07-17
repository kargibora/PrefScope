"""Tests for the pure arena->corpus transform (no HF I/O)."""
import numpy as np
import pandas as pd

from prefscope.data.arenas import conversations_to_corpus


def _lmarena_frame():
    # lmarena 100k/140k shape: conversation_a/b are message lists.
    return pd.DataFrame({
        "model_a": ["gpt-x", "llama-y"],
        "model_b": ["claude-z", "qwen-w"],
        "language": ["English", "Russian"],
        "conversation_a": [
            [{"role": "user", "content": "Compare A and B"},
             {"role": "assistant", "content": "A is bigger."}],
            np.array([{"role": "user", "content": "Hi"},
                      {"role": "assistant", "content": "Hello there"}], dtype=object),
        ],
        "conversation_b": [
            [{"role": "user", "content": "Compare A and B"},
             {"role": "assistant", "content": "B is better."}],
            [{"role": "user", "content": "Hi"},
             {"role": "assistant", "content": "Salut"}],
        ],
    })


def test_lmarena_extracts_prompt_and_responses():
    out = conversations_to_corpus(_lmarena_frame(), "lmarena-140k")
    r0 = out.iloc[0]
    assert r0["prompt"] == "Compare A and B"
    assert r0["completion_a"] == "A is bigger."
    assert r0["completion_b"] == "B is better."
    assert r0["model_a"] == "gpt-x"
    assert r0["language"] == "English"
    assert (out["source"] == "lmarena-140k").all()
    # numpy-array conversation (row 1) handled too
    assert out.iloc[1]["completion_b"] == "Salut"


def test_comparia_prompt_from_opening_msg_and_named_models():
    df = pd.DataFrame({
        "model_a_name": ["mistral"],
        "model_b_name": ["gpt-fr"],
        "opening_msg": ["Quelle est la capitale ?"],
        "conversation_a": [[{"role": "user", "content": "Quelle est la capitale ?"},
                            {"role": "assistant", "content": "Paris."}]],
        "conversation_b": [[{"role": "user", "content": "Quelle est la capitale ?"},
                            {"role": "assistant", "content": "C'est Paris."}]],
    })
    out = conversations_to_corpus(df, "comparia", model_a="model_a_name",
                                  model_b="model_b_name", prompt_col="opening_msg")
    r = out.iloc[0]
    assert r["prompt"] == "Quelle est la capitale ?"
    assert r["completion_a"] == "Paris."
    assert r["completion_b"] == "C'est Paris."
    assert r["model_a"] == "mistral"
    assert r["language"] == ""          # comparia frame has no language column


def test_list_content_is_flattened_and_empty_dropped():
    df = pd.DataFrame({
        "model_a": ["m1", "m2"],
        "model_b": ["n1", "n2"],
        "conversation_a": [
            [{"role": "user", "content": "q"},
             {"role": "assistant", "content": [{"type": "text", "text": "part1"},
                                               {"type": "text", "text": "part2"}]}],
            [{"role": "user", "content": "q2"}],   # no assistant -> empty completion
        ],
        "conversation_b": [
            [{"role": "user", "content": "q"},
             {"role": "assistant", "content": "resp"}],
            [{"role": "assistant", "content": "only b"}],
        ],
    })
    out = conversations_to_corpus(df, "lmarena-100k")
    assert len(out) == 1                       # second row dropped (empty completion_a)
    assert out.iloc[0]["completion_a"] == "part1\npart2"


def test_human_pref_mapping():
    import pandas as pd
    from prefscope.data.arenas import human_pref
    lm = pd.DataFrame({"winner": ["model_a", "model_b", "tie", "tie (bothbad)"]})
    assert list(human_pref(lm, "lmarena-140k")) == [1.0, 0.0, 0.5, 0.5]
    cmp = pd.DataFrame({"model_a_name": ["m", "m", "m"],
                        "model_b_name": ["n", "n", "n"],
                        "chosen_model_name": ["m", "n", "m"],
                        "both_equal": [False, False, True]})
    assert list(human_pref(cmp, "comparia")) == [1.0, 0.0, 0.5]
