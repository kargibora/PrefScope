import numpy as np

from prefscope.activations.extract import filter_outlier_rows, build_chat_inputs


def test_filter_outlier_rows_drops_high_norm():
    # 20 unit-norm rows + one large outlier; mean is not dominated by the outlier,
    # so the 6x-mean threshold drops it (the paper's per-span outlier filter).
    rows = [[1.0, 0.0]] * 20 + [[100.0, 100.0]]
    v = np.array(rows, dtype=np.float32)
    keep = filter_outlier_rows(v, mult=6.0)
    assert keep.dtype == bool
    assert keep[:20].all()          # the 20 normal rows are kept
    assert not keep[20]             # the outlier is dropped


def test_filter_outlier_rows_keeps_all_when_uniform():
    v = np.ones((5, 3), dtype=np.float32)
    assert filter_outlier_rows(v, mult=6.0).all()


class _StubTokenizer:
    """Minimal prefix-consistent chat template with one assistant-header token."""

    def apply_chat_template(self, messages, add_generation_prompt=False,
                            tokenize=True, **kw):
        ids = list(range(1, len(messages[0]["content"].split()) + 1))
        if len(messages) == 2:
            ids += [999]
            ids += list(range(100, 100 + len(messages[1]["content"].split())))
        elif add_generation_prompt:
            ids += [999]
        return ids


def test_build_chat_inputs_boundary():
    tok = _StubTokenizer()
    out = build_chat_inputs(tok, "hello there world", "good answer here", max_tokens=128)
    assert out["resp_start"] == 4
    assert out["resp_end"] == out["resp_start"] + 3
    assert len(out["input_ids"]) == out["resp_end"]
    assert out["input_ids"].count(999) == 1


def test_build_chat_inputs_truncates_to_max_tokens():
    tok = _StubTokenizer()
    out = build_chat_inputs(tok, "a b c d e", "f g h i j", max_tokens=6)
    assert len(out["input_ids"]) == 6
    assert out["resp_end"] <= 6


def test_build_chat_inputs_rejects_non_prefix_tokenizer():
    import pytest

    class _NonPrefixTokenizer:
        def apply_chat_template(self, messages, add_generation_prompt=False,
                                tokenize=True, **kw):
            # full chat reorders the user content, breaking the prefix property
            if len(messages) == 1:
                ids = [1, 2, 3]
                if add_generation_prompt:
                    ids += [999]
                return ids
            return [7, 8, 9, 4, 5, 6]  # first 3 tokens != the prompt-only [1,2,3]

    with pytest.raises(ValueError):
        build_chat_inputs(_NonPrefixTokenizer(), "a b c", "d e f", max_tokens=128)


def test_build_chat_inputs_rejects_generation_prefix_mismatch():
    import pytest

    class _MismatchedAssistantHeader:
        def apply_chat_template(self, messages, add_generation_prompt=False,
                                tokenize=True, **kw):
            if len(messages) == 1:
                return [1, 2, 9] if add_generation_prompt else [1, 2]
            return [1, 2, 8, 3, 4]

    with pytest.raises(ValueError, match="generation-prefix-consistent"):
        build_chat_inputs(_MismatchedAssistantHeader(), "a", "b", max_tokens=128)
