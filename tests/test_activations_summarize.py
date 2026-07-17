import numpy as np
import pandas as pd
import pytest

from prefscope.activations.summarize import reduce_span_summaries, summarize_spans


def test_reduce_span_summaries_max_and_freq():
    # 2 features, tokens grouped by (battle_id, span)
    codes = np.array([
        [0.0, 2.0],   # battle x / prompt, tok0
        [1.0, 0.0],   # battle x / prompt, tok1
        [3.0, 0.0],   # battle x / a, tok0
    ], dtype=np.float32)
    index = pd.DataFrame({
        "battle_id": ["x", "x", "x"],
        "span": ["prompt", "prompt", "a"],
        "token_idx": [0, 1, 0],
    })
    df = reduce_span_summaries(codes, index)
    xp = df[(df.battle_id == "x") & (df.span == "prompt")].set_index("feature_id")
    assert xp.loc[0, "x_max"] == 1.0 and xp.loc[0, "x_freq"] == 0.5   # f0 fires on 1/2 tokens
    assert xp.loc[1, "x_max"] == 2.0 and xp.loc[1, "x_freq"] == 0.5
    xa = df[(df.battle_id == "x") & (df.span == "a")].set_index("feature_id")
    assert xa.loc[0, "x_max"] == 3.0 and xa.loc[0, "x_freq"] == 1.0
    assert 1 not in xa.index   # f1 never fires on span a -> dropped


# ---------------------------------------------------------------------------
# End-to-end test for summarize_spans using a fake projector / cache stub.
# Pure NumPy / pandas — no torch required.
# ---------------------------------------------------------------------------

class _FakeProjector:
    """Identity projector: project(x) returns x unchanged (hidden == M == 2)."""
    def __init__(self):
        self.m_total = 2

    def project(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.float32)


class _FakeCache:
    """Minimal cache-like object with .index DataFrame and .acts ndarray."""
    def __init__(self, acts: np.ndarray, index: pd.DataFrame):
        self.acts = acts
        self.index = index


def test_summarize_spans_end_to_end():
    # 4 tokens: 2 in (battle "b1", span "prompt"), 2 in (battle "b1", span "a")
    acts = np.array([
        [0.0, 5.0],   # b1/prompt tok0  — feature 1 fires
        [2.0, 0.0],   # b1/prompt tok1  — feature 0 fires
        [4.0, 0.0],   # b1/a     tok0  — feature 0 fires
        [0.0, 0.0],   # b1/a     tok1  — nothing fires
    ], dtype=np.float32)
    index = pd.DataFrame({
        "battle_id": ["b1", "b1", "b1", "b1"],
        "span":      ["prompt", "prompt", "a", "a"],
        "token_idx": [0, 1, 0, 1],
    })
    cache = _FakeCache(acts, index)
    projector = _FakeProjector()

    summaries, span_meta = summarize_spans(cache, projector, batch=8192)

    # --- span_meta checks ---
    meta = span_meta.set_index(["battle_id", "span"])
    assert meta.loc[("b1", "prompt"), "n_tokens"] == 2
    assert meta.loc[("b1", "a"), "n_tokens"] == 2

    # --- summaries checks for b1/prompt ---
    sp = summaries[(summaries.battle_id == "b1") & (summaries.span == "prompt")].set_index("feature_id")
    assert sp.loc[0, "x_max"] == 2.0
    assert sp.loc[0, "x_freq"] == pytest.approx(0.5)
    assert sp.loc[1, "x_max"] == 5.0
    assert sp.loc[1, "x_freq"] == pytest.approx(0.5)

    # --- summaries checks for b1/a ---
    sa = summaries[(summaries.battle_id == "b1") & (summaries.span == "a")].set_index("feature_id")
    assert sa.loc[0, "x_max"] == 4.0
    assert sa.loc[0, "x_freq"] == pytest.approx(0.5)   # fires on 1 of 2 tokens
    assert 1 not in sa.index   # feature 1 never fires on span a


def test_summarize_spans_batch_boundary():
    """Flushing mid-stream (batch < n_tokens) must produce same result as large batch."""
    acts = np.array([
        [1.0, 0.0],
        [0.0, 3.0],
        [2.0, 0.0],
        [0.0, 0.0],
    ], dtype=np.float32)
    index = pd.DataFrame({
        "battle_id": ["x", "x", "y", "y"],
        "span":      ["s", "s", "s", "s"],
        "token_idx": [0, 1, 0, 1],
    })
    cache = _FakeCache(acts, index)
    projector = _FakeProjector()

    big, meta_big = summarize_spans(cache, projector, batch=8192)
    small, meta_small = summarize_spans(cache, projector, batch=2)   # forces mid-buffer flush

    # span_meta must be identical (order may differ — sort to compare)
    assert set(zip(meta_big.battle_id, meta_big.span, meta_big.n_tokens)) == \
           set(zip(meta_small.battle_id, meta_small.span, meta_small.n_tokens))

    # summaries must have the same rows (sort by all columns to compare)
    sort_cols = ["battle_id", "span", "feature_id"]
    big_sorted   = big.sort_values(sort_cols).reset_index(drop=True)
    small_sorted = small.sort_values(sort_cols).reset_index(drop=True)
    pd.testing.assert_frame_equal(big_sorted, small_sorted, check_like=False)
