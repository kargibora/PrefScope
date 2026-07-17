"""run_elicitation (the `prefscope elicit` core): loads two lenses, aligns, restricts
to verified axes, returns the co-activation lift edge table."""
import numpy as np
import pandas as pd
import pytest

from prefscope.artifacts import BATTLES, Z_A, Z_B, Z_PROMPT
from prefscope.pipeline.elicit import run_elicitation


def _lens(d, arrays, ids):
    d.mkdir(parents=True, exist_ok=True)
    for name, arr in arrays.items():
        np.save(d / name, arr)
    pd.DataFrame({"battle_id": [str(i) for i in ids]}).to_parquet(d / BATTLES)


def test_run_elicitation_aligns_and_returns_edges(tmp_path):
    rng = np.random.default_rng(0)
    N, Mc, Mp = 300, 10, 6
    za = (rng.random((N, Mc)) * (rng.random((N, Mc)) < 0.2)).astype(np.float32)
    zb = (rng.random((N, Mc)) * (rng.random((N, Mc)) < 0.2)).astype(np.float32)
    zp = (rng.random((N, Mp)) * (rng.random((N, Mp)) < 0.3)).astype(np.float32)
    # inject prompt-feature 1 -> response-feature 2 co-occurrence
    zb[:, 2] = np.where((zp[:, 1] > 0) & (rng.random(N) < 0.7), 1.0, zb[:, 2]).astype(np.float32)

    clens, plens = tmp_path / "c", tmp_path / "p"
    _lens(clens, {Z_A: za, Z_B: zb}, range(N))
    _lens(plens, {Z_PROMPT: zp}, range(N))

    edges = run_elicitation(clens, plens, min_support=20, min_cooccur=5, log=lambda *_: None)
    assert {"prompt_feature", "completion_feature", "lift", "significant"} <= set(edges.columns)
    hit = edges[(edges.prompt_feature == 1) & (edges.completion_feature == 2)]
    assert len(hit) == 1 and hit["lift"].iloc[0] > 1.5   # the injected elicitation surfaces


def test_run_elicitation_rejects_difference_lens(tmp_path):
    # a difference lens has z_diff but no z_a/z_b
    d = tmp_path / "diff"
    _lens(d, {"z_diff.npy": np.zeros((5, 4), np.float32)}, range(5))
    p = tmp_path / "p"
    _lens(p, {Z_PROMPT: np.zeros((5, 4), np.float32)}, range(5))
    with pytest.raises(ValueError, match="INDIVIDUAL"):
        run_elicitation(d, p)
