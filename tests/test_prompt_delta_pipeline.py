"""run_prompt_conditioned_delta (the `prefscope conditional-delta` core): orients the
completion lens by human_pref, conditions on the prompt concept, writes Δ_{k,f} (+ the
length-controlled conditional δ_{f,k})."""
import json

import numpy as np
import pandas as pd
import pytest

from prefscope.artifacts import BATTLES, Z_DIFF, Z_PROMPT
from prefscope.pipeline.prompt_delta import run_prompt_conditioned_delta


def _corpus(path, ids, rng):
    n = len(ids)
    pd.DataFrame({
        "battle_id": ids, "source": ["t"] * n, "language": ["en"] * n,
        "prompt": [f"p-{i}" for i in range(n)],
        "model_a": ["A"] * n, "model_b": ["B"] * n,
        "completion_a": ["a a a a a"] * n, "completion_b": ["b b b"] * n,
        "human_pref": rng.choice([0.0, 1.0], n),
    }).to_parquet(path)


def test_run_prompt_conditioned_delta_writes_delta_and_conditional(tmp_path):
    rng = np.random.default_rng(0)
    N, Mc = 500, 8
    ids = [str(i) for i in range(N)]

    # IMBALANCED prompt concepts so the conditional path actually computes: concept 0
    # gets 400 battles (>= conditional_win_relevance min_battles=300), concept 1 gets 100.
    zp = np.zeros((N, 2), dtype=np.float32)
    zp[:400, 0] = rng.random(400).astype(np.float32) + 5.0
    zp[400:, 1] = rng.random(100).astype(np.float32) + 5.0

    clens, plens = tmp_path / "c", tmp_path / "p"
    clens.mkdir()
    plens.mkdir()
    (clens / "manifest.json").write_text(json.dumps({"input_rep": "individual"}))
    np.save(clens / Z_DIFF, rng.standard_normal((N, Mc)).astype(np.float32))
    np.save(plens / Z_PROMPT, zp)
    pd.DataFrame({"battle_id": ids}).to_parquet(clens / BATTLES)
    pd.DataFrame({"battle_id": ids}).to_parquet(plens / BATTLES)
    corpus = tmp_path / "corpus.parquet"
    _corpus(corpus, ids, rng)

    out, cond = tmp_path / "delta.csv", tmp_path / "cond.csv"
    # permute=2 also exercises the fork-based permutation-null path
    run_prompt_conditioned_delta(clens, plens, out, corpus=str(corpus),
                                 conditional_out=str(cond), permute=2, jobs=1,
                                 log=lambda *_: None)

    d = pd.read_csv(out)
    assert {"prompt_concept", "completion_feature", "delta", "stable"} <= set(d.columns)
    assert len(d) > 0
    c = pd.read_csv(cond)
    assert {"prompt_concept", "feature_id", "delta_win_rate", "cond_significant"} <= set(c.columns)
    assert len(c) > 0                        # concept 0 cleared min_battles -> real rows
    assert (c["prompt_concept"] == 0).all()  # only concept 0 (400 battles) computed

    (clens / "manifest.json").write_text(json.dumps({"input_rep": "difference"}))
    with pytest.raises(ValueError, match="cannot be reversed"):
        run_prompt_conditioned_delta(
            clens, plens, tmp_path / "bad.csv", corpus=str(corpus), log=lambda *_: None)
