import json

import numpy as np
import pandas as pd

from prefscope.viewer.data import (
    diagnosis_battles, feature_table, load_lens_for_view, top_battles,
)


def _z():
    # 4 battles, 2 features
    return np.array([[2.0, 0.0],
                     [-1.0, 0.0],
                     [0.0, 3.0],
                     [4.0, -5.0]], dtype=np.float32)


def _battles():
    return pd.DataFrame({
        "instruction_id": ["0", "1", "2", "3"],
        "prompt": ["p0", "p1", "p2", "p3"],
        "completion_a": ["a0", "a1", "a2", "a3"],
        "completion_b": ["b0", "b1", "b2", "b3"],
        "model_a": ["X", "Y", "X", "Y"],
        "model_b": ["Y", "X", "Y", "X"],
        "y_judge": [1.0, 0.0, 0.5, 1.0],
    })


def test_feature_table_stats_and_merge():
    names = pd.DataFrame({"feature_id": [0, 1], "concept": ["direct", "humor"],
                          "concept_abbrev": ["d", "h"]})
    fid = pd.DataFrame({"feature_id": [0, 1], "correlation": [0.7, 0.2],
                        "p_bonferroni": [0.01, 0.4], "fidelity_pass": [True, False]})
    df = feature_table(_z(), names=names, fidelity=fid).set_index("feature_id")
    assert df.loc[0, "fire_rate"] == 0.75            # 3 of 4 fire
    assert df.loc[0, "self_more_rate"] == 0.5         # 2,4 > 0
    assert df.loc[0, "self_less_rate"] == 0.25        # -1 < 0
    assert df.loc[0, "concept"] == "direct"
    assert bool(df.loc[0, "fidelity_pass"]) is True
    # concept/fidelity columns float to the front
    assert list(feature_table(_z(), names=names).columns)[:2] == ["feature_id", "concept"]


def test_feature_table_no_metadata():
    df = feature_table(_z())
    assert set(df["feature_id"]) == {0, 1}
    assert "concept" not in df.columns


def test_top_battles_modes():
    z, b = _z(), _battles()
    pos = top_battles(z, b, 0, mode="pos", n=10)
    assert list(pos["z"]) == [4.0, 2.0]               # only positives, desc
    neg = top_battles(z, b, 0, mode="neg", n=10)
    assert list(neg["z"]) == [-1.0]
    abs_ = top_battles(z, b, 0, mode="abs", n=10)
    assert list(abs_["z"]) == [4.0, 2.0, -1.0]        # by |z|, zeros dropped
    assert abs_.iloc[0]["prompt"] == "p3"


def test_top_battles_bad_mode():
    import pytest
    with pytest.raises(ValueError):
        top_battles(_z(), _battles(), 0, mode="nonsense")


def _per_battle():
    return pd.DataFrame({
        "self_model": ["M", "M", "M", "M"],
        "other_model": ["X", "Y", "X", "Y"],
        "prompt": ["p0", "p1", "p2", "p3"],
        "self_completion": ["s0", "s1", "s2", "s3"],
        "other_completion": ["o0", "o1", "o2", "o3"],
        "outcome": ["win", "loss", "tie", "win"],
        "win": [1.0, 0.0, 0.5, 1.0],
        "z0": [2.0, -1.0, 0.0, 4.0],
    })


def test_diagnosis_battles_more_less_abs():
    pb = _per_battle()
    more = diagnosis_battles(pb, 0, mode="more", n=10)
    assert list(more["z"]) == [4.0, 2.0]                  # z>0 desc
    assert more.iloc[0]["self_completion"] == "s3"
    less = diagnosis_battles(pb, 0, mode="less", n=10)
    assert list(less["z"]) == [-1.0]                      # z<0
    abs_ = diagnosis_battles(pb, 0, mode="abs", n=10)
    assert list(abs_["z"]) == [4.0, 2.0, -1.0]            # |z|, zeros dropped


def test_diagnosis_battles_missing_feature():
    import pytest
    with pytest.raises(ValueError):
        diagnosis_battles(_per_battle(), 9, mode="more")


def test_load_lens_for_view_from_corpus(tmp_path):
    # lens meta (ids 0,1) + z_diff, plus a corpus parquet supplying the text
    pd.DataFrame({"instruction_id": ["0", "1"], "model_a": ["X", "Y"],
                  "model_b": ["Y", "X"]}).to_parquet(tmp_path / "battles.parquet")
    np.save(tmp_path / "z_diff.npy", _z()[:2])
    (tmp_path / "manifest.json").write_text(json.dumps({"input_rep": "difference"}))
    from prefscope.data.corpus import normalize, write_corpus
    raw = pd.DataFrame({"prompt": ["p0", "p1"], "model_a": ["X", "Y"],
                        "model_b": ["Y", "X"], "completion_a": ["ca0", "ca1"],
                        "completion_b": ["cb0", "cb1"]})
    corp = normalize(raw, "lmarena-100k")
    corp["battle_id"] = ["0", "1"]
    write_corpus(corp, tmp_path / "corp.parquet")

    battles, z, _ = load_lens_for_view(tmp_path, corpus=str(tmp_path / "corp.parquet"))
    assert list(battles["instruction_id"]) == ["0", "1"]
    assert list(battles["completion_a"]) == ["ca0", "ca1"]   # text attached
    assert len(z) == 2


def test_load_lens_for_view_without_annotations(tmp_path):
    # lens folder alone: meta parquet + z_diff, no completion text
    pd.DataFrame({"instruction_id": ["0", "1"], "model_a": ["X", "Y"],
                  "model_b": ["Y", "X"], "y_judge": [1.0, 0.0]}
                 ).to_parquet(tmp_path / "battles.parquet")
    np.save(tmp_path / "z_diff.npy", _z()[:2])
    (tmp_path / "manifest.json").write_text(json.dumps({"input_rep": "difference"}))

    battles, z, manifest = load_lens_for_view(tmp_path, annotations=None)
    assert len(battles) == 2 == len(z)
    # text columns are synthesized as empty so the viewer still renders
    assert (battles["completion_a"] == "").all()
    assert manifest["input_rep"] == "difference"
