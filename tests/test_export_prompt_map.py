"""export_prompt_map: aligns prompt+completion lenses, orients by winner, badges Δ."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
from export_viewer_data import export_prompt_map  # noqa: E402


def _setup(tmp: Path):
    plens, clens, pint = tmp / "plens", tmp / "clens", tmp / "pint"
    for d in (plens, clens, pint):
        d.mkdir()
    ids = ["b0", "b1", "b2"]
    # b0: dom prompt feat 0; b1: dom prompt feat 1; b2: tie (dropped)
    np.save(plens / "z_prompt.npy", np.array([[2.0, 0.1], [0.1, 1.5], [1.0, 0.2]], np.float32))
    np.save(clens / "z_diff.npy", np.array([[1.0, 0.1], [-0.5, 0.9], [0.3, 0.3]], np.float32))
    pd.DataFrame({"instruction_id": ids, "prompt": ["pA", "pB", "pC"]}).to_parquet(
        plens / "battles.parquet")
    pd.DataFrame({"instruction_id": ids, "prompt": ["pA", "pB", "pC"],
                  "human_pref": [1.0, 0.0, 0.5], "model_a": ["mA"] * 3, "model_b": ["mB"] * 3,
                  "completion_a": ["ca"] * 3, "completion_b": ["cb"] * 3}).to_parquet(
        clens / "battles.parquet")
    pd.DataFrame({"feature_id": [0, 1], "concept": ["c0", "c1"]}).to_csv(
        clens / "feature_names.csv", index=False)
    pd.DataFrame({"feature_id": [0, 1], "concept": ["p0", "p1"]}).to_csv(
        pint / "prompt_feature_names.csv", index=False)
    pd.DataFrame({"feature_id": [0, 1], "cluster_id": [10, 11], "behavior": ["B10", "B11"]}).to_csv(
        pint / "prompt_feature_clusters.csv", index=False)
    # one significant Δ cell: prompt cluster 10 × completion feature 0
    pd.DataFrame({"prompt_concept": [10], "completion_feature": [0], "delta": [0.2],
                  "p_bonferroni": [0.01], "stable": [True]}).to_csv(tmp / "delta.csv", index=False)
    return plens, clens, pint, tmp / "delta.csv"


def test_export_prompt_map(tmp_path):
    plens, clens, pint, delta = _setup(tmp_path)
    out = export_prompt_map(plens, clens, delta, pint)
    assert out is not None
    pts = out["points"]
    assert len(pts) == 2  # tie b2 dropped
    byw = {p["win"]: p for p in pts}

    a = byw["A"]  # b0, A wins -> oriented = +z_diff
    assert a["pf"][0]["concept"] == "p0"
    cf0 = next(c for c in a["cf"] if c["id"] == 0)
    assert cf0["sig"] is True and abs(cf0["delta"] - 0.2) < 1e-9
    assert abs(cf0["z"] - 1.0) < 1e-6

    b = byw["B"]  # b1, B wins -> oriented = -z_diff, sign flips
    cfb0 = next(c for c in b["cf"] if c["id"] == 0)
    assert abs(cfb0["z"] - 0.5) < 1e-6


def test_export_prompt_map_corpus_fallback(tmp_path):
    """Diff-lens battles often lack prompt/human_pref/model text — pull from --corpus."""
    plens, clens, pint = tmp_path / "plens", tmp_path / "clens", tmp_path / "pint"
    for d in (plens, clens, pint):
        d.mkdir()
    ids = ["b0", "b1", "b2"]
    np.save(plens / "z_prompt.npy", np.array([[2.0, 0.1], [0.1, 1.5], [1.0, 0.2]], np.float32))
    np.save(clens / "z_diff.npy", np.array([[1.0, 0.1], [-0.5, 0.9], [0.3, 0.3]], np.float32))
    # completion lens battles carry ONLY the id (like the difference lens)
    pd.DataFrame({"instruction_id": ids}).to_parquet(clens / "battles.parquet")
    pd.DataFrame({"instruction_id": ids, "prompt": ["pA", "pB", "pC"]}).to_parquet(
        plens / "battles.parquet")
    pd.DataFrame({"feature_id": [0, 1], "concept": ["c0", "c1"]}).to_csv(
        clens / "feature_names.csv", index=False)
    pd.DataFrame({"feature_id": [0, 1], "concept": ["p0", "p1"]}).to_csv(
        pint / "prompt_feature_names.csv", index=False)
    corpus = tmp_path / "corpus.parquet"
    pd.DataFrame({"battle_id": ids, "prompt": ["pA", "pB", "pC"],
                  "human_pref": [1.0, 0.0, 0.5], "model_a": ["mA"] * 3,
                  "model_b": ["mB"] * 3}).to_parquet(corpus)

    out = export_prompt_map(plens, clens, None, pint, corpus_path=str(corpus))
    assert out is not None
    assert len(out["points"]) == 2  # tie dropped
    a = next(p for p in out["points"] if p["win"] == "A")
    assert a["p"] == "pA" and a["ma"] == "mA"


def test_export_prompt_map_all_negative_prompt_has_no_concept(tmp_path):
    plens, clens, pint, delta = _setup(tmp_path)
    zp = np.load(plens / "z_prompt.npy")
    zp[0] = [-8.0, -1.0]
    np.save(plens / "z_prompt.npy", zp)

    out = export_prompt_map(plens, clens, delta, pint, mode="random", sample=3)
    point = next(p for p in out["points"] if p["p"] == "pA")
    assert point["f"] == -1
    assert point["pc"] == -1
    assert point["m"] == 0.0
    assert point["pf"] == []
