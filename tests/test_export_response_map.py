"""export_response_map: one point per single response (A/B), not per A/B pair."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
from export_viewer_data import export_response_map  # noqa: E402


def test_response_map_is_per_response(tmp_path):
    lens = tmp_path / "lens"; lens.mkdir()
    ids = [str(i) for i in range(6)]
    # feature 0 fires on the A responses, feature 1 on the B responses
    np.save(lens / "z_a.npy", np.array([[2.0, 0.0]] * 6, np.float32))
    np.save(lens / "z_b.npy", np.array([[0.0, 2.0]] * 6, np.float32))
    pd.DataFrame({"instruction_id": ids}).to_parquet(lens / "battles.parquet")
    corpus = tmp_path / "corpus.parquet"
    pd.DataFrame({"battle_id": ids, "prompt": ["p%d" % i for i in range(6)],
                  "completion_a": ["A-resp %d" % i for i in range(6)],
                  "completion_b": ["B-resp %d" % i for i in range(6)],
                  "model_a": ["mA"] * 6, "model_b": ["mB"] * 6}).to_parquet(corpus)
    feats = pd.DataFrame({"feature_id": [0, 1], "concept": ["c0", "c1"],
                          "fidelity_pass": [True, True]})

    out = export_response_map(lens, str(corpus), feats, mode="top-activating")
    assert out is not None
    sides = {p["side"] for p in out["points"]}
    assert sides == {"A", "B"}                       # both single-response sides present
    a = next(p for p in out["points"] if p["side"] == "A")
    assert a["r"].startswith("A-resp") and a["model"] == "mA"   # ONE response shown
    assert a["f"] == 0                                # A responses colored by feature 0
    b = next(p for p in out["points"] if p["side"] == "B")
    assert b["r"].startswith("B-resp") and b["f"] == 1
    assert "r" in a and "ca" not in a and "cb" not in a          # single response, not a pair


def test_response_map_uses_named_positive_pole_and_sentinel(tmp_path):
    lens = tmp_path / "lens"; lens.mkdir()
    ids = ["positive", "negative"]
    # Row 0 has a modest positive feature 0 and a much larger negative feature 1.
    # Row 1 has no named positive pole at all.
    np.save(lens / "z_a.npy", np.array([[1.0, -9.0], [-2.0, -3.0]], np.float32))
    np.save(lens / "z_b.npy", np.zeros((2, 2), np.float32))
    pd.DataFrame({"instruction_id": ids}).to_parquet(lens / "battles.parquet")
    corpus = tmp_path / "corpus.parquet"
    pd.DataFrame({
        "battle_id": ids, "prompt": ["p0", "p1"],
        "completion_a": ["positive row", "all negative row"],
        "completion_b": ["silent b0", "silent b1"],
        "model_a": ["mA", "mA"], "model_b": ["mB", "mB"],
    }).to_parquet(corpus)
    feats = pd.DataFrame({"feature_id": [0, 1], "concept": ["c0", "c1"],
                          "fidelity_pass": [True, True]})

    out = export_response_map(lens, str(corpus), feats, mode="random", sample=4, seed=0)
    assert out is not None
    by_response = {p["r"]: p for p in out["points"]}
    assert by_response["positive row"]["f"] == 0
    assert by_response["positive row"]["m"] == 1.0
    assert by_response["all negative row"]["f"] == -1
    assert by_response["all negative row"]["m"] == 0.0


def test_response_map_supports_single_response_lens_without_corpus(tmp_path):
    lens = tmp_path / "lens"; lens.mkdir()
    np.save(lens / "z_a.npy", np.array([[2.0, 0.0], [0.0, 1.0]], np.float32))
    pd.DataFrame({
        "instruction_id": ["0", "1"], "prompt": ["p0", "p1"],
        "completion_a": ["only a0", "only a1"], "model_a": ["m0", "m1"],
    }).to_parquet(lens / "battles.parquet")
    feats = pd.DataFrame({"feature_id": [0, 1], "concept": ["c0", "c1"],
                          "fidelity_pass": [True, True]})

    out = export_response_map(lens, "", feats, mode="random", sample=2)
    assert out is not None and out["n_total"] == 2
    assert {p["side"] for p in out["points"]} == {"A"}
    assert {p["r"] for p in out["points"]} == {"only a0", "only a1"}
