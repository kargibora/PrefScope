"""export_diagnosis: per-model fire_rate / prompt_types / relations from a bank.

Synthetic bank with a planted prompt→response signal; no GPU / embeddings.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
from export_viewer_data import (  # noqa: E402
    export_conditional,
    export_diagnosis,
    export_report_battles,
)
from prefscope.pipeline.oriented_bank import build_oriented_codes, save_bank  # noqa: E402


class _Id:
    def project(self, x):
        return np.asarray(x, dtype=np.float32)


def _bank_lens(tmp_path):
    """Lens dir with a bank where model M expresses feature 0 (code>0) on the
    battles it wins, so within the single prompt concept the prompt→response
    edge has delta_win = +1.0 and fire_rate = 0.5."""
    ids = [f"b{i}" for i in range(6)]
    battles = pd.DataFrame({"instruction_id": ids,
                            "model_a": ["M"] * 6, "model_b": ["O"] * 6,
                            "y_judge": [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]})
    e_a = np.array([[1], [1], [1], [0], [0], [0]], np.float32)
    e_b = np.zeros((6, 1), np.float32)
    Z, meta = build_oriented_codes(e_a, e_b, battles, _Id())
    lens = tmp_path / "lens"
    lens.mkdir()
    save_bank(lens / "bank", Z, meta)

    plens = tmp_path / "plens"
    plens.mkdir()
    np.save(plens / "z_prompt.npy", np.tile(np.array([[5.0, 0.0]], np.float32), (6, 1)))
    pd.DataFrame({"battle_id": ids}).to_parquet(plens / "battles.parquet")
    return lens, plens, ids


def test_export_diagnosis_emits_fire_rate_prompt_types_relations(tmp_path):
    lens, plens, _ = _bank_lens(tmp_path)
    features = pd.DataFrame({"feature_id": [0], "concept": ["code blocks"]})
    diag = export_diagnosis(lens, features, 2, prompt_lens=str(plens),
                            prompt_names={0: "coding"})
    assert diag is not None
    row = diag["rows"]["M"]
    assert {"fire_rate", "prompt_types", "relations"} <= set(row)
    np.testing.assert_allclose(row["fire_rate"][0], 0.5)
    # prompt-types: one concept, win rate = mean(1,1,1,0,0,0) = 0.5
    assert row["prompt_types"] and row["prompt_types"][0]["concept"] == "coding"
    np.testing.assert_allclose(row["prompt_types"][0]["win_rate"], 0.5)
    # relation: coding ⇒ code blocks, +1.0 Δwin
    rels = row["relations"]
    assert len(rels) == 1
    assert rels[0]["prompt_concept"] == "coding"
    assert rels[0]["response_concept"] == "code blocks"
    np.testing.assert_allclose(rels[0]["delta_win"], 1.0)
    assert rels[0]["n"] == 3


def test_export_diagnosis_fire_rate_is_absolute_prevalence_with_per_side_codes(tmp_path):
    """When the lens has per-side codes z_a/z_b, 'Does a lot' fire_rate is the ABSOLUTE
    prevalence P(z_self>0) — the fraction of the model's OWN answers expressing the +pole —
    NOT the bank's contrast disagreement rate (#1)."""
    lens, _, ids = _bank_lens(tmp_path)  # bank has 6 battles, model_a=M, model_b=O
    # per-side absolute codes for feature 0: M (model_a) fires +pole in 3 of 6 answers -> 0.5;
    # O (model_b) fires +pole in 1 of 6 -> 1/6. (The bank's contrast rate for M was 0.5 too,
    # so use a DIFFERENT O rate to prove the absolute path is what's used, not the contrast.)
    np.save(lens / "z_a.npy", np.array([[1.], [2.], [1.], [0.], [0.], [0.]], np.float32))
    np.save(lens / "z_b.npy", np.array([[0.], [0.], [0.], [0.], [0.], [5.]], np.float32))
    pd.DataFrame({"battle_id": ids, "model_a": ["M"] * 6, "model_b": ["O"] * 6}
                 ).to_parquet(lens / "battles.parquet")
    features = pd.DataFrame({"feature_id": [0], "concept": ["code blocks"]})
    diag = export_diagnosis(lens, features, 2)
    assert diag["fire_rate_kind"] == "absolute"
    np.testing.assert_allclose(diag["rows"]["M"]["fire_rate"][0], 0.5)      # 3/6
    np.testing.assert_allclose(diag["rows"]["O"]["fire_rate"][0], 1 / 6, atol=1e-5)


def test_export_diagnosis_fire_rate_falls_back_to_contrast_without_per_side_codes(tmp_path):
    """A difference lens (no z_a/z_b) can't do absolute prevalence -> fire_rate is the bank
    contrast rate, honestly labeled fire_rate_kind='contrast'."""
    lens, _, _ = _bank_lens(tmp_path)  # no z_a/z_b/battles written
    features = pd.DataFrame({"feature_id": [0], "concept": ["code blocks"]})
    diag = export_diagnosis(lens, features, 2)
    assert diag["fire_rate_kind"] == "contrast"
    np.testing.assert_allclose(diag["rows"]["M"]["fire_rate"][0], 0.5)      # (pm+nm)/n


def test_export_diagnosis_relations_empty_without_prompt_lens(tmp_path):
    lens, _, _ = _bank_lens(tmp_path)
    features = pd.DataFrame({"feature_id": [0], "concept": ["code blocks"]})
    diag = export_diagnosis(lens, features, 2)  # no prompt lens
    row = diag["rows"]["M"]
    assert row["relations"] == [] and row["prompt_types"] == []
    assert "fire_rate" in row  # fire_rate needs no prompt lens


def test_export_diagnosis_returns_none_without_bank(tmp_path):
    lens = tmp_path / "lens"
    lens.mkdir()
    features = pd.DataFrame({"feature_id": [0], "concept": ["x"]})
    assert export_diagnosis(lens, features) is None


# --- conditional δ: prompt-CLUSTER names come from the conditional CSV itself ---

def test_export_conditional_uses_cluster_names_from_conditional_csv(tmp_path):
    # conditional CSV is keyed by prompt CLUSTERS and carries prompt_concept_name
    # (cluster behavior). The delta CSV (raw-concept keyspace) must NOT override it.
    cond = tmp_path / "conditional_win_relevance.csv"
    pd.DataFrame({
        "prompt_concept": [0, 1],
        "prompt_concept_name": ["coding help", "creative writing"],
        "feature_id": [7, 7],
        "delta_win_rate": [0.05, -0.03],
        "cond_significant": [True, False],
        "n_battles": [400, 350],
    }).to_csv(cond, index=False)
    features = pd.DataFrame({"feature_id": [7], "concept": ["code blocks"]})
    out = export_conditional(str(cond), features)
    names = {p["id"]: p["name"] for p in out["prompt_concepts"]}
    assert names == {0: "coding help", 1: "creative writing"}
    assert out["features"][0]["concept"] == "code blocks"


# --- per (model × prompt-concept) drill-in battles ---

def _corpus(tmp_path):
    df = pd.DataFrame({
        "battle_id": ["b0", "b1", "b2"],
        "source": ["arena"] * 3,
        "language": ["en"] * 3,
        "prompt": ["write a function", "write a function", "write a function"],
        "model_a": ["M", "M", "O"],
        "model_b": ["O", "O", "M"],
        "completion_a": ["A0", "A1", "A2"],
        "completion_b": ["B0", "B1", "B2"],
        "human_pref": [1.0, 0.0, 1.0],  # b0: M(A) wins; b1: M(A) loses; b2: A=O wins so M(B) loses
    })
    p = tmp_path / "corpus.parquet"
    df.to_parquet(p)
    plens = tmp_path / "plens"
    plens.mkdir()
    np.save(plens / "z_prompt.npy", np.tile(np.array([[5.0, 0.0]], np.float32), (3, 1)))
    pd.DataFrame({"battle_id": ["b0", "b1", "b2"]}).to_parquet(plens / "battles.parquet")
    return p, plens


def test_export_report_battles_orients_and_groups_by_prompt_concept(tmp_path):
    corpus, plens = _corpus(tmp_path)
    diag = {"models": ["M"],
            "rows": {"M": {"prompt_types": [{"concept": "coding", "win_rate": 0.5, "n": 3}]}}}
    pnames = pd.DataFrame({"feature_id": [0], "concept": ["coding"]})
    rb = export_report_battles(tmp_path, str(corpus), str(plens), diag, pnames, per_type=5)
    assert "M" in rb and "coding" in rb["M"]
    rows = rb["M"]["coding"]
    assert len(rows) == 3
    # b0: M is model_a and A preferred -> self="A0", win
    b0 = next(r for r in rows if r["self"] == "A0")
    assert b0["other"] == "B0" and b0["outcome"] == "win"
    # b2: M is model_b, A(=O) preferred -> self="B2", loss
    b2 = next(r for r in rows if r["self"] == "B2")
    assert b2["other"] == "A2" and b2["outcome"] == "loss"


def test_export_report_battles_skips_concepts_not_in_prompt_types(tmp_path):
    corpus, plens = _corpus(tmp_path)
    # prompt_types names a concept that no battle maps to -> nothing kept -> None
    diag = {"models": ["M"], "rows": {"M": {"prompt_types": [{"concept": "math", "win_rate": 0.5, "n": 3}]}}}
    pnames = pd.DataFrame({"feature_id": [0], "concept": ["coding"]})
    assert export_report_battles(tmp_path, str(corpus), str(plens), diag, pnames) is None


def test_export_report_battles_does_not_label_all_negative_prompt(tmp_path):
    corpus, plens = _corpus(tmp_path)
    np.save(plens / "z_prompt.npy", np.tile(
        np.array([[-5.0, -1.0]], np.float32), (3, 1)))
    diag = {"models": ["M"],
            "rows": {"M": {"prompt_types": [{"concept": "coding", "win_rate": 0.5, "n": 3}]}}}
    pnames = pd.DataFrame({"feature_id": [0, 1], "concept": ["coding", "other"]})
    assert export_report_battles(tmp_path, str(corpus), str(plens), diag, pnames) is None


# --- unnamed concepts emit null, never the string "nan" (A1) ---

def test_export_elicitation_emits_null_for_unnamed(tmp_path):
    from export_viewer_data import export_elicitation
    csv = tmp_path / "elic.csv"
    pd.DataFrame({
        "prompt_feature": [0, 0],
        "completion_feature": [1, 2],
        "prompt_feature_name": ["coding", "coding"],
        "completion_feature_name": ["code blocks", np.nan],  # feature 2 unnamed
        "lift": [2.0, 0.5],
        "significant": [True, False],
    }).to_csv(csv, index=False)
    out = export_elicitation(str(csv))
    rc = {c["id"]: c["concept"] for c in out["response_concepts"]}
    assert rc[1] == "code blocks"
    assert rc[2] is None  # not the string "nan"


def test_export_elicitation_keeps_each_features_top_edges(tmp_path):
    """Per-feature coverage cap: a feature whose best edge is weak GLOBALLY (dwarfed by
    other features' stronger edges) must still be kept, so the Feature panel's 'activated
    by' isn't empty for it (the old global top-N by |lift| dropped exactly these)."""
    import numpy as np
    from export_viewer_data import export_elicitation
    rows = []
    for px in range(60):  # 60 very strong edges for feature 1 would dominate a global top-N
        rows.append({"prompt_feature": px, "completion_feature": 1, "lift": 6.0, "significant": False})
    rows.append({"prompt_feature": 3, "completion_feature": 100, "lift": 1.4, "significant": False})  # weak, but feature 100's best
    pd.DataFrame(rows).to_csv(tmp_path / "e.csv", index=False)
    out = export_elicitation(str(tmp_path / "e.csv"), max_edges=1000, per_feature=15, per_prompt=30)
    cys = {e["cy"] for e in out["edges"]}
    assert 100 in cys and 1 in cys        # feature 100's edge survived despite feature 1's flood
    assert out["n_edges"] == 61           # true total reported honestly, not the kept count


def test_export_conditional_emits_null_for_unnamed_feature(tmp_path):
    cond = tmp_path / "cond.csv"
    pd.DataFrame({
        "prompt_concept": [0],
        "prompt_concept_name": ["coding help"],
        "feature_id": [99],          # not in features -> unnamed
        "delta_win_rate": [0.05],
        "cond_significant": [True],
    }).to_csv(cond, index=False)
    features = pd.DataFrame({"feature_id": [7], "concept": ["code blocks"]})
    out = export_conditional(str(cond), features)
    assert out["features"][0]["concept"] is None
    assert out["prompt_concepts"][0]["name"] == "coding help"


# --- generality = pervasiveness (fire rate) + n_prompt_types (secondary) ---

def test_feature_fire_rate_from_per_side_codes(tmp_path):
    """generality = fraction of responses (both sides) where a feature's top-k code != 0."""
    import numpy as np
    from export_viewer_data import feature_fire_rate
    lens = tmp_path / "lens"
    lens.mkdir()
    # 2 battles, 2 features. feature 0 fires: A[b0], B[b0], B[b1] -> 3 of 4 responses = 0.75
    #                        feature 1 fires: never -> 0.0
    np.save(lens / "z_a.npy", np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32))
    np.save(lens / "z_b.npy", np.array([[2.0, 0.0], [3.0, 0.0]], dtype=np.float32))
    fr = feature_fire_rate(lens)
    assert fr[0] == 0.75
    assert fr[1] == 0.0


def test_feature_fire_rate_needs_per_side_codes(tmp_path):
    """A difference lens has only z_diff (no per-side firing) -> {} (generality absent)."""
    import numpy as np
    from export_viewer_data import feature_fire_rate
    lens = tmp_path / "lens"
    lens.mkdir()
    np.save(lens / "z_diff.npy", np.zeros((2, 2), dtype=np.float32))
    assert feature_fire_rate(lens) == {}


def test_feature_prompt_types_counts_significant_elicitors(tmp_path):
    """n_prompt_types = # prompt concepts that significantly (lift>1) elicit the feature."""
    from export_viewer_data import feature_prompt_types
    csv = tmp_path / "elic.csv"
    pd.DataFrame({
        "prompt_feature":     [0, 1, 2,  0, 1, 2],
        "completion_feature": [10, 10, 10, 20, 20, 20],
        "lift":        [2.0, 2.0, 0.5,  2.0, 1.0, 1.0],   # 10: 2 sig; 20: 1 sig
        "significant": [True, True, True, True, True, False],
    }).to_csv(csv, index=False)
    npt = feature_prompt_types(str(csv))
    assert npt[10] == 2
    assert npt[20] == 1
    assert feature_prompt_types(None) == {}


def test_dumps_emits_valid_json_for_nan_inf(tmp_path):
    """NaN/Inf floats must serialize to null (bare NaN/Infinity is invalid JSON and breaks
    the browser's JSON.parse). String content 'NaN' must be preserved."""
    import json
    import math
    from export_viewer_data import _dumps
    import numpy as np
    s = _dumps({"a": float("nan"), "b": math.inf, "c": -math.inf, "d": 1.5,
                "e": ["x", float("nan"), 2], "text": "value is NaN",
                # numpy scalars must not crash json.dumps (they aren't python-float subclasses)
                "nf32": np.float32("nan"), "ni": np.int64(7), "nb": np.bool_(True)})
    assert "NaN" not in s.replace('"value is NaN"', "")  # no bare NaN token
    d = json.loads(s)  # strict parse (rejects NaN) must succeed
    assert d["a"] is None and d["b"] is None and d["c"] is None
    assert d["d"] == 1.5 and d["e"] == ["x", None, 2]
    assert d["text"] == "value is NaN"  # string untouched
    assert d["nf32"] is None and d["ni"] == 7 and d["nb"] is True


# --- per-model example answers (report-card drill-in shows the model's OWN answers) ---

def test_export_examples_by_model(tmp_path, monkeypatch):
    """Each (model, feature) gets that model's own top-activating answers, from the correct
    side (z_a for model_a, z_b for model_b), with outcome from the model's perspective."""
    import numpy as np
    import export_viewer_data as ev
    from prefscope import interpret as _interpret  # noqa: F401
    import prefscope.interpret.io as io

    lens = tmp_path / "lens"
    lens.mkdir()
    # battle 0: A fires feature 0 (z=2), A won. battle 1: B fires feature 0 (z=3), B won.
    np.save(lens / "z_a.npy", np.array([[2.0], [0.0]], dtype=np.float32))
    np.save(lens / "z_b.npy", np.array([[0.0], [3.0]], dtype=np.float32))
    battles = pd.DataFrame({
        "model_a": ["A", "A"], "model_b": ["B", "B"],
        "prompt": ["p0", "p1"], "completion_a": ["a0", "a1"], "completion_b": ["b0", "b1"],
        "y_judge": [1.0, 0.0],           # battle0 A preferred; battle1 B preferred
    })
    monkeypatch.setattr(io, "load_lens_battles",
                        lambda lens_dir, corpus=None: (battles, np.zeros((2, 1)), {}))

    features = pd.DataFrame({"feature_id": [0], "fidelity_pass": [True]})
    diag = {"models": ["A", "B"], "features": [0], "concepts": ["c"]}
    ebm = ev.export_examples_by_model(lens, "corpus.parquet", features, diag, n_per=4)

    assert set(ebm) == {"A", "B"}
    ea = ebm["A"]["0"][0]
    assert ea["answer"] == "a0" and ea["z"] == 2.0 and ea["outcome"] == "win"
    eb = ebm["B"]["0"][0]
    assert eb["answer"] == "b1" and eb["z"] == 3.0 and eb["outcome"] == "win"   # B preferred


def test_export_examples_by_model_needs_per_side_codes(tmp_path):
    import export_viewer_data as ev
    features = pd.DataFrame({"feature_id": [0], "fidelity_pass": [True]})
    diag = {"models": ["A", "B"], "features": [0], "concepts": ["c"]}
    assert ev.export_examples_by_model(tmp_path, "corpus.parquet", features, diag) is None


def test_export_examples_by_model_surfaces_concept_pole_not_magnitude(tmp_path, monkeypatch):
    """The concept NAME describes the positive pole; a strongly-NEGATIVE activation is the
    OPPOSITE pole (a different concept), so it must NOT be surfaced under 'answers exhibiting
    <concept>'. Selection is by signed activation (concept pole), NOT |activation|."""
    import numpy as np
    import export_viewer_data as ev
    import prefscope.interpret.io as io

    lens = tmp_path / "lens"
    lens.mkdir()
    # model A, feature 0: battle0 = +1.0 (concept pole), battle1 = -5.0 (opposite pole, larger |·|)
    np.save(lens / "z_a.npy", np.array([[1.0], [-5.0]], dtype=np.float32))
    np.save(lens / "z_b.npy", np.array([[0.0], [0.0]], dtype=np.float32))
    battles = pd.DataFrame({
        "model_a": ["A", "A"], "model_b": ["B", "B"],
        "prompt": ["p0", "p1"], "completion_a": ["a0", "a1"], "completion_b": ["b0", "b1"],
        "y_judge": [1.0, 1.0],
    })
    monkeypatch.setattr(io, "load_lens_battles",
                        lambda lens_dir, corpus=None: (battles, np.zeros((2, 1)), {}))
    features = pd.DataFrame({"feature_id": [0], "fidelity_pass": [True]})
    diag = {"models": ["A", "B"], "features": [0], "concepts": ["c"]}
    ebm = ev.export_examples_by_model(lens, "corpus.parquet", features, diag, n_per=4)

    exs = ebm["A"]["0"]
    # ONLY the concept-pole (positive) answer survives; the larger-magnitude opposite pole is dropped
    assert [e["answer"] for e in exs] == ["a0"]
    assert exs[0]["z"] == 1.0
    assert all(e["z"] > 0 for e in exs)


# --- head-to-head (#1): paired prompt-matched feature contrast between model pairs ---

def _write_side_lens(tmp_path, z_a, z_b, model_a, model_b):
    """A minimal individual-lens dir: per-side codes + a battles frame with model_a/model_b."""
    import numpy as np
    lens = tmp_path / "lens"
    lens.mkdir(exist_ok=True)
    np.save(lens / "z_a.npy", np.asarray(z_a, dtype=np.float32))
    np.save(lens / "z_b.npy", np.asarray(z_b, dtype=np.float32))
    pd.DataFrame({"model_a": model_a, "model_b": model_b}).to_parquet(lens / "battles.parquet")
    return lens


def test_export_head_to_head_paired_discordant_counts(tmp_path):
    """Discordant counts come from per-side codes z_a/z_b (NOT the sign-flipped bank), and
    accumulate per unordered model pair oriented a<b by model index."""
    import numpy as np
    import export_viewer_data as ev

    # 3 A-vs-B battles, 1 feature ("fires" == z != 0):
    #  b1: A fires, B doesn't   b2: A fires, B doesn't   b3: A doesn't, B fires
    lens = _write_side_lens(tmp_path,
                            z_a=np.array([[1.0], [2.0], [0.0]]),
                            z_b=np.array([[0.0], [0.0], [3.0]]),
                            model_a=["A", "A", "A"], model_b=["B", "B", "B"])
    features = pd.DataFrame({"feature_id": [0], "fidelity_pass": [True], "concept": ["greets back"]})
    diag = {"models": ["A", "B", "C"], "features": [0], "concepts": ["greets back"]}
    h2h = ev.export_head_to_head(lens, features, diag, min_shared=1)

    assert h2h["models"] == ["A", "B", "C"] and h2h["features"] == [0]
    assert len(h2h["pairs"]) == 1
    p = h2h["pairs"][0]
    assert (p["a"], p["b"]) == (0, 1)          # A<B by model index
    assert p["n"] == 3
    assert p["bpos"] == [2]                     # A fires, B doesn't: 2 battles
    assert p["cpos"] == [1]                     # B fires, A doesn't: 1 battle


def test_export_head_to_head_orientation_independent_of_column_order(tmp_path):
    """Counts land on the same (lo, hi) orientation regardless of which model is model_a
    (exercises the a_is_lo=False swap branch)."""
    import numpy as np
    import export_viewer_data as ev

    # both battles are "A expresses it, B doesn't" — but battle 2 puts B in the model_a slot
    lens = _write_side_lens(tmp_path,
                            z_a=np.array([[1.0], [0.0]]),   # b1: A(model_a) fires; b2: B(model_a) doesn't
                            z_b=np.array([[0.0], [1.0]]),   # b1: B(model_b) doesn't; b2: A(model_b) fires
                            model_a=["A", "B"], model_b=["B", "A"])
    features = pd.DataFrame({"feature_id": [0], "fidelity_pass": [True], "concept": ["c"]})
    diag = {"models": ["A", "B"], "features": [0], "concepts": ["c"]}
    h2h = ev.export_head_to_head(lens, features, diag, min_shared=1)
    p = h2h["pairs"][0]
    assert (p["a"], p["b"]) == (0, 1)          # A<B
    assert p["n"] == 2
    assert p["bpos"] == [2]                     # A expresses it, B doesn't in BOTH battles
    assert p["cpos"] == [0]


def test_export_head_to_head_needs_per_side_codes(tmp_path):
    """A difference lens (only z_diff, no z_a/z_b) can't do a per-side head-to-head -> None."""
    import numpy as np
    import export_viewer_data as ev
    lens = tmp_path / "lens"
    lens.mkdir()
    np.save(lens / "z_diff.npy", np.zeros((2, 1), dtype=np.float32))
    pd.DataFrame({"model_a": ["A", "A"], "model_b": ["B", "B"]}).to_parquet(lens / "battles.parquet")
    features = pd.DataFrame({"feature_id": [0], "fidelity_pass": [True]})
    diag = {"models": ["A", "B"], "features": [0], "concepts": ["x"]}
    assert ev.export_head_to_head(lens, features, diag, min_shared=1) is None


def test_export_meta_has_preference_flag(tmp_path):
    """has_preference is True only when win-relevance columns reached the features table."""
    import json
    from export_viewer_data import export_meta

    lens = tmp_path / "lens"
    lens.mkdir()
    (lens / "manifest.json").write_text(json.dumps({
        "input_rep": "individual", "embed_model_id": "X", "m_total": 8,
        "k": 2, "input_dim": 16, "n_battles": 100,
    }))

    labeled = pd.DataFrame({
        "feature_id": [0, 1], "concept": ["a", "b"],
        "fidelity_pass": [True, False], "delta_win_rate": [0.1, -0.2],
    })
    unlabeled = pd.DataFrame({
        "feature_id": [0, 1], "concept": ["a", "b"], "fidelity_pass": [True, False],
    })

    assert export_meta(lens, None, labeled)["has_preference"] is True
    assert export_meta(lens, None, unlabeled)["has_preference"] is False


def test_export_meta_r2_is_loo_semantics(tmp_path):
    """r2/is_loo are honest: loo_r2 is null unless predictions are genuinely LOO."""
    import json
    from export_viewer_data import export_meta

    lens = tmp_path / "lens"
    lens.mkdir()
    (lens / "manifest.json").write_text(json.dumps({"m_total": 4, "k": 2}))
    feats = pd.DataFrame({"feature_id": [0], "concept": ["a"], "delta_win_rate": [0.1]})

    insample = pd.DataFrame({
        "model": ["m1", "m2", "m3"],
        "predicted_score": [0.1, 0.5, 0.9],
        "actual_win_rate": [0.2, 0.5, 0.8],
    })
    m = export_meta(lens, insample, feats)
    assert m["is_loo"] is False
    assert m["loo_r2"] is None          # never labeled held-out
    assert m["r2"] is not None and 0.9 < m["r2"] <= 1.0  # near-perfect linear relation

    loo = insample.rename(columns={"predicted_score": "predicted_score_loo"})
    m = export_meta(lens, loo, feats)
    assert m["is_loo"] is True
    assert m["loo_r2"] == m["r2"]
