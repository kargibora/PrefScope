"""Tests for the public Lens object: encode / encode_one / concept_names /
top_concepts / save, the back-compat aliases, pairs_to_battles, and Lens.train."""
import numpy as np
import pandas as pd
import pytest

from prefscope.api.loaded_lens import Lens, LoadedLens, pairs_to_battles
from prefscope.core.dataset import Dataset
from prefscope.core.types import PairItem


class FakeEmbedder:
    """Deterministic: each row filled with the completion's (or prompt's) length."""
    def encode(self, prompts, completions):
        return np.array([[float(len(c))] * 4 for c in completions], dtype=np.float32)

    def encode_prompts(self, prompts):
        return np.array([[float(len(p))] * 4 for p in prompts], dtype=np.float32)


class FakeProjector:
    m_total = 3
    def project(self, x):
        x = np.asarray(x, dtype=np.float32)
        return np.stack([x[:, 0], -x[:, 0], np.zeros(len(x))], axis=1).astype(np.float32)


def _names():
    return pd.DataFrame({"feature_id": [0, 1, 2], "concept": ["a", "b", "c"],
                         "fidelity_pass": [True, False, True]})


def _lens(manifest=None, names=None):
    return Lens(FakeProjector(), FakeEmbedder(),
                names=names, manifest=manifest or {"input_rep": "individual"})


# ---- aliases -------------------------------------------------------------

def test_loadedlens_is_lens():
    assert LoadedLens is Lens


def test_load_alias_matches_from_dir():
    assert Lens.load.__func__ is Lens.from_dir.__func__


def test_constructed_directly_has_no_lens_dir():
    assert _lens().lens_dir is None


# ---- encode --------------------------------------------------------------

def test_encode_individual_shape():
    lens = _lens()
    out = lens.encode(["q1", "q2"], ["aaaa", "bb"])
    assert out.shape == (2, 3)
    np.testing.assert_allclose(out[:, 0], [4.0, 2.0])


def test_encode_prompt_lens_uses_prompts():
    lens = _lens(manifest={"input_rep": "prompt"})
    out = lens.encode(["abc", "de"])
    assert out.shape == (2, 3)
    np.testing.assert_allclose(out[:, 0], [3.0, 2.0])


def test_encode_accepts_single_str():
    lens = _lens()
    out = lens.encode("q", "aaaa")
    assert out.shape == (1, 3)


def test_encode_one_shape():
    lens = _lens()
    out = lens.encode_one("q", "aaaa")
    assert out.shape == (3,)


def test_encode_difference_guard_raises():
    lens = _lens(manifest={"input_rep": "difference"})
    with pytest.raises(ValueError, match="difference lens is contrast-only"):
        lens.encode(["q"], ["a"])


def test_encode_individual_none_completions_raises():
    lens = _lens()
    with pytest.raises(ValueError, match="individual lens needs completions"):
        lens.encode(["p"])
    with pytest.raises(ValueError, match="individual lens needs completions"):
        lens.encode_one("p")


def test_encode_prompt_lens_none_completions_ok():
    lens = _lens(manifest={"input_rep": "prompt"})
    out = lens.encode(["p"])
    assert out.shape == (1, 3)


def test_encode_length_mismatch_raises():
    lens = _lens()
    with pytest.raises(ValueError, match="length mismatch"):
        lens.encode(["p1", "p2"], ["c1"])
    # equal lengths still return (N, M)
    assert lens.encode(["p1", "p2"], ["c1", "c2"]).shape == (2, 3)


# ---- concept_names / top_concepts ---------------------------------------

def test_concept_names_maps_ids():
    s = _lens(names=_names()).concept_names
    assert s.loc[0] == "a" and s.loc[1] == "b" and s.loc[2] == "c"
    assert _lens().concept_names is None


def test_top_concepts_named_only_and_sorted():
    names = pd.DataFrame({"feature_id": [0, 2], "concept": ["a", "c"]})  # 1 unnamed
    lens = _lens(names=names)
    codes = np.array([[1.0, -9.0, 2.0]])   # feature 1 has biggest |code| but no name
    top = lens.top_concepts(codes, k=5)
    assert len(top) == 1
    concepts = [c for c, _ in top[0]]
    assert "b" not in concepts and concepts == ["c", "a"]   # |2| > |1|, both named
    assert top[0][0] == ("c", 2.0)


def test_top_concepts_no_names():
    assert _lens().top_concepts(np.array([[1.0, 2.0, 3.0]])) == [[]]


def test_concept_names_dedupes_duplicate_feature_id():
    names = pd.DataFrame({"feature_id": [0, 0, 1], "concept": ["a", "a2", "b"]})
    lens = _lens(names=names)
    s = lens.concept_names
    assert s.index.is_unique
    # top_concepts must not raise on the duplicate id
    top = lens.top_concepts(np.array([[3.0, 1.0, 0.0]]), k=2)
    assert len(top) == 1


def test_top_concepts_k_zero_returns_empty_lists():
    lens = _lens(names=_names())
    out = lens.top_concepts(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), k=0)
    assert out == [[], []]


def test_top_concepts_skips_nan_codes():
    lens = _lens(names=_names())
    out = lens.top_concepts(np.array([[np.nan, 2.0, np.nan]]), k=5)
    concepts = [c for c, _ in out[0]]
    assert concepts == ["b"]   # only the non-NaN named code survives


# ---- back-compat project alias ------------------------------------------

class _PairData(Dataset):
    def __iter__(self):
        yield PairItem(id="1", x="q", y_a="aaaa", y_b="b", pref=1.0,
                       model_a="m1", model_b="m2")
        yield PairItem(id="2", x="q", y_a="a", y_b="bbbb", pref=0.0,
                       model_a="m1", model_b="m2")


class _SingleData(Dataset):
    def __iter__(self):
        yield PairItem(id="s1", x="q", y_a="aaaa", model_a="m1")
        yield PairItem(id="s2", x="q", y_a="bb", model_a="m2")


def test_project_alias_returns_tuple():
    lens = _lens(manifest={"input_rep": "difference"})
    out = lens.project(_PairData())
    assert isinstance(out, tuple) and len(out) == 2
    codes, meta = out
    assert codes.shape == (2, 3)
    assert list(meta.columns) == ["id", "pref", "model_a", "model_b"]


def test_encode_pairs_same_as_project():
    lens = _lens(manifest={"input_rep": "difference"})
    c1, _ = lens.encode_pairs(_PairData())
    c2, _ = lens.project(_PairData())
    np.testing.assert_array_equal(c1, c2)


def test_encode_items_single_response_individual_lens():
    codes, meta = _lens().encode_items(_SingleData())
    np.testing.assert_allclose(codes[:, 0], [4.0, 2.0])
    assert list(meta.columns) == ["id", "pref", "model_a", "model_b"]
    assert list(meta["id"]) == ["s1", "s2"]


def test_encode_items_pairs_match_encode_pairs():
    lens = _lens(manifest={"input_rep": "difference"})
    got, _ = lens.encode_items(_PairData())
    expected, _ = lens.encode_pairs(_PairData())
    np.testing.assert_array_equal(got, expected)


def test_encode_items_single_requires_individual_and_rejects_mixed():
    difference = _lens(manifest={"input_rep": "difference"})
    with pytest.raises(ValueError, match="individual lens"):
        difference.encode_items(_SingleData())
    mixed = [PairItem(id="1", x="q", y_a="a"),
             PairItem(id="2", x="q", y_a="a", y_b="b")]
    with pytest.raises(ValueError, match="homogeneous"):
        _lens().encode_items(mixed)


# ---- save ----------------------------------------------------------------

def test_save_no_backing_dir_raises():
    with pytest.raises(ValueError, match="no backing directory"):
        _lens().save("/tmp/whatever")


def test_save_copies_dir(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "manifest.json").write_text('{"input_rep": "individual"}')
    lens = _lens()
    lens.lens_dir = src
    dest = tmp_path / "dest"
    lens.save(dest)
    assert (dest / "manifest.json").exists()
    # no-op safe when dest == src
    lens.save(src)


# ---- pairs_to_battles ----------------------------------------------------

def test_pairs_to_battles_from_pairitems():
    pairs = [PairItem(id="1", x="q1", y_a="a1", y_b="b1", pref=1.0,
                      model_a="m1", model_b="m2"),
             PairItem(id="2", x="q2", y_a="a2", y_b="b2", pref=0.0)]
    df = pairs_to_battles(pairs)
    for col in ("prompt", "completion_a", "completion_b", "instruction_id"):
        assert col in df.columns
    assert list(df["prompt"]) == ["q1", "q2"]
    assert list(df["instruction_id"]) == ["1", "2"]
    assert list(df["human_pref"]) == [1.0, 0.0]
    assert df["model_a"].iloc[0] == "m1" and pd.isna(df["model_a"].iloc[1])


def test_pairs_to_battles_from_dataframe_with_rename():
    raw = pd.DataFrame({"q": ["p1"], "ca": ["x"], "cb": ["y"], "iid": ["i1"],
                        "human_pref": [1.0]})
    df = pairs_to_battles(raw, columns={"q": "prompt", "ca": "completion_a",
                                        "cb": "completion_b", "iid": "instruction_id"})
    for col in ("prompt", "completion_a", "completion_b", "instruction_id"):
        assert col in df.columns
    assert list(df["human_pref"]) == [1.0]


def test_pairs_to_battles_missing_cols_raises():
    raw = pd.DataFrame({"prompt": ["p"]})
    with pytest.raises(ValueError, match="missing required columns"):
        pairs_to_battles(raw)


def test_pairs_to_battles_dataframe_accepts_single_response_rows():
    raw = pd.DataFrame({"prompt": ["p"], "completion_a": ["a"],
                        "instruction_id": ["i"]})
    out = pairs_to_battles(raw)
    assert list(out["completion_a"]) == ["a"]
    assert "completion_b" not in out.columns


# ---- Lens.train wiring ---------------------------------------------------

def test_lens_train_wires_config(monkeypatch, tmp_path):
    import prefscope.pipeline.build_lens as bl
    import prefscope.encode.embed as emb_mod
    from prefscope.api.config import SAEConfig, TrainConfig

    calls = {}

    def fake_build_lens(battles, embedder, out_dir, **kw):
        calls["battles"] = battles
        calls["kw"] = kw
        return {"ok": True}

    class FakeEmb:
        def __init__(self, cache, **kw):
            calls["emb_kw"] = kw

    monkeypatch.setattr(bl, "build_lens", fake_build_lens)
    monkeypatch.setattr(emb_mod, "Embedder", FakeEmb)
    monkeypatch.setattr(Lens, "load", classmethod(lambda cls, out, **kw: "LOADED"))

    pairs = [PairItem(id="1", x="q1", y_a="a1", y_b="b1", pref=1.0)]
    cfg = TrainConfig(sae=SAEConfig(m=64, k=8, input_rep="individual",
                                    matryoshka_prefix=(4, 16)),
                      embed_model_id="emb-x", val_frac=0.2, device="cpu",
                      max_train_rows=123, train_kwargs={"epochs": 2})
    result = Lens.train(pairs, cfg, out=tmp_path / "lens")

    assert result == "LOADED"
    assert calls["kw"]["m_total"] == 64
    assert calls["kw"]["k"] == 8
    assert calls["kw"]["input_rep"] == "individual"
    assert calls["kw"]["matryoshka_prefix"] == (4, 16)
    assert calls["kw"]["val_frac"] == 0.2
    assert calls["kw"]["device"] == "cpu"
    assert calls["kw"]["embed_model_id"] == "emb-x"
    assert calls["kw"]["max_train_rows"] == 123
    assert calls["kw"]["epochs"] == 2
    assert "prompt" in calls["battles"].columns


def test_lens_train_rejects_colliding_train_kwargs(monkeypatch, tmp_path):
    import prefscope.pipeline.build_lens as bl
    import prefscope.encode.embed as emb_mod
    from prefscope.api.config import SAEConfig, TrainConfig

    monkeypatch.setattr(bl, "build_lens", lambda *a, **k: {"ok": True})
    monkeypatch.setattr(emb_mod, "Embedder", lambda *a, **k: object())
    monkeypatch.setattr(Lens, "load", classmethod(lambda cls, out, **kw: "LOADED"))

    pairs = [PairItem(id="1", x="q1", y_a="a1", y_b="b1", pref=1.0)]
    cfg = TrainConfig(train_kwargs={"m_total": 999, "k": 3})
    with pytest.raises(ValueError, match="train_kwargs may not override"):
        Lens.train(pairs, cfg, out=tmp_path / "lens")
