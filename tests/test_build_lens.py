import hashlib

import numpy as np
import pandas as pd
import pytest

from prefscope.pipeline.build_lens import build_lens, build_lens_from_embeddings
from prefscope.encode.sae import SAEProjector


class FakeEmbedder:
    """Deterministic embeddings keyed on (prompt, completion) text."""
    dim = 8

    def encode(self, prompts, completions):
        out = np.zeros((len(prompts), self.dim), dtype=np.float32)
        for i, (p, c) in enumerate(zip(prompts, completions)):
            h = hashlib.sha1(f"{p}||{c}".encode()).digest()
            vec = np.frombuffer(h[: self.dim], dtype=np.uint8).astype(np.float32)
            out[i] = (vec - 128.0) / 128.0
        return out


def _battles(n=40):
    rows = []
    for i in range(n):
        rows.append({
            "instruction_id": str(i),
            "model_a": "A" if i % 2 == 0 else "B",
            "model_b": "B" if i % 2 == 0 else "A",
            "prompt": f"prompt {i}",
            "completion_a": f"resp a {i}",
            "completion_b": f"resp b {i}",
            "y_judge": float(i % 3) / 2.0,
            "lang": "en",
        })
    return pd.DataFrame(rows)


def test_build_lens_difference_default(tmp_path):
    battles = _battles()
    emb = FakeEmbedder()
    out = build_lens(
        battles, emb, tmp_path,
        m_total=8, k=2, matryoshka_prefix=(4,),
        n_epochs=3, min_epochs=3, patience=3, batch=16, device="cpu", seed=0)

    assert out["input_rep"] == "difference"
    assert out["output_arrays"] == ["z_diff"]
    assert out["n_battles"] == 40
    assert out["m_total"] == 8

    assert (tmp_path / "sae_model.pt").exists()
    proj = SAEProjector(tmp_path, device="cpu")
    assert proj.m_total == 8

    # only z_diff is written in difference mode
    assert (tmp_path / "z_diff.npy").exists()
    assert not (tmp_path / "z_a.npy").exists()
    assert not (tmp_path / "z_b.npy").exists()

    z_diff = np.load(tmp_path / "z_diff.npy")
    assert z_diff.shape == (40, 8)
    # the saved code IS the projection of the contrast vector e_a - e_b
    prompts = battles["prompt"].tolist()
    e_a = emb.encode(prompts, battles["completion_a"].tolist())
    e_b = emb.encode(prompts, battles["completion_b"].tolist())
    expected = proj.project((e_a - e_b).astype(np.float32))
    assert np.allclose(z_diff, expected)

    meta = pd.read_parquet(tmp_path / "battles.parquet")
    assert list(meta["instruction_id"]) == list(battles["instruction_id"])


def test_build_lens_individual_mode(tmp_path):
    battles = _battles()
    out = build_lens(
        battles, FakeEmbedder(), tmp_path, input_rep="individual",
        m_total=8, k=2, matryoshka_prefix=(4,),
        n_epochs=3, min_epochs=3, patience=3, batch=16, device="cpu", seed=0)
    assert out["input_rep"] == "individual"
    assert out["output_arrays"] == ["z_a", "z_b", "z_diff"]
    assert (tmp_path / "sae_model.pt").exists()
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "sae_training_log.csv").exists()
    for name in ("z_a.npy", "z_b.npy", "z_diff.npy"):
        assert (tmp_path / name).exists()
    za = np.load(tmp_path / "z_a.npy")
    zb = np.load(tmp_path / "z_b.npy")
    zd = np.load(tmp_path / "z_diff.npy")
    assert za.shape == (40, 8) and zb.shape == (40, 8) and zd.shape == (40, 8)
    assert np.allclose(zd, za - zb)


def test_build_lens_individual_single_response_mode(tmp_path):
    items = _battles().drop(columns=["completion_b", "model_b"])
    emb_dir = tmp_path / "emb"
    out = build_lens(
        items, FakeEmbedder(), tmp_path / "lens", input_rep="individual",
        dump_embeddings=emb_dir, m_total=8, k=2, matryoshka_prefix=(4,),
        n_epochs=3, min_epochs=3, patience=3, batch=16, device="cpu", seed=0)

    assert out["dataset_mode"] == "single"
    assert out["output_arrays"] == ["z_a"]
    assert np.load(tmp_path / "lens" / "z_a.npy").shape == (40, 8)
    assert not (tmp_path / "lens" / "z_b.npy").exists()
    assert not (tmp_path / "lens" / "z_diff.npy").exists()
    saved = pd.read_parquet(tmp_path / "lens" / "battles.parquet")
    assert {"instruction_id", "prompt", "completion_a"} <= set(saved.columns)
    assert (emb_dir / "e_a.npy").exists() and not (emb_dir / "e_b.npy").exists()

    swept = build_lens_from_embeddings(
        emb_dir, tmp_path / "swept", input_rep="individual",
        m_total=8, k=2, matryoshka_prefix=(4,), n_epochs=3, min_epochs=3,
        patience=3, batch=16, device="cpu", seed=0)
    assert swept["dataset_mode"] == "single"
    assert np.load(tmp_path / "swept" / "z_a.npy").shape == (40, 8)


def test_dump_and_train_from_embeddings_matches(tmp_path):
    """A lens trained from dumped embeddings equals one trained inline (same seed)."""
    battles = _battles()
    emb_dir = tmp_path / "emb"
    direct = tmp_path / "direct"
    swept = tmp_path / "swept"
    kw = dict(m_total=8, k=2, matryoshka_prefix=(4,), n_epochs=3, min_epochs=3,
              patience=3, batch=16, device="cpu", seed=0)

    # build inline + dump the embeddings
    build_lens(battles, FakeEmbedder(), direct, dump_embeddings=emb_dir, **kw)
    assert (emb_dir / "e_a.npy").exists() and (emb_dir / "meta.parquet").exists()

    # retrain from the dump (no embedder) — must reproduce the same codes
    build_lens_from_embeddings(emb_dir, swept, **kw)
    assert np.allclose(np.load(direct / "z_diff.npy"), np.load(swept / "z_diff.npy"))

    # and a different M works straight from the dump
    out32 = build_lens_from_embeddings(emb_dir, tmp_path / "m4", **{**kw, "m_total": 4})
    assert out32["m_total"] == 4
    assert np.load(tmp_path / "m4" / "z_diff.npy").shape == (40, 4)


def test_build_lens_from_memmap_matches_in_memory(tmp_path):
    """build_lens_from_embeddings memmaps the dump; codes must equal a build that
    loaded the dump fully into RAM."""
    import numpy as np

    battles = _battles()
    emb_dir = tmp_path / "emb"
    kw = dict(m_total=8, k=2, matryoshka_prefix=(4,), n_epochs=3, min_epochs=3,
              patience=3, batch=16, device="cpu", seed=0)
    build_lens(battles, FakeEmbedder(), tmp_path / "seed", dump_embeddings=emb_dir,
               **kw)

    # the production path (memmap'd)
    out_mm = tmp_path / "mm"
    build_lens_from_embeddings(emb_dir, out_mm, **kw)
    z_mm = np.load(out_mm / "z_diff.npy")
    assert z_mm.shape == (40, 8)

    # a reference build that forces the full arrays into RAM first
    from prefscope.pipeline.build_lens import _train_and_save
    e_a = np.array(np.load(emb_dir / "e_a.npy"))   # real (non-memmap) copy
    e_b = np.array(np.load(emb_dir / "e_b.npy"))
    meta = pd.read_parquet(emb_dir / "meta.parquet")
    out_ram = tmp_path / "ram"
    _train_and_save(e_a, e_b, meta, out_ram, input_rep="difference",
                    val_frac=0.1, embed_model_id=None, **kw)
    z_ram = np.load(out_ram / "z_diff.npy")
    assert np.allclose(z_mm, z_ram)


def test_build_lens_max_train_rows_caps_trained_rows(tmp_path):
    """A reservoir cap < n_train trains on at most cap rows (per the manifest)."""
    battles = _battles(n=80)
    emb_dir = tmp_path / "emb"
    kw = dict(m_total=8, k=2, matryoshka_prefix=(4,), n_epochs=2, min_epochs=2,
              patience=2, batch=16, device="cpu", seed=0)
    build_lens(battles, FakeEmbedder(), tmp_path / "seed", dump_embeddings=emb_dir,
               **kw)

    out = build_lens_from_embeddings(emb_dir, tmp_path / "capped",
                                     max_train_rows=10, **kw)
    assert out["n_train_rows_used"] == 10
    assert out["n_train_rows_used"] <= out["n_train_battles"]
    # z arrays still cover every battle (projection is independent of the cap)
    z = np.load(tmp_path / "capped" / "z_diff.npy")
    assert z.shape[0] == out["n_battles"]


def test_build_lens_rejects_bad_input_rep(tmp_path):
    with pytest.raises(ValueError, match="input_rep"):
        build_lens(_battles(), FakeEmbedder(), tmp_path, input_rep="diff",
                   m_total=8, k=2, n_epochs=2, device="cpu")


def test_build_lens_rejects_prompt_rep_before_embedding(tmp_path):
    class CountingEmbedder(FakeEmbedder):
        calls = 0

        def encode(self, prompts, completions):
            CountingEmbedder.calls += 1
            return super().encode(prompts, completions)

    emb = CountingEmbedder()
    with pytest.raises(ValueError, match="contrastive"):
        build_lens(_battles(), emb, tmp_path, input_rep="prompt",
                   m_total=8, k=2, n_epochs=2, device="cpu")
    assert CountingEmbedder.calls == 0   # rejected before the costly embed


def test_build_lens_rejects_tiny_val(tmp_path):
    with pytest.raises(ValueError):
        build_lens(_battles(1), FakeEmbedder(), tmp_path,
                   m_total=8, k=2, n_epochs=2, device="cpu")


def test_build_difference_lens_rejects_single_response_data(tmp_path):
    bad = pd.DataFrame([{"instruction_id": "1", "prompt": "p",
                         "completion_a": "a"}])  # no completion_b
    with pytest.raises(ValueError, match="requires paired data with completion_b"):
        build_lens(bad, FakeEmbedder(), tmp_path, m_total=8, k=2,
                   n_epochs=2, device="cpu")


def test_build_lens_rejects_mixed_pair_and_single_rows_before_embedding(tmp_path):
    bad = _battles(4)
    bad.loc[0, "completion_b"] = None

    class CountingEmbedder(FakeEmbedder):
        calls = 0
        def encode(self, prompts, completions):
            CountingEmbedder.calls += 1
            return super().encode(prompts, completions)

    with pytest.raises(ValueError, match="mixed paired/single"):
        build_lens(bad, CountingEmbedder(), tmp_path, input_rep="individual",
                   m_total=8, k=2, n_epochs=2, device="cpu")
    assert CountingEmbedder.calls == 0
