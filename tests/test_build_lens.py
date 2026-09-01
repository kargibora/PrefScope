import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from prefscope.pipeline.build_lens import (
    build_lens, build_lens_from_embeddings, build_prompt_lens)
from prefscope.encode.sae import SAEProjector


class FakeEmbedder:
    """Deterministic embeddings keyed on (prompt, completion) text."""
    dim = 8
    model_id = "test/embedder"

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
    assert out["sae_type"] == "batchtopk"
    assert out["activation_polarity"] == "signed"
    assert out["code_semantics"] == "axis"
    assert len(out["dataset_hash"]) == 64

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
    assert out["sae_type"] == "batchtopk-relu"
    assert out["activation_polarity"] == "nonnegative"
    assert out["code_semantics"] == "presence"
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
    assert (za >= 0).all() and (zb >= 0).all()
    assert np.allclose(zd, za - zb)


def test_build_prompt_lens_auto_uses_nonnegative_sae_and_no_matryoshka(tmp_path):
    emb_dir = tmp_path / "prompt-emb"
    emb_dir.mkdir()
    rng = np.random.default_rng(4)
    np.save(emb_dir / "e_prompt.npy", rng.standard_normal((40, 8)).astype(np.float32))
    pd.DataFrame({"instruction_id": [str(i) for i in range(40)]}).to_parquet(
        emb_dir / "meta.parquet")

    out_dir = tmp_path / "prompt-lens"
    out_dir.mkdir()
    # A prompt build is unwhitened and must not auto-load preprocessing left by an
    # older lens in the same directory.
    np.savez(out_dir / "whiten.npz", method=np.array("standardize"),
             mean=np.zeros(3), std=np.ones(3))
    (out_dir / "stale.txt").write_text("old prompt lens")
    out = build_prompt_lens(
        emb_dir, out_dir, m_total=8, k=2, n_epochs=2, min_epochs=2,
        patience=2, batch=16, device="cpu", seed=0, embed_model_id="test/embed")
    assert out["input_rep"] == "prompt"
    assert out["sae_type"] == "batchtopk-relu"
    assert out["activation_polarity"] == "nonnegative"
    assert out["matryoshka_prefix_lengths"] == []
    assert len(out["dataset_hash"]) == 64
    assert (np.load(out_dir / "z_prompt.npy") >= 0).all()
    assert not (out_dir / "whiten.npz").exists()
    assert not (out_dir / "stale.txt").exists()


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
    swept_manifest = build_lens_from_embeddings(emb_dir, swept, **kw)
    assert np.allclose(np.load(direct / "z_diff.npy"), np.load(swept / "z_diff.npy"))
    direct_manifest = json.loads((direct / "manifest.json").read_text())
    assert direct_manifest["dataset_hash"] == swept_manifest["dataset_hash"]

    # and a different M works straight from the dump
    out32 = build_lens_from_embeddings(emb_dir, tmp_path / "m4", **{**kw, "m_total": 4})
    assert out32["m_total"] == 4
    assert out32["dataset_hash"] == direct_manifest["dataset_hash"]
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
                    val_frac=0.1, embed_model_id=FakeEmbedder.model_id, **kw)
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


@pytest.mark.parametrize(
    "bad",
    [np.ones(8, dtype=np.float32), np.ones((39, 8), dtype=np.float32),
     np.full((40, 8), np.nan, dtype=np.float32)],
)
def test_build_lens_rejects_invalid_embedding_shape_rows_or_values(tmp_path, bad):
    class BadEmbedder(FakeEmbedder):
        def encode(self, prompts, completions):
            return bad

    with pytest.raises(ValueError, match="2-D|rows|non-finite"):
        build_lens(
            _battles(), BadEmbedder(), tmp_path, input_rep="individual",
            m_total=8, k=2, n_epochs=2, device="cpu")


def test_build_lens_from_embeddings_rejects_metadata_misalignment(tmp_path):
    emb_dir = tmp_path / "emb"
    emb_dir.mkdir()
    np.save(emb_dir / "e_a.npy", np.ones((4, 8), dtype=np.float32))
    np.save(emb_dir / "e_b.npy", np.ones((4, 8), dtype=np.float32))
    _battles(3).to_parquet(emb_dir / "meta.parquet")

    with pytest.raises(ValueError, match="metadata has 3 rows"):
        build_lens_from_embeddings(
            emb_dir, tmp_path / "lens", m_total=8, k=2, device="cpu")


def test_build_prompt_lens_rejects_nonfinite_or_misaligned_embeddings(tmp_path):
    emb_dir = tmp_path / "prompt-emb"
    emb_dir.mkdir()
    values = np.ones((4, 8), dtype=np.float32)
    values[0, 0] = np.inf
    np.save(emb_dir / "e_prompt.npy", values)
    pd.DataFrame({"instruction_id": ["0", "1", "2", "3"]}).to_parquet(
        emb_dir / "meta.parquet")

    with pytest.raises(ValueError, match="e_prompt contains non-finite"):
        build_prompt_lens(
            emb_dir, tmp_path / "lens", m_total=8, k=2, device="cpu")


def test_build_lens_rejects_nonfinite_projection(tmp_path, monkeypatch):
    import importlib

    module = importlib.import_module("prefscope.pipeline.build_lens")

    class BadProjector:
        def __init__(self, model_path, device="cpu"):
            self.calls = 0

        def project(self, values):
            self.calls += 1
            fill = 1.0 if self.calls == 1 else np.inf
            return np.full((len(values), 8), fill, dtype=np.float32)

    monkeypatch.setattr(module, "SAEProjector", BadProjector)
    with pytest.raises(ValueError, match="contains non-finite"):
        build_lens(
            _battles(), FakeEmbedder(), tmp_path, input_rep="individual",
            m_total=8, k=2, n_epochs=2, min_epochs=2, patience=2,
            batch=16, device="cpu", seed=0)


def test_unwhitened_build_removes_stale_whitener(tmp_path):
    out_dir = tmp_path / "lens"
    out_dir.mkdir()
    np.savez(out_dir / "whiten.npz", method=np.array("standardize"),
             mean=np.zeros(3), std=np.ones(3))

    build_lens(
        _battles(), FakeEmbedder(), out_dir, input_rep="individual",
        m_total=8, k=2, n_epochs=2, min_epochs=2, patience=2,
        batch=16, device="cpu", seed=0)

    assert not (out_dir / "whiten.npz").exists()


def test_rebuild_replaces_whole_completion_directory(tmp_path):
    out_dir = tmp_path / "lens"
    out_dir.mkdir()
    (out_dir / "stale.txt").write_text("old artifact")
    np.save(out_dir / "z_prompt.npy", np.ones((1, 1), dtype=np.float32))

    manifest = build_lens(
        _battles(), FakeEmbedder(), out_dir, input_rep="difference",
        m_total=8, k=2, n_epochs=2, min_epochs=2, patience=2,
        batch=16, device="cpu", seed=0)

    expected = {
        "manifest.json", "sae_model.pt", "sae_training_log.csv",
        "battles.parquet", "z_diff.npy",
    }
    assert {path.name for path in out_dir.iterdir()} == expected
    assert json.loads((out_dir / "manifest.json").read_text()) == manifest


def test_failed_completion_build_preserves_existing_destination(tmp_path, monkeypatch):
    import importlib

    module = importlib.import_module("prefscope.pipeline.build_lens")
    out_dir = tmp_path / "lens"
    out_dir.mkdir()
    (out_dir / "keep.txt").write_text("previous lens")
    emb_dir = tmp_path / "embedding-dump"

    def fail_training(*args, **kwargs):
        raise RuntimeError("synthetic training failure")

    monkeypatch.setattr(module, "train_sae", fail_training)
    with pytest.raises(RuntimeError, match="synthetic training failure"):
        build_lens(
            _battles(), FakeEmbedder(), out_dir, input_rep="difference",
            dump_embeddings=emb_dir, m_total=8, k=2,
            n_epochs=2, device="cpu", seed=0)

    assert {path.name for path in out_dir.iterdir()} == {"keep.txt"}
    assert (out_dir / "keep.txt").read_text() == "previous lens"
    assert {path.name for path in emb_dir.iterdir()} == {
        "e_a.npy", "e_b.npy", "meta.parquet", "embedding_manifest.json"}
    assert not list(tmp_path.glob(".lens.tmp-*"))
    assert not list(tmp_path.glob(".lens.bak-*"))


def test_failed_prompt_build_preserves_existing_destination(tmp_path, monkeypatch):
    import importlib

    module = importlib.import_module("prefscope.pipeline.build_lens")
    emb_dir = tmp_path / "prompt-emb"
    emb_dir.mkdir()
    np.save(emb_dir / "e_prompt.npy", np.ones((40, 8), dtype=np.float32))
    pd.DataFrame({"instruction_id": [str(i) for i in range(40)]}).to_parquet(
        emb_dir / "meta.parquet")
    out_dir = tmp_path / "prompt-lens"
    out_dir.mkdir()
    (out_dir / "keep.txt").write_text("previous prompt lens")

    monkeypatch.setattr(
        module, "train_sae",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("prompt failure")))
    with pytest.raises(RuntimeError, match="prompt failure"):
        build_prompt_lens(
            emb_dir, out_dir, m_total=8, k=2, n_epochs=2,
            device="cpu", seed=0, embed_model_id="test/embed")

    assert {path.name for path in out_dir.iterdir()} == {"keep.txt"}
    assert (out_dir / "keep.txt").read_text() == "previous prompt lens"


def test_transaction_commit_failure_rolls_back_destination(tmp_path, monkeypatch):
    import importlib

    module = importlib.import_module("prefscope.pipeline.build_lens")
    destination = tmp_path / "lens"
    destination.mkdir()
    (destination / "old.txt").write_text("old")
    real_replace = module.os.replace

    def replace_with_failed_commit(source, target):
        source = module.Path(source)
        target = module.Path(target)
        if source.name.startswith(".lens.tmp-") and target == destination:
            raise OSError("synthetic commit failure")
        return real_replace(source, target)

    monkeypatch.setattr(module.os, "replace", replace_with_failed_commit)

    def builder(staging):
        (staging / "new.txt").write_text("new")
        return {"ok": True}

    with pytest.raises(OSError, match="synthetic commit failure"):
        module._transactional_build(destination, builder)

    assert {path.name for path in destination.iterdir()} == {"old.txt"}
    assert (destination / "old.txt").read_text() == "old"
    assert not list(tmp_path.glob(".lens.tmp-*"))
    assert not list(tmp_path.glob(".lens.bak-*"))


def test_dataset_hash_binds_ordered_metadata_and_arrays():
    from prefscope.pipeline.build_lens import _ordered_dataset_hash

    metadata = pd.DataFrame({"instruction_id": ["a", "b"], "model": ["x", "y"]})
    values = np.arange(8, dtype=np.float32).reshape(2, 4)
    baseline = _ordered_dataset_hash(metadata, {"e_a": values})

    reordered = _ordered_dataset_hash(
        metadata.iloc[::-1].reset_index(drop=True), {"e_a": values[::-1]})
    metadata_changed = metadata.copy()
    metadata_changed.loc[0, "model"] = "other"
    rebound_metadata = _ordered_dataset_hash(metadata_changed, {"e_a": values})
    changed_values = values.copy()
    changed_values[0, 0] += 1
    rebound_values = _ordered_dataset_hash(metadata, {"e_a": changed_values})

    assert len(baseline) == 64
    assert len({baseline, reordered, rebound_metadata, rebound_values}) == 4



def test_dataset_hash_rejects_lossy_array_dtypes_and_ambiguous_metadata():
    from prefscope.core.provenance import ordered_dataset_hash

    metadata = pd.DataFrame({"id": ["a"]})
    with pytest.raises(ValueError, match="float32"):
        ordered_dataset_hash(metadata, {"x": np.array([[1.0]], dtype=np.float64)})
    with pytest.raises(ValueError, match="mapping keys must be strings"):
        ordered_dataset_hash(
            pd.DataFrame({"meta": [{1: "integer", "1": "string"}]}),
            {"x": np.array([[1.0]], dtype=np.float32)},
        )


def test_transaction_rejects_active_publication_lock(tmp_path):
    import importlib
    import socket
    import uuid

    module = importlib.import_module("prefscope.pipeline.build_lens")
    destination = tmp_path / "lens"
    lock = tmp_path / ".lens.lock"
    lock.write_text(json.dumps({
        "pid": module.os.getpid(),
        "hostname": socket.gethostname(),
        "owner_id": uuid.uuid4().hex,
    }))
    called = False

    def builder(staging):
        nonlocal called
        called = True

    with pytest.raises(RuntimeError, match="another active publisher"):
        module._transactional_build(destination, builder)
    assert called is False
    assert lock.exists()


def test_transaction_removes_identifiable_stale_lock(tmp_path, monkeypatch):
    import importlib
    import socket
    import uuid

    module = importlib.import_module("prefscope.pipeline.build_lens")
    destination = tmp_path / "lens"
    lock = tmp_path / ".lens.lock"
    lock.write_text(json.dumps({
        "pid": 999999,
        "hostname": socket.gethostname(),
        "owner_id": uuid.uuid4().hex,
    }))
    monkeypatch.setattr(module, "_pid_is_alive", lambda pid: False)

    module._transactional_build(
        destination,
        lambda staging: (staging / "new.txt").write_text("new"),
    )

    assert (destination / "new.txt").read_text() == "new"
    assert not lock.exists()


def test_transaction_refuses_unidentifiable_lock(tmp_path):
    import importlib

    module = importlib.import_module("prefscope.pipeline.build_lens")
    lock = tmp_path / ".lens.lock"
    lock.write_text("not valid lock metadata")

    with pytest.raises(RuntimeError, match="refusing to remove"):
        module._transactional_build(tmp_path / "lens", lambda staging: None)
    assert lock.read_text() == "not valid lock metadata"


def test_transaction_recovers_sole_orphan_backup_before_build(tmp_path):
    import importlib
    import uuid

    module = importlib.import_module("prefscope.pipeline.build_lens")
    destination = tmp_path / "lens"
    backup = tmp_path / f".lens.bak-{uuid.uuid4().hex}"
    backup.mkdir()
    (backup / "old.txt").write_text("old")

    def fail(staging):
        assert (destination / "old.txt").read_text() == "old"
        raise RuntimeError("stop after recovery")

    with pytest.raises(RuntimeError, match="stop after recovery"):
        module._transactional_build(destination, fail)
    assert (destination / "old.txt").read_text() == "old"
    assert not backup.exists()
    assert not (tmp_path / ".lens.lock").exists()


def test_transaction_uses_uuid_staging_and_backup_names(tmp_path, monkeypatch):
    import importlib
    import re

    module = importlib.import_module("prefscope.pipeline.build_lens")
    destination = tmp_path / "lens"
    destination.mkdir()
    (destination / "old.txt").write_text("old")
    seen = []
    real_replace = module.os.replace

    def capture(source, target):
        seen.append((module.Path(source).name, module.Path(target).name))
        return real_replace(source, target)

    monkeypatch.setattr(module.os, "replace", capture)
    module._transactional_build(
        destination,
        lambda staging: (staging / "new.txt").write_text("new"),
    )

    names = {name for pair in seen for name in pair}
    assert any(re.fullmatch(r"\.lens\.tmp-[0-9a-f]{32}", name) for name in names)
    assert any(re.fullmatch(r"\.lens\.bak-[0-9a-f]{32}", name) for name in names)



def test_transaction_refuses_ambiguous_orphan_backups(tmp_path):
    import importlib
    import uuid

    module = importlib.import_module("prefscope.pipeline.build_lens")
    for _ in range(2):
        backup = tmp_path / f".lens.bak-{uuid.uuid4().hex}"
        backup.mkdir()
        (backup / "old.txt").write_text("old")
    called = False

    def builder(staging):
        nonlocal called
        called = True

    with pytest.raises(RuntimeError, match="multiple orphan backups"):
        module._transactional_build(tmp_path / "lens", builder)
    assert called is False
    assert len(list(tmp_path.glob(".lens.bak-*"))) == 2
    assert not (tmp_path / ".lens.lock").exists()
