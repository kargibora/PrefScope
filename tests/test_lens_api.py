"""Tests for the public Lens object and its feature-extraction helpers."""

import os

import numpy as np
import pandas as pd
import pytest

from prefscope.api.loaded_lens import Lens, pairs_to_battles
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
        return np.stack([x[:, 0], -x[:, 0], np.zeros(len(x))], axis=1).astype(
            np.float32
        )


def _names():
    return pd.DataFrame(
        {
            "feature_id": [0, 1, 2],
            "concept": ["a", "b", "c"],
            "fidelity_pass": [True, False, True],
        }
    )


def _lens(manifest=None, names=None):
    return Lens(
        FakeProjector(),
        FakeEmbedder(),
        names=names,
        manifest=manifest or {"input_rep": "individual"},
    )


# ---- loading -------------------------------------------------------------


def test_load_alias_matches_from_dir():
    assert Lens.load.__func__ is Lens.from_dir.__func__


def test_constructed_directly_has_no_lens_dir():
    assert _lens().lens_dir is None


# ---- encode --------------------------------------------------------------


# ---- concept_names / top_concepts ---------------------------------------


def test_concept_names_maps_ids():
    s = _lens(names=_names()).concept_names
    assert s.loc[0] == "a" and s.loc[1] == "b" and s.loc[2] == "c"
    assert _lens().concept_names is None


def test_concept_names_dedupes_duplicate_feature_id():
    names = pd.DataFrame({"feature_id": [0, 0, 1], "concept": ["a", "a2", "b"]})
    lens = _lens(names=names)
    s = lens.concept_names
    assert s.index.is_unique
    assert s.to_dict() == {0: "a", 1: "b"}


# ---- paired encoding ----------------------------------------------------


class _PairData(Dataset):
    def __iter__(self):
        yield PairItem(
            id="1", x="q", y_a="aaaa", y_b="b", pref=1.0, model_a="m1", model_b="m2"
        )
        yield PairItem(
            id="2", x="q", y_a="a", y_b="bbbb", pref=0.0, model_a="m1", model_b="m2"
        )


class _SingleData(Dataset):
    def __iter__(self):
        yield PairItem(id="s1", x="q", y_a="aaaa", model_a="m1")
        yield PairItem(id="s2", x="q", y_a="bb", model_a="m2")


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


def test_save_rejects_source_destination_overlap_and_file_destinations(tmp_path):
    src = tmp_path / "source"
    src.mkdir()
    (src / "manifest.json").write_text('{"input_rep": "individual"}')
    lens = _lens()
    lens.lens_dir = src
    with pytest.raises(ValueError, match="ancestors or descendants"):
        lens.save(src / "nested")
    with pytest.raises(ValueError, match="ancestors or descendants"):
        lens.save(tmp_path, overwrite=True)
    destination_file = tmp_path / "artifact.bin"
    destination_file.write_bytes(b"old")
    with pytest.raises(ValueError, match="directory path"):
        lens.save(destination_file, overwrite=True)


def test_publication_lock_cleans_up_when_initial_write_fails(tmp_path, monkeypatch):
    import fcntl
    import os
    import stat

    import prefscope.api.loaded_lens as loaded_lens

    destination = tmp_path / "lens"
    lock = tmp_path / ".lens.lock"

    def fail_write(*args, **kwargs):
        raise OSError("injected lock write failure")

    monkeypatch.setattr(loaded_lens.os, "write", fail_write)
    with pytest.raises(OSError, match="injected"):
        with loaded_lens._publication_lock(destination):
            pass

    assert lock.exists()
    assert stat.S_ISREG(lock.stat(follow_symlinks=False).st_mode)
    assert stat.S_IMODE(lock.stat().st_mode) == 0o600
    descriptor = os.open(lock, os.O_RDWR | os.O_NOFOLLOW)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def test_publication_lock_rejects_hard_link_without_modifying_target(tmp_path):
    from prefscope.api import loaded_lens

    destination = tmp_path / "lens"
    lock = tmp_path / ".lens.lock"
    victim = tmp_path / "victim.txt"
    victim.write_text("do not modify")
    try:
        os.link(victim, lock)
    except OSError as exc:
        pytest.skip(f"hard links unavailable: {exc}")

    with pytest.raises(RuntimeError, match="exactly one hard link"):
        with loaded_lens._publication_lock(destination):
            pass

    assert victim.read_text() == "do not modify"


def test_publication_lock_is_persistent_advisory_and_released(tmp_path):
    import fcntl
    import os

    import prefscope.api.loaded_lens as loaded_lens

    destination = tmp_path / "lens"
    lock = tmp_path / ".lens.lock"
    with loaded_lens._publication_lock(destination):
        descriptor = os.open(lock, os.O_RDWR | os.O_NOFOLLOW)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(descriptor)

    assert lock.exists()
    descriptor = os.open(lock, os.O_RDWR | os.O_NOFOLLOW)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def test_publication_lock_rejects_symlink_and_nonregular_file(tmp_path):
    import os

    import prefscope.api.loaded_lens as loaded_lens

    destination = tmp_path / "lens"
    lock = tmp_path / ".lens.lock"
    target = tmp_path / "target"
    target.write_text("unchanged")
    lock.symlink_to(target)
    with pytest.raises(RuntimeError, match="securely open"):
        with loaded_lens._publication_lock(destination):
            pass
    assert target.read_text() == "unchanged"

    lock.unlink()
    if hasattr(os, "mkfifo"):
        os.mkfifo(lock)
        with pytest.raises(RuntimeError, match="regular file"):
            with loaded_lens._publication_lock(destination):
                pass


# ---- pairs_to_battles ----------------------------------------------------


def test_pairs_to_battles_empty_preserves_canonical_schema():
    frame = pairs_to_battles([])
    assert list(frame.columns) == [
        "instruction_id",
        "prompt",
        "completion_a",
        "completion_b",
        "human_pref",
        "model_a",
        "model_b",
    ]


def test_pairs_to_battles_from_pairitems():
    pairs = [
        PairItem(
            id="1", x="q1", y_a="a1", y_b="b1", pref=1.0, model_a="m1", model_b="m2"
        ),
        PairItem(id="2", x="q2", y_a="a2", y_b="b2", pref=0.0),
    ]
    df = pairs_to_battles(pairs)
    for col in ("prompt", "completion_a", "completion_b", "instruction_id"):
        assert col in df.columns
    assert list(df["prompt"]) == ["q1", "q2"]
    assert list(df["instruction_id"]) == ["1", "2"]
    assert list(df["human_pref"]) == [1.0, 0.0]
    assert df["model_a"].iloc[0] == "m1" and pd.isna(df["model_a"].iloc[1])


def test_pairs_to_battles_from_dataframe_with_rename():
    raw = pd.DataFrame(
        {"q": ["p1"], "ca": ["x"], "cb": ["y"], "iid": ["i1"], "human_pref": [1.0]}
    )
    df = pairs_to_battles(
        raw,
        columns={
            "q": "prompt",
            "ca": "completion_a",
            "cb": "completion_b",
            "iid": "instruction_id",
        },
    )
    for col in ("prompt", "completion_a", "completion_b", "instruction_id"):
        assert col in df.columns
    assert list(df["human_pref"]) == [1.0]


def test_pairs_to_battles_missing_cols_raises():
    raw = pd.DataFrame({"prompt": ["p"]})
    with pytest.raises(ValueError, match="missing required columns"):
        pairs_to_battles(raw)


def test_pairs_to_battles_dataframe_accepts_single_response_rows():
    raw = pd.DataFrame(
        {"prompt": ["p"], "completion_a": ["a"], "instruction_id": ["i"]}
    )
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
    cfg = TrainConfig(
        sae=SAEConfig(m=64, k=8, input_rep="individual", matryoshka_prefix=(4, 16)),
        embed_model_id="emb-x",
        val_frac=0.2,
        device="cpu",
        max_train_rows=123,
        train_kwargs={"epochs": 2},
    )
    result = Lens.train(pairs, cfg, out=tmp_path / "lens")

    assert result == "LOADED"
    assert calls["kw"]["m_total"] == 64
    assert calls["kw"]["k"] == 8
    assert calls["kw"]["input_rep"] == "individual"
    assert calls["kw"]["sae_type"] == "auto"
    assert calls["kw"]["sparsity_coef"] == 1e-3
    assert calls["kw"]["bandwidth"] == 1e-3
    assert calls["kw"]["sparsity_warmup_steps"] == 0
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
    from prefscope.api.config import TrainConfig

    monkeypatch.setattr(bl, "build_lens", lambda *a, **k: {"ok": True})
    monkeypatch.setattr(emb_mod, "Embedder", lambda *a, **k: object())
    monkeypatch.setattr(Lens, "load", classmethod(lambda cls, out, **kw: "LOADED"))

    pairs = [PairItem(id="1", x="q1", y_a="a1", y_b="b1", pref=1.0)]
    cfg = TrainConfig(train_kwargs={"m_total": 999, "k": 3})
    with pytest.raises(ValueError, match="train_kwargs may not override"):
        Lens.train(pairs, cfg, out=tmp_path / "lens")


def test_save_rejects_active_publication_lock(tmp_path):
    import fcntl
    import json
    import os
    import socket
    import uuid

    src = tmp_path / "src-lock"
    src.mkdir()
    (src / "manifest.json").write_text('{"input_rep": "individual"}')
    lens = _lens()
    lens.lens_dir = src
    dest = tmp_path / "published"
    lock = tmp_path / ".published.lock"
    lock.write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "hostname": socket.gethostname(),
                "owner_id": uuid.uuid4().hex,
            }
        )
    )
    descriptor = os.open(lock, os.O_RDWR | os.O_NOFOLLOW)
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        with pytest.raises(RuntimeError, match="another active publisher"):
            lens.save(dest)
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)

    assert not dest.exists()
    assert lock.exists()


def test_save_recovers_sole_orphan_backup_deterministically(tmp_path):
    import uuid

    src = tmp_path / "src-recovery"
    src.mkdir()
    (src / "manifest.json").write_text('{"input_rep": "individual"}')
    lens = _lens()
    lens.lens_dir = src
    dest = tmp_path / "published"
    backup = tmp_path / f".published.bak-{uuid.uuid4().hex}"
    backup.mkdir()
    (backup / "old.txt").write_text("old")

    # Recovery happens before overwrite policy. The recovered prior artifact is
    # preserved, and the caller must still explicitly authorize its replacement.
    with pytest.raises(FileExistsError, match="overwrite=True"):
        lens.save(dest)
    assert (dest / "old.txt").read_text() == "old"
    assert not backup.exists()

    lens.save(dest, overwrite=True)
    assert (dest / "manifest.json").exists()
    assert not (dest / "old.txt").exists()


def test_save_uses_uuid_staging_and_backup_names(tmp_path, monkeypatch):
    import importlib
    import re

    module = importlib.import_module("prefscope.api.loaded_lens")
    src = tmp_path / "src-uuid"
    src.mkdir()
    (src / "manifest.json").write_text('{"input_rep": "individual"}')
    lens = _lens()
    lens.lens_dir = src
    dest = tmp_path / "published"
    dest.mkdir()
    (dest / "old.txt").write_text("old")
    seen = []
    real_replace = module.os.replace

    def capture(source, target):
        seen.append((module.Path(source).name, module.Path(target).name))
        return real_replace(source, target)

    monkeypatch.setattr(module.os, "replace", capture)
    lens.save(dest, overwrite=True)

    names = {name for pair in seen for name in pair}
    assert any(re.fullmatch(r"\.published\.tmp-[0-9a-f]{32}", name) for name in names)
    assert any(re.fullmatch(r"\.published\.bak-[0-9a-f]{32}", name) for name in names)
    lock = tmp_path / ".published.lock"
    assert lock.exists()
    assert lock.is_file() and not lock.is_symlink()
    assert lock.stat().st_mode & 0o777 == 0o600
    assert not list(tmp_path.glob(".published.tmp-*"))
    assert not list(tmp_path.glob(".published.bak-*"))
