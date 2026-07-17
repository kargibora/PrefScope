"""Lens.save must be atomic and never produce a hybrid lens.

The old copytree(dirs_exist_ok=True) merged into an existing dir, so a stale file from a
previous artifact could survive next to the new one. save() now refuses a non-empty dest
unless overwrite=True, and when it does write it replaces the dest wholesale.
"""
from __future__ import annotations

import pytest

from prefscope.api.loaded_lens import Lens


def _fake_lens_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    (path / "sae_model.pt").write_bytes(b"weights")
    (path / "manifest.json").write_text("{}")
    (path / "feature_names.csv").write_text("feature_id,concept\n0,new\n")
    return path


def _lens_with_dir(src):
    lens = Lens.__new__(Lens)          # bypass full init; save() only needs lens_dir
    lens.lens_dir = src
    return lens


def test_save_refuses_nonempty_dest_without_overwrite(tmp_path):
    lens = _lens_with_dir(_fake_lens_dir(tmp_path / "src"))
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "stale.txt").write_text("old artifact leftover")
    with pytest.raises(FileExistsError):
        lens.save(dest)


def test_save_overwrite_replaces_wholesale_no_hybrid(tmp_path):
    lens = _lens_with_dir(_fake_lens_dir(tmp_path / "src"))
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "stale.txt").write_text("old artifact leftover")
    (dest / "feature_names.csv").write_text("feature_id,concept\n0,STALE\n")

    lens.save(dest, overwrite=True)

    assert not (dest / "stale.txt").exists()                 # stale file gone
    assert (dest / "sae_model.pt").read_bytes() == b"weights"
    assert "new" in (dest / "feature_names.csv").read_text()  # replaced, not merged
    assert "STALE" not in (dest / "feature_names.csv").read_text()


def test_save_into_empty_dest_ok(tmp_path):
    lens = _lens_with_dir(_fake_lens_dir(tmp_path / "src"))
    dest = tmp_path / "empty_dest"
    dest.mkdir()                                             # exists but empty → allowed
    out = lens.save(dest)
    assert (out / "sae_model.pt").exists()


def test_save_same_dir_is_noop(tmp_path):
    src = _fake_lens_dir(tmp_path / "src")
    lens = _lens_with_dir(src)
    assert lens.save(src) == src
    assert (src / "sae_model.pt").exists()
