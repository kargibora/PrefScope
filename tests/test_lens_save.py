"""Lens.save must stage a whole-directory replacement and never produce a hybrid lens.

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
    lens.input_rep = "individual"
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


def test_save_can_bundle_external_annotations_without_merging(tmp_path):
    src = _fake_lens_dir(tmp_path / "src")
    interpret = tmp_path / "interpret"
    interpret.mkdir()
    (interpret / "feature_fidelity.csv").write_text(
        "feature_id,concept,fidelity_pass\n0,new,True\n")
    (interpret / "feature_calibration.csv").write_text(
        "feature_id,semantic_threshold,presence_pass\n0,0.4,True\n")
    (interpret / "feature_roles.csv").write_text(
        "feature_id,semantic_role\n0,presentation\n")

    dest = tmp_path / "release"
    _lens_with_dir(src).save(dest, annotations=interpret)

    assert (dest / "feature_fidelity.csv").exists()
    assert (dest / "feature_calibration.csv").exists()
    assert (dest / "feature_roles.csv").exists()


def test_save_inference_only_omits_corpus_codes_and_marks_manifest(tmp_path):
    import json

    src = _fake_lens_dir(tmp_path / "src")
    (src / "manifest.json").write_text(json.dumps({
        "input_rep": "individual",
        "output_arrays": ["z_a", "z_b", "z_diff"],
    }))
    (src / "z_a.npy").write_bytes(b"large codes")
    (src / "battles.parquet").write_bytes(b"private corpus")
    (src / "whiten.npz").write_bytes(b"transform")

    dest = tmp_path / "hub"
    _lens_with_dir(src).save(dest, inference_only=True)

    assert (dest / "sae_model.pt").exists() and (dest / "whiten.npz").exists()
    assert (dest / "feature_names.csv").exists()
    assert not (dest / "z_a.npy").exists()
    assert not (dest / "battles.parquet").exists()
    manifest = json.loads((dest / "manifest.json").read_text())
    assert manifest["artifact_scope"] == "inference"
    assert manifest["output_arrays"] == []
    assert manifest["source_output_arrays"] == ["z_a", "z_b", "z_diff"]


def test_save_inference_only_migrates_prompt_manifest_and_omits_codes(tmp_path):
    import json

    src = _fake_lens_dir(tmp_path / "prompt")
    (src / "manifest.json").write_text(json.dumps({
        "schema_version": 1, "lens_kind": "prompt", "input_rep": "prompt",
        "m_total": 16, "k": 4, "input_dim": 8,
        "matryoshka_prefix_lengths": [], "output_arrays": ["z_prompt"],
        "embed_model_id": "example/embedder", "sae_type": "batchtopk-relu",
    }))
    (src / "z_prompt.npy").write_bytes(b"private corpus codes")

    dest = tmp_path / "prompt-hub"
    lens = _lens_with_dir(src)
    lens.input_rep = "prompt"
    lens.save(dest, inference_only=True)

    manifest = json.loads((dest / "manifest.json").read_text())
    assert manifest["schema_version"] == 2
    assert manifest["artifact_scope"] == "inference"
    assert manifest["output_arrays"] == []
    assert manifest["source_output_arrays"] == ["z_prompt"]
    assert manifest["activation_polarity"] == "nonnegative"
    assert not (dest / "z_prompt.npy").exists()


def test_repackaging_inference_artifact_preserves_array_provenance(tmp_path):
    import json

    src = _fake_lens_dir(tmp_path / "source")
    (src / "manifest.json").write_text(json.dumps({
        "input_rep": "individual", "output_arrays": [],
        "artifact_scope": "inference",
        "source_output_arrays": ["z_a", "z_b", "z_diff"],
    }))
    dest = tmp_path / "repacked"
    _lens_with_dir(src).save(dest, inference_only=True)
    manifest = json.loads((dest / "manifest.json").read_text())
    assert manifest["source_output_arrays"] == ["z_a", "z_b", "z_diff"]
