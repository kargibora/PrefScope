import hashlib
import json

import pytest

from prefscope import Lens, LensBundle, load_bundle, load_lens
from prefscope.api._bundle_publication import package_bundle
from prefscope.api.lens_bundle import validate_bundle_manifest


def _child(root, name, *, artifact_type=None):
    path = root / name
    path.mkdir()
    manifest = {"schema_version": 1, "input_rep": "prompt", "m_total": 2,
                "artifact_scope": "inference", "output_arrays": []}
    if artifact_type:
        manifest["artifact_type"] = artifact_type
    (path / "manifest.json").write_text(json.dumps(manifest))
    (path / "sae_model.pt").write_bytes(b"weights")
    return path


def _bundle_manifest(root, members):
    descriptors = {}
    for name, path in members.items():
        descriptors[name] = {
            "path": path.name,
            "manifest_sha256": hashlib.sha256((path / "manifest.json").read_bytes()).hexdigest(),
        }
    manifest = {"artifact_type": "lens_bundle", "bundle_schema_version": 1,
                "members": descriptors}
    (root / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def test_local_bundle_is_lazy_cached_and_keeps_member_paths(tmp_path, monkeypatch):
    prompt = _child(tmp_path, "prompt-m256")
    completion = _child(tmp_path, "completion-m2048")
    root = tmp_path / "bundle"
    root.mkdir()
    prompt.rename(root / prompt.name)
    completion.rename(root / completion.name)
    _bundle_manifest(root, {"prompt": root / "prompt-m256", "completion": root / "completion-m2048"})
    loaded = []
    from prefscope.api.loaded_lens import Lens
    monkeypatch.setattr(Lens, "from_dir", classmethod(lambda cls, path, **kwargs: loaded.append(path) or path))

    bundle = LensBundle.from_dir(root)
    assert list(bundle) == ["prompt", "completion"]
    assert bundle["prompt"] is bundle["prompt"]
    assert bundle["completion"] == root / "completion-m2048"
    assert loaded == [root / "prompt-m256", root / "completion-m2048"]
    with pytest.raises(KeyError, match="available"):
        bundle["missing"]


def test_bundle_rejects_unsafe_or_invalid_members(tmp_path):
    child = _child(tmp_path, "child")
    for index, (name, path) in enumerate((("bad", "../child"), ("bad", "/tmp/child"), ("bad", "child\\nested"))):
        root = tmp_path / f"{name}-{index}"
        root.mkdir()
        manifest = {"artifact_type": "lens_bundle", "bundle_schema_version": 1,
                    "members": {name: {"path": path}}}
        with pytest.raises(ValueError):
            validate_bundle_manifest(manifest, root)
    root = tmp_path / "valid"
    root.mkdir()
    manifest = {"artifact_type": "lens_bundle", "bundle_schema_version": 1,
                "members": {"child": {"path": "child", "manifest_sha256": "bad"}}}
    child.rename(root / "child")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_bundle_manifest(manifest, root)


def test_package_bundle_writes_root_manifest_and_validates_children(tmp_path):
    prompt = _child(tmp_path, "prompt")
    completion = _child(tmp_path, "completion")
    out = tmp_path / "out"
    package_bundle({"prompt": prompt, "completion": completion}, out)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["artifact_type"] == "lens_bundle"
    assert set(manifest["members"]) == {"prompt", "completion"}
    assert set(LensBundle.from_dir(out)) == {"prompt", "completion"}


@pytest.mark.parametrize("name", ["../outside", "nested/path", "nested\\path", "/absolute", 123])
def test_package_bundle_rejects_unsafe_names_before_writing(tmp_path, name):
    child = _child(tmp_path, "child")
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="invalid lens bundle member name"):
        package_bundle({name: child}, out)
    assert not out.exists()
    assert not (tmp_path / "outside").exists()
    assert not list(tmp_path.glob(".out.tmp-*"))


@pytest.mark.parametrize("extra", ["private_notes.csv", "raw_text.txt", "nested/private.csv"])
def test_package_bundle_rejects_unknown_child_files(tmp_path, extra):
    child = _child(tmp_path, "child")
    path = child / extra
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("private data")
    out = tmp_path / "out"
    out.mkdir()
    (out / "keep.txt").write_text("old bundle")
    with pytest.raises(ValueError, match="unexpected file"):
        package_bundle({"prompt": child}, out, overwrite=True)
    assert (out / "keep.txt").read_text() == "old bundle"
    assert not (out / "prompt").exists()


@pytest.mark.parametrize("destination_inside_source", [True, False])
def test_package_bundle_rejects_overlapping_source_and_destination(
    tmp_path, destination_inside_source,
):
    if destination_inside_source:
        child = _child(tmp_path, "child")
        out = child / "bundle"
    else:
        out = tmp_path / "bundle"
        out.mkdir()
        child = _child(out, "child")
    with pytest.raises(ValueError, match="must not overlap"):
        package_bundle({"prompt": child}, out)
    assert not (out / "prompt").exists()


def test_package_bundle_keeps_package_lens_artifacts(tmp_path):
    child = _child(tmp_path, "prompt")
    artifacts = (
        "whiten.npz", "README.md", "LICENSE", "prompt_feature_names.csv",
        "prompt_feature_fidelity.csv", "feature_context.csv", "feature_catalog.json",
        "derived_feature_catalog.json",
    )
    for name in artifacts:
        (child / name).write_text("published lens artifact")
    out = tmp_path / "out"
    package_bundle({"prompt": child}, out)
    assert all((out / "prompt" / name).is_file() for name in artifacts)


def test_package_bundle_rejects_symlinked_source(tmp_path):
    child = _child(tmp_path, "child")
    alias = tmp_path / "alias"
    alias.symlink_to(child, target_is_directory=True)
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="real directory"):
        package_bundle({"prompt": alias}, out)
    assert not out.exists()


@pytest.mark.parametrize("change", [
    "bundle_id", "member", "path", "manifest_hash", "missing_manifest_hash",
    "missing_weights_hash",
])
def test_bundle_rejects_tampered_identified_manifest(tmp_path, change):
    child = _child(tmp_path, "child")
    out = tmp_path / "out"
    package_bundle({"prompt": child}, out)
    manifest = json.loads((out / "manifest.json").read_text())
    descriptor = manifest["members"]["prompt"]
    if change == "bundle_id":
        manifest["bundle_id"] = "sha256:" + "0" * 64
    elif change == "member":
        manifest["members"] = {"other": descriptor}
    elif change == "path":
        descriptor["path"] = "other"
    elif change == "manifest_hash":
        descriptor["manifest_sha256"] = "0" * 64
    elif change == "missing_manifest_hash":
        del descriptor["manifest_sha256"]
    else:
        del descriptor["weights_sha256"]
    with pytest.raises(ValueError, match="hashes|bundle ID"):
        validate_bundle_manifest(manifest, out)


@pytest.mark.parametrize("filename", ["sae_model.pt", "whiten.npz"])
def test_bundle_rejects_tampered_lens_weights(tmp_path, filename):
    child = _child(tmp_path, "child")
    (child / "whiten.npz").write_bytes(b"whitener")
    out = tmp_path / "out"
    package_bundle({"prompt": child}, out)
    (out / "prompt" / filename).write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoint hash mismatch|whitener hash mismatch"):
        LensBundle.from_dir(out)


def test_package_bundle_recovers_orphan_backup_before_overwrite_check(tmp_path):
    child = _child(tmp_path, "child")
    orphan = tmp_path / ".out.bak-old"
    orphan.mkdir()
    (orphan / "keep.txt").write_text("old bundle")
    out = tmp_path / "out"
    with pytest.raises(FileExistsError, match="exists and is not empty"):
        package_bundle({"prompt": child}, out)
    assert (out / "keep.txt").read_text() == "old bundle"
    assert not orphan.exists()
    assert not (out / "prompt").exists()


@pytest.mark.parametrize("filename", ["manifest.json", "sae_model.pt", "whiten.npz"])
def test_bundle_rejects_child_file_symlink_outside_root(tmp_path, filename):
    child = _child(tmp_path, "child")
    (child / "whiten.npz").write_bytes(b"whitener")
    out = tmp_path / "out"
    package_bundle({"prompt": child}, out)
    path = out / "prompt" / filename
    outside = tmp_path / f"outside-{filename}"
    outside.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        LensBundle.from_dir(out)


def test_load_bundle_local_entrypoint(tmp_path):
    child = _child(tmp_path, "prompt")
    root = tmp_path / "bundle"
    root.mkdir()
    child.rename(root / "prompt")
    _bundle_manifest(root, {"prompt": root / "prompt"})
    assert isinstance(load_bundle(root), LensBundle)


def test_single_lens_loader_rejects_bundle_root(tmp_path):
    root = tmp_path / "bundle"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps({"artifact_type": "lens_bundle"}))
    for loader in (Lens.from_dir, load_lens):
        with pytest.raises(ValueError, match="lens bundle"):
            loader(root)
