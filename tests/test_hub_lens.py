"""Hugging Face lenses resolve to the ordinary, validated local lens loader."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from prefscope import Lens, load_lens
from prefscope.api import hub


def _snapshot(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "manifest.json").write_text("{}")
    (path / "sae_model.pt").write_bytes(b"weights")
    return path


def test_download_lens_passes_hub_options_and_supports_subfolder(tmp_path, monkeypatch):
    root = tmp_path / "snapshot"
    lens_dir = _snapshot(root / "completion")
    seen = {}

    def fake_snapshot_download(**kwargs):
        seen.update(kwargs)
        return str(root)

    resolved = "a" * 40
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(hub, "resolve_hf_revision", lambda *a, **k: resolved)
    got = hub.download_lens(
        "owner/lenses", revision="v1", cache_dir=tmp_path / "cache",
        token="secret", local_files_only=True, subfolder="completion")

    assert got == lens_dir
    assert seen["repo_id"] == "owner/lenses" and seen["revision"] == resolved
    assert seen["local_files_only"] is True and seen["token"] == "secret"


def test_download_lens_rejects_missing_artifact_and_path_traversal(tmp_path, monkeypatch):
    root = tmp_path / "snapshot"
    root.mkdir()
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **_: str(root))
    monkeypatch.setattr(hub, "resolve_hf_revision", lambda *a, **k: "b" * 40)

    with pytest.raises(FileNotFoundError, match="not a PrefScope lens"):
        hub.download_lens("owner/broken")
    with pytest.raises(ValueError, match="relative path"):
        hub.download_lens("owner/broken", subfolder="../outside")


def test_from_pretrained_delegates_downloaded_dir_to_regular_loader(tmp_path, monkeypatch):
    lens_dir = _snapshot(tmp_path / "lens")
    seen = {}
    resolved = "c" * 40
    published_manifest = {"embed_model_id": "public/model"}
    sentinel = SimpleNamespace(manifest=published_manifest.copy())
    monkeypatch.setattr(hub, "resolve_hf_revision", lambda *a, **k: resolved)

    def fake_download(repo_id, **kwargs):
        seen["download"] = {"repo_id": repo_id, **kwargs}
        return lens_dir

    monkeypatch.setattr(hub, "download_lens", fake_download)

    def fake_from_dir(cls, path, **kwargs):
        seen.update(path=path, **kwargs)
        return sentinel

    monkeypatch.setattr(Lens, "from_dir", classmethod(fake_from_dir))
    got = Lens.from_pretrained(
        "owner/lens", revision="v1", token="top-secret", device="cpu")

    assert got is sentinel
    assert seen["path"] == lens_dir and seen["device"] == "cpu"
    assert seen["download"]["revision"] == resolved
    assert got.requested_revision == "v1"
    assert got.resolved_revision == resolved
    assert got.pretrained_revision == "v1"
    assert got.pretrained_resolved_revision == resolved
    assert got.manifest == published_manifest
    assert "top-secret" not in repr(vars(got))


def test_load_lens_hf_uri_parses_repo_and_subfolder(monkeypatch):
    seen = {}
    sentinel = object()

    def fake(cls, repo_id, **kwargs):
        seen.update(repo_id=repo_id, **kwargs)
        return sentinel

    monkeypatch.setattr(Lens, "from_pretrained", classmethod(fake))
    got = load_lens("hf://owner/lenses/completion", revision="v2")

    assert got is sentinel
    assert seen["repo_id"] == "owner/lenses"
    assert seen["subfolder"] == "completion" and seen["revision"] == "v2"


def test_split_hf_source_rejects_incomplete_repo():
    with pytest.raises(ValueError, match="owner/repository"):
        hub.split_hf_source("hf://only-one-part")



def test_resolve_hf_revision_uses_hub_sha_for_mutable_ref(monkeypatch):
    import huggingface_hub

    resolved = "d" * 40
    seen = {}

    class FakeApi:
        def repo_info(self, **kwargs):
            seen.update(kwargs)
            return SimpleNamespace(sha=resolved)

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    assert hub.resolve_hf_revision(
        "owner/lens", revision="release", token="secret") == resolved
    assert seen == {
        "repo_id": "owner/lens",
        "repo_type": "model",
        "revision": "release",
        "token": "secret",
    }


def test_explicit_commit_revision_resolves_offline_without_lookup(monkeypatch):
    import huggingface_hub

    class NoNetwork:
        def __init__(self):
            raise AssertionError("an explicit commit must not query the Hub")

    monkeypatch.setattr(huggingface_hub, "HfApi", NoNetwork)
    commit = "E" * 40
    assert hub.resolve_hf_revision(
        "owner/lens", revision=commit, local_files_only=True) == commit.lower()


def test_mutable_revision_cannot_be_resolved_in_local_only_mode():
    with pytest.raises(ValueError, match="exact 40-character commit"):
        hub.resolve_hf_revision(
            "owner/lens", revision="main", local_files_only=True)
