def test_top_level_imports():
    from prefscope import (
        Lens, LoadedLens, load_lens, PairItem, Dataset,
        TrainConfig, SAEConfig, diagnose, evaluate_preference,
        feature_preference_relevance, registry,
    )
    assert LoadedLens is Lens
    assert callable(load_lens)
    assert callable(diagnose)
    assert callable(evaluate_preference)
    assert callable(feature_preference_relevance)
    assert hasattr(registry, "make")


def test_load_lens_delegates(monkeypatch):
    import prefscope
    captured = {}

    def fake_load(cls, path, *, device="cpu"):
        captured["path"] = path
        captured["device"] = device
        return "L"

    monkeypatch.setattr(prefscope.Lens, "load", classmethod(fake_load))
    out = prefscope.load_lens("some/dir", device="cuda")
    assert out == "L"
    assert captured == {"path": "some/dir", "device": "cuda"}


def test_import_prefscope_is_torch_free():
    import subprocess
    import sys
    code = "import prefscope; import sys; assert 'torch' not in sys.modules, sorted(m for m in sys.modules if 'torch' in m)"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
