def test_top_level_imports():
    from prefscope import (
        Dataset,
        FeatureBatch,
        FeatureCatalog,
        FeatureMatrix,
        Lens,
        LensBackend,
        PairItem,
        Report,
        activation_summary,
        coactivation_counts,
        feature_activation_table,
        load_lens,
    )

    assert all(
        item is not None
        for item in (
            PairItem, Dataset, Lens, FeatureBatch, FeatureMatrix, FeatureCatalog,
            LensBackend, Report,
        )
    )
    assert all(
        callable(item)
        for item in (
            load_lens, activation_summary, coactivation_counts,
            feature_activation_table,
        )
    )


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


def test_import_prefscope_does_not_require_fcntl():
    import subprocess
    import sys

    code = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == 'fcntl':
        raise ImportError('fcntl unavailable')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import prefscope
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
