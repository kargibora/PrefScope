import tomllib
from pathlib import Path

import prefscope
from prefscope.config import CONFIG


def test_package_imports():
    assert CONFIG.frozen_sae_dir.name == "features_m128_k16"


def test_version_matches_pyproject():
    """__version__ must equal pyproject's version — they drifted (0.1.0 vs 0.1.0a0) (#8)."""
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    declared = tomllib.loads(pyproject.read_text())["project"]["version"]
    assert prefscope.__version__ == declared
