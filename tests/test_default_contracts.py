from __future__ import annotations

import pytest

from prefscope.pipeline.analyze_config import (
    ANALYSIS_DEFAULTS,
    CONCEPT_DEFAULTS,
    EMBEDDING_DEFAULTS,
    VIEWER_DEFAULTS,
)
from prefscope.config import VIEWER_EXPORT_DEFAULTS


def test_analyze_default_profiles_are_named_and_immutable():
    assert CONCEPT_DEFAULTS["presence_policy"] == "mixed"
    assert ANALYSIS_DEFAULTS["relationships"] == "auto"
    assert EMBEDDING_DEFAULTS["backend"] == "hf"
    with pytest.raises(TypeError):
        CONCEPT_DEFAULTS["presence_policy"] = "calibrated"


def test_analyze_viewer_profile_reuses_export_defaults():
    for key, value in VIEWER_EXPORT_DEFAULTS.items():
        assert VIEWER_DEFAULTS[key] == value
    assert VIEWER_DEFAULTS["enabled"] is True
