from __future__ import annotations

import sys
import uuid

import pytest

from prefscope.core.plugins import load_plugins, normalize_plugin_modules
from prefscope.core import registry
from prefscope.interpret.strategy import resolve_name_mode, resolve_verify_mode
from prefscope.pipeline.run import PipelineConfig, _accepted_params


def _write_plugin(tmp_path, *, component_name: str) -> str:
    module_name = f"prefscope_test_plugin_{uuid.uuid4().hex}"
    (tmp_path / f"{module_name}.py").write_text(
        "from prefscope.core import registry\n"
        f"@registry.register('clusterer', {component_name!r})\n"
        "class CustomClusterer:\n"
        "    def __init__(self): pass\n"
    )
    return module_name


def test_plugin_module_names_are_explicit_and_deduplicated():
    assert normalize_plugin_modules(["package.plugin", "package.plugin"]) == (
        "package.plugin",
    )
    with pytest.raises(ValueError, match="must be a list"):
        normalize_plugin_modules("package.plugin")
    with pytest.raises(ValueError, match="dotted Python module name"):
        normalize_plugin_modules(["../plugin.py"])


def test_load_plugins_reports_the_module_that_failed():
    with pytest.raises(ValueError, match="failed to import PrefScope plugin"):
        load_plugins(["prefscope_plugin_that_does_not_exist"])


def test_pipeline_config_loads_plugins_before_registry_resolution(tmp_path, monkeypatch):
    component_name = f"custom-{uuid.uuid4().hex}"
    module_name = _write_plugin(tmp_path, component_name=component_name)
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        config = PipelineConfig.from_dict({
            "lens_dir": "lens",
            "out_dir": "out",
            "plugins": [module_name],
            "stages": ["cluster"],
            "clusterer": {"name": component_name},
        })
    finally:
        sys.modules.pop(module_name, None)
    assert config.plugins == (module_name,)
    assert config.clusterer.component == component_name


def test_auto_parameter_contract_does_not_depend_on_plugin_sort_order():
    before = _accepted_params("interpreter", "auto")
    component_name = f"000-plugin-{uuid.uuid4().hex}"

    @registry.register("interpreter", component_name)
    class LexicallyEarlyInterpreter:
        def __init__(self, unrelated_plugin_option=None):
            self.unrelated_plugin_option = unrelated_plugin_option

    assert _accepted_params("interpreter", "auto") == before




def test_invalid_config_does_not_execute_declared_plugin(tmp_path, monkeypatch):
    module_name = f"prefscope_test_plugin_{uuid.uuid4().hex}"
    marker = tmp_path / "plugin-imported"
    (tmp_path / f"{module_name}.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('imported')\n")
    monkeypatch.syspath_prepend(str(tmp_path))

    with pytest.raises(ValueError, match="unknown llm keys"):
        PipelineConfig.from_dict({
            "lens_dir": "lens",
            "out_dir": "out",
            "plugins": [module_name],
            "llm": {"unknown": True},
        })
    assert not marker.exists()


def test_prompt_config_preserves_explicit_plugin_component_names(tmp_path, monkeypatch):
    module_name = f"prefscope_test_plugin_{uuid.uuid4().hex}"
    interpreter_name = f"prompt-interpreter-{uuid.uuid4().hex}"
    verifier_name = f"prompt-verifier-{uuid.uuid4().hex}"
    (tmp_path / f"{module_name}.py").write_text(
        "from prefscope.core import registry\n"
        f"@registry.register('interpreter', {interpreter_name!r})\n"
        "class PromptInterpreter:\n"
        "    def __init__(self): pass\n"
        f"@registry.register('verifier', {verifier_name!r})\n"
        "class PromptVerifier:\n"
        "    def __init__(self): pass\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        config = PipelineConfig.from_dict({
            "lens_dir": "lens",
            "out_dir": "out",
            "lens_kind": "prompt",
            "plugins": [module_name],
            "interpreter": interpreter_name,
            "verifier": verifier_name,
        })
    finally:
        sys.modules.pop(module_name, None)

    assert resolve_name_mode(
        config.interpreter.component, "prompt", "prompt") == interpreter_name
    assert resolve_verify_mode(
        config.verifier.component, "prompt", "prompt") == verifier_name
