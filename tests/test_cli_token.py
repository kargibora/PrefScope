from types import SimpleNamespace

import pytest

from prefscope.cli.token import _cmd_extract_activations


def test_extract_activations_refuses_existing_output_before_reading_args(tmp_path):
    (tmp_path / "manifest.json").write_text("{}")
    args = SimpleNamespace(out=tmp_path)

    with pytest.raises(FileExistsError, match="not empty"):
        _cmd_extract_activations(args)
