"""CLI smoke tests: the parser builds and every command exposes --help.

These also guard the lazy-import contract — building the parser and printing help must
not require torch (heavy training/diagnose imports are deferred into their handlers), so
`prefscope --help` works after a bare `pip install prefscope`.
"""
from __future__ import annotations

import subprocess
import sys

import pytest

from prefscope.__main__ import main


def test_top_level_help_exits_zero():
    with pytest.raises(SystemExit) as e:
        main(["--help"])
    assert e.value.code == 0


@pytest.mark.parametrize("cmd", [
    "inspect", "build-lens", "build-prompt-lens", "diagnose",
    "win-relevance", "associate-outcomes", "prepare-dataset", "encode-dataset", "concepts",
    "compare-responses", "cluster-features", "feature-relations", "screen-confounds",
    "init-demo", "package-lens", "extract-concepts",
])
def test_subcommand_help_exits_zero(cmd):
    # exercises full sub-parser construction for the heavy commands too
    with pytest.raises(SystemExit) as e:
        main([cmd, "--help"])
    assert e.value.code == 0


def test_no_command_does_not_crash():
    # argparse with required subcommand exits non-zero but must not raise a bare error
    with pytest.raises(SystemExit):
        main([])


def test_cli_parser_import_does_not_eagerly_import_torch():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import prefscope.cli; assert 'torch' not in sys.modules",
        ],
        check=True,
    )
