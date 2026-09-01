"""Compose PrefScope's domain-specific command registrars."""

from __future__ import annotations

import argparse

from prefscope.cli.commands import (
    register_analysis_commands,
    register_bank_commands,
    register_data_commands,
    register_interpret_commands,
    register_lens_commands,
    register_token_commands,
    register_workflow_commands,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the complete CLI without importing heavy runtime dependencies."""
    parser = argparse.ArgumentParser(
        prog="prefscope",
        description="Analyze prompt and response datasets through reusable concept lenses.",
        epilog=(
            "Train a new lens with prepare-dataset -> build-lens -> run. "
            "Apply published lenses to a dataset with analyze, to one text pair with "
            "extract-concepts, or stage by stage with concepts and the advanced commands."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    register_workflow_commands(commands)
    register_data_commands(commands)
    register_lens_commands(commands)
    register_interpret_commands(commands)
    register_analysis_commands(commands)
    register_bank_commands(commands)
    register_token_commands(commands)
    return parser


__all__ = ["build_parser"]
