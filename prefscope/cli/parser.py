"""Compose PrefScope's domain-specific command registrars."""

from __future__ import annotations

import argparse

from prefscope.cli.commands import (
    register_data_commands,
    register_interpret_commands,
    register_lens_commands,
    register_token_commands,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the complete CLI without importing heavy runtime dependencies."""
    parser = argparse.ArgumentParser(
        prog="prefscope",
        description="Extract and inspect features through reusable concept lenses.",
        epilog=(
            "Build or load a lens, featurize data, then use the Python analysis "
            "primitives or optional recipes."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    register_data_commands(commands)
    register_lens_commands(commands)
    register_interpret_commands(commands)
    register_token_commands(commands)
    return parser


__all__ = ["build_parser"]
