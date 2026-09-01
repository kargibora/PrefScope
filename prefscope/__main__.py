"""Run the PrefScope command-line interface with ``python -m prefscope``."""

from __future__ import annotations

from prefscope.cli import build_parser, console_main, main

__all__ = ["build_parser", "console_main", "main"]


if __name__ == "__main__":
    console_main()
