"""Command-line interface for PrefScope."""

from __future__ import annotations

import os
import sys

from prefscope.cli.parser import build_parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, ImportError, ValueError) as exc:
        if os.environ.get("PREFSCOPE_DEBUG"):
            raise
        print(f"error: {exc}", file=sys.stderr)
        return 2


def console_main(argv=None) -> None:
    """Entry point for the installed ``prefscope`` script; never returns.

    Exits without interpreter finalization, which native dependencies can block
    indefinitely by leaving non-daemon threads alive. Use ``main`` to call the CLI
    from Python.
    """
    try:
        code = main(argv)
    except SystemExit as exc:
        code = exc.code
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(int(code or 0))


__all__ = ["build_parser", "console_main", "main"]
