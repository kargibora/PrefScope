"""Installed launcher for the optional Streamlit viewer."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    try:
        from streamlit.web import cli as streamlit_cli
    except ImportError as exc:
        raise SystemExit(
            "prefscope-view needs the viewer dependencies; install 'prefscope[viewer]'"
        ) from exc
    app = Path(__file__).with_name("app.py")
    sys.argv = ["streamlit", "run", str(app), "--", *sys.argv[1:]]
    return int(streamlit_cli.main())


if __name__ == "__main__":
    raise SystemExit(main())
