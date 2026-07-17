"""Export a PrefScope lens + results into a compact JSON bundle for the web viewer.

Thin shim: the implementation lives in ``prefscope.viewer_export`` (sanitize /
features / diagnosis / examples / tables / maps / cli). This file only keeps the
historical entry point (`python scripts/export_viewer_data.py`) and import path
(`from export_viewer_data import ...`) working.

Usage:
    python scripts/export_viewer_data.py --lens-dir lens_arena8b_m32_k4 \
        --corpus corpora/arena_merged.parquet --out viewer-web/public/data
"""
from __future__ import annotations

import sys
from pathlib import Path

# run as `python scripts/export_viewer_data.py` puts scripts/ on the path, not
# the repo root — add it so `import prefscope` resolves.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# re-export EVERYTHING the historical monolith exposed (the package's __all__
# includes the underscore helpers tests rely on, e.g. `_dumps`).
from prefscope.viewer_export import *  # noqa: E402,F401,F403
from prefscope.viewer_export import main  # noqa: E402,F401

if __name__ == "__main__":
    raise SystemExit(main())
