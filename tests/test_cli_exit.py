import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

_LINGERING_THREAD = """
import sys, threading
# A non-daemon thread that never finishes, as HF streaming leaves behind.
threading.Thread(target=threading.Event().wait, daemon=False).start()
from prefscope.cli import console_main
console_main({argv})
"""


def _run(argv: list[str], tmp_path: Path) -> subprocess.CompletedProcess:
    script = tmp_path / "run_cli.py"
    script.write_text(_LINGERING_THREAD.format(argv=repr(argv)))
    return subprocess.run(
        [sys.executable, str(script)],
        capture_output=True, text=True, timeout=60, cwd=REPO,
    )


def test_console_main_exits_with_lingering_thread(tmp_path):
    try:
        result = _run(["--help"], tmp_path)
    except subprocess.TimeoutExpired:
        pytest.fail("console_main hung at interpreter shutdown")
    assert result.returncode == 0
    assert "usage: prefscope" in result.stdout


def test_console_main_propagates_error_exit_code(tmp_path):
    try:
        result = _run(["inspect", "--corpus", "does-not-exist.parquet"], tmp_path)
    except subprocess.TimeoutExpired:
        pytest.fail("console_main hung on the error path")
    assert result.returncode == 2
