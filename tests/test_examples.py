from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.core


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_example(script_name: str, args: list[str]) -> None:
    script_path = REPO_ROOT / "examples" / script_name
    old_argv = sys.argv[:]
    try:
        sys.argv = [str(script_path), *args]
        runpy.run_path(str(script_path), run_name="__main__")
    finally:
        sys.argv = old_argv


def test_pipeline_demo_help_smoke(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        _run_example("pipeline_demo.py", ["--help"])

    assert exc_info.value.code == 0
    assert "Run the default pipeline demo" in capsys.readouterr().out


def test_sync_pipeline_demo_help_smoke(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        _run_example("sync_pipeline_demo.py", ["--help"])

    assert exc_info.value.code == 0
    assert "Run the synchronous pipeline demo" in capsys.readouterr().out
