#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

Command = Sequence[str]
ROOT = Path(__file__).resolve().parents[1]


def _run_step(label: str, command: Command) -> None:
    print(f"[quality-gate] {label}", flush=True)
    subprocess.run(command, check=True, cwd=ROOT)


def main() -> int:
    python = sys.executable
    steps: list[tuple[str, Command]] = [
        ("ruff check", [python, "-m", "ruff", "check", "."]),
        ("ruff format --check", [python, "-m", "ruff", "format", "--check", "."]),
        ("pytest core", [python, "-m", "pytest", "-q", "-m", "core", "tests"]),
        (
            "pytest integration",
            [python, "-m", "pytest", "-q", "-m", "integration", "tests"],
        ),
        ("mypy", [python, "-m", "mypy", "src", "tests"]),
    ]
    for label, command in steps:
        _run_step(label, command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
