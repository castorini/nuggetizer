#!/usr/bin/env python3
from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path

Command = Sequence[str]
ROOT = Path(__file__).resolve().parents[1]


def _run_step(label: str, command: Command) -> None:
    print(f"[quality-gate] {label}", flush=True)
    subprocess.run(command, check=True, cwd=ROOT)


def main() -> int:
    steps: list[tuple[str, Command]] = [
        ("uv lock --check", ["uv", "lock", "--check"]),
        ("ruff check", ["uv", "run", "ruff", "check", "."]),
        ("ruff format --check", ["uv", "run", "ruff", "format", "--check", "."]),
        ("pytest core", ["uv", "run", "pytest", "-q", "-m", "core", "tests"]),
        (
            "pytest integration",
            ["uv", "run", "pytest", "-q", "-m", "integration", "tests"],
        ),
        ("mypy", ["uv", "run", "mypy", "src", "tests"]),
    ]
    for label, command in steps:
        _run_step(label, command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
