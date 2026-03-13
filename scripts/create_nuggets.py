#!/usr/bin/env python3
from __future__ import annotations

import sys


def cli_compatible_main(argv: list[str] | None = None) -> int:
    from nuggetizer.cli.main import main as cli_main

    argv = sys.argv[1:] if argv is None else argv
    translated: list[str] = ["create"]
    for token in argv:
        translated.append(
            f"--{token[2:].replace('_', '-')}" if token.startswith("--") else token
        )
    translated.append("--resume")
    return cli_main(translated)


if __name__ == "__main__":
    cli_compatible_main()
