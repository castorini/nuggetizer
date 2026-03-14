#!/usr/bin/env python3
from __future__ import annotations

import sys


def cli_compatible_main(argv: list[str] | None = None) -> int:
    from nuggetizer.cli.main import main as cli_main

    argv = sys.argv[1:] if argv is None else argv
    translated = ["assign", "--input-kind", "retrieval"]
    rename_map = {
        "--nugget_file": "--nuggets",
        "--retrieve_results_file": "--contexts",
    }
    for token in argv:
        if token.startswith("--"):
            translated.append(rename_map.get(token, f"--{token[2:].replace('_', '-')}"))
        else:
            translated.append(token)
    translated.append("--resume")
    return cli_main(translated)


if __name__ == "__main__":
    cli_compatible_main()
