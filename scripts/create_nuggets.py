#!/usr/bin/env python3
from __future__ import annotations

import sys


def cli_compatible_main(argv: list[str] | None = None) -> int:
    from nuggetizer.cli.main import main as cli_main
    from nuggetizer.cli.script_compat import translate_legacy_argv

    argv = sys.argv[1:] if argv is None else argv
    translated = translate_legacy_argv(
        argv, command_prefix=["create"], append_resume=True
    )
    return cli_main(translated)


if __name__ == "__main__":
    cli_compatible_main()
