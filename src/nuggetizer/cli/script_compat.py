from __future__ import annotations


def translate_legacy_argv(
    argv: list[str],
    *,
    command_prefix: list[str],
    rename_map: dict[str, str] | None = None,
    append_resume: bool = False,
) -> list[str]:
    translated = list(command_prefix)
    resolved_rename_map = rename_map or {}
    for token in argv:
        if token.startswith("--"):
            translated.append(
                resolved_rename_map.get(token, f"--{token[2:].replace('_', '-')}")
            )
        else:
            translated.append(token)
    if append_resume:
        translated.append("--resume")
    return translated
