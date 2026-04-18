from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NoReturn, cast

from .responses import CommandResponse

INVALID_ARGS_EXIT_CODE = 2
MISSING_RESOURCE_EXIT_CODE = 4
VALIDATION_EXIT_CODE = 5
RUNTIME_EXIT_CODE = 6
KNOWN_COMMANDS = (
    "create",
    "assign",
    "metrics",
    "serve",
    "view",
    "prompt",
    "describe",
    "schema",
    "doctor",
    "validate",
)
TOP_LEVEL_EXAMPLES = (
    "nuggetizer create --input-file pool.jsonl --output-file nuggets.jsonl",
    (
        "nuggetizer assign --input-kind answers --nuggets nuggets.jsonl "
        "--contexts answers.jsonl --output-file assignments.jsonl"
    ),
    "nuggetizer serve --port 8085",
    "nuggetizer doctor --output json",
)


class CLIError(Exception):
    """Structured CLI error carrying shell exit and envelope metadata."""

    def __init__(
        self,
        message: str,
        *,
        exit_code: int,
        status: str,
        error_code: str,
        command: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.exit_code = exit_code
        self.status = status
        self.error_code = error_code
        self.command = command or "unknown"
        self.details = details or {}


class CLIArgumentParser(argparse.ArgumentParser):
    """ArgumentParser that raises structured CLI errors instead of exiting."""

    def error(self, message: str) -> NoReturn:
        if message == "the following arguments are required: command":
            raise CLIError(
                build_missing_command_message(),
                exit_code=INVALID_ARGS_EXIT_CODE,
                status="validation_error",
                error_code="missing_command",
                command="unknown",
                details={
                    "available_commands": list(KNOWN_COMMANDS),
                    "examples": list(TOP_LEVEL_EXAMPLES),
                    "help_hint": "Run `nuggetizer --help` for full usage.",
                },
            )
        raise CLIError(
            message,
            exit_code=INVALID_ARGS_EXIT_CODE,
            status="validation_error",
            error_code="invalid_arguments",
            command=detect_command(sys.argv[1:]),
        )


def detect_command(argv: Sequence[str]) -> str:
    for token in argv:
        if token in KNOWN_COMMANDS:
            return token
    return "unknown"


def build_missing_command_message() -> str:
    command_list = ", ".join(KNOWN_COMMANDS)
    examples = "\n".join(f"  {example}" for example in TOP_LEVEL_EXAMPLES)
    return (
        "No command provided. Choose one of: "
        f"{command_list}\n"
        "Examples:\n"
        f"{examples}\n"
        "Run `nuggetizer --help` for full usage."
    )


def wants_json(argv: Sequence[str]) -> bool:
    for index, token in enumerate(argv):
        if token == "--output" and index + 1 < len(argv):
            return argv[index + 1] == "json"
    return False


def emit_json(data: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(data) + "\n")


def build_error_response(error: CLIError) -> CommandResponse:
    return CommandResponse(
        command=error.command,
        status=error.status,
        exit_code=error.exit_code,
        errors=[
            {
                "code": error.error_code,
                "message": error.message,
                "details": error.details,
                "retryable": False,
            }
        ],
    )


def build_runtime_error_response(command: str, error: Exception) -> CommandResponse:
    return CommandResponse(
        command=command,
        status="runtime_error",
        exit_code=RUNTIME_EXIT_CODE,
        errors=[
            {
                "code": "runtime_error",
                "message": str(error),
                "details": {},
                "retryable": False,
            }
        ],
    )


def ensure_file_exists(path: str, *, command: str, field_name: str) -> None:
    if not Path(path).exists():
        raise CLIError(
            f"{field_name} does not exist: {path}",
            exit_code=MISSING_RESOURCE_EXIT_CODE,
            status="validation_error",
            error_code="missing_input",
            command=command,
            details={"field": field_name, "path": path},
        )


def write_manifest(manifest_path: str | None, response: CommandResponse) -> None:
    if manifest_path is None:
        return
    Path(manifest_path).write_text(
        json.dumps(response.to_envelope(), indent=2) + "\n", encoding="utf-8"
    )


def resolve_write_policy(args: argparse.Namespace) -> str:
    if getattr(args, "resume", False):
        return "resume"
    if getattr(args, "overwrite", False):
        return "overwrite"
    if getattr(args, "fail_if_exists", False):
        return "fail_if_exists"
    return "default_fail_if_exists"


def prepare_output_path(args: argparse.Namespace, *, command: str) -> str:
    output_path = getattr(args, "output_file", None)
    if output_path is None:
        raise CLIError(
            f"{command} requires --output-file",
            exit_code=INVALID_ARGS_EXIT_CODE,
            status="validation_error",
            error_code="missing_output_file",
            command=command,
        )

    output_path_str = cast(str, output_path)
    output_file = Path(output_path_str)
    write_policy = resolve_write_policy(args)
    if output_file.exists():
        if write_policy == "resume":
            return output_path_str
        if write_policy == "overwrite":
            output_file.write_text("", encoding="utf-8")
            return output_path_str
        raise CLIError(
            f"Output file already exists: {output_path}",
            exit_code=VALIDATION_EXIT_CODE,
            status="validation_error",
            error_code="write_policy_conflict",
            command=command,
            details={"path": output_path, "write_policy": write_policy},
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    return output_path_str


def read_direct_payload(args: argparse.Namespace) -> dict[str, Any]:
    try:
        if args.stdin:
            return cast(dict[str, Any], json.loads(sys.stdin.read()))
        if args.input_json is not None:
            return cast(dict[str, Any], json.loads(args.input_json))
    except json.JSONDecodeError as exc:
        raise CLIError(
            "Input payload is not valid JSON",
            exit_code=INVALID_ARGS_EXIT_CODE,
            status="validation_error",
            error_code="invalid_json",
            command=args.command,
            details={"error": str(exc)},
        ) from exc
    raise CLIError(
        "Direct input requires --stdin or --input-json",
        exit_code=INVALID_ARGS_EXIT_CODE,
        status="validation_error",
        error_code="missing_direct_input",
        command=args.command,
    )


def format_reasoning_traces(
    traces: list[str], *, label_prefix: str = "Reasoning Trace"
) -> str:
    return "\n".join(
        f"{label_prefix} {index}: {trace}"
        for index, trace in enumerate(traces, start=1)
    )


def format_direct_nugget_output(
    nuggets: list[dict[str, Any]],
    *,
    include_reasoning: bool,
    include_assignment: bool,
    query: str | None = None,
    creator_reasoning_traces: Sequence[str] = (),
    scoring_reasoning_traces: Sequence[str] = (),
) -> str:
    lines: list[str] = []
    if query is not None:
        lines.append(f"query: {query}")
        lines.append("nuggets:")
    for nugget in nuggets:
        if include_assignment:
            lines.append(
                f"{nugget['assignment']}: {nugget['importance']} {nugget['text']}"
            )
        else:
            lines.append(f"{nugget['importance']}: {nugget['text']}")
    if include_reasoning:
        creator_traces = [trace for trace in creator_reasoning_traces if trace.strip()]
        scoring_traces = [trace for trace in scoring_reasoning_traces if trace.strip()]
        if creator_traces or scoring_traces:
            lines.append("")
        if creator_traces:
            lines.append(
                format_reasoning_traces(
                    creator_traces, label_prefix="creator reasoning trace"
                )
            )
        if creator_traces and scoring_traces:
            lines.append("")
        if scoring_traces:
            lines.append(
                format_reasoning_traces(
                    scoring_traces, label_prefix="scoring reasoning trace"
                )
            )
    return "\n".join(lines)
