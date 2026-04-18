from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NoReturn, cast

from nuggetizer.api.runtime import (
    ServerConfig,
)
from nuggetizer.core.types import Nugget, NuggetAssignMode
from nuggetizer.prompts import (
    create_assign_prompt,
    create_nugget_prompt,
    create_score_prompt,
)

from .adapters import (
    request_from_create_record_with_threshold,
)
from .adapters_common import make_data_artifact, make_file_artifact
from .config import load_config
from .execution import (
    AssignExecutionConfig,
    CreateExecutionConfig,
    ExecutionMode,
    execute_direct_assign,
    execute_direct_create,
)
from .introspection import (
    COMMAND_DESCRIPTIONS,
    SCHEMAS,
    doctor_report,
    validate_assign_batch_files,
    validate_assign_input,
    validate_create_batch_file,
    validate_create_input,
)
from .io import read_jsonl
from .logging_utils import setup_logging
from .normalize import (
    direct_assign_inputs,
    direct_create_record,
    joined_assign_batch_records,
)
from .operations import (
    AssignAnswersBatchConfig,
    AssignRetrievalBatchConfig,
    CreateBatchConfig,
    MetricsBatchConfig,
    async_run_assign_answers_batch,
    async_run_assign_retrieval_batch,
    async_run_create_batch,
    run_assign_answers_batch,
    run_assign_retrieval_batch,
    run_create_batch,
    run_metrics,
)
from .prompt_view import (
    build_prompt_template_view,
    build_rendered_prompt_view,
    list_prompt_templates,
    render_prompt_catalog_text,
    render_prompt_template_text,
    render_rendered_prompt_text,
    resolve_prompt_template,
)
from .responses import CommandResponse
from .view import (
    ViewError,
    build_view_summary,
    detect_artifact_type,
    load_records,
    render_view_summary,
)

_shtab: Any | None
try:
    import shtab as _shtab
except ModuleNotFoundError:  # optional dev dependency
    _shtab = None

shtab = cast(Any, _shtab)

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
                _build_missing_command_message(),
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
            command=_detect_command(sys.argv[1:]),
        )


def _detect_command(argv: Sequence[str]) -> str:
    for token in argv:
        if token in KNOWN_COMMANDS:
            return token
    return "unknown"


def _build_missing_command_message() -> str:
    command_list = ", ".join(KNOWN_COMMANDS)
    examples = "\n".join(f"  {example}" for example in TOP_LEVEL_EXAMPLES)
    return (
        "No command provided. Choose one of: "
        f"{command_list}\n"
        "Examples:\n"
        f"{examples}\n"
        "Run `nuggetizer --help` for full usage."
    )


def _wants_json(argv: Sequence[str]) -> bool:
    for index, token in enumerate(argv):
        if token == "--output" and index + 1 < len(argv):
            return argv[index + 1] == "json"
    return False


def _emit_json(data: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(data) + "\n")


def _build_error_response(error: CLIError) -> CommandResponse:
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


def _build_runtime_error_response(command: str, error: Exception) -> CommandResponse:
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


def _ensure_file_exists(path: str, *, command: str, field_name: str) -> None:
    if not Path(path).exists():
        raise CLIError(
            f"{field_name} does not exist: {path}",
            exit_code=MISSING_RESOURCE_EXIT_CODE,
            status="validation_error",
            error_code="missing_input",
            command=command,
            details={"field": field_name, "path": path},
        )


def _write_manifest(manifest_path: str | None, response: CommandResponse) -> None:
    if manifest_path is None:
        return
    Path(manifest_path).write_text(
        json.dumps(response.to_envelope(), indent=2) + "\n", encoding="utf-8"
    )


def _resolve_write_policy(args: argparse.Namespace) -> str:
    if getattr(args, "resume", False):
        return "resume"
    if getattr(args, "overwrite", False):
        return "overwrite"
    if getattr(args, "fail_if_exists", False):
        return "fail_if_exists"
    return "default_fail_if_exists"


def _prepare_output_path(args: argparse.Namespace, *, command: str) -> str:
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
    write_policy = _resolve_write_policy(args)
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


def _read_direct_payload(args: argparse.Namespace) -> dict[str, Any]:
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


def _format_reasoning_traces(
    traces: list[str], *, label_prefix: str = "Reasoning Trace"
) -> str:
    return "\n".join(
        f"{label_prefix} {index}: {trace}"
        for index, trace in enumerate(traces, start=1)
    )


def _format_direct_nugget_output(
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
                _format_reasoning_traces(
                    creator_traces, label_prefix="creator reasoning trace"
                )
            )
        if creator_traces and scoring_traces:
            lines.append("")
        if scoring_traces:
            lines.append(
                _format_reasoning_traces(
                    scoring_traces, label_prefix="scoring reasoning trace"
                )
            )
    return "\n".join(lines)


def build_parser() -> CLIArgumentParser:
    parser = CLIArgumentParser(
        prog="nuggetizer",
        description=(
            "Nuggetizer packaged CLI for nugget creation, assignment, metrics, "
            "validation, and artifact inspection."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Common patterns:\n"
            "  nuggetizer create --input-file pool.jsonl --output-file nuggets.jsonl\n"
            "  nuggetizer assign --input-kind answers --nuggets nuggets.jsonl "
            "--contexts answers.jsonl --output-file assignments.jsonl\n"
            "  nuggetizer serve --port 8085\n"
            "  nuggetizer prompt show create\n"
            "  nuggetizer doctor --output json"
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {importlib.metadata.version('nuggetizer')}",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        default=False,
        help="Suppress all log output (sets log level to CRITICAL).",
    )
    if shtab is not None:
        shtab.add_argument_to(parser, ["--print-completion"])
    subparsers = parser.add_subparsers(
        dest="command", required=True, parser_class=CLIArgumentParser
    )

    create_parser = subparsers.add_parser(
        "create",
        help="Create and score nuggets from direct JSON input or batch JSONL input.",
        description=(
            "Create and score nuggets from direct JSON input or batch JSONL input."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    create_inputs = create_parser.add_mutually_exclusive_group(required=True)
    create_inputs.add_argument(
        "--input-file",
        type=str,
        help="Batch JSONL request file in the shared query-candidate schema.",
    )
    create_inputs.add_argument(
        "--stdin",
        action="store_true",
        help="Read one direct JSON request from standard input.",
    )
    create_inputs.add_argument(
        "--input-json",
        type=str,
        help="Direct JSON request in the shared query-candidate schema.",
    )
    create_parser.add_argument(
        "--output-file", type=str, help="Output JSONL path for batch nugget creation."
    )
    create_parser.add_argument(
        "--output",
        choices=["text", "json", "jsonl"],
        default="text",
        help="Human-readable text, machine-readable JSON envelope, or JSONL output.",
    )
    create_parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="Default model used for creation and scoring.",
    )
    create_parser.add_argument(
        "--execution-mode",
        choices=["sync", "async"],
        default="sync",
        help="Execution mode for direct JSON input or batch JSONL input.",
    )
    create_parser.add_argument(
        "--creator-model", type=str, help="Override the nugget creation model."
    )
    create_parser.add_argument(
        "--scorer-model", type=str, help="Override the nugget scoring model."
    )
    create_parser.add_argument(
        "--window-size", type=int, help="Window size for chunked nugget creation."
    )
    create_parser.add_argument(
        "--max-nuggets", type=int, help="Maximum nuggets to emit per request."
    )
    create_parser.add_argument(
        "--min-judgment",
        type=int,
        default=2,
        help="Minimum candidate judgment to include when a judgment field is present.",
    )
    create_parser.add_argument(
        "--log-level",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="Logging verbosity: 0=warnings, 1=info, 2=debug.",
    )
    create_parser.add_argument(
        "--use-azure-openai",
        action="store_true",
        help="Use Azure OpenAI environment settings for OpenAI-compatible requests.",
    )
    create_parser.add_argument(
        "--use-openrouter",
        action="store_true",
        help="Use OpenRouter for OpenAI-compatible requests.",
    )
    create_parser.add_argument(
        "--reasoning-effort",
        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
        help="Reasoning effort for OpenAI-compatible models that support it.",
    )
    create_parser.add_argument(
        "--resume",
        action="store_true",
        help="Allow appending to an existing output file without truncating it.",
    )
    create_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Truncate an existing output file before writing results.",
    )
    create_parser.add_argument(
        "--fail-if-exists",
        action="store_true",
        help="Fail if the target output path already exists.",
    )
    create_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve inputs and write policy without running models.",
    )
    create_parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the declared contract without running models.",
    )
    create_parser.add_argument(
        "--include-trace",
        action="store_true",
        help="Include model trace details in emitted results where available.",
    )
    create_parser.add_argument(
        "--include-reasoning",
        action="store_true",
        help="Include model reasoning fields in emitted results where available.",
    )
    create_parser.add_argument(
        "--redact-prompts",
        action="store_true",
        help="Redact prompt content from emitted trace fields.",
    )
    create_parser.add_argument(
        "--manifest-path",
        type=str,
        help="Write the final JSON envelope to a manifest file.",
    )

    assign_parser = subparsers.add_parser(
        "assign",
        help="Assign nuggets to a direct context or to batch answers or retrieval results.",
        description="Assign nuggets to a direct context or to batch answers or retrieval results.",
    )
    assign_inputs = assign_parser.add_mutually_exclusive_group(required=True)
    assign_inputs.add_argument(
        "--stdin",
        action="store_true",
        help="Read one direct JSON payload from standard input.",
    )
    assign_inputs.add_argument(
        "--input-json",
        type=str,
        help="Direct JSON payload containing query, context, and nuggets.",
    )
    assign_inputs.add_argument(
        "--contexts",
        type=str,
        help="Batch JSONL contexts file for answers or retrieval candidates.",
    )
    assign_parser.add_argument(
        "--nuggets",
        type=str,
        help="Batch nugget JSONL file aligned by query identifier.",
    )
    assign_parser.add_argument(
        "--input-kind",
        choices=["answers", "retrieval"],
        help="Batch input type for --contexts.",
    )
    assign_parser.add_argument(
        "--output-file", type=str, help="Output JSONL path for batch nugget assignment."
    )
    assign_parser.add_argument(
        "--output",
        choices=["text", "json", "jsonl"],
        default="text",
        help="Human-readable text, machine-readable JSON envelope, or JSONL output.",
    )
    assign_parser.add_argument(
        "--model", type=str, default="gpt-4o", help="Model used for nugget assignment."
    )
    assign_parser.add_argument(
        "--execution-mode",
        choices=["sync", "async"],
        default="sync",
        help="Execution mode for direct JSON input or batch JSONL input.",
    )
    assign_parser.add_argument(
        "--log-level",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="Logging verbosity: 0=warnings, 1=info, 2=debug.",
    )
    assign_parser.add_argument(
        "--use-azure-openai",
        action="store_true",
        help="Use Azure OpenAI environment settings for OpenAI-compatible requests.",
    )
    assign_parser.add_argument(
        "--use-openrouter",
        action="store_true",
        help="Use OpenRouter for OpenAI-compatible requests.",
    )
    assign_parser.add_argument(
        "--reasoning-effort",
        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
        help="Reasoning effort for OpenAI-compatible models that support it.",
    )
    assign_parser.add_argument(
        "--resume",
        action="store_true",
        help="Allow appending to an existing output file without truncating it.",
    )
    assign_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Truncate an existing output file before writing results.",
    )
    assign_parser.add_argument(
        "--fail-if-exists",
        action="store_true",
        help="Fail if the target output path already exists.",
    )
    assign_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve inputs and write policy without running models.",
    )
    assign_parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the declared contract without running models.",
    )
    assign_parser.add_argument(
        "--include-trace",
        action="store_true",
        help="Include model trace details in emitted results where available.",
    )
    assign_parser.add_argument(
        "--include-reasoning",
        action="store_true",
        help="Include model reasoning fields in emitted results where available.",
    )
    assign_parser.add_argument(
        "--redact-prompts",
        action="store_true",
        help="Redact prompt content from emitted trace fields.",
    )
    assign_parser.add_argument(
        "--manifest-path",
        type=str,
        help="Write the final JSON envelope to a manifest file.",
    )

    metrics_parser = subparsers.add_parser(
        "metrics",
        help="Calculate per-query and global nugget metrics from assignment JSONL.",
        description="Calculate per-query and global nugget metrics from assignment JSONL.",
    )
    metrics_parser.add_argument(
        "--input-file", required=True, type=str, help="Assignment JSONL file to score."
    )
    metrics_parser.add_argument(
        "--output-file", required=True, type=str, help="Metrics JSONL file to write."
    )
    metrics_parser.add_argument(
        "--output",
        choices=["text", "json", "jsonl"],
        default="text",
        help="Human-readable text, machine-readable JSON envelope, or JSONL output.",
    )
    metrics_parser.add_argument(
        "--resume",
        action="store_true",
        help="Allow writing to an existing output file without truncating it.",
    )
    metrics_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Truncate an existing output file before writing results.",
    )
    metrics_parser.add_argument(
        "--fail-if-exists",
        action="store_true",
        help="Fail if the target output path already exists.",
    )
    metrics_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve inputs and write policy without computing metrics.",
    )
    metrics_parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the declared contract without computing metrics.",
    )
    metrics_parser.add_argument(
        "--manifest-path",
        type=str,
        help="Write the final JSON envelope to a manifest file.",
    )

    serve_parser = subparsers.add_parser(
        "serve",
        help="Start a FastAPI server for direct nugget creation and assignment.",
        description=(
            "Start a FastAPI server that exposes Nuggetizer direct create and "
            "assign operations over HTTP."
        ),
    )
    serve_parser.add_argument("--host", type=str, default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=8085)
    serve_parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="Default model used for create and assign requests.",
    )
    serve_parser.add_argument(
        "--creator-model", type=str, help="Override the nugget creation model."
    )
    serve_parser.add_argument(
        "--scorer-model", type=str, help="Override the nugget scoring model."
    )
    serve_parser.add_argument(
        "--window-size", type=int, help="Window size for chunked nugget creation."
    )
    serve_parser.add_argument(
        "--max-nuggets", type=int, help="Maximum nuggets to emit per request."
    )
    serve_parser.add_argument(
        "--min-judgment",
        type=int,
        default=2,
        help="Minimum candidate judgment to include when a judgment field is present.",
    )
    serve_parser.add_argument(
        "--execution-mode",
        choices=["sync", "async"],
        default="sync",
        help="Execution mode for direct JSON requests.",
    )
    serve_parser.add_argument(
        "--log-level",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="Logging verbosity: 0=warnings, 1=info, 2=debug.",
    )
    serve_parser.add_argument(
        "--use-azure-openai",
        action="store_true",
        help="Use Azure OpenAI environment settings for OpenAI-compatible requests.",
    )
    serve_parser.add_argument(
        "--use-openrouter",
        action="store_true",
        help="Use OpenRouter for OpenAI-compatible requests.",
    )
    serve_parser.add_argument(
        "--reasoning-effort",
        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
        help="Reasoning effort for OpenAI-compatible models that support it.",
    )
    serve_parser.add_argument("--include-trace", action="store_true")
    serve_parser.add_argument("--include-reasoning", action="store_true")
    serve_parser.add_argument("--redact-prompts", action="store_true")

    describe_parser = subparsers.add_parser(
        "describe",
        help="Inspect structured metadata for a public Nuggetizer command.",
    )
    describe_parser.add_argument(
        "target",
        choices=sorted(COMMAND_DESCRIPTIONS),
        help="Public command to describe.",
    )
    describe_parser.add_argument(
        "--output",
        choices=["text", "json"],
        default="text",
        help="Human-readable description or JSON envelope.",
    )

    schema_parser = subparsers.add_parser(
        "schema",
        help="Print JSON schemas for supported Nuggetizer inputs, outputs, and envelopes.",
    )
    schema_parser.add_argument(
        "target", choices=sorted(SCHEMAS), help="Schema artifact to print."
    )
    schema_parser.add_argument(
        "--output",
        choices=["text", "json"],
        default="text",
        help="Human-readable schema or JSON envelope.",
    )

    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Report environment and backend readiness for the packaged Nuggetizer CLI.",
    )
    doctor_parser.add_argument(
        "--output",
        choices=["text", "json"],
        default="text",
        help="Human-readable readiness report or JSON envelope.",
    )

    view_parser = subparsers.add_parser(
        "view",
        help="Inspect an existing Nuggetizer artifact.",
        description="Inspect an existing Nuggetizer artifact and render a stable summary.",
    )
    view_parser.add_argument("path", type=str, help="Artifact path to inspect.")
    view_parser.add_argument(
        "--type",
        dest="artifact_type",
        type=str,
        help="Explicit artifact type when automatic detection is ambiguous.",
    )
    view_parser.add_argument(
        "--records",
        type=int,
        default=3,
        help="Number of records to sample in the inspection summary.",
    )
    view_parser.add_argument(
        "--nugget-limit",
        type=int,
        default=5,
        help="Maximum nuggets to show per sampled record.",
    )
    view_parser.add_argument(
        "--color",
        choices=["auto", "always", "never"],
        default="auto",
        help="Color policy for text-mode rendering.",
    )
    view_parser.add_argument(
        "--output",
        choices=["text", "json"],
        default="text",
        help="Human-readable summary or JSON envelope.",
    )

    prompt_parser = subparsers.add_parser(
        "prompt",
        help="Inspect built-in Nuggetizer prompt templates.",
        description="Inspect built-in Nuggetizer prompt templates.",
    )
    prompt_subparsers = prompt_parser.add_subparsers(
        dest="prompt_command", required=True, parser_class=CLIArgumentParser
    )

    prompt_list_parser = prompt_subparsers.add_parser(
        "list",
        help="List built-in prompt templates.",
    )
    prompt_list_parser.add_argument(
        "--output",
        choices=["text", "json"],
        default="text",
        help="Human-readable catalog or JSON envelope.",
    )

    prompt_show_parser = prompt_subparsers.add_parser(
        "show",
        help="Show a built-in prompt template.",
    )
    prompt_show_parser.add_argument(
        "target",
        choices=["create", "assign", "score"],
        help="Prompt family to inspect.",
    )
    prompt_show_parser.add_argument(
        "--assign-mode",
        choices=["support_grade_2", "support_grade_3"],
        default="support_grade_3",
        help="Assignment prompt mode when target is assign.",
    )
    prompt_show_parser.add_argument(
        "--output",
        choices=["text", "json"],
        default="text",
        help="Human-readable prompt template or JSON envelope.",
    )

    prompt_render_parser = prompt_subparsers.add_parser(
        "render",
        help="Render a built-in prompt template against direct input.",
    )
    prompt_render_parser.add_argument(
        "target",
        choices=["create", "assign", "score"],
        help="Prompt family to render.",
    )
    prompt_render_parser.add_argument(
        "--assign-mode",
        choices=["support_grade_2", "support_grade_3"],
        default="support_grade_3",
        help="Assignment prompt mode when target is assign.",
    )
    prompt_render_inputs = prompt_render_parser.add_mutually_exclusive_group(
        required=True
    )
    prompt_render_inputs.add_argument(
        "--stdin",
        action="store_true",
        help="Read one direct JSON payload from standard input.",
    )
    prompt_render_inputs.add_argument(
        "--input-json",
        type=str,
        help="Direct JSON payload for the selected prompt family.",
    )
    prompt_render_parser.add_argument(
        "--max-nuggets",
        type=int,
        default=30,
        help="Maximum nuggets to include in create prompt rendering.",
    )
    prompt_render_parser.add_argument(
        "--min-judgment",
        type=int,
        default=2,
        help="Minimum candidate judgment to include when rendering create prompts.",
    )
    prompt_render_parser.add_argument(
        "--part",
        choices=["system", "user", "all"],
        default="all",
        help="Rendered prompt section to show in text mode.",
    )
    prompt_render_parser.add_argument(
        "--output",
        choices=["text", "json"],
        default="text",
        help="Human-readable rendered prompt or JSON envelope.",
    )

    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate direct JSON input or batch JSONL inputs without running models.",
        description="Validate direct JSON input or batch JSONL inputs without running models.",
    )
    validate_parser.add_argument(
        "target", choices=["create", "assign"], help="Validation target to inspect."
    )
    validate_inputs = validate_parser.add_mutually_exclusive_group(required=True)
    validate_inputs.add_argument(
        "--input-file", type=str, help="Batch JSONL request file to validate."
    )
    validate_inputs.add_argument(
        "--stdin",
        action="store_true",
        help="Read one direct JSON payload from standard input.",
    )
    validate_inputs.add_argument(
        "--input-json", type=str, help="Direct JSON payload to validate."
    )
    validate_inputs.add_argument(
        "--contexts", type=str, help="Batch contexts file to validate for assignment."
    )
    validate_parser.add_argument(
        "--nuggets", type=str, help="Batch nuggets file to validate for assignment."
    )
    validate_parser.add_argument(
        "--input-kind",
        choices=["answers", "retrieval"],
        help="Batch input type for assignment validation.",
    )
    validate_parser.add_argument(
        "--output-file",
        type=str,
        help="Optional output path to validate for write-policy planning.",
    )
    validate_parser.add_argument(
        "--output",
        choices=["text", "json"],
        default="text",
        help="Human-readable validation summary or JSON envelope.",
    )

    return parser


def _run_direct_create(args: argparse.Namespace) -> CommandResponse:
    payload = _read_direct_payload(args)
    validation = validate_create_input(payload)
    request_obj = request_from_create_record_with_threshold(
        direct_create_record(payload),
        min_judgment=args.min_judgment,
    )
    if args.validate_only or args.dry_run:
        return CommandResponse(
            command="create",
            mode="validate" if args.validate_only else "dry-run",
            inputs={"source": "direct"},
            resolved={
                "input_mode": "direct",
                "execution_mode": args.execution_mode,
                "min_judgment": args.min_judgment,
            },
            validation=validation,
            metrics={"candidate_count": len(request_obj.documents)},
        )
    response = execute_direct_create(
        payload,
        config=CreateExecutionConfig(
            model=args.model,
            creator_model=args.creator_model,
            scorer_model=args.scorer_model,
            window_size=args.window_size,
            max_nuggets=args.max_nuggets,
            min_judgment=args.min_judgment,
            execution_mode=cast(ExecutionMode, args.execution_mode),
            log_level=args.log_level,
            use_azure_openai=args.use_azure_openai,
            use_openrouter=args.use_openrouter,
            reasoning_effort=args.reasoning_effort,
            include_trace=args.include_trace,
            include_reasoning=args.include_reasoning,
            redact_prompts=args.redact_prompts,
            quiet=getattr(args, "quiet", False),
            output=args.output,
        ),
    )
    direct_output = cast(dict[str, Any], response.artifacts[0]["data"])
    if args.output == "json":
        return response

    sys.stdout.write(
        _format_direct_nugget_output(
            cast(list[dict[str, Any]], direct_output["nuggets"]),
            include_reasoning=args.include_reasoning,
            include_assignment=False,
            query=cast(str, direct_output["query"]),
            creator_reasoning_traces=cast(
                list[str], direct_output.get("creator_reasoning_traces", [])
            ),
            scoring_reasoning_traces=cast(
                list[str], direct_output.get("scoring_reasoning_traces", [])
            ),
        )
        + "\n"
    )
    return CommandResponse(command="create")


def _run_direct_assign(args: argparse.Namespace) -> CommandResponse:
    payload = _read_direct_payload(args)
    validation = validate_assign_input(payload)
    nuggets: list[Any]
    if all(key in payload for key in ["answer_records", "nugget_record"]) or all(
        key in payload for key in ["answers_envelope", "nugget_envelope"]
    ):
        batch_records = joined_assign_batch_records(payload)
        nuggets = cast(list[Any], batch_records[0]["nuggets"])
    else:
        _query, _context, nuggets = direct_assign_inputs(payload)
    if args.validate_only or args.dry_run:
        return CommandResponse(
            command="assign",
            mode="validate" if args.validate_only else "dry-run",
            inputs={"source": "direct"},
            resolved={
                "input_mode": "direct",
                "assign_mode": "context",
                "execution_mode": args.execution_mode,
            },
            validation=validation,
            metrics={
                "record_count": validation["record_count"],
                "nugget_count": len(nuggets),
            },
        )
    response = execute_direct_assign(
        payload,
        config=AssignExecutionConfig(
            model=args.model,
            execution_mode=cast(ExecutionMode, args.execution_mode),
            log_level=args.log_level,
            use_azure_openai=args.use_azure_openai,
            use_openrouter=args.use_openrouter,
            reasoning_effort=args.reasoning_effort,
            include_trace=args.include_trace,
            include_reasoning=args.include_reasoning,
            redact_prompts=args.redact_prompts,
            quiet=getattr(args, "quiet", False),
            output=args.output,
        ),
    )
    direct_output = response.artifacts[0]["data"]
    if args.output == "json":
        return response

    if isinstance(direct_output, list):
        sys.stdout.write(
            "\n".join(
                json.dumps(record, ensure_ascii=False) for record in direct_output
            )
            + "\n"
        )
        return CommandResponse(command="assign")

    sys.stdout.write(
        _format_direct_nugget_output(
            cast(list[dict[str, Any]], cast(dict[str, Any], direct_output)["nuggets"]),
            include_reasoning=args.include_reasoning,
            include_assignment=True,
            scoring_reasoning_traces=cast(
                list[str],
                cast(dict[str, Any], direct_output).get("reasoning_traces", []),
            ),
        )
        + "\n"
    )
    return CommandResponse(command="assign")


def _run_create_batch_command(args: argparse.Namespace) -> CommandResponse:
    _ensure_file_exists(args.input_file, command="create", field_name="input_file")
    output_path = _prepare_output_path(args, command="create")
    write_policy = _resolve_write_policy(args)
    validation = validate_create_batch_file(args.input_file)
    if args.validate_only or args.dry_run:
        response = CommandResponse(
            command="create",
            mode="validate" if args.validate_only else "dry-run",
            inputs={"input_file": args.input_file},
            resolved={
                "input_mode": "batch",
                "execution_mode": args.execution_mode,
                "min_judgment": args.min_judgment,
                "write_policy": write_policy,
            },
            artifacts=[make_file_artifact("create-output", output_path)],
            validation=validation,
            metrics={"record_count": validation["record_count"]},
        )
        return response
    batch_config = CreateBatchConfig(
        input_file=args.input_file,
        output_file=output_path,
        execution=CreateExecutionConfig(
            model=args.model,
            creator_model=args.creator_model,
            scorer_model=args.scorer_model,
            window_size=args.window_size,
            max_nuggets=args.max_nuggets,
            min_judgment=args.min_judgment,
            execution_mode=cast(ExecutionMode, args.execution_mode),
            log_level=args.log_level,
            use_azure_openai=args.use_azure_openai,
            use_openrouter=args.use_openrouter,
            reasoning_effort=args.reasoning_effort,
            include_trace=args.include_trace,
            include_reasoning=args.include_reasoning,
            redact_prompts=args.redact_prompts,
            quiet=getattr(args, "quiet", False),
            output=args.output,
        ),
    )
    if args.execution_mode == "async":
        response = asyncio.run(
            async_run_create_batch(
                batch_config,
                setup_logging(args.log_level, quiet=getattr(args, "quiet", False)),
            )
        )
    else:
        response = run_create_batch(
            batch_config,
            setup_logging(args.log_level, quiet=getattr(args, "quiet", False)),
        )
    response.inputs = {"input_file": args.input_file}
    response.resolved = {
        "input_mode": "batch",
        "execution_mode": args.execution_mode,
        "min_judgment": args.min_judgment,
        "write_policy": write_policy,
    }
    response.artifacts = [make_file_artifact("create-output", output_path)]
    _write_manifest(args.manifest_path, response)
    return response


def _run_assign_batch_command(args: argparse.Namespace) -> CommandResponse:
    if not args.nuggets or not args.output_file or not args.input_kind:
        raise CLIError(
            "batch assign requires --nuggets, --contexts, --input-kind, and --output-file",
            exit_code=INVALID_ARGS_EXIT_CODE,
            status="validation_error",
            error_code="missing_batch_assign_args",
            command="assign",
        )
    _ensure_file_exists(args.nuggets, command="assign", field_name="nuggets")
    _ensure_file_exists(args.contexts, command="assign", field_name="contexts")
    output_path = _prepare_output_path(args, command="assign")
    write_policy = _resolve_write_policy(args)
    validation = validate_assign_batch_files(args.nuggets, args.contexts)
    if args.validate_only or args.dry_run:
        response = CommandResponse(
            command="assign",
            mode="validate" if args.validate_only else "dry-run",
            inputs={
                "nuggets": args.nuggets,
                "contexts": args.contexts,
                "input_kind": args.input_kind,
            },
            resolved={
                "input_mode": "batch",
                "assign_mode": args.input_kind,
                "execution_mode": args.execution_mode,
                "write_policy": write_policy,
            },
            artifacts=[make_file_artifact("assign-output", output_path)],
            validation=validation,
            metrics=validation,
        )
        return response

    if args.input_kind == "answers":
        answers_batch_config = AssignAnswersBatchConfig(
            nugget_file=args.nuggets,
            answer_file=args.contexts,
            output_file=output_path,
            execution=AssignExecutionConfig(
                model=args.model,
                execution_mode=cast(ExecutionMode, args.execution_mode),
                log_level=args.log_level,
                use_azure_openai=args.use_azure_openai,
                use_openrouter=args.use_openrouter,
                reasoning_effort=args.reasoning_effort,
                include_trace=args.include_trace,
                include_reasoning=args.include_reasoning,
                redact_prompts=args.redact_prompts,
                quiet=getattr(args, "quiet", False),
                output=args.output,
            ),
        )
        if args.execution_mode == "async":
            response = asyncio.run(
                async_run_assign_answers_batch(
                    answers_batch_config,
                    setup_logging(args.log_level, quiet=getattr(args, "quiet", False)),
                )
            )
        else:
            response = run_assign_answers_batch(
                answers_batch_config,
                setup_logging(args.log_level, quiet=getattr(args, "quiet", False)),
            )
    else:
        retrieval_batch_config = AssignRetrievalBatchConfig(
            nugget_file=args.nuggets,
            retrieve_results_file=args.contexts,
            output_file=output_path,
            execution=AssignExecutionConfig(
                model=args.model,
                execution_mode=cast(ExecutionMode, args.execution_mode),
                log_level=args.log_level,
                use_azure_openai=args.use_azure_openai,
                use_openrouter=args.use_openrouter,
                reasoning_effort=args.reasoning_effort,
                include_trace=args.include_trace,
                include_reasoning=args.include_reasoning,
                redact_prompts=args.redact_prompts,
                quiet=getattr(args, "quiet", False),
                output=args.output,
            ),
        )
        if args.execution_mode == "async":
            response = asyncio.run(
                async_run_assign_retrieval_batch(
                    retrieval_batch_config,
                    setup_logging(args.log_level, quiet=getattr(args, "quiet", False)),
                )
            )
        else:
            response = run_assign_retrieval_batch(
                retrieval_batch_config,
                setup_logging(args.log_level, quiet=getattr(args, "quiet", False)),
            )

    response.command = "assign"
    response.inputs = {
        "nuggets": args.nuggets,
        "contexts": args.contexts,
        "input_kind": args.input_kind,
    }
    response.resolved = {
        "input_mode": "batch",
        "assign_mode": args.input_kind,
        "execution_mode": args.execution_mode,
        "write_policy": write_policy,
    }
    response.artifacts = [make_file_artifact("assign-output", output_path)]
    _write_manifest(args.manifest_path, response)
    return response


def _run_metrics_command(args: argparse.Namespace) -> CommandResponse:
    _ensure_file_exists(args.input_file, command="metrics", field_name="input_file")
    output_path = _prepare_output_path(args, command="metrics")
    config = MetricsBatchConfig(input_file=args.input_file, output_file=output_path)
    input_records = read_jsonl(args.input_file)
    if args.validate_only or args.dry_run:
        return CommandResponse(
            command="metrics",
            mode="validate" if args.validate_only else "dry-run",
            inputs={"input_file": args.input_file},
            resolved={
                "input_mode": "batch",
                "write_policy": _resolve_write_policy(args),
            },
            artifacts=[make_file_artifact("metrics-output", output_path)],
            validation={"valid": True, "record_count": len(input_records)},
            metrics={"record_count": len(input_records)},
        )
    processed_records, global_metrics = run_metrics(config)
    with open(output_path, "w", encoding="utf-8") as file_obj:
        for record in processed_records:
            file_obj.write(json.dumps(record) + "\n")
        file_obj.write(json.dumps(global_metrics) + "\n")
    response = CommandResponse(
        command="metrics",
        inputs={"input_file": args.input_file},
        resolved={
            "input_mode": "batch",
            "write_policy": _resolve_write_policy(args),
        },
        artifacts=[make_file_artifact("metrics-output", output_path)],
        metrics={
            "record_count": len(processed_records),
            "global_metrics": global_metrics,
        },
    )
    _write_manifest(args.manifest_path, response)
    return response


def _run_serve_command(args: argparse.Namespace) -> CommandResponse:
    try:
        import uvicorn

        from nuggetizer.api.app import create_app
    except ModuleNotFoundError as error:
        raise CLIError(
            "serve requires FastAPI dependencies; install the `api` extra",
            exit_code=MISSING_RESOURCE_EXIT_CODE,
            status="validation_error",
            error_code="missing_api_dependencies",
            command="serve",
            details={"missing_dependencies": ["fastapi", "uvicorn"]},
        ) from error

    app = create_app(
        ServerConfig(
            host=args.host,
            port=args.port,
            model=args.model,
            creator_model=args.creator_model,
            scorer_model=args.scorer_model,
            window_size=args.window_size,
            max_nuggets=args.max_nuggets,
            min_judgment=args.min_judgment,
            execution_mode=args.execution_mode,
            log_level=args.log_level,
            use_azure_openai=args.use_azure_openai,
            use_openrouter=args.use_openrouter,
            reasoning_effort=args.reasoning_effort,
            include_trace=args.include_trace,
            include_reasoning=args.include_reasoning,
            redact_prompts=args.redact_prompts,
            quiet=getattr(args, "quiet", False),
        )
    )
    uvicorn.run(app, host=args.host, port=args.port)
    return CommandResponse(
        command="serve", resolved={"host": args.host, "port": args.port}
    )


def _run_describe_command(args: argparse.Namespace) -> CommandResponse:
    description = COMMAND_DESCRIPTIONS[args.target]
    response = CommandResponse(
        command="describe",
        inputs={"target": args.target},
        resolved={"target_command": args.target},
        artifacts=[make_data_artifact(args.target, description)],
    )
    if args.output == "text":
        sys.stdout.write(json.dumps(description, indent=2) + "\n")
    return response


def _run_schema_command(args: argparse.Namespace) -> CommandResponse:
    schema = SCHEMAS[args.target]
    response = CommandResponse(
        command="schema",
        inputs={"target": args.target},
        resolved={"target": args.target},
        artifacts=[make_data_artifact(args.target, schema)],
    )
    if args.output == "text":
        sys.stdout.write(json.dumps(schema, indent=2) + "\n")
    return response


def _run_doctor_command(args: argparse.Namespace) -> CommandResponse:
    report = doctor_report()
    config_path = getattr(args, "_config_path", None)
    report["config_file"] = str(config_path) if config_path else None
    response = CommandResponse(
        command="doctor",
        metrics=report,
        validation={"python_ok": report["python_ok"]},
        warnings=[] if report["env_file_present"] else [".env file not found"],
    )
    if args.output == "text":
        sys.stdout.write(json.dumps(report, indent=2) + "\n")
    return response


def _run_view_command(args: argparse.Namespace) -> CommandResponse:
    try:
        records = load_records(args.path)
        artifact_type = detect_artifact_type(records, args.artifact_type)
    except ViewError as error:
        raise CLIError(
            str(error),
            exit_code=VALIDATION_EXIT_CODE,
            status="validation_error",
            error_code="invalid_view_input",
            command="view",
            details={"path": args.path, "artifact_type": args.artifact_type},
        ) from error

    view_summary = build_view_summary(
        args.path,
        records,
        artifact_type,
        record_limit=args.records,
        nugget_limit=args.nugget_limit,
    )
    response = CommandResponse(
        command="view",
        mode="inspect",
        inputs={"path": args.path},
        resolved={
            "artifact_type": artifact_type,
            "records": args.records,
            "nugget_limit": args.nugget_limit,
            "color": args.color,
        },
        artifacts=[make_data_artifact("view-summary", view_summary)],
        metrics=view_summary["summary"],
    )
    if args.output == "text":
        sys.stdout.write(render_view_summary(view_summary, color=args.color) + "\n")
    return response


def _run_prompt_command(args: argparse.Namespace) -> CommandResponse:
    if args.prompt_command == "list":
        catalog = list_prompt_templates()
        response = CommandResponse(
            command="prompt",
            mode="inspect",
            artifacts=[make_data_artifact("prompt-catalog", catalog)],
        )
        if args.output == "text":
            sys.stdout.write(render_prompt_catalog_text(catalog) + "\n")
        return response

    assign_mode = NuggetAssignMode(args.assign_mode)
    template_name, template = resolve_prompt_template(args.target, assign_mode)
    view = build_prompt_template_view(
        args.target,
        template_name,
        template,
        assign_mode=assign_mode if args.target == "assign" else None,
    )
    response = CommandResponse(
        command="prompt",
        mode="inspect",
        inputs={"target": args.target},
        resolved={
            "prompt_command": "show",
            "target": args.target,
            "assign_mode": view["assign_mode"],
        },
        artifacts=[make_data_artifact("prompt-template", view)],
    )
    if args.output == "text":
        sys.stdout.write(render_prompt_template_text(view) + "\n")
    return response


def _normalize_score_payload(payload: dict[str, Any]) -> tuple[str, list[Nugget]]:
    if "query" not in payload or "nuggets" not in payload:
        raise CLIError(
            "score prompt render requires query and nuggets",
            exit_code=INVALID_ARGS_EXIT_CODE,
            status="validation_error",
            error_code="invalid_score_prompt_input",
            command="prompt",
        )
    nuggets = [
        Nugget(
            text=nugget if isinstance(nugget, str) else nugget["text"],
        )
        for nugget in cast(list[Any], payload["nuggets"])
    ]
    return str(payload["query"]), nuggets


def _run_prompt_render_command(args: argparse.Namespace) -> CommandResponse:
    payload = _read_direct_payload(args)
    assign_mode = NuggetAssignMode(args.assign_mode)
    template_name, _template = resolve_prompt_template(args.target, assign_mode)

    if args.target == "create":
        validation = validate_create_input(payload)
        if not validation.get("valid", False):
            raise CLIError(
                "create prompt input failed validation",
                exit_code=INVALID_ARGS_EXIT_CODE,
                status="validation_error",
                error_code="invalid_create_prompt_input",
                command="prompt",
                details=validation,
            )
        request_obj = request_from_create_record_with_threshold(
            direct_create_record(payload),
            min_judgment=args.min_judgment,
        )
        messages = create_nugget_prompt(
            request_obj,
            0,
            len(request_obj.documents),
            [],
            creator_max_nuggets=args.max_nuggets,
        )
        inputs = {
            "query": request_obj.query.text,
            "candidate_count": len(request_obj.documents),
            "max_nuggets": args.max_nuggets,
            "min_judgment": args.min_judgment,
        }
        resolved_assign_mode = None
    elif args.target == "assign":
        validation = validate_assign_input(payload)
        if not validation.get("valid", False):
            raise CLIError(
                "assign prompt input failed validation",
                exit_code=INVALID_ARGS_EXIT_CODE,
                status="validation_error",
                error_code="invalid_assign_prompt_input",
                command="prompt",
                details=validation,
            )
        query, context, nuggets = direct_assign_inputs(payload)
        messages = create_assign_prompt(
            query,
            context,
            nuggets,
            assigner_mode=assign_mode,
        )
        inputs = {
            "query": query,
            "context": context,
            "nugget_count": len(nuggets),
        }
        resolved_assign_mode = assign_mode
    else:
        query, score_nuggets = _normalize_score_payload(payload)
        messages = create_score_prompt(query, score_nuggets)
        inputs = {"query": query, "nugget_count": len(score_nuggets)}
        resolved_assign_mode = None

    view = build_rendered_prompt_view(
        args.target,
        template_name,
        messages,
        assign_mode=resolved_assign_mode,
        inputs=inputs,
    )
    response = CommandResponse(
        command="prompt",
        mode="inspect",
        inputs={"target": args.target},
        resolved={
            "prompt_command": "render",
            "target": args.target,
            "assign_mode": view["assign_mode"],
            "part": args.part,
        },
        artifacts=[make_data_artifact("rendered-prompt", view)],
    )
    if args.output == "text":
        sys.stdout.write(render_rendered_prompt_text(view, part=args.part) + "\n")
    return response


def _run_validate_command(args: argparse.Namespace) -> CommandResponse:
    if args.target == "create":
        if args.input_file is not None:
            _ensure_file_exists(
                args.input_file, command="validate", field_name="input_file"
            )
            validation = validate_create_batch_file(args.input_file)
            resolved = {"target_command": "create", "input_mode": "batch"}
        else:
            payload = _read_direct_payload(args)
            validation = validate_create_input(payload)
            resolved = {"target_command": "create", "input_mode": "direct"}
    else:
        if args.contexts is not None:
            if not args.nuggets:
                raise CLIError(
                    "validate assign batch mode requires --nuggets",
                    exit_code=INVALID_ARGS_EXIT_CODE,
                    status="validation_error",
                    error_code="missing_nuggets",
                    command="validate",
                )
            _ensure_file_exists(args.nuggets, command="validate", field_name="nuggets")
            _ensure_file_exists(
                args.contexts, command="validate", field_name="contexts"
            )
            validation = validate_assign_batch_files(args.nuggets, args.contexts)
            resolved = {"target_command": "assign", "input_mode": "batch"}
        else:
            payload = _read_direct_payload(args)
            validation = validate_assign_input(payload)
            resolved = {"target_command": "assign", "input_mode": "direct"}

    status = "success" if validation.get("valid", False) else "validation_error"
    exit_code = 0 if validation.get("valid", False) else INVALID_ARGS_EXIT_CODE
    response = CommandResponse(
        command="validate",
        status=status,
        exit_code=exit_code,
        inputs={"target": args.target},
        resolved=resolved,
        validation=validation,
    )
    if status != "success":
        response.errors.append(
            {
                "code": "validation_failed",
                "message": f"{args.target} input failed validation",
                "details": validation,
                "retryable": False,
            }
        )
    if args.output == "text":
        sys.stdout.write(json.dumps(validation, indent=2) + "\n")
    return response


def _run_command(args: argparse.Namespace) -> CommandResponse:
    if getattr(args, "log_level", None) is not None:
        setup_logging(args.log_level, quiet=getattr(args, "quiet", False))

    if args.command == "create":
        if args.input_file:
            return _run_create_batch_command(args)
        return _run_direct_create(args)
    if args.command == "assign":
        if args.contexts is not None:
            return _run_assign_batch_command(args)
        return _run_direct_assign(args)
    if args.command == "metrics":
        return _run_metrics_command(args)
    if args.command == "serve":
        return _run_serve_command(args)
    if args.command == "describe":
        return _run_describe_command(args)
    if args.command == "schema":
        return _run_schema_command(args)
    if args.command == "doctor":
        return _run_doctor_command(args)
    if args.command == "view":
        return _run_view_command(args)
    if args.command == "prompt":
        if args.prompt_command == "render":
            return _run_prompt_render_command(args)
        return _run_prompt_command(args)
    if args.command == "validate":
        return _run_validate_command(args)
    raise CLIError(
        f"Unknown command: {args.command}",
        exit_code=INVALID_ARGS_EXIT_CODE,
        status="validation_error",
        error_code="unknown_command",
        command=args.command,
    )


def main(argv: Sequence[str] | None = None) -> int:
    argv_list = list(argv) if argv is not None else sys.argv[1:]
    parser = build_parser()
    config, config_path = load_config()
    wants_json = _wants_json(argv_list)

    try:
        args = parser.parse_args(argv_list)
        args._config_path = config_path
        for key, value in config.items():
            flag = f"--{key.replace('_', '-')}"
            if flag not in argv_list:
                setattr(args, key, value)
        response = _run_command(args)
        if getattr(args, "output", "text") == "json":
            _emit_json(response.to_envelope())
        return response.exit_code
    except CLIError as error:
        if wants_json:
            _emit_json(_build_error_response(error).to_envelope())
        else:
            sys.stderr.write(f"error: {error.message}\n")
        return error.exit_code
    except Exception as error:  # pragma: no cover - defensive runtime envelope
        response = _build_runtime_error_response(_detect_command(argv_list), error)
        if wants_json:
            _emit_json(response.to_envelope())
        else:
            sys.stderr.write(f"error: {error}\n")
        return response.exit_code
