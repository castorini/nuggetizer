from __future__ import annotations

import asyncio
import argparse
import json
import sys
from pathlib import Path
from typing import Any, NoReturn, Sequence, cast

from nuggetizer.models.nuggetizer import Nuggetizer

from .adapters import create_output_record, request_from_create_record
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
from .normalize import direct_assign_inputs, direct_create_record
from .operations import (
    async_run_assign_answers_batch,
    async_run_assign_retrieval_batch,
    async_run_create_batch,
    build_create_nuggetizer_kwargs,
    run_assign_answers_batch,
    run_assign_retrieval_batch,
    run_create_batch,
    run_metrics,
)
from .responses import CommandResponse

INVALID_ARGS_EXIT_CODE = 2
MISSING_RESOURCE_EXIT_CODE = 4
VALIDATION_EXIT_CODE = 5
RUNTIME_EXIT_CODE = 6


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
        raise CLIError(
            message,
            exit_code=INVALID_ARGS_EXIT_CODE,
            status="validation_error",
            error_code="invalid_arguments",
            command=_detect_command(sys.argv[1:]),
        )


def _detect_command(argv: Sequence[str]) -> str:
    known_commands = {
        "create",
        "assign",
        "assign-retrieval",
        "metrics",
        "describe",
        "schema",
        "doctor",
        "validate",
    }
    for token in argv:
        if token in known_commands:
            return token
    return "unknown"


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


def build_parser() -> CLIArgumentParser:
    parser = CLIArgumentParser(prog="nuggetizer")
    subparsers = parser.add_subparsers(
        dest="command", required=True, parser_class=CLIArgumentParser
    )

    create_parser = subparsers.add_parser("create")
    create_inputs = create_parser.add_mutually_exclusive_group(required=True)
    create_inputs.add_argument("--input-file", type=str)
    create_inputs.add_argument("--stdin", action="store_true")
    create_inputs.add_argument("--input-json", type=str)
    create_parser.add_argument("--output-file", type=str)
    create_parser.add_argument(
        "--output", choices=["text", "json", "jsonl"], default="text"
    )
    create_parser.add_argument("--model", type=str, default="gpt-4o")
    create_parser.add_argument(
        "--execution-mode", choices=["sync", "async"], default="sync"
    )
    create_parser.add_argument("--creator-model", type=str)
    create_parser.add_argument("--scorer-model", type=str)
    create_parser.add_argument("--window-size", type=int)
    create_parser.add_argument("--max-nuggets", type=int)
    create_parser.add_argument("--log-level", type=int, default=0, choices=[0, 1, 2])
    create_parser.add_argument("--use-azure-openai", action="store_true")
    create_parser.add_argument("--resume", action="store_true")
    create_parser.add_argument("--overwrite", action="store_true")
    create_parser.add_argument("--fail-if-exists", action="store_true")
    create_parser.add_argument("--dry-run", action="store_true")
    create_parser.add_argument("--validate-only", action="store_true")
    create_parser.add_argument("--manifest-path", type=str)

    assign_parser = subparsers.add_parser("assign")
    assign_inputs = assign_parser.add_mutually_exclusive_group(required=True)
    assign_inputs.add_argument("--stdin", action="store_true")
    assign_inputs.add_argument("--input-json", type=str)
    assign_inputs.add_argument("--contexts", type=str)
    assign_parser.add_argument("--nuggets", type=str)
    assign_parser.add_argument("--input-kind", choices=["answers", "retrieval"])
    assign_parser.add_argument("--output-file", type=str)
    assign_parser.add_argument(
        "--output", choices=["text", "json", "jsonl"], default="text"
    )
    assign_parser.add_argument("--model", type=str, default="gpt-4o")
    assign_parser.add_argument(
        "--execution-mode", choices=["sync", "async"], default="sync"
    )
    assign_parser.add_argument("--log-level", type=int, default=0, choices=[0, 1, 2])
    assign_parser.add_argument("--use-azure-openai", action="store_true")
    assign_parser.add_argument("--resume", action="store_true")
    assign_parser.add_argument("--overwrite", action="store_true")
    assign_parser.add_argument("--fail-if-exists", action="store_true")
    assign_parser.add_argument("--dry-run", action="store_true")
    assign_parser.add_argument("--validate-only", action="store_true")
    assign_parser.add_argument("--manifest-path", type=str)

    assign_retrieval_parser = subparsers.add_parser("assign-retrieval")
    assign_retrieval_parser.add_argument("--nuggets", required=True, type=str)
    assign_retrieval_parser.add_argument("--contexts", required=True, type=str)
    assign_retrieval_parser.add_argument("--output-file", required=True, type=str)
    assign_retrieval_parser.add_argument(
        "--output", choices=["text", "json", "jsonl"], default="text"
    )
    assign_retrieval_parser.add_argument("--model", type=str, default="gpt-4")
    assign_retrieval_parser.add_argument(
        "--execution-mode", choices=["sync", "async"], default="sync"
    )
    assign_retrieval_parser.add_argument(
        "--log-level", type=int, default=0, choices=[0, 1, 2]
    )
    assign_retrieval_parser.add_argument("--use-azure-openai", action="store_true")
    assign_retrieval_parser.add_argument("--resume", action="store_true")
    assign_retrieval_parser.add_argument("--overwrite", action="store_true")
    assign_retrieval_parser.add_argument("--fail-if-exists", action="store_true")
    assign_retrieval_parser.add_argument("--dry-run", action="store_true")
    assign_retrieval_parser.add_argument("--validate-only", action="store_true")
    assign_retrieval_parser.add_argument("--manifest-path", type=str)

    metrics_parser = subparsers.add_parser("metrics")
    metrics_parser.add_argument("--input-file", required=True, type=str)
    metrics_parser.add_argument("--output-file", required=True, type=str)
    metrics_parser.add_argument(
        "--output", choices=["text", "json", "jsonl"], default="text"
    )
    metrics_parser.add_argument("--resume", action="store_true")
    metrics_parser.add_argument("--overwrite", action="store_true")
    metrics_parser.add_argument("--fail-if-exists", action="store_true")
    metrics_parser.add_argument("--dry-run", action="store_true")
    metrics_parser.add_argument("--validate-only", action="store_true")
    metrics_parser.add_argument("--manifest-path", type=str)

    describe_parser = subparsers.add_parser("describe")
    describe_parser.add_argument("target", choices=sorted(COMMAND_DESCRIPTIONS))
    describe_parser.add_argument("--output", choices=["text", "json"], default="text")

    schema_parser = subparsers.add_parser("schema")
    schema_parser.add_argument("artifact", choices=sorted(SCHEMAS))
    schema_parser.add_argument("--output", choices=["text", "json"], default="text")

    doctor_parser = subparsers.add_parser("doctor")
    doctor_parser.add_argument("--output", choices=["text", "json"], default="text")

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("target", choices=["create", "assign"])
    validate_inputs = validate_parser.add_mutually_exclusive_group(required=True)
    validate_inputs.add_argument("--input-file", type=str)
    validate_inputs.add_argument("--stdin", action="store_true")
    validate_inputs.add_argument("--input-json", type=str)
    validate_inputs.add_argument("--contexts", type=str)
    validate_parser.add_argument("--nuggets", type=str)
    validate_parser.add_argument("--input-kind", choices=["answers", "retrieval"])
    validate_parser.add_argument("--output-file", type=str)
    validate_parser.add_argument("--output", choices=["text", "json"], default="text")

    return parser


def _run_direct_create(args: argparse.Namespace) -> CommandResponse:
    payload = _read_direct_payload(args)
    validation = validate_create_input(payload)
    if args.validate_only or args.dry_run:
        return CommandResponse(
            command="create",
            mode="validate" if args.validate_only else "dry-run",
            inputs={"source": "direct"},
            resolved={
                "input_mode": "direct",
                "execution_mode": args.execution_mode,
            },
            validation=validation,
            metrics={"candidate_count": len(payload["candidates"])},
        )
    nuggetizer = Nuggetizer(**build_create_nuggetizer_kwargs(args))
    request_obj = request_from_create_record(direct_create_record(payload))
    if args.execution_mode == "async":
        scored_nuggets = asyncio.run(nuggetizer.async_create(request_obj))
    else:
        scored_nuggets = nuggetizer.create(request_obj)
    output_record = create_output_record(request_obj, scored_nuggets)
    direct_output = {
        "query": output_record["query"],
        "nuggets": output_record["nuggets"],
    }
    if args.output == "json":
        return CommandResponse(
            command="create",
            inputs={"source": "direct"},
            resolved={
                "input_mode": "direct",
                "execution_mode": args.execution_mode,
            },
            artifacts=[{"type": "inline_result", "data": direct_output}],
            metrics={"nugget_count": len(direct_output["nuggets"])},
        )

    for nugget in direct_output["nuggets"]:
        sys.stdout.write(f"{nugget['importance']}: {nugget['text']}\n")
    return CommandResponse(command="create")


def _run_direct_assign(args: argparse.Namespace) -> CommandResponse:
    payload = _read_direct_payload(args)
    validation = validate_assign_input(payload)
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
            metrics={"nugget_count": len(payload["nuggets"])},
        )
    nuggetizer = Nuggetizer(
        assigner_model=args.model,
        log_level=args.log_level,
        use_azure_openai=args.use_azure_openai,
    )
    query, context, nuggets = direct_assign_inputs(payload)
    if args.execution_mode == "async":
        assigned_nuggets = asyncio.run(
            nuggetizer.async_assign(query, context, nuggets=nuggets)
        )
    else:
        assigned_nuggets = nuggetizer.assign(query, context, nuggets=nuggets)
    direct_output = {
        "query": query,
        "nuggets": [
            {
                "text": nugget.text,
                "importance": nugget.importance,
                "assignment": nugget.assignment,
            }
            for nugget in assigned_nuggets
        ],
    }
    if args.output == "json":
        return CommandResponse(
            command="assign",
            inputs={"source": "direct"},
            resolved={
                "input_mode": "direct",
                "assign_mode": "context",
                "execution_mode": args.execution_mode,
            },
            artifacts=[{"type": "inline_result", "data": direct_output}],
            metrics={"nugget_count": len(direct_output["nuggets"])},
        )

    for nugget in cast(list[dict[str, str]], direct_output["nuggets"]):
        sys.stdout.write(
            f"{nugget['assignment']}: {nugget['importance']} {nugget['text']}\n"
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
                "write_policy": write_policy,
            },
            artifacts=[{"path": output_path, "type": "jsonl"}],
            validation=validation,
            metrics={"record_count": validation["record_count"]},
        )
        return response
    compat_args = argparse.Namespace(
        input_file=args.input_file,
        output_file=output_path,
        model=args.model,
        creator_model=args.creator_model,
        scorer_model=args.scorer_model,
        window_size=args.window_size,
        max_nuggets=args.max_nuggets,
        log_level=args.log_level,
        use_azure_openai=args.use_azure_openai,
    )
    if args.execution_mode == "async":
        response = asyncio.run(
            async_run_create_batch(compat_args, setup_logging(args.log_level))
        )
    else:
        response = run_create_batch(compat_args, setup_logging(args.log_level))
    response.inputs = {"input_file": args.input_file}
    response.resolved = {
        "input_mode": "batch",
        "execution_mode": args.execution_mode,
        "write_policy": write_policy,
    }
    response.artifacts = [{"path": output_path, "type": "jsonl"}]
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
            artifacts=[{"path": output_path, "type": "jsonl"}],
            validation=validation,
            metrics=validation,
        )
        return response

    if args.input_kind == "answers":
        compat_args = argparse.Namespace(
            nugget_file=args.nuggets,
            answer_file=args.contexts,
            output_file=output_path,
            model=args.model,
            use_azure_openai=args.use_azure_openai,
            log_level=args.log_level,
        )
        if args.execution_mode == "async":
            response = asyncio.run(
                async_run_assign_answers_batch(
                    compat_args, setup_logging(args.log_level)
                )
            )
        else:
            response = run_assign_answers_batch(
                compat_args, setup_logging(args.log_level)
            )
    else:
        compat_args = argparse.Namespace(
            nugget_file=args.nuggets,
            retrieve_results_file=args.contexts,
            output_file=output_path,
            model=args.model,
            log_level=args.log_level,
            use_azure_openai=args.use_azure_openai,
        )
        if args.execution_mode == "async":
            response = asyncio.run(
                async_run_assign_retrieval_batch(
                    compat_args, setup_logging(args.log_level)
                )
            )
        else:
            response = run_assign_retrieval_batch(
                compat_args, setup_logging(args.log_level)
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
    response.artifacts = [{"path": output_path, "type": "jsonl"}]
    _write_manifest(args.manifest_path, response)
    return response


def _run_assign_retrieval_alias(args: argparse.Namespace) -> CommandResponse:
    _ensure_file_exists(args.nuggets, command="assign-retrieval", field_name="nuggets")
    _ensure_file_exists(
        args.contexts, command="assign-retrieval", field_name="contexts"
    )
    output_path = _prepare_output_path(args, command="assign-retrieval")
    write_policy = _resolve_write_policy(args)
    validation = validate_assign_batch_files(args.nuggets, args.contexts)
    if args.validate_only or args.dry_run:
        return CommandResponse(
            command="assign-retrieval",
            mode="validate" if args.validate_only else "dry-run",
            inputs={"nuggets": args.nuggets, "contexts": args.contexts},
            resolved={
                "input_mode": "batch",
                "assign_mode": "retrieval",
                "alias_for": "assign",
                "execution_mode": args.execution_mode,
                "write_policy": write_policy,
            },
            artifacts=[{"path": output_path, "type": "jsonl"}],
            validation=validation,
            metrics=validation,
        )
    compat_args = argparse.Namespace(
        nugget_file=args.nuggets,
        retrieve_results_file=args.contexts,
        output_file=output_path,
        model=args.model,
        log_level=args.log_level,
        use_azure_openai=args.use_azure_openai,
    )
    if args.execution_mode == "async":
        response = asyncio.run(
            async_run_assign_retrieval_batch(compat_args, setup_logging(args.log_level))
        )
    else:
        response = run_assign_retrieval_batch(
            compat_args, setup_logging(args.log_level)
        )
    response.command = "assign-retrieval"
    response.inputs = {"nuggets": args.nuggets, "contexts": args.contexts}
    response.resolved = {
        "input_mode": "batch",
        "assign_mode": "retrieval",
        "alias_for": "assign",
        "execution_mode": args.execution_mode,
        "write_policy": write_policy,
    }
    response.artifacts = [{"path": output_path, "type": "jsonl"}]
    _write_manifest(args.manifest_path, response)
    return response


def _run_metrics_command(args: argparse.Namespace) -> CommandResponse:
    _ensure_file_exists(args.input_file, command="metrics", field_name="input_file")
    output_path = _prepare_output_path(args, command="metrics")
    compat_args = argparse.Namespace(
        input_file=args.input_file, output_file=output_path
    )
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
            artifacts=[{"path": output_path, "type": "jsonl"}],
            validation={"valid": True, "record_count": len(input_records)},
            metrics={"record_count": len(input_records)},
        )
    processed_records, global_metrics = run_metrics(compat_args)
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
        artifacts=[{"path": output_path, "type": "jsonl"}],
        metrics={
            "record_count": len(processed_records),
            "global_metrics": global_metrics,
        },
    )
    _write_manifest(args.manifest_path, response)
    return response


def _run_describe_command(args: argparse.Namespace) -> CommandResponse:
    description = COMMAND_DESCRIPTIONS[args.target]
    response = CommandResponse(
        command="describe",
        inputs={"target": args.target},
        resolved={"target_command": args.target},
        artifacts=[{"type": "inline_result", "data": description}],
    )
    if args.output == "text":
        sys.stdout.write(json.dumps(description, indent=2) + "\n")
    return response


def _run_schema_command(args: argparse.Namespace) -> CommandResponse:
    schema = SCHEMAS[args.artifact]
    response = CommandResponse(
        command="schema",
        inputs={"artifact": args.artifact},
        resolved={"artifact": args.artifact},
        artifacts=[{"type": "inline_result", "data": schema}],
    )
    if args.output == "text":
        sys.stdout.write(json.dumps(schema, indent=2) + "\n")
    return response


def _run_doctor_command(args: argparse.Namespace) -> CommandResponse:
    report = doctor_report()
    response = CommandResponse(
        command="doctor",
        metrics=report,
        validation={"python_ok": report["python_ok"]},
        warnings=[] if report["env_file_present"] else [".env file not found"],
    )
    if args.output == "text":
        sys.stdout.write(json.dumps(report, indent=2) + "\n")
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
        setup_logging(args.log_level)

    if args.command == "create":
        if args.input_file:
            return _run_create_batch_command(args)
        return _run_direct_create(args)
    if args.command == "assign":
        if args.contexts is not None:
            return _run_assign_batch_command(args)
        return _run_direct_assign(args)
    if args.command == "assign-retrieval":
        return _run_assign_retrieval_alias(args)
    if args.command == "metrics":
        return _run_metrics_command(args)
    if args.command == "describe":
        return _run_describe_command(args)
    if args.command == "schema":
        return _run_schema_command(args)
    if args.command == "doctor":
        return _run_doctor_command(args)
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
    wants_json = _wants_json(argv_list)

    try:
        args = parser.parse_args(argv_list)
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
        response = CommandResponse(
            command=_detect_command(argv_list),
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
        if wants_json:
            _emit_json(response.to_envelope())
        else:
            sys.stderr.write(f"error: {error}\n")
        return response.exit_code
