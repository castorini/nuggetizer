from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Sequence

from nuggetizer.models.nuggetizer import Nuggetizer

from .adapters import create_output_record, request_from_create_record
from .logging_utils import setup_logging
from .normalize import direct_assign_inputs, direct_create_record
from .operations import (
    build_create_nuggetizer_kwargs,
    run_assign_answers_batch,
    run_assign_retrieval_batch,
    run_create_batch,
    run_metrics,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nuggetizer")
    subparsers = parser.add_subparsers(dest="command", required=True)

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
    create_parser.add_argument("--creator-model", type=str)
    create_parser.add_argument("--scorer-model", type=str)
    create_parser.add_argument("--window-size", type=int)
    create_parser.add_argument("--max-nuggets", type=int)
    create_parser.add_argument("--log-level", type=int, default=0, choices=[0, 1, 2])
    create_parser.add_argument("--use-azure-openai", action="store_true")
    create_parser.add_argument("--resume", action="store_true")
    create_parser.add_argument("--overwrite", action="store_true")
    create_parser.add_argument("--fail-if-exists", action="store_true")
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
    assign_parser.add_argument("--log-level", type=int, default=0, choices=[0, 1, 2])
    assign_parser.add_argument("--use-azure-openai", action="store_true")
    assign_parser.add_argument("--resume", action="store_true")
    assign_parser.add_argument("--overwrite", action="store_true")
    assign_parser.add_argument("--fail-if-exists", action="store_true")
    assign_parser.add_argument("--manifest-path", type=str)

    assign_retrieval_parser = subparsers.add_parser("assign-retrieval")
    assign_retrieval_parser.add_argument("--nuggets", required=True, type=str)
    assign_retrieval_parser.add_argument("--contexts", required=True, type=str)
    assign_retrieval_parser.add_argument("--output-file", required=True, type=str)
    assign_retrieval_parser.add_argument("--model", type=str, default="gpt-4")
    assign_retrieval_parser.add_argument(
        "--log-level", type=int, default=0, choices=[0, 1, 2]
    )
    assign_retrieval_parser.add_argument("--use-azure-openai", action="store_true")

    metrics_parser = subparsers.add_parser("metrics")
    metrics_parser.add_argument("--input-file", required=True, type=str)
    metrics_parser.add_argument("--output-file", required=True, type=str)

    return parser


def _read_direct_payload(args: argparse.Namespace) -> dict[str, Any]:
    if args.stdin:
        return json.loads(sys.stdin.read())
    if args.input_json is not None:
        return json.loads(args.input_json)
    raise ValueError("Direct input requires --stdin or --input-json")


def _emit_json(data: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(data) + "\n")


def _run_direct_create(args: argparse.Namespace) -> int:
    payload = _read_direct_payload(args)
    nuggetizer = Nuggetizer(**build_create_nuggetizer_kwargs(args))
    request_obj = request_from_create_record(direct_create_record(payload))
    scored_nuggets = nuggetizer.create(request_obj)
    output_record = create_output_record(request_obj, scored_nuggets)
    direct_output = {
        "query": output_record["query"],
        "nuggets": output_record["nuggets"],
    }
    if args.output == "json":
        _emit_json(direct_output)
    else:
        for nugget in direct_output["nuggets"]:
            sys.stdout.write(f"{nugget['importance']}: {nugget['text']}\n")
    return 0


def _run_direct_assign(args: argparse.Namespace) -> int:
    payload = _read_direct_payload(args)
    nuggetizer = Nuggetizer(
        assigner_model=args.model,
        log_level=args.log_level,
        use_azure_openai=args.use_azure_openai,
    )
    query, context, nuggets = direct_assign_inputs(payload)
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
        _emit_json(direct_output)
    else:
        for nugget in direct_output["nuggets"]:
            sys.stdout.write(
                f"{nugget['assignment']}: {nugget['importance']} {nugget['text']}\n"
            )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if getattr(args, "log_level", None) is not None:
        setup_logging(args.log_level)

    if args.command == "create":
        if args.input_file:
            if not args.output_file:
                parser.error("create --input-file requires --output-file")
            compat_args = argparse.Namespace(
                input_file=args.input_file,
                output_file=args.output_file,
                model=args.model,
                creator_model=args.creator_model,
                scorer_model=args.scorer_model,
                window_size=args.window_size,
                max_nuggets=args.max_nuggets,
                log_level=args.log_level,
                use_azure_openai=args.use_azure_openai,
            )
            run_create_batch(compat_args, setup_logging(args.log_level))
            return 0
        return _run_direct_create(args)

    if args.command == "assign":
        if args.contexts is not None:
            if not args.nuggets or not args.output_file or not args.input_kind:
                parser.error(
                    "batch assign requires --nuggets, --contexts, --input-kind, and --output-file"
                )
            if args.input_kind == "answers":
                compat_args = argparse.Namespace(
                    nugget_file=args.nuggets,
                    answer_file=args.contexts,
                    output_file=args.output_file,
                    model=args.model,
                    use_azure_openai=args.use_azure_openai,
                    log_level=args.log_level,
                )
                run_assign_answers_batch(compat_args, setup_logging(args.log_level))
                return 0

            compat_args = argparse.Namespace(
                nugget_file=args.nuggets,
                retrieve_results_file=args.contexts,
                output_file=args.output_file,
                model=args.model,
                log_level=args.log_level,
                use_azure_openai=args.use_azure_openai,
            )
            run_assign_retrieval_batch(compat_args, setup_logging(args.log_level))
            return 0
        return _run_direct_assign(args)

    if args.command == "assign-retrieval":
        compat_args = argparse.Namespace(
            nugget_file=args.nuggets,
            retrieve_results_file=args.contexts,
            output_file=args.output_file,
            model=args.model,
            log_level=args.log_level,
            use_azure_openai=args.use_azure_openai,
        )
        run_assign_retrieval_batch(compat_args, setup_logging(args.log_level))
        return 0

    if args.command == "metrics":
        compat_args = argparse.Namespace(
            input_file=args.input_file, output_file=args.output_file
        )
        processed_records, global_metrics = run_metrics(compat_args)
        with open(args.output_file, "w", encoding="utf-8") as file_obj:
            for record in processed_records:
                file_obj.write(json.dumps(record) + "\n")
            file_obj.write(json.dumps(global_metrics) + "\n")
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2
