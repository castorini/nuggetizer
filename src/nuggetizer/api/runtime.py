from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, dataclass, replace
from typing import Any

from nuggetizer.cli.adapters import (
    assign_answer_output_record,
    collect_nonempty_reasoning_traces,
    collect_reasoning_traces,
    request_from_create_record_with_threshold,
    serialize_nugget,
)
from nuggetizer.cli.adapters_common import make_data_artifact
from nuggetizer.cli.introspection import validate_assign_input, validate_create_input
from nuggetizer.cli.normalize import (
    direct_assign_inputs,
    direct_create_record,
    joined_assign_batch_records,
    unwrap_direct_create_payload,
)
from nuggetizer.cli.operations import (
    build_assign_nuggetizer_kwargs,
    build_create_nuggetizer_kwargs,
)
from nuggetizer.cli.responses import CommandResponse
from nuggetizer.models.nuggetizer import Nuggetizer


@dataclass(frozen=True)
class ServerConfig:
    host: str
    port: int
    model: str = "gpt-4o"
    creator_model: str | None = None
    scorer_model: str | None = None
    window_size: int | None = None
    max_nuggets: int | None = None
    min_judgment: int = 2
    execution_mode: str = "sync"
    log_level: int = 0
    use_azure_openai: bool = False
    use_openrouter: bool = False
    reasoning_effort: str | None = None
    include_trace: bool = False
    include_reasoning: bool = False
    redact_prompts: bool = False
    quiet: bool = False


_CREATE_OVERRIDABLE_FIELDS = {
    "model",
    "creator_model",
    "scorer_model",
    "window_size",
    "max_nuggets",
    "min_judgment",
    "execution_mode",
    "log_level",
    "use_azure_openai",
    "use_openrouter",
    "reasoning_effort",
    "include_trace",
    "include_reasoning",
    "redact_prompts",
}

_ASSIGN_OVERRIDABLE_FIELDS = {
    "model",
    "execution_mode",
    "log_level",
    "use_azure_openai",
    "use_openrouter",
    "reasoning_effort",
    "include_trace",
    "include_reasoning",
    "redact_prompts",
}


def _base_args(config: ServerConfig) -> argparse.Namespace:
    return argparse.Namespace(
        model=config.model,
        creator_model=config.creator_model,
        scorer_model=config.scorer_model,
        window_size=config.window_size,
        max_nuggets=config.max_nuggets,
        min_judgment=config.min_judgment,
        execution_mode=config.execution_mode,
        log_level=config.log_level,
        use_azure_openai=config.use_azure_openai,
        use_openrouter=config.use_openrouter,
        reasoning_effort=config.reasoning_effort,
        include_trace=config.include_trace,
        include_reasoning=config.include_reasoning,
        redact_prompts=config.redact_prompts,
        quiet=config.quiet,
        output="json",
    )


def _extract_override_payload(
    payload: dict[str, Any], *, allowed_fields: set[str], unwrap_create: bool = False
) -> dict[str, Any]:
    override_payload = payload.get("overrides", {})
    if not isinstance(override_payload, dict):
        raise ValueError("overrides must be an object when provided")
    combined = dict(override_payload)
    if unwrap_create:
        unwrapped_payload = unwrap_direct_create_payload(payload)
        unwrapped_override_payload = unwrapped_payload.get("overrides", {})
        if not isinstance(unwrapped_override_payload, dict):
            raise ValueError("overrides must be an object when provided")
        combined.update(unwrapped_override_payload)
    unknown_keys = sorted(set(combined) - allowed_fields)
    if unknown_keys:
        raise ValueError(
            "unsupported nuggetizer override field(s): " + ", ".join(unknown_keys)
        )
    if combined.get("use_azure_openai") and combined.get("use_openrouter"):
        raise ValueError(
            "use_azure_openai and use_openrouter cannot both be true in overrides"
        )
    return combined


def _merge_config_with_payload(
    payload: dict[str, Any],
    *,
    config: ServerConfig,
    allowed_fields: set[str],
    unwrap_create: bool = False,
) -> ServerConfig:
    overrides = _extract_override_payload(
        payload,
        allowed_fields=allowed_fields,
        unwrap_create=unwrap_create,
    )
    if not overrides:
        return config
    effective_values = asdict(config)
    effective_values.update(overrides)
    return replace(config, **effective_values)


def execute_direct_create(
    payload: dict[str, Any], *, args: argparse.Namespace
) -> CommandResponse:
    validation = validate_create_input(payload)
    nuggetizer = Nuggetizer(**build_create_nuggetizer_kwargs(args))
    request_obj = request_from_create_record_with_threshold(
        direct_create_record(payload),
        min_judgment=args.min_judgment,
    )
    if args.execution_mode == "async":
        scored_nuggets = asyncio.run(nuggetizer.async_create(request_obj))
    else:
        scored_nuggets = nuggetizer.create(request_obj)
    direct_output: dict[str, Any] = {
        "query": request_obj.query.text,
        "nuggets": [
            serialize_nugget(
                nugget,
                include_reasoning=args.include_reasoning,
                include_trace=args.include_trace,
                redact_prompts=args.redact_prompts,
            )
            for nugget in scored_nuggets
        ],
    }
    if args.include_reasoning:
        creator_reasoning_traces = collect_nonempty_reasoning_traces(
            nuggetizer.get_creator_reasoning_traces()
        )
        scoring_reasoning_traces = collect_reasoning_traces(scored_nuggets)
        if creator_reasoning_traces:
            direct_output["creator_reasoning_traces"] = creator_reasoning_traces
        if scoring_reasoning_traces:
            direct_output["scoring_reasoning_traces"] = scoring_reasoning_traces
    return CommandResponse(
        command="create",
        inputs={"source": "direct"},
        resolved={
            "input_mode": "direct",
            "execution_mode": args.execution_mode,
            "model": args.model,
            "creator_model": args.creator_model or args.model,
            "scorer_model": args.scorer_model or args.model,
            "min_judgment": args.min_judgment,
            "reasoning_effort": args.reasoning_effort,
        },
        validation=validation,
        artifacts=[make_data_artifact("create-result", direct_output)],
        metrics={"nugget_count": len(direct_output["nuggets"])},
    )


def execute_direct_assign(
    payload: dict[str, Any], *, args: argparse.Namespace
) -> CommandResponse:
    validation = validate_assign_input(payload)
    nuggetizer = Nuggetizer(**build_assign_nuggetizer_kwargs(args))
    if all(key in payload for key in ["answer_records", "nugget_record"]) or all(
        key in payload for key in ["answers_envelope", "nugget_envelope"]
    ):
        batch_records = joined_assign_batch_records(payload)
        batch_output: list[dict[str, Any]] = []
        for batch_record in batch_records:
            query = batch_record["query"]
            context = batch_record["context"]
            nuggets = batch_record["nuggets"]
            if args.execution_mode == "async":
                assigned_nuggets = asyncio.run(
                    nuggetizer.async_assign(query, context, nuggets=nuggets)
                )
            else:
                assigned_nuggets = nuggetizer.assign(query, context, nuggets=nuggets)
            batch_output.append(
                assign_answer_output_record(
                    batch_record["answer_record"],
                    batch_record["nugget_record"],
                    batch_record["run_id"],
                    assigned_nuggets,
                    include_reasoning=args.include_reasoning,
                    include_trace=args.include_trace,
                    redact_prompts=args.redact_prompts,
                )
            )
        return CommandResponse(
            command="assign",
            inputs={"source": "direct"},
            resolved={
                "input_mode": "direct",
                "assign_mode": "context",
                "execution_mode": args.execution_mode,
                "model": args.model,
                "reasoning_effort": args.reasoning_effort,
            },
            validation=validation,
            artifacts=[make_data_artifact("assign-result", batch_output)],
            metrics={"record_count": len(batch_output)},
        )

    query, context, nuggets = direct_assign_inputs(payload)
    if args.execution_mode == "async":
        assigned_nuggets = asyncio.run(
            nuggetizer.async_assign(query, context, nuggets=nuggets)
        )
    else:
        assigned_nuggets = nuggetizer.assign(query, context, nuggets=nuggets)
    direct_output: dict[str, Any] = {
        "query": query,
        "nuggets": [
            serialize_nugget(
                nugget,
                include_reasoning=args.include_reasoning,
                include_trace=args.include_trace,
                redact_prompts=args.redact_prompts,
            )
            for nugget in assigned_nuggets
        ],
    }
    if args.include_reasoning:
        reasoning_traces = collect_reasoning_traces(assigned_nuggets)
        if reasoning_traces:
            direct_output["reasoning_traces"] = reasoning_traces
    return CommandResponse(
        command="assign",
        inputs={"source": "direct"},
        resolved={
            "input_mode": "direct",
            "assign_mode": "context",
            "execution_mode": args.execution_mode,
            "model": args.model,
            "reasoning_effort": args.reasoning_effort,
        },
        validation=validation,
        artifacts=[make_data_artifact("assign-result", direct_output)],
        metrics={"nugget_count": len(direct_output["nuggets"])},
    )


def run_create_request(
    payload: dict[str, Any], *, config: ServerConfig
) -> CommandResponse:
    effective_config = _merge_config_with_payload(
        payload,
        config=config,
        allowed_fields=_CREATE_OVERRIDABLE_FIELDS,
        unwrap_create=True,
    )
    return execute_direct_create(payload, args=_base_args(effective_config))


def run_assign_request(
    payload: dict[str, Any], *, config: ServerConfig
) -> CommandResponse:
    effective_config = _merge_config_with_payload(
        payload,
        config=config,
        allowed_fields=_ASSIGN_OVERRIDABLE_FIELDS,
    )
    return execute_direct_assign(payload, args=_base_args(effective_config))


def validation_error_response(command: str, message: str) -> CommandResponse:
    return CommandResponse(
        command=command,
        status="validation_error",
        exit_code=5,
        errors=[
            {
                "code": "validation_error",
                "message": message,
                "details": {},
                "retryable": False,
            }
        ],
    )


def runtime_error_response(command: str, error: Exception) -> CommandResponse:
    return CommandResponse(
        command=command,
        status="runtime_error",
        exit_code=6,
        errors=[
            {
                "code": "runtime_error",
                "message": str(error),
                "details": {},
                "retryable": False,
            }
        ],
    )
