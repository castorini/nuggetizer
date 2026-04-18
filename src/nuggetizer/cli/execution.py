from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Literal

from nuggetizer.models.nuggetizer import Nuggetizer

from .adapters import (
    assign_answer_output_record,
    collect_nonempty_reasoning_traces,
    collect_reasoning_traces,
    request_from_create_record_with_threshold,
    serialize_nugget,
)
from .adapters_common import make_data_artifact
from .introspection import validate_assign_input, validate_create_input
from .normalize import (
    direct_assign_inputs,
    direct_create_record,
    joined_assign_batch_records,
)
from .responses import CommandResponse

ExecutionMode = Literal["sync", "async"]


@dataclass(frozen=True)
class CreateExecutionConfig:
    model: str
    creator_model: str | None = None
    scorer_model: str | None = None
    window_size: int | None = None
    max_nuggets: int | None = None
    min_judgment: int = 2
    execution_mode: ExecutionMode = "sync"
    log_level: int = 0
    use_azure_openai: bool = False
    use_openrouter: bool = False
    reasoning_effort: str | None = None
    include_trace: bool = False
    include_reasoning: bool = False
    redact_prompts: bool = False
    quiet: bool = False
    output: str = "json"


@dataclass(frozen=True)
class AssignExecutionConfig:
    model: str
    execution_mode: ExecutionMode = "sync"
    log_level: int = 0
    use_azure_openai: bool = False
    use_openrouter: bool = False
    reasoning_effort: str | None = None
    include_trace: bool = False
    include_reasoning: bool = False
    redact_prompts: bool = False
    quiet: bool = False
    output: str = "json"


def build_create_nuggetizer_kwargs(config: CreateExecutionConfig) -> dict[str, Any]:
    nuggetizer_kwargs: dict[str, Any] = {
        "log_level": config.log_level,
        "use_azure_openai": config.use_azure_openai,
        "use_openrouter": config.use_openrouter,
        "store_trace": config.include_trace,
        "store_reasoning": config.include_reasoning,
        "reasoning_effort": config.reasoning_effort,
    }
    if config.creator_model or config.scorer_model:
        nuggetizer_kwargs.update(
            {
                "creator_model": config.creator_model or config.model,
                "scorer_model": config.scorer_model or config.model,
            }
        )
    else:
        nuggetizer_kwargs["model"] = config.model
    if config.window_size:
        nuggetizer_kwargs["window_size"] = config.window_size
    if config.max_nuggets:
        nuggetizer_kwargs["max_nuggets"] = config.max_nuggets
    return nuggetizer_kwargs


def build_assign_nuggetizer_kwargs(config: AssignExecutionConfig) -> dict[str, Any]:
    return {
        "assigner_model": config.model,
        "log_level": config.log_level,
        "use_azure_openai": config.use_azure_openai,
        "use_openrouter": config.use_openrouter,
        "store_trace": config.include_trace,
        "store_reasoning": config.include_reasoning,
        "reasoning_effort": config.reasoning_effort,
    }


def execute_direct_create(
    payload: dict[str, Any], *, config: CreateExecutionConfig
) -> CommandResponse:
    validation = validate_create_input(payload)
    nuggetizer = Nuggetizer(**build_create_nuggetizer_kwargs(config))
    request_obj = request_from_create_record_with_threshold(
        direct_create_record(payload),
        min_judgment=config.min_judgment,
    )
    if config.execution_mode == "async":
        scored_nuggets = asyncio.run(nuggetizer.async_create(request_obj))
    else:
        scored_nuggets = nuggetizer.create(request_obj)
    direct_output: dict[str, Any] = {
        "query": request_obj.query.text,
        "nuggets": [
            serialize_nugget(
                nugget,
                include_reasoning=config.include_reasoning,
                include_trace=config.include_trace,
                redact_prompts=config.redact_prompts,
            )
            for nugget in scored_nuggets
        ],
    }
    if config.include_reasoning:
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
            "execution_mode": config.execution_mode,
            "model": config.model,
            "creator_model": config.creator_model or config.model,
            "scorer_model": config.scorer_model or config.model,
            "min_judgment": config.min_judgment,
            "reasoning_effort": config.reasoning_effort,
        },
        validation=validation,
        artifacts=[make_data_artifact("create-result", direct_output)],
        metrics={"nugget_count": len(direct_output["nuggets"])},
    )


def execute_direct_assign(
    payload: dict[str, Any], *, config: AssignExecutionConfig
) -> CommandResponse:
    validation = validate_assign_input(payload)
    nuggetizer = Nuggetizer(**build_assign_nuggetizer_kwargs(config))
    if all(key in payload for key in ["answer_records", "nugget_record"]) or all(
        key in payload for key in ["answers_envelope", "nugget_envelope"]
    ):
        batch_records = joined_assign_batch_records(payload)
        batch_output: list[dict[str, Any]] = []
        for batch_record in batch_records:
            query = batch_record["query"]
            context = batch_record["context"]
            nuggets = batch_record["nuggets"]
            if config.execution_mode == "async":
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
                    include_reasoning=config.include_reasoning,
                    include_trace=config.include_trace,
                    redact_prompts=config.redact_prompts,
                )
            )
        return CommandResponse(
            command="assign",
            inputs={"source": "direct"},
            resolved={
                "input_mode": "direct",
                "assign_mode": "context",
                "execution_mode": config.execution_mode,
                "model": config.model,
                "reasoning_effort": config.reasoning_effort,
            },
            validation=validation,
            artifacts=[make_data_artifact("assign-result", batch_output)],
            metrics={"record_count": len(batch_output)},
        )

    query, context, nuggets = direct_assign_inputs(payload)
    if config.execution_mode == "async":
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
                include_reasoning=config.include_reasoning,
                include_trace=config.include_trace,
                redact_prompts=config.redact_prompts,
            )
            for nugget in assigned_nuggets
        ],
    }
    if config.include_reasoning:
        reasoning_traces = collect_reasoning_traces(assigned_nuggets)
        if reasoning_traces:
            direct_output["reasoning_traces"] = reasoning_traces
    return CommandResponse(
        command="assign",
        inputs={"source": "direct"},
        resolved={
            "input_mode": "direct",
            "assign_mode": "context",
            "execution_mode": config.execution_mode,
            "model": config.model,
            "reasoning_effort": config.reasoning_effort,
        },
        validation=validation,
        artifacts=[make_data_artifact("assign-result", direct_output)],
        metrics={"nugget_count": len(direct_output["nuggets"])},
    )
