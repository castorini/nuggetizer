from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, cast

from nuggetizer.cli.execution import (
    AssignExecutionConfig,
    CreateExecutionConfig,
    ExecutionMode,
    execute_direct_assign,
    execute_direct_create,
)
from nuggetizer.cli.normalize import unwrap_direct_create_payload
from nuggetizer.cli.responses import CommandResponse


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


def run_create_request(
    payload: dict[str, Any], *, config: ServerConfig
) -> CommandResponse:
    effective_config = _merge_config_with_payload(
        payload,
        config=config,
        allowed_fields=_CREATE_OVERRIDABLE_FIELDS,
        unwrap_create=True,
    )
    return execute_direct_create(
        payload,
        config=CreateExecutionConfig(
            model=effective_config.model,
            creator_model=effective_config.creator_model,
            scorer_model=effective_config.scorer_model,
            window_size=effective_config.window_size,
            max_nuggets=effective_config.max_nuggets,
            min_judgment=effective_config.min_judgment,
            execution_mode=cast(ExecutionMode, effective_config.execution_mode),
            log_level=effective_config.log_level,
            use_azure_openai=effective_config.use_azure_openai,
            use_openrouter=effective_config.use_openrouter,
            reasoning_effort=effective_config.reasoning_effort,
            include_trace=effective_config.include_trace,
            include_reasoning=effective_config.include_reasoning,
            redact_prompts=effective_config.redact_prompts,
            quiet=effective_config.quiet,
        ),
    )


def run_assign_request(
    payload: dict[str, Any], *, config: ServerConfig
) -> CommandResponse:
    effective_config = _merge_config_with_payload(
        payload,
        config=config,
        allowed_fields=_ASSIGN_OVERRIDABLE_FIELDS,
    )
    return execute_direct_assign(
        payload,
        config=AssignExecutionConfig(
            model=effective_config.model,
            execution_mode=cast(ExecutionMode, effective_config.execution_mode),
            log_level=effective_config.log_level,
            use_azure_openai=effective_config.use_azure_openai,
            use_openrouter=effective_config.use_openrouter,
            reasoning_effort=effective_config.reasoning_effort,
            include_trace=effective_config.include_trace,
            include_reasoning=effective_config.include_reasoning,
            redact_prompts=effective_config.redact_prompts,
            quiet=effective_config.quiet,
        ),
    )


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
