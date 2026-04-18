from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import tiktoken

SUPPORTED_REASONING_EFFORTS = (
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
)


def validate_reasoning_effort(reasoning_effort: str | None) -> None:
    if reasoning_effort is None:
        return
    if reasoning_effort not in SUPPORTED_REASONING_EFFORTS:
        raise ValueError(
            "Unsupported reasoning_effort: "
            f"{reasoning_effort}. Expected one of "
            f"{', '.join(SUPPORTED_REASONING_EFFORTS)}."
        )


def resolve_api_settings(
    *,
    model: str,
    api_keys: str | list[str] | None,
    api_type: str | None,
    api_base: str | None,
    api_version: str | None,
    use_azure_openai: bool,
    use_openrouter: bool,
    use_vllm: bool,
    openrouter_api_key: str | None,
    vllm_port: int,
    get_azure_openai_args: Callable[[], dict[str, str | None]],
    get_openai_api_key: Callable[[], str | None],
    get_openrouter_api_key: Callable[[], str | None],
    get_vllm_api_key: Callable[[], str | None],
) -> tuple[str, str | None, str | None, list[str]]:
    if use_azure_openai and (
        api_keys is None
        or (api_type == "azure" or (api_type is None and "gpt" in model.lower()))
    ):
        azure_args = get_azure_openai_args()
        api_type = "azure"
        api_base = azure_args.get("api_base")
        api_version = azure_args.get("api_version")
        api_keys = azure_args.get("api_key", api_keys) or get_openai_api_key()
    elif api_keys is None:
        if use_vllm:
            api_keys = get_vllm_api_key()
            api_base = f"http://localhost:{vllm_port}/v1"
            api_type = "vllm"
        elif use_openrouter:
            resolved_openrouter_key = openrouter_api_key or get_openrouter_api_key()
            if resolved_openrouter_key is None:
                raise ValueError("use_openrouter=True but no OpenRouter API key found")
            api_keys = resolved_openrouter_key
            api_base = "https://openrouter.ai/api/v1"
            api_type = "openrouter"
        else:
            openai_key = get_openai_api_key()
            if openai_key is not None:
                api_keys = openai_key
                api_type = "openai"
            else:
                resolved_openrouter_key = openrouter_api_key or get_openrouter_api_key()
                if resolved_openrouter_key is not None:
                    api_keys = resolved_openrouter_key
                    api_base = "https://openrouter.ai/api/v1"
                    api_type = "openrouter"
    else:
        api_type = "openai"

    if api_type is None or api_keys is None:
        raise ValueError(
            "No valid API key found. Please provide either:\n"
            "1. OpenAI API key (OPENAI_API_KEY environment variable)\n"
            "2. OpenRouter API key (OPENROUTER_API_KEY environment variable)\n"
            "3. Azure OpenAI credentials (AZURE_OPENAI_API_KEY, etc.)\n"
            "4. Use vLLM local server (use_vllm=True)\n"
            "5. Pass api_keys parameter directly to Nuggetizer constructor"
        )

    resolved_api_keys = [api_keys] if isinstance(api_keys, str) else api_keys
    return api_type, api_base, api_version, resolved_api_keys


def uses_reasoning_style_api(model: str) -> bool:
    return "o1" in model or "o3" in model or "o4" in model or "gpt-5" in model


def uses_responses_reasoning_api(model: str, reasoning_effort: str | None) -> bool:
    return reasoning_effort is not None and uses_reasoning_style_api(model)


def normalize_messages(
    messages: list[dict[str, str]], model: str
) -> list[dict[str, str]]:
    if ("o1" in model or "o3" in model or "o4" in model) and len(messages) >= 2:
        normalized_messages = [message.copy() for message in messages[1:]]
        normalized_messages[0]["content"] = (
            messages[0]["content"] + "\n" + messages[1]["content"]
        )
        return normalized_messages
    return messages


def build_responses_input(
    messages: list[dict[str, str]], model: str
) -> list[dict[str, Any]]:
    normalized_messages = normalize_messages(messages, model)
    return [
        {
            "type": "message",
            "role": message["role"],
            "content": [{"type": "input_text", "text": message["content"]}],
        }
        for message in normalized_messages
    ]


def get_mapping_value(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def get_sequence_value(item: Any, key: str) -> Sequence[Any]:
    value = get_mapping_value(item, key, [])
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return value
    return []


def get_text_value(item: Any, key: str) -> str | None:
    value = get_mapping_value(item, key)
    if value:
        return str(value)
    return None


def is_openrouter_client(client: Any) -> bool:
    return "openrouter.ai" in str(getattr(client, "base_url", ""))


def extract_responses_text_and_reasoning(
    response: Any, *, prefer_direct_reasoning: bool
) -> tuple[str, str | None]:
    text = get_text_value(response, "output_text") or ""
    if not text:
        for item in get_sequence_value(response, "output"):
            if get_text_value(item, "type") != "message":
                continue
            for content in get_sequence_value(item, "content"):
                if get_text_value(content, "type") == "output_text":
                    text = get_text_value(content, "text") or ""

    reasoning_parts: list[str] = []
    for item in get_sequence_value(response, "output"):
        if get_text_value(item, "type") != "reasoning":
            continue
        direct_parts: list[str] = []
        summary_parts: list[str] = []
        for summary in get_sequence_value(item, "summary"):
            if isinstance(summary, str):
                summary_parts.append(summary)
                continue
            summary_type = get_text_value(summary, "type")
            if summary_type not in (None, "summary_text"):
                continue
            summary_text = get_text_value(summary, "text")
            if summary_text:
                summary_parts.append(summary_text)
        for key in ("reasoning", "reasoning_content"):
            reasoning_text = get_text_value(item, key)
            if reasoning_text:
                direct_parts.append(reasoning_text)
        for content in get_sequence_value(item, "content"):
            if isinstance(content, str):
                direct_parts.append(content)
                continue
            content_text = get_text_value(content, "text")
            if content_text:
                direct_parts.append(content_text)
        reasoning_parts.extend(
            direct_parts or summary_parts
            if prefer_direct_reasoning
            else summary_parts or direct_parts
        )

    reasoning = "\n".join(reasoning_parts) if reasoning_parts else None
    return text, reasoning


def extract_reasoning_content(message: Any) -> str | None:
    for key in ("reasoning", "reasoning_content"):
        value = get_text_value(message, key)
        if value:
            return value
    model_extra = get_mapping_value(message, "model_extra")
    if isinstance(model_extra, dict):
        for key in ("reasoning", "reasoning_content"):
            value = model_extra.get(key)
            if value:
                return str(value)
    return None


def build_usage_metadata(usage: Any) -> dict[str, Any] | None:
    if not usage:
        return None
    return {
        "prompt_tokens": get_mapping_value(
            usage, "input_tokens", get_mapping_value(usage, "prompt_tokens")
        ),
        "completion_tokens": get_mapping_value(
            usage, "output_tokens", get_mapping_value(usage, "completion_tokens")
        ),
        "total_tokens": get_mapping_value(usage, "total_tokens"),
    }


def get_response_encoding(model: str) -> Any:
    try:
        if "gpt-4o" in model or "gpt-4.1" in model:
            return tiktoken.get_encoding("o200k_base")
        return tiktoken.get_encoding(model)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")
