from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, cast

import pytest
from openai import AsyncOpenAI, OpenAI

from nuggetizer.core.async_llm import AsyncLLMHandler
from nuggetizer.core.llm import LLMHandler

pytestmark = pytest.mark.core


def _fake_completion(message: Any) -> Any:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
        usage=SimpleNamespace(
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
        ),
    )


def _fake_responses_response(
    *,
    output_text: str,
    reasoning_summary: list[Any] | None = None,
) -> Any:
    return SimpleNamespace(
        output_text=output_text,
        output=[
            SimpleNamespace(type="message", content=[]),
            SimpleNamespace(type="reasoning", summary=reasoning_summary or []),
        ],
        usage=SimpleNamespace(input_tokens=1, output_tokens=1, total_tokens=2),
    )


def test_sync_llm_handler_uses_responses_api_for_openai_reasoning_models() -> None:
    recorded_kwargs: dict[str, Any] = {}
    handler = LLMHandler(
        model="gpt-5.4",
        api_keys="test-key",
        reasoning_effort="minimal",
    )

    def fake_create(**kwargs: Any) -> Any:
        recorded_kwargs.update(kwargs)
        return _fake_responses_response(
            output_text="response",
            reasoning_summary=[SimpleNamespace(type="summary_text", text="chain")],
        )

    handler.client = cast(
        OpenAI,
        SimpleNamespace(responses=SimpleNamespace(create=fake_create)),
    )

    response, _, _, reasoning = handler.run([{"role": "user", "content": "prompt"}])

    assert response == "response"
    assert reasoning == "chain"
    assert recorded_kwargs["reasoning"] == {
        "effort": "minimal",
        "summary": "auto",
    }
    assert recorded_kwargs["input"] == [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "prompt"}],
        }
    ]


def test_async_llm_handler_uses_responses_api_for_openai_reasoning_models() -> None:
    recorded_kwargs: dict[str, Any] = {}
    handler = AsyncLLMHandler(
        model="gpt-5.4",
        api_keys="test-key",
        reasoning_effort="xhigh",
    )

    async def fake_create(**kwargs: Any) -> Any:
        recorded_kwargs.update(kwargs)
        return _fake_responses_response(
            output_text="response",
            reasoning_summary=[SimpleNamespace(type="summary_text", text="trace")],
        )

    handler.client = cast(
        AsyncOpenAI,
        SimpleNamespace(responses=SimpleNamespace(create=fake_create)),
    )

    response, _, _, reasoning = asyncio.run(
        handler.run([{"role": "user", "content": "prompt"}])
    )

    assert response == "response"
    assert reasoning == "trace"
    assert recorded_kwargs["reasoning"] == {
        "effort": "xhigh",
        "summary": "auto",
    }
    assert recorded_kwargs["input"] == [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "prompt"}],
        }
    ]


def test_sync_llm_handler_uses_responses_api_for_openrouter_reasoning_models() -> None:
    recorded_kwargs: dict[str, Any] = {}
    handler = LLMHandler(
        model="openrouter/openai/o4-mini",
        api_keys="test-key",
        api_type="openrouter",
        api_base="https://openrouter.ai/api/v1",
        reasoning_effort="high",
    )

    def fake_create(**kwargs: Any) -> Any:
        recorded_kwargs.update(kwargs)
        return _fake_responses_response(
            output_text="response",
            reasoning_summary=["openrouter summary"],
        )

    handler.client = cast(
        OpenAI,
        SimpleNamespace(
            base_url="https://openrouter.ai/api/v1",
            responses=SimpleNamespace(create=fake_create),
        ),
    )

    response, _, _, reasoning = handler.run([{"role": "user", "content": "prompt"}])

    assert response == "response"
    assert reasoning == "openrouter summary"
    assert recorded_kwargs["reasoning"] == {
        "effort": "high",
        "summary": "auto",
    }


def test_sync_llm_handler_prefers_direct_reasoning_for_openrouter_responses() -> None:
    handler = LLMHandler(
        model="openrouter/openai/o4-mini",
        api_keys="test-key",
        api_type="openrouter",
        api_base="https://openrouter.ai/api/v1",
        reasoning_effort="high",
    )

    def fake_create(**kwargs: Any) -> Any:
        del kwargs
        return SimpleNamespace(
            output_text="response",
            output=[
                SimpleNamespace(
                    type="reasoning",
                    reasoning="raw openrouter reasoning",
                    summary=["openrouter summary"],
                )
            ],
            usage=SimpleNamespace(input_tokens=1, output_tokens=1, total_tokens=2),
        )

    handler.client = cast(
        OpenAI,
        SimpleNamespace(
            base_url="https://openrouter.ai/api/v1",
            responses=SimpleNamespace(create=fake_create),
        ),
    )

    _, _, _, reasoning = handler.run([{"role": "user", "content": "prompt"}])

    assert reasoning == "raw openrouter reasoning"


def test_sync_llm_handler_uses_openrouter_reasoning_payload() -> None:
    recorded_kwargs: dict[str, Any] = {}
    handler = LLMHandler(
        model="openrouter/hunter-alpha",
        api_keys="test-key",
        api_type="openrouter",
        api_base="https://openrouter.ai/api/v1",
        reasoning_effort="high",
    )

    def fake_create(**kwargs: Any) -> Any:
        recorded_kwargs.update(kwargs)
        return _fake_completion(
            SimpleNamespace(
                content="response", model_extra={"reasoning": "openrouter-chain"}
            )
        )

    handler.client = cast(
        OpenAI,
        SimpleNamespace(
            base_url="https://openrouter.ai/api/v1",
            chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create)),
        ),
    )

    response, _, _, reasoning = handler.run([{"role": "user", "content": "prompt"}])

    assert response == "response"
    assert reasoning == "openrouter-chain"
    assert recorded_kwargs["extra_body"] == {
        "reasoning": {"effort": "high", "exclude": False}
    }
    assert "reasoning_effort" not in recorded_kwargs


def test_async_llm_handler_uses_openrouter_reasoning_payload() -> None:
    recorded_kwargs: dict[str, Any] = {}
    handler = AsyncLLMHandler(
        model="openrouter/hunter-alpha",
        api_keys="test-key",
        api_type="openrouter",
        api_base="https://openrouter.ai/api/v1",
        reasoning_effort="high",
    )

    async def fake_create(**kwargs: Any) -> Any:
        recorded_kwargs.update(kwargs)
        return _fake_completion(
            SimpleNamespace(
                content="response", model_extra={"reasoning": "openrouter-chain"}
            )
        )

    handler.client = cast(
        AsyncOpenAI,
        SimpleNamespace(
            base_url="https://openrouter.ai/api/v1",
            chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create)),
        ),
    )

    response, _, _, reasoning = asyncio.run(
        handler.run([{"role": "user", "content": "prompt"}])
    )

    assert response == "response"
    assert reasoning == "openrouter-chain"
    assert recorded_kwargs["extra_body"] == {
        "reasoning": {"effort": "high", "exclude": False}
    }
    assert "reasoning_effort" not in recorded_kwargs
