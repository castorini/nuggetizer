from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from typing import cast

from nuggetizer.core.async_llm import AsyncLLMHandler
from nuggetizer.core.llm import LLMHandler
from openai import AsyncOpenAI, OpenAI


def _fake_completion(message: Any) -> Any:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
        usage=SimpleNamespace(
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
        ),
    )


def test_sync_llm_handler_forwards_reasoning_effort() -> None:
    recorded_kwargs: dict[str, Any] = {}
    handler = LLMHandler(
        model="gpt-5.4",
        api_keys="test-key",
        reasoning_effort="minimal",
    )

    def fake_create(**kwargs: Any) -> Any:
        recorded_kwargs.update(kwargs)
        return _fake_completion(SimpleNamespace(content="response", reasoning="chain"))

    handler.client = cast(
        OpenAI,
        SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))
        ),
    )

    response, _, _, reasoning = handler.run([{"role": "user", "content": "prompt"}])

    assert response == "response"
    assert reasoning == "chain"
    assert recorded_kwargs["reasoning_effort"] == "minimal"


def test_async_llm_handler_forwards_reasoning_effort() -> None:
    recorded_kwargs: dict[str, Any] = {}
    handler = AsyncLLMHandler(
        model="gpt-5.4",
        api_keys="test-key",
        reasoning_effort="xhigh",
    )

    async def fake_create(**kwargs: Any) -> Any:
        recorded_kwargs.update(kwargs)
        return _fake_completion(
            SimpleNamespace(content="response", reasoning_content="trace")
        )

    handler.client = cast(
        AsyncOpenAI,
        SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))
        ),
    )

    response, _, _, reasoning = asyncio.run(
        handler.run([{"role": "user", "content": "prompt"}])
    )

    assert response == "response"
    assert reasoning == "trace"
    assert recorded_kwargs["reasoning_effort"] == "xhigh"


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
    assert recorded_kwargs["reasoning"] == {"effort": "high", "exclude": False}
    assert "reasoning_effort" not in recorded_kwargs
