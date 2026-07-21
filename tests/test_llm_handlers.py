from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from openai import AsyncOpenAI, OpenAI

from nuggetizer.core.async_llm import AsyncLLMHandler
from nuggetizer.core.cache import _get_database_path
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


def test_llm_handler_prefers_openai_key_before_openrouter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("nuggetizer.core.llm.get_openai_api_key", lambda: "openai-key")
    monkeypatch.setattr(
        "nuggetizer.core.llm.get_openrouter_api_key", lambda: "openrouter-key"
    )

    handler = LLMHandler(model="gpt-4o")

    assert handler.api_keys == ["openai-key"]


def test_llm_handler_falls_back_to_openrouter_when_openai_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("nuggetizer.core.llm.get_openai_api_key", lambda: None)
    monkeypatch.setattr(
        "nuggetizer.core.llm.get_openrouter_api_key", lambda: "openrouter-key"
    )

    handler = LLMHandler(model="openrouter/model-name")

    assert handler.api_keys == ["openrouter-key"]


def test_llm_handler_requires_openrouter_key_when_forced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("nuggetizer.core.llm.get_openrouter_api_key", lambda: None)

    with pytest.raises(ValueError, match="use_openrouter=True"):
        LLMHandler(model="openrouter/model-name", use_openrouter=True)


def test_llm_handler_uses_vllm_placeholder_key_and_local_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("nuggetizer.core.llm.get_vllm_api_key", lambda: "EMPTY")

    handler = LLMHandler(model="meta-llama", use_vllm=True, vllm_port=8012)

    assert handler.api_keys == ["EMPTY"]
    assert "localhost:8012" in str(handler.client.base_url)


def test_async_llm_handler_uses_vllm_placeholder_key_and_local_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("nuggetizer.core.async_llm.get_vllm_api_key", lambda: "EMPTY")

    handler = AsyncLLMHandler(model="meta-llama", use_vllm=True, vllm_port=8012)

    assert handler.api_keys == ["EMPTY"]
    assert "localhost:8012" in str(handler.client.base_url)


def test_llm_handler_prefers_azure_settings_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "nuggetizer.core.llm.get_azure_openai_args",
        lambda: {
            "api_type": "azure",
            "api_version": "2024-01-01",
            "api_base": "https://azure.example.com",
            "api_key": "azure-key",
        },
    )
    monkeypatch.setattr(
        "nuggetizer.core.llm.get_openai_api_key", lambda: "fallback-openai"
    )

    handler = LLMHandler(model="gpt-4o", use_azure_openai=True)

    assert handler.api_keys == ["azure-key"]
    assert cast(Any, handler.client)._api_version == "2024-01-01"


def test_sync_reasoning_models_merge_system_and_user_messages() -> None:
    merged = LLMHandler._normalize_messages(
        [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
            {"role": "assistant", "content": "assistant"},
        ],
        "o4-mini",
    )

    assert merged == [
        {"role": "user", "content": "system\nuser"},
        {"role": "assistant", "content": "assistant"},
    ]


def test_sync_llm_handler_uses_sqlite_cache_when_enabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CACHE_DIR", str(tmp_path))
    call_count = 0
    handler = LLMHandler(model="gpt-4o", api_keys="test-key")

    def fake_create(**kwargs: Any) -> Any:
        nonlocal call_count
        del kwargs
        call_count += 1
        return _fake_completion(SimpleNamespace(content="cached response"))

    handler.client = cast(
        OpenAI,
        SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))
        ),
    )

    first_response, _, _, _ = handler.run([{"role": "user", "content": "prompt"}])
    second_response, _, _, _ = handler.run([{"role": "user", "content": "prompt"}])

    assert first_response == "cached response"
    assert second_response == "cached response"
    assert call_count == 1
    assert _get_database_path(str(tmp_path)).exists()


def test_sync_llm_handler_stores_request_json_in_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CACHE_DIR", str(tmp_path))
    handler = LLMHandler(model="gpt-4o", api_keys="test-key")

    def fake_create(**kwargs: Any) -> Any:
        del kwargs
        return _fake_completion(SimpleNamespace(content="cached response"))

    handler.client = cast(
        OpenAI,
        SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))
        ),
    )

    handler.run([{"role": "user", "content": "prompt"}], temperature=0.25)

    with sqlite3.connect(_get_database_path(str(tmp_path))) as connection:
        row = connection.execute(
            "SELECT request_json, response_json FROM chat_completions_cache"
        ).fetchone()

    assert row is not None
    request_json, response_json = row
    assert json.loads(request_json) == {
        "max_completion_tokens": 4096,
        "messages": [{"content": "prompt", "role": "user"}],
        "model": "gpt-4o",
        "temperature": 0.25,
        "timeout": 60,
    }
    assert json.loads(response_json)["choices"][0]["message"]["content"] == (
        "cached response"
    )


def test_sync_llm_handler_cache_key_distinguishes_models(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CACHE_DIR", str(tmp_path))
    call_count = 0

    def fake_create(**kwargs: Any) -> Any:
        nonlocal call_count
        call_count += 1
        return _fake_completion(SimpleNamespace(content=kwargs["model"]))

    first_handler = LLMHandler(model="gpt-4o", api_keys="test-key")
    first_handler.client = cast(
        OpenAI,
        SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))
        ),
    )
    second_handler = LLMHandler(model="gpt-4.1", api_keys="test-key")
    second_handler.client = cast(
        OpenAI,
        SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))
        ),
    )

    first_response, _, _, _ = first_handler.run([{"role": "user", "content": "prompt"}])
    second_response, _, _, _ = second_handler.run(
        [{"role": "user", "content": "prompt"}]
    )

    with sqlite3.connect(_get_database_path(str(tmp_path))) as connection:
        row_count = connection.execute(
            "SELECT COUNT(*) FROM chat_completions_cache"
        ).fetchone()

    assert first_response == "gpt-4o"
    assert second_response == "gpt-4.1"
    assert call_count == 2
    assert row_count is not None
    assert row_count[0] == 2
