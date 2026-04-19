from __future__ import annotations

import asyncio
from typing import Any, cast

import pytest

from nuggetizer.core.types import Document, Query, Request, ScoredNugget, Trace
from nuggetizer.models.nuggetizer import Nuggetizer

pytestmark = pytest.mark.core


class FakeSyncLLM:
    def __init__(self, responses: list[tuple[str, dict[str, Any] | None, str | None]]):
        self.responses = list(responses)
        self.model = "fake-sync-model"
        self.calls: list[list[dict[str, str]]] = []

    def run(
        self, messages: list[dict[str, str]], temperature: float = 0
    ) -> tuple[str, int, dict[str, Any] | None, str | None]:
        del temperature
        self.calls.append(messages)
        response, usage, reasoning = self.responses.pop(0)
        return response, len(response), usage, reasoning


class FakeAsyncLLM:
    def __init__(self, responses: list[tuple[str, dict[str, Any] | None, str | None]]):
        self.responses = list(responses)
        self.model = "fake-async-model"
        self.calls: list[list[dict[str, str]]] = []

    async def run(
        self, messages: list[dict[str, str]], temperature: float = 0
    ) -> tuple[str, int, dict[str, Any] | None, str | None]:
        del temperature
        self.calls.append(messages)
        response, usage, reasoning = self.responses.pop(0)
        return response, len(response), usage, reasoning


def _request(doc_count: int = 3) -> Request:
    return Request(
        query=Query(qid="q1", text="What is Python used for?"),
        documents=[
            Document(
                docid=f"d{i}", segment=f"Document {i} mentions Python use case {i}."
            )
            for i in range(1, doc_count + 1)
        ],
    )


def test_create_respects_window_size_and_collects_reasoning_and_trace() -> None:
    nuggetizer = Nuggetizer(
        window_size=2, max_nuggets=3, store_trace=True, store_reasoning=True
    )
    creator_llm = FakeSyncLLM(
        [
            ("['alpha', 'beta']", {"prompt_tokens": 2}, "creator-1"),
            ("['alpha', 'beta', 'gamma', 'delta']", {"prompt_tokens": 3}, "creator-2"),
        ]
    )
    scorer_llm = FakeSyncLLM(
        [
            ("['vital', 'okay']", {"completion_tokens": 2}, "score-1"),
            ("['vital']", {"completion_tokens": 1}, "score-2"),
        ]
    )
    cast(Any, nuggetizer).creator_llm = creator_llm
    cast(Any, nuggetizer).scorer_llm = scorer_llm

    nuggets = nuggetizer.create(_request())

    assert [n.text for n in nuggets] == ["alpha", "beta", "gamma"]
    assert [n.importance for n in nuggets] == ["vital", "okay", "vital"]
    assert [n.reasoning for n in nuggets] == ["score-1", "score-1", "score-2"]
    assert nuggetizer.get_creator_reasoning() == "creator-2"
    assert nuggetizer.get_creator_reasoning_traces() == ["creator-1", "creator-2"]
    assert len(creator_llm.calls) == 2
    assert len(scorer_llm.calls) == 2
    assert nuggets[0].trace == Trace(
        component="scorer",
        model="fake-sync-model",
        params={"temperature": 0.0},
        messages=scorer_llm.calls[0],
        usage={"completion_tokens": 2},
        raw_output="['vital', 'okay']",
        window_start=0,
        window_end=2,
        timestamp_utc=nuggets[0].trace.timestamp_utc if nuggets[0].trace else None,
    )
    assert nuggets[2].trace is not None
    assert nuggets[2].trace.window_start == 2
    assert nuggets[2].trace.window_end == 3


def test_create_falls_back_to_okay_after_parse_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("nuggetizer.models.nuggetizer.MAX_TRIALS", 1)
    nuggetizer = Nuggetizer(store_reasoning=True)
    cast(Any, nuggetizer).creator_llm = FakeSyncLLM([("['alpha']", None, "creator")])
    cast(Any, nuggetizer).scorer_llm = FakeSyncLLM(
        [("not-a-python-list", {"tokens": 1}, "score")]
    )

    nuggets = nuggetizer.create(_request(doc_count=1))

    assert len(nuggets) == 1
    assert nuggets[0].text == "alpha"
    assert nuggets[0].importance == "okay"
    assert nuggets[0].reasoning == "score"


def test_assign_falls_back_to_failed_after_parse_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("nuggetizer.models.nuggetizer.MAX_TRIALS", 1)
    nuggetizer = Nuggetizer(store_reasoning=True)
    scored_nuggets = [
        ScoredNugget(text="alpha", importance="vital"),
        ScoredNugget(text="beta", importance="okay"),
    ]
    cast(Any, nuggetizer).assigner_llm = FakeSyncLLM(
        [("not-a-python-list", {"tokens": 1}, "assign")]
    )

    assigned = nuggetizer.assign("query", "context", scored_nuggets)

    assert [n.assignment for n in assigned] == ["failed", "failed"]
    assert [n.reasoning for n in assigned] == ["assign", "assign"]


def test_async_create_and_assign_match_sync_behavior() -> None:
    request = _request(doc_count=2)

    sync_nuggetizer = Nuggetizer(window_size=2, store_trace=True, store_reasoning=True)
    cast(Any, sync_nuggetizer).creator_llm = FakeSyncLLM(
        [("['alpha', 'beta']", {"tokens": 1}, "creator")]
    )
    cast(Any, sync_nuggetizer).scorer_llm = FakeSyncLLM(
        [("['vital', 'okay']", {"tokens": 2}, "score")]
    )
    cast(Any, sync_nuggetizer).assigner_llm = FakeSyncLLM(
        [("['support', 'not_support']", {"tokens": 3}, "assign")]
    )

    async_nuggetizer = Nuggetizer(window_size=2, store_trace=True, store_reasoning=True)
    cast(Any, async_nuggetizer).creator_llm_async = FakeAsyncLLM(
        [("['alpha', 'beta']", {"tokens": 1}, "creator")]
    )
    cast(Any, async_nuggetizer).scorer_llm_async = FakeAsyncLLM(
        [("['vital', 'okay']", {"tokens": 2}, "score")]
    )
    cast(Any, async_nuggetizer).assigner_llm_async = FakeAsyncLLM(
        [("['support', 'not_support']", {"tokens": 3}, "assign")]
    )

    sync_created = sync_nuggetizer.create(request)
    async_created = asyncio.run(async_nuggetizer.async_create(request))
    assert [(n.text, n.importance, n.reasoning) for n in async_created] == [
        (n.text, n.importance, n.reasoning) for n in sync_created
    ]
    assert [n.trace.component if n.trace else None for n in async_created] == [
        n.trace.component if n.trace else None for n in sync_created
    ]

    sync_assigned = sync_nuggetizer.assign(request.query.text, "context", sync_created)
    async_assigned = asyncio.run(
        async_nuggetizer.async_assign(request.query.text, "context", async_created)
    )
    assert [
        (n.text, n.importance, n.assignment, n.reasoning) for n in async_assigned
    ] == [(n.text, n.importance, n.assignment, n.reasoning) for n in sync_assigned]


def test_batch_methods_preserve_order_and_validate_lengths() -> None:
    nuggetizer = Nuggetizer()

    request_one = _request(doc_count=1)
    request_two = Request(
        query=Query(qid="q2", text="Who created Python?"),
        documents=[Document(docid="d2", segment="Guido van Rossum created Python.")],
    )

    cast(Any, nuggetizer).creator_llm = FakeSyncLLM(
        [("['alpha']", None, None), ("['beta']", None, None)]
    )
    cast(Any, nuggetizer).scorer_llm = FakeSyncLLM(
        [("['vital']", None, None), ("['okay']", None, None)]
    )

    created = nuggetizer.create_batch([request_one, request_two])
    assert [[n.text for n in batch] for batch in created] == [["alpha"], ["beta"]]

    with pytest.raises(ValueError, match="same length"):
        nuggetizer.assign_batch(["q1"], ["context-1", "context-2"], [created[0]])


def test_operations_only_initialize_required_handlers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_API_BASE", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_API_VERSION", raising=False)

    create_nuggetizer = Nuggetizer()
    cast(Any, create_nuggetizer).creator_llm = FakeSyncLLM([("['alpha']", None, None)])
    cast(Any, create_nuggetizer).scorer_llm = FakeSyncLLM([("['vital']", None, None)])

    created = create_nuggetizer.create(_request(doc_count=1))

    assert [n.text for n in created] == ["alpha"]
    assert create_nuggetizer.assigner_llm is None

    assign_nuggetizer = Nuggetizer()
    cast(Any, assign_nuggetizer).assigner_llm = FakeSyncLLM(
        [("['support']", None, None)]
    )

    assigned = assign_nuggetizer.assign("query", "context", created)

    assert [n.assignment for n in assigned] == ["support"]
    assert assign_nuggetizer.creator_llm is None
    assert assign_nuggetizer.scorer_llm is None
