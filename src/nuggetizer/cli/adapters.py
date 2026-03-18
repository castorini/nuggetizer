from __future__ import annotations

from typing import Any, Sequence

from nuggetizer.core.metrics import calculate_nugget_scores
from nuggetizer.core.types import (
    AssignedScoredNugget,
    Document,
    Query,
    Request,
    ScoredNugget,
)

from .validation import require_keys


def collect_reasoning_traces(
    nuggets: Sequence[ScoredNugget | AssignedScoredNugget],
) -> list[str]:
    """Return de-duplicated reasoning traces in first-seen order."""
    traces: list[str] = []
    seen: set[str] = set()
    for nugget in nuggets:
        reasoning = (nugget.reasoning or "").strip()
        if not reasoning or reasoning in seen:
            continue
        seen.add(reasoning)
        traces.append(reasoning)
    return traces


def collect_nonempty_reasoning_traces(traces: Sequence[str | None]) -> list[str]:
    """Return non-empty reasoning traces in original order."""
    collected: list[str] = []
    for trace in traces:
        normalized = (trace or "").strip()
        if normalized:
            collected.append(normalized)
    return collected


def request_from_create_record(record: dict[str, Any]) -> Request:
    """Convert a batch create record into a Request object."""
    require_keys(record, ["query", "candidates"])
    query_data = record["query"]
    query = Query(qid=query_data["qid"], text=query_data["text"])

    documents = []
    for candidate in record["candidates"]:
        if "judgment" in candidate and candidate["judgment"] <= 0:
            continue
        doc = candidate["doc"]
        segment = doc.get("segment") or doc.get("contents", "")
        documents.append(Document(docid=candidate["docid"], segment=segment))
    return Request(query=query, documents=documents)


def scored_nuggets_from_record(record: dict[str, Any]) -> list[ScoredNugget]:
    """Convert serialized nuggets into ScoredNugget objects."""
    return [
        ScoredNugget(text=nugget["text"], importance=nugget.get("importance", "vital"))
        for nugget in record["nuggets"]
    ]


def create_output_record(
    request: Request,
    scored_nuggets: list[ScoredNugget],
    *,
    creator_reasoning_traces: Sequence[str | None] = (),
    include_reasoning: bool = False,
    include_trace: bool = False,
    redact_prompts: bool = False,
) -> dict[str, Any]:
    """Serialize create output in the legacy JSONL schema."""
    output = {
        "query": request.query.text,
        "qid": request.query.qid,
        "nuggets": [
            serialize_nugget(
                nugget,
                include_reasoning=include_reasoning,
                include_trace=include_trace,
                redact_prompts=redact_prompts,
            )
            for nugget in scored_nuggets
        ],
    }
    if include_reasoning:
        creator_traces = collect_nonempty_reasoning_traces(creator_reasoning_traces)
        scoring_traces = collect_reasoning_traces(scored_nuggets)
        if creator_traces:
            output["creator_reasoning_traces"] = creator_traces
        if scoring_traces:
            output["scoring_reasoning_traces"] = scoring_traces
    return output


def assign_answer_output_record(
    answer_record: dict[str, Any],
    nugget_record: dict[str, Any],
    run_id: str,
    assigned_nuggets: Sequence[AssignedScoredNugget],
    *,
    include_reasoning: bool = False,
    include_trace: bool = False,
    redact_prompts: bool = False,
) -> dict[str, Any]:
    """Serialize answer assignment output in the legacy JSONL schema."""
    answer_text = " ".join(answer["text"] for answer in answer_record["answer"])
    output = {
        "query": nugget_record["query"],
        "qid": nugget_record["qid"],
        "answer_text": answer_text,
        "response_length": answer_record["response_length"],
        "run_id": run_id,
        "nuggets": [
            serialize_nugget(
                nugget,
                include_reasoning=include_reasoning,
                include_trace=include_trace,
                redact_prompts=redact_prompts,
            )
            for nugget in assigned_nuggets
        ],
    }
    if include_reasoning:
        reasoning_traces = collect_reasoning_traces(assigned_nuggets)
        if reasoning_traces:
            output["reasoning_traces"] = reasoning_traces
    return output


def assign_retrieval_output_record(
    nugget_record: dict[str, Any],
    candidate: dict[str, Any],
    assigned_nuggets: Sequence[AssignedScoredNugget],
    *,
    include_reasoning: bool = False,
    include_trace: bool = False,
    redact_prompts: bool = False,
) -> dict[str, Any]:
    """Serialize retrieval assignment output in the legacy JSONL schema."""
    output = {
        "text": nugget_record["query"],
        "qid": nugget_record["qid"],
        "candidate_text": candidate["doc"]["segment"],
        "docid": candidate["docid"],
        "nuggets": [
            serialize_nugget(
                nugget,
                include_reasoning=include_reasoning,
                include_trace=include_trace,
                redact_prompts=redact_prompts,
            )
            for nugget in assigned_nuggets
        ],
    }
    if include_reasoning:
        reasoning_traces = collect_reasoning_traces(assigned_nuggets)
        if reasoning_traces:
            output["reasoning_traces"] = reasoning_traces
    return output


def serialize_nugget(
    nugget: ScoredNugget | AssignedScoredNugget,
    *,
    include_reasoning: bool,
    include_trace: bool,
    redact_prompts: bool,
) -> dict[str, Any]:
    """Serialize a nugget with optional reasoning and trace fields."""
    serialized: dict[str, Any] = {
        "text": nugget.text,
        "importance": nugget.importance,
    }
    if isinstance(nugget, AssignedScoredNugget):
        serialized["assignment"] = nugget.assignment
    if include_reasoning and nugget.reasoning is not None:
        serialized["reasoning"] = nugget.reasoning
    if include_trace and nugget.trace is not None:
        trace_dict = {
            "component": nugget.trace.component,
            "model": nugget.trace.model,
            "params": nugget.trace.params,
            "messages": None if redact_prompts else nugget.trace.messages,
            "usage": nugget.trace.usage,
            "raw_output": nugget.trace.raw_output,
            "window_start": nugget.trace.window_start,
            "window_end": nugget.trace.window_end,
            "timestamp_utc": nugget.trace.timestamp_utc,
        }
        serialized["trace"] = trace_dict
    return serialized


def metrics_output_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Calculate per-record metric rows in the legacy JSONL schema."""
    output_records: list[dict[str, Any]] = []
    for record in records:
        metrics = calculate_nugget_scores(record["qid"], record["nuggets"])
        output_records.append(
            {
                "qid": metrics.qid,
                "strict_vital_score": metrics.strict_vital_score,
                "strict_all_score": metrics.strict_all_score,
                "vital_score": metrics.vital_score,
                "all_score": metrics.all_score,
            }
        )
    return output_records
