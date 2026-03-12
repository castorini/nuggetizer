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


def request_from_create_record(record: dict[str, Any]) -> Request:
    """Convert a batch create record into a Request object."""
    require_keys(record, ["query", "candidates"])
    query_data = record["query"]
    query = Query(qid=query_data["qid"], text=query_data["text"])

    documents = []
    for candidate in record["candidates"]:
        if "judgment" in candidate and candidate["judgment"] <= 0:
            continue
        documents.append(
            Document(docid=candidate["docid"], segment=candidate["doc"]["segment"])
        )
    return Request(query=query, documents=documents)


def scored_nuggets_from_record(record: dict[str, Any]) -> list[ScoredNugget]:
    """Convert serialized nuggets into ScoredNugget objects."""
    return [
        ScoredNugget(text=nugget["text"], importance=nugget.get("importance", "vital"))
        for nugget in record["nuggets"]
    ]


def create_output_record(
    request: Request, scored_nuggets: list[ScoredNugget]
) -> dict[str, Any]:
    """Serialize create output in the legacy JSONL schema."""
    return {
        "query": request.query.text,
        "qid": request.query.qid,
        "nuggets": [
            {"text": nugget.text, "importance": nugget.importance}
            for nugget in scored_nuggets
        ],
    }


def assign_answer_output_record(
    answer_record: dict[str, Any],
    nugget_record: dict[str, Any],
    run_id: str,
    assigned_nuggets: Sequence[AssignedScoredNugget],
) -> dict[str, Any]:
    """Serialize answer assignment output in the legacy JSONL schema."""
    answer_text = " ".join(answer["text"] for answer in answer_record["answer"])
    return {
        "query": nugget_record["query"],
        "qid": nugget_record["qid"],
        "answer_text": answer_text,
        "response_length": answer_record["response_length"],
        "run_id": run_id,
        "nuggets": [
            {
                "text": nugget.text,
                "importance": nugget.importance,
                "assignment": nugget.assignment,
            }
            for nugget in assigned_nuggets
        ],
    }


def assign_retrieval_output_record(
    nugget_record: dict[str, Any],
    candidate: dict[str, Any],
    assigned_nuggets: Sequence[AssignedScoredNugget],
) -> dict[str, Any]:
    """Serialize retrieval assignment output in the legacy JSONL schema."""
    return {
        "text": nugget_record["query"],
        "qid": nugget_record["qid"],
        "candidate_text": candidate["doc"]["segment"],
        "docid": candidate["docid"],
        "nuggets": [
            {
                "text": nugget.text,
                "importance": nugget.importance,
                "assignment": nugget.assignment,
            }
            for nugget in assigned_nuggets
        ],
    }


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
