from __future__ import annotations

from typing import Any

from nuggetizer.core.types import ScoredNugget

from .adapters import scored_nuggets_from_record


def direct_create_record(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize direct create input into the batch-like adapter shape."""
    query = payload["query"]
    query_text = (
        query
        if isinstance(query, str)
        else query["text"]
        if isinstance(query, dict) and isinstance(query.get("text"), str)
        else None
    )
    if query_text is None:
        raise ValueError(
            "direct create input requires `query` as a string or object with `text`"
        )
    return {
        "query": {
            "qid": "q0" if isinstance(query, str) else str(query.get("qid", "q0")),
            "text": query_text,
        },
        "candidates": [
            (
                {"docid": f"d{index}", "doc": {"segment": candidate}}
                if isinstance(candidate, str)
                else {
                    "docid": candidate.get("docid", f"d{index}"),
                    "doc": {
                        "segment": candidate.get("text")
                        or (
                            candidate.get("doc")
                            if isinstance(candidate.get("doc"), str)
                            else None
                        )
                        or (candidate.get("doc") or {}).get("segment")
                        or (candidate.get("doc") or {}).get("contents")
                    },
                }
            )
            for index, candidate in enumerate(payload["candidates"])
        ],
    }


def direct_assign_inputs(
    payload: dict[str, Any],
) -> tuple[str, str, list[ScoredNugget]]:
    """Normalize direct assign input into the core assign call shape."""
    nugget_record = {
        "nuggets": payload["nuggets"],
    }
    return (
        payload["query"],
        payload["context"],
        scored_nuggets_from_record(nugget_record),
    )
