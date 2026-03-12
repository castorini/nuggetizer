from __future__ import annotations

from typing import Any

from nuggetizer.core.types import ScoredNugget

from .adapters import scored_nuggets_from_record


def direct_create_record(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize direct create input into the batch-like adapter shape."""
    return {
        "query": {"qid": "q0", "text": payload["query"]},
        "candidates": [
            (
                {"docid": f"d{index}", "doc": {"segment": candidate}}
                if isinstance(candidate, str)
                else {
                    "docid": candidate.get("docid", f"d{index}"),
                    "doc": {"segment": candidate["text"]},
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
