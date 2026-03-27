from __future__ import annotations

from typing import Any

from nuggetizer.core.types import ScoredNugget

from .adapters import scored_nuggets_from_record


def _unwrap_single_artifact_record(
    payload: dict[str, Any], *, artifact_name: str, record_name: str
) -> dict[str, Any]:
    schema_version = payload.get("schema_version")
    artifacts = payload.get("artifacts")
    if schema_version != "castorini.cli.v1" or not isinstance(artifacts, list):
        raise ValueError(f"{record_name} input must be a castorini.cli.v1 envelope")

    matched_artifact: dict[str, Any] | None = None
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        if artifact.get("name") != artifact_name:
            continue
        matched_artifact = artifact
        break

    if matched_artifact is None:
        raise ValueError(
            f"{record_name} envelope must contain artifact `{artifact_name}`"
        )

    artifact_payload = matched_artifact.get("data", matched_artifact.get("value"))
    if isinstance(artifact_payload, dict):
        return artifact_payload
    if isinstance(artifact_payload, list):
        if len(artifact_payload) != 1:
            raise ValueError(f"{record_name} envelope must contain exactly one record")
        record = artifact_payload[0]
        if isinstance(record, dict):
            return record

    raise ValueError(
        f"{record_name} envelope artifact `{artifact_name}` must contain a record object"
    )


def unwrap_generation_record(payload: dict[str, Any]) -> dict[str, Any]:
    return _unwrap_single_artifact_record(
        payload,
        artifact_name="generation-results",
        record_name="generation record",
    )


def unwrap_nugget_record(payload: dict[str, Any]) -> dict[str, Any]:
    return _unwrap_single_artifact_record(
        payload,
        artifact_name="create-result",
        record_name="nugget record",
    )


def _unwrap_castorini_envelope(payload: dict[str, Any]) -> dict[str, Any]:
    schema_version = payload.get("schema_version")
    artifacts = payload.get("artifacts")
    if schema_version != "castorini.cli.v1" or not isinstance(artifacts, list):
        return payload

    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        artifact_payload = artifact.get("data", artifact.get("value"))
        if (
            isinstance(artifact_payload, dict)
            and {
                "query",
                "candidates",
            }
            <= artifact_payload.keys()
        ):
            return artifact_payload
        if isinstance(artifact_payload, list):
            if len(artifact_payload) != 1:
                raise ValueError(
                    "direct create envelope input requires exactly one record"
                )
            record = artifact_payload[0]
            if isinstance(record, dict) and {"query", "candidates"} <= record.keys():
                return record

    raise ValueError(
        "direct create envelope input must contain a single artifact record "
        "with query and candidates"
    )


def direct_create_record(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize direct create input into the batch-like adapter shape."""
    payload = _unwrap_castorini_envelope(payload)
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
