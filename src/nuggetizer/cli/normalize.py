from __future__ import annotations

from typing import Any

from nuggetizer.core.types import ScoredNugget

from .adapters import scored_nuggets_from_record


def _normalize_umbrela_judgments(judgments: list[dict[str, Any]]) -> dict[str, Any]:
    if not judgments:
        raise ValueError("umbrela create input requires a non-empty `judgments` list")

    query_text: str | None = None
    candidates: list[dict[str, Any]] = []
    for index, judgment in enumerate(judgments):
        if not isinstance(judgment, dict):
            raise ValueError("umbrela create input `judgments` must contain objects")
        judgment_query = judgment.get("query")
        passage = judgment.get("passage")
        if not isinstance(judgment_query, str):
            raise ValueError(
                "umbrela create input judgments must contain `query` as a string"
            )
        if not isinstance(passage, str):
            raise ValueError(
                "umbrela create input judgments must contain `passage` as a string"
            )
        if query_text is None:
            query_text = judgment_query
        elif judgment_query != query_text:
            raise ValueError(
                "umbrela create input requires all judgments to share the same query"
            )

        candidate: dict[str, Any] = {
            "docid": str(judgment.get("docid", f"d{index}")),
            "doc": {"segment": passage},
        }
        if "judgment" in judgment:
            candidate["judgment"] = judgment["judgment"]
        candidates.append(candidate)

    return {
        "query": {"qid": "q0", "text": query_text},
        "candidates": candidates,
    }


def _unwrap_artifact_payload(
    payload: dict[str, Any], *, artifact_name: str, record_name: str
) -> Any:
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

    return matched_artifact.get("data", matched_artifact.get("value"))


def _unwrap_single_artifact_record(
    payload: dict[str, Any], *, artifact_name: str, record_name: str
) -> dict[str, Any]:
    artifact_payload = _unwrap_artifact_payload(
        payload,
        artifact_name=artifact_name,
        record_name=record_name,
    )
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


def _unwrap_artifact_records(
    payload: dict[str, Any], *, artifact_name: str, record_name: str
) -> list[dict[str, Any]]:
    artifact_payload = _unwrap_artifact_payload(
        payload,
        artifact_name=artifact_name,
        record_name=record_name,
    )
    if not isinstance(artifact_payload, list) or not artifact_payload:
        raise ValueError(
            f"{record_name} envelope artifact `{artifact_name}` must contain a record list"
        )
    if not all(isinstance(record, dict) for record in artifact_payload):
        raise ValueError(
            f"{record_name} envelope artifact `{artifact_name}` must contain record objects"
        )
    return list(artifact_payload)


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


def _answer_record_to_query_context(answer_record: dict[str, Any]) -> tuple[str, str]:
    topic = answer_record.get("topic")
    answer = answer_record.get("answer")
    if not isinstance(topic, str):
        raise ValueError("answer record must contain `topic` as a string")
    if not isinstance(answer, list):
        raise ValueError("answer record must contain `answer` as a list")
    try:
        context = " ".join(str(answer_sentence["text"]) for answer_sentence in answer)
    except (TypeError, KeyError) as error:
        raise ValueError(
            "answer record `answer` entries must contain string `text` fields"
        ) from error
    return topic, context


def _answer_record_to_assignment_input(
    answer_record: dict[str, Any], nugget_record: dict[str, Any]
) -> dict[str, Any]:
    query, context = _answer_record_to_query_context(answer_record)
    topic_id = answer_record.get("topic_id")
    qid = nugget_record.get("qid")
    if not isinstance(topic_id, str):
        raise ValueError("answer record must contain `topic_id` as a string")
    if not isinstance(qid, str):
        raise ValueError("nugget record must contain `qid` as a string")
    if topic_id != qid:
        raise ValueError("answer record `topic_id` must match nugget record `qid`")

    response_length = answer_record.get("response_length", 0)
    if not isinstance(response_length, int):
        raise ValueError("answer record `response_length` must be an integer")

    run_id = answer_record.get("run_id", "direct-assign")
    if not isinstance(run_id, str):
        raise ValueError("answer record `run_id` must be a string when provided")

    answer = answer_record.get("answer")
    if not isinstance(answer, list):
        raise ValueError("answer record must contain `answer` as a list")

    return {
        "query": query,
        "qid": qid,
        "context": context,
        "answer_text": context,
        "response_length": response_length,
        "run_id": run_id,
        "answer_record": answer_record,
        "nugget_record": nugget_record,
        "nuggets": scored_nuggets_from_record(nugget_record),
    }


def unwrap_direct_create_payload(payload: dict[str, Any]) -> dict[str, Any]:
    schema_version = payload.get("schema_version")
    artifacts = payload.get("artifacts")
    if schema_version != "castorini.cli.v1" or not isinstance(artifacts, list):
        return payload

    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        artifact_payload = artifact.get("data", artifact.get("value"))
        artifact_name = artifact.get("name")
        if artifact_name == "judgments":
            if not isinstance(artifact_payload, list):
                raise ValueError(
                    "direct create envelope input `judgments` artifact must contain a list"
                )
            return {"judgments": artifact_payload}
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
    payload = unwrap_direct_create_payload(payload)
    if "judgments" in payload:
        judgments = payload.get("judgments")
        if not isinstance(judgments, list):
            raise ValueError("direct create input `judgments` must be a list")
        return _normalize_umbrela_judgments(judgments)
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
                    **(
                        {"judgment": candidate["judgment"]}
                        if "judgment" in candidate
                        else {}
                    ),
                }
            )
            for index, candidate in enumerate(payload["candidates"])
        ],
    }


def direct_assign_inputs(
    payload: dict[str, Any],
) -> tuple[str, str, list[ScoredNugget]]:
    """Normalize direct assign input into the core assign call shape."""
    if all(key in payload for key in ["query", "context", "nuggets"]):
        nugget_record = {
            "nuggets": payload["nuggets"],
        }
        return (
            payload["query"],
            payload["context"],
            scored_nuggets_from_record(nugget_record),
        )

    if all(key in payload for key in ["answer_record", "nugget_record"]):
        query, context = _answer_record_to_query_context(payload["answer_record"])
        nuggets = scored_nuggets_from_record(payload["nugget_record"])
        return query, context, nuggets

    if all(key in payload for key in ["answer_envelope", "nugget_envelope"]):
        answer_record = unwrap_generation_record(payload["answer_envelope"])
        nugget_record = unwrap_nugget_record(payload["nugget_envelope"])
        query, context = _answer_record_to_query_context(answer_record)
        nuggets = scored_nuggets_from_record(nugget_record)
        return query, context, nuggets

    raise ValueError(
        "direct assign input requires either `query`/`context`/`nuggets`, "
        "`answer_record`/`nugget_record`, or "
        "`answer_envelope`/`nugget_envelope`"
    )


def joined_assign_batch_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize batch joined assign input into execution-ready records."""
    if all(key in payload for key in ["answer_records", "nugget_record"]):
        answer_records = payload["answer_records"]
        nugget_record = payload["nugget_record"]
    elif all(key in payload for key in ["answers_envelope", "nugget_envelope"]):
        answer_records = _unwrap_artifact_records(
            payload["answers_envelope"],
            artifact_name="generation-results",
            record_name="generation record",
        )
        nugget_record = unwrap_nugget_record(payload["nugget_envelope"])
    else:
        raise ValueError(
            "batch joined assign input requires either `answer_records`/"
            "`nugget_record` or `answers_envelope`/`nugget_envelope`"
        )

    if not isinstance(answer_records, list) or not answer_records:
        raise ValueError(
            "batch joined assign input requires `answer_records` as a non-empty list"
        )
    if not isinstance(nugget_record, dict):
        raise ValueError(
            "batch joined assign input requires `nugget_record` as an object"
        )

    return [
        _answer_record_to_assignment_input(answer_record, nugget_record)
        for answer_record in answer_records
    ]
