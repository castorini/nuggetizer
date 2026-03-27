from __future__ import annotations

import pytest

from nuggetizer.cli.normalize import (
    unwrap_generation_record,
    unwrap_nugget_record,
)

pytestmark = pytest.mark.core


def test_unwrap_generation_record_accepts_single_record() -> None:
    payload = {
        "schema_version": "castorini.cli.v1",
        "repo": "ragnarok",
        "command": "generate",
        "artifacts": [
            {
                "name": "generation-results",
                "kind": "data",
                "data": [
                    {
                        "run_id": "ragnarok",
                        "topic_id": "q1",
                        "topic": "What is Python used for?",
                        "references": [],
                        "response_length": 4,
                        "answer": [{"text": "Python is used for web development."}],
                    }
                ],
            }
        ],
    }

    record = unwrap_generation_record(payload)

    assert record["topic_id"] == "q1"
    assert record["topic"] == "What is Python used for?"


def test_unwrap_generation_record_rejects_multiple_records() -> None:
    payload = {
        "schema_version": "castorini.cli.v1",
        "artifacts": [
            {
                "name": "generation-results",
                "kind": "data",
                "data": [{"topic_id": "q1"}, {"topic_id": "q2"}],
            }
        ],
    }

    with pytest.raises(ValueError, match="exactly one record"):
        unwrap_generation_record(payload)


def test_unwrap_generation_record_rejects_missing_artifact() -> None:
    payload = {
        "schema_version": "castorini.cli.v1",
        "artifacts": [{"name": "other", "kind": "data", "data": []}],
    }

    with pytest.raises(ValueError, match="generation-results"):
        unwrap_generation_record(payload)


def test_unwrap_nugget_record_accepts_single_data_record() -> None:
    payload = {
        "schema_version": "castorini.cli.v1",
        "repo": "nuggetizer",
        "command": "create",
        "artifacts": [
            {
                "name": "create-result",
                "kind": "data",
                "data": {
                    "query": "What is Python used for?",
                    "qid": "q1",
                    "nuggets": [{"text": "Python is used for web development."}],
                },
            }
        ],
    }

    record = unwrap_nugget_record(payload)

    assert record["qid"] == "q1"
    assert record["nuggets"][0]["text"] == "Python is used for web development."


def test_unwrap_nugget_record_rejects_non_envelope_payload() -> None:
    payload = {"query": "What is Python used for?", "nuggets": []}

    with pytest.raises(ValueError, match="castorini.cli.v1 envelope"):
        unwrap_nugget_record(payload)
