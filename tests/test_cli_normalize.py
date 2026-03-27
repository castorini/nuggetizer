from __future__ import annotations

import pytest

from nuggetizer.cli.normalize import (
    direct_assign_inputs,
    joined_assign_batch_records,
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


def test_direct_assign_inputs_accepts_single_joined_records() -> None:
    query, context, nuggets = direct_assign_inputs(
        {
            "answer_record": {
                "topic_id": "q1",
                "topic": "What is Python used for?",
                "answer": [
                    {"text": "Python is used for web development."},
                    {"text": "It is also used for data analysis."},
                ],
            },
            "nugget_record": {
                "query": "What is Python used for?",
                "qid": "q1",
                "nuggets": [
                    {
                        "text": "Python is used for web development.",
                        "importance": "vital",
                    }
                ],
            },
        }
    )

    assert query == "What is Python used for?"
    assert (
        context
        == "Python is used for web development. It is also used for data analysis."
    )
    assert nuggets[0].text == "Python is used for web development."


def test_direct_assign_inputs_accepts_single_joined_envelopes() -> None:
    query, context, nuggets = direct_assign_inputs(
        {
            "answer_envelope": {
                "schema_version": "castorini.cli.v1",
                "artifacts": [
                    {
                        "name": "generation-results",
                        "kind": "data",
                        "data": [
                            {
                                "topic_id": "q1",
                                "topic": "What is Python used for?",
                                "answer": [
                                    {"text": "Python is used for web development."}
                                ],
                            }
                        ],
                    }
                ],
            },
            "nugget_envelope": {
                "schema_version": "castorini.cli.v1",
                "artifacts": [
                    {
                        "name": "create-result",
                        "kind": "data",
                        "data": {
                            "query": "What is Python used for?",
                            "qid": "q1",
                            "nuggets": [
                                {
                                    "text": "Python is used for web development.",
                                    "importance": "vital",
                                }
                            ],
                        },
                    }
                ],
            },
        }
    )

    assert query == "What is Python used for?"
    assert context == "Python is used for web development."
    assert nuggets[0].text == "Python is used for web development."


def test_direct_assign_inputs_rejects_malformed_joined_envelope() -> None:
    with pytest.raises(ValueError, match="generation-results"):
        direct_assign_inputs(
            {
                "answer_envelope": {
                    "schema_version": "castorini.cli.v1",
                    "artifacts": [],
                },
                "nugget_envelope": {
                    "schema_version": "castorini.cli.v1",
                    "artifacts": [],
                },
            }
        )


def test_joined_assign_batch_records_accepts_joined_records() -> None:
    records = joined_assign_batch_records(
        {
            "answer_records": [
                {
                    "run_id": "demo-run",
                    "topic_id": "q1",
                    "topic": "What is Python used for?",
                    "response_length": 10,
                    "answer": [
                        {"text": "Python is used for web development."},
                        {"text": "It is also used for data analysis."},
                    ],
                },
                {
                    "topic_id": "q1",
                    "topic": "What is Python used for?",
                    "response_length": 8,
                    "answer": [{"text": "Python is also used for automation."}],
                },
            ],
            "nugget_record": {
                "query": "What is Python used for?",
                "qid": "q1",
                "nuggets": [
                    {
                        "text": "Python is used for web development.",
                        "importance": "vital",
                    }
                ],
            },
        }
    )

    assert len(records) == 2
    assert records[0]["query"] == "What is Python used for?"
    assert (
        records[0]["answer_text"]
        == "Python is used for web development. It is also used for data analysis."
    )
    assert records[0]["response_length"] == 10
    assert records[0]["run_id"] == "demo-run"
    assert records[1]["run_id"] == "direct-assign"
    assert records[1]["qid"] == "q1"
    assert records[1]["nuggets"][0].text == "Python is used for web development."


def test_joined_assign_batch_records_accepts_joined_envelopes() -> None:
    records = joined_assign_batch_records(
        {
            "answers_envelope": {
                "schema_version": "castorini.cli.v1",
                "artifacts": [
                    {
                        "name": "generation-results",
                        "kind": "data",
                        "data": [
                            {
                                "topic_id": "q1",
                                "topic": "What is Python used for?",
                                "response_length": 10,
                                "answer": [
                                    {"text": "Python is used for web development."}
                                ],
                            },
                            {
                                "run_id": "demo-run",
                                "topic_id": "q1",
                                "topic": "What is Python used for?",
                                "response_length": 6,
                                "answer": [
                                    {"text": "Python is also used for testing."}
                                ],
                            },
                        ],
                    }
                ],
            },
            "nugget_envelope": {
                "schema_version": "castorini.cli.v1",
                "artifacts": [
                    {
                        "name": "create-result",
                        "kind": "data",
                        "data": {
                            "query": "What is Python used for?",
                            "qid": "q1",
                            "nuggets": [
                                {
                                    "text": "Python is used for web development.",
                                    "importance": "vital",
                                }
                            ],
                        },
                    }
                ],
            },
        }
    )

    assert len(records) == 2
    assert records[0]["qid"] == "q1"
    assert records[0]["run_id"] == "direct-assign"
    assert records[1]["run_id"] == "demo-run"


def test_joined_assign_batch_records_rejects_mismatched_ids() -> None:
    with pytest.raises(ValueError, match="topic_id` must match nugget record `qid"):
        joined_assign_batch_records(
            {
                "answer_records": [
                    {
                        "topic_id": "q2",
                        "topic": "What is Python used for?",
                        "response_length": 10,
                        "answer": [{"text": "Python is used for web development."}],
                    }
                ],
                "nugget_record": {
                    "query": "What is Python used for?",
                    "qid": "q1",
                    "nuggets": [{"text": "Python is used for web development."}],
                },
            }
        )


def test_joined_assign_batch_records_rejects_empty_answer_records() -> None:
    with pytest.raises(ValueError, match="non-empty list"):
        joined_assign_batch_records(
            {
                "answer_records": [],
                "nugget_record": {
                    "query": "What is Python used for?",
                    "qid": "q1",
                    "nuggets": [{"text": "Python is used for web development."}],
                },
            }
        )
