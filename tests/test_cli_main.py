from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nuggetizer.cli.main import main
from nuggetizer.core.types import AssignedScoredNugget, ScoredNugget
from nuggetizer.models.nuggetizer import Nuggetizer


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_direct_create_via_input_json(monkeypatch: Any, capsys: Any) -> None:
    def fake_create(self: Nuggetizer, request: Any) -> list[ScoredNugget]:
        assert request.query.qid == "q0"
        assert request.documents[0].docid == "d0"
        return [
            ScoredNugget(text="Python is used for web development.", importance="vital")
        ]

    monkeypatch.setattr(Nuggetizer, "create", fake_create)

    exit_code = main(
        [
            "create",
            "--input-json",
            json.dumps(
                {
                    "query": "What is Python used for?",
                    "candidates": ["Python is widely used for web development."],
                }
            ),
            "--output",
            "json",
        ]
    )

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["schema_version"] == "castorini.cli.v1"
    assert output["command"] == "create"
    assert output["status"] == "success"
    assert output["artifacts"][0]["data"] == {
        "query": "What is Python used for?",
        "nuggets": [
            {"text": "Python is used for web development.", "importance": "vital"}
        ],
    }


def test_direct_assign_via_input_json(monkeypatch: Any, capsys: Any) -> None:
    def fake_assign(
        self: Nuggetizer, query: str, context: str, nuggets: list[ScoredNugget]
    ) -> list[AssignedScoredNugget]:
        assert query == "What is Python used for?"
        assert context == "Python is commonly used for web development."
        assert nuggets[0].text == "Python is used for web development."
        return [
            AssignedScoredNugget(
                text=nuggets[0].text,
                importance=nuggets[0].importance,
                assignment="support",
            )
        ]

    monkeypatch.setattr(Nuggetizer, "assign", fake_assign)

    exit_code = main(
        [
            "assign",
            "--input-json",
            json.dumps(
                {
                    "query": "What is Python used for?",
                    "context": "Python is commonly used for web development.",
                    "nuggets": [
                        {
                            "text": "Python is used for web development.",
                            "importance": "vital",
                        }
                    ],
                }
            ),
            "--output",
            "json",
        ]
    )

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["schema_version"] == "castorini.cli.v1"
    assert output["command"] == "assign"
    assert output["status"] == "success"
    assert output["artifacts"][0]["data"] == {
        "query": "What is Python used for?",
        "nuggets": [
            {
                "text": "Python is used for web development.",
                "importance": "vital",
                "assignment": "support",
            }
        ],
    }


def test_batch_assign_retrieval_alias_uses_same_flow(
    tmp_path: Path, monkeypatch: Any
) -> None:
    nugget_path = tmp_path / "nuggets.jsonl"
    retrieval_path = tmp_path / "retrieval.jsonl"
    output_path = tmp_path / "assignments.jsonl"
    write_jsonl(
        nugget_path,
        [
            {
                "query": "What is Python used for?",
                "qid": "q1",
                "nuggets": [
                    {
                        "text": "Python is used for web development.",
                        "importance": "vital",
                    }
                ],
            }
        ],
    )
    write_jsonl(
        retrieval_path,
        [
            {
                "query": {"qid": "q1", "text": "What is Python used for?"},
                "candidates": [
                    {
                        "docid": "d1",
                        "doc": {"segment": "Python is used for web development."},
                    }
                ],
            }
        ],
    )

    def fake_assign(
        self: Nuggetizer, query: str, context: str, nuggets: list[ScoredNugget]
    ) -> list[AssignedScoredNugget]:
        return [
            AssignedScoredNugget(
                text=nuggets[0].text,
                importance=nuggets[0].importance,
                assignment="support",
            )
        ]

    monkeypatch.setattr(Nuggetizer, "assign", fake_assign)

    exit_code = main(
        [
            "assign-retrieval",
            "--nuggets",
            str(nugget_path),
            "--contexts",
            str(retrieval_path),
            "--output-file",
            str(output_path),
        ]
    )

    assert exit_code == 0
    records = read_jsonl(output_path)
    assert records[0]["docid"] == "d1"
    assert records[0]["nuggets"][0]["assignment"] == "support"


def test_batch_create_missing_input_returns_json_error(capsys: Any) -> None:
    exit_code = main(
        [
            "create",
            "--input-file",
            "/tmp/does-not-exist.jsonl",
            "--output-file",
            "/tmp/out.jsonl",
            "--output",
            "json",
        ]
    )

    assert exit_code == 4
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "validation_error"
    assert output["exit_code"] == 4
    assert output["errors"][0]["code"] == "missing_input"


def test_describe_assign_returns_json_envelope(capsys: Any) -> None:
    exit_code = main(["describe", "assign", "--output", "json"])

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["command"] == "describe"
    assert output["artifacts"][0]["data"]["batch_input_kinds"] == [
        "answers",
        "retrieval",
    ]


def test_schema_assign_output_answers_returns_json_envelope(capsys: Any) -> None:
    exit_code = main(["schema", "assign-output-answers", "--output", "json"])

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["command"] == "schema"
    assert "run_id" in output["artifacts"][0]["data"]["required"]


def test_doctor_returns_json_envelope(capsys: Any) -> None:
    exit_code = main(["doctor", "--output", "json"])

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["command"] == "doctor"
    assert "python_version" in output["metrics"]


def test_validate_create_batch_returns_json_envelope(
    tmp_path: Path, capsys: Any
) -> None:
    input_path = tmp_path / "pool.jsonl"
    write_jsonl(
        input_path,
        [
            {
                "query": {"qid": "q1", "text": "What is Python used for?"},
                "candidates": [
                    {
                        "docid": "d1",
                        "doc": {"segment": "Python is used for web development."},
                    }
                ],
            }
        ],
    )

    exit_code = main(
        ["validate", "create", "--input-file", str(input_path), "--output", "json"]
    )

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["command"] == "validate"
    assert output["validation"]["valid"] is True
    assert output["validation"]["record_count"] == 1


def test_direct_create_validate_only_does_not_call_llm(
    monkeypatch: Any, capsys: Any
) -> None:
    def fail_create(self: Nuggetizer, request: Any) -> list[ScoredNugget]:
        raise AssertionError("LLM should not be called during validate-only")

    monkeypatch.setattr(Nuggetizer, "create", fail_create)

    exit_code = main(
        [
            "create",
            "--input-json",
            json.dumps(
                {
                    "query": "What is Python used for?",
                    "candidates": ["Python is used for web development."],
                }
            ),
            "--validate-only",
            "--output",
            "json",
        ]
    )

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["mode"] == "validate"
    assert output["validation"]["valid"] is True


def test_batch_create_dry_run_reports_write_policy_conflict(
    tmp_path: Path, capsys: Any
) -> None:
    input_path = tmp_path / "pool.jsonl"
    output_path = tmp_path / "nuggets.jsonl"
    write_jsonl(
        input_path,
        [
            {
                "query": {"qid": "q1", "text": "What is Python used for?"},
                "candidates": [
                    {
                        "docid": "d1",
                        "doc": {"segment": "Python is used for web development."},
                    }
                ],
            }
        ],
    )
    output_path.write_text("existing\n", encoding="utf-8")

    exit_code = main(
        [
            "create",
            "--input-file",
            str(input_path),
            "--output-file",
            str(output_path),
            "--dry-run",
            "--output",
            "json",
        ]
    )

    assert exit_code == 5
    output = json.loads(capsys.readouterr().out)
    assert output["errors"][0]["code"] == "write_policy_conflict"


def test_batch_assign_dry_run_returns_counts_without_writing(
    tmp_path: Path, capsys: Any
) -> None:
    nugget_path = tmp_path / "nuggets.jsonl"
    answer_path = tmp_path / "answers.jsonl"
    output_path = tmp_path / "assignments.jsonl"
    write_jsonl(
        nugget_path,
        [
            {
                "query": "What is Python used for?",
                "qid": "q1",
                "nuggets": [
                    {
                        "text": "Python is used for web development.",
                        "importance": "vital",
                    }
                ],
            }
        ],
    )
    write_jsonl(
        answer_path,
        [
            {
                "run_id": "demo-run",
                "topic_id": "q1",
                "topic": "What is Python used for?",
                "response_length": 10,
                "answer": [
                    {"text": "Python is used for web development.", "citations": [0]}
                ],
            }
        ],
    )

    exit_code = main(
        [
            "assign",
            "--input-kind",
            "answers",
            "--nuggets",
            str(nugget_path),
            "--contexts",
            str(answer_path),
            "--output-file",
            str(output_path),
            "--dry-run",
            "--output",
            "json",
        ]
    )

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["mode"] == "dry-run"
    assert output["validation"]["nugget_record_count"] == 1
    assert not output_path.exists()
