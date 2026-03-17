from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from nuggetizer.cli.main import main
from nuggetizer.core.types import AssignedScoredNugget, ScoredNugget
from nuggetizer.models.nuggetizer import Nuggetizer

pytestmark = pytest.mark.integration


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


def test_create_assign_and_metrics_pipeline_regression(
    tmp_path: Path, monkeypatch: Any
) -> None:
    pool_path = tmp_path / "pool.jsonl"
    nuggets_path = tmp_path / "nuggets.jsonl"
    answers_path = tmp_path / "answers.jsonl"
    assignments_path = tmp_path / "assignments.jsonl"
    metrics_path = tmp_path / "metrics.jsonl"

    write_jsonl(
        pool_path,
        [
            {
                "query": {"qid": "q1", "text": "What is Python used for?"},
                "candidates": [
                    {
                        "docid": "d1",
                        "doc": {
                            "segment": (
                                "Python is used for web development and data analysis."
                            )
                        },
                    }
                ],
            }
        ],
    )
    write_jsonl(
        answers_path,
        [
            {
                "run_id": "demo-run",
                "topic_id": "q1",
                "topic": "What is Python used for?",
                "response_length": 8,
                "answer": [
                    {
                        "text": "Python is used for web development and data analysis.",
                        "citations": [0],
                    }
                ],
            }
        ],
    )

    def fake_create(self: Nuggetizer, request: Any) -> list[ScoredNugget]:
        assert request.query.qid == "q1"
        return [
            ScoredNugget(
                text="Python is used for web development.",
                importance="vital",
            ),
            ScoredNugget(
                text="Python is used for data analysis.",
                importance="okay",
            ),
        ]

    def fake_assign(
        self: Nuggetizer, query: str, context: str, nuggets: list[ScoredNugget]
    ) -> list[AssignedScoredNugget]:
        del query, context
        return [
            AssignedScoredNugget(
                text=nuggets[0].text,
                importance=nuggets[0].importance,
                assignment="support",
            ),
            AssignedScoredNugget(
                text=nuggets[1].text,
                importance=nuggets[1].importance,
                assignment="partial_support",
            ),
        ]

    monkeypatch.setattr(Nuggetizer, "create", fake_create)
    monkeypatch.setattr(Nuggetizer, "assign", fake_assign)

    create_exit = main(
        [
            "create",
            "--model",
            "gpt-4o-mini",
            "--input-file",
            str(pool_path),
            "--output-file",
            str(nuggets_path),
            "--output",
            "json",
        ]
    )
    assert create_exit == 0

    nugget_records = read_jsonl(nuggets_path)
    assert nugget_records[0]["qid"] == "q1"
    assert [nugget["importance"] for nugget in nugget_records[0]["nuggets"]] == [
        "vital",
        "okay",
    ]

    assign_exit = main(
        [
            "assign",
            "--model",
            "gpt-4o-mini",
            "--input-kind",
            "answers",
            "--nuggets",
            str(nuggets_path),
            "--contexts",
            str(answers_path),
            "--output-file",
            str(assignments_path),
            "--output",
            "json",
        ]
    )
    assert assign_exit == 0

    assignment_records = read_jsonl(assignments_path)
    assert assignment_records[0]["qid"] == "q1"
    assert [nugget["assignment"] for nugget in assignment_records[0]["nuggets"]] == [
        "support",
        "partial_support",
    ]

    metrics_exit = main(
        [
            "metrics",
            "--input-file",
            str(assignments_path),
            "--output-file",
            str(metrics_path),
            "--output",
            "json",
        ]
    )
    assert metrics_exit == 0

    metric_records = read_jsonl(metrics_path)
    assert metric_records[0]["qid"] == "q1"
    assert metric_records[0]["strict_vital_score"] == 1.0
    assert metric_records[0]["strict_all_score"] == 0.5
    assert metric_records[0]["vital_score"] == 1.0
    assert metric_records[0]["all_score"] == 0.75
    assert metric_records[1]["qid"] == "all"
    assert metric_records[1]["all_score"] == 0.75
