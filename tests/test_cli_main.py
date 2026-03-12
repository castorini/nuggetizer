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


def test_direct_create_via_input_json(
    monkeypatch: Any, capsys: Any
) -> None:
    def fake_create(self: Nuggetizer, request: Any) -> list[ScoredNugget]:
        assert request.query.qid == "q0"
        assert request.documents[0].docid == "d0"
        return [ScoredNugget(text="Python is used for web development.", importance="vital")]

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
    assert output == {
        "query": "What is Python used for?",
        "nuggets": [
            {"text": "Python is used for web development.", "importance": "vital"}
        ],
    }


def test_direct_assign_via_input_json(
    monkeypatch: Any, capsys: Any
) -> None:
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
    assert output == {
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
