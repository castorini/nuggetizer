from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path
from typing import Any

from nuggetizer.core.types import AssignedScoredNugget, ScoredNugget
from nuggetizer.models.nuggetizer import Nuggetizer


REPO_ROOT = Path(__file__).resolve().parents[1]


def run_script(script_name: str, args: list[str]) -> None:
    script_path = REPO_ROOT / "scripts" / script_name
    old_argv = sys.argv[:]
    try:
        sys.argv = [str(script_path), *args]
        runpy.run_path(str(script_path), run_name="__main__")
    finally:
        sys.argv = old_argv


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


def test_create_nuggets_script_skips_processed_qids(
    tmp_path: Path, monkeypatch: Any
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
                        "doc": {
                            "segment": "Python is widely used for web development."
                        },
                        "judgment": 1,
                    }
                ],
            },
            {
                "query": {"qid": "q2", "text": "Who created Python?"},
                "candidates": [
                    {
                        "docid": "d2",
                        "doc": {"segment": "Python was created by Guido van Rossum."},
                        "judgment": 1,
                    }
                ],
            },
        ],
    )
    write_jsonl(
        output_path,
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

    def fake_create(self: Nuggetizer, request: Any) -> list[ScoredNugget]:
        assert request.query.qid == "q2"
        assert request.documents[0].docid == "d2"
        return [
            ScoredNugget(text="Guido van Rossum created Python.", importance="vital")
        ]

    monkeypatch.setattr(Nuggetizer, "create", fake_create)

    run_script(
        "create_nuggets.py",
        ["--input_file", str(input_path), "--output_file", str(output_path)],
    )

    records = read_jsonl(output_path)
    assert [record["qid"] for record in records] == ["q1", "q2"]
    assert records[1]["nuggets"] == [
        {"text": "Guido van Rossum created Python.", "importance": "vital"}
    ]


def test_assign_nuggets_script_uses_missing_answer_fallback(
    tmp_path: Path, monkeypatch: Any
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
            },
            {
                "query": "Who created Python?",
                "qid": "q2",
                "nuggets": [
                    {
                        "text": "Python was created by Guido van Rossum.",
                        "importance": "vital",
                    }
                ],
            },
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
    write_jsonl(
        output_path,
        [
            {
                "query": "What is Python used for?",
                "qid": "q1",
                "answer_text": "Python is used for web development.",
                "response_length": 10,
                "run_id": "answers",
                "nuggets": [
                    {
                        "text": "Python is used for web development.",
                        "importance": "vital",
                        "assignment": "support",
                    }
                ],
            }
        ],
    )

    def fake_assign(
        self: Nuggetizer, query: str, context: str, nuggets: list[ScoredNugget]
    ) -> list[AssignedScoredNugget]:
        assert query == "Who created Python?"
        assert context == ""
        assert nuggets[0].text == "Python was created by Guido van Rossum."
        return [
            AssignedScoredNugget(
                text=nuggets[0].text,
                importance=nuggets[0].importance,
                assignment="not_support",
            )
        ]

    monkeypatch.setattr(Nuggetizer, "assign", fake_assign)

    run_script(
        "assign_nuggets.py",
        [
            "--nugget_file",
            str(nugget_path),
            "--answer_file",
            str(answer_path),
            "--output_file",
            str(output_path),
        ],
    )

    records = read_jsonl(output_path)
    assert [record["qid"] for record in records] == ["q1", "q2"]
    assert records[1]["answer_text"] == ""
    assert records[1]["response_length"] == 0
    assert records[1]["run_id"] == "answers"
    assert records[1]["nuggets"][0]["assignment"] == "not_support"


def test_assign_retrieval_script_skips_processed_entries(
    tmp_path: Path, monkeypatch: Any
) -> None:
    nugget_path = tmp_path / "nuggets.jsonl"
    retrieval_path = tmp_path / "retrieval.jsonl"
    output_path = tmp_path / "retrieval_assignments.jsonl"
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
                    },
                    {
                        "docid": "d2",
                        "doc": {"segment": "Python was created by Guido van Rossum."},
                    },
                ],
            }
        ],
    )
    write_jsonl(
        output_path,
        [
            {
                "text": "What is Python used for?",
                "qid": "q1",
                "candidate_text": "Python is used for web development.",
                "docid": "d1",
                "nuggets": [
                    {
                        "text": "Python is used for web development.",
                        "importance": "vital",
                        "assignment": "support",
                    }
                ],
            }
        ],
    )

    def fake_assign(
        self: Nuggetizer, query: str, context: str, nuggets: list[ScoredNugget]
    ) -> list[AssignedScoredNugget]:
        assert query == "What is Python used for?"
        assert context == "Python was created by Guido van Rossum."
        return [
            AssignedScoredNugget(
                text=nuggets[0].text,
                importance=nuggets[0].importance,
                assignment="not_support",
            )
        ]

    monkeypatch.setattr(Nuggetizer, "assign", fake_assign)

    run_script(
        "assign_nuggets_retrieve_results.py",
        [
            "--nugget_file",
            str(nugget_path),
            "--retrieve_results_file",
            str(retrieval_path),
            "--output_file",
            str(output_path),
        ],
    )

    records = read_jsonl(output_path)
    assert [(record["qid"], record["docid"]) for record in records] == [
        ("q1", "d1"),
        ("q1", "d2"),
    ]
    assert records[1]["candidate_text"] == "Python was created by Guido van Rossum."
    assert records[1]["nuggets"][0]["assignment"] == "not_support"


def test_calculate_metrics_script_writes_per_query_and_global_metrics(
    tmp_path: Path, capsys: Any
) -> None:
    input_path = tmp_path / "assignments.jsonl"
    output_path = tmp_path / "metrics.jsonl"
    write_jsonl(
        input_path,
        [
            {
                "qid": "q1",
                "nuggets": [
                    {"text": "a", "importance": "vital", "assignment": "support"},
                    {
                        "text": "b",
                        "importance": "okay",
                        "assignment": "partial_support",
                    },
                ],
            }
        ],
    )

    run_script(
        "calculate_metrics.py",
        ["--input_file", str(input_path), "--output_file", str(output_path)],
    )

    records = read_jsonl(output_path)
    assert records == [
        {
            "qid": "q1",
            "strict_vital_score": 1.0,
            "strict_all_score": 0.5,
            "vital_score": 1.0,
            "all_score": 0.75,
        },
        {
            "qid": "all",
            "strict_vital_score": 1.0,
            "strict_all_score": 0.5,
            "vital_score": 1.0,
            "all_score": 0.75,
        },
    ]

    stdout = capsys.readouterr().out
    assert "'qid': 'all'" in stdout
