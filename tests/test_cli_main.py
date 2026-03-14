from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from nuggetizer.cli.main import main
from nuggetizer.core.types import AssignedScoredNugget, ScoredNugget, Trace
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
    assert output["artifacts"][0]["kind"] == "data"
    assert output["artifacts"][0]["name"] == "create-result"
    assert output["artifacts"][0]["data"] == {
        "query": "What is Python used for?",
        "nuggets": [
            {"text": "Python is used for web development.", "importance": "vital"}
        ],
    }


def test_direct_create_forwards_openrouter_and_reasoning_effort(
    monkeypatch: Any, capsys: Any
) -> None:
    captured_kwargs: dict[str, Any] = {}

    def fake_init(self: Nuggetizer, *args: Any, **kwargs: Any) -> None:
        del args
        captured_kwargs.update(kwargs)

    def fake_create(self: Nuggetizer, request: Any) -> list[ScoredNugget]:
        del request
        return [ScoredNugget(text="router", importance="vital")]

    monkeypatch.setattr(Nuggetizer, "__init__", fake_init)
    monkeypatch.setattr(Nuggetizer, "create", fake_create)

    exit_code = main(
        [
            "create",
            "--input-json",
            json.dumps({"query": "q", "candidates": ["c"]}),
            "--use-openrouter",
            "--reasoning-effort",
            "minimal",
            "--output",
            "json",
        ]
    )

    assert exit_code == 0
    json.loads(capsys.readouterr().out)
    assert captured_kwargs["use_openrouter"] is True
    assert captured_kwargs["reasoning_effort"] == "minimal"


def test_direct_create_text_output_prints_reasoning(
    monkeypatch: Any, capsys: Any
) -> None:
    def fake_create(self: Nuggetizer, request: Any) -> list[ScoredNugget]:
        del request
        self.creator_reasoning_traces = [
            "Creator window 1 trace.",
            "Creator window 2 trace.",
        ]
        return [
            ScoredNugget(
                text="Python is used for web development.",
                importance="vital",
                reasoning="Scored as vital because it directly answers the query.",
            )
        ]

    monkeypatch.setattr(Nuggetizer, "create", fake_create)

    exit_code = main(
        [
            "create",
            "--input-json",
            json.dumps({"query": "What is Python used for?", "candidates": ["c"]}),
            "--include-reasoning",
        ]
    )

    assert exit_code == 0
    assert capsys.readouterr().out == (
        "query: What is Python used for?\n"
        "nuggets:\n"
        "vital: Python is used for web development.\n"
        "\n"
        "creator reasoning trace 1: Creator window 1 trace.\n"
        "creator reasoning trace 2: Creator window 2 trace.\n"
        "\n"
        "scoring reasoning trace 1: Scored as vital because it directly answers the query.\n"
    )


def test_direct_create_json_output_aggregates_unique_reasoning_traces(
    monkeypatch: Any, capsys: Any
) -> None:
    def fake_create(self: Nuggetizer, request: Any) -> list[ScoredNugget]:
        del request
        self.creator_reasoning_traces = ["creator trace 1", "creator trace 2"]
        return [
            ScoredNugget(text="A", importance="vital", reasoning="same trace"),
            ScoredNugget(text="B", importance="okay", reasoning="same trace"),
            ScoredNugget(text="C", importance="okay", reasoning="different trace"),
        ]

    monkeypatch.setattr(Nuggetizer, "create", fake_create)

    exit_code = main(
        [
            "create",
            "--input-json",
            json.dumps({"query": "What is Python used for?", "candidates": ["c"]}),
            "--include-reasoning",
            "--output",
            "json",
        ]
    )

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["artifacts"][0]["data"]["creator_reasoning_traces"] == [
        "creator trace 1",
        "creator trace 2",
    ]
    assert output["artifacts"][0]["data"]["scoring_reasoning_traces"] == [
        "same trace",
        "different trace",
    ]


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


def test_direct_assign_text_output_prints_unique_reasoning_traces(
    monkeypatch: Any, capsys: Any
) -> None:
    def fake_assign(
        self: Nuggetizer, query: str, context: str, nuggets: list[ScoredNugget]
    ) -> list[AssignedScoredNugget]:
        del query, context, nuggets
        return [
            AssignedScoredNugget(
                text="A",
                importance="vital",
                assignment="support",
                reasoning="same trace",
            ),
            AssignedScoredNugget(
                text="B",
                importance="okay",
                assignment="partial_support",
                reasoning="same trace",
            ),
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
                    "nuggets": [{"text": "n", "importance": "vital"}],
                }
            ),
            "--include-reasoning",
        ]
    )

    assert exit_code == 0
    assert capsys.readouterr().out == (
        "support: vital A\n"
        "partial_support: okay B\n"
        "\n"
        "scoring reasoning trace 1: same trace\n"
    )


def test_direct_assign_forwards_openrouter_and_reasoning_effort(
    monkeypatch: Any, capsys: Any
) -> None:
    captured_kwargs: dict[str, Any] = {}

    def fake_init(self: Nuggetizer, *args: Any, **kwargs: Any) -> None:
        del args
        captured_kwargs.update(kwargs)

    def fake_assign(
        self: Nuggetizer, query: str, context: str, nuggets: list[ScoredNugget]
    ) -> list[AssignedScoredNugget]:
        del query, context
        return [
            AssignedScoredNugget(
                text=nuggets[0].text,
                importance=nuggets[0].importance,
                assignment="support",
            )
        ]

    monkeypatch.setattr(Nuggetizer, "__init__", fake_init)
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
            "--use-openrouter",
            "--reasoning-effort",
            "xhigh",
            "--output",
            "json",
        ]
    )

    assert exit_code == 0
    json.loads(capsys.readouterr().out)
    assert captured_kwargs["use_openrouter"] is True
    assert captured_kwargs["reasoning_effort"] == "xhigh"


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


def test_missing_command_returns_descriptive_text_error(capsys: Any) -> None:
    exit_code = main([])

    assert exit_code == 2
    captured = capsys.readouterr()
    assert "No command provided." in captured.err
    assert (
        "create, assign, assign-retrieval, metrics, view, describe, schema, doctor, validate"
        in captured.err
    )
    assert (
        "nuggetizer create --input-file pool.jsonl --output-file nuggets.jsonl"
        in captured.err
    )
    assert "Run `nuggetizer --help` for full usage." in captured.err


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
    assert "backend_readiness" in output["metrics"]
    assert "command_readiness" in output["metrics"]


def test_top_level_help_includes_command_summaries(capsys: Any) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])

    assert exc_info.value.code == 0
    stdout = capsys.readouterr().out
    assert "Nuggetizer packaged CLI" in stdout
    assert "create and score nuggets" in stdout.lower()
    assert "inspect an existing nuggetizer artifact" in stdout.lower()


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


def test_direct_create_defaults_to_sync_execution(
    monkeypatch: Any, capsys: Any
) -> None:
    def fake_create(self: Nuggetizer, request: Any) -> list[ScoredNugget]:
        return [ScoredNugget(text="sync", importance="vital")]

    async def fail_async_create(self: Nuggetizer, request: Any) -> list[ScoredNugget]:
        raise AssertionError("async_create should not run by default")

    monkeypatch.setattr(Nuggetizer, "create", fake_create)
    monkeypatch.setattr(Nuggetizer, "async_create", fail_async_create)

    exit_code = main(
        [
            "create",
            "--input-json",
            json.dumps({"query": "q", "candidates": ["c"]}),
            "--output",
            "json",
        ]
    )

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["resolved"]["execution_mode"] == "sync"
    assert output["artifacts"][0]["data"]["nuggets"][0]["text"] == "sync"


def test_direct_create_async_execution_is_opt_in(monkeypatch: Any, capsys: Any) -> None:
    def fail_create(self: Nuggetizer, request: Any) -> list[ScoredNugget]:
        raise AssertionError("create should not run in async mode")

    async def fake_async_create(self: Nuggetizer, request: Any) -> list[ScoredNugget]:
        return [ScoredNugget(text="async", importance="vital")]

    monkeypatch.setattr(Nuggetizer, "create", fail_create)
    monkeypatch.setattr(Nuggetizer, "async_create", fake_async_create)

    exit_code = main(
        [
            "create",
            "--input-json",
            json.dumps({"query": "q", "candidates": ["c"]}),
            "--execution-mode",
            "async",
            "--output",
            "json",
        ]
    )

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["resolved"]["execution_mode"] == "async"
    assert output["artifacts"][0]["data"]["nuggets"][0]["text"] == "async"


def test_direct_create_trace_and_reasoning_are_opt_in(
    monkeypatch: Any, capsys: Any
) -> None:
    def fake_create(self: Nuggetizer, request: Any) -> list[ScoredNugget]:
        return [
            ScoredNugget(
                text="Python is used for web development.",
                importance="vital",
                reasoning="Model explanation",
                trace=Trace(
                    component="creator",
                    model="gpt-4o",
                    params={"temperature": 0.0},
                    messages=[{"role": "user", "content": "prompt"}],
                    usage={"total_tokens": 10},
                    raw_output="raw",
                ),
            )
        ]

    monkeypatch.setattr(Nuggetizer, "create", fake_create)

    exit_code = main(
        [
            "create",
            "--input-json",
            json.dumps({"query": "q", "candidates": ["c"]}),
            "--include-trace",
            "--include-reasoning",
            "--redact-prompts",
            "--output",
            "json",
        ]
    )

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    nugget = output["artifacts"][0]["data"]["nuggets"][0]
    assert nugget["reasoning"] == "Model explanation"
    assert nugget["trace"]["messages"] is None


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


def test_view_create_output_returns_json_summary(tmp_path: Path, capsys: Any) -> None:
    path = tmp_path / "nuggets.jsonl"
    write_jsonl(
        path,
        [
            {
                "query": "What is Python used for? " * 10,
                "qid": "q1",
                "nuggets": [
                    {
                        "text": "Python is used for web development.",
                        "importance": "vital",
                    },
                    {"text": "Python is used for data analysis.", "importance": "okay"},
                ],
            }
        ],
    )

    exit_code = main(["view", str(path), "--records", "1", "--output", "json"])

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["command"] == "view"
    assert output["artifacts"][0]["data"]["artifact_type"] == "create-output"
    assert output["artifacts"][0]["data"]["summary"]["total_nuggets"] == 2
    assert (
        output["artifacts"][0]["data"]["sampled_records"][0]["query"]
        == ("What is Python used for? " * 10).strip()
    )


def test_view_create_output_honors_nugget_limit(tmp_path: Path, capsys: Any) -> None:
    path = tmp_path / "nuggets.jsonl"
    write_jsonl(
        path,
        [
            {
                "query": "What is Python used for?",
                "qid": "q1",
                "nuggets": [
                    {
                        "text": "Python is used for web development.",
                        "importance": "vital",
                    },
                    {"text": "Python is used for data analysis.", "importance": "okay"},
                ],
            }
        ],
    )

    exit_code = main(
        ["view", str(path), "--records", "1", "--nugget-limit", "1", "--output", "json"]
    )

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert len(output["artifacts"][0]["data"]["sampled_records"][0]["nuggets"]) == 1


def test_view_assign_answers_text_renders_assignments(
    tmp_path: Path, capsys: Any
) -> None:
    path = tmp_path / "assignments.jsonl"
    write_jsonl(
        path,
        [
            {
                "query": "What is Python used for?",
                "qid": "q1",
                "answer_text": "Python is used for web development and data analysis.",
                "response_length": 12,
                "run_id": "demo-run",
                "nuggets": [
                    {
                        "text": "Python is used for web development.",
                        "importance": "vital",
                        "assignment": "support",
                    },
                    {
                        "text": "Python is used for data analysis.",
                        "importance": "okay",
                        "assignment": "partial_support",
                    },
                ],
            }
        ],
    )

    exit_code = main(["view", str(path), "--color", "never"])

    assert exit_code == 0
    stdout = capsys.readouterr().out
    assert "Nuggetizer View" in stdout
    assert "run_ids: demo-run" in stdout
    assert "assignments: support=1, partial_support=1, not_support=0" in stdout


def test_view_assign_answers_text_color_codes_importance_and_assignment(
    tmp_path: Path, capsys: Any
) -> None:
    path = tmp_path / "assignments-color.jsonl"
    write_jsonl(
        path,
        [
            {
                "query": "What is Python used for?",
                "qid": "q1",
                "answer_text": "Python is used for web development, data analysis, and automation.",
                "response_length": 12,
                "run_id": "demo-run",
                "nuggets": [
                    {
                        "text": "Python is used for web development.",
                        "importance": "vital",
                        "assignment": "support",
                    },
                    {
                        "text": "Python is used for data analysis.",
                        "importance": "okay",
                        "assignment": "partial_support",
                    },
                    {
                        "text": "Python is only used in browsers.",
                        "importance": "okay",
                        "assignment": "not_support",
                    },
                ],
            }
        ],
    )

    exit_code = main(["view", str(path), "--color", "always"])

    assert exit_code == 0
    stdout = capsys.readouterr().out
    assert "\033[32mvital\033[0m/\033[32msupport\033[0m" in stdout
    assert "\033[33mokay\033[0m/\033[33mpartial_support\033[0m" in stdout
    assert "\033[31mnot_support\033[0m=1" in stdout


def test_view_metrics_output_reports_global_metrics(
    tmp_path: Path, capsys: Any
) -> None:
    path = tmp_path / "metrics.jsonl"
    write_jsonl(
        path,
        [
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
        ],
    )

    exit_code = main(["view", str(path), "--output", "json"])

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["artifacts"][0]["data"]["artifact_type"] == "metrics-output"
    assert output["artifacts"][0]["data"]["summary"]["record_count"] == 1
    assert output["artifacts"][0]["data"]["summary"]["has_global_metrics"] is True
    assert output["artifacts"][0]["data"]["summary"]["global_metrics"]["qid"] == "all"


def test_view_empty_file_returns_json_error(tmp_path: Path, capsys: Any) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")

    exit_code = main(["view", str(path), "--output", "json"])

    assert exit_code == 5
    output = json.loads(capsys.readouterr().out)
    assert output["command"] == "view"
    assert output["errors"][0]["code"] == "invalid_view_input"


def test_view_malformed_file_returns_json_error(tmp_path: Path, capsys: Any) -> None:
    path = tmp_path / "broken.jsonl"
    path.write_text("{not-json}\n", encoding="utf-8")

    exit_code = main(["view", str(path), "--output", "json"])

    assert exit_code == 5
    output = json.loads(capsys.readouterr().out)
    assert output["command"] == "view"
    assert output["errors"][0]["code"] == "invalid_view_input"
