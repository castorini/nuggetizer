from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from nuggetizer.cli.main import main
from nuggetizer.core.types import AssignedScoredNugget, ScoredNugget, Trace
from nuggetizer.models.nuggetizer import Nuggetizer

pytestmark = pytest.mark.core


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


def test_batch_json_output_suppresses_progress_bar(
    tmp_path: Path, monkeypatch: Any, capsys: Any
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

    def fake_create(self: Nuggetizer, request: Any) -> list[ScoredNugget]:
        return [ScoredNugget(text="nugget", importance="vital")]

    monkeypatch.setattr(Nuggetizer, "create", fake_create)

    exit_code = main(
        [
            "create",
            "--input-file",
            str(input_path),
            "--output-file",
            str(output_path),
            "--output",
            "json",
        ]
    )

    assert exit_code == 0
    assert capsys.readouterr().err == ""


def test_nuggetizer_init_does_not_require_api_key_until_runtime(
    tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_API_BASE", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_API_VERSION", raising=False)

    nuggetizer = Nuggetizer()

    assert nuggetizer.creator_llm is None
    assert nuggetizer.scorer_llm is None
    assert nuggetizer.assigner_llm is None


def test_quiet_flag_suppresses_stderr(capsys: Any) -> None:
    exit_code = main(["--quiet", "doctor", "--output", "json"])

    assert exit_code == 0
    assert capsys.readouterr().err == ""


def test_no_color_env_suppresses_ansi_codes(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    monkeypatch.setenv("NO_COLOR", "")
    path = tmp_path / "assignments.jsonl"
    write_jsonl(
        path,
        [
            {
                "query": "What is Python?",
                "qid": "q1",
                "answer_text": "Python is used for web development.",
                "response_length": 12,
                "run_id": "demo-run",
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

    exit_code = main(["view", str(path), "--color", "always"])

    assert exit_code == 0
    stdout = capsys.readouterr().out
    assert "\033[" not in stdout


def test_print_completion_outputs_bash_script(capsys: Any) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--print-completion", "bash"])

    assert exc_info.value.code == 0
    stdout = capsys.readouterr().out
    assert "complete" in stdout.lower() or "_nuggetizer" in stdout


def test_version_flag_prints_version_and_exits(capsys: Any) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])

    assert exc_info.value.code == 0
    stdout = capsys.readouterr().out
    assert "nuggetizer" in stdout


def test_prompt_list_returns_json_catalog(capsys: Any) -> None:
    exit_code = main(["prompt", "list", "--output", "json"])

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["command"] == "prompt"
    assert output["artifacts"][0]["name"] == "prompt-catalog"
    catalog = output["artifacts"][0]["data"]
    assert len(catalog) == 4
    assert {entry["target"] for entry in catalog} == {"create", "assign", "score"}


def test_prompt_show_create_returns_text_template(capsys: Any) -> None:
    exit_code = main(["prompt", "show", "create"])

    assert exit_code == 0
    stdout = capsys.readouterr().out
    assert "Nuggetizer Prompt Template" in stdout
    assert "target: create" in stdout
    assert "template_name: creator_template" in stdout
    assert "[user]" in stdout
    assert "Search Query: {query}" in stdout


def test_prompt_show_assign_support_grade_2_returns_json(capsys: Any) -> None:
    exit_code = main(
        [
            "prompt",
            "show",
            "assign",
            "--assign-mode",
            "support_grade_2",
            "--output",
            "json",
        ]
    )

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["command"] == "prompt"
    view = output["artifacts"][0]["data"]
    assert view["target"] == "assign"
    assert view["assign_mode"] == "support_grade_2"
    assert view["template_name"] == "assigner_2grade_template"


def test_prompt_render_create_returns_text_prompt(capsys: Any) -> None:
    exit_code = main(
        [
            "prompt",
            "render",
            "create",
            "--input-json",
            json.dumps({"query": "What is Python used for?", "candidates": ["web"]}),
            "--part",
            "user",
        ]
    )

    assert exit_code == 0
    stdout = capsys.readouterr().out
    assert "Nuggetizer Rendered Prompt" in stdout
    assert "target: create" in stdout
    assert "[user]" in stdout
    assert "[system]" not in stdout
    assert "Search Query: What is Python used for?" in stdout


def test_prompt_render_assign_returns_json_prompt(capsys: Any) -> None:
    exit_code = main(
        [
            "prompt",
            "render",
            "assign",
            "--assign-mode",
            "support_grade_2",
            "--input-json",
            json.dumps(
                {
                    "query": "What is Python used for?",
                    "context": "Python is used for web development.",
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
    view = output["artifacts"][0]["data"]
    assert view["target"] == "assign"
    assert view["assign_mode"] == "support_grade_2"
    assert view["messages"][0]["role"] == "system"
    assert "support or not_support" in view["messages"][1]["content"]


def test_prompt_render_score_returns_json_prompt(capsys: Any) -> None:
    exit_code = main(
        [
            "prompt",
            "render",
            "score",
            "--input-json",
            json.dumps(
                {
                    "query": "What is Python used for?",
                    "nuggets": ["Python is used for web development."],
                }
            ),
            "--output",
            "json",
        ]
    )

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    view = output["artifacts"][0]["data"]
    assert view["target"] == "score"
    assert "label each of the 1 nuggets" in view["messages"][1]["content"]


def test_prompt_render_score_requires_query_and_nuggets(capsys: Any) -> None:
    exit_code = main(
        [
            "prompt",
            "render",
            "score",
            "--input-json",
            json.dumps({"query": "q"}),
            "--output",
            "json",
        ]
    )

    assert exit_code == 2
    output = json.loads(capsys.readouterr().out)
    assert output["command"] == "prompt"
    assert output["errors"][0]["code"] == "invalid_score_prompt_input"


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


def test_direct_create_filters_judgments_by_default(
    monkeypatch: Any, capsys: Any
) -> None:
    def fake_create(self: Nuggetizer, request: Any) -> list[ScoredNugget]:
        del self
        assert [document.docid for document in request.documents] == ["d1", "d2"]
        return [ScoredNugget(text="filtered", importance="vital")]

    monkeypatch.setattr(Nuggetizer, "create", fake_create)

    exit_code = main(
        [
            "create",
            "--input-json",
            json.dumps(
                {
                    "query": "What is Python used for?",
                    "candidates": [
                        {"docid": "d0", "doc": {"segment": "bad"}, "judgment": 1},
                        {"docid": "d1", "doc": {"segment": "good"}, "judgment": 2},
                        {"docid": "d2", "doc": {"segment": "best"}, "judgment": 3},
                    ],
                }
            ),
            "--output",
            "json",
        ]
    )

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["resolved"]["min_judgment"] == 2


def test_direct_create_accepts_min_judgment_override(
    monkeypatch: Any, capsys: Any
) -> None:
    def fake_create(self: Nuggetizer, request: Any) -> list[ScoredNugget]:
        del self
        assert [document.docid for document in request.documents] == ["d0", "d1"]
        return [ScoredNugget(text="filtered", importance="vital")]

    monkeypatch.setattr(Nuggetizer, "create", fake_create)

    exit_code = main(
        [
            "create",
            "--input-json",
            json.dumps(
                {
                    "query": "What is Python used for?",
                    "candidates": [
                        {"docid": "d0", "doc": {"segment": "okay"}, "judgment": 1},
                        {"docid": "d1", "doc": {"segment": "good"}, "judgment": 2},
                    ],
                }
            ),
            "--min-judgment",
            "1",
            "--output",
            "json",
        ]
    )

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["resolved"]["min_judgment"] == 1


def test_direct_create_accepts_umbrela_judgments(monkeypatch: Any, capsys: Any) -> None:
    def fake_create(self: Nuggetizer, request: Any) -> list[ScoredNugget]:
        del self
        assert request.query.text == "What is Python used for?"
        assert [document.docid for document in request.documents] == ["d1"]
        assert request.documents[0].segment == "Python is used for web development."
        return [ScoredNugget(text="nugget", importance="vital")]

    monkeypatch.setattr(Nuggetizer, "create", fake_create)

    exit_code = main(
        [
            "create",
            "--input-json",
            json.dumps(
                {
                    "judgments": [
                        {
                            "query": "What is Python used for?",
                            "passage": "Python can be hard to learn.",
                            "judgment": 1,
                        },
                        {
                            "query": "What is Python used for?",
                            "passage": "Python is used for web development.",
                            "judgment": 2,
                        },
                    ]
                }
            ),
            "--output",
            "json",
        ]
    )

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["artifacts"][0]["name"] == "create-result"


def test_direct_create_returns_empty_nuggets_when_all_candidates_are_filtered(
    monkeypatch: Any, capsys: Any
) -> None:
    def fake_create(self: Nuggetizer, request: Any) -> list[ScoredNugget]:
        del self
        assert request.documents == []
        return []

    monkeypatch.setattr(Nuggetizer, "create", fake_create)

    exit_code = main(
        [
            "create",
            "--input-json",
            json.dumps(
                {
                    "judgments": [
                        {
                            "query": "What is Python used for?",
                            "passage": "Not relevant",
                            "judgment": 0,
                        }
                    ]
                }
            ),
            "--output",
            "json",
        ]
    )

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["artifacts"][0]["data"]["nuggets"] == []


def test_prompt_render_create_accepts_umbrela_judgments(capsys: Any) -> None:
    exit_code = main(
        [
            "prompt",
            "render",
            "create",
            "--input-json",
            json.dumps(
                {
                    "judgments": [
                        {
                            "query": "What is Python used for?",
                            "passage": "Bad passage",
                            "judgment": 1,
                        },
                        {
                            "query": "What is Python used for?",
                            "passage": "Good passage",
                            "judgment": 2,
                        },
                    ]
                }
            ),
            "--output",
            "json",
        ]
    )

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    view = output["artifacts"][0]["data"]
    assert view["inputs"]["candidate_count"] == 1
    assert view["inputs"]["min_judgment"] == 2


def test_direct_create_accepts_anserini_style_doc_contents(
    monkeypatch: Any, capsys: Any
) -> None:
    def fake_create(self: Nuggetizer, request: Any) -> list[ScoredNugget]:
        assert request.query.qid == "q0"
        assert request.documents[0].docid == "96854"
        assert (
            request.documents[0].segment
            == "Python is used for web development and data analysis."
        )
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
                    "candidates": [
                        {
                            "docid": "96854",
                            "doc": {
                                "id": "96854",
                                "contents": "Python is used for web development and data analysis.",
                            },
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
    assert output["command"] == "create"
    assert output["artifacts"][0]["name"] == "create-result"


def test_direct_create_accepts_anserini_rest_payload(
    monkeypatch: Any, capsys: Any
) -> None:
    def fake_create(self: Nuggetizer, request: Any) -> list[ScoredNugget]:
        assert request.query.qid == "q0"
        assert request.query.text == "What is Python used for?"
        assert request.documents[0].docid == "1737459"
        assert (
            request.documents[0].segment == "Python is widely used for web development."
        )
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
                    "api": "v1",
                    "index": "msmarco-v1-passage",
                    "query": {"text": "What is Python used for?"},
                    "candidates": [
                        {
                            "docid": "1737459",
                            "score": 10.58,
                            "rank": 1,
                            "doc": "Python is widely used for web development.",
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
    assert output["command"] == "create"
    assert output["artifacts"][0]["name"] == "create-result"


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


def test_direct_assign_via_joined_record_input_json(
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
                    "answer_record": {
                        "topic_id": "q1",
                        "topic": "What is Python used for?",
                        "answer": [
                            {"text": "Python is commonly used for web development."}
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
            ),
            "--output",
            "json",
        ]
    )

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["artifacts"][0]["data"]["query"] == "What is Python used for?"


def test_direct_assign_via_joined_envelope_input_json(
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
                                            {
                                                "text": "Python is commonly used for web development."
                                            }
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
            ),
            "--output",
            "json",
        ]
    )

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["artifacts"][0]["data"]["query"] == "What is Python used for?"


def test_direct_assign_via_joined_batch_input_json_supports_metrics(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    def fake_assign(
        self: Nuggetizer, query: str, context: str, nuggets: list[ScoredNugget]
    ) -> list[AssignedScoredNugget]:
        del self, query, context
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
                    "answer_records": [
                        {
                            "run_id": "demo-run",
                            "topic_id": "q1",
                            "topic": "What is Python used for?",
                            "response_length": 10,
                            "answer": [
                                {"text": "Python is commonly used for web development."}
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
            ),
            "--output",
            "json",
        ]
    )

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    records = output["artifacts"][0]["data"]
    assert len(records) == 2
    assert records[0]["qid"] == "q1"
    assert records[0]["answer_text"] == "Python is commonly used for web development."
    assert records[0]["response_length"] == 10
    assert records[0]["run_id"] == "demo-run"
    assert records[1]["run_id"] == "direct-assign"

    assignments_path = tmp_path / "assignments.jsonl"
    metrics_path = tmp_path / "metrics.jsonl"
    write_jsonl(assignments_path, records)

    metrics_exit_code = main(
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

    assert metrics_exit_code == 0
    metrics_output = json.loads(capsys.readouterr().out)
    metric_records = read_jsonl(metrics_path)
    assert metric_records[0]["qid"] == "q1"
    assert metric_records[1]["qid"] == "q1"
    assert metrics_output["metrics"]["global_metrics"]["qid"] == "all"


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
            "assign",
            "--input-kind",
            "retrieval",
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
        "create, assign, metrics, serve, view, prompt, describe, schema, doctor, validate"
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
    assert (
        output["artifacts"][0]["data"]["direct_input"]["shapes"][2]["name"]
        == "joined-single-envelopes"
    )


def test_schema_assign_output_answers_returns_json_envelope(capsys: Any) -> None:
    exit_code = main(["schema", "assign-output-answers", "--output", "json"])

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["command"] == "schema"
    assert "run_id" in output["artifacts"][0]["data"]["required"]


def test_schema_assign_direct_input_includes_joined_batch_forms(capsys: Any) -> None:
    exit_code = main(["schema", "assign-direct-input", "--output", "json"])

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["command"] == "schema"
    one_of = output["artifacts"][0]["data"]["oneOf"]
    assert any(
        set(option["required"]) == {"answer_records", "nugget_record"}
        for option in one_of
    )
    assert any(
        set(option["required"]) == {"answers_envelope", "nugget_envelope"}
        for option in one_of
    )
    assert "overrides" in output["artifacts"][0]["data"]["properties"]


def test_schema_create_direct_input_includes_overrides(capsys: Any) -> None:
    exit_code = main(["schema", "create-direct-input", "--output", "json"])

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    overrides = output["artifacts"][0]["data"]["properties"]["overrides"]["properties"]
    assert "creator_model" in overrides
    assert "scorer_model" in overrides
    assert "min_judgment" in overrides


def test_doctor_returns_json_envelope(capsys: Any) -> None:
    exit_code = main(["doctor", "--output", "json"])

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["command"] == "doctor"
    assert "python_version" in output["metrics"]
    assert "backend_readiness" in output["metrics"]
    assert "command_readiness" in output["metrics"]
    assert "serve" in output["metrics"]["command_readiness"]


def test_top_level_help_includes_command_summaries(capsys: Any) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])

    assert exc_info.value.code == 0
    stdout = capsys.readouterr().out
    assert "Nuggetizer packaged CLI" in stdout
    assert "create and score nuggets" in stdout.lower()
    assert "inspect an existing nuggetizer artifact" in stdout.lower()


def test_serve_command_starts_uvicorn(monkeypatch: Any) -> None:
    pytest.importorskip("fastapi")
    seen: dict[str, Any] = {}

    def fake_run(app: Any, host: str, port: int) -> None:
        seen["app"] = app
        seen["host"] = host
        seen["port"] = port

    monkeypatch.setattr("uvicorn.run", fake_run)

    exit_code = main(["serve", "--port", "8085"])

    assert exit_code == 0
    assert seen["host"] == "0.0.0.0"
    assert seen["port"] == 8085


def test_serve_app_health_create_and_assign(monkeypatch: Any) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from nuggetizer.api.app import create_app
    from nuggetizer.api.runtime import ServerConfig

    def fake_create(self: Nuggetizer, request: Any) -> list[ScoredNugget]:
        return [ScoredNugget(text="nugget", importance="vital")]

    def fake_assign(
        self: Nuggetizer, query: str, context: str, nuggets: list[Any]
    ) -> list[AssignedScoredNugget]:
        del query, context, nuggets
        return [
            AssignedScoredNugget(
                text="nugget",
                importance="vital",
                assignment="support",
            )
        ]

    monkeypatch.setattr(Nuggetizer, "create", fake_create)
    monkeypatch.setattr(Nuggetizer, "assign", fake_assign)

    client = TestClient(create_app(ServerConfig(host="127.0.0.1", port=8085)))

    health_response = client.get("/healthz")
    create_response = client.post(
        "/v1/create",
        json={
            "api": "v1",
            "index": "msmarco-v1-passage",
            "query": {"text": "What is Python used for?"},
            "candidates": [
                {
                    "docid": "d0",
                    "score": 1.0,
                    "rank": 1,
                    "doc": "web",
                }
            ],
        },
    )
    assign_response = client.post(
        "/v1/assign",
        json={
            "query": "What is Python used for?",
            "context": "Python is used for web development.",
            "nuggets": [
                {"text": "Python is used for web development.", "importance": "vital"}
            ],
        },
    )

    assert health_response.status_code == 200
    assert health_response.json() == {"status": "ok"}
    assert create_response.status_code == 200
    assert create_response.json()["artifacts"][0]["name"] == "create-result"
    assert assign_response.status_code == 200
    assert assign_response.json()["artifacts"][0]["name"] == "assign-result"


def test_serve_app_assign_accepts_joined_records(monkeypatch: Any) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from nuggetizer.api.app import create_app
    from nuggetizer.api.runtime import ServerConfig

    seen: dict[str, Any] = {}

    def fake_assign(
        self: Nuggetizer, query: str, context: str, nuggets: list[Any]
    ) -> list[AssignedScoredNugget]:
        seen["query"] = query
        seen["context"] = context
        seen["nuggets"] = nuggets
        return [
            AssignedScoredNugget(
                text="nugget",
                importance="vital",
                assignment="support",
            )
        ]

    monkeypatch.setattr(Nuggetizer, "assign", fake_assign)

    client = TestClient(create_app(ServerConfig(host="127.0.0.1", port=8085)))
    response = client.post(
        "/v1/assign",
        json={
            "answer_record": {
                "topic_id": "q1",
                "topic": "What is Python used for?",
                "answer": [
                    {"text": "Python is used for web development."},
                    {"text": "It is also used for data science."},
                ],
            },
            "nugget_record": {
                "qid": "q1",
                "query": "What is Python used for?",
                "nuggets": [
                    {
                        "text": "Python is used for web development.",
                        "importance": "vital",
                    }
                ],
            },
        },
    )

    assert response.status_code == 200
    assert response.json()["artifacts"][0]["name"] == "assign-result"
    assert seen["query"] == "What is Python used for?"
    assert (
        seen["context"]
        == "Python is used for web development. It is also used for data science."
    )
    assert seen["nuggets"] == [
        ScoredNugget(text="Python is used for web development.", importance="vital")
    ]


def test_serve_app_assign_accepts_joined_envelopes(monkeypatch: Any) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from nuggetizer.api.app import create_app
    from nuggetizer.api.runtime import ServerConfig

    seen: dict[str, Any] = {}

    def fake_assign(
        self: Nuggetizer, query: str, context: str, nuggets: list[Any]
    ) -> list[AssignedScoredNugget]:
        seen["query"] = query
        seen["context"] = context
        seen["nuggets"] = nuggets
        return [
            AssignedScoredNugget(
                text="nugget",
                importance="vital",
                assignment="support",
            )
        ]

    monkeypatch.setattr(Nuggetizer, "assign", fake_assign)

    client = TestClient(create_app(ServerConfig(host="127.0.0.1", port=8085)))
    response = client.post(
        "/v1/assign",
        json={
            "answer_envelope": {
                "schema_version": "castorini.cli.v1",
                "repo": "ragnarok",
                "command": "generate",
                "artifacts": [
                    {
                        "name": "generation-results",
                        "kind": "data",
                        "data": [
                            {
                                "topic_id": "q1",
                                "topic": "What is Python used for?",
                                "answer": [
                                    {
                                        "text": "Python is used for web development.",
                                    },
                                    {
                                        "text": "It is also used for data science.",
                                    },
                                ],
                            }
                        ],
                    }
                ],
            },
            "nugget_envelope": {
                "schema_version": "castorini.cli.v1",
                "repo": "nuggetizer",
                "command": "create",
                "artifacts": [
                    {
                        "name": "create-result",
                        "kind": "data",
                        "data": {
                            "qid": "q1",
                            "query": "What is Python used for?",
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
        },
    )

    assert response.status_code == 200
    assert response.json()["artifacts"][0]["name"] == "assign-result"
    assert seen["query"] == "What is Python used for?"
    assert (
        seen["context"]
        == "Python is used for web development. It is also used for data science."
    )
    assert seen["nuggets"] == [
        ScoredNugget(text="Python is used for web development.", importance="vital")
    ]


def test_serve_app_assign_accepts_joined_batch_records(monkeypatch: Any) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from nuggetizer.api.app import create_app
    from nuggetizer.api.runtime import ServerConfig

    seen_contexts: list[str] = []

    def fake_assign(
        self: Nuggetizer, query: str, context: str, nuggets: list[Any]
    ) -> list[AssignedScoredNugget]:
        del self, query, nuggets
        seen_contexts.append(context)
        return [
            AssignedScoredNugget(
                text="nugget",
                importance="vital",
                assignment="support",
            )
        ]

    monkeypatch.setattr(Nuggetizer, "assign", fake_assign)

    client = TestClient(create_app(ServerConfig(host="127.0.0.1", port=8085)))
    response = client.post(
        "/v1/assign",
        json={
            "answer_records": [
                {
                    "run_id": "demo-run",
                    "topic_id": "q1",
                    "topic": "What is Python used for?",
                    "response_length": 10,
                    "answer": [
                        {"text": "Python is used for web development."},
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
                "qid": "q1",
                "query": "What is Python used for?",
                "nuggets": [
                    {
                        "text": "Python is used for web development.",
                        "importance": "vital",
                    }
                ],
            },
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["artifacts"][0]["name"] == "assign-result"
    assert len(body["artifacts"][0]["data"]) == 2
    assert body["artifacts"][0]["data"][0]["run_id"] == "demo-run"
    assert body["artifacts"][0]["data"][1]["run_id"] == "direct-assign"
    assert seen_contexts == [
        "Python is used for web development.",
        "Python is also used for automation.",
    ]


def test_serve_app_assign_accepts_joined_batch_envelopes(monkeypatch: Any) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from nuggetizer.api.app import create_app
    from nuggetizer.api.runtime import ServerConfig

    seen_contexts: list[str] = []

    def fake_assign(
        self: Nuggetizer, query: str, context: str, nuggets: list[Any]
    ) -> list[AssignedScoredNugget]:
        del self, query, nuggets
        seen_contexts.append(context)
        return [
            AssignedScoredNugget(
                text="nugget",
                importance="vital",
                assignment="support",
            )
        ]

    monkeypatch.setattr(Nuggetizer, "assign", fake_assign)

    client = TestClient(create_app(ServerConfig(host="127.0.0.1", port=8085)))
    response = client.post(
        "/v1/assign",
        json={
            "answers_envelope": {
                "schema_version": "castorini.cli.v1",
                "repo": "ragnarok",
                "command": "generate",
                "artifacts": [
                    {
                        "name": "generation-results",
                        "kind": "data",
                        "data": [
                            {
                                "run_id": "demo-run",
                                "topic_id": "q1",
                                "topic": "What is Python used for?",
                                "response_length": 10,
                                "answer": [
                                    {"text": "Python is used for web development."},
                                ],
                            },
                            {
                                "topic_id": "q1",
                                "topic": "What is Python used for?",
                                "response_length": 8,
                                "answer": [
                                    {"text": "Python is also used for automation."}
                                ],
                            },
                        ],
                    }
                ],
            },
            "nugget_envelope": {
                "schema_version": "castorini.cli.v1",
                "repo": "nuggetizer",
                "command": "create",
                "artifacts": [
                    {
                        "name": "create-result",
                        "kind": "data",
                        "data": {
                            "qid": "q1",
                            "query": "What is Python used for?",
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
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["artifacts"][0]["name"] == "assign-result"
    assert len(body["artifacts"][0]["data"]) == 2
    assert body["artifacts"][0]["data"][0]["qid"] == "q1"
    assert body["artifacts"][0]["data"][1]["run_id"] == "direct-assign"
    assert seen_contexts == [
        "Python is used for web development.",
        "Python is also used for automation.",
    ]


def test_serve_app_assign_rejects_joined_batch_id_mismatch() -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from nuggetizer.api.app import create_app
    from nuggetizer.api.runtime import ServerConfig

    client = TestClient(create_app(ServerConfig(host="127.0.0.1", port=8085)))
    response = client.post(
        "/v1/assign",
        json={
            "answer_records": [
                {
                    "topic_id": "q2",
                    "topic": "What is Python used for?",
                    "response_length": 10,
                    "answer": [{"text": "Python is used for web development."}],
                }
            ],
            "nugget_record": {
                "qid": "q1",
                "query": "What is Python used for?",
                "nuggets": [
                    {
                        "text": "Python is used for web development.",
                        "importance": "vital",
                    }
                ],
            },
        },
    )

    assert response.status_code == 400
    assert response.json()["status"] == "validation_error"


def test_serve_app_rejects_invalid_payload() -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from nuggetizer.api.app import create_app
    from nuggetizer.api.runtime import ServerConfig

    client = TestClient(create_app(ServerConfig(host="127.0.0.1", port=8085)))

    response = client.post("/v1/create", json={"query": 1})

    assert response.status_code == 400
    assert response.json()["status"] == "validation_error"


def test_serve_app_create_applies_request_overrides(monkeypatch: Any) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from nuggetizer.api.app import create_app
    from nuggetizer.api.runtime import ServerConfig

    captured: dict[str, Any] = {}

    def fake_build_create_nuggetizer_kwargs(args: Any) -> dict[str, Any]:
        captured["creator_model"] = args.creator_model
        captured["scorer_model"] = args.scorer_model
        captured["model"] = args.model
        captured["min_judgment"] = args.min_judgment
        return {"model": args.model}

    def fake_create(self: Nuggetizer, request: Any) -> list[ScoredNugget]:
        del self, request
        return [ScoredNugget(text="nugget", importance="vital")]

    monkeypatch.setattr(
        "nuggetizer.api.runtime.build_create_nuggetizer_kwargs",
        fake_build_create_nuggetizer_kwargs,
    )
    monkeypatch.setattr(Nuggetizer, "create", fake_create)

    client = TestClient(create_app(ServerConfig(host="127.0.0.1", port=8085)))
    response = client.post(
        "/v1/create",
        json={
            "query": "What is Python used for?",
            "candidates": ["web"],
            "overrides": {
                "creator_model": "gpt-4.1-mini",
                "scorer_model": "gpt-4.1-mini",
                "min_judgment": 3,
            },
        },
    )

    assert response.status_code == 200
    assert captured["model"] == "gpt-4o"
    assert captured["creator_model"] == "gpt-4.1-mini"
    assert captured["scorer_model"] == "gpt-4.1-mini"
    assert captured["min_judgment"] == 3
    assert response.json()["resolved"]["creator_model"] == "gpt-4.1-mini"
    assert response.json()["resolved"]["scorer_model"] == "gpt-4.1-mini"
    assert response.json()["resolved"]["min_judgment"] == 3
    assert response.json()["resolved"]["reasoning_effort"] is None


def test_serve_app_assign_applies_request_model_override(monkeypatch: Any) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from nuggetizer.api.app import create_app
    from nuggetizer.api.runtime import ServerConfig

    captured: dict[str, Any] = {}

    def fake_build_assign_nuggetizer_kwargs(args: Any) -> dict[str, Any]:
        captured["model"] = args.model
        return {"assigner_model": args.model}

    def fake_assign(
        self: Nuggetizer, query: str, context: str, nuggets: list[Any]
    ) -> list[AssignedScoredNugget]:
        del self, query, context, nuggets
        return [
            AssignedScoredNugget(
                text="nugget",
                importance="vital",
                assignment="support",
            )
        ]

    monkeypatch.setattr(
        "nuggetizer.api.runtime.build_assign_nuggetizer_kwargs",
        fake_build_assign_nuggetizer_kwargs,
    )
    monkeypatch.setattr(Nuggetizer, "assign", fake_assign)

    client = TestClient(create_app(ServerConfig(host="127.0.0.1", port=8085)))
    response = client.post(
        "/v1/assign",
        json={
            "query": "What is Python used for?",
            "context": "Python is used for web development.",
            "nuggets": [
                {"text": "Python is used for web development.", "importance": "vital"}
            ],
            "overrides": {"model": "gpt-4.1-mini"},
        },
    )

    assert response.status_code == 200
    assert captured["model"] == "gpt-4.1-mini"
    assert response.json()["resolved"]["model"] == "gpt-4.1-mini"


def test_serve_app_create_exposes_default_model_resolution(monkeypatch: Any) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from nuggetizer.api.app import create_app
    from nuggetizer.api.runtime import ServerConfig

    def fake_create(self: Nuggetizer, request: Any) -> list[ScoredNugget]:
        del self, request
        return [ScoredNugget(text="nugget", importance="vital")]

    monkeypatch.setattr(Nuggetizer, "create", fake_create)

    client = TestClient(
        create_app(
            ServerConfig(
                host="127.0.0.1",
                port=8085,
                model="gpt-4o",
            )
        )
    )
    response = client.post(
        "/v1/create",
        json={
            "query": "What is Python used for?",
            "candidates": ["web"],
        },
    )

    assert response.status_code == 200
    assert response.json()["resolved"]["model"] == "gpt-4o"
    assert response.json()["resolved"]["creator_model"] == "gpt-4o"
    assert response.json()["resolved"]["scorer_model"] == "gpt-4o"


def test_serve_app_rejects_invalid_override_combinations() -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from nuggetizer.api.app import create_app
    from nuggetizer.api.runtime import ServerConfig

    client = TestClient(create_app(ServerConfig(host="127.0.0.1", port=8085)))
    response = client.post(
        "/v1/create",
        json={
            "query": "What is Python used for?",
            "candidates": ["web"],
            "overrides": {
                "use_azure_openai": True,
                "use_openrouter": True,
            },
        },
    )

    assert response.status_code == 400
    assert response.json()["status"] == "validation_error"


def test_serve_app_create_accepts_rank_llm_envelope(monkeypatch: Any) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from nuggetizer.api.app import create_app
    from nuggetizer.api.runtime import ServerConfig

    def fake_create(self: Nuggetizer, request: Any) -> list[ScoredNugget]:
        del request
        return [ScoredNugget(text="nugget", importance="vital")]

    monkeypatch.setattr(Nuggetizer, "create", fake_create)

    client = TestClient(create_app(ServerConfig(host="127.0.0.1", port=8085)))
    response = client.post(
        "/v1/create",
        json={
            "schema_version": "castorini.cli.v1",
            "repo": "rank_llm",
            "command": "rerank",
            "artifacts": [
                {
                    "name": "rerank-results",
                    "kind": "data",
                    "value": [
                        {
                            "query": {"text": "What is Python used for?", "qid": ""},
                            "candidates": [
                                {
                                    "docid": "d0",
                                    "score": 1.0,
                                    "doc": {"contents": "web"},
                                }
                            ],
                        }
                    ],
                }
            ],
        },
    )

    assert response.status_code == 200
    assert response.json()["artifacts"][0]["name"] == "create-result"


def test_serve_app_create_accepts_umbrela_judgments_envelope(
    monkeypatch: Any,
) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from nuggetizer.api.app import create_app
    from nuggetizer.api.runtime import ServerConfig

    def fake_create(self: Nuggetizer, request: Any) -> list[ScoredNugget]:
        del self
        assert [document.docid for document in request.documents] == ["d1"]
        assert request.documents[0].segment == "Python is used for web development."
        return [ScoredNugget(text="nugget", importance="vital")]

    monkeypatch.setattr(Nuggetizer, "create", fake_create)

    client = TestClient(create_app(ServerConfig(host="127.0.0.1", port=8085)))
    response = client.post(
        "/v1/create",
        json={
            "schema_version": "castorini.cli.v1",
            "repo": "umbrela",
            "command": "judge",
            "artifacts": [
                {
                    "name": "judgments",
                    "kind": "data",
                    "data": [
                        {
                            "query": "What is Python used for?",
                            "passage": "Not relevant",
                            "judgment": 1,
                        },
                        {
                            "query": "What is Python used for?",
                            "passage": "Python is used for web development.",
                            "judgment": 2,
                        },
                    ],
                }
            ],
        },
    )

    assert response.status_code == 200
    assert response.json()["artifacts"][0]["name"] == "create-result"


def test_serve_app_create_accepts_umbrela_curl_payload(monkeypatch: Any) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from nuggetizer.api.app import create_app
    from nuggetizer.api.runtime import ServerConfig

    def fake_create(self: Nuggetizer, request: Any) -> list[ScoredNugget]:
        del self
        assert [document.docid for document in request.documents] == ["d0"]
        return [ScoredNugget(text="nugget", importance="vital")]

    monkeypatch.setattr(Nuggetizer, "create", fake_create)

    client = TestClient(create_app(ServerConfig(host="127.0.0.1", port=8085)))
    response = client.post(
        "/v1/create",
        json={
            "schema_version": "castorini.cli.v1",
            "command": "judge",
            "artifacts": [
                {
                    "name": "judgments",
                    "kind": "data",
                    "data": [
                        {
                            "query": "What is Python used for?",
                            "passage": "Python is used for web development.",
                            "judgment": 2,
                        }
                    ],
                }
            ],
            "overrides": {"min_judgment": 2},
        },
    )

    assert response.status_code == 200
    assert response.json()["resolved"]["min_judgment"] == 2


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
                "query": "What is Python used for? " * 5,
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
        == ("What is Python used for? " * 5).strip()
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
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
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


def test_config_file_sets_default_output_format(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    config_dir = tmp_path / "config" / "nuggetizer"
    config_dir.mkdir(parents=True)
    config_file = config_dir / "config.toml"
    config_file.write_text('output = "json"\n', encoding="utf-8")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))

    exit_code = main(["doctor"])

    assert exit_code == 0
    stdout = capsys.readouterr().out
    output = json.loads(stdout)
    assert output["command"] == "doctor"
    assert output["metrics"]["config_file"] == str(config_file)


def test_pipe_create_jsonl_output_is_valid_for_assign(
    monkeypatch: Any, capsys: Any
) -> None:
    """Verify that create's JSON output contains nuggets consumable by assign."""

    def fake_create(self: Nuggetizer, request: Any) -> list[ScoredNugget]:
        return [ScoredNugget(text="Python is useful.", importance="vital")]

    monkeypatch.setattr(Nuggetizer, "create", fake_create)

    exit_code = main(
        [
            "create",
            "--input-json",
            json.dumps(
                {"query": "What is Python?", "candidates": ["Python is useful."]}
            ),
            "--output",
            "json",
        ]
    )

    assert exit_code == 0
    envelope = json.loads(capsys.readouterr().out)
    data = envelope["artifacts"][0]["data"]
    assert "query" in data
    assert "nuggets" in data
    assert all(
        "text" in nugget and "importance" in nugget for nugget in data["nuggets"]
    )
