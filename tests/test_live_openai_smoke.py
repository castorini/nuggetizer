from __future__ import annotations

import json
import os
from textwrap import indent

import pytest

from nuggetizer.cli.main import main


pytestmark = pytest.mark.skipif(
    os.getenv("NUGGETIZER_LIVE_OPENAI_SMOKE") != "1",
    reason="Set NUGGETIZER_LIVE_OPENAI_SMOKE=1 to run live OpenAI smoke tests.",
)


def _run_json_command(
    args: list[str], capsys: pytest.CaptureFixture[str]
) -> dict[str, object]:
    exit_code = main(args)
    assert exit_code == 0
    output = capsys.readouterr().out
    lines = [line for line in output.splitlines() if line.strip()]
    return json.loads(lines[-1])


def _pretty_print_create(model: str, result: dict[str, object]) -> None:
    nuggets = result["nuggets"]
    lines = [
        "Nuggetizer live smoke result",
        f"model: {model}",
        f"query: {result['query']}",
        "nuggets:",
    ]
    for index, nugget in enumerate(nuggets, start=1):
        lines.extend(
            [
                f"  {index}. importance: {nugget['importance']}",
                "     text:",
                indent(str(nugget["text"]), "       "),
            ]
        )
        reasoning = nugget.get("reasoning")
        if reasoning:
            lines.extend(["     reasoning:", indent(str(reasoning), "       ")])
    print("\n".join(lines))


def _pretty_print_assign(
    label: str, query: str, context: str, nuggets: list[dict[str, object]]
) -> None:
    lines = [
        "",
        f"assignment case: {label}",
        f"query: {query}",
        "context:",
        indent(context, "  "),
        "assigned nuggets:",
    ]
    for index, nugget in enumerate(nuggets, start=1):
        lines.extend(
            [
                f"  {index}. assignment: {nugget['assignment']} importance: {nugget['importance']}",
                "     text:",
                indent(str(nugget["text"]), "       "),
            ]
        )
        reasoning = nugget.get("reasoning")
        if reasoning:
            lines.extend(["     reasoning:", indent(str(reasoning), "       ")])
    print("\n".join(lines))


def test_direct_create_and_assign_openai_smoke(
    capsys: pytest.CaptureFixture[str],
) -> None:
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("OPENROUTER_API_KEY"):
        pytest.skip("OPENAI_API_KEY or OPENROUTER_API_KEY is required.")

    model = os.getenv("NUGGETIZER_LIVE_OPENAI_MODEL", "gpt-4o-mini")
    query = "What is Python used for?"
    create_payload = {
        "query": query,
        "candidates": [
            (
                "Python is widely used for web development, data analysis, "
                "automation, scripting, machine learning, artificial intelligence, "
                "scientific computing, data visualization, and backend development."
            )
        ],
    }
    create_result = _run_json_command(
        [
            "create",
            "--model",
            model,
            "--input-json",
            json.dumps(create_payload),
            "--output",
            "json",
        ],
        capsys,
    )
    assert create_result["command"] == "create"
    assert create_result["status"] == "success"
    created = create_result["artifacts"][0]["data"]
    nuggets = created["nuggets"]
    assert nuggets
    _pretty_print_create(model, created)

    assignment_cases = {
        "comprehensive": (
            "Python is used for web development, backend services, data analysis, "
            "automation, scripting, machine learning, artificial intelligence, "
            "scientific computing, and data visualization."
        ),
        "sparse": "Python is used for web development and data analysis.",
        "random": (
            "I had toast for breakfast and then took the tram to the museum. "
            "The weather was cloudy and the coffee was average."
        ),
    }
    for label, context in assignment_cases.items():
        assign_result = _run_json_command(
            [
                "assign",
                "--model",
                model,
                "--input-json",
                json.dumps(
                    {
                        "query": query,
                        "context": context,
                        "nuggets": nuggets,
                    }
                ),
                "--output",
                "json",
            ],
            capsys,
        )
        assert assign_result["command"] == "assign"
        assert assign_result["status"] == "success"
        assigned = assign_result["artifacts"][0]["data"]
        assert assigned["nuggets"]
        _pretty_print_assign(label, query, context, assigned["nuggets"])
