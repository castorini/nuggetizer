from __future__ import annotations

import asyncio
import json
import os
import sys
from textwrap import indent
from typing import TextIO, TypedDict, cast

import pytest

from nuggetizer.cli.adapters import scored_nuggets_from_record
from nuggetizer.cli.main import main
from nuggetizer.models.nuggetizer import Nuggetizer


pytestmark = pytest.mark.skipif(
    os.getenv("NUGGETIZER_LIVE_OPENAI_SMOKE") != "1",
    reason="Set NUGGETIZER_LIVE_OPENAI_SMOKE=1 to run live OpenAI smoke tests.",
)


class SmokeNuggetRecord(TypedDict):
    text: object
    importance: object
    assignment: object


class SmokeArtifactData(TypedDict, total=False):
    query: object
    nuggets: list[SmokeNuggetRecord]
    creator_reasoning_traces: list[object]
    scoring_reasoning_traces: list[object]
    reasoning_traces: list[object]


class SmokeArtifact(TypedDict):
    data: SmokeArtifactData


class SmokeCommandResult(TypedDict):
    command: str
    status: str
    artifacts: list[SmokeArtifact]


def _stdout() -> TextIO:
    stdout = sys.__stdout__
    assert stdout is not None
    return stdout


def _trace_list(value: object) -> list[object]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _run_json_command(
    args: list[str], capsys: pytest.CaptureFixture[str]
) -> SmokeCommandResult:
    exit_code = main(args)
    assert exit_code == 0
    output = capsys.readouterr().out
    lines = [line for line in output.splitlines() if line.strip()]
    return cast(SmokeCommandResult, json.loads(lines[-1]))


def _unique_nonempty_traces(traces: list[object] | tuple[object, ...]) -> list[str]:
    unique_traces: list[str] = []
    seen: set[str] = set()
    for trace in traces:
        normalized = str(trace).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique_traces.append(normalized)
    return unique_traces


def _append_reasoning_traces(
    lines: list[str], traces: list[object] | tuple[object, ...], *, label_prefix: str
) -> None:
    unique_traces = _unique_nonempty_traces(traces)
    if not unique_traces:
        return
    lines.append("")
    for index, trace in enumerate(unique_traces, start=1):
        lines.append(f"{label_prefix} {index}: {trace}")


def _pretty_print_create(model: str, result: SmokeArtifactData) -> None:
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
    _append_reasoning_traces(
        lines,
        _trace_list(result.get("creator_reasoning_traces")),
        label_prefix="creator reasoning trace",
    )
    _append_reasoning_traces(
        lines,
        _trace_list(result.get("scoring_reasoning_traces")),
        label_prefix="scoring reasoning trace",
    )
    stdout = _stdout()
    stdout.write("\n".join(lines) + "\n")
    stdout.flush()


def _pretty_print_assign(
    label: str,
    query: str,
    context: str,
    nuggets: list[SmokeNuggetRecord],
    *,
    scoring_reasoning_traces: list[object] | tuple[object, ...] = (),
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
    _append_reasoning_traces(
        lines,
        scoring_reasoning_traces,
        label_prefix="scoring reasoning trace",
    )
    stdout = _stdout()
    stdout.write("\n".join(lines) + "\n")
    stdout.flush()


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
    nuggetizer = Nuggetizer(assigner_model=model)
    assigned_batches = asyncio.run(
        nuggetizer.async_assign_batch(
            [query] * len(assignment_cases),
            list(assignment_cases.values()),
            [scored_nuggets_from_record({"nuggets": nuggets})] * len(assignment_cases),
        )
    )
    for (label, context), assigned_batch in zip(
        assignment_cases.items(), assigned_batches, strict=True
    ):
        assigned_nuggets: list[SmokeNuggetRecord] = [
            {
                "text": nugget.text,
                "importance": nugget.importance,
                "assignment": nugget.assignment,
            }
            for nugget in assigned_batch
        ]
        assert assigned_nuggets
        _pretty_print_assign(label, query, context, assigned_nuggets)


def test_direct_create_and_assign_reasoning_openai_smoke(
    capsys: pytest.CaptureFixture[str],
) -> None:
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("OPENROUTER_API_KEY"):
        pytest.skip("OPENAI_API_KEY or OPENROUTER_API_KEY is required.")

    model = os.getenv("NUGGETIZER_LIVE_OPENAI_REASONING_MODEL", "gpt-5-mini")
    create_result = _run_json_command(
        [
            "create",
            "--model",
            model,
            "--execution-mode",
            "async",
            "--reasoning-effort",
            "medium",
            "--include-reasoning",
            "--input-json",
            json.dumps(
                {
                    "query": "What is Python used for?",
                    "candidates": [
                        (
                            "Python is widely used for web development, data analysis, "
                            "automation, scripting, machine learning, artificial intelligence, "
                            "scientific computing, data visualization, and backend development."
                        )
                    ],
                }
            ),
            "--output",
            "json",
        ],
        capsys,
    )
    assert create_result["command"] == "create"
    assert create_result["status"] == "success"
    created = create_result["artifacts"][0]["data"]
    assert created["nuggets"]
    creator_traces = _trace_list(created.get("creator_reasoning_traces"))
    scoring_traces = _trace_list(created.get("scoring_reasoning_traces"))
    assert creator_traces or scoring_traces
    _pretty_print_create(model, created)

    query = str(created["query"])
    context = "Python is used for web development and data analysis."
    assign_result = _run_json_command(
        [
            "assign",
            "--model",
            model,
            "--execution-mode",
            "async",
            "--reasoning-effort",
            "medium",
            "--include-reasoning",
            "--input-json",
            json.dumps(
                {
                    "query": query,
                    "context": context,
                    "nuggets": created["nuggets"],
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
    assert assigned.get("reasoning_traces")
    _pretty_print_assign(
        "reasoning",
        query,
        context,
        assigned["nuggets"],
        scoring_reasoning_traces=_trace_list(assigned.get("reasoning_traces")),
    )


def test_direct_create_and_assign_reasoning_openrouter_smoke(
    capsys: pytest.CaptureFixture[str],
) -> None:
    if not os.getenv("OPENROUTER_API_KEY"):
        pytest.skip("OPENROUTER_API_KEY is required.")

    model = os.getenv(
        "NUGGETIZER_LIVE_OPENROUTER_REASONING_MODEL", "openrouter/hunter-alpha"
    )
    create_result = _run_json_command(
        [
            "create",
            "--model",
            model,
            "--use-openrouter",
            "--execution-mode",
            "async",
            "--reasoning-effort",
            "medium",
            "--include-reasoning",
            "--input-json",
            json.dumps(
                {
                    "query": "What is Python used for?",
                    "candidates": [
                        (
                            "Python is widely used for web development, data analysis, "
                            "automation, scripting, machine learning, artificial intelligence, "
                            "scientific computing, data visualization, and backend development."
                        )
                    ],
                }
            ),
            "--output",
            "json",
        ],
        capsys,
    )
    assert create_result["command"] == "create"
    assert create_result["status"] == "success"
    created = create_result["artifacts"][0]["data"]
    assert created["nuggets"]
    creator_traces = _trace_list(created.get("creator_reasoning_traces"))
    scoring_traces = _trace_list(created.get("scoring_reasoning_traces"))
    assert creator_traces or scoring_traces
    _pretty_print_create(model, created)

    query = str(created["query"])
    context = "Python is used for web development and data analysis."
    assign_result = _run_json_command(
        [
            "assign",
            "--model",
            model,
            "--use-openrouter",
            "--execution-mode",
            "async",
            "--reasoning-effort",
            "medium",
            "--include-reasoning",
            "--input-json",
            json.dumps(
                {
                    "query": query,
                    "context": context,
                    "nuggets": created["nuggets"],
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
    assert assigned.get("reasoning_traces")
    _pretty_print_assign(
        "openrouter reasoning",
        query,
        context,
        assigned["nuggets"],
        scoring_reasoning_traces=_trace_list(assigned.get("reasoning_traces")),
    )
