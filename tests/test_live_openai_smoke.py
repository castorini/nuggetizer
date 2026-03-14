from __future__ import annotations

import json
import os

import pytest

from nuggetizer.cli.main import main


pytestmark = pytest.mark.skipif(
    os.getenv("NUGGETIZER_LIVE_OPENAI_SMOKE") != "1",
    reason="Set NUGGETIZER_LIVE_OPENAI_SMOKE=1 to run live OpenAI smoke tests.",
)


def test_direct_create_openai_smoke(capsys: pytest.CaptureFixture[str]) -> None:
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("OPENROUTER_API_KEY"):
        pytest.skip("OPENAI_API_KEY or OPENROUTER_API_KEY is required.")

    model = os.getenv("NUGGETIZER_LIVE_OPENAI_MODEL", "gpt-4o-mini")
    exit_code = main(
        [
            "create",
            "--model",
            model,
            "--input-json",
            json.dumps(
                {
                    "query": "What is Python used for?",
                    "candidates": [
                        "Python is widely used for web development and data analysis."
                    ],
                }
            ),
            "--output",
            "json",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == "create"
    assert payload["status"] == "success"
    nuggets = payload["artifacts"][0]["data"]["nuggets"]
    assert nuggets
    assert "text" in nuggets[0]
