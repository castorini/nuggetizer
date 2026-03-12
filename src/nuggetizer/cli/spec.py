from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CommandSpec:
    """Internal command metadata used by scripts and the future public CLI."""

    name: str
    description: str


CREATE_COMMAND = CommandSpec(
    name="create", description="Extract and score nuggets from input JSONL file"
)
ASSIGN_COMMAND = CommandSpec(
    name="assign", description="Assign nuggets to answer text from input JSONL files"
)
ASSIGN_RETRIEVAL_COMMAND = CommandSpec(
    name="assign-retrieval",
    description="Assign nuggets to retrieved candidate segments",
)
METRICS_COMMAND = CommandSpec(
    name="metrics", description="Calculate metrics for nugget assignments"
)
