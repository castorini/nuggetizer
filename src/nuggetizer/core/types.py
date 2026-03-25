from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal


class NuggetMode(Enum):
    ATOMIC = "atomic"
    NOUN_PHRASE = "noun_phrase"
    QUESTION = "question"


class NuggetScoreMode(Enum):
    VITAL_OKAY = "vital_okay"


class NuggetAssignMode(Enum):
    SUPPORT_GRADE_2 = "support_grade_2"
    SUPPORT_GRADE_3 = "support_grade_3"


@dataclass
class Query:
    qid: str
    text: str


@dataclass
class Document:
    docid: str
    segment: str
    title: str | None = None


@dataclass
class Request:
    query: Query
    documents: list[Document]


@dataclass
class Trace:
    """Trace information for debugging and transparency."""

    # Which stage produced this artifact
    component: Literal["creator", "scorer", "assigner"]
    # LLM plumbing
    model: str | None = None
    # e.g., {"temperature": 0.0}
    params: dict[str, Any] = field(default_factory=dict)
    # The messages we sent to the LLM (or the prompt content)
    messages: list[dict[str, str]] | None = None
    # Usage and outputs
    usage: dict[str, Any] | None = None  # e.g., tokens, cost
    raw_output: str | None = None  # raw text as returned
    # Helpful for debugging batched calls
    window_start: int | None = None
    window_end: int | None = None
    # When the call happened (optional)
    timestamp_utc: str | None = None  # ISO8601 string


@dataclass
class BaseNugget:
    """Base class for all nuggets with common fields."""

    text: str
    # Optional metadata
    reasoning: str | None = None
    trace: Trace | None = None


@dataclass
class Nugget(BaseNugget):
    pass


@dataclass
class ScoredNugget(BaseNugget):
    importance: str = "okay"  # e.g., "vital" | "okay" | "failed"


@dataclass
class AssignedNugget(BaseNugget):
    assignment: str = "not_support"  # e.g., "support", "partial_support", "not_support"


@dataclass
class AssignedScoredNugget(ScoredNugget):
    assignment: str = "not_support"  # e.g., "support", "partial_support", "not_support"
