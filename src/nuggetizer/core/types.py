from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union


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
    title: Optional[str] = None


@dataclass
class Request:
    query: Query
    documents: List[Document]


@dataclass
class Trace:
    """Trace information for debugging and transparency."""
    # Which stage produced this artifact
    component: Literal["creator", "scorer", "assigner"]
    # LLM plumbing
    model: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)  # e.g., {"temperature": 0.0}
    # The messages we sent to the LLM (or the prompt content)
    messages: Optional[List[Dict[str, str]]] = None
    # Usage and outputs
    usage: Optional[Dict[str, Any]] = None               # e.g., tokens, cost
    raw_output: Optional[str] = None                     # raw text as returned
    # Helpful for debugging batched calls
    window_start: Optional[int] = None
    window_end: Optional[int] = None
    # When the call happened (optional)
    timestamp_utc: Optional[str] = None                  # ISO8601 string

@dataclass
class BaseNugget:
    """Base class for all nuggets with common fields."""
    text: str
    # Optional metadata
    reasoning: Optional[str] = None
    trace: Optional[Trace] = None


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