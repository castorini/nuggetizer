from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CommandResponse:
    """Internal response scaffold for the future structured CLI envelope."""

    command: str
    status: str = "success"
    metrics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
