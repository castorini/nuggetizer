from __future__ import annotations

from typing import Any


def require_keys(record: dict[str, Any], required_keys: list[str]) -> None:
    """Ensure required top-level keys are present."""
    missing_keys = [key for key in required_keys if key not in record]
    if missing_keys:
        raise KeyError(f"Missing required keys: {', '.join(missing_keys)}")
