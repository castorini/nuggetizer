from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_jsonl(file_path: str) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dictionaries."""
    data: list[dict[str, Any]] = []
    with open(file_path, encoding="utf-8") as file_obj:
        for line in file_obj:
            data.append(json.loads(line))
    return data


def append_jsonl_record(file_obj: Any, record: dict[str, Any]) -> None:
    """Write a single JSONL record and flush immediately."""
    file_obj.write(json.dumps(record) + "\n")
    file_obj.flush()


def get_run_id(file_path: str) -> str:
    """Extract a run identifier from a file path."""
    return Path(file_path).stem


def get_processed_values(output_file: str, *keys: str) -> set[Any]:
    """Read the output file and return processed key values or tuples."""
    processed_values: set[Any] = set()
    try:
        with open(output_file, encoding="utf-8") as file_obj:
            for line in file_obj:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if len(keys) == 1:
                    key = keys[0]
                    if key in record:
                        processed_values.add(record[key])
                    continue

                if all(key in record for key in keys):
                    processed_values.add(tuple(record[key] for key in keys))
    except FileNotFoundError:
        pass
    return processed_values
