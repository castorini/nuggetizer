from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from .io import read_jsonl


COMMAND_DESCRIPTIONS: dict[str, dict[str, Any]] = {
    "create": {
        "summary": "Create and score nuggets from either batch JSONL or direct JSON input.",
        "execution_mode_default": "sync",
        "examples": [
            "nuggetizer create --input-file pool.jsonl --output-file nuggets.jsonl",
            (
                "nuggetizer create --input-json "
                '\'{"query":"What is Python used for?","candidates":["Python is used for web development."]}\' '
                "--output json"
            ),
        ],
        "direct_input": {
            "ids_optional": True,
            "shape": {
                "query": "string",
                "candidates": ["string | {text: string, docid?: string}"],
            },
        },
        "batch_input": "JSONL records with query.qid/query.text and candidates[].doc.segment",
    },
    "assign": {
        "summary": "Assign nuggets to either a direct context string or batch answers/retrieval records.",
        "execution_mode_default": "sync",
        "examples": [
            (
                "nuggetizer assign --input-kind answers --nuggets nuggets.jsonl "
                "--contexts answers.jsonl --output-file assignments.jsonl"
            ),
            (
                "nuggetizer assign --input-json "
                '\'{"query":"What is Python used for?","context":"Python is used for web development.","nuggets":[{"text":"Python is used for web development.","importance":"vital"}]}\' '
                "--output json"
            ),
        ],
        "direct_input": {
            "ids_optional": True,
            "shape": {
                "query": "string",
                "context": "string",
                "nuggets": [{"text": "string", "importance": "string"}],
            },
        },
        "batch_input_kinds": ["answers", "retrieval"],
        "legacy_mappings": [
            "scripts/assign_nuggets.py",
            "scripts/assign_nuggets_retrieve_results.py",
        ],
    },
    "metrics": {
        "summary": "Calculate per-query and global nugget metrics from assignment JSONL.",
        "examples": [
            "nuggetizer metrics --input-file assignments.jsonl --output-file metrics.jsonl"
        ],
    },
    "view": {
        "summary": "Inspect Nuggetizer artifact files with a human-readable preview.",
        "examples": [
            "nuggetizer view nuggets.jsonl",
            "nuggetizer view assignments.jsonl --records 1",
        ],
        "supported_types": [
            "create-output",
            "assign-output-answers",
            "assign-output-retrieval",
            "metrics-output",
        ],
    },
}


SCHEMAS: dict[str, dict[str, Any]] = {
    "create-direct-input": {
        "type": "object",
        "required": ["query", "candidates"],
        "properties": {
            "query": {"type": "string"},
            "candidates": {
                "type": "array",
                "items": {
                    "oneOf": [
                        {"type": "string"},
                        {
                            "type": "object",
                            "required": ["text"],
                            "properties": {
                                "text": {"type": "string"},
                                "docid": {"type": "string"},
                            },
                        },
                    ]
                },
            },
        },
    },
    "create-batch-input-record": {
        "type": "object",
        "required": ["query", "candidates"],
        "properties": {
            "query": {
                "type": "object",
                "required": ["qid", "text"],
                "properties": {
                    "qid": {"type": "string"},
                    "text": {"type": "string"},
                },
            },
            "candidates": {"type": "array"},
        },
    },
    "create-output": {
        "type": "object",
        "required": ["query", "qid", "nuggets"],
        "properties": {
            "query": {"type": "string"},
            "qid": {"type": "string"},
            "nuggets": {"type": "array"},
        },
    },
    "assign-direct-input": {
        "type": "object",
        "required": ["query", "context", "nuggets"],
        "properties": {
            "query": {"type": "string"},
            "context": {"type": "string"},
            "nuggets": {"type": "array"},
        },
    },
    "assign-contexts-answers-input": {
        "type": "object",
        "required": ["topic_id", "answer"],
        "properties": {
            "topic_id": {"type": "string"},
            "answer": {"type": "array"},
        },
    },
    "assign-contexts-retrieval-input": {
        "type": "object",
        "required": ["query", "candidates"],
        "properties": {
            "query": {"type": "object"},
            "candidates": {"type": "array"},
        },
    },
    "assign-output-answers": {
        "type": "object",
        "required": [
            "query",
            "qid",
            "answer_text",
            "response_length",
            "run_id",
            "nuggets",
        ],
    },
    "assign-output-retrieval": {
        "type": "object",
        "required": ["text", "qid", "candidate_text", "docid", "nuggets"],
    },
    "metrics-output": {
        "type": "object",
        "required": [
            "qid",
            "strict_vital_score",
            "strict_all_score",
            "vital_score",
            "all_score",
        ],
    },
    "view-summary": {
        "type": "object",
        "required": ["path", "artifact_type", "summary", "sampled_records"],
    },
    "cli-envelope": {
        "type": "object",
        "required": [
            "schema_version",
            "repo",
            "command",
            "mode",
            "status",
            "exit_code",
            "inputs",
            "resolved",
            "artifacts",
            "validation",
            "metrics",
            "warnings",
            "errors",
        ],
    },
}


def doctor_report() -> dict[str, Any]:
    """Return a lightweight environment readiness report."""
    env_path = Path(".env")
    return {
        "python_version": sys.version.split()[0],
        "python_ok": sys.version_info >= (3, 11),
        "package_importable": True,
        "env_file_present": env_path.exists(),
        "provider_keys": {
            "openai": bool(os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_AI_API_KEY")),
            "openrouter": bool(os.getenv("OPENROUTER_API_KEY")),
            "azure": bool(
                os.getenv("AZURE_OPENAI_API_BASE")
                and os.getenv("AZURE_OPENAI_API_VERSION")
                and os.getenv("AZURE_OPENAI_API_KEY")
            ),
        },
    }


def validate_create_input(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate a direct create payload."""
    return {
        "valid": isinstance(payload.get("query"), str)
        and isinstance(payload.get("candidates"), list),
        "record_count": 1,
    }


def validate_assign_input(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate a direct assign payload."""
    return {
        "valid": all(key in payload for key in ["query", "context", "nuggets"]),
        "record_count": 1,
    }


def validate_create_batch_file(path: str) -> dict[str, Any]:
    """Validate a batch create JSONL file."""
    records = read_jsonl(path)
    valid = all("query" in record and "candidates" in record for record in records)
    return {"valid": valid, "record_count": len(records)}


def validate_assign_batch_files(
    nuggets_path: str, contexts_path: str
) -> dict[str, Any]:
    """Validate a batch assign input pair."""
    nugget_records = read_jsonl(nuggets_path)
    context_records = read_jsonl(contexts_path)
    valid = bool(nugget_records) and bool(context_records)
    return {
        "valid": valid,
        "nugget_record_count": len(nugget_records),
        "context_record_count": len(context_records),
    }
