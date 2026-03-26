from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

from .io import read_jsonl

COMMAND_DESCRIPTIONS: dict[str, dict[str, Any]] = {
    "create": {
        "summary": "Create and score nuggets from either batch JSONL or direct JSON input.",
        "execution_mode_default": "sync",
        "inspection_safe": False,
        "examples": [
            "nuggetizer create --input-file pool.jsonl --output-file nuggets.jsonl",
            (
                "nuggetizer create --input-json "
                '\'{"query":"What is Python used for?","candidates":["Python is used for web development."]}\' '
                "--output json"
            ),
            (
                'curl -s "http://127.0.0.1:8081/v1/msmarco-v1-passage/search?query=q" '
                "| curl -s -X POST http://127.0.0.1:8085/v1/create "
                '-H "content-type: application/json" --data-binary @- | jq'
            ),
            (
                'curl -s "http://127.0.0.1:8081/v1/msmarco-v1-passage/search?query=q" '
                "| curl -s -X POST http://127.0.0.1:8082/v1/rerank "
                '-H "content-type: application/json" --data-binary @- '
                "| curl -s -X POST http://127.0.0.1:8085/v1/create "
                '-H "content-type: application/json" --data-binary @- | jq'
            ),
        ],
        "direct_input": {
            "ids_optional": True,
            "shape": {
                "query": "string | {text: string, qid?: string}",
                "candidates": [
                    "string | {text: string, docid?: string} | "
                    "{doc: string | {segment: string} | {contents: string}, docid?: string}"
                ],
            },
        },
        "batch_input": "JSONL records with query.qid/query.text and candidates[].doc.segment",
    },
    "assign": {
        "summary": "Assign nuggets to either a direct context string or batch answers/retrieval records.",
        "execution_mode_default": "sync",
        "inspection_safe": False,
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
        "script_wrappers": [
            "scripts/assign_nuggets.py",
            "scripts/assign_nuggets_retrieve_results.py",
        ],
    },
    "metrics": {
        "summary": "Calculate per-query and global nugget metrics from assignment JSONL.",
        "inspection_safe": False,
        "examples": [
            "nuggetizer metrics --input-file assignments.jsonl --output-file metrics.jsonl"
        ],
    },
    "serve": {
        "summary": "Start a FastAPI server for direct nugget creation and assignment requests.",
        "examples": [
            "nuggetizer serve --port 8085",
            (
                "curl -X POST http://127.0.0.1:8085/v1/create "
                "-H 'content-type: application/json' "
                '-d \'{"query":"q","candidates":["p"]}\''
            ),
            (
                'curl -s "http://127.0.0.1:8081/v1/msmarco-v1-passage/search?query=q" '
                "| curl -s -X POST http://127.0.0.1:8085/v1/create "
                '-H "content-type: application/json" --data-binary @- | jq'
            ),
            (
                'curl -s "http://127.0.0.1:8081/v1/msmarco-v1-passage/search?query=q" '
                "| curl -s -X POST http://127.0.0.1:8082/v1/rerank "
                '-H "content-type: application/json" --data-binary @- '
                "| curl -s -X POST http://127.0.0.1:8085/v1/create "
                '-H "content-type: application/json" --data-binary @- | jq'
            ),
        ],
        "routes": ["GET /healthz", "POST /v1/create", "POST /v1/assign"],
        "inspection_safe": True,
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
        "inspection_safe": True,
    },
    "prompt": {
        "summary": "Inspect built-in Nuggetizer prompt templates.",
        "examples": [
            "nuggetizer prompt list",
            "nuggetizer prompt show create",
            'nuggetizer prompt render create --input-json \'{"query":"q","candidates":["p"]}\'',
            "nuggetizer prompt show assign --assign-mode support_grade_2 --output json",
        ],
        "inspection_safe": True,
        "subcommands": ["list", "show", "render"],
    },
    "describe": {
        "summary": "Inspect structured metadata for a public Nuggetizer command.",
        "inspection_safe": True,
    },
    "schema": {
        "summary": "Print JSON schemas for supported Nuggetizer inputs, outputs, and envelopes.",
        "inspection_safe": True,
    },
    "doctor": {
        "summary": "Report environment and backend readiness for the packaged Nuggetizer CLI.",
        "inspection_safe": True,
    },
    "validate": {
        "summary": "Validate direct JSON input or batch JSONL inputs without running models.",
        "targets": ["create", "assign"],
        "inspection_safe": True,
    },
}


SCHEMAS: dict[str, dict[str, Any]] = {
    "create-direct-input": {
        "type": "object",
        "required": ["query", "candidates"],
        "properties": {
            "query": {
                "oneOf": [
                    {"type": "string"},
                    {
                        "type": "object",
                        "required": ["text"],
                        "properties": {
                            "qid": {"type": "string"},
                            "text": {"type": "string"},
                        },
                    },
                ]
            },
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
                        {
                            "type": "object",
                            "required": ["doc"],
                            "properties": {
                                "docid": {"type": "string"},
                                "doc": {
                                    "oneOf": [
                                        {"type": "string"},
                                        {
                                            "type": "object",
                                            "properties": {
                                                "segment": {"type": "string"},
                                                "contents": {"type": "string"},
                                            },
                                        },
                                    ]
                                },
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
    "prompt-catalog": {
        "type": "array",
        "items": {
            "type": "object",
            "required": ["target", "template_name", "assign_mode", "template"],
        },
    },
    "prompt-template": {
        "type": "object",
        "required": ["target", "template_name", "assign_mode", "template"],
    },
    "rendered-prompt": {
        "type": "object",
        "required": ["target", "template_name", "assign_mode", "messages", "inputs"],
    },
    "doctor-output": {
        "type": "object",
        "required": [
            "python_version",
            "python_ok",
            "env_file_present",
            "backend_readiness",
            "command_readiness",
            "overall_status",
        ],
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
    python_ok = sys.version_info >= (3, 11)
    openai_ready = bool(os.getenv("OPENAI_API_KEY"))
    openrouter_ready = bool(os.getenv("OPENROUTER_API_KEY"))
    azure_ready = bool(
        os.getenv("AZURE_OPENAI_API_BASE")
        and os.getenv("AZURE_OPENAI_API_VERSION")
        and os.getenv("AZURE_OPENAI_API_KEY")
    )
    openai_dep_ready = importlib.util.find_spec("openai") is not None
    fastapi_dep_ready = importlib.util.find_spec("fastapi") is not None
    uvicorn_dep_ready = importlib.util.find_spec("uvicorn") is not None

    def status(
        *,
        ready: bool,
        missing_env: list[str] | None = None,
        missing_deps: list[str] | None = None,
    ) -> dict[str, Any]:
        missing_env = missing_env or []
        missing_deps = missing_deps or []
        return {
            "status": "ready"
            if ready
            else (
                "missing_env"
                if missing_env
                else ("missing_dependency" if missing_deps else "blocked")
            ),
            "missing_env": missing_env,
            "missing_dependencies": missing_deps,
        }

    backend_readiness = {
        "openai": status(
            ready=python_ok and openai_ready and openai_dep_ready,
            missing_env=[] if openai_ready else ["OPENAI_API_KEY"],
            missing_deps=[] if openai_dep_ready else ["openai"],
        ),
        "openrouter": status(
            ready=python_ok and openrouter_ready and openai_dep_ready,
            missing_env=[] if openrouter_ready else ["OPENROUTER_API_KEY"],
            missing_deps=[] if openai_dep_ready else ["openai"],
        ),
        "azure_openai": status(
            ready=python_ok and azure_ready and openai_dep_ready,
            missing_env=[]
            if azure_ready
            else [
                "AZURE_OPENAI_API_BASE",
                "AZURE_OPENAI_API_VERSION",
                "AZURE_OPENAI_API_KEY",
            ],
            missing_deps=[] if openai_dep_ready else ["openai"],
        ),
        "vllm_local": status(
            ready=python_ok and openai_dep_ready,
            missing_deps=[] if openai_dep_ready else ["openai"],
        ),
    }
    command_readiness = {
        command: status(ready=python_ok)
        for command in [
            "create",
            "assign",
            "metrics",
            "view",
            "describe",
            "schema",
            "doctor",
            "validate",
        ]
    }
    command_readiness["serve"] = status(
        ready=python_ok and fastapi_dep_ready and uvicorn_dep_ready,
        missing_deps=[
            dependency
            for dependency, available in (
                ("fastapi", fastapi_dep_ready),
                ("uvicorn", uvicorn_dep_ready),
            )
            if not available
        ],
    )
    return {
        "python_version": sys.version.split()[0],
        "python_ok": python_ok,
        "package_importable": True,
        "env_file_present": env_path.exists(),
        "provider_keys": {
            "openai": openai_ready,
            "openrouter": openrouter_ready,
            "azure": azure_ready,
        },
        "backend_readiness": backend_readiness,
        "command_readiness": command_readiness,
        "optional_dependencies": {
            "openai": openai_dep_ready,
            "fastapi": fastapi_dep_ready,
            "uvicorn": uvicorn_dep_ready,
        },
        "overall_status": "ready" if python_ok else "blocked",
    }


def validate_create_input(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate a direct create payload."""
    from .normalize import direct_create_record

    direct_create_record(payload)
    return {"valid": True, "record_count": 1}


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
