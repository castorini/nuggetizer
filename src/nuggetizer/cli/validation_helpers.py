from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

from .io import read_jsonl


def doctor_report() -> dict[str, Any]:
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
        resolved_missing_env = missing_env or []
        resolved_missing_deps = missing_deps or []
        return {
            "status": "ready"
            if ready
            else (
                "missing_env"
                if resolved_missing_env
                else ("missing_dependency" if resolved_missing_deps else "blocked")
            ),
            "missing_env": resolved_missing_env,
            "missing_dependencies": resolved_missing_deps,
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
    from .normalize import direct_create_record

    direct_create_record(payload)
    return {"valid": True, "record_count": 1}


def validate_assign_input(payload: dict[str, Any]) -> dict[str, Any]:
    from .normalize import direct_assign_inputs, joined_assign_batch_records

    try:
        direct_assign_inputs(payload)
    except ValueError:
        batch_records = joined_assign_batch_records(payload)
        return {"valid": True, "record_count": len(batch_records)}
    return {"valid": True, "record_count": 1}


def validate_create_batch_file(path: str) -> dict[str, Any]:
    records = read_jsonl(path)
    valid = all("query" in record and "candidates" in record for record in records)
    return {"valid": valid, "record_count": len(records)}


def validate_assign_batch_files(
    nuggets_path: str, contexts_path: str
) -> dict[str, Any]:
    nugget_records = read_jsonl(nuggets_path)
    context_records = read_jsonl(contexts_path)
    valid = bool(nugget_records) and bool(context_records)
    return {
        "valid": valid,
        "nugget_record_count": len(nugget_records),
        "context_record_count": len(context_records),
    }
