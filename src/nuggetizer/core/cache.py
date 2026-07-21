from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


class AttrDict(dict[str, Any]):
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def cache_is_enabled():
    return os.getenv("CACHE_DIR") is not None and os.getenv("CACHE_DIR")


def create_cached_chat_completion(
    client: Any, completion_params: dict[str, Any]
) -> Any:
    import sqlite3
    cache_dir = os.getenv("CACHE_DIR")
    if not cache_dir:
        return client.chat.completions.create(**completion_params)

    request_json = json.dumps(_to_jsonable(completion_params), sort_keys=True)
    database_path = _get_database_path(cache_dir)
    cache_key = _build_cache_key(
        {
            "model": completion_params.get("model"),
            "request": request_json,
        }
    )
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_completions_cache (
                cache_key TEXT PRIMARY KEY,
                request_json TEXT NOT NULL,
                response_json TEXT NOT NULL
            )
            """
        )
        row = connection.execute(
            "SELECT response_json FROM chat_completions_cache WHERE cache_key = ?",
            (cache_key,),
        ).fetchone()
        if row is not None:
            return _from_jsonable(json.loads(row[0]))

        completion = client.chat.completions.create(**completion_params)
        connection.execute(
            """
            INSERT OR REPLACE INTO chat_completions_cache (
                cache_key,
                request_json,
                response_json
            ) VALUES (?, ?, ?)
            """,
            (
                cache_key,
                request_json,
                json.dumps(_to_jsonable(completion), sort_keys=True),
            ),
        )
        connection.commit()
    return completion


def _get_database_path(cache_dir: str) -> Path:
    path = Path(cache_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path / "llm_cache.sqlite"


def _build_cache_key(payload: dict[str, Any]) -> str:
    normalized_payload = json.dumps(_to_jsonable(payload), sort_keys=True)
    return hashlib.sha256(normalized_payload.encode("utf-8")).hexdigest()


def _to_jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return _to_jsonable(value.model_dump(mode="json"))
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    if hasattr(value, "__dict__"):
        return {key: _to_jsonable(item) for key, item in vars(value).items()}
    return str(value)


def _from_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return AttrDict({key: _from_jsonable(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_from_jsonable(item) for item in value]
    return value
