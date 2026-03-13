from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


ANSI_CODES = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "cyan": "\033[36m",
    "green": "\033[32m",
    "yellow": "\033[33m",
}

CREATE_KEYS = {"query", "qid", "nuggets"}
ASSIGN_ANSWERS_KEYS = {
    "query",
    "qid",
    "answer_text",
    "response_length",
    "run_id",
    "nuggets",
}
ASSIGN_RETRIEVAL_KEYS = {"text", "qid", "candidate_text", "docid", "nuggets"}
METRICS_KEYS = {
    "qid",
    "strict_vital_score",
    "strict_all_score",
    "vital_score",
    "all_score",
}


class ViewError(ValueError):
    """Raised when a file cannot be rendered as a supported Nuggetizer artifact."""


def _color_enabled(color: str) -> bool:
    if color == "always":
        return True
    if color == "never":
        return False
    return sys.stdout.isatty()


def _style(text: str, color: str, enabled: bool) -> str:
    if not enabled:
        return text
    return f"{ANSI_CODES[color]}{text}{ANSI_CODES['reset']}"


def _truncate(text: str, limit: int = 140) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[: limit - 3]}..."


def load_records(path: str) -> list[dict[str, Any]]:
    file_path = Path(path)
    try:
        raw_text = file_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ViewError(f"path does not exist: {path}") from exc

    if not raw_text.strip():
        raise ViewError(f"file is empty: {path}")

    try:
        if file_path.suffix == ".json":
            payload = json.loads(raw_text)
            if isinstance(payload, dict):
                return [payload]
            if isinstance(payload, list):
                return payload
            raise ViewError(f"unsupported JSON payload in {path}")
        records = [
            json.loads(line)
            for line in raw_text.splitlines()
            if line.strip()
        ]
    except json.JSONDecodeError as exc:
        raise ViewError(f"file is not valid JSON/JSONL: {path}") from exc

    if not records:
        raise ViewError(f"file is empty: {path}")
    return records


def detect_artifact_type(
    records: list[dict[str, Any]], requested_type: str | None
) -> str:
    supported_types = {
        "create-output",
        "assign-output-answers",
        "assign-output-retrieval",
        "metrics-output",
    }
    if requested_type is not None:
        if requested_type not in supported_types:
            raise ViewError(
                "unsupported --type for nuggetizer view; expected one of "
                "create-output, assign-output-answers, assign-output-retrieval, metrics-output"
            )
        return requested_type

    first_record = records[0]
    if METRICS_KEYS.issubset(first_record.keys()):
        return "metrics-output"
    if ASSIGN_ANSWERS_KEYS.issubset(first_record.keys()):
        return "assign-output-answers"
    if ASSIGN_RETRIEVAL_KEYS.issubset(first_record.keys()):
        return "assign-output-retrieval"
    if CREATE_KEYS.issubset(first_record.keys()):
        return "create-output"
    raise ViewError(
        "could not detect Nuggetizer artifact type; use --type to specify it"
    )


def _assignment_counts(nuggets: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"support": 0, "partial_support": 0, "not_support": 0}
    for nugget in nuggets:
        assignment = str(nugget.get("assignment", ""))
        if assignment in counts:
            counts[assignment] += 1
    return counts


def build_view_summary(
    path: str, records: list[dict[str, Any]], artifact_type: str, *, record_limit: int
) -> dict[str, Any]:
    limit = max(record_limit, 0)
    sampled_records: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"record_count": len(records)}

    if artifact_type == "create-output":
        summary["total_nuggets"] = sum(len(record.get("nuggets", [])) for record in records)
        for record in records[:limit]:
            sampled_records.append(
                {
                    "qid": record["qid"],
                    "query": _truncate(str(record["query"]), 150),
                    "nugget_count": len(record["nuggets"]),
                    "nuggets": [
                        {
                            "text": _truncate(str(nugget.get("text", "")), 120),
                            "importance": nugget.get("importance"),
                        }
                        for nugget in record["nuggets"][:5]
                    ],
                }
            )
    elif artifact_type == "assign-output-answers":
        summary["run_ids"] = sorted(
            {str(record.get("run_id", "")) for record in records if record.get("run_id")}
        )
        for record in records[:limit]:
            sampled_records.append(
                {
                    "qid": record["qid"],
                    "query": _truncate(str(record["query"]), 140),
                    "answer_text": _truncate(str(record["answer_text"]), 180),
                    "assignment_counts": _assignment_counts(record["nuggets"]),
                    "nuggets": [
                        {
                            "text": _truncate(str(nugget.get("text", "")), 110),
                            "importance": nugget.get("importance"),
                            "assignment": nugget.get("assignment"),
                        }
                        for nugget in record["nuggets"][:5]
                    ],
                }
            )
    elif artifact_type == "assign-output-retrieval":
        for record in records[:limit]:
            sampled_records.append(
                {
                    "qid": record["qid"],
                    "text": _truncate(str(record["text"]), 140),
                    "docid": record["docid"],
                    "candidate_text": _truncate(str(record["candidate_text"]), 160),
                    "assignment_counts": _assignment_counts(record["nuggets"]),
                    "nuggets": [
                        {
                            "text": _truncate(str(nugget.get("text", "")), 110),
                            "importance": nugget.get("importance"),
                            "assignment": nugget.get("assignment"),
                        }
                        for nugget in record["nuggets"][:5]
                    ],
                }
            )
    else:
        aggregate = None
        per_query_records = records
        if records and str(records[-1].get("qid")) == "all":
            aggregate = records[-1]
            per_query_records = records[:-1]
        summary["record_count"] = len(per_query_records)
        summary["has_global_metrics"] = aggregate is not None
        summary["global_metrics"] = aggregate
        for record in per_query_records[:limit]:
            sampled_records.append(
                {
                    "qid": record["qid"],
                    "strict_vital_score": record["strict_vital_score"],
                    "strict_all_score": record["strict_all_score"],
                    "vital_score": record["vital_score"],
                    "all_score": record["all_score"],
                }
            )

    return {
        "path": str(Path(path)),
        "artifact_type": artifact_type,
        "summary": summary,
        "sampled_records": sampled_records,
        "requested_records": limit,
    }


def render_view_summary(view: dict[str, Any], *, color: str) -> str:
    enabled = _color_enabled(color)
    summary = view["summary"]
    lines = [
        _style("Nuggetizer View", "bold", enabled),
        f"path: {view['path']}",
        f"type: {view['artifact_type']}",
        f"records: {summary['record_count']}",
    ]
    if "total_nuggets" in summary:
        lines.append(f"total_nuggets: {summary['total_nuggets']}")
    if summary.get("run_ids"):
        lines.append(f"run_ids: {', '.join(summary['run_ids'])}")
    if summary.get("global_metrics"):
        aggregate = summary["global_metrics"]
        lines.append(
            "global_metrics: "
            f"strict_vital={aggregate['strict_vital_score']:.3f}, "
            f"strict_all={aggregate['strict_all_score']:.3f}, "
            f"vital={aggregate['vital_score']:.3f}, "
            f"all={aggregate['all_score']:.3f}"
        )

    for index, record in enumerate(view["sampled_records"], start=1):
        lines.append("")
        if view["artifact_type"] == "metrics-output":
            lines.append(f"[{index}] qid={_style(str(record['qid']), 'cyan', enabled)}")
            lines.append(
                "scores: "
                f"strict_vital={record['strict_vital_score']:.3f}, "
                f"strict_all={record['strict_all_score']:.3f}, "
                f"vital={record['vital_score']:.3f}, "
                f"all={record['all_score']:.3f}"
            )
            continue

        if view["artifact_type"] == "assign-output-retrieval":
            lines.append(
                f"[{index}] qid={_style(str(record['qid']), 'cyan', enabled)} "
                f"docid={_style(str(record['docid']), 'green', enabled)}"
            )
            lines.append(f"query: {record['text']}")
            lines.append(f"candidate: {record['candidate_text']}")
        else:
            lines.append(f"[{index}] qid={_style(str(record['qid']), 'cyan', enabled)}")
            lines.append(f"query: {record['query']}")
            if "answer_text" in record:
                lines.append(f"answer: {record['answer_text']}")

        if "assignment_counts" in record:
            counts = record["assignment_counts"]
            lines.append(
                "assignments: "
                f"support={counts['support']}, "
                f"partial_support={counts['partial_support']}, "
                f"not_support={counts['not_support']}"
            )
        else:
            lines.append(f"nuggets: {record['nugget_count']}")

        for nugget_index, nugget in enumerate(record["nuggets"], start=1):
            prefix = f"{nugget_index}. {nugget['importance']}"
            if "assignment" in nugget:
                prefix += f"/{nugget['assignment']}"
            lines.append(f"{prefix}: {nugget['text']}")
    return "\n".join(lines)
