from __future__ import annotations

import asyncio
import logging
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from tqdm import tqdm

from nuggetizer.core.metrics import calculate_global_metrics
from nuggetizer.core.types import AssignedScoredNugget, Request, ScoredNugget
from nuggetizer.models.nuggetizer import Nuggetizer

from .adapters import (
    assign_answer_output_record,
    assign_retrieval_output_record,
    create_output_record,
    metrics_output_records,
    request_from_create_record_with_threshold,
    scored_nuggets_from_record,
)
from .execution import (
    AssignExecutionConfig,
    CreateExecutionConfig,
    build_assign_nuggetizer_kwargs,
    build_create_nuggetizer_kwargs,
)
from .io import append_jsonl_record, get_processed_values, get_run_id, read_jsonl
from .responses import CommandResponse


@dataclass(frozen=True)
class CreateBatchConfig:
    input_file: str
    output_file: str
    execution: CreateExecutionConfig

    @property
    def min_judgment(self) -> int:
        return self.execution.min_judgment

    @property
    def quiet(self) -> bool:
        return self.execution.quiet

    @property
    def output(self) -> str:
        return self.execution.output


@dataclass(frozen=True)
class AssignAnswersBatchConfig:
    nugget_file: str
    answer_file: str
    output_file: str
    execution: AssignExecutionConfig

    @property
    def quiet(self) -> bool:
        return self.execution.quiet

    @property
    def output(self) -> str:
        return self.execution.output


@dataclass(frozen=True)
class AssignRetrievalBatchConfig:
    nugget_file: str
    retrieve_results_file: str
    output_file: str
    execution: AssignExecutionConfig

    @property
    def quiet(self) -> bool:
        return self.execution.quiet

    @property
    def output(self) -> str:
        return self.execution.output


@dataclass(frozen=True)
class MetricsBatchConfig:
    input_file: str
    output_file: str


def _disable_progress(config: object) -> bool:
    quiet = getattr(config, "quiet", False)
    output_format = getattr(config, "output", "text")
    return quiet or output_format in ("json", "jsonl") or not sys.stderr.isatty()


def run_create_batch(
    config: CreateBatchConfig, logger: logging.Logger
) -> CommandResponse:
    return asyncio.run(
        _run_create_batch(
            config, logger, lambda nuggetizer, request: nuggetizer.create(request)
        )
    )


async def async_run_create_batch(
    config: CreateBatchConfig, logger: logging.Logger
) -> CommandResponse:
    async def _runner(nuggetizer: Nuggetizer, request: Request) -> list[ScoredNugget]:
        return await nuggetizer.async_create(request)

    return await _run_create_batch(config, logger, _runner)


def run_assign_answers_batch(
    config: AssignAnswersBatchConfig, logger: logging.Logger
) -> CommandResponse:
    return asyncio.run(
        _run_assign_answers_batch(
            config,
            logger,
            lambda nuggetizer, query, context, nuggets: nuggetizer.assign(
                query, context, nuggets=nuggets
            ),
        )
    )


async def async_run_assign_answers_batch(
    config: AssignAnswersBatchConfig, logger: logging.Logger
) -> CommandResponse:
    async def _runner(
        nuggetizer: Nuggetizer,
        query: str,
        context: str,
        nuggets: list[ScoredNugget],
    ) -> list[AssignedScoredNugget]:
        return await nuggetizer.async_assign(query, context, nuggets=nuggets)

    return await _run_assign_answers_batch(config, logger, _runner)


def run_assign_retrieval_batch(
    config: AssignRetrievalBatchConfig, logger: logging.Logger
) -> CommandResponse:
    return asyncio.run(
        _run_assign_retrieval_batch(
            config,
            logger,
            lambda nuggetizer, query, context, nuggets: nuggetizer.assign(
                query, context, nuggets=nuggets
            ),
        )
    )


async def async_run_assign_retrieval_batch(
    config: AssignRetrievalBatchConfig, logger: logging.Logger
) -> CommandResponse:
    async def _runner(
        nuggetizer: Nuggetizer,
        query: str,
        context: str,
        nuggets: list[ScoredNugget],
    ) -> list[AssignedScoredNugget]:
        return await nuggetizer.async_assign(query, context, nuggets=nuggets)

    return await _run_assign_retrieval_batch(config, logger, _runner)


def run_metrics(
    config: MetricsBatchConfig,
) -> tuple[list[dict[str, Any]], dict[str, float | str]]:
    records = read_jsonl(config.input_file)
    return metrics_output_records(records), calculate_global_metrics(records)


async def _maybe_await(value: Any) -> Any:
    if isinstance(value, Awaitable):
        return await value
    return value


async def _run_create_batch(
    config: CreateBatchConfig,
    logger: logging.Logger,
    runner: Callable[[Nuggetizer, Request], Any],
) -> CommandResponse:
    processed_qids = get_processed_values(config.output_file, "qid")
    logger.info("Found %d already processed records", len(processed_qids))

    nuggetizer = Nuggetizer(**build_create_nuggetizer_kwargs(config.execution))
    input_data = read_jsonl(config.input_file)
    logger.info("Processing %d records", len(input_data))

    generated_count = 0
    skipped_count = 0
    failed_count = 0
    disable = _disable_progress(config)
    with open(config.output_file, "a", encoding="utf-8") as file_obj:
        for i, record in enumerate(
            tqdm(input_data, desc="Creating", file=sys.stderr, disable=disable), 1
        ):
            qid = record["query"]["qid"]
            if qid in processed_qids:
                logger.info("Skipping already processed record %s", qid)
                skipped_count += 1
                continue

            logger.info("Processing record %d/%d", i, len(input_data))
            try:
                request = request_from_create_record_with_threshold(
                    record, min_judgment=config.min_judgment
                )
                scored_nuggets = await _maybe_await(runner(nuggetizer, request))
                append_jsonl_record(
                    file_obj,
                    create_output_record(
                        request,
                        scored_nuggets,
                        creator_reasoning_traces=nuggetizer.get_creator_reasoning_traces(),
                        include_reasoning=config.execution.include_reasoning,
                        include_trace=config.execution.include_trace,
                        redact_prompts=config.execution.redact_prompts,
                    ),
                )
                generated_count += 1
                logger.info(
                    "Generated %d nuggets for record %d", len(scored_nuggets), i
                )
            except Exception as exc:
                logger.error("Error processing record %s: %s", qid, str(exc))
                failed_count += 1

    return CommandResponse(
        command="create",
        metrics={
            "attempted_records": len(input_data),
            "generated_records": generated_count,
            "skipped_records": skipped_count,
            "failed_records": failed_count,
        },
    )


async def _run_assign_answers_batch(
    config: AssignAnswersBatchConfig,
    logger: logging.Logger,
    runner: Callable[[Nuggetizer, str, str, list[ScoredNugget]], Any],
) -> CommandResponse:
    processed_qids = get_processed_values(config.output_file, "qid")
    logger.info("Found %d already processed records", len(processed_qids))

    run_id = get_run_id(config.answer_file)
    logger.info("Using run_id: %s", run_id)
    nuggetizer = Nuggetizer(**build_assign_nuggetizer_kwargs(config.execution))

    nugget_data = read_jsonl(config.nugget_file)
    answer_data = read_jsonl(config.answer_file)
    qid_to_answer_data = {answer["topic_id"]: answer for answer in answer_data}

    assigned_count = 0
    skipped_count = 0
    failed_count = 0
    disable = _disable_progress(config)
    with open(config.output_file, "a", encoding="utf-8") as file_obj:
        for i, nugget_record in enumerate(
            tqdm(nugget_data, desc="Assigning", file=sys.stderr, disable=disable), 1
        ):
            qid = nugget_record["qid"]
            if qid in processed_qids:
                logger.info("Skipping already processed record %s", qid)
                skipped_count += 1
                continue

            logger.info("Processing record pair %d/%d", i, len(nugget_data))
            answer_record = qid_to_answer_data.get(
                qid, {"answer": [], "response_length": 0, "qid": qid}
            )
            try:
                answer_text = " ".join(
                    answer["text"] for answer in answer_record["answer"]
                )
                nuggets = scored_nuggets_from_record(nugget_record)
                logger.info(
                    "Assigning %d nuggets to answer text (length: %d)",
                    len(nuggets),
                    len(answer_text),
                )
                assigned_nuggets = await _maybe_await(
                    runner(
                        nuggetizer,
                        nugget_record.get("query", "N/A"),
                        answer_text,
                        nuggets,
                    )
                )
                append_jsonl_record(
                    file_obj,
                    assign_answer_output_record(
                        answer_record,
                        nugget_record,
                        run_id,
                        assigned_nuggets,
                        include_reasoning=config.execution.include_reasoning,
                        include_trace=config.execution.include_trace,
                        redact_prompts=config.execution.redact_prompts,
                    ),
                )
                assigned_count += 1
            except Exception as exc:
                logger.error("Error processing record %s: %s", qid, str(exc))
                failed_count += 1

    return CommandResponse(
        command="assign",
        metrics={
            "attempted_records": len(nugget_data),
            "assigned_records": assigned_count,
            "skipped_records": skipped_count,
            "failed_records": failed_count,
        },
    )


async def _run_assign_retrieval_batch(
    config: AssignRetrievalBatchConfig,
    logger: logging.Logger,
    runner: Callable[[Nuggetizer, str, str, list[ScoredNugget]], Any],
) -> CommandResponse:
    processed_entries = get_processed_values(config.output_file, "qid", "docid")
    logger.info("Found %d already processed entries", len(processed_entries))

    nuggetizer = Nuggetizer(**build_assign_nuggetizer_kwargs(config.execution))

    nugget_data = read_jsonl(config.nugget_file)
    retrieve_data = read_jsonl(config.retrieve_results_file)
    qid_to_nuggets = {record["qid"]: record for record in nugget_data}

    assigned_count = 0
    skipped_count = 0
    failed_count = 0
    disable = _disable_progress(config)
    with open(config.output_file, "a", encoding="utf-8") as file_obj:
        for retrieve_record in tqdm(
            retrieve_data, desc="Assigning", file=sys.stderr, disable=disable
        ):
            qid = retrieve_record["query"]["qid"]
            nugget_record = qid_to_nuggets.get(qid)
            if not nugget_record:
                logger.warning("No nuggets found for qid %s", qid)
                continue

            for index, candidate in enumerate(retrieve_record["candidates"], start=1):
                processed_key = (qid, candidate["docid"])
                if processed_key in processed_entries:
                    logger.info(
                        "Skipping already processed entry (qid: %s, docid: %s)",
                        qid,
                        candidate["docid"],
                    )
                    skipped_count += 1
                    continue

                logger.info("Processing candidate %d for query %s", index, qid)
                try:
                    nuggets = scored_nuggets_from_record(nugget_record)
                    context = candidate["doc"]["segment"]
                    logger.info(
                        "Assigning %d nuggets to candidate text (length: %d)",
                        len(nuggets),
                        len(context),
                    )
                    assigned_nuggets = await _maybe_await(
                        runner(
                            nuggetizer,
                            nugget_record.get("query", "N/A"),
                            context,
                            nuggets,
                        )
                    )
                    append_jsonl_record(
                        file_obj,
                        assign_retrieval_output_record(
                            nugget_record,
                            candidate,
                            assigned_nuggets,
                            include_reasoning=config.execution.include_reasoning,
                            include_trace=config.execution.include_trace,
                            redact_prompts=config.execution.redact_prompts,
                        ),
                    )
                    assigned_count += 1
                except Exception as exc:
                    logger.error(
                        "Error processing candidate for qid %s, docid %s: %s",
                        qid,
                        candidate["docid"],
                        str(exc),
                    )
                    failed_count += 1

    return CommandResponse(
        command="assign",
        metrics={
            "assigned_records": assigned_count,
            "skipped_records": skipped_count,
            "failed_records": failed_count,
        },
    )
