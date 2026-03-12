from __future__ import annotations

import logging
from typing import Any

from nuggetizer.core.metrics import calculate_global_metrics
from nuggetizer.models.nuggetizer import Nuggetizer

from .adapters import (
    assign_answer_output_record,
    assign_retrieval_output_record,
    create_output_record,
    metrics_output_records,
    request_from_create_record,
    scored_nuggets_from_record,
)
from .io import append_jsonl_record, get_processed_values, get_run_id, read_jsonl
from .responses import CommandResponse


def build_create_nuggetizer_kwargs(args: Any) -> dict[str, Any]:
    """Build Nuggetizer kwargs for create scripts."""
    nuggetizer_kwargs: dict[str, Any] = {
        "log_level": args.log_level,
        "use_azure_openai": args.use_azure_openai,
    }
    if args.creator_model or args.scorer_model:
        nuggetizer_kwargs.update(
            {
                "creator_model": args.creator_model or args.model,
                "scorer_model": args.scorer_model or args.model,
            }
        )
    else:
        nuggetizer_kwargs["model"] = args.model
    if args.window_size:
        nuggetizer_kwargs["window_size"] = args.window_size
    if args.max_nuggets:
        nuggetizer_kwargs["max_nuggets"] = args.max_nuggets
    return nuggetizer_kwargs


def run_create_batch(args: Any, logger: logging.Logger) -> CommandResponse:
    """Run the legacy batch create flow."""
    processed_qids = get_processed_values(args.output_file, "qid")
    logger.info("Found %d already processed records", len(processed_qids))

    nuggetizer = Nuggetizer(**build_create_nuggetizer_kwargs(args))
    input_data = read_jsonl(args.input_file)
    logger.info("Processing %d records", len(input_data))

    generated_count = 0
    skipped_count = 0
    failed_count = 0
    with open(args.output_file, "a", encoding="utf-8") as file_obj:
        for i, record in enumerate(input_data, 1):
            qid = record["query"]["qid"]
            if qid in processed_qids:
                logger.info("Skipping already processed record %s", qid)
                skipped_count += 1
                continue

            logger.info("Processing record %d/%d", i, len(input_data))
            try:
                request = request_from_create_record(record)
                scored_nuggets = nuggetizer.create(request)
                append_jsonl_record(
                    file_obj, create_output_record(request, scored_nuggets)
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


def run_assign_answers_batch(args: Any, logger: logging.Logger) -> CommandResponse:
    """Run the legacy answer-assignment flow."""
    processed_qids = get_processed_values(args.output_file, "qid")
    logger.info("Found %d already processed records", len(processed_qids))

    run_id = get_run_id(args.answer_file)
    logger.info("Using run_id: %s", run_id)
    nuggetizer = Nuggetizer(
        assigner_model=args.model,
        log_level=args.log_level,
        use_azure_openai=args.use_azure_openai,
    )

    nugget_data = read_jsonl(args.nugget_file)
    answer_data = read_jsonl(args.answer_file)
    qid_to_answer_data = {answer["topic_id"]: answer for answer in answer_data}

    assigned_count = 0
    skipped_count = 0
    failed_count = 0
    with open(args.output_file, "a", encoding="utf-8") as file_obj:
        for i, nugget_record in enumerate(nugget_data, 1):
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
                assigned_nuggets = nuggetizer.assign(
                    nugget_record.get("query", "N/A"), answer_text, nuggets=nuggets
                )
                append_jsonl_record(
                    file_obj,
                    assign_answer_output_record(
                        answer_record, nugget_record, run_id, assigned_nuggets
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


def run_assign_retrieval_batch(args: Any, logger: logging.Logger) -> CommandResponse:
    """Run the legacy retrieval-assignment flow."""
    processed_entries = get_processed_values(args.output_file, "qid", "docid")
    logger.info("Found %d already processed entries", len(processed_entries))

    nuggetizer = Nuggetizer(
        assigner_model=args.model,
        log_level=args.log_level,
        use_azure_openai=args.use_azure_openai,
    )

    nugget_data = read_jsonl(args.nugget_file)
    retrieve_data = read_jsonl(args.retrieve_results_file)
    qid_to_nuggets = {record["qid"]: record for record in nugget_data}

    assigned_count = 0
    skipped_count = 0
    failed_count = 0
    with open(args.output_file, "a", encoding="utf-8") as file_obj:
        for retrieve_record in retrieve_data:
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
                    assigned_nuggets = nuggetizer.assign(
                        nugget_record.get("query", "N/A"),
                        context,
                        nuggets=nuggets,
                    )
                    append_jsonl_record(
                        file_obj,
                        assign_retrieval_output_record(
                            nugget_record, candidate, assigned_nuggets
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
        command="assign-retrieval",
        metrics={
            "assigned_records": assigned_count,
            "skipped_records": skipped_count,
            "failed_records": failed_count,
        },
    )


def run_metrics(args: Any) -> tuple[list[dict[str, Any]], dict[str, float | str]]:
    """Run the legacy metrics flow."""
    records = read_jsonl(args.input_file)
    return metrics_output_records(records), calculate_global_metrics(records)


async def async_run_create_batch(args: Any, logger: logging.Logger) -> CommandResponse:
    """Run the batch create flow using async Nuggetizer methods."""
    processed_qids = get_processed_values(args.output_file, "qid")
    logger.info("Found %d already processed records", len(processed_qids))

    nuggetizer = Nuggetizer(**build_create_nuggetizer_kwargs(args))
    input_data = read_jsonl(args.input_file)
    logger.info("Processing %d records", len(input_data))

    generated_count = 0
    skipped_count = 0
    failed_count = 0
    with open(args.output_file, "a", encoding="utf-8") as file_obj:
        for i, record in enumerate(input_data, 1):
            qid = record["query"]["qid"]
            if qid in processed_qids:
                logger.info("Skipping already processed record %s", qid)
                skipped_count += 1
                continue
            logger.info("Processing record %d/%d", i, len(input_data))
            try:
                request = request_from_create_record(record)
                scored_nuggets = await nuggetizer.async_create(request)
                append_jsonl_record(
                    file_obj, create_output_record(request, scored_nuggets)
                )
                generated_count += 1
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


async def async_run_assign_answers_batch(
    args: Any, logger: logging.Logger
) -> CommandResponse:
    """Run the answer assignment flow using async Nuggetizer methods."""
    processed_qids = get_processed_values(args.output_file, "qid")
    logger.info("Found %d already processed records", len(processed_qids))

    run_id = get_run_id(args.answer_file)
    nuggetizer = Nuggetizer(
        assigner_model=args.model,
        log_level=args.log_level,
        use_azure_openai=args.use_azure_openai,
    )
    nugget_data = read_jsonl(args.nugget_file)
    answer_data = read_jsonl(args.answer_file)
    qid_to_answer_data = {answer["topic_id"]: answer for answer in answer_data}

    assigned_count = 0
    skipped_count = 0
    failed_count = 0
    with open(args.output_file, "a", encoding="utf-8") as file_obj:
        for nugget_record in nugget_data:
            qid = nugget_record["qid"]
            if qid in processed_qids:
                skipped_count += 1
                continue
            answer_record = qid_to_answer_data.get(
                qid, {"answer": [], "response_length": 0, "qid": qid}
            )
            try:
                answer_text = " ".join(
                    answer["text"] for answer in answer_record["answer"]
                )
                nuggets = scored_nuggets_from_record(nugget_record)
                assigned_nuggets = await nuggetizer.async_assign(
                    nugget_record.get("query", "N/A"), answer_text, nuggets=nuggets
                )
                append_jsonl_record(
                    file_obj,
                    assign_answer_output_record(
                        answer_record, nugget_record, run_id, assigned_nuggets
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


async def async_run_assign_retrieval_batch(
    args: Any, logger: logging.Logger
) -> CommandResponse:
    """Run the retrieval assignment flow using async Nuggetizer methods."""
    processed_entries = get_processed_values(args.output_file, "qid", "docid")
    nuggetizer = Nuggetizer(
        assigner_model=args.model,
        log_level=args.log_level,
        use_azure_openai=args.use_azure_openai,
    )
    nugget_data = read_jsonl(args.nugget_file)
    retrieve_data = read_jsonl(args.retrieve_results_file)
    qid_to_nuggets = {record["qid"]: record for record in nugget_data}

    assigned_count = 0
    skipped_count = 0
    failed_count = 0
    with open(args.output_file, "a", encoding="utf-8") as file_obj:
        for retrieve_record in retrieve_data:
            qid = retrieve_record["query"]["qid"]
            nugget_record = qid_to_nuggets.get(qid)
            if not nugget_record:
                continue
            for candidate in retrieve_record["candidates"]:
                processed_key = (qid, candidate["docid"])
                if processed_key in processed_entries:
                    skipped_count += 1
                    continue
                try:
                    nuggets = scored_nuggets_from_record(nugget_record)
                    assigned_nuggets = await nuggetizer.async_assign(
                        nugget_record.get("query", "N/A"),
                        candidate["doc"]["segment"],
                        nuggets=nuggets,
                    )
                    append_jsonl_record(
                        file_obj,
                        assign_retrieval_output_record(
                            nugget_record, candidate, assigned_nuggets
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
        command="assign-retrieval",
        metrics={
            "assigned_records": assigned_count,
            "skipped_records": skipped_count,
            "failed_records": failed_count,
        },
    )
