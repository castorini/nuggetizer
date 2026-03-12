#!/usr/bin/env python3
import argparse

from nuggetizer.cli.logging_utils import setup_logging
from nuggetizer.cli.operations import run_assign_answers_batch
from nuggetizer.cli.spec import ASSIGN_COMMAND


def main() -> None:
    parser = argparse.ArgumentParser(description=ASSIGN_COMMAND.description)
    parser.add_argument(
        "--nugget_file", type=str, required=True, help="Path to nugget JSONL file"
    )
    parser.add_argument(
        "--answer_file", type=str, required=True, help="Path to answer JSONL file"
    )
    parser.add_argument(
        "--output_file", type=str, required=True, help="Path to output JSONL file"
    )
    parser.add_argument(
        "--model", type=str, default="gpt-4o", help="Model to use for assignment"
    )
    parser.add_argument(
        "--use_azure_openai", action="store_true", help="Use Azure OpenAI"
    )
    parser.add_argument(
        "--log_level",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="Logging level: 0=warnings only, 1=info, 2=debug",
    )
    args = parser.parse_args()

    logger = setup_logging(args.log_level)
    logger.info("Initializing Nuggetizer with model: %s", args.model)
    logger.info("Reading nugget file: %s", args.nugget_file)
    logger.info("Reading answer file: %s", args.answer_file)
    run_assign_answers_batch(args, logger)
    logger.info("Processing complete")


if __name__ == "__main__":
    main()
