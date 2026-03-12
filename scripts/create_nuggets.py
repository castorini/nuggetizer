#!/usr/bin/env python3
import argparse

from nuggetizer.cli.logging_utils import setup_logging
from nuggetizer.cli.operations import run_create_batch
from nuggetizer.cli.spec import CREATE_COMMAND


def main() -> None:
    parser = argparse.ArgumentParser(description=CREATE_COMMAND.description)
    parser.add_argument(
        "--input_file", type=str, required=True, help="Path to input JSONL file"
    )
    parser.add_argument(
        "--output_file", type=str, required=True, help="Path to output JSONL file"
    )
    parser.add_argument(
        "--model", type=str, default="gpt-4o", help="Model to use for all operations"
    )
    parser.add_argument(
        "--creator_model", type=str, help="Model to use for nugget creation"
    )
    parser.add_argument(
        "--scorer_model", type=str, help="Model to use for nugget scoring"
    )
    parser.add_argument("--window_size", type=int, help="Window size for processing")
    parser.add_argument(
        "--max_nuggets", type=int, help="Maximum number of nuggets to extract"
    )
    parser.add_argument(
        "--log_level",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="Logging level: 0=warnings only, 1=info, 2=debug",
    )
    parser.add_argument(
        "--use_azure_openai", action="store_true", help="Use Azure OpenAI"
    )
    args = parser.parse_args()

    logger = setup_logging(args.log_level)
    logger.info("Initializing Nuggetizer")
    logger.info("Reading input file: %s", args.input_file)
    run_create_batch(args, logger)
    logger.info("Processing complete")


if __name__ == "__main__":
    main()
