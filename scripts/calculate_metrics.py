#!/usr/bin/env python3
import argparse
import json

from nuggetizer.cli.operations import run_metrics
from nuggetizer.cli.spec import METRICS_COMMAND


def main() -> None:
    parser = argparse.ArgumentParser(description=METRICS_COMMAND.description)
    parser.add_argument(
        "--input_file", type=str, help="Path to input JSONL file with assignments"
    )
    parser.add_argument("--output_file", type=str, help="Path to output JSONL file")
    args = parser.parse_args()

    processed_records, global_metrics = run_metrics(args)

    with open(args.output_file, "w", encoding="utf-8") as file_obj:
        for record in processed_records:
            file_obj.write(json.dumps(record) + "\n")
        print(global_metrics)
        file_obj.write(json.dumps(global_metrics) + "\n")


if __name__ == "__main__":
    main()
