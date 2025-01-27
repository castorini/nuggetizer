#!/usr/bin/env python3
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

from nuggetizer.core.types import Query, Document, Request
from nuggetizer.models.nuggetizer import Nuggetizer


def setup_logging(log_level: int) -> None:
    """Configure logging based on verbosity level."""
    logging_level = logging.WARNING
    if log_level >= 2:
        logging_level = logging.DEBUG
    elif log_level >= 1:
        logging_level = logging.INFO
    
    logging.basicConfig(
        level=logging_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def read_jsonl(file_path: str) -> List[Dict]:
    """Read JSONL file and return list of dictionaries."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def process_input_record(record: Dict) -> Request:
    """Convert input JSON record to Request object."""
    query = Query(
        qid=record['query']['qid'],
        text=record['query']['text']
    )
    
    documents = []
    for candidate in record['candidates']:
        doc = Document(
            docid=candidate['docid'],
            segment=candidate['doc']['segment']
        )
        if 'judgment' in candidate and candidate['judgment'] > 0:
            documents.append(doc)
    
    return Request(query=query, documents=documents)


def format_output(request: Request, nuggets: List[Dict], nugget_trajectory: List[List[Dict]] = None) -> Dict:
    """Format output according to required schema."""
    if nugget_trajectory is None:
        return {
            "query": request.query.text,
            "qid": request.query.qid,
            "nuggets": [{"text": n.text} for n in nuggets],
        }
    else:
        return {
            "query": request.query.text,
            "qid": request.query.qid,
            "nuggets": [{"text": n.text} for n in nuggets],
            "nugget_trajectory": [[{"text": n.text} for n in nugget_list] for nugget_list in nugget_trajectory],
        }


def get_processed_qids(output_file: str) -> set:
    """Read the output file and return a set of already processed qids."""
    processed_qids = set()
    try:
        with open(output_file, 'r') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    processed_qids.add(record['qid'])
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        pass
    return processed_qids


def main():
    parser = argparse.ArgumentParser(description='Extract nuggets from input JSONL file')
    parser.add_argument('--input_file', type=str, help='Path to input JSONL file')
    parser.add_argument('--output_file', type=str, help='Path to output JSONL file')
    parser.add_argument('--model', type=str, default='gpt-4o', help='Model to use for nugget extraction')
    parser.add_argument('--store_trajectory', action='store_true', help='Store nugget trajectory')
    parser.add_argument('--log_level', type=int, default=0, choices=[0, 1, 2],
                      help='Logging level: 0=warnings only, 1=info, 2=debug')
    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # Get already processed qids
    processed_qids = get_processed_qids(args.output_file)
    logger.info("Found %d already processed records", len(processed_qids))

    # Initialize nuggetizer
    logger.info("Initializing Nuggetizer with model: %s", args.model)
    nuggetizer = Nuggetizer(model=args.model, log_level=args.log_level)
    
    # Process each record
    logger.info("Reading input file: %s", args.input_file)
    input_data = read_jsonl(args.input_file)
    
    logger.info("Processing %d records", len(input_data))
    with open(args.output_file, 'a') as f:
        for i, record in enumerate(input_data, 1):
            if record['query']['qid'] in processed_qids:
                logger.info("Skipping already processed record %s", record['query']['qid'])
                continue
                
            logger.info("Processing record %d/%d", i, len(input_data))
            try:
                request = process_input_record(record)
                nuggets, nugget_trajectory = nuggetizer.process(request)
                if args.store_trajectory:
                    output_record = format_output(request, nuggets, nugget_trajectory)
                else:
                    output_record = format_output(request, nuggets)
                f.write(json.dumps(output_record) + '\n')
                f.flush()  # Ensure the record is written immediately
                logger.info("Generated %d nuggets for record %d", len(nuggets), i)
            except Exception as e:
                logger.error("Error processing record %s: %s", record['query']['qid'], str(e))
                continue
    
    logger.info("Processing complete")


if __name__ == '__main__':
    main()
