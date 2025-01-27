#!/usr/bin/env python3
import argparse
import json
import logging
from typing import Dict, List

from nuggetizer.core.types import Nugget
from nuggetizer.models.scorer import NuggetScorer


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


def process_record(record: Dict, scorer: NuggetScorer, logger: logging.Logger) -> Dict:
    """Process a single record and add importance scores."""
    # Convert nuggets to Nugget objects
    nuggets = [Nugget(text=n['text']) for n in record['nuggets']]
    
    logger.info("Scoring %d nuggets for query: %s", len(nuggets), record.get('query', 'N/A'))
    # Score nuggets
    scored_nuggets = scorer.score(nuggets)
    
    # Update record with scores
    record['nuggets'] = [
        {
            'text': n.text,
            'importance': n.importance
        }
        for n in scored_nuggets
    ]
    
    logger.info("Completed scoring with results: %s", 
                {importance: sum(1 for n in scored_nuggets if n.importance == importance)
                 for importance in ['vital', 'okay']})
    return record


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
    parser = argparse.ArgumentParser(description='Score nuggets from input JSONL file')
    parser.add_argument('--input_file', type=str, help='Path to input JSONL file')
    parser.add_argument('--output_file', type=str, help='Path to output JSONL file')
    parser.add_argument('--model', type=str, default='gpt-4o', help='Model to use for scoring')
    parser.add_argument('--log_level', type=int, default=0, choices=[0, 1, 2],
                      help='Logging level: 0=warnings only, 1=info, 2=debug')
    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # Get already processed qids
    processed_qids = get_processed_qids(args.output_file)
    logger.info("Found %d already processed records", len(processed_qids))

    # Initialize scorer
    logger.info("Initializing NuggetScorer with model: %s", args.model)
    scorer = NuggetScorer(model=args.model, log_level=args.log_level)
    
    # Process each record
    logger.info("Reading input file: %s", args.input_file)
    input_data = read_jsonl(args.input_file)
    
    logger.info("Processing %d records", len(input_data))
    with open(args.output_file, 'a') as f:
        for i, record in enumerate(input_data, 1):
            if record['qid'] in processed_qids:
                logger.info("Skipping already processed record %s", record['qid'])
                continue
                
            logger.info("Processing record %d/%d", i, len(input_data))
            try:
                scored_record = process_record(record, scorer, logger)
                f.write(json.dumps(scored_record) + '\n')
                f.flush()  # Ensure the record is written immediately
            except Exception as e:
                logger.error("Error processing record %s: %s", record['qid'], str(e))
                continue
    
    logger.info("Processing complete")


if __name__ == '__main__':
    main() 
