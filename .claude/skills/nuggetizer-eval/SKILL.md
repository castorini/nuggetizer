---
name: nuggetizer-eval
description: Use when analyzing nuggetizer metrics outputs — comparing runs, computing agreement between models, reporting token usage and latency from trace fields, and building side-by-side evaluation tables. Use after running metrics to interpret and compare results.
---

# Nuggetizer Eval

Analyze and compare nuggetizer metrics outputs across runs, models, or assignment configurations.

## When to Use

- After `nuggetizer metrics` — interpret scores and compare runs
- When comparing different models (e.g., gpt-4o vs gpt-4o-mini)
- When comparing assignment modes (2-grade vs 3-grade)
- When reporting agreement, token usage, or latency from trace fields

## What It Does

### Run Comparison
- Load two metrics JSONL files and produce a side-by-side table
- Show per-query score differences and global averages
- Highlight queries where runs diverge significantly

### Agreement Analysis
- Compare assignment labels between two assign output files
- Report per-nugget agreement rate and Cohen's kappa
- Break down by importance level (vital vs okay)

### Token Usage and Latency
- Extract trace fields from `--include-trace` outputs
- Report total/average input and output token counts
- Estimate cost based on token counts and model pricing

## Usage

Compare two metrics runs:
```bash
python3 .claude/skills/nuggetizer-eval/scripts/compare.py \
  --run-a metrics_gpt4o.jsonl --run-b metrics_mini.jsonl
```

Or use the CLI directly:
```bash
# View metrics summary
nuggetizer view metrics.jsonl --records 0 --output json

# Run metrics on an assignment file
nuggetizer metrics --input-file assignments.jsonl --output-file metrics.jsonl --output json
```

## Reference Files

- `references/metrics.md` — Metric definitions, interpretation guide, and scoring examples

## Comparison Script

See `scripts/compare.py` for the side-by-side comparison tool.

## Gotchas

- Metrics are computed from assign output, not create output. Make sure you're comparing assign outputs from the same input nuggets and contexts.
- `strict_*` scores give no partial credit for `partial_support`. Non-strict scores give 0.5 credit.
- When comparing runs, ensure both used the same nuggets file — otherwise differences reflect nugget variation, not assignment quality.
- Token usage requires `--include-trace` during create/assign. Without it, trace fields are absent.
- `--resume` runs may mix records from different sessions — check for consistency before comparing.
