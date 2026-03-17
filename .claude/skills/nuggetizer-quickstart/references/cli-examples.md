# Nuggetizer CLI Examples

## create

```bash
# Batch: create nuggets from pool JSONL
nuggetizer create --input-file pool.jsonl --output-file nuggets.jsonl --model gpt-4o

# With resume (skip already-processed qids)
nuggetizer create --input-file pool.jsonl --output-file nuggets.jsonl --model gpt-4o --resume

# Direct input (single query, stdout)
nuggetizer create --input-json '{"query":"What is Python?","candidates":["Python is a programming language."]}' --output json

# With trace and reasoning fields
nuggetizer create --input-file pool.jsonl --output-file nuggets.jsonl --model gpt-4o --include-trace --include-reasoning

# Dry run (resolve inputs, no model calls)
nuggetizer create --input-file pool.jsonl --output-file nuggets.jsonl --model gpt-4o --dry-run

# Separate creator and scorer models
nuggetizer create --input-file pool.jsonl --output-file nuggets.jsonl --creator-model gpt-4o --scorer-model gpt-4o-mini

# Async execution
nuggetizer create --input-file pool.jsonl --output-file nuggets.jsonl --model gpt-4o --execution-mode async

# OpenRouter backend
nuggetizer create --input-file pool.jsonl --output-file nuggets.jsonl --model openai/gpt-4o --use-openrouter

# Azure OpenAI backend
nuggetizer create --input-file pool.jsonl --output-file nuggets.jsonl --model gpt-4o --use-azure-openai
```

## assign

```bash
# Assign nuggets to answers (3-grade)
nuggetizer assign --contexts answers.jsonl --nuggets nuggets.jsonl \
  --input-kind answers --output-file assignments.jsonl --model gpt-4o

# Assign nuggets to retrieval results
nuggetizer assign --contexts retrieval.jsonl --nuggets nuggets.jsonl \
  --input-kind retrieval --output-file assign_retrieval.jsonl --model gpt-4o

# Direct input (single query, stdin)
echo '{"query":"What is Python?","context":"Python is a language.","nuggets":[{"text":"Python is a language","importance":"vital"}]}' | \
  nuggetizer assign --stdin --output json

# Validate assign input without running
nuggetizer validate assign --input-file answers.jsonl --nuggets nuggets.jsonl --input-kind answers
```

## metrics

```bash
# Calculate metrics from assignment output
nuggetizer metrics --input-file assignments.jsonl --output-file metrics.jsonl

# JSON output
nuggetizer metrics --input-file assignments.jsonl --output-file metrics.jsonl --output json

# With manifest
nuggetizer metrics --input-file assignments.jsonl --output-file metrics.jsonl --manifest-path manifest.json
```

## Introspection

```bash
# Environment check
nuggetizer doctor
nuggetizer doctor --output json

# Command contract
nuggetizer describe create
nuggetizer describe assign --output json

# JSON Schemas
nuggetizer schema create-batch-input-record
nuggetizer schema assign-output-answers
nuggetizer schema cli-envelope

# Prompt inspection
nuggetizer prompt list
nuggetizer prompt show create
nuggetizer prompt show assign --assign-mode support_grade_3
nuggetizer prompt render create --input-json '{"query":"test","candidates":["passage"]}' --part user

# View artifacts
nuggetizer view nuggets.jsonl
nuggetizer view assignments.jsonl --records 5 --output json

# Validate inputs
nuggetizer validate create --input-file pool.jsonl
nuggetizer validate assign --contexts answers.jsonl --nuggets nuggets.jsonl --input-kind answers
```
