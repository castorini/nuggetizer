# Nuggetizer CLI For Agents

`nuggetizer ...` is the canonical command-line interface for this repository.
Use it in preference to `scripts/*.py`. If the virtual environment is not
activated, the development fallback is `uv run nuggetizer ...`.

`nuggetizer view` renders an existing output artifact for humans after the main
pipeline command has finished writing it.
`nuggetizer prompt` inspects built-in prompt templates and renders exact prompt
messages for direct input payloads without calling a model.

## Command Mapping

Old:
```bash
python3 scripts/create_nuggets.py --input_file pool.jsonl --output_file nuggets.jsonl
```

New:
```bash
nuggetizer create --input-file pool.jsonl --output-file nuggets.jsonl
```

Old:
```bash
python3 scripts/assign_nuggets.py --nugget_file nuggets.jsonl --answer_file answers.jsonl --output_file assignments.jsonl
```

New:
```bash
nuggetizer assign --input-kind answers --nuggets nuggets.jsonl --contexts answers.jsonl --output-file assignments.jsonl
```

Old:
```bash
python3 scripts/assign_nuggets_retrieve_results.py --nugget_file nuggets.jsonl --retrieve_results_file retrieval.jsonl --output_file retrieval_assignments.jsonl
```

New:
```bash
nuggetizer assign --input-kind retrieval --nuggets nuggets.jsonl --contexts retrieval.jsonl --output-file retrieval_assignments.jsonl
```

Old:
```bash
python3 scripts/calculate_metrics.py --input_file assignments.jsonl --output_file metrics.jsonl
```

New:
```bash
nuggetizer metrics --input-file assignments.jsonl --output-file metrics.jsonl
```

## Direct Single-Object Examples

Create without caller-supplied IDs:

```bash
nuggetizer create \
  --input-json '{"query":"What is Python used for?","candidates":["Python is used for web development.","Python is used for data analysis."]}' \
  --output json
```

Create directly from `umbrela` judgments. The default threshold keeps only
`judgment >= 2`; set `--min-judgment 1` to include weakly relevant passages:

```bash
nuggetizer create \
  --input-json '{"judgments":[{"query":"What is Python used for?","passage":"Python can be difficult for beginners.","judgment":1},{"query":"What is Python used for?","passage":"Python is used for web development.","judgment":2}]}' \
  --output json
```

Assign without caller-supplied IDs:

```bash
nuggetizer assign \
  --input-json '{"query":"What is Python used for?","context":"Python is used for web development and data analysis.","nuggets":[{"text":"Python is used for web development.","importance":"vital"},{"text":"Python is used for data analysis.","importance":"okay"}]}' \
  --output json
```

Assign by joining a single answer artifact with a nugget pool artifact:

```bash
nuggetizer assign \
  --input-json '{"answer_record":{"topic_id":"q1","topic":"What is Python used for?","response_length":10,"answer":[{"text":"Python is used for web development."}]},"nugget_record":{"query":"What is Python used for?","qid":"q1","nuggets":[{"text":"Python is used for web development.","importance":"vital"}]}}' \
  --output json
```

Assign one nugget pool against multiple answer records:

```bash
nuggetizer assign \
  --input-json '{"answer_records":[{"run_id":"demo-run","topic_id":"q1","topic":"What is Python used for?","response_length":10,"answer":[{"text":"Python is used for web development."}]},{"topic_id":"q1","topic":"What is Python used for?","response_length":8,"answer":[{"text":"Python is also used for automation."}]}],"nugget_record":{"query":"What is Python used for?","qid":"q1","nuggets":[{"text":"Python is used for web development.","importance":"vital"}]}}' \
  --output json
```

Direct mode synthesizes internal positional IDs only when internal helpers need
them. Callers do not need to provide `qid` or `docid`.

## JSON And Dry-Run Examples

Describe command contract:

```bash
nuggetizer describe assign --output json
```

Dry-run a batch assignment without writing output:

```bash
nuggetizer assign \
  --input-kind answers \
  --nuggets nuggets.jsonl \
  --contexts answers.jsonl \
  --output-file assignments.jsonl \
  --dry-run \
  --output json
```

Validate a create input file before execution:

```bash
nuggetizer validate create --input-file pool.jsonl --output json
```

View existing create, assign, or metrics artifacts:

```bash
nuggetizer view nuggets.jsonl
nuggetizer view assignments.jsonl --records 1
nuggetizer view metrics.jsonl
```

Inspect or render prompt templates:

```bash
nuggetizer prompt list
nuggetizer prompt show create
nuggetizer prompt show assign --assign-mode support_grade_2
nuggetizer prompt render create \
  --input-json '{"query":"What is Python used for?","candidates":["Python is used for web development."]}'
nuggetizer prompt render create \
  --input-json '{"judgments":[{"query":"What is Python used for?","passage":"Python can be difficult for beginners.","judgment":1},{"query":"What is Python used for?","passage":"Python is used for web development.","judgment":2}]}'
nuggetizer prompt render assign \
  --assign-mode support_grade_3 \
  --input-json '{"query":"What is Python used for?","context":"Python is used for web development.","nuggets":[{"text":"Python is used for web development.","importance":"vital"}]}'
nuggetizer prompt render score \
  --input-json '{"query":"What is Python used for?","nuggets":["Python is used for web development."]}'
```

Serve-to-serve piping works for both rerank-style input and `umbrela` judgments:

```bash
curl -s "http://127.0.0.1:8081/v1/msmarco-v1-passage/search?query=what%20is%20python%20used%20for" \
  | curl -s -X POST http://127.0.0.1:8085/v1/create \
      -H 'content-type: application/json' \
      --data-binary @-

curl -s -X POST http://127.0.0.1:8086/v1/judge \
    -H 'content-type: application/json' \
    --data-binary '{"query":"what is python used for","candidates":["Python can be difficult for beginners.","Python is used for web development."]}' \
  | curl -s -X POST http://127.0.0.1:8085/v1/create \
      -H 'content-type: application/json' \
      --data-binary @-

curl -s -X POST http://127.0.0.1:8086/v1/judge \
    -H 'content-type: application/json' \
    --data-binary '{"query":"what is python used for","candidates":["Python can be difficult for beginners.","Python is used for web development."]}' \
  | python - <<'PY'
import json, sys
payload = json.load(sys.stdin)
payload["overrides"] = {"min_judgment": 1}
print(json.dumps(payload))
PY
  | curl -s -X POST http://127.0.0.1:8085/v1/create \
      -H 'content-type: application/json' \
      --data-binary @-
```

## Failure Example

Missing input files produce a structured JSON error and a non-zero exit code:

```bash
nuggetizer create \
  --input-file missing.jsonl \
  --output-file nuggets.jsonl \
  --output json
```

## Notes

- Default execution mode is synchronous.
- Async execution is opt-in via `--execution-mode async`.
- `nuggetizer create` defaults to `--min-judgment 2` when a candidate has a `judgment` field.
- Trace and reasoning fields are opt-in via `--include-trace` and `--include-reasoning`.
- Prompt content inside traces can be removed with `--redact-prompts`.
