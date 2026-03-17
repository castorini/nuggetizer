---
name: nuggetizer-quickstart
description: Use when working with nuggetizer CLI commands (create, assign, metrics), prompt templates, JSONL formats, or introspection (doctor, describe, schema, validate). Covers all entry points, flags, and the YAML prompt system.
---

# Nuggetizer Quickstart

Reference for the `nuggetizer` CLI — the tool for creating, scoring, and assigning factual nuggets for RAG evaluation using LLM backends.

## CLI Entry Point

```bash
nuggetizer <command> [options]
```

## Primary Commands

| Command | Purpose |
|---------|---------|
| `create` | Create and score nuggets from query + candidate passages |
| `assign` | Assign nuggets to contexts (answers or retrieval results) |
| `metrics` | Calculate per-query and global nugget-based scores |

## Introspection Commands

| Command | Purpose |
|---------|---------|
| `doctor` | Check Python version, API keys, backend readiness |
| `describe <cmd>` | Machine-readable command contract (flags, defaults, examples) |
| `schema <name>` | Print JSON Schema for inputs/outputs |
| `validate <target>` | Validate inputs without running models |
| `prompt list\|show\|render` | Inspect and render YAML prompt templates |
| `view <path>` | Inspect existing artifact files |

## Quick Workflow

```bash
# 1. Check environment
nuggetizer doctor

# 2. Create nuggets from a pool file
nuggetizer create --input-file pool.jsonl --output-file nuggets.jsonl --model gpt-4o

# 3. Assign nuggets to answers
nuggetizer assign --contexts answers.jsonl --nuggets nuggets.jsonl \
  --input-kind answers --output-file assignments.jsonl --model gpt-4o

# 4. Calculate metrics
nuggetizer metrics --input-file assignments.jsonl --output-file metrics.jsonl
```

## Reference Files

Read these on demand for details:

- `references/cli-examples.md` — Common invocations for each command
- `references/input-output-examples.md` — JSONL format examples (create input, create output, assign input/output, metrics output)
- `references/prompt-templates.md` — How the YAML prompt template system works

## Key Concepts

- **Nuggets**: Atomic facts extracted from candidate passages, each scored as `vital` or `okay`
- **Assignment modes**: 3-grade (`support`, `partial_support`, `not_support`) or 2-grade (`support`, `not_support`)
- **Backends**: OpenAI (default), Azure OpenAI (`--use-azure-openai`), OpenRouter (`--use-openrouter`), vLLM (local)
- **Write policies**: `--resume` (append, skip processed), `--overwrite` (truncate), `--fail-if-exists`

## Gotchas

- `create` has separate `--creator-model` and `--scorer-model` overrides. If only `--model` is set, both stages use it.
- `assign --input-kind answers` expects `--contexts` to be a JSONL with `topic_id` and `answer` fields. `assign --input-kind retrieval` expects the same format as `create` input.
- `--nuggets` is required for batch `assign` but not for direct input (nuggets are in the JSON payload).
- `metrics` operates on assign output only — it reads assignment labels, not raw nuggets.
- `--execution-mode async` is available but the default is `sync`.
- Legacy scripts in `scripts/` translate to CLI calls with `--resume` by default.
