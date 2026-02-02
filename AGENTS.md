# AGENTS.md

## Project Snapshot

Nuggetizer is a Python package for **information nugget creation, scoring, and assignment** for RAG evaluation. Three-stage pipeline:

1. **Creation** — Extract atomic nuggets (1–12 word facts) from candidate documents.
2. **Scoring** — Label each nugget `vital` or `okay`.
3. **Assignment** — Determine whether answer text `support`s, `partial_support`s, or `not_support`s each nugget.

Source: `src/nuggetizer/` with subpackages `core`, `models`, `prompts`, `utils`.

## Architecture

```
Request (Query + Documents)
  → Nuggetizer.create()
    → [windowed] creator prompt → LLM → ast.literal_eval
    → [windowed] scorer prompt  → LLM → ast.literal_eval
  → List[ScoredNugget]
  → Nuggetizer.assign(query, context, nuggets)
    → [windowed] assigner prompt → LLM → ast.literal_eval
  → List[AssignedScoredNugget]
  → calculate_nugget_scores() → NuggetMetrics
```

`Nuggetizer` in `models/nuggetizer.py` is the central orchestrator — owns three LLM handlers (creator, scorer, assigner) and delegates prompt construction to `prompts/`.

## Module Layout

- `core/types.py` — All dataclasses and enums
- `core/base.py` — ABC + `@runtime_checkable` Protocol contracts
- `core/llm.py` / `core/async_llm.py` — Sync and async LLM handlers (OpenAI SDK wrappers)
- `core/metrics.py` — Nugget scoring math
- `models/nuggetizer.py` — Main `Nuggetizer` class (public API)
- `prompts/template_loader.py` — YAML template loading + caching
- `prompts/*_prompts.py` — Prompt builders for each stage
- `prompts/prompt_templates/*.yaml` — The actual prompt text
- `utils/api.py` — Env-var loaders for API keys
- `utils/display.py` — Pretty-printing utilities
- `scripts/` — CLI pipeline (create, assign, metrics)
- `examples/` — End-to-end sync and async demos

## Key Patterns

1. **Dataclass hierarchy** — `BaseNugget → Nugget | ScoredNugget → AssignedScoredNugget`. Use dataclasses for domain objects, not plain dicts.
2. **`ast.literal_eval` for LLM parsing** — LLMs return Python list literals. Always parse with `ast.literal_eval`, never `eval()`. Response cleaning strips markdown code fences before parsing.
3. **Windowed processing** — Documents/nuggets are chunked into configurable windows (default size 10) for LLM calls.
4. **Temperature escalation** — Starts at `temperature=0.0`; bumps to `0.2` on parse failure.
5. **Graceful degradation** — Scoring failures default to `importance="okay"`, assignment failures to `assignment="failed"`.
6. **Lazy async init** — Async LLM clients created on first async call via `_ensure_async_llm()`.
7. **Round-robin key rotation** — API keys stored as a list, rotated on failure.
8. **Resume support in scripts** — Scripts read existing output to skip already-processed entries.

## LLM Handlers

Four providers supported — Azure OpenAI (`"azure"`), OpenAI (`"openai"`), OpenRouter (`"openrouter"`), vLLM (`"vllm"`) — all via the OpenAI SDK with different configs.

**Sync/async asymmetries to watch:** The sync handler (`llm.py`) has 5 retries, 4096 max tokens, 60s timeout, vLLM param branching, and content filter abort. The async handler (`async_llm.py`) retries infinitely, uses 2048 max tokens, 30s timeout, and lacks vLLM branching and content filter handling. If you fix a bug in one, check the other.

**Special model handling:** Models starting with `o1`, `o3`, `o4`, or `gpt-5` collapse system messages into the user message and force `temperature=1.0`.

## Prompt System

- Prompts are YAML templates in `prompts/prompt_templates/`. Each has `system_message` and `prefix_user` with `str.format()` placeholders.
- Always edit YAML templates, never hard-code prompt text in Python.
- Templates are cached at module level. New templates must match the `*.yaml` glob in `pyproject.toml` package-data.
- Keep template format variables in sync with the Python prompt builder that passes them.

## Environment & Secrets

API keys loaded from `.env` via `python-dotenv` (legacy keys from `.env.local`). Never hard-code keys.

- Azure: `AZURE_OPENAI_API_BASE`, `AZURE_OPENAI_API_VERSION`, `AZURE_OPENAI_API_KEY`
- OpenAI: `OPEN_AI_API_KEY` (or `OPENAI_API_KEY`)
- OpenRouter: `OPENROUTER_API_KEY`
- vLLM: No auth needed

## Tooling

- Python `>=3.10`. All new code must be fully typed.
- Pre-commit enforces `ruff check --fix`, `ruff format`, and `mypy` (strict settings in `pyproject.toml`).
- Run `pre-commit run --all-files` before committing.
- Version bumps via `bumpver` — updates `pyproject.toml` and `README.md`.

## Scripts

Sequential pipeline for TREC RAG Track evaluation:
1. `create_nuggets.py` — Extract + score nuggets from query/document JSONL
2. `assign_nuggets.py` — Assign nuggets to RAG answer JSONL
3. `assign_nuggets_retrieve_results.py` — Variant for individual retrieved segments
4. `calculate_metrics.py` — Compute per-query and global metrics

All scripts use `argparse`, support resume via output file scanning, handle errors per-record with `continue`, and flush output immediately.

## Changes Checklist

- [ ] New modules under `src/nuggetizer/` and registered in `pyproject.toml` packages if needed
- [ ] `__init__.py` and `__all__` updated for new public API
- [ ] Prompt changes in YAML templates, not Python; format variables match prompt builders
- [ ] All functions fully typed
- [ ] `pre-commit run --all-files` passes
- [ ] Changes to `core/llm.py` checked against `core/async_llm.py` (and vice versa)
- [ ] Scripts maintain resume support
- [ ] No hard-coded API keys
