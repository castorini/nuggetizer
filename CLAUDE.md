# Repo Project Instructions

## Scope
- Repository: `castorini/nuggetizer`
- Primary language: Python 3.11+
- Purpose: create/score/assign factual nuggets for RAG evaluation using LLM backends (OpenAI, Azure OpenAI, OpenRouter, vLLM).

## Project Layout
- `src/nuggetizer/models/nuggetizer.py`: main orchestration (`Nuggetizer`) for create/score/assign.
- `src/nuggetizer/core/`: core types, metrics, sync/async LLM handlers, base protocols.
- `src/nuggetizer/prompts/`: prompt builders and YAML prompt templates.
- `scripts/`: CLI pipelines for dataset-scale JSONL processing.
- `examples/`: end-to-end usage examples. `pipeline_demo.py` is the default async example and `sync_pipeline_demo.py` is the synchronous variant.
- `docs/`: assets only (logo currently).

## Packaging And Environment
- Build backend: `setuptools.build_meta` via `pyproject.toml`.
- Base runtime dependencies are declared in `pyproject.toml` under `[project.dependencies]`.
- Development tooling is defined in the `dev` dependency group in `pyproject.toml`.
- `.python-version` pins Python `3.11` for local `uv` workflows.
- Install for development with `uv sync --group dev`.
- Recommended local environment from README: `uv`-managed Python 3.11 virtual environment.
- Do not rely on `source "$HOME/.local/bin/env"` for `uv`; only add `~/.local/bin` to `PATH` when `uv` is not already available in the shell.

## LLM Provider Conventions
- API keys are loaded from `.env` by `src/nuggetizer/utils/api.py`.
- Supported env vars:
  - OpenAI: `OPENAI_API_KEY`
  - OpenRouter: `OPENROUTER_API_KEY`
  - Azure OpenAI: `AZURE_OPENAI_API_BASE`, `AZURE_OPENAI_API_VERSION`, `AZURE_OPENAI_API_KEY`
- Keep provider fallback behavior intact in `LLMHandler`/`AsyncLLMHandler`:
  - OpenAI first when available, OpenRouter fallback when enabled/available.
  - vLLM uses local base URL (`http://localhost:<port>/v1`) with placeholder key.

## Coding Standards
- Formatting, linting, tests, and type checks are enforced by the repo-local quality gate:
  - `uv run python scripts/quality_gate.py`
- The gate runs, in order:
  - `uv lock --check`
  - Ruff check
  - Ruff format check
  - Core tests
  - Integration tests
  - MyPy (strict-ish config in `pyproject.toml`)
- `pre-commit` and `pre-push` both invoke the same `uv run`-backed gate from the repository environment.
- Type hints are expected for new/changed code (`disallow_untyped_defs = true`).
- Preserve dataclass and Enum-based type contracts in `core/types.py`.

## CI And Contribution Workflow
- PR CI (`.github/workflows/pr-format.yml`) runs on PRs to `main`.
- CI runs the quality gate in order: Ruff, core tests, integration tests, then MyPy.
- Test tiers:
  - `core`: `uv run pytest -q -m core tests`
  - `integration`: `uv run pytest -q -m integration tests`
  - `live`: opt-in smoke tests such as `NUGGETIZER_LIVE_OPENAI_SMOKE=1 uv run pytest tests/test_live_openai_smoke.py`
- Keep `core` and `integration` coverage offline and deterministic; live-provider checks stay opt-in.
- Apply the shared pytest markers `core`, `integration`, and `live` at the module level when adding or moving tests.

## Validation Commands
- Full quality gate:
  - `uv run python scripts/quality_gate.py`
- Live smoke:
  - `NUGGETIZER_LIVE_OPENAI_SMOKE=1 uv run pytest tests/test_live_openai_smoke.py`
- Quick CLI help checks:
  - `uv run python examples/pipeline_demo.py --help`
  - `uv run python examples/sync_pipeline_demo.py --help`
  - `uv run python scripts/create_nuggets.py --help`
  - `uv run python scripts/assign_nuggets.py --help`
  - `uv run python scripts/calculate_metrics.py --help`

## Data And Pipeline Expectations
- `scripts/create_nuggets.py` expects JSONL records with `query` and `candidates`.
- `scripts/assign_nuggets.py` joins nugget JSONL with answer JSONL (`topic_id` mapping).
- `scripts/calculate_metrics.py` computes per-record and global metrics from assignments.
- Scripts append to output JSONL in some paths; avoid accidental duplicate processing.

## Change Guidelines
- Keep public constructor behavior stable in `Nuggetizer` (model args, provider flags, window/max controls).
- Avoid breaking JSONL schemas produced by `scripts/` unless all downstream consumers are updated.
- When editing prompt templates, verify prompt loader paths and assignment/score label compatibility.
- Preserve retry and key-rotation logic in LLM handlers unless intentionally redesigning error handling.

## Versioning
- Version is defined in `pyproject.toml` (`project.version`) and managed with `bumpver` config.
- If doing a release bump, update versioned references consistently per bumpver patterns.
- Keep user-visible changes documented in `docs/release-notes/`, including migration notes for CLI/schema/metric changes.
