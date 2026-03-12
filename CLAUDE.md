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
- `examples/`: end-to-end usage examples (sync and async).
- `docs/`: assets only (logo currently).

## Packaging And Environment
- Build backend: `setuptools.build_meta` via `pyproject.toml`.
- Dependencies are dynamic and sourced from `requirements.txt`.
- Development tooling is defined in the `dev` dependency group in `pyproject.toml`.
- Install for development with `uv sync --group dev`.
- Recommended local environment from README: `uv`-managed Python 3.11 virtual environment.

## LLM Provider Conventions
- API keys are loaded from `.env` by `src/nuggetizer/utils/api.py`.
- Supported env vars:
  - OpenAI: `OPEN_AI_API_KEY` or `OPENAI_API_KEY`
  - OpenRouter: `OPENROUTER_API_KEY`
  - Azure OpenAI: `AZURE_OPENAI_API_BASE`, `AZURE_OPENAI_API_VERSION`, `AZURE_OPENAI_API_KEY`
- Keep provider fallback behavior intact in `LLMHandler`/`AsyncLLMHandler`:
  - OpenAI first when available, OpenRouter fallback when enabled/available.
  - vLLM uses local base URL (`http://localhost:<port>/v1`) with placeholder key.

## Coding Standards
- Formatting/linting/type checks are enforced by pre-commit:
  - Ruff (`ruff-check --fix`, `ruff-format`)
  - MyPy (strict-ish config in `pyproject.toml`)
- Run before committing:
  - `pre-commit run --all-files`
- Type hints are expected for new/changed code (`disallow_untyped_defs = true`).
- Preserve dataclass and Enum-based type contracts in `core/types.py`.

## CI And Contribution Workflow
- PR CI (`.github/workflows/pr-format.yml`) runs on PRs to `main`.
- CI currently validates style/type only via pre-commit (ruff + mypy).
- No dedicated automated test suite is present; validate behavior using examples/scripts locally.

## Validation Commands
- Lint/type:
  - `uv run pre-commit run --all-files`
- Quick smoke checks:
  - `uv run python examples/e2e.py --help`
  - `uv run python examples/async_e2e.py --help`
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
