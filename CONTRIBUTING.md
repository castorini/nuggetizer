# Contributing to nuggetizer

Thank you for contributing to nuggetizer. This repository is the Castorini toolkit for nugget creation, nugget assignment, and nugget-based retrieval-augmented generation evaluation, so changes should preserve CLI stability, JSONL artifact compatibility, and deterministic offline validation.

## Before You Start

- Open or reference a GitHub issue for significant bug fixes, features, or refactors when possible.
- Keep pull requests focused. Separate API changes, large prompt rewrites, and infrastructure cleanups into different pull requests unless they are inseparable.
- If your change affects the public CLI, structured JSON output, JSONL schemas, or prompt behavior, update the README and any relevant examples in the same pull request.

## Development Setup

This repository uses `uv` and the `dev` dependency group defined in `pyproject.toml`.

```bash
uv python install 3.11
uv venv --python 3.11
uv sync --group dev
pre-commit install
```

The base package and `dev` tools are installed by default with `uv sync --group dev`. Optional extras such as `api` remain opt-in:

```bash
uv sync --group dev --extra api
```

Prefer `uv run ...` instead of activating the virtual environment manually.

## Local Quality Gate

Run this ordered gate before opening a pull request:

```bash
uv run python scripts/quality_gate.py
```

The repo-local gate runs `uv lock --check`, Ruff, core tests, integration tests, and MyPy in that order. Both `pre-commit` and `pre-push` hooks invoke the same `uv run`-backed gate in the repository virtual environment, so push-time checks stay aligned with local validation.

## Testing Expectations

- Add or update tests for non-trivial behavior changes.
- Keep tests in one of these layers:
  - `core`: fast deterministic CLI, prompt, and handler coverage that always runs in PR CI
  - `integration`: deterministic offline pipeline regressions backed by frozen fixtures
  - `live`: provider-backed smoke tests gated behind explicit environment variables
- Apply the shared pytest markers (`core`, `integration`, `live`) at the module level so CI and local commands stay aligned across Castorini Python repos.
- Prefer offline, deterministic tests under `tests/`.
- If you change CLI behavior, request normalization, JSON envelopes, or compatibility shims in `scripts/`, add regression coverage.
- If a change depends on live model providers, keep the default automated tests offline and explain any manual validation steps in the pull request.

## Prompt, Schema, and Pipeline Safety

- Do not silently change JSONL schemas emitted by `create`, `assign`, `assign`, or `metrics`.
- Preserve the canonical `nuggetizer` CLI interface and legacy script compatibility shims unless the pull request explicitly documents a migration.
- When editing prompt templates, verify that the loader paths and expected label semantics still match the code paths that consume them.
- Preserve provider fallback and retry behavior unless the pull request is intentionally redesigning those paths.

## Documentation Expectations

- Update `README.md` for any user-facing CLI, install, or environment changes.
- Update examples when the recommended invocation path changes.
- Mention new environment variables, model-provider requirements, or optional extras in the pull request description.
- Add or update a file in `docs/release-notes/` for user-visible changes.
- If prompt semantics, JSONL schemas, metric definitions, or CLI defaults change, include a migration note in the release note.

## Pull Request Checklist

Before submitting:

1. Run the local quality gate commands listed above.
2. Summarize the user-visible behavior change and any schema impact.
3. Mention whether the change affects prompt outputs, metrics, latency, cost, or validation behavior.
4. Include benchmark or comparison data when the change could affect quality or efficiency.

## Reporting Issues

GitHub issues are the public tracker for bugs and feature requests. Good reports include:

- the exact `nuggetizer ...` or `scripts/*.py` command used
- a minimal input artifact or fixture
- expected versus observed behavior
- relevant logs, tracebacks, or sample output JSON or JSONL

## License

By contributing to nuggetizer, you agree that your contributions will be licensed under the `LICENSE` file in the root of this repository.
