---
name: nuggetizer-install
description: Set up a nuggetizer development environment — checks Python 3.11+, installs via uv or pip, and verifies with doctor. Use when someone is onboarding, setting up a fresh clone, or troubleshooting their environment.
---

# nuggetizer Install

Development environment setup for [nuggetizer](https://github.com/castorini/nuggetizer) — nugget creation, assignment, and evaluation.

## Prerequisites

- Python 3.11+
- Git (SSH access to `github.com:castorini`)

## Verify Runtime

```bash
python3 --version   # must be 3.11+
command -v uv       # if present, use uv path; otherwise recommend uv
```

If `uv` is on PATH, use it silently. If not, ask the user once: install uv or proceed with pip.

## Clone (if needed)

If no `pyproject.toml` in cwd:

```bash
git clone git@github.com:castorini/nuggetizer.git && cd nuggetizer
```

## Install (source — preferred)

### uv path

```bash
uv venv --python 3.11
source .venv/bin/activate
uv sync --group dev
```

### pip path

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install pre-commit pytest mypy ruff
```

### PyPI alternative (mention but don't default to)

```bash
pip install nuggetizer
```

## Smoke Test

```bash
nuggetizer doctor --output json
nuggetizer --help
```

## Pre-commit (source installs)

```bash
pre-commit install
```

## Reference Files

- `references/extras.md` — Optional dependency details (read when user asks about specific backends)

## Gotchas

- MyPy is strict: `disallow_untyped_defs = true`. All new functions need type annotations.
- `[dependency-groups]` in pyproject.toml is a PEP 735 feature — not pip-installable directly. Use `uv sync --group dev` or install dev deps manually with pip.
- OpenAI is a base dependency — no extras needed for the most common workflow.
- Test directory is `tests/` (with an s).
