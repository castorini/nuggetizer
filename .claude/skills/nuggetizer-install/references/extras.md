# nuggetizer Dependency Details

nuggetizer has no optional extras — all runtime dependencies (including OpenAI) are in the base install.

## Base Dependencies

Loaded dynamically from `requirements.txt` at build time. Key packages:

- `openai` — LLM backend for nugget generation and assignment
- `pyyaml` — YAML config parsing
- `tqdm` — Progress bars
- `requests` — HTTP client

## Dev Dependencies (dependency-group)

Available via `uv sync --group dev` or manual pip install:

| Package | Purpose |
|---------|---------|
| `mypy` | Static type checking (strict mode) |
| `pre-commit` | Git hook management |
| `pytest` | Test runner |
| `ruff` | Linter and formatter |
| `shtab` | Shell tab-completion generation |
| `types-PyYAML` | MyPy stubs for PyYAML |
| `types-requests` | MyPy stubs for requests |
| `types-tqdm` | MyPy stubs for tqdm |

## Notes

- Since there are no optional extras, `pip install -e .` or `uv sync` gets the full runtime.
- Dev dependencies use PEP 735 `[dependency-groups]` — only `uv sync --group dev` resolves them natively. With pip, install each package manually.
