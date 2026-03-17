# nuggetizer v0.0.6

Initial release-note scaffold for the packaged CLI and JSONL pipeline workflow.

## Included In This Baseline

- Packaged `nuggetizer` CLI for create, assign, metrics, validation, doctor, prompt inspection, and view commands.
- Legacy `scripts/*.py` compatibility wrappers around the packaged CLI.
- Offline-first contributor workflow built around `uv`.

## Migration Notes

This baseline establishes the release-note policy for future changes.

Document a migration note whenever a change affects:

- prompt semantics or rendered prompt structure
- JSON or JSONL schema emitted by `create`, `assign`, or `metrics`
- metric definitions or aggregation behavior
- CLI defaults or compatibility-wrapper behavior
