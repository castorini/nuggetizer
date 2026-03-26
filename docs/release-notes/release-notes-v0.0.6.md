# nuggetizer v0.0.6

Initial release-note scaffold for the packaged CLI and JSONL pipeline workflow.

## Included In This Baseline

- Packaged `nuggetizer` CLI for create, assign, metrics, validation, doctor, prompt inspection, and view commands.
- FastAPI `nuggetizer serve` command exposing `GET /healthz`, `POST /v1/create`, and `POST /v1/assign` on port `8085` by default.
- Direct `create` input now also accepts Anserini REST payloads with `query.text` plus candidates whose `doc` is either a plain string or an object containing `contents`, so Anserini search results can be piped directly into `POST /v1/create` without a `jq` reshape step.
- Direct `create` input now also accepts single-record `castorini.cli.v1` envelopes from upstream tools such as `rank_llm`, so `search | rerank | create` can be piped through `POST /v1/create` without unwrapping `.artifacts[0].value[0]` first.
- Legacy `scripts/*.py` compatibility wrappers around the packaged CLI.
- Offline-first contributor workflow built around `uv`.

## Migration Notes

This baseline establishes the release-note policy for future changes.

Document a migration note whenever a change affects:

- prompt semantics or rendered prompt structure
- JSON or JSONL schema emitted by `create`, `assign`, or `metrics`
- metric definitions or aggregation behavior
- CLI defaults or compatibility-wrapper behavior
