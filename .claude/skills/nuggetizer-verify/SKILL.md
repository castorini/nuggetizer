---
name: nuggetizer-verify
description: Use when validating nuggetizer batch outputs — checks JSONL integrity, duplicate qids, missing nugget fields, assignment label validity, and metrics consistency. Wraps `nuggetizer validate` plus custom assertions. Use after running create, assign, or metrics to verify output correctness.
---

# Nuggetizer Verify

Validates nuggetizer batch outputs for correctness, completeness, and consistency.

## When to Use

- After `nuggetizer create` — verify nugget output integrity
- After `nuggetizer assign` — verify assignment labels and nugget coverage
- After `nuggetizer metrics` — verify metric ranges and qid completeness
- Before submitting results for evaluation or comparison

## What It Checks

### JSONL Integrity
- Every line is valid JSON
- No trailing commas, no truncated records
- Consistent field presence across records

### Create Output
- Every record has `qid`, `query`, and `nuggets` array
- Every nugget has `text` (non-empty string) and `importance` (`vital` or `okay`)
- No duplicate `qid` values
- No empty `nuggets` arrays

### Assign Output
- Every record has `qid`, `query`, and `nuggets` array with `assignment` field
- Assignment labels are valid: `support`, `partial_support`, `not_support` (3-grade) or `support`, `not_support` (2-grade)
- No mixed assignment modes within a single file
- Every nugget retains `text` and `importance` from creation

### Metrics Output
- Every record has `qid` and all four score fields
- Scores are in [0.0, 1.0] range
- `strict_*` scores ≤ corresponding non-strict scores
- `vital_score` computed only over vital nuggets (not all)

## Usage

Run the verification script:

```bash
bash .claude/skills/nuggetizer-verify/scripts/verify.sh <artifact-path> [artifact-type]
```

Or use the built-in validator first:

```bash
nuggetizer validate create --input-file nuggets.jsonl
nuggetizer validate assign --contexts answers.jsonl --nuggets nuggets.jsonl --input-kind answers
```

Then run custom assertions via the verify script.

## Verification Script

See `scripts/verify.sh` for the runnable verification wrapper.

## Gotchas

- `nuggetizer validate` checks *input* contracts (before running). The verify script checks *output* artifacts (after running).
- A record with `importance: "failed"` on a nugget indicates a scoring failure — the nugget was created but not successfully scored.
- `--resume` can produce files with records from different model versions if the model flag changed between runs. Check `trace.model` consistency if `--include-trace` was used.
- Metrics require assign output as input. Running metrics on create output will fail silently or produce meaningless results.
