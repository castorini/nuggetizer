# Nuggetizer Metrics Reference

## Score Fields

| Metric | Scope | Partial Credit | Description |
|--------|-------|----------------|-------------|
| `strict_vital_score` | vital nuggets only | No | Fraction of vital nuggets assigned `support` |
| `strict_all_score` | all nuggets | No | Fraction of all nuggets assigned `support` |
| `vital_score` | vital nuggets only | Yes | Score with 0.5 credit for `partial_support` |
| `all_score` | all nuggets | Yes | Score with 0.5 credit for `partial_support` |

## Scoring Formula

For strict scores:
```
strict_score = count(assignment == "support") / total_nuggets
```

For non-strict scores:
```
score = (count("support") + 0.5 * count("partial_support")) / total_nuggets
```

## Interpretation Guide

| Score Range | Interpretation |
|-------------|----------------|
| 0.9 – 1.0 | Excellent coverage — answer addresses nearly all nuggets |
| 0.7 – 0.9 | Good coverage — most key facts present |
| 0.5 – 0.7 | Moderate — significant facts missing |
| 0.3 – 0.5 | Weak — many important nuggets unsupported |
| 0.0 – 0.3 | Poor — answer fails to address the query's key facts |

## Relationships Between Scores

- `strict_vital_score ≤ vital_score` (always, because partial credit adds)
- `strict_all_score ≤ all_score` (always)
- `vital_score` focuses on the most important nuggets
- Large gap between `vital_score` and `all_score` suggests answer covers key facts but misses secondary details

## Token Usage (from trace fields)

When `--include-trace` was used during create/assign:

```json
{
  "trace": {
    "input_token_count": 245,
    "output_token_count": 89,
    "model": "gpt-4o"
  }
}
```

Aggregate across records for total cost estimation.

## Comparing Runs

When comparing two runs, control for:
1. **Same nuggets file** — both assign runs used the same `create` output
2. **Same contexts** — both ran on the same answers/retrieval file
3. **Same assignment mode** — both used 3-grade or both used 2-grade
4. **Same model** (if comparing across something else like prompt changes)

The compare script flags queries with score differences exceeding a configurable threshold.
