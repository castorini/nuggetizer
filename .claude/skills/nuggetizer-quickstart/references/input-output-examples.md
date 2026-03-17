# Nuggetizer Input/Output Examples

## Create Input (batch JSONL)

Each line in the input file:

```json
{
  "query": {"qid": "q1", "text": "What is Python?"},
  "candidates": [
    {"docid": "d1", "score": 10.0, "doc": {"segment": "Python is a high-level programming language created by Guido van Rossum."}},
    {"docid": "d2", "score": 8.0, "doc": {"segment": "Python is used for web development, data science, and automation."}}
  ]
}
```

Lightweight shorthand (candidates as plain strings):

```json
{
  "query": "What is Python?",
  "candidates": ["Python is a programming language.", "Python is used for web development."]
}
```

## Create Output (JSONL)

```json
{
  "query": "What is Python?",
  "qid": "q1",
  "nuggets": [
    {"text": "Python is a high-level programming language", "importance": "vital"},
    {"text": "Python was created by Guido van Rossum", "importance": "vital"},
    {"text": "Python is used for web development", "importance": "okay"},
    {"text": "Python is used for data science", "importance": "okay"}
  ]
}
```

With `--include-trace`:

```json
{
  "query": "What is Python?",
  "qid": "q1",
  "nuggets": [...],
  "trace": {
    "prompt": "...",
    "response": "...",
    "input_token_count": 245,
    "output_token_count": 89
  }
}
```

## Assign Input — Answers Mode

Contexts file (`--contexts`, `--input-kind answers`):

```json
{"topic_id": "q1", "answer": [{"text": "Python is a high-level programming language created by Guido van Rossum. It is widely used for web development and data science."}]}
```

Nuggets file (`--nuggets`): same format as create output.

## Assign Input — Retrieval Mode

Contexts file (`--contexts`, `--input-kind retrieval`): same format as create input (query + candidates).

## Assign Output — Answers

```json
{
  "query": "What is Python?",
  "qid": "q1",
  "answer_text": "Python is a high-level programming language...",
  "response_length": 42,
  "run_id": "run-123",
  "nuggets": [
    {"text": "Python is a high-level programming language", "importance": "vital", "assignment": "support"},
    {"text": "Python was created by Guido van Rossum", "importance": "vital", "assignment": "support"},
    {"text": "Python is used for data science", "importance": "okay", "assignment": "partial_support"},
    {"text": "Python supports multithreading", "importance": "okay", "assignment": "not_support"}
  ]
}
```

## Assign Output — Retrieval

```json
{
  "text": "What is Python?",
  "qid": "q1",
  "candidate_text": "Python is a high-level programming language...",
  "docid": "d1",
  "nuggets": [
    {"text": "Python is a programming language", "importance": "vital", "assignment": "support"},
    {"text": "Python supports multithreading", "importance": "okay", "assignment": "not_support"}
  ]
}
```

## Metrics Output (JSONL)

```json
{
  "qid": "q1",
  "strict_vital_score": 0.8,
  "strict_all_score": 0.7,
  "vital_score": 0.85,
  "all_score": 0.75
}
```

## Metric Definitions

| Metric | Scope | Description |
|--------|-------|-------------|
| `strict_vital_score` | vital nuggets only | Fraction assigned `support` (strict: no partial credit) |
| `strict_all_score` | all nuggets | Fraction assigned `support` (strict) |
| `vital_score` | vital nuggets only | Score with partial credit for `partial_support` |
| `all_score` | all nuggets | Score with partial credit |
