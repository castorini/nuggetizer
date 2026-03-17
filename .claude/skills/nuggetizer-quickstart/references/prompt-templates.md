# Nuggetizer Prompt Template System

## Template Location

Templates live in `src/nuggetizer/prompts/prompt_templates/`:

| File | Method | Purpose |
|------|--------|---------|
| `creator_template.yaml` | `nugget_creation` | Extract atomic nuggets from query + candidates |
| `scorer_template.yaml` | `nugget_scoring` | Score each nugget as `vital` or `okay` |
| `assigner_template.yaml` | `nugget_assignment` | 3-grade assignment: `support`, `partial_support`, `not_support` |
| `assigner_2grade_template.yaml` | `nugget_assignment_2grade` | 2-grade assignment: `support`, `not_support` |

## Template Structure (YAML)

```yaml
method: "nugget_creation"
system_message: |
  You are an expert at extracting atomic factual nuggets...
prefix_user: |
  Query: {query}
  Candidates: {context}
  ...
```

Each template has:
- `method`: identifier used for dispatch
- `system_message`: LLM system prompt
- `prefix_user`: user prompt with placeholder variables

## Placeholder Variables

| Variable | Used In | Description |
|----------|---------|-------------|
| `{query}` | all templates | The query text |
| `{context}` | creator, assigner | Candidate passage(s) or answer text |
| `{nuggets}` | scorer, assigner | List of nuggets to score/assign |
| `{nuggets_length}` | creator | Current nugget count (for iterative creation) |
| `{num_nuggets}` | scorer, assigner | Number of nuggets in the list |
| `{creator_max_nuggets}` | creator | Max nuggets to extract |

## Inspecting Templates

```bash
# List all templates
nuggetizer prompt list

# Show a specific template
nuggetizer prompt show create
nuggetizer prompt show assign --assign-mode support_grade_3
nuggetizer prompt show score

# Render with real input
nuggetizer prompt render create \
  --input-json '{"query":"What is Python?","candidates":["Python is a language."]}' \
  --part user

# Render assign prompt
nuggetizer prompt render assign \
  --input-json '{"query":"What is Python?","context":"Python is a language.","nuggets":[{"text":"Python is a language","importance":"vital"}]}' \
  --assign-mode support_grade_3 \
  --part all
```

## Assignment Modes

| Mode | Labels | Template |
|------|--------|----------|
| `support_grade_3` (default) | `support`, `partial_support`, `not_support` | `assigner_template.yaml` |
| `support_grade_2` | `support`, `not_support` | `assigner_2grade_template.yaml` |

## Importance Labels

| Label | Meaning |
|-------|---------|
| `vital` | Nugget captures a key fact essential to answering the query |
| `okay` | Nugget is relevant but not essential |
| `failed` | Scoring failed for this nugget (fallback) |
