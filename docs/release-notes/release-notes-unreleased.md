# nuggetizer (unreleased)

## Fixes

- `create` now forwards the configured `max_nuggets` / `creator_max_nuggets` into the nugget-creation prompt. Previously the prompt always rendered the template default ("at most 30 nuggets"), so `--max-nuggets` values above 30 were silently ignored by the model and only acted as a post-hoc truncation.

## Migration Notes

- Prompt semantics change: runs that pass `--max-nuggets N` (or `max_nuggets=N` / `creator_max_nuggets=N`) with `N != 30` now render "at most N nuggets" in the creator prompt. Runs using the default of 30 are unaffected. Re-running a pool with a non-default cap can produce a different number of nuggets than before.
