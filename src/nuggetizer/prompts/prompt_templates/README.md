# YAML Prompt Templates

Use with `use_yaml=True` flag.

- `creator_template.yaml`for nugget creation
- `scorer_template.yaml` for nugget scoring  
- `assigner_template.yaml`for 3 grade assignment
- `assigner_2grade_template.yaml` for 2 grade assignment

```python
create_nugget_prompt(request, start, end, nuggets, use_yaml=True)
create_score_prompt(query, nuggets, use_yaml=True)
create_assign_prompt(query, context, nuggets, use_yaml=True)
```