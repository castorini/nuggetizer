from __future__ import annotations

from typing import Any

from nuggetizer.core.types import Nugget, NuggetAssignMode, Request, ScoredNugget

from .template_loader import PromptTemplate, format_template, get_template

PROMPT_TARGETS = {
    "create": "creator_template",
    "score": "scorer_template",
}


def resolve_template_name(
    target: str, assign_mode: NuggetAssignMode = NuggetAssignMode.SUPPORT_GRADE_3
) -> str:
    if target == "assign":
        return (
            "assigner_template"
            if assign_mode == NuggetAssignMode.SUPPORT_GRADE_3
            else "assigner_2grade_template"
        )
    return PROMPT_TARGETS[target]


def resolve_template(
    target: str, assign_mode: NuggetAssignMode = NuggetAssignMode.SUPPORT_GRADE_3
) -> tuple[str, PromptTemplate]:
    template_name = resolve_template_name(target, assign_mode)
    return template_name, get_template(template_name)


def render_messages(template_name: str, **kwargs: Any) -> list[dict[str, str]]:
    template_data = format_template(template_name, **kwargs)
    return [
        {"role": "system", "content": template_data["system"]},
        {"role": "user", "content": template_data["user"]},
    ]


def create_nugget_messages(
    request: Request,
    start: int,
    end: int,
    nuggets: list[str],
    creator_max_nuggets: int = 30,
) -> list[dict[str, str]]:
    context = "\n".join(
        f"[{i + 1}] {doc.segment}" for i, doc in enumerate(request.documents[start:end])
    )
    return render_messages(
        "creator_template",
        query=request.query.text,
        context=context,
        nuggets=nuggets,
        nuggets_length=len(nuggets),
        creator_max_nuggets=creator_max_nuggets,
    )


def create_score_messages(query: str, nuggets: list[Nugget]) -> list[dict[str, str]]:
    return render_messages(
        "scorer_template",
        query=query,
        nuggets=[nugget.text for nugget in nuggets],
        num_nuggets=len(nuggets),
    )


def create_assign_messages(
    query: str,
    context: str,
    nuggets: list[ScoredNugget],
    assigner_mode: NuggetAssignMode = NuggetAssignMode.SUPPORT_GRADE_3,
) -> list[dict[str, str]]:
    return render_messages(
        resolve_template_name("assign", assigner_mode),
        query=query,
        context=context,
        nuggets=[nugget.text for nugget in nuggets],
        num_nuggets=len(nuggets),
    )
