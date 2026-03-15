from __future__ import annotations

from typing import Any

from nuggetizer.core.types import NuggetAssignMode
from nuggetizer.prompts.template_loader import PromptTemplate, get_template


PROMPT_TARGETS = {
    "create": "creator_template",
    "score": "scorer_template",
}


def resolve_prompt_template(
    target: str, assign_mode: NuggetAssignMode = NuggetAssignMode.SUPPORT_GRADE_3
) -> tuple[str, PromptTemplate]:
    if target == "assign":
        template_name = (
            "assigner_template"
            if assign_mode == NuggetAssignMode.SUPPORT_GRADE_3
            else "assigner_2grade_template"
        )
        return template_name, get_template(template_name)
    template_name = PROMPT_TARGETS[target]
    return template_name, get_template(template_name)


def list_prompt_templates() -> list[dict[str, Any]]:
    catalog: list[dict[str, Any]] = []
    for target in ("create", "score"):
        template_name, template = resolve_prompt_template(target)
        catalog.append(
            {
                "target": target,
                "template_name": template_name,
                "assign_mode": None,
                "template": template.metadata(),
            }
        )
    for assign_mode in (
        NuggetAssignMode.SUPPORT_GRADE_3,
        NuggetAssignMode.SUPPORT_GRADE_2,
    ):
        template_name, template = resolve_prompt_template("assign", assign_mode)
        catalog.append(
            {
                "target": "assign",
                "template_name": template_name,
                "assign_mode": assign_mode.value,
                "template": template.metadata(),
            }
        )
    return catalog


def build_prompt_template_view(
    target: str,
    template_name: str,
    template: PromptTemplate,
    *,
    assign_mode: NuggetAssignMode | None,
) -> dict[str, Any]:
    return {
        "target": target,
        "template_name": template_name,
        "assign_mode": None if assign_mode is None else assign_mode.value,
        "template": template.metadata(),
    }


def render_prompt_catalog_text(catalog: list[dict[str, Any]]) -> str:
    lines = ["Nuggetizer Prompt Catalog"]
    for entry in catalog:
        lines.append("")
        lines.append(f"- target: {entry['target']}")
        if entry["assign_mode"] is not None:
            lines.append(f"  assign_mode: {entry['assign_mode']}")
        lines.append(f"  template_name: {entry['template_name']}")
        lines.append(f"  method: {entry['template']['method']}")
        lines.append(f"  source: {entry['template']['source_path']}")
        lines.append(
            "  placeholders: "
            + (
                ", ".join(entry["template"]["placeholders"])
                if entry["template"]["placeholders"]
                else "(none)"
            )
        )
    return "\n".join(lines)


def render_prompt_template_text(view: dict[str, Any]) -> str:
    lines = ["Nuggetizer Prompt Template"]
    lines.append(f"target: {view['target']}")
    if view["assign_mode"] is not None:
        lines.append(f"assign_mode: {view['assign_mode']}")
    lines.append(f"template_name: {view['template_name']}")
    lines.append(f"method: {view['template']['method']}")
    lines.append(f"source: {view['template']['source_path']}")
    lines.append(
        "placeholders: "
        + (
            ", ".join(view["template"]["placeholders"])
            if view["template"]["placeholders"]
            else "(none)"
        )
    )
    lines.append("")
    lines.append("[system]")
    lines.append(view["template"]["system_message"] or "(empty)")
    lines.append("")
    lines.append("[user]")
    lines.append(view["template"]["prefix_user"])
    return "\n".join(lines)


def build_rendered_prompt_view(
    target: str,
    template_name: str,
    messages: list[dict[str, str]],
    *,
    assign_mode: NuggetAssignMode | None,
    inputs: dict[str, Any],
) -> dict[str, Any]:
    return {
        "target": target,
        "template_name": template_name,
        "assign_mode": None if assign_mode is None else assign_mode.value,
        "messages": messages,
        "inputs": inputs,
    }


def render_rendered_prompt_text(view: dict[str, Any], *, part: str) -> str:
    lines = ["Nuggetizer Rendered Prompt"]
    lines.append(f"target: {view['target']}")
    if view["assign_mode"] is not None:
        lines.append(f"assign_mode: {view['assign_mode']}")
    lines.append(f"template_name: {view['template_name']}")
    for name, value in view["inputs"].items():
        lines.append(f"{name}: {value}")
    for message in view["messages"]:
        if part != "all" and message["role"] != part:
            continue
        lines.append("")
        lines.append(f"[{message['role']}]")
        lines.append(message["content"])
    return "\n".join(lines)
