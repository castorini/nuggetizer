from __future__ import annotations

from nuggetizer.prompts.template_loader import format_template, get_template


def test_get_template_metadata_reports_creator_template_details() -> None:
    template = get_template("creator_template")

    assert template.method == "nugget_creation"
    assert template.source_path.endswith("prompt_templates/creator_template.yaml")
    assert template.placeholders == (
        "creator_max_nuggets",
        "query",
        "context",
        "query",
        "nuggets",
        "nuggets_length",
    )
    assert template.metadata()["placeholders"] == [
        "creator_max_nuggets",
        "query",
        "context",
        "query",
        "nuggets",
        "nuggets_length",
    ]


def test_format_template_renders_assigner_template() -> None:
    rendered = format_template(
        "assigner_template",
        query="What is Python used for?",
        context="Python is commonly used for web development.",
        nuggets=["Python is used for web development."],
        num_nuggets=1,
    )

    assert rendered["system"] == (
        "You are NuggetizeAssignerLLM, an intelligent assistant that can "
        "label a list of atomic nuggets based on if they are captured by a given passage."
    )
    assert "Search Query: What is Python used for?" in rendered["user"]
    assert "Passage: Python is commonly used for web development." in rendered["user"]
    assert "Nugget List: ['Python is used for web development.']" in rendered["user"]


def test_get_template_reports_assigner_2grade_placeholders() -> None:
    template = get_template("assigner_2grade_template")

    assert template.method == "nugget_assignment_2grade"
    assert template.placeholders == ("num_nuggets", "query", "context", "nuggets")
