"""
Prompts for nugget scoring
"""

from typing import Dict, List

from ..core.types import Nugget
from .template_loader import format_template


def create_score_prompt(query: str, nuggets: List[Nugget]) -> List[Dict[str, str]]:
    """
    Creates a prompt for nugget scoring using YAML template.
    """

    # format template with variables
    template_data = format_template(
        "scorer_template",
        query=query,
        nuggets=[nugget.text for nugget in nuggets],
        num_nuggets=len(nuggets),
    )

    return [
        {"role": "system", "content": template_data["system"]},
        {"role": "user", "content": template_data["user"]},
    ]
