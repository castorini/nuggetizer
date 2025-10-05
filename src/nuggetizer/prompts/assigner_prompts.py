"""
Prompts for nugget assignment
"""

from typing import List, Dict
from ..core.types import ScoredNugget, NuggetAssignMode
from .template_loader import format_template

def create_assign_prompt(query: str, context: str, nuggets: List[ScoredNugget], assigner_mode: NuggetAssignMode = NuggetAssignMode.SUPPORT_GRADE_3) -> List[Dict[str, str]]:
    """
    Creates a prompt for nugget assignment using YAML template.
    """

    # choose template based on assignment mode
    template_name = "assigner_template" if assigner_mode == NuggetAssignMode.SUPPORT_GRADE_3 else "assigner_2grade_template"

    # format template with variables
    template_data = format_template(
        template_name,
        query=query,
        context=context,
        nuggets=[nugget.text for nugget in nuggets],
        num_nuggets=len(nuggets)
    )

    return [
        {"role": "system", "content": template_data['system']},
        {"role": "user", "content": template_data['user']}
    ]
