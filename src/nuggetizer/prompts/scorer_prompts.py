"""
Prompts for nugget scoring
"""

from typing import List, Dict
from ..core.types import Nugget

def create_score_prompt(query: str, nuggets: List[Nugget], use_yaml: bool = False) -> List[Dict[str, str]]:
    """
    Creates a prompt for nugget scoring. Set use_yaml=True to use YAML templates.
    """
    if use_yaml:
        from .template_loader import format_template
        
        # format template with variables
        template_data = format_template(
            "scorer_template",
            query=query,
            nuggets=[nugget.text for nugget in nuggets],
            num_nuggets=len(nuggets)
        )
        
        return [
            {"role": "system", "content": template_data['system']},
            {"role": "user", "content": template_data['user']}
        ]
    else:
        messages = [
            {
                "role": "system",
                "content": "You are NuggetizeScoreLLM, an intelligent assistant that can label a list of atomic nuggets based on their importance for a given search query."
            },
            {
                "role": "user",
                "content": f"""Based on the query, label each of the {len(nuggets)} nuggets either a vital or okay based on the following criteria. Vital nuggets represent concepts that must be present in a "good" answer; on the other hand, okay nuggets contribute worthwhile information about the target but are not essential. Return the list of labels in a Pythonic list format (type: List[str]). The list should be in the same order as the input nuggets. Make sure to provide a label for each nugget.

Search Query: {query}
Nugget List: {[nugget.text for nugget in nuggets]}

Only return the list of labels (List[str]). Do not explain.
Labels:"""
        }
    ]
    return messages 