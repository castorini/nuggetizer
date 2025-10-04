"""
Prompts for nugget creation
"""

from typing import List, Dict
from ..core.types import Request
from .template_loader import format_template

def create_nugget_prompt(request: Request, start: int, end: int, nuggets: List[str], creator_max_nuggets: int = 30) -> List[Dict[str, str]]:
    """
    Creates a prompt for nugget creation using YAML template.
    """
    
    # prepare context from docs
    context = "\n".join([
        f"[{i+1}] {doc.segment}" 
        for i, doc in enumerate(request.documents[start:end])
    ])
    
    # format template with variables
    template_data = format_template(
        "creator_template",
        query=request.query.text,
        context=context,
        nuggets=nuggets,
        nuggets_length=len(nuggets),
        creator_max_nuggets=creator_max_nuggets
    )
    
    return [
        {"role": "system", "content": template_data['system']},
        {"role": "user", "content": template_data['user']}
    ]

def get_nugget_prompt_content(request: Request, start: int, end: int, nuggets: List[str], creator_max_nuggets: int = 30) -> str:
    """
    Gets the content for the nugget creation prompt
    """
    messages = create_nugget_prompt(request, start, end, nuggets, creator_max_nuggets)
    return messages[1]['content']
