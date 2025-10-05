"""
Template loader for YAML-based prompts
"""

import yaml
from typing import Dict, Any
from pathlib import Path

# template cache to avoid reloading files
_template_cache: Dict[str, Dict[str, Any]] = {}

def load_template(template_name: str) -> Dict[str, Any]:
    """
    Load a YAML template from prompt_templates directory (cached)
    """
    if template_name not in _template_cache:
        template_dir = Path(__file__).parent / "prompt_templates"
        template_path = template_dir / f"{template_name}.yaml"

        if not template_path.exists():
            raise FileNotFoundError(f"Template {template_name} not found at {template_path}")

        with open(template_path, 'r', encoding='utf-8') as f:
            _template_cache[template_name] = yaml.safe_load(f)

    return _template_cache[template_name]

def format_template(template_name: str, **kwargs) -> Dict[str, str]:
    """
    Load and format a template with variables
    """
    template = load_template(template_name)
    user_content = template['prefix_user'].format(**kwargs)

    return {
        'system': template['system_message'],
        'user': user_content
    }