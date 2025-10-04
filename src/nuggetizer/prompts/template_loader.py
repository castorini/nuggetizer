import yaml
from typing import Dict, Any
from pathlib import Path

def load_template(template_name: str) -> Dict[str, Any]:
    """
    Load a YAML template from prompt_templates directory
    """
    template_dir = Path(__file__).parent / "prompt_templates"
    template_path = template_dir / f"{template_name}.yaml"
    
    if not template_path.exists():
        raise FileNotFoundError(f"Template {template_name} not found at {template_path}")
    
    with open(template_path, 'r', encoding='utf-8') as f:
        template_data = yaml.safe_load(f)
    
    return template_data

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
