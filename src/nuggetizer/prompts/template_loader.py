"""
Template loader for YAML-based prompts
"""

from dataclasses import dataclass
from string import Formatter
from pathlib import Path
from typing import Any, Dict

import yaml

# template cache to avoid reloading files
_template_cache: Dict[str, Dict[str, Any]] = {}


@dataclass(frozen=True)
class PromptTemplate:
    method: str
    system_message: str
    prefix_user: str
    source_path: str

    @property
    def placeholders(self) -> tuple[str, ...]:
        return tuple(
            field_name
            for _, field_name, _, _ in Formatter().parse(self.prefix_user)
            if field_name is not None
        )

    def render(self, **kwargs: Any) -> Dict[str, str]:
        return {
            "system": self.system_message,
            "user": self.prefix_user.format(**kwargs),
        }

    def metadata(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "system_message": self.system_message,
            "prefix_user": self.prefix_user,
            "source_path": self.source_path,
            "placeholders": list(self.placeholders),
        }


def load_template(template_name: str) -> Dict[str, Any]:
    """
    Load a YAML template from prompt_templates directory (cached)
    """
    if template_name not in _template_cache:
        template_dir = Path(__file__).parent / "prompt_templates"
        template_path = template_dir / f"{template_name}.yaml"

        if not template_path.exists():
            raise FileNotFoundError(
                f"Template {template_name} not found at {template_path}"
            )

        with open(template_path, "r", encoding="utf-8") as f:
            _template_cache[template_name] = yaml.safe_load(f)

    return _template_cache[template_name]


def get_template(template_name: str) -> PromptTemplate:
    template = load_template(template_name)
    template_dir = Path(__file__).parent / "prompt_templates"
    template_path = template_dir / f"{template_name}.yaml"
    return PromptTemplate(
        method=str(template["method"]),
        system_message=str(template["system_message"]),
        prefix_user=str(template["prefix_user"]),
        source_path=str(template_path),
    )


def format_template(template_name: str, **kwargs: Any) -> Dict[str, str]:
    """
    Load and format a template with variables
    """
    return get_template(template_name).render(**kwargs)
