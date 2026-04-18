from ..core.types import Nugget
from .service import create_score_messages


def create_score_prompt(query: str, nuggets: list[Nugget]) -> list[dict[str, str]]:
    return create_score_messages(query, nuggets)
