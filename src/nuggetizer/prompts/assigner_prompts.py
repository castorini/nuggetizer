from ..core.types import NuggetAssignMode, ScoredNugget
from .service import create_assign_messages


def create_assign_prompt(
    query: str,
    context: str,
    nuggets: list[ScoredNugget],
    assigner_mode: NuggetAssignMode = NuggetAssignMode.SUPPORT_GRADE_3,
) -> list[dict[str, str]]:
    return create_assign_messages(query, context, nuggets, assigner_mode)
