from ..core.types import Request
from .service import create_nugget_messages


def create_nugget_prompt(
    request: Request,
    start: int,
    end: int,
    nuggets: list[str],
    creator_max_nuggets: int = 30,
) -> list[dict[str, str]]:
    return create_nugget_messages(
        request,
        start,
        end,
        nuggets,
        creator_max_nuggets=creator_max_nuggets,
    )
