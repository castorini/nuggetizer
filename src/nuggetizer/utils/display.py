"""Display utilities for pretty printing nuggets and assignments."""

from collections.abc import Sequence

from nuggetizer.core.types import (
    AssignedNugget,
    AssignedScoredNugget,
    Document,
    Nugget,
    ScoredNugget,
)


def print_nuggets(
    nuggets: Sequence[Nugget | ScoredNugget],
    numbered: bool = True,
    show_importance: bool = True,
) -> None:
    """Pretty print a list of nuggets or scored nuggets.

    Args:
        nuggets: List of Nugget or ScoredNugget objects to print
        numbered: Whether to number the nuggets (default: True)
        show_importance: Whether to show importance for ScoredNuggets (default: True)
    """
    for i, nugget in enumerate(nuggets, 1):
        prefix = f"{i}. " if numbered else ""

        # Check if this is a ScoredNugget with importance attribute
        if show_importance and hasattr(nugget, "importance"):
            importance_emoji = "🥇" if nugget.importance == "vital" else "🥈"
            print(
                f"{prefix}{importance_emoji} {nugget.text} (Importance: {nugget.importance})"
            )
        else:
            print(f"{prefix}{nugget.text}")


def print_assigned_nuggets(
    document: Document,
    assigned_nuggets: Sequence[AssignedNugget | AssignedScoredNugget],
    show_importance: bool = True,
) -> None:
    """Pretty print assigned nuggets for a document.

    Args:
        document: Document object containing docid and segment
        assigned_nuggets: List of AssignedNugget or AssignedScoredNugget objects
        show_importance: Whether to show importance for AssignedScoredNuggets (default: True)
    """
    print(f"\nAssignments for document: {document.docid}")
    print(f"Segment: {document.segment}")
    for nugget in assigned_nuggets:
        # Determine assignment emoji
        assignment_emoji = {
            "support": "✅",
            "partial_support": "🟡",
            "not_support": "❌",
        }.get(nugget.assignment, "❓")

        print(f"{nugget.text}")

        # Show importance if available and requested
        if show_importance and hasattr(nugget, "importance"):
            importance_emoji = "🥇" if nugget.importance == "vital" else "🥈"
            print(f"  Importance: {nugget.importance} {importance_emoji}")

        print(f"  Assignment: {nugget.assignment} {assignment_emoji}")
