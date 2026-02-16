"""Artifact status transition rules.

This module defines the valid status transitions for artifacts using
the TransitionValidator from dataknobs-common. Invalid transitions
raise InvalidTransitionError.

Example:
    >>> from dataknobs_bots.artifacts.transitions import validate_transition
    >>> from dataknobs_bots.artifacts.models import ArtifactStatus
    >>> validate_transition(ArtifactStatus.DRAFT, ArtifactStatus.PENDING_REVIEW)
"""

from __future__ import annotations

from dataknobs_common.transitions import TransitionValidator

from .models import ArtifactStatus

ARTIFACT_STATUS = TransitionValidator(
    "artifact_status",
    {
        "draft": {"pending_review", "archived", "superseded"},
        "pending_review": {"in_review", "draft", "archived"},
        "in_review": {"approved", "needs_revision", "rejected"},
        "needs_revision": {"draft", "archived", "superseded"},
        "approved": {"superseded", "archived"},
        "rejected": {"archived"},
        "superseded": {"archived"},
        "archived": set(),
    },
)


def validate_transition(
    current: ArtifactStatus, target: ArtifactStatus
) -> None:
    """Validate an artifact status transition.

    Args:
        current: The current status.
        target: The desired target status.

    Raises:
        InvalidTransitionError: If the transition is not allowed.
    """
    ARTIFACT_STATUS.validate(current.value, target.value)
