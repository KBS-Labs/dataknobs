"""Tests for artifact status transitions."""

from __future__ import annotations

import pytest
from dataknobs_common.transitions import InvalidTransitionError

from dataknobs_bots.artifacts.models import ArtifactStatus
from dataknobs_bots.artifacts.transitions import ARTIFACT_STATUS, validate_transition


class TestArtifactStatusTransitions:
    def test_draft_to_pending_review(self) -> None:
        validate_transition(ArtifactStatus.DRAFT, ArtifactStatus.PENDING_REVIEW)

    def test_draft_to_archived(self) -> None:
        validate_transition(ArtifactStatus.DRAFT, ArtifactStatus.ARCHIVED)

    def test_pending_review_to_in_review(self) -> None:
        validate_transition(
            ArtifactStatus.PENDING_REVIEW, ArtifactStatus.IN_REVIEW
        )

    def test_pending_review_back_to_draft(self) -> None:
        validate_transition(
            ArtifactStatus.PENDING_REVIEW, ArtifactStatus.DRAFT
        )

    def test_in_review_to_approved(self) -> None:
        validate_transition(
            ArtifactStatus.IN_REVIEW, ArtifactStatus.APPROVED
        )

    def test_in_review_to_needs_revision(self) -> None:
        validate_transition(
            ArtifactStatus.IN_REVIEW, ArtifactStatus.NEEDS_REVISION
        )

    def test_in_review_to_rejected(self) -> None:
        validate_transition(
            ArtifactStatus.IN_REVIEW, ArtifactStatus.REJECTED
        )

    def test_draft_to_superseded(self) -> None:
        validate_transition(ArtifactStatus.DRAFT, ArtifactStatus.SUPERSEDED)

    def test_needs_revision_to_draft(self) -> None:
        validate_transition(
            ArtifactStatus.NEEDS_REVISION, ArtifactStatus.DRAFT
        )

    def test_needs_revision_to_superseded(self) -> None:
        validate_transition(
            ArtifactStatus.NEEDS_REVISION, ArtifactStatus.SUPERSEDED
        )

    def test_approved_to_superseded(self) -> None:
        validate_transition(
            ArtifactStatus.APPROVED, ArtifactStatus.SUPERSEDED
        )

    def test_rejected_to_archived(self) -> None:
        validate_transition(
            ArtifactStatus.REJECTED, ArtifactStatus.ARCHIVED
        )

    def test_invalid_draft_to_approved(self) -> None:
        with pytest.raises(InvalidTransitionError):
            validate_transition(
                ArtifactStatus.DRAFT, ArtifactStatus.APPROVED
            )

    def test_invalid_approved_to_draft(self) -> None:
        with pytest.raises(InvalidTransitionError):
            validate_transition(
                ArtifactStatus.APPROVED, ArtifactStatus.DRAFT
            )

    def test_invalid_archived_to_anything(self) -> None:
        for target in ArtifactStatus:
            if target == ArtifactStatus.ARCHIVED:
                continue
            with pytest.raises(InvalidTransitionError):
                validate_transition(ArtifactStatus.ARCHIVED, target)

    def test_archived_is_terminal(self) -> None:
        reachable = ARTIFACT_STATUS.get_reachable("archived")
        assert reachable == set()

    def test_all_states_can_reach_archived(self) -> None:
        for status in ArtifactStatus:
            if status == ArtifactStatus.ARCHIVED:
                continue
            reachable = ARTIFACT_STATUS.get_reachable(status.value)
            assert "archived" in reachable, (
                f"{status.value} cannot reach archived"
            )
