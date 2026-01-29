"""Tests for artifact data models."""

import pytest

from dataknobs_bots.artifacts.models import (
    Artifact,
    ArtifactDefinition,
    ArtifactLineage,
    ArtifactMetadata,
    ArtifactReview,
)


class TestArtifactMetadata:
    """Tests for ArtifactMetadata."""

    def test_default_values(self) -> None:
        """Test default values are set."""
        metadata = ArtifactMetadata()
        assert metadata.created_at > 0
        assert metadata.created_by is None
        assert metadata.stage is None
        assert metadata.tags == []
        assert metadata.custom == {}

    def test_with_values(self) -> None:
        """Test metadata with values."""
        metadata = ArtifactMetadata(
            created_by="test_bot",
            stage="build_questions",
            task_id="task_1",
            purpose="Generate questions",
            tags=["assessment", "math"],
            custom={"priority": "high"},
        )
        assert metadata.created_by == "test_bot"
        assert metadata.stage == "build_questions"
        assert metadata.task_id == "task_1"
        assert metadata.purpose == "Generate questions"
        assert metadata.tags == ["assessment", "math"]
        assert metadata.custom == {"priority": "high"}

    def test_serialization(self) -> None:
        """Test to_dict and from_dict."""
        original = ArtifactMetadata(
            created_by="test_bot",
            stage="build_questions",
            tags=["test"],
        )
        data = original.to_dict()
        restored = ArtifactMetadata.from_dict(data)

        assert restored.created_by == original.created_by
        assert restored.stage == original.stage
        assert restored.tags == original.tags


class TestArtifactLineage:
    """Tests for ArtifactLineage."""

    def test_default_values(self) -> None:
        """Test default values are set."""
        lineage = ArtifactLineage()
        assert lineage.parent_id is None
        assert lineage.source_ids == []
        assert lineage.version == 1
        assert lineage.derived_from is None

    def test_with_values(self) -> None:
        """Test lineage with values."""
        lineage = ArtifactLineage(
            parent_id="art_parent",
            source_ids=["art_source1", "art_source2"],
            version=2,
            derived_from="Updated based on feedback",
        )
        assert lineage.parent_id == "art_parent"
        assert lineage.source_ids == ["art_source1", "art_source2"]
        assert lineage.version == 2
        assert lineage.derived_from == "Updated based on feedback"

    def test_serialization(self) -> None:
        """Test to_dict and from_dict."""
        original = ArtifactLineage(
            parent_id="art_parent",
            version=3,
        )
        data = original.to_dict()
        restored = ArtifactLineage.from_dict(data)

        assert restored.parent_id == original.parent_id
        assert restored.version == original.version


class TestArtifactReview:
    """Tests for ArtifactReview."""

    def test_default_values(self) -> None:
        """Test default values are set."""
        review = ArtifactReview()
        assert review.id.startswith("rev_")
        assert review.artifact_id == ""
        assert review.reviewer == ""
        assert review.review_type == "persona"
        assert review.passed is False
        assert review.score is None
        assert review.feedback == []
        assert review.suggestions == []
        assert review.issues == []
        assert review.timestamp > 0

    def test_with_values(self) -> None:
        """Test review with values."""
        review = ArtifactReview(
            artifact_id="art_123",
            reviewer="adversarial",
            review_type="persona",
            passed=True,
            score=0.85,
            feedback=["Good overall"],
            issues=["Minor issue"],
            suggestions=["Consider edge case"],
        )
        assert review.artifact_id == "art_123"
        assert review.reviewer == "adversarial"
        assert review.passed is True
        assert review.score == 0.85
        assert len(review.feedback) == 1
        assert len(review.issues) == 1
        assert len(review.suggestions) == 1

    def test_serialization(self) -> None:
        """Test to_dict and from_dict."""
        original = ArtifactReview(
            artifact_id="art_123",
            reviewer="skeptical",
            passed=True,
            score=0.9,
            issues=["Test issue"],
        )
        data = original.to_dict()
        restored = ArtifactReview.from_dict(data)

        assert restored.artifact_id == original.artifact_id
        assert restored.reviewer == original.reviewer
        assert restored.passed == original.passed
        assert restored.score == original.score
        assert restored.issues == original.issues


class TestArtifact:
    """Tests for Artifact."""

    def test_default_values(self) -> None:
        """Test default values are set."""
        artifact = Artifact()
        assert artifact.id.startswith("art_")
        assert artifact.type == "content"
        assert artifact.name == ""
        assert artifact.content is None
        assert artifact.content_type == "application/json"
        assert artifact.status == "draft"
        assert artifact.schema_id is None
        assert isinstance(artifact.metadata, ArtifactMetadata)
        assert isinstance(artifact.lineage, ArtifactLineage)
        assert artifact.reviews == []
        assert artifact.definition_id is None

    def test_with_content(self) -> None:
        """Test artifact with content."""
        content = {"questions": [{"id": "q1", "text": "What is 2+2?"}]}
        artifact = Artifact(
            type="content",
            name="Math Questions",
            content=content,
            metadata=ArtifactMetadata(
                stage="build_questions",
                purpose="Generate math questions",
            ),
        )
        assert artifact.name == "Math Questions"
        assert artifact.content == content
        assert artifact.metadata.stage == "build_questions"

    def test_is_approved(self) -> None:
        """Test is_approved property."""
        artifact = Artifact(status="draft")
        assert artifact.is_approved is False

        artifact.status = "approved"
        assert artifact.is_approved is True

    def test_is_reviewable(self) -> None:
        """Test is_reviewable property."""
        # Reviewable statuses
        for status in ["draft", "pending_review", "needs_revision"]:
            artifact = Artifact(status=status)  # type: ignore[arg-type]
            assert artifact.is_reviewable is True

        # Non-reviewable statuses
        for status in ["approved", "rejected", "superseded", "in_review"]:
            artifact = Artifact(status=status)  # type: ignore[arg-type]
            assert artifact.is_reviewable is False

    def test_latest_review(self) -> None:
        """Test latest_review property."""
        artifact = Artifact()
        assert artifact.latest_review is None

        review1 = ArtifactReview(reviewer="reviewer1", passed=False)
        review2 = ArtifactReview(reviewer="reviewer2", passed=True)
        artifact.reviews = [review1, review2]

        assert artifact.latest_review == review2

    def test_serialization(self) -> None:
        """Test to_dict and from_dict."""
        original = Artifact(
            type="config",
            name="Bot Config",
            content={"name": "TestBot"},
            status="approved",
            metadata=ArtifactMetadata(stage="finalize"),
            lineage=ArtifactLineage(version=2, parent_id="art_old"),
            reviews=[ArtifactReview(reviewer="validation", passed=True)],
            definition_id="bot_config",
        )
        data = original.to_dict()
        restored = Artifact.from_dict(data)

        assert restored.id == original.id
        assert restored.type == original.type
        assert restored.name == original.name
        assert restored.content == original.content
        assert restored.status == original.status
        assert restored.metadata.stage == original.metadata.stage
        assert restored.lineage.version == original.lineage.version
        assert len(restored.reviews) == 1
        assert restored.definition_id == original.definition_id


class TestArtifactDefinition:
    """Tests for ArtifactDefinition."""

    def test_default_values(self) -> None:
        """Test default values are set."""
        definition = ArtifactDefinition(id="test")
        assert definition.id == "test"
        assert definition.type == "content"
        assert definition.name == ""
        assert definition.reviews == []
        assert definition.approval_threshold == 0.8
        assert definition.require_all_reviews is True
        assert definition.auto_submit_for_review is False

    def test_from_config(self) -> None:
        """Test creation from config dict."""
        config = {
            "type": "content",
            "name": "Assessment Questions",
            "description": "Questions for assessment",
            "reviews": ["adversarial", "downstream"],
            "approval_threshold": 0.9,
            "auto_submit_for_review": True,
            "tags": ["assessment"],
        }
        definition = ArtifactDefinition.from_config("questions", config)

        assert definition.id == "questions"
        assert definition.type == "content"
        assert definition.name == "Assessment Questions"
        assert definition.description == "Questions for assessment"
        assert definition.reviews == ["adversarial", "downstream"]
        assert definition.approval_threshold == 0.9
        assert definition.auto_submit_for_review is True
        assert definition.tags == ["assessment"]

    def test_from_config_minimal(self) -> None:
        """Test creation from minimal config."""
        definition = ArtifactDefinition.from_config("minimal", {})
        assert definition.id == "minimal"
        assert definition.name == "minimal"  # Defaults to ID
        assert definition.type == "content"

    def test_serialization(self) -> None:
        """Test to_dict."""
        definition = ArtifactDefinition(
            id="test",
            type="config",
            name="Test Definition",
            reviews=["validation"],
        )
        data = definition.to_dict()

        assert data["id"] == "test"
        assert data["type"] == "config"
        assert data["name"] == "Test Definition"
        assert data["reviews"] == ["validation"]
