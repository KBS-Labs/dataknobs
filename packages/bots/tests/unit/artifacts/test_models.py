"""Tests for revised artifact data models."""

from __future__ import annotations

from dataknobs_bots.artifacts.models import (
    Artifact,
    ArtifactStatus,
    ArtifactTypeDefinition,
)
from dataknobs_bots.artifacts.provenance import (
    ProvenanceRecord,
    SourceReference,
    create_provenance,
)


class TestArtifactStatus:
    def test_enum_values(self) -> None:
        assert ArtifactStatus.DRAFT.value == "draft"
        assert ArtifactStatus.PENDING_REVIEW.value == "pending_review"
        assert ArtifactStatus.IN_REVIEW.value == "in_review"
        assert ArtifactStatus.NEEDS_REVISION.value == "needs_revision"
        assert ArtifactStatus.APPROVED.value == "approved"
        assert ArtifactStatus.REJECTED.value == "rejected"
        assert ArtifactStatus.SUPERSEDED.value == "superseded"
        assert ArtifactStatus.ARCHIVED.value == "archived"

    def test_str_enum(self) -> None:
        assert ArtifactStatus("draft") == ArtifactStatus.DRAFT
        assert ArtifactStatus("approved") == ArtifactStatus.APPROVED


class TestArtifact:
    def test_default_values(self) -> None:
        artifact = Artifact()
        assert artifact.id.startswith("art_")
        assert artifact.type == "content"
        assert artifact.name == ""
        assert artifact.version == "1.0.0"
        assert artifact.status == ArtifactStatus.DRAFT
        assert artifact.content == {}
        assert artifact.content_schema is None
        assert isinstance(artifact.provenance, ProvenanceRecord)
        assert artifact.tags == []
        assert artifact.rubric_ids == []
        assert artifact.evaluation_ids == []
        assert artifact.created_at != ""
        assert artifact.updated_at != ""

    def test_with_content(self) -> None:
        content = {"questions": [{"id": "q1", "text": "What is 2+2?"}]}
        provenance = create_provenance(
            created_by="bot:edubot",
            creation_method="generator",
        )
        artifact = Artifact(
            type="assessment",
            name="Math Questions",
            content=content,
            provenance=provenance,
            tags=["math", "assessment"],
        )
        assert artifact.name == "Math Questions"
        assert artifact.content == content
        assert artifact.provenance.created_by == "bot:edubot"
        assert artifact.tags == ["math", "assessment"]

    def test_is_approved(self) -> None:
        artifact = Artifact(status=ArtifactStatus.DRAFT)
        assert artifact.is_approved is False

        artifact.status = ArtifactStatus.APPROVED
        assert artifact.is_approved is True

    def test_is_reviewable(self) -> None:
        reviewable = [
            ArtifactStatus.DRAFT,
            ArtifactStatus.PENDING_REVIEW,
            ArtifactStatus.NEEDS_REVISION,
        ]
        for status in reviewable:
            artifact = Artifact(status=status)
            assert artifact.is_reviewable is True, f"{status} should be reviewable"

        non_reviewable = [
            ArtifactStatus.IN_REVIEW,
            ArtifactStatus.APPROVED,
            ArtifactStatus.REJECTED,
            ArtifactStatus.SUPERSEDED,
            ArtifactStatus.ARCHIVED,
        ]
        for status in non_reviewable:
            artifact = Artifact(status=status)
            assert artifact.is_reviewable is False, f"{status} should not be reviewable"

    def test_serialization_round_trip(self) -> None:
        original = Artifact(
            id="art_test123",
            type="config",
            name="Bot Config",
            version="2.0.0",
            status=ArtifactStatus.APPROVED,
            content={"name": "TestBot"},
            provenance=create_provenance(
                created_by="user:jane",
                creation_method="manual",
                sources=[
                    SourceReference(source_id="doc_1", source_type="document"),
                ],
            ),
            tags=["config"],
            rubric_ids=["rubric_001"],
            evaluation_ids=["eval_001"],
        )
        d = original.to_dict()
        restored = Artifact.from_dict(d)

        assert restored.id == original.id
        assert restored.type == original.type
        assert restored.name == original.name
        assert restored.version == original.version
        assert restored.status == original.status
        assert restored.content == original.content
        assert restored.provenance.created_by == "user:jane"
        assert len(restored.provenance.sources) == 1
        assert restored.tags == original.tags
        assert restored.rubric_ids == original.rubric_ids
        assert restored.evaluation_ids == original.evaluation_ids

    def test_unique_ids_generated(self) -> None:
        a1 = Artifact()
        a2 = Artifact()
        assert a1.id != a2.id


class TestArtifactTypeDefinition:
    def test_default_values(self) -> None:
        defn = ArtifactTypeDefinition(id="test")
        assert defn.id == "test"
        assert defn.description == ""
        assert defn.rubrics == []
        assert defn.auto_review is False
        assert defn.requires_approval is False
        assert defn.approval_threshold == 0.7

    def test_from_config(self) -> None:
        config = {
            "description": "Questions for assessment",
            "rubrics": ["content_quality", "pedagogical_quality"],
            "auto_review": True,
            "requires_approval": True,
            "approval_threshold": 0.8,
            "tags": ["assessment"],
        }
        defn = ArtifactTypeDefinition.from_config("questions", config)

        assert defn.id == "questions"
        assert defn.description == "Questions for assessment"
        assert defn.rubrics == ["content_quality", "pedagogical_quality"]
        assert defn.auto_review is True
        assert defn.requires_approval is True
        assert defn.approval_threshold == 0.8
        assert defn.tags == ["assessment"]

    def test_from_config_minimal(self) -> None:
        defn = ArtifactTypeDefinition.from_config("minimal", {})
        assert defn.id == "minimal"
        assert defn.description == ""
        assert defn.rubrics == []

    def test_serialization(self) -> None:
        defn = ArtifactTypeDefinition(
            id="test",
            description="Test type",
            rubrics=["quality"],
            auto_review=True,
        )
        d = defn.to_dict()
        assert d["id"] == "test"
        assert d["description"] == "Test type"
        assert d["rubrics"] == ["quality"]
        assert d["auto_review"] is True
