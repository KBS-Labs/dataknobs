"""Tests for artifact registry."""

import pytest

from dataknobs_bots.artifacts.models import (
    Artifact,
    ArtifactDefinition,
    ArtifactReview,
)
from dataknobs_bots.artifacts.registry import ArtifactRegistry


class TestArtifactRegistryBasic:
    """Basic tests for ArtifactRegistry."""

    def test_init_empty(self) -> None:
        """Test empty initialization."""
        registry = ArtifactRegistry()
        assert len(registry.get_all()) == 0
        assert registry.get_available_definitions() == []

    def test_init_with_definitions(self) -> None:
        """Test initialization with definitions."""
        definitions = {
            "questions": ArtifactDefinition(
                id="questions",
                type="content",
                reviews=["adversarial"],
            )
        }
        registry = ArtifactRegistry(definitions=definitions)
        assert "questions" in registry.get_available_definitions()

    def test_from_config(self) -> None:
        """Test creation from config dict."""
        config = {
            "artifacts": {
                "definitions": {
                    "questions": {
                        "type": "content",
                        "reviews": ["adversarial"],
                    },
                    "config": {
                        "type": "config",
                        "reviews": ["validation"],
                    },
                }
            }
        }
        registry = ArtifactRegistry.from_config(config)
        assert "questions" in registry.get_available_definitions()
        assert "config" in registry.get_available_definitions()


class TestArtifactRegistryCreation:
    """Tests for artifact creation."""

    def test_create_simple(self) -> None:
        """Test simple artifact creation."""
        registry = ArtifactRegistry()
        content = {"key": "value"}
        artifact = registry.create(content=content, name="Test Artifact")

        assert artifact.id.startswith("art_")
        assert artifact.content == content
        assert artifact.name == "Test Artifact"
        assert artifact.status == "draft"
        assert registry.get(artifact.id) == artifact

    def test_create_with_metadata(self) -> None:
        """Test artifact creation with metadata."""
        registry = ArtifactRegistry()
        artifact = registry.create(
            content={"data": "test"},
            name="Test",
            stage="build_stage",
            task_id="task_1",
            purpose="Testing",
            tags=["test", "demo"],
        )

        assert artifact.metadata.stage == "build_stage"
        assert artifact.metadata.task_id == "task_1"
        assert artifact.metadata.purpose == "Testing"
        assert set(artifact.metadata.tags) == {"test", "demo"}

    def test_create_with_definition(self) -> None:
        """Test artifact creation using a definition."""
        definitions = {
            "questions": ArtifactDefinition(
                id="questions",
                type="content",
                name="Assessment Questions",
                tags=["assessment"],
            )
        }
        registry = ArtifactRegistry(definitions=definitions)

        artifact = registry.create(
            content={"questions": []},
            definition_id="questions",
        )

        assert artifact.type == "content"
        assert artifact.name == "Assessment Questions"
        assert artifact.definition_id == "questions"
        assert "assessment" in artifact.metadata.tags

    def test_create_with_definition_auto_submit(self) -> None:
        """Test artifact creation with auto-submit for review."""
        definitions = {
            "auto_review": ArtifactDefinition(
                id="auto_review",
                auto_submit_for_review=True,
            )
        }
        registry = ArtifactRegistry(definitions=definitions)

        artifact = registry.create(
            content={"data": "test"},
            definition_id="auto_review",
        )

        assert artifact.status == "pending_review"


class TestArtifactRegistryUpdate:
    """Tests for artifact updates."""

    def test_update_creates_new_version(self) -> None:
        """Test that update creates a new version."""
        registry = ArtifactRegistry()
        original = registry.create(content={"v": 1}, name="Test")
        updated = registry.update(original.id, content={"v": 2})

        assert updated.id != original.id
        assert updated.content == {"v": 2}
        assert updated.lineage.version == 2
        assert updated.lineage.parent_id == original.id
        assert original.status == "superseded"

    def test_update_preserves_metadata(self) -> None:
        """Test that update preserves metadata."""
        registry = ArtifactRegistry()
        original = registry.create(
            content={"v": 1},
            name="Test",
            stage="test_stage",
            purpose="Testing",
        )
        updated = registry.update(original.id, content={"v": 2})

        assert updated.name == original.name
        assert updated.metadata.stage == original.metadata.stage
        assert updated.metadata.purpose == original.metadata.purpose

    def test_update_with_derived_from(self) -> None:
        """Test update with derived_from description."""
        registry = ArtifactRegistry()
        original = registry.create(content={"v": 1}, name="Test")
        updated = registry.update(
            original.id,
            content={"v": 2},
            derived_from="Fixed typo",
        )

        assert updated.lineage.derived_from == "Fixed typo"

    def test_update_not_found_raises(self) -> None:
        """Test that updating non-existent artifact raises."""
        registry = ArtifactRegistry()
        with pytest.raises(KeyError):
            registry.update("nonexistent", content={"data": "test"})


class TestArtifactRegistryStatus:
    """Tests for status management."""

    def test_submit_for_review(self) -> None:
        """Test submitting artifact for review."""
        registry = ArtifactRegistry()
        artifact = registry.create(content={"data": "test"}, name="Test")

        assert artifact.status == "draft"
        registry.submit_for_review(artifact.id)
        assert artifact.status == "pending_review"

    def test_submit_for_review_not_found_raises(self) -> None:
        """Test that submitting non-existent artifact raises."""
        registry = ArtifactRegistry()
        with pytest.raises(KeyError):
            registry.submit_for_review("nonexistent")

    def test_submit_for_review_non_reviewable_raises(self) -> None:
        """Test that submitting non-reviewable artifact raises."""
        registry = ArtifactRegistry()
        artifact = registry.create(content={"data": "test"}, name="Test")
        artifact.status = "approved"

        with pytest.raises(ValueError):
            registry.submit_for_review(artifact.id)

    def test_set_status(self) -> None:
        """Test setting artifact status directly."""
        registry = ArtifactRegistry()
        artifact = registry.create(content={"data": "test"}, name="Test")

        registry.set_status(artifact.id, "approved")
        assert artifact.status == "approved"


class TestArtifactRegistryReviews:
    """Tests for review management."""

    def test_add_review(self) -> None:
        """Test adding a review to artifact."""
        registry = ArtifactRegistry()
        artifact = registry.create(content={"data": "test"}, name="Test")

        review = ArtifactReview(
            reviewer="adversarial",
            passed=True,
            score=0.9,
        )
        registry.add_review(artifact.id, review)

        assert len(artifact.reviews) == 1
        assert artifact.reviews[0].reviewer == "adversarial"
        assert artifact.reviews[0].artifact_id == artifact.id

    def test_add_review_not_found_raises(self) -> None:
        """Test that adding review to non-existent artifact raises."""
        registry = ArtifactRegistry()
        review = ArtifactReview(reviewer="test", passed=True)

        with pytest.raises(KeyError):
            registry.add_review("nonexistent", review)

    def test_add_review_auto_approval(self) -> None:
        """Test automatic approval when all reviews pass."""
        definitions = {
            "test": ArtifactDefinition(
                id="test",
                reviews=["reviewer1", "reviewer2"],
                require_all_reviews=True,
            )
        }
        registry = ArtifactRegistry(definitions=definitions)
        artifact = registry.create(content={"data": "test"}, definition_id="test")
        registry.submit_for_review(artifact.id)

        # Add first review - not approved yet
        review1 = ArtifactReview(reviewer="reviewer1", passed=True)
        registry.add_review(artifact.id, review1)
        assert artifact.status != "approved"

        # Add second review - should be approved now
        review2 = ArtifactReview(reviewer="reviewer2", passed=True)
        registry.add_review(artifact.id, review2)
        assert artifact.status == "approved"

    def test_add_review_needs_revision(self) -> None:
        """Test that failed review sets needs_revision status."""
        definitions = {
            "test": ArtifactDefinition(
                id="test",
                reviews=["reviewer1"],
                require_all_reviews=True,
            )
        }
        registry = ArtifactRegistry(definitions=definitions)
        artifact = registry.create(content={"data": "test"}, definition_id="test")
        registry.submit_for_review(artifact.id)

        review = ArtifactReview(reviewer="reviewer1", passed=False)
        registry.add_review(artifact.id, review)
        assert artifact.status == "needs_revision"


class TestArtifactRegistryQueries:
    """Tests for artifact queries."""

    def test_get(self) -> None:
        """Test getting artifact by ID."""
        registry = ArtifactRegistry()
        artifact = registry.create(content={"data": "test"}, name="Test")

        found = registry.get(artifact.id)
        assert found == artifact

        not_found = registry.get("nonexistent")
        assert not_found is None

    def test_get_by_definition(self) -> None:
        """Test getting artifacts by definition."""
        definitions = {
            "type_a": ArtifactDefinition(id="type_a"),
            "type_b": ArtifactDefinition(id="type_b"),
        }
        registry = ArtifactRegistry(definitions=definitions)

        registry.create(content={"a": 1}, definition_id="type_a")
        registry.create(content={"a": 2}, definition_id="type_a")
        registry.create(content={"b": 1}, definition_id="type_b")

        type_a_artifacts = registry.get_by_definition("type_a")
        assert len(type_a_artifacts) == 2

        type_b_artifacts = registry.get_by_definition("type_b")
        assert len(type_b_artifacts) == 1

    def test_get_by_status(self) -> None:
        """Test getting artifacts by status."""
        registry = ArtifactRegistry()
        artifact1 = registry.create(content={"v": 1}, name="Draft")
        artifact2 = registry.create(content={"v": 2}, name="Also Draft")
        registry.set_status(artifact2.id, "approved")

        drafts = registry.get_by_status("draft")
        assert len(drafts) == 1
        assert drafts[0].id == artifact1.id

        approved = registry.get_by_status("approved")
        assert len(approved) == 1
        assert approved[0].id == artifact2.id

    def test_get_by_stage(self) -> None:
        """Test getting artifacts by wizard stage."""
        registry = ArtifactRegistry()
        registry.create(content={"v": 1}, stage="stage_a")
        registry.create(content={"v": 2}, stage="stage_a")
        registry.create(content={"v": 3}, stage="stage_b")

        stage_a = registry.get_by_stage("stage_a")
        assert len(stage_a) == 2

        stage_b = registry.get_by_stage("stage_b")
        assert len(stage_b) == 1

    def test_get_pending_review(self) -> None:
        """Test getting artifacts pending review."""
        registry = ArtifactRegistry()
        artifact1 = registry.create(content={"v": 1}, name="Draft")
        artifact2 = registry.create(content={"v": 2}, name="Pending")
        registry.submit_for_review(artifact2.id)

        pending = registry.get_pending_review()
        assert len(pending) == 1
        assert pending[0].id == artifact2.id

    def test_get_all(self) -> None:
        """Test getting all artifacts."""
        registry = ArtifactRegistry()
        registry.create(content={"v": 1}, name="One")
        registry.create(content={"v": 2}, name="Two")

        all_artifacts = registry.get_all()
        assert len(all_artifacts) == 2


class TestArtifactRegistryVersioning:
    """Tests for version navigation."""

    def test_get_latest_version(self) -> None:
        """Test getting latest version."""
        registry = ArtifactRegistry()
        v1 = registry.create(content={"v": 1}, name="Test")
        v2 = registry.update(v1.id, content={"v": 2})
        v3 = registry.update(v2.id, content={"v": 3})

        # Starting from any version should get v3
        assert registry.get_latest_version(v1.id) == v3
        assert registry.get_latest_version(v2.id) == v3
        assert registry.get_latest_version(v3.id) == v3

    def test_get_version_history(self) -> None:
        """Test getting version history."""
        registry = ArtifactRegistry()
        v1 = registry.create(content={"v": 1}, name="Test")
        v2 = registry.update(v1.id, content={"v": 2})
        v3 = registry.update(v2.id, content={"v": 3})

        # Starting from any version should get full history
        history = registry.get_version_history(v1.id)
        assert len(history) == 3
        assert history[0] == v1
        assert history[1] == v2
        assert history[2] == v3

        history = registry.get_version_history(v3.id)
        assert len(history) == 3


class TestArtifactRegistryHooks:
    """Tests for lifecycle hooks."""

    def test_on_create_hook(self) -> None:
        """Test on_create hook is called."""
        created_artifacts: list[Artifact] = []

        def on_create(artifact: Artifact) -> None:
            created_artifacts.append(artifact)

        registry = ArtifactRegistry()
        registry.on("on_create", on_create)

        artifact = registry.create(content={"data": "test"}, name="Test")

        assert len(created_artifacts) == 1
        assert created_artifacts[0] == artifact

    def test_on_update_hook(self) -> None:
        """Test on_update hook is called."""
        update_events: list[tuple[Artifact, Artifact]] = []

        def on_update(new: Artifact, original: Artifact) -> None:
            update_events.append((new, original))

        registry = ArtifactRegistry()
        registry.on("on_update", on_update)

        original = registry.create(content={"v": 1}, name="Test")
        updated = registry.update(original.id, content={"v": 2})

        assert len(update_events) == 1
        assert update_events[0] == (updated, original)

    def test_on_review_hook(self) -> None:
        """Test on_review hook is called."""
        review_events: list[tuple[Artifact, ArtifactReview]] = []

        def on_review(artifact: Artifact, review: ArtifactReview) -> None:
            review_events.append((artifact, review))

        registry = ArtifactRegistry()
        registry.on("on_review", on_review)

        artifact = registry.create(content={"data": "test"}, name="Test")
        review = ArtifactReview(reviewer="test", passed=True)
        registry.add_review(artifact.id, review)

        assert len(review_events) == 1
        assert review_events[0] == (artifact, review)

    def test_invalid_hook_raises(self) -> None:
        """Test that registering invalid hook raises."""
        registry = ArtifactRegistry()
        with pytest.raises(ValueError):
            registry.on("invalid_event", lambda: None)


class TestArtifactRegistrySerialization:
    """Tests for registry serialization."""

    def test_to_dict(self) -> None:
        """Test serializing registry to dict."""
        registry = ArtifactRegistry()
        artifact = registry.create(content={"data": "test"}, name="Test")

        data = registry.to_dict()
        assert "artifacts" in data
        assert artifact.id in data["artifacts"]

    def test_from_dict(self) -> None:
        """Test restoring registry from dict."""
        original_registry = ArtifactRegistry()
        artifact = original_registry.create(content={"data": "test"}, name="Test")
        data = original_registry.to_dict()

        restored_registry = ArtifactRegistry.from_dict(data)
        restored_artifact = restored_registry.get(artifact.id)

        assert restored_artifact is not None
        assert restored_artifact.id == artifact.id
        assert restored_artifact.content == artifact.content
        assert restored_artifact.name == artifact.name

    def test_clear(self) -> None:
        """Test clearing registry."""
        registry = ArtifactRegistry()
        registry.create(content={"data": "test"}, name="Test")
        assert len(registry.get_all()) == 1

        registry.clear()
        assert len(registry.get_all()) == 0
