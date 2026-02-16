"""Tests for artifact registry."""

from __future__ import annotations

import pytest
from dataknobs_common.transitions import InvalidTransitionError
from dataknobs_data.backends.memory import AsyncMemoryDatabase

from dataknobs_bots.artifacts.models import (
    Artifact,
    ArtifactStatus,
    ArtifactTypeDefinition,
)
from dataknobs_bots.artifacts.provenance import create_provenance
from dataknobs_bots.artifacts.registry import ArtifactRegistry


@pytest.fixture
async def db() -> AsyncMemoryDatabase:
    """Create an in-memory database for testing."""
    return AsyncMemoryDatabase()


@pytest.fixture
async def registry(db: AsyncMemoryDatabase) -> ArtifactRegistry:
    """Create a registry for testing."""
    return ArtifactRegistry(db=db)


@pytest.fixture
def type_defs() -> dict[str, ArtifactTypeDefinition]:
    """Create type definitions for testing."""
    return {
        "questions": ArtifactTypeDefinition(
            id="questions",
            description="Assessment questions",
            rubrics=["quality_check"],
            tags=["assessment"],
        ),
        "config": ArtifactTypeDefinition(
            id="config",
            description="Configuration artifact",
            rubrics=[],
            tags=["config"],
        ),
    }


@pytest.fixture
async def registry_with_types(
    db: AsyncMemoryDatabase, type_defs: dict[str, ArtifactTypeDefinition]
) -> ArtifactRegistry:
    """Create a registry with type definitions."""
    return ArtifactRegistry(db=db, type_definitions=type_defs)


class TestArtifactRegistryCRUD:
    """Tests for basic CRUD operations."""

    @pytest.mark.asyncio
    async def test_create_simple(self, registry: ArtifactRegistry) -> None:
        """Test simple artifact creation."""
        artifact = await registry.create(
            artifact_type="content",
            name="Test Artifact",
            content={"key": "value"},
        )

        assert artifact.id.startswith("art_")
        assert artifact.name == "Test Artifact"
        assert artifact.content == {"key": "value"}
        assert artifact.type == "content"
        assert artifact.status == ArtifactStatus.DRAFT
        assert artifact.version == "1.0.0"

    @pytest.mark.asyncio
    async def test_create_with_provenance(
        self, registry: ArtifactRegistry
    ) -> None:
        """Test artifact creation with explicit provenance."""
        provenance = create_provenance(
            created_by="bot:test",
            creation_method="generator",
        )
        artifact = await registry.create(
            artifact_type="content",
            name="Provenance Test",
            content={"data": "value"},
            provenance=provenance,
        )

        assert artifact.provenance.created_by == "bot:test"
        assert artifact.provenance.creation_method == "generator"

    @pytest.mark.asyncio
    async def test_create_with_tags(self, registry: ArtifactRegistry) -> None:
        """Test artifact creation with custom tags."""
        artifact = await registry.create(
            artifact_type="content",
            name="Tagged",
            content={"data": "test"},
            tags=["tag1", "tag2"],
        )

        assert "tag1" in artifact.tags
        assert "tag2" in artifact.tags

    @pytest.mark.asyncio
    async def test_create_with_type_definition(
        self, registry_with_types: ArtifactRegistry
    ) -> None:
        """Test creation inherits type definition defaults."""
        artifact = await registry_with_types.create(
            artifact_type="questions",
            name="Math Questions",
            content={"questions": []},
        )

        assert "assessment" in artifact.tags
        assert "quality_check" in artifact.rubric_ids

    @pytest.mark.asyncio
    async def test_create_merges_tags_with_type_defaults(
        self, registry_with_types: ArtifactRegistry
    ) -> None:
        """Test custom tags are merged with type definition defaults."""
        artifact = await registry_with_types.create(
            artifact_type="questions",
            name="Tagged Questions",
            content={"questions": []},
            tags=["custom_tag"],
        )

        assert "assessment" in artifact.tags
        assert "custom_tag" in artifact.tags

    @pytest.mark.asyncio
    async def test_get(self, registry: ArtifactRegistry) -> None:
        """Test retrieving an artifact by ID."""
        created = await registry.create(
            artifact_type="content",
            name="Test",
            content={"data": "value"},
        )

        retrieved = await registry.get(created.id)

        assert retrieved is not None
        assert retrieved.id == created.id
        assert retrieved.name == "Test"
        assert retrieved.content == {"data": "value"}

    @pytest.mark.asyncio
    async def test_get_not_found(self, registry: ArtifactRegistry) -> None:
        """Test get returns None for missing artifact."""
        result = await registry.get("nonexistent")
        assert result is None


class TestArtifactRegistryVersioning:
    """Tests for versioning via revise()."""

    @pytest.mark.asyncio
    async def test_revise_creates_new_version(
        self, registry: ArtifactRegistry
    ) -> None:
        """Test that revise bumps version and updates content."""
        original = await registry.create(
            artifact_type="content",
            name="Test",
            content={"v": 1},
        )
        assert original.version == "1.0.0"

        revised = await registry.revise(
            artifact_id=original.id,
            new_content={"v": 2},
            reason="Updated content",
            triggered_by="test",
        )

        assert revised.version == "1.0.1"
        assert revised.content == {"v": 2}
        assert revised.id == original.id

    @pytest.mark.asyncio
    async def test_revise_marks_old_superseded(
        self, registry: ArtifactRegistry
    ) -> None:
        """Test that revise marks old version as superseded."""
        original = await registry.create(
            artifact_type="content",
            name="Test",
            content={"v": 1},
        )
        original_version = original.version

        await registry.revise(
            artifact_id=original.id,
            new_content={"v": 2},
            reason="Updated",
            triggered_by="test",
        )

        old = await registry.get_version(original.id, original_version)
        assert old is not None
        assert old.status == ArtifactStatus.SUPERSEDED

    @pytest.mark.asyncio
    async def test_revise_adds_revision_record(
        self, registry: ArtifactRegistry
    ) -> None:
        """Test that revise adds a revision to provenance."""
        original = await registry.create(
            artifact_type="content",
            name="Test",
            content={"v": 1},
        )

        revised = await registry.revise(
            artifact_id=original.id,
            new_content={"v": 2},
            reason="Fixed typo",
            triggered_by="user",
        )

        assert len(revised.provenance.revision_history) == 1
        revision = revised.provenance.revision_history[0]
        assert revision.reason == "Fixed typo"
        assert revision.triggered_by == "user"
        assert revision.previous_version == "1.0.0"

    @pytest.mark.asyncio
    async def test_revise_not_found_raises(
        self, registry: ArtifactRegistry
    ) -> None:
        """Test that revising non-existent artifact raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            await registry.revise(
                artifact_id="nonexistent",
                new_content={"v": 2},
                reason="Update",
                triggered_by="test",
            )

    @pytest.mark.asyncio
    async def test_get_version(self, registry: ArtifactRegistry) -> None:
        """Test retrieving a specific version."""
        original = await registry.create(
            artifact_type="content",
            name="Test",
            content={"v": 1},
        )

        await registry.revise(
            artifact_id=original.id,
            new_content={"v": 2},
            reason="Update",
            triggered_by="test",
        )

        v1 = await registry.get_version(original.id, "1.0.0")
        assert v1 is not None
        assert v1.content == {"v": 1}

        latest = await registry.get(original.id)
        assert latest is not None
        assert latest.content == {"v": 2}
        assert latest.version == "1.0.1"


class TestArtifactRegistryStatusTransitions:
    """Tests for status management."""

    @pytest.mark.asyncio
    async def test_set_status_valid(
        self, registry: ArtifactRegistry
    ) -> None:
        """Test valid status transition."""
        artifact = await registry.create(
            artifact_type="content",
            name="Test",
            content={"data": "test"},
        )
        assert artifact.status == ArtifactStatus.DRAFT

        await registry.set_status(artifact.id, ArtifactStatus.PENDING_REVIEW)

        updated = await registry.get(artifact.id)
        assert updated is not None
        assert updated.status == ArtifactStatus.PENDING_REVIEW

    @pytest.mark.asyncio
    async def test_set_status_invalid_raises(
        self, registry: ArtifactRegistry
    ) -> None:
        """Test invalid status transition raises."""
        artifact = await registry.create(
            artifact_type="content",
            name="Test",
            content={"data": "test"},
        )

        with pytest.raises(InvalidTransitionError):
            await registry.set_status(artifact.id, ArtifactStatus.APPROVED)

    @pytest.mark.asyncio
    async def test_set_status_not_found_raises(
        self, registry: ArtifactRegistry
    ) -> None:
        """Test set_status for missing artifact raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            await registry.set_status("nonexistent", ArtifactStatus.ARCHIVED)


class TestArtifactRegistryQuery:
    """Tests for querying artifacts."""

    @pytest.mark.asyncio
    async def test_query_by_type(self, registry: ArtifactRegistry) -> None:
        """Test querying by artifact type."""
        await registry.create(
            artifact_type="content", name="A1", content={"a": 1}
        )
        await registry.create(
            artifact_type="content", name="A2", content={"a": 2}
        )
        await registry.create(
            artifact_type="config", name="B1", content={"b": 1}
        )

        content_artifacts = await registry.query(artifact_type="content")
        assert len(content_artifacts) == 2

        config_artifacts = await registry.query(artifact_type="config")
        assert len(config_artifacts) == 1

    @pytest.mark.asyncio
    async def test_query_by_status(
        self, registry: ArtifactRegistry
    ) -> None:
        """Test querying by status."""
        a1 = await registry.create(
            artifact_type="content", name="Draft", content={"v": 1}
        )
        a2 = await registry.create(
            artifact_type="content", name="Reviewed", content={"v": 2}
        )
        await registry.set_status(a2.id, ArtifactStatus.PENDING_REVIEW)

        drafts = await registry.query(status=ArtifactStatus.DRAFT)
        assert len(drafts) == 1
        assert drafts[0].id == a1.id

        pending = await registry.query(status=ArtifactStatus.PENDING_REVIEW)
        assert len(pending) == 1
        assert pending[0].id == a2.id

    @pytest.mark.asyncio
    async def test_query_by_tags(self, registry: ArtifactRegistry) -> None:
        """Test querying by tags."""
        await registry.create(
            artifact_type="content",
            name="Tagged",
            content={"v": 1},
            tags=["math", "assessment"],
        )
        await registry.create(
            artifact_type="content",
            name="Untagged",
            content={"v": 2},
        )

        results = await registry.query(tags=["math"])
        assert len(results) == 1
        assert results[0].name == "Tagged"

    @pytest.mark.asyncio
    async def test_query_no_results(
        self, registry: ArtifactRegistry
    ) -> None:
        """Test querying with no matches."""
        results = await registry.query(artifact_type="nonexistent")
        assert len(results) == 0


class TestArtifactRegistryReviewIntegration:
    """Tests for rubric-based review integration."""

    @pytest.mark.asyncio
    async def test_submit_for_review_no_executor(
        self, registry: ArtifactRegistry
    ) -> None:
        """Test submit_for_review without rubric components."""
        artifact = await registry.create(
            artifact_type="content",
            name="Test",
            content={"data": "value"},
        )

        evaluations = await registry.submit_for_review(artifact.id)
        assert evaluations == []

        updated = await registry.get(artifact.id)
        assert updated is not None
        assert updated.status == ArtifactStatus.IN_REVIEW

    @pytest.mark.asyncio
    async def test_submit_for_review_not_found(
        self, registry: ArtifactRegistry
    ) -> None:
        """Test submit_for_review with missing artifact."""
        with pytest.raises(ValueError, match="not found"):
            await registry.submit_for_review("nonexistent")


class TestArtifactRegistryHooks:
    """Tests for lifecycle hooks."""

    @pytest.mark.asyncio
    async def test_on_create_hook(
        self, registry: ArtifactRegistry
    ) -> None:
        """Test on_create hook is called."""
        created_artifacts: list[Artifact] = []

        async def on_create(artifact: Artifact) -> None:
            created_artifacts.append(artifact)

        registry.on_create(on_create)

        artifact = await registry.create(
            artifact_type="content",
            name="Hooked",
            content={"data": "test"},
        )

        assert len(created_artifacts) == 1
        assert created_artifacts[0].id == artifact.id

    @pytest.mark.asyncio
    async def test_on_status_change_hook(
        self, registry: ArtifactRegistry
    ) -> None:
        """Test on_status_change hook is called."""
        status_changes: list[Artifact] = []

        async def on_status(artifact: Artifact) -> None:
            status_changes.append(artifact)

        registry.on_status_change(on_status)

        artifact = await registry.create(
            artifact_type="content",
            name="Status Test",
            content={"data": "test"},
        )
        await registry.set_status(artifact.id, ArtifactStatus.PENDING_REVIEW)

        assert len(status_changes) == 1
        assert status_changes[0].status == ArtifactStatus.PENDING_REVIEW


class TestArtifactRegistryConfig:
    """Tests for configuration-driven creation."""

    @pytest.mark.asyncio
    async def test_from_config(self) -> None:
        """Test creating registry from configuration."""
        config = {
            "artifact_types": {
                "questions": {
                    "description": "Assessment questions",
                    "rubrics": ["quality"],
                    "tags": ["assessment"],
                },
                "config": {
                    "description": "Config artifact",
                    "tags": ["config"],
                },
            }
        }
        db = AsyncMemoryDatabase()
        registry = await ArtifactRegistry.from_config(config, db=db)

        artifact = await registry.create(
            artifact_type="questions",
            name="Math Questions",
            content={"questions": []},
        )

        assert "assessment" in artifact.tags
        assert "quality" in artifact.rubric_ids
