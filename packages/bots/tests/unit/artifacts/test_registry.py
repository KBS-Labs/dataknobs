"""Tests for artifact registry."""

from __future__ import annotations

import asyncio

import pytest
from dataknobs_common.transitions import InvalidTransitionError
from dataknobs_data import SortOrder, SortSpec
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
    async def test_revise_concurrent_serialized_in_process(
        self, registry: ArtifactRegistry
    ) -> None:
        """Concurrent revise() calls on the same artifact must serialize.

        Before the per-id lock, two concurrent ``revise`` callers could
        each read ``v1.0.0`` and both compute ``v1.0.1`` — last-write
        wins on both the pointer and the snapshot, silently dropping
        one revision.  With the lock, the second caller observes the
        first caller's bumped version and computes ``v1.0.2``.
        """
        original = await registry.create(
            artifact_type="content",
            name="Concurrent",
            content={"v": 1},
        )

        results = await asyncio.gather(
            registry.revise(
                artifact_id=original.id,
                new_content={"v": 2},
                reason="A",
                triggered_by="A",
            ),
            registry.revise(
                artifact_id=original.id,
                new_content={"v": 3},
                reason="B",
                triggered_by="B",
            ),
        )
        versions = sorted(r.version for r in results)
        # Without serialization both calls produce v1.0.1; with
        # serialization the second observes the first and produces
        # v1.0.2.
        assert versions == ["1.0.1", "1.0.2"]
        # Both snapshots persisted distinctly.
        v_old = await registry.get_version(original.id, "1.0.0")
        v_one = await registry.get_version(original.id, "1.0.1")
        v_two = await registry.get_version(original.id, "1.0.2")
        assert v_old is not None
        assert v_one is not None
        assert v_two is not None
        # The pointer reflects whichever revise won the second slot.
        latest = await registry.get(original.id)
        assert latest is not None
        assert latest.version == "1.0.2"

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


class TestArtifactRegistryMetadata:
    """Tests for the metadata channel routed through AsyncKeyedRecordStore.

    These pin the metadata-preservation contract that the structural
    Record.data/metadata split enforces.  Together with the existing
    test_models.test_metadata_round_trip, they verify that metadata
    flows from caller → serializer → backend → deserializer → caller
    without leakage into the data channel.
    """

    @pytest.mark.asyncio
    async def test_create_with_metadata_preserved(
        self, registry: ArtifactRegistry
    ) -> None:
        """Metadata passed to create() round-trips through get()."""
        artifact = await registry.create(
            artifact_type="content",
            name="Metadata Test",
            content={"v": 1},
            metadata={"tenant_id": "acme", "correlation_id": "req-1"},
        )
        assert artifact.metadata == {"tenant_id": "acme", "correlation_id": "req-1"}

        retrieved = await registry.get(artifact.id)
        assert retrieved is not None
        assert retrieved.metadata == {"tenant_id": "acme", "correlation_id": "req-1"}

    @pytest.mark.asyncio
    async def test_create_without_metadata_defaults_empty(
        self, registry: ArtifactRegistry
    ) -> None:
        """Omitting metadata stores an empty dict, not None — matches model default."""
        artifact = await registry.create(
            artifact_type="content",
            name="No Metadata",
            content={"v": 1},
        )
        retrieved = await registry.get(artifact.id)
        assert retrieved is not None
        assert retrieved.metadata == {}

    @pytest.mark.asyncio
    async def test_get_version_returns_metadata(
        self, registry: ArtifactRegistry
    ) -> None:
        """Versioned snapshots preserve metadata too (both writes go through the store)."""
        artifact = await registry.create(
            artifact_type="content",
            name="Versioned",
            content={"v": 1},
            metadata={"tenant_id": "acme"},
        )
        snapshot = await registry.get_version(artifact.id, artifact.version)
        assert snapshot is not None
        assert snapshot.metadata == {"tenant_id": "acme"}

    @pytest.mark.asyncio
    async def test_query_filter_metadata_matches(
        self, registry: ArtifactRegistry
    ) -> None:
        """``filter_metadata`` routes through the metadata column and matches AND-style."""
        await registry.create(
            artifact_type="content",
            name="Acme One",
            content={"v": 1},
            metadata={"tenant_id": "acme"},
        )
        await registry.create(
            artifact_type="content",
            name="Globex One",
            content={"v": 1},
            metadata={"tenant_id": "globex"},
        )

        acme = await registry.query(filter_metadata={"tenant_id": "acme"})
        names = sorted(a.name for a in acme)
        assert names == ["Acme One"]

    @pytest.mark.asyncio
    async def test_query_empty_filter_metadata_is_no_filter(
        self, registry: ArtifactRegistry
    ) -> None:
        """``filter_metadata={}`` is equivalent to ``None`` — no filter applied."""
        await registry.create(
            artifact_type="content",
            name="A",
            content={"v": 1},
            metadata={"tenant_id": "acme"},
        )
        await registry.create(
            artifact_type="content",
            name="B",
            content={"v": 1},
            metadata={"tenant_id": "globex"},
        )

        all_with_empty_filter = await registry.query(filter_metadata={})
        all_unfiltered = await registry.query()
        assert len(all_with_empty_filter) == len(all_unfiltered) == 2

    @pytest.mark.asyncio
    async def test_query_filter_metadata_combined_with_type(
        self, registry: ArtifactRegistry
    ) -> None:
        """``filter_metadata`` AND-combines with structural filters like artifact_type."""
        await registry.create(
            artifact_type="content",
            name="Acme Content",
            content={"v": 1},
            metadata={"tenant_id": "acme"},
        )
        await registry.create(
            artifact_type="config",
            name="Acme Config",
            content={"v": 1},
            metadata={"tenant_id": "acme"},
        )
        await registry.create(
            artifact_type="content",
            name="Globex Content",
            content={"v": 1},
            metadata={"tenant_id": "globex"},
        )

        results = await registry.query(
            artifact_type="content",
            filter_metadata={"tenant_id": "acme"},
        )
        assert [a.name for a in results] == ["Acme Content"]

    @pytest.mark.asyncio
    async def test_revise_inherits_metadata_by_default(
        self, registry: ArtifactRegistry
    ) -> None:
        """``revise()`` without explicit metadata carries the current version's metadata."""
        original = await registry.create(
            artifact_type="content",
            name="Inherited",
            content={"v": 1},
            metadata={"tenant_id": "acme"},
        )

        revised = await registry.revise(
            artifact_id=original.id,
            new_content={"v": 2},
            reason="Update",
            triggered_by="test",
        )
        assert revised.metadata == {"tenant_id": "acme"}

    @pytest.mark.asyncio
    async def test_revise_with_explicit_metadata_overrides(
        self, registry: ArtifactRegistry
    ) -> None:
        """``revise(metadata=...)`` overrides the inherited metadata for the new version."""
        original = await registry.create(
            artifact_type="content",
            name="Override",
            content={"v": 1},
            metadata={"tenant_id": "acme"},
        )

        revised = await registry.revise(
            artifact_id=original.id,
            new_content={"v": 2},
            reason="Tenant migration",
            triggered_by="test",
            metadata={"tenant_id": "acme", "audit": {"migrated_from": "acme"}},
        )
        assert revised.metadata == {
            "tenant_id": "acme",
            "audit": {"migrated_from": "acme"},
        }
        # And the old version's metadata is unchanged in its snapshot.
        v1 = await registry.get_version(original.id, "1.0.0")
        assert v1 is not None
        assert v1.metadata == {"tenant_id": "acme"}


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


class TestArtifactRegistryQueryPagination:
    """Tests for ``query()`` sort/limit/offset and ``count()``.

    Pins the post-dedup pagination contract: ``sort`` pushes to the
    database but ``limit``/``offset`` apply to the deduplicated
    artifact list — never to the pre-dedup row stream — because the
    dual-write storage shape (latest pointer + versioned snapshots)
    means N rows in the database can collapse to anywhere between
    ``ceil(N/2)`` and ``1`` artifact.
    """

    @pytest.mark.asyncio
    async def test_query_sort_by_name_asc(
        self, registry: ArtifactRegistry
    ) -> None:
        """Sort by name ascending returns artifacts in name order."""
        await registry.create(
            artifact_type="content", name="Charlie", content={"v": 1}
        )
        await registry.create(
            artifact_type="content", name="Alice", content={"v": 1}
        )
        await registry.create(
            artifact_type="content", name="Bob", content={"v": 1}
        )

        results = await registry.query(
            sort=[SortSpec(field="name", order=SortOrder.ASC)],
        )
        assert [a.name for a in results] == ["Alice", "Bob", "Charlie"]

    @pytest.mark.asyncio
    async def test_query_sort_by_name_desc(
        self, registry: ArtifactRegistry
    ) -> None:
        """Sort by name descending returns artifacts in reverse name order."""
        await registry.create(
            artifact_type="content", name="Charlie", content={"v": 1}
        )
        await registry.create(
            artifact_type="content", name="Alice", content={"v": 1}
        )
        await registry.create(
            artifact_type="content", name="Bob", content={"v": 1}
        )

        results = await registry.query(
            sort=[SortSpec(field="name", order=SortOrder.DESC)],
        )
        assert [a.name for a in results] == ["Charlie", "Bob", "Alice"]

    @pytest.mark.asyncio
    async def test_query_limit_applies_after_dedup(
        self, registry: ArtifactRegistry
    ) -> None:
        """``limit`` returns N deduplicated artifacts, not N database rows.

        Each create writes both a latest pointer and a versioned
        snapshot, so 3 artifacts produce 6 records.  ``limit=2`` must
        return 2 artifacts (not, for example, 2 records that may all
        be snapshots).
        """
        await registry.create(
            artifact_type="content", name="A", content={"v": 1}
        )
        await registry.create(
            artifact_type="content", name="B", content={"v": 1}
        )
        await registry.create(
            artifact_type="content", name="C", content={"v": 1}
        )

        results = await registry.query(
            sort=[SortSpec(field="name", order=SortOrder.ASC)],
            limit=2,
        )
        assert len(results) == 2
        assert [a.name for a in results] == ["A", "B"]

    @pytest.mark.asyncio
    async def test_query_offset_applies_after_dedup(
        self, registry: ArtifactRegistry
    ) -> None:
        """``offset`` skips deduplicated artifacts, not raw rows."""
        await registry.create(
            artifact_type="content", name="A", content={"v": 1}
        )
        await registry.create(
            artifact_type="content", name="B", content={"v": 1}
        )
        await registry.create(
            artifact_type="content", name="C", content={"v": 1}
        )

        results = await registry.query(
            sort=[SortSpec(field="name", order=SortOrder.ASC)],
            offset=1,
        )
        assert [a.name for a in results] == ["B", "C"]

    @pytest.mark.asyncio
    async def test_query_limit_and_offset_combine(
        self, registry: ArtifactRegistry
    ) -> None:
        """``offset`` first, then ``limit`` — standard pagination semantics."""
        for name in ["A", "B", "C", "D", "E"]:
            await registry.create(
                artifact_type="content", name=name, content={"v": 1}
            )

        page = await registry.query(
            sort=[SortSpec(field="name", order=SortOrder.ASC)],
            offset=1,
            limit=2,
        )
        assert [a.name for a in page] == ["B", "C"]

    @pytest.mark.asyncio
    async def test_query_limit_with_versioned_artifacts(
        self, registry: ArtifactRegistry
    ) -> None:
        """Pre-existing versions don't leak through ``limit``.

        Revising an artifact creates additional snapshot rows.  After
        2 revisions on artifact A, the database has 4 rows for A
        (pointer + 3 snapshots).  A query with ``limit=2`` must still
        deduplicate first, then take 2 artifacts — not 2 rows.

        DRAFT → SUPERSEDED is a direct allowed transition, so
        ``revise()`` runs from DRAFT without manual state shepherding.
        """
        a = await registry.create(
            artifact_type="content", name="A", content={"v": 1}
        )
        await registry.revise(
            artifact_id=a.id,
            new_content={"v": 2},
            reason="r2",
            triggered_by="t",
        )
        await registry.revise(
            artifact_id=a.id,
            new_content={"v": 3},
            reason="r3",
            triggered_by="t",
        )
        await registry.create(
            artifact_type="content", name="B", content={"v": 1}
        )

        results = await registry.query(
            sort=[SortSpec(field="name", order=SortOrder.ASC)],
            limit=2,
        )
        # Must be 2 distinct artifacts, not 2 raw rows
        assert len(results) == 2
        assert {r.name for r in results} == {"A", "B"}
        # And A should be the latest-pointer version (v=3)
        latest_a = next(r for r in results if r.name == "A")
        assert latest_a.content == {"v": 3}

    @pytest.mark.asyncio
    async def test_query_sort_combines_with_filters(
        self, registry: ArtifactRegistry
    ) -> None:
        """Sort works alongside ``artifact_type`` and ``filter_metadata``."""
        await registry.create(
            artifact_type="content",
            name="Charlie",
            content={"v": 1},
            metadata={"tenant_id": "acme"},
        )
        await registry.create(
            artifact_type="content",
            name="Alice",
            content={"v": 1},
            metadata={"tenant_id": "acme"},
        )
        await registry.create(
            artifact_type="config",
            name="Bob",
            content={"v": 1},
            metadata={"tenant_id": "acme"},
        )
        await registry.create(
            artifact_type="content",
            name="Dan",
            content={"v": 1},
            metadata={"tenant_id": "globex"},
        )

        results = await registry.query(
            artifact_type="content",
            filter_metadata={"tenant_id": "acme"},
            sort=[SortSpec(field="name", order=SortOrder.ASC)],
        )
        assert [a.name for a in results] == ["Alice", "Charlie"]

    @pytest.mark.asyncio
    async def test_query_limit_zero_returns_empty(
        self, registry: ArtifactRegistry
    ) -> None:
        """``limit=0`` returns no artifacts (consistent with Python slice)."""
        await registry.create(
            artifact_type="content", name="A", content={"v": 1}
        )

        results = await registry.query(limit=0)
        assert results == []

    @pytest.mark.asyncio
    async def test_query_offset_beyond_count_returns_empty(
        self, registry: ArtifactRegistry
    ) -> None:
        """``offset`` past the end returns empty list, not error."""
        await registry.create(
            artifact_type="content", name="A", content={"v": 1}
        )

        results = await registry.query(offset=10)
        assert results == []

    @pytest.mark.asyncio
    async def test_count_empty_registry_returns_zero(
        self, registry: ArtifactRegistry
    ) -> None:
        """Count over an empty registry is zero."""
        assert await registry.count() == 0

    @pytest.mark.asyncio
    async def test_count_returns_distinct_artifact_count(
        self, registry: ArtifactRegistry
    ) -> None:
        """Count returns the number of distinct artifacts, not raw rows."""
        await registry.create(
            artifact_type="content", name="A", content={"v": 1}
        )
        await registry.create(
            artifact_type="content", name="B", content={"v": 1}
        )
        await registry.create(
            artifact_type="config", name="C", content={"v": 1}
        )

        assert await registry.count() == 3

    @pytest.mark.asyncio
    async def test_count_with_versioned_artifacts_dedups(
        self, registry: ArtifactRegistry
    ) -> None:
        """Versioned snapshots are deduplicated in the count.

        After one revision, artifact A has 3 raw rows (pointer + 2
        snapshots).  ``count()`` must return 1 for the single distinct
        artifact, not 3.
        """
        a = await registry.create(
            artifact_type="content", name="A", content={"v": 1}
        )
        await registry.revise(
            artifact_id=a.id,
            new_content={"v": 2},
            reason="r2",
            triggered_by="t",
        )

        assert await registry.count() == 1

    @pytest.mark.asyncio
    async def test_count_by_type(
        self, registry: ArtifactRegistry
    ) -> None:
        """Count respects ``artifact_type`` filter."""
        await registry.create(
            artifact_type="content", name="A", content={"v": 1}
        )
        await registry.create(
            artifact_type="content", name="B", content={"v": 1}
        )
        await registry.create(
            artifact_type="config", name="C", content={"v": 1}
        )

        assert await registry.count(artifact_type="content") == 2
        assert await registry.count(artifact_type="config") == 1
        assert await registry.count(artifact_type="nonexistent") == 0

    @pytest.mark.asyncio
    async def test_count_by_status(
        self, registry: ArtifactRegistry
    ) -> None:
        """Count respects ``status`` filter."""
        a = await registry.create(
            artifact_type="content", name="A", content={"v": 1}
        )
        await registry.create(
            artifact_type="content", name="B", content={"v": 1}
        )
        await registry.set_status(a.id, ArtifactStatus.PENDING_REVIEW)

        assert await registry.count(status=ArtifactStatus.DRAFT) == 1
        assert await registry.count(status=ArtifactStatus.PENDING_REVIEW) == 1
        assert await registry.count(status=ArtifactStatus.ARCHIVED) == 0

    @pytest.mark.asyncio
    async def test_count_by_tags(
        self, registry: ArtifactRegistry
    ) -> None:
        """Count respects ``tags`` filter (AND-style match)."""
        await registry.create(
            artifact_type="content",
            name="A",
            content={"v": 1},
            tags=["math", "assessment"],
        )
        await registry.create(
            artifact_type="content",
            name="B",
            content={"v": 1},
            tags=["math"],
        )
        await registry.create(
            artifact_type="content",
            name="C",
            content={"v": 1},
            tags=["science"],
        )

        assert await registry.count(tags=["math"]) == 2
        assert await registry.count(tags=["math", "assessment"]) == 1
        assert await registry.count(tags=["science"]) == 1

    @pytest.mark.asyncio
    async def test_count_with_filter_metadata(
        self, registry: ArtifactRegistry
    ) -> None:
        """Count routes through the metadata channel."""
        await registry.create(
            artifact_type="content",
            name="A",
            content={"v": 1},
            metadata={"tenant_id": "acme"},
        )
        await registry.create(
            artifact_type="content",
            name="B",
            content={"v": 1},
            metadata={"tenant_id": "acme"},
        )
        await registry.create(
            artifact_type="content",
            name="C",
            content={"v": 1},
            metadata={"tenant_id": "globex"},
        )

        assert await registry.count(filter_metadata={"tenant_id": "acme"}) == 2
        assert (
            await registry.count(filter_metadata={"tenant_id": "globex"}) == 1
        )
        assert await registry.count(filter_metadata={"tenant_id": "none"}) == 0

    @pytest.mark.asyncio
    async def test_count_combines_filters(
        self, registry: ArtifactRegistry
    ) -> None:
        """Count AND-combines ``artifact_type`` and ``filter_metadata``."""
        await registry.create(
            artifact_type="content",
            name="A",
            content={"v": 1},
            metadata={"tenant_id": "acme"},
        )
        await registry.create(
            artifact_type="config",
            name="B",
            content={"v": 1},
            metadata={"tenant_id": "acme"},
        )
        await registry.create(
            artifact_type="content",
            name="C",
            content={"v": 1},
            metadata={"tenant_id": "globex"},
        )

        assert (
            await registry.count(
                artifact_type="content",
                filter_metadata={"tenant_id": "acme"},
            )
            == 1
        )

    @pytest.mark.asyncio
    async def test_count_matches_query_length(
        self, registry: ArtifactRegistry
    ) -> None:
        """count(...) and len(query(...)) agree for the same filter shape."""
        for name in ["A", "B", "C", "D"]:
            await registry.create(
                artifact_type="content",
                name=name,
                content={"v": 1},
                metadata={"tenant_id": "acme"},
            )
        await registry.create(
            artifact_type="content",
            name="E",
            content={"v": 1},
            metadata={"tenant_id": "globex"},
        )

        kwargs: dict[str, object] = {
            "artifact_type": "content",
            "filter_metadata": {"tenant_id": "acme"},
        }
        results = await registry.query(**kwargs)  # type: ignore[arg-type]
        count = await registry.count(**kwargs)  # type: ignore[arg-type]
        assert count == len(results) == 4
