"""Tests for prompt versioning functionality."""

import pytest
from datetime import datetime

from dataknobs_llm.prompts.versioning import (
    VersionManager,
    PromptVersion,
    VersionStatus,
    VersioningError,
)


class TestVersionManager:
    """Test VersionManager functionality."""

    @pytest.fixture
    def manager(self):
        """Create a VersionManager instance for testing."""
        return VersionManager()

    @pytest.mark.asyncio
    async def test_create_version(self, manager):
        """Test creating a new prompt version."""
        version = await manager.create_version(
            name="greeting",
            prompt_type="system",
            template="Hello {{name}}!",
            version="1.0.0",
            defaults={"name": "World"},
            metadata={"author": "alice"},
        )

        assert version.name == "greeting"
        assert version.prompt_type == "system"
        assert version.version == "1.0.0"
        assert version.template == "Hello {{name}}!"
        assert version.defaults == {"name": "World"}
        assert version.metadata["author"] == "alice"
        assert version.status == VersionStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_auto_increment_version(self, manager):
        """Test automatic version incrementing."""
        # Create first version
        v1 = await manager.create_version(
            name="test",
            prompt_type="user",
            template="Version 1",
            version="1.0.0"
        )

        # Create second version without specifying version
        v2 = await manager.create_version(
            name="test",
            prompt_type="user",
            template="Version 2"
        )

        assert v1.version == "1.0.0"
        assert v2.version == "1.0.1"
        assert v2.parent_version == v1.version_id

    @pytest.mark.asyncio
    async def test_auto_increment_from_no_versions(self, manager):
        """Test auto-increment creates 1.0.0 for first version."""
        version = await manager.create_version(
            name="new_prompt",
            prompt_type="system",
            template="First version"
        )

        assert version.version == "1.0.0"
        assert version.parent_version is None

    @pytest.mark.asyncio
    async def test_invalid_version_format(self, manager):
        """Test that invalid version format raises error."""
        with pytest.raises(VersioningError, match="Invalid version format"):
            await manager.create_version(
                name="test",
                prompt_type="system",
                template="Test",
                version="1.0"  # Invalid: missing patch version
            )

    @pytest.mark.asyncio
    async def test_duplicate_version(self, manager):
        """Test that creating duplicate version raises error."""
        await manager.create_version(
            name="test",
            prompt_type="system",
            template="V1",
            version="1.0.0"
        )

        with pytest.raises(VersioningError, match="already exists"):
            await manager.create_version(
                name="test",
                prompt_type="system",
                template="V1 again",
                version="1.0.0"
            )

    @pytest.mark.asyncio
    async def test_get_version_by_number(self, manager):
        """Test retrieving specific version."""
        created = await manager.create_version(
            name="test",
            prompt_type="system",
            template="Test",
            version="1.0.0"
        )

        retrieved = await manager.get_version(
            name="test",
            prompt_type="system",
            version="1.0.0"
        )

        assert retrieved is not None
        assert retrieved.version_id == created.version_id
        assert retrieved.version == "1.0.0"

    @pytest.mark.asyncio
    async def test_get_latest_version(self, manager):
        """Test getting latest version."""
        await manager.create_version(
            name="test",
            prompt_type="system",
            template="V1",
            version="1.0.0"
        )

        v2 = await manager.create_version(
            name="test",
            prompt_type="system",
            template="V2",
            version="1.1.0"
        )

        await manager.create_version(
            name="test",
            prompt_type="system",
            template="V3",
            version="1.0.1"
        )

        latest = await manager.get_version(
            name="test",
            prompt_type="system",
            version="latest"
        )

        assert latest is not None
        assert latest.version == "1.1.0"  # Highest version
        assert latest.version_id == v2.version_id

    @pytest.mark.asyncio
    async def test_get_nonexistent_version(self, manager):
        """Test getting nonexistent version returns None."""
        result = await manager.get_version(
            name="nonexistent",
            prompt_type="system",
            version="1.0.0"
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_list_versions(self, manager):
        """Test listing all versions of a prompt."""
        await manager.create_version(
            name="test",
            prompt_type="system",
            template="V1",
            version="1.0.0"
        )

        await manager.create_version(
            name="test",
            prompt_type="system",
            template="V2",
            version="1.1.0"
        )

        await manager.create_version(
            name="test",
            prompt_type="system",
            template="V3",
            version="2.0.0"
        )

        versions = await manager.list_versions("test", "system")

        assert len(versions) == 3
        # Should be sorted newest first
        assert versions[0].version == "2.0.0"
        assert versions[1].version == "1.1.0"
        assert versions[2].version == "1.0.0"

    @pytest.mark.asyncio
    async def test_list_versions_with_tag_filter(self, manager):
        """Test filtering versions by tags."""
        v1 = await manager.create_version(
            name="test",
            prompt_type="system",
            template="V1",
            version="1.0.0",
            tags=["production"]
        )

        await manager.create_version(
            name="test",
            prompt_type="system",
            template="V2",
            version="1.1.0",
            tags=["experimental"]
        )

        versions = await manager.list_versions(
            name="test",
            prompt_type="system",
            tags=["production"]
        )

        assert len(versions) == 1
        assert versions[0].version_id == v1.version_id

    @pytest.mark.asyncio
    async def test_list_versions_with_status_filter(self, manager):
        """Test filtering versions by status."""
        v1 = await manager.create_version(
            name="test",
            prompt_type="system",
            template="V1",
            version="1.0.0",
            status=VersionStatus.PRODUCTION
        )

        await manager.create_version(
            name="test",
            prompt_type="system",
            template="V2",
            version="1.1.0",
            status=VersionStatus.DRAFT
        )

        versions = await manager.list_versions(
            name="test",
            prompt_type="system",
            status=VersionStatus.PRODUCTION
        )

        assert len(versions) == 1
        assert versions[0].version_id == v1.version_id

    @pytest.mark.asyncio
    async def test_tag_version(self, manager):
        """Test adding tags to a version."""
        version = await manager.create_version(
            name="test",
            prompt_type="system",
            template="Test",
            version="1.0.0"
        )

        assert "production" not in version.tags

        updated = await manager.tag_version(version.version_id, "production")

        assert "production" in updated.tags

    @pytest.mark.asyncio
    async def test_tag_nonexistent_version(self, manager):
        """Test tagging nonexistent version raises error."""
        with pytest.raises(VersioningError, match="not found"):
            await manager.tag_version("nonexistent-id", "production")

    @pytest.mark.asyncio
    async def test_untag_version(self, manager):
        """Test removing tags from a version."""
        version = await manager.create_version(
            name="test",
            prompt_type="system",
            template="Test",
            version="1.0.0",
            tags=["production", "stable"]
        )

        assert "production" in version.tags

        updated = await manager.untag_version(version.version_id, "production")

        assert "production" not in updated.tags
        assert "stable" in updated.tags

    @pytest.mark.asyncio
    async def test_update_status(self, manager):
        """Test updating version status."""
        version = await manager.create_version(
            name="test",
            prompt_type="system",
            template="Test",
            version="1.0.0",
            status=VersionStatus.DRAFT
        )

        assert version.status == VersionStatus.DRAFT

        updated = await manager.update_status(
            version.version_id,
            VersionStatus.PRODUCTION
        )

        assert updated.status == VersionStatus.PRODUCTION

    @pytest.mark.asyncio
    async def test_delete_version(self, manager):
        """Test deleting a version."""
        version = await manager.create_version(
            name="test",
            prompt_type="system",
            template="Test",
            version="1.0.0"
        )

        # Verify it exists
        retrieved = await manager.get_version(
            name="test",
            prompt_type="system",
            version="1.0.0"
        )
        assert retrieved is not None

        # Delete it
        deleted = await manager.delete_version(version.version_id)
        assert deleted is True

        # Verify it's gone
        retrieved = await manager.get_version(
            name="test",
            prompt_type="system",
            version="1.0.0"
        )
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_version(self, manager):
        """Test deleting nonexistent version returns False."""
        deleted = await manager.delete_version("nonexistent-id")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_get_version_history(self, manager):
        """Test getting version history in chronological order."""
        # Create versions with slight time gaps
        v1 = await manager.create_version(
            name="test",
            prompt_type="system",
            template="V1",
            version="1.0.0"
        )

        v2 = await manager.create_version(
            name="test",
            prompt_type="system",
            template="V2",
            version="1.1.0"
        )

        v3 = await manager.create_version(
            name="test",
            prompt_type="system",
            template="V3",
            version="2.0.0"
        )

        history = await manager.get_version_history("test", "system")

        # Should be in chronological order (oldest first)
        assert len(history) == 3
        assert history[0].version_id == v1.version_id
        assert history[1].version_id == v2.version_id
        assert history[2].version_id == v3.version_id

    @pytest.mark.asyncio
    async def test_latest_version_excludes_deprecated(self, manager):
        """Test that latest version excludes deprecated/archived versions."""
        v1 = await manager.create_version(
            name="test",
            prompt_type="system",
            template="V1",
            version="1.0.0",
            status=VersionStatus.PRODUCTION
        )

        # Create higher version but mark as deprecated
        await manager.create_version(
            name="test",
            prompt_type="system",
            template="V2",
            version="1.1.0",
            status=VersionStatus.DEPRECATED
        )

        latest = await manager.get_version(
            name="test",
            prompt_type="system",
            version="latest"
        )

        # Should get v1 (production) not v2 (deprecated)
        assert latest.version_id == v1.version_id

    @pytest.mark.asyncio
    async def test_version_to_dict_and_from_dict(self, manager):
        """Test serialization/deserialization of PromptVersion."""
        original = await manager.create_version(
            name="test",
            prompt_type="system",
            template="Test {{var}}",
            version="1.0.0",
            defaults={"var": "value"},
            metadata={"author": "alice"},
            tags=["production"],
            status=VersionStatus.PRODUCTION
        )

        # Convert to dict
        version_dict = original.to_dict()

        # Convert back
        restored = PromptVersion.from_dict(version_dict)

        assert restored.version_id == original.version_id
        assert restored.name == original.name
        assert restored.version == original.version
        assert restored.template == original.template
        assert restored.defaults == original.defaults
        assert restored.metadata == original.metadata
        assert restored.tags == original.tags
        assert restored.status == original.status
