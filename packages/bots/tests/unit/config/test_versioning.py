"""Tests for configuration versioning."""

import pytest

from dataknobs_bots.config.versioning import (
    ConfigVersion,
    ConfigVersionManager,
    VersionConflictError,
)


class TestConfigVersion:
    """Tests for ConfigVersion."""

    def test_basic_creation(self) -> None:
        """Test basic version creation."""
        version = ConfigVersion(
            version=1,
            config={"name": "MyBot"},
            reason="Initial",
        )

        assert version.version == 1
        assert version.config["name"] == "MyBot"
        assert version.reason == "Initial"
        assert version.previous_version is None

    def test_to_dict_from_dict(self) -> None:
        """Test serialization round-trip."""
        version = ConfigVersion(
            version=2,
            config={"name": "MyBot", "llm": "gpt-4"},
            reason="Added LLM",
            previous_version=1,
            created_by="user1",
            metadata={"env": "production"},
        )

        data = version.to_dict()
        restored = ConfigVersion.from_dict(data)

        assert restored.version == version.version
        assert restored.config == version.config
        assert restored.reason == version.reason
        assert restored.previous_version == version.previous_version
        assert restored.created_by == version.created_by
        assert restored.metadata == version.metadata

    def test_equality(self) -> None:
        """Test version equality is based on version number."""
        v1 = ConfigVersion(version=1, config={"a": 1})
        v1_duplicate = ConfigVersion(version=1, config={"b": 2})
        v2 = ConfigVersion(version=2, config={"a": 1})

        assert v1 == v1_duplicate
        assert v1 != v2

    def test_hash(self) -> None:
        """Test version can be used in sets/dicts."""
        v1 = ConfigVersion(version=1, config={})
        v2 = ConfigVersion(version=2, config={})

        versions = {v1, v2}
        assert len(versions) == 2


class TestConfigVersionManager:
    """Tests for ConfigVersionManager."""

    def test_init(self) -> None:
        """Test manager initialization."""
        manager = ConfigVersionManager()

        assert manager.current_version == 0
        assert manager.current_config is None
        assert len(manager) == 0

    def test_create_initial(self) -> None:
        """Test creating initial version."""
        manager = ConfigVersionManager()

        version = manager.create(
            config={"name": "MyBot"},
            reason="Initial configuration",
        )

        assert version.version == 1
        assert version.config["name"] == "MyBot"
        assert version.previous_version is None
        assert manager.current_version == 1
        assert len(manager) == 1

    def test_create_when_exists(self) -> None:
        """Test create fails when versions exist."""
        manager = ConfigVersionManager()
        manager.create(config={"v": 1})

        with pytest.raises(ValueError, match="already exists"):
            manager.create(config={"v": 2})

    def test_update(self) -> None:
        """Test updating configuration."""
        manager = ConfigVersionManager()
        v1 = manager.create(config={"name": "Bot1"})

        v2 = manager.update(
            config={"name": "Bot2"},
            reason="Renamed bot",
        )

        assert v2.version == 2
        assert v2.config["name"] == "Bot2"
        assert v2.previous_version == 1
        assert manager.current_version == 2
        assert len(manager) == 2

    def test_update_without_initial(self) -> None:
        """Test update fails without initial version."""
        manager = ConfigVersionManager()

        with pytest.raises(ValueError, match="No configuration exists"):
            manager.update(config={"v": 1})

    def test_update_with_expected_version(self) -> None:
        """Test update with expected version check."""
        manager = ConfigVersionManager()
        manager.create(config={"v": 1})

        # Correct expected version
        v2 = manager.update(
            config={"v": 2},
            expected_version=1,
        )
        assert v2.version == 2

    def test_update_version_conflict(self) -> None:
        """Test update fails with wrong expected version."""
        manager = ConfigVersionManager()
        manager.create(config={"v": 1})

        with pytest.raises(VersionConflictError) as exc_info:
            manager.update(
                config={"v": 2},
                expected_version=5,  # Wrong version
            )

        assert exc_info.value.expected_version == 5
        assert exc_info.value.actual_version == 1

    def test_rollback(self) -> None:
        """Test rolling back to previous version."""
        manager = ConfigVersionManager()
        v1 = manager.create(config={"name": "Bot1", "llm": "gpt-3.5"})
        v2 = manager.update(config={"name": "Bot1", "llm": "gpt-4"})
        v3 = manager.update(config={"name": "Bot1", "llm": "gpt-4o"})

        # Rollback to v1
        v4 = manager.rollback(to_version=1, reason="Reverting LLM change")

        assert v4.version == 4
        assert v4.config == v1.config
        assert v4.metadata["rollback_to"] == 1
        assert v4.metadata["rollback_from"] == 3

    def test_rollback_invalid_version(self) -> None:
        """Test rollback to non-existent version."""
        manager = ConfigVersionManager()
        manager.create(config={"v": 1})

        with pytest.raises(ValueError, match="not found"):
            manager.rollback(to_version=99)

    def test_get_version(self) -> None:
        """Test getting specific version."""
        manager = ConfigVersionManager()
        v1 = manager.create(config={"v": 1})
        v2 = manager.update(config={"v": 2})

        found = manager.get_version(1)
        assert found == v1

        found = manager.get_version(2)
        assert found == v2

        not_found = manager.get_version(99)
        assert not_found is None

    def test_get_history(self) -> None:
        """Test getting version history."""
        manager = ConfigVersionManager()
        manager.create(config={"v": 1})
        manager.update(config={"v": 2})
        manager.update(config={"v": 3})

        # Default: newest first
        history = manager.get_history()
        assert len(history) == 3
        assert history[0].version == 3
        assert history[2].version == 1

        # With limit
        history = manager.get_history(limit=2)
        assert len(history) == 2
        assert history[0].version == 3

        # Since version
        history = manager.get_history(since_version=1)
        assert len(history) == 2
        assert all(v.version > 1 for v in history)

    def test_current_config(self) -> None:
        """Test getting current config."""
        manager = ConfigVersionManager()

        assert manager.current_config is None

        manager.create(config={"name": "Bot"})
        assert manager.current_config == {"name": "Bot"}

        manager.update(config={"name": "Bot2"})
        assert manager.current_config == {"name": "Bot2"}

    def test_diff(self) -> None:
        """Test diff between versions."""
        manager = ConfigVersionManager()
        manager.create(config={"name": "Bot", "llm": "gpt-3.5", "temp": 0.7})
        manager.update(config={"name": "Bot", "llm": "gpt-4", "memory": True})

        diff = manager.diff(from_version=1, to_version=2)

        assert diff["from_version"] == 1
        assert diff["to_version"] == 2
        assert diff["added"] == {"memory": True}
        assert diff["removed"] == {"temp": 0.7}
        assert "llm" in diff["changed"]
        assert diff["changed"]["llm"]["from"] == "gpt-3.5"
        assert diff["changed"]["llm"]["to"] == "gpt-4"

    def test_diff_invalid_version(self) -> None:
        """Test diff with invalid version."""
        manager = ConfigVersionManager()
        manager.create(config={"v": 1})

        with pytest.raises(ValueError, match="not found"):
            manager.diff(from_version=1, to_version=99)

    def test_to_dict_from_dict(self) -> None:
        """Test serialization round-trip."""
        manager = ConfigVersionManager()
        manager.create(config={"v": 1}, reason="Initial")
        manager.update(config={"v": 2}, reason="Update")

        data = manager.to_dict()
        restored = ConfigVersionManager.from_dict(data)

        assert restored.current_version == manager.current_version
        assert len(restored) == len(manager)
        assert restored.current_config == manager.current_config

    def test_config_isolation(self) -> None:
        """Test that configs are deeply copied."""
        manager = ConfigVersionManager()

        original = {"name": "Bot", "nested": {"key": "value"}}
        manager.create(config=original)

        # Modify original
        original["name"] = "Modified"
        original["nested"]["key"] = "modified"

        # Stored config should be unchanged
        stored = manager.current_config
        assert stored["name"] == "Bot"
        assert stored["nested"]["key"] == "value"

    def test_created_by_tracking(self) -> None:
        """Test tracking who created versions."""
        manager = ConfigVersionManager()

        v1 = manager.create(
            config={"v": 1},
            created_by="admin",
        )
        assert v1.created_by == "admin"

        v2 = manager.update(
            config={"v": 2},
            created_by="user1",
        )
        assert v2.created_by == "user1"

    def test_metadata_tracking(self) -> None:
        """Test custom metadata on versions."""
        manager = ConfigVersionManager()

        v1 = manager.create(
            config={"v": 1},
            metadata={"env": "staging", "approved": True},
        )

        assert v1.metadata["env"] == "staging"
        assert v1.metadata["approved"] is True
