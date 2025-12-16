"""Tests for registry models."""

from datetime import datetime, timezone

import pytest

from dataknobs_bots.registry import Registration


class TestRegistration:
    """Tests for Registration dataclass."""

    def test_create_with_defaults(self):
        """Test creating Registration with default values."""
        reg = Registration(
            bot_id="test-bot",
            config={"llm": {"provider": "echo"}},
        )

        assert reg.bot_id == "test-bot"
        assert reg.config == {"llm": {"provider": "echo"}}
        assert reg.status == "active"
        assert reg.created_at is not None
        assert reg.updated_at is not None
        assert reg.last_accessed_at is not None

    def test_create_with_status(self):
        """Test creating Registration with custom status."""
        reg = Registration(
            bot_id="test-bot",
            config={},
            status="inactive",
        )

        assert reg.status == "inactive"

    def test_create_with_timestamps(self):
        """Test creating Registration with custom timestamps."""
        created = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        updated = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        accessed = datetime(2024, 12, 1, 12, 0, 0, tzinfo=timezone.utc)

        reg = Registration(
            bot_id="test-bot",
            config={},
            created_at=created,
            updated_at=updated,
            last_accessed_at=accessed,
        )

        assert reg.created_at == created
        assert reg.updated_at == updated
        assert reg.last_accessed_at == accessed

    def test_to_dict(self):
        """Test converting Registration to dict."""
        reg = Registration(
            bot_id="test-bot",
            config={"key": "value"},
            status="active",
        )

        data = reg.to_dict()

        assert data["bot_id"] == "test-bot"
        assert data["config"] == {"key": "value"}
        assert data["status"] == "active"
        assert data["created_at"] is not None
        assert isinstance(data["created_at"], str)  # ISO format string

    def test_from_dict(self):
        """Test creating Registration from dict."""
        data = {
            "bot_id": "restored-bot",
            "config": {"restored": True},
            "status": "inactive",
            "created_at": "2024-01-01T12:00:00+00:00",
            "updated_at": "2024-06-01T12:00:00+00:00",
            "last_accessed_at": "2024-12-01T12:00:00+00:00",
        }

        reg = Registration.from_dict(data)

        assert reg.bot_id == "restored-bot"
        assert reg.config == {"restored": True}
        assert reg.status == "inactive"
        assert reg.created_at.year == 2024
        assert reg.created_at.month == 1

    def test_from_dict_with_defaults(self):
        """Test from_dict with missing optional fields."""
        data = {
            "bot_id": "minimal-bot",
            "config": {},
        }

        reg = Registration.from_dict(data)

        assert reg.bot_id == "minimal-bot"
        assert reg.status == "active"  # Default
        assert reg.created_at is not None  # Generated

    def test_roundtrip(self):
        """Test to_dict -> from_dict roundtrip."""
        original = Registration(
            bot_id="roundtrip-bot",
            config={"nested": {"key": "value"}},
            status="error",
        )

        data = original.to_dict()
        restored = Registration.from_dict(data)

        assert restored.bot_id == original.bot_id
        assert restored.config == original.config
        assert restored.status == original.status

    def test_repr(self):
        """Test string representation."""
        reg = Registration(
            bot_id="repr-bot",
            config={},
            status="active",
        )

        repr_str = repr(reg)

        assert "Registration" in repr_str
        assert "repr-bot" in repr_str
        assert "active" in repr_str
