"""Tests for BotContext."""

import pytest

from dataknobs_bots.bot.context import BotContext


class TestBotContext:
    """Tests for BotContext dataclass."""

    def test_basic_creation(self) -> None:
        """Test creating a basic BotContext."""
        ctx = BotContext(
            conversation_id="conv-123",
            client_id="client-456",
        )
        assert ctx.conversation_id == "conv-123"
        assert ctx.client_id == "client-456"
        assert ctx.user_id is None
        assert ctx.session_metadata == {}
        assert ctx.request_metadata == {}

    def test_full_creation(self) -> None:
        """Test creating BotContext with all fields."""
        ctx = BotContext(
            conversation_id="conv-123",
            client_id="client-456",
            user_id="user-789",
            session_metadata={"key": "value"},
            request_metadata={"request_key": "request_value"},
        )
        assert ctx.user_id == "user-789"
        assert ctx.session_metadata == {"key": "value"}
        assert ctx.request_metadata == {"request_key": "request_value"}

    def test_dict_like_setitem(self) -> None:
        """Test setting items using dict-like access."""
        ctx = BotContext(
            conversation_id="conv-123",
            client_id="client-456",
        )
        ctx["rag_query"] = "test query"
        assert ctx.request_metadata["rag_query"] == "test query"

    def test_dict_like_getitem(self) -> None:
        """Test getting items using dict-like access."""
        ctx = BotContext(
            conversation_id="conv-123",
            client_id="client-456",
            request_metadata={"key": "value"},
        )
        assert ctx["key"] == "value"

    def test_dict_like_getitem_keyerror(self) -> None:
        """Test that KeyError is raised for missing keys."""
        ctx = BotContext(
            conversation_id="conv-123",
            client_id="client-456",
        )
        with pytest.raises(KeyError):
            _ = ctx["nonexistent"]

    def test_dict_like_contains(self) -> None:
        """Test checking if key exists using 'in' operator."""
        ctx = BotContext(
            conversation_id="conv-123",
            client_id="client-456",
            request_metadata={"exists": True},
        )
        assert "exists" in ctx
        assert "missing" not in ctx

    def test_get_with_default(self) -> None:
        """Test get method with default value."""
        ctx = BotContext(
            conversation_id="conv-123",
            client_id="client-456",
            request_metadata={"key": "value"},
        )
        assert ctx.get("key") == "value"
        assert ctx.get("missing") is None
        assert ctx.get("missing", "default") == "default"

    def test_get_without_default(self) -> None:
        """Test get method returns None for missing keys."""
        ctx = BotContext(
            conversation_id="conv-123",
            client_id="client-456",
        )
        assert ctx.get("nonexistent") is None

    def test_multiple_operations(self) -> None:
        """Test multiple dict-like operations."""
        ctx = BotContext(
            conversation_id="conv-123",
            client_id="client-456",
        )

        # Set multiple values
        ctx["key1"] = "value1"
        ctx["key2"] = "value2"

        # Check they exist
        assert "key1" in ctx
        assert "key2" in ctx

        # Get values
        assert ctx["key1"] == "value1"
        assert ctx.get("key2") == "value2"

        # Verify request_metadata
        assert ctx.request_metadata == {"key1": "value1", "key2": "value2"}

    def test_session_metadata_separate_from_dict_access(self) -> None:
        """Test that session_metadata is separate from dict-like access."""
        ctx = BotContext(
            conversation_id="conv-123",
            client_id="client-456",
            session_metadata={"session_key": "session_value"},
        )

        # Session metadata should not be accessible via dict-like access
        assert "session_key" not in ctx
        assert ctx.get("session_key") is None

        # But should be accessible via attribute
        assert ctx.session_metadata["session_key"] == "session_value"

    def test_overwrite_value(self) -> None:
        """Test overwriting an existing value."""
        ctx = BotContext(
            conversation_id="conv-123",
            client_id="client-456",
            request_metadata={"key": "old_value"},
        )

        ctx["key"] = "new_value"
        assert ctx["key"] == "new_value"

    def test_complex_values(self) -> None:
        """Test storing complex values (dict, list)."""
        ctx = BotContext(
            conversation_id="conv-123",
            client_id="client-456",
        )

        ctx["nested"] = {"a": 1, "b": [1, 2, 3]}
        ctx["list"] = [1, 2, 3]

        assert ctx["nested"]["a"] == 1
        assert ctx["list"][1] == 2

    def test_copy_no_overrides(self) -> None:
        """Test copy creates identical context."""
        ctx = BotContext(
            conversation_id="conv-123",
            client_id="client-456",
            user_id="user-789",
            session_metadata={"session": "data"},
            request_metadata={"request": "data"},
        )

        copy = ctx.copy()

        assert copy.conversation_id == "conv-123"
        assert copy.client_id == "client-456"
        assert copy.user_id == "user-789"
        assert copy.session_metadata == {"session": "data"}
        assert copy.request_metadata == {"request": "data"}

    def test_copy_with_overrides(self) -> None:
        """Test copy with field overrides."""
        ctx = BotContext(
            conversation_id="conv-123",
            client_id="client-456",
            user_id="user-789",
        )

        copy = ctx.copy(conversation_id="conv-new", user_id="user-new")

        assert copy.conversation_id == "conv-new"
        assert copy.client_id == "client-456"  # unchanged
        assert copy.user_id == "user-new"

    def test_copy_metadata_isolation(self) -> None:
        """Test that copy creates independent dict copies."""
        ctx = BotContext(
            conversation_id="conv-123",
            client_id="client-456",
            session_metadata={"key": "value"},
            request_metadata={"req": "data"},
        )

        copy = ctx.copy()

        # Modify copy's metadata
        copy.session_metadata["new_key"] = "new_value"
        copy.request_metadata["new_req"] = "new_data"

        # Original should be unchanged
        assert "new_key" not in ctx.session_metadata
        assert "new_req" not in ctx.request_metadata

    def test_copy_override_metadata(self) -> None:
        """Test copy can override metadata dicts entirely."""
        ctx = BotContext(
            conversation_id="conv-123",
            client_id="client-456",
            session_metadata={"old": "session"},
            request_metadata={"old": "request"},
        )

        copy = ctx.copy(
            session_metadata={"new": "session"},
            request_metadata={"new": "request"},
        )

        assert copy.session_metadata == {"new": "session"}
        assert copy.request_metadata == {"new": "request"}
        # Original unchanged
        assert ctx.session_metadata == {"old": "session"}
        assert ctx.request_metadata == {"old": "request"}
