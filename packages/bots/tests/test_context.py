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
