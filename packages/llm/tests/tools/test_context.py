"""Tests for ToolExecutionContext and WizardStateSnapshot."""

from typing import Any
from unittest.mock import MagicMock

import pytest

from dataknobs_llm.tools.context import ToolExecutionContext, WizardStateSnapshot


class TestWizardStateSnapshot:
    """Tests for WizardStateSnapshot."""

    def test_default_values(self) -> None:
        """Test that default values are set correctly."""
        snapshot = WizardStateSnapshot()

        assert snapshot.current_stage is None
        assert snapshot.collected_data == {}
        assert snapshot.history == []
        assert snapshot.completed is False
        assert snapshot.stage_metadata == {}

    def test_custom_values(self) -> None:
        """Test creating snapshot with custom values."""
        snapshot = WizardStateSnapshot(
            current_stage="configure",
            collected_data={"name": "test-bot", "template": "tutor"},
            history=["welcome", "select_template", "configure"],
            completed=False,
            stage_metadata={"prompt": "Configure your bot"},
        )

        assert snapshot.current_stage == "configure"
        assert snapshot.collected_data["name"] == "test-bot"
        assert len(snapshot.history) == 3
        assert snapshot.completed is False

    def test_from_manager_metadata_empty(self) -> None:
        """Test creating snapshot from empty metadata."""
        snapshot = WizardStateSnapshot.from_manager_metadata({})

        assert snapshot.current_stage is None
        assert snapshot.collected_data == {}
        assert snapshot.history == []
        assert snapshot.completed is False

    def test_from_manager_metadata_with_wizard_data(self) -> None:
        """Test creating snapshot from metadata with wizard state."""
        metadata = {
            "wizard": {
                "fsm_state": {
                    "current_stage": "review",
                    "data": {"domain_id": "math-tutor", "domain_name": "Math Tutor"},
                    "history": ["welcome", "configure", "review"],
                    "completed": False,
                }
            }
        }

        snapshot = WizardStateSnapshot.from_manager_metadata(metadata)

        assert snapshot.current_stage == "review"
        assert snapshot.collected_data["domain_id"] == "math-tutor"
        assert snapshot.collected_data["domain_name"] == "Math Tutor"
        assert len(snapshot.history) == 3
        assert snapshot.completed is False

    def test_from_manager_metadata_completed(self) -> None:
        """Test creating snapshot from completed wizard."""
        metadata = {
            "wizard": {
                "fsm_state": {
                    "current_stage": "complete",
                    "data": {"domain_id": "my-bot"},
                    "history": ["welcome", "configure", "complete"],
                    "completed": True,
                }
            }
        }

        snapshot = WizardStateSnapshot.from_manager_metadata(metadata)

        assert snapshot.current_stage == "complete"
        assert snapshot.completed is True


class TestToolExecutionContext:
    """Tests for ToolExecutionContext."""

    def test_empty_context(self) -> None:
        """Test creating empty context."""
        context = ToolExecutionContext.empty()

        assert context.conversation_id is None
        assert context.user_id is None
        assert context.client_id is None
        assert context.conversation_metadata == {}
        assert context.wizard_state is None
        assert context.request_metadata == {}
        assert context.extra == {}

    def test_custom_context(self) -> None:
        """Test creating context with custom values."""
        wizard_state = WizardStateSnapshot(
            current_stage="configure",
            collected_data={"name": "test"},
        )

        context = ToolExecutionContext(
            conversation_id="conv-123",
            user_id="user-456",
            client_id="client-789",
            conversation_metadata={"key": "value"},
            wizard_state=wizard_state,
            request_metadata={"header": "x-custom"},
            extra={"custom_key": "custom_value"},
        )

        assert context.conversation_id == "conv-123"
        assert context.user_id == "user-456"
        assert context.client_id == "client-789"
        assert context.conversation_metadata["key"] == "value"
        assert context.wizard_state is not None
        assert context.wizard_state.current_stage == "configure"
        assert context.request_metadata["header"] == "x-custom"
        assert context.extra["custom_key"] == "custom_value"

    def test_from_manager_basic(self) -> None:
        """Test creating context from manager without wizard state."""
        manager = MagicMock()
        manager.conversation_id = "conv-abc"
        manager.metadata = {"some_key": "some_value"}

        context = ToolExecutionContext.from_manager(manager)

        assert context.conversation_id == "conv-abc"
        assert context.conversation_metadata["some_key"] == "some_value"
        assert context.wizard_state is None

    def test_from_manager_with_wizard_state(self) -> None:
        """Test creating context from manager with wizard state."""
        manager = MagicMock()
        manager.conversation_id = "conv-xyz"
        manager.metadata = {
            "wizard": {
                "fsm_state": {
                    "current_stage": "configure",
                    "data": {"domain_id": "test-bot"},
                    "history": ["welcome", "configure"],
                    "completed": False,
                }
            }
        }

        context = ToolExecutionContext.from_manager(manager)

        assert context.conversation_id == "conv-xyz"
        assert context.wizard_state is not None
        assert context.wizard_state.current_stage == "configure"
        assert context.wizard_state.collected_data["domain_id"] == "test-bot"

    def test_from_manager_with_extra(self) -> None:
        """Test creating context with extra values."""
        manager = MagicMock()
        manager.conversation_id = "conv-123"
        manager.metadata = {}

        context = ToolExecutionContext.from_manager(
            manager,
            request_metadata={"trace_id": "trace-456"},
            extra={"custom": "data"},
        )

        assert context.request_metadata["trace_id"] == "trace-456"
        assert context.extra["custom"] == "data"

    def test_get_from_extra(self) -> None:
        """Test dict-like access to extra values."""
        context = ToolExecutionContext(
            extra={"key1": "value1", "key2": 42}
        )

        assert context.get("key1") == "value1"
        assert context.get("key2") == 42
        assert context.get("missing") is None
        assert context.get("missing", "default") == "default"

    def test_with_extra_creates_new_context(self) -> None:
        """Test that with_extra creates a new context."""
        original = ToolExecutionContext(
            conversation_id="conv-123",
            extra={"key1": "value1"},
        )

        new_context = original.with_extra(key2="value2", key3="value3")

        # Original is unchanged
        assert "key2" not in original.extra
        assert "key3" not in original.extra

        # New context has all values
        assert new_context.extra["key1"] == "value1"
        assert new_context.extra["key2"] == "value2"
        assert new_context.extra["key3"] == "value3"

        # Other fields are preserved
        assert new_context.conversation_id == "conv-123"

    def test_from_manager_handles_missing_attributes(self) -> None:
        """Test that from_manager handles managers without expected attributes."""
        # Manager without conversation_id
        manager = MagicMock(spec=[])
        manager.metadata = {}

        context = ToolExecutionContext.from_manager(manager)

        assert context.conversation_id is None
