"""Tests for context builder."""

from unittest.mock import MagicMock

import pytest

from dataknobs_bots.artifacts.models import Artifact
from dataknobs_bots.artifacts.registry import ArtifactRegistry
from dataknobs_bots.context.builder import ContextBuilder, ContextPersister


class TestContextBuilderBasic:
    """Basic tests for ContextBuilder."""

    def test_init_empty(self) -> None:
        """Test empty initialization."""
        builder = ContextBuilder()
        assert builder._artifact_registry is None
        assert builder._tool_registry is None

    def test_init_with_registries(self) -> None:
        """Test initialization with registries."""
        artifact_registry = ArtifactRegistry()
        builder = ContextBuilder(artifact_registry=artifact_registry)
        assert builder._artifact_registry == artifact_registry


class TestContextBuilderBuild:
    """Tests for building context from manager."""

    def test_build_empty_manager(self) -> None:
        """Test building context from empty manager."""
        builder = ContextBuilder()
        manager = MagicMock()
        manager.metadata = {}
        manager.conversation_id = "conv_123"

        context = builder.build(manager)

        assert context.conversation_id == "conv_123"
        assert context.wizard_stage is None
        assert context.artifacts == []
        assert context.assumptions == []

    def test_build_with_wizard_state(self) -> None:
        """Test building context with wizard state."""
        builder = ContextBuilder()
        manager = MagicMock()
        manager.conversation_id = "conv_123"
        manager.metadata = {
            "wizard": {
                "progress": 0.5,
                "fsm_state": {
                    "current_stage": "collect_info",
                    "data": {"name": "test"},
                    "tasks": {
                        "tasks": [
                            {"id": "t1", "status": "pending"}
                        ]
                    },
                    "transitions": [
                        {"from_stage": "welcome", "to_stage": "collect_info"}
                    ],
                },
            }
        }

        context = builder.build(manager)

        assert context.wizard_stage == "collect_info"
        assert context.wizard_data == {"name": "test"}
        assert context.wizard_progress == 0.5
        assert len(context.wizard_tasks) == 1
        assert len(context.transitions) == 1

    def test_build_with_artifact_registry(self) -> None:
        """Test building context with artifact registry."""
        artifact_registry = ArtifactRegistry()
        artifact_registry.create(content={"v": 1}, name="Test 1")
        artifact_registry.create(content={"v": 2}, name="Test 2")

        builder = ContextBuilder(artifact_registry=artifact_registry)
        manager = MagicMock()
        manager.metadata = {}
        manager.conversation_id = "conv_123"

        context = builder.build(manager)

        assert len(context.artifacts) == 2

    def test_build_with_artifacts_in_metadata(self) -> None:
        """Test building context with artifacts in metadata (no registry)."""
        builder = ContextBuilder()
        manager = MagicMock()
        manager.conversation_id = "conv_123"
        manager.metadata = {
            "artifacts": [
                {"id": "a1", "name": "Test", "status": "draft"}
            ]
        }

        context = builder.build(manager)

        assert len(context.artifacts) == 1
        assert context.artifacts[0]["id"] == "a1"

    def test_build_with_assumptions(self) -> None:
        """Test building context with assumptions from metadata."""
        builder = ContextBuilder()
        manager = MagicMock()
        manager.conversation_id = "conv_123"
        manager.metadata = {
            "context": {
                "assumptions": [
                    {
                        "id": "asn_1",
                        "content": "Test assumption",
                        "source": "inferred",
                        "confidence": 0.7,
                    }
                ]
            }
        }

        context = builder.build(manager)

        assert len(context.assumptions) == 1
        assert context.assumptions[0].content == "Test assumption"
        assert context.assumptions[0].confidence == 0.7

    def test_build_with_tool_history(self) -> None:
        """Test building context with tool history from metadata."""
        builder = ContextBuilder()
        manager = MagicMock()
        manager.conversation_id = "conv_123"
        manager.metadata = {
            "tool_history": [
                {"tool_name": "search", "success": True, "duration_ms": 100}
            ]
        }

        context = builder.build(manager)

        assert len(context.tool_history) == 1
        assert context.tool_history[0]["tool_name"] == "search"


class TestContextBuilderFromMetadata:
    """Tests for building context from metadata dict directly."""

    def test_build_from_metadata(self) -> None:
        """Test building context directly from metadata."""
        builder = ContextBuilder()
        metadata = {
            "wizard": {
                "progress": 0.75,
                "fsm_state": {
                    "current_stage": "review",
                    "data": {"item": "value"},
                },
            },
            "context": {
                "assumptions": [
                    {"content": "Test", "source": "inferred", "confidence": 0.5}
                ]
            },
        }

        context = builder.build_from_metadata(metadata, conversation_id="conv_123")

        assert context.conversation_id == "conv_123"
        assert context.wizard_stage == "review"
        assert context.wizard_progress == 0.75
        assert len(context.assumptions) == 1


class TestContextPersister:
    """Tests for ContextPersister."""

    def test_persist_to_manager(self) -> None:
        """Test persisting context to manager."""
        from dataknobs_bots.context.accumulator import ConversationContext

        context = ConversationContext(conversation_id="conv_123")
        context.add_assumption(content="Test assumption", confidence=0.8)
        context.add_section(name="custom", content="data", priority=80)

        manager = MagicMock()
        manager.metadata = {}

        persister = ContextPersister()
        persister.persist(context, manager)

        # Check that metadata was updated
        saved_metadata = manager.metadata
        assert "context" in saved_metadata
        assert len(saved_metadata["context"]["assumptions"]) == 1
        assert len(saved_metadata["context"]["sections"]) == 1
        assert saved_metadata["context"]["assumptions"][0]["content"] == "Test assumption"

    def test_persist_to_existing_metadata(self) -> None:
        """Test persisting context to manager with existing metadata."""
        from dataknobs_bots.context.accumulator import ConversationContext

        context = ConversationContext()
        context.add_assumption(content="New assumption")

        manager = MagicMock()
        manager.metadata = {
            "wizard": {"progress": 0.5},
            "other": "data",
        }

        persister = ContextPersister()
        persister.persist(context, manager)

        # Existing data should be preserved
        assert "wizard" in manager.metadata
        assert "other" in manager.metadata
        # Context data should be added
        assert "context" in manager.metadata

    def test_persist_to_dict(self) -> None:
        """Test getting context as dict without manager."""
        from dataknobs_bots.context.accumulator import ConversationContext

        context = ConversationContext()
        context.add_assumption(content="Test", confidence=0.9)
        context.add_section(name="custom", content="test", priority=60)

        persister = ContextPersister()
        data = persister.persist_to_dict(context)

        assert "context" in data
        assert len(data["context"]["assumptions"]) == 1
        assert len(data["context"]["sections"]) == 1
        assert "updated_at" in data["context"]


class TestContextBuilderToolRegistry:
    """Tests for tool registry integration."""

    def test_build_with_tool_registry(self) -> None:
        """Test building context with tool registry."""
        # Create mock tool registry
        tool_registry = MagicMock()
        execution_record = MagicMock()
        execution_record.tool_name = "search"
        execution_record.timestamp = 12345.0
        execution_record.success = True
        execution_record.duration_ms = 150
        tool_registry.get_execution_history.return_value = [execution_record]

        builder = ContextBuilder(tool_registry=tool_registry)
        manager = MagicMock()
        manager.conversation_id = "conv_123"
        manager.metadata = {}

        context = builder.build(manager)

        assert len(context.tool_history) == 1
        assert context.tool_history[0]["tool_name"] == "search"
        assert context.tool_history[0]["success"] is True

    def test_build_with_tool_registry_error(self) -> None:
        """Test building context when tool registry throws error."""
        # Create mock tool registry that throws
        tool_registry = MagicMock()
        tool_registry.get_execution_history.side_effect = Exception("Registry error")

        builder = ContextBuilder(tool_registry=tool_registry)
        manager = MagicMock()
        manager.conversation_id = "conv_123"
        manager.metadata = {
            "tool_history": [{"tool_name": "fallback", "success": True}]
        }

        # Should fall back to metadata
        context = builder.build(manager)

        assert len(context.tool_history) == 1
        assert context.tool_history[0]["tool_name"] == "fallback"
