"""Tests for context builder."""

from unittest.mock import MagicMock

import pytest

from dataknobs_bots.artifacts.registry import ArtifactRegistry
from dataknobs_bots.context.builder import ContextBuilder, ContextPersister


class TestContextBuilderBasic:
    """Basic tests for ContextBuilder."""

    def test_init_empty(self) -> None:
        """Test empty initialization."""
        builder = ContextBuilder()
        assert builder._artifact_registry is None
        assert builder._tool_registry is None

    @pytest.mark.asyncio
    async def test_init_with_registries(self) -> None:
        """Test initialization with registries."""
        from dataknobs_data.backends.memory import AsyncMemoryDatabase

        db = AsyncMemoryDatabase()
        artifact_registry = ArtifactRegistry(db=db)
        builder = ContextBuilder(artifact_registry=artifact_registry)
        assert builder._artifact_registry == artifact_registry


class TestContextBuilderBuild:
    """Tests for building context from manager."""

    @pytest.mark.asyncio
    async def test_build_empty_manager(self) -> None:
        """Test building context from empty manager."""
        builder = ContextBuilder()
        manager = MagicMock()
        manager.metadata = {}
        manager.conversation_id = "conv_123"

        context = await builder.build(manager)

        assert context.conversation_id == "conv_123"
        assert context.wizard_stage is None
        assert context.artifacts == []
        assert context.assumptions == []

    @pytest.mark.asyncio
    async def test_build_with_wizard_state(self) -> None:
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

        context = await builder.build(manager)

        assert context.wizard_stage == "collect_info"
        assert context.wizard_data == {"name": "test"}
        assert context.wizard_progress == 0.5
        assert len(context.wizard_tasks) == 1
        assert len(context.transitions) == 1

    @pytest.mark.asyncio
    async def test_build_with_artifact_registry(self) -> None:
        """Test building context with artifact registry."""
        from dataknobs_data.backends.memory import AsyncMemoryDatabase

        db = AsyncMemoryDatabase()
        registry = ArtifactRegistry(db=db)
        await registry.create("content", "Test 1", {"v": 1})
        await registry.create("content", "Test 2", {"v": 2})

        builder = ContextBuilder(artifact_registry=registry)
        manager = MagicMock()
        manager.metadata = {}
        manager.conversation_id = "conv_123"

        context = await builder.build(manager)

        assert len(context.artifacts) == 2

    @pytest.mark.asyncio
    async def test_build_with_artifacts_in_metadata(self) -> None:
        """Test building context with artifacts in metadata (no registry)."""
        builder = ContextBuilder()
        manager = MagicMock()
        manager.conversation_id = "conv_123"
        manager.metadata = {
            "artifacts": [
                {"id": "a1", "name": "Test", "status": "draft"}
            ]
        }

        context = await builder.build(manager)

        assert len(context.artifacts) == 1
        assert context.artifacts[0]["id"] == "a1"

    @pytest.mark.asyncio
    async def test_build_with_assumptions(self) -> None:
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

        context = await builder.build(manager)

        assert len(context.assumptions) == 1
        assert context.assumptions[0].content == "Test assumption"
        assert context.assumptions[0].confidence == 0.7

    @pytest.mark.asyncio
    async def test_build_with_tool_history(self) -> None:
        """Test building context with tool history from metadata."""
        builder = ContextBuilder()
        manager = MagicMock()
        manager.conversation_id = "conv_123"
        manager.metadata = {
            "tool_history": [
                {"tool_name": "search", "success": True, "duration_ms": 100}
            ]
        }

        context = await builder.build(manager)

        assert len(context.tool_history) == 1
        assert context.tool_history[0]["tool_name"] == "search"


class TestContextBuilderFromMetadata:
    """Tests for building context from metadata dict directly."""

    @pytest.mark.asyncio
    async def test_build_from_metadata(self) -> None:
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

        context = await builder.build_from_metadata(metadata, conversation_id="conv_123")

        assert context.conversation_id == "conv_123"
        assert context.wizard_stage == "review"
        assert context.wizard_progress == 0.75
        assert len(context.assumptions) == 1


class TestContextPersister:
    """Tests for ContextPersister.

    Behavioural pins against a real :class:`ConversationManager` live in
    :class:`TestContextPersisterAgainstRealManager` below; those cover
    the persist→read round trip end-to-end. This class is now scoped to
    the manager-free :meth:`ContextPersister.persist_to_dict` shape.
    """

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

    @pytest.mark.asyncio
    async def test_build_with_tool_registry(self) -> None:
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

        context = await builder.build(manager)

        assert len(context.tool_history) == 1
        assert context.tool_history[0]["tool_name"] == "search"
        assert context.tool_history[0]["success"] is True

    @pytest.mark.asyncio
    async def test_build_with_tool_registry_error(self) -> None:
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
        context = await builder.build(manager)

        assert len(context.tool_history) == 1
        assert context.tool_history[0]["tool_name"] == "fallback"


class TestContextPersisterAgainstRealManager:
    """Behavioural pins exercising :class:`ContextPersister` against a
    real :class:`ConversationManager`.

    The existing :class:`TestContextPersister` uses ``MagicMock`` for the
    manager, which silently accepts attribute assignment to ``metadata``
    — so it did not catch the production bug where the real manager's
    ``metadata`` is a read-only ``@property``. These tests pin both
    paths (pre-state and post-state).
    """

    @staticmethod
    async def _new_manager():
        """Build a real :class:`ConversationManager` with echo provider
        and in-memory storage."""
        import tempfile
        from pathlib import Path

        import yaml as _yaml

        from dataknobs_data.backends.memory import AsyncMemoryDatabase
        from dataknobs_llm.conversations import (
            ConversationManager,
            DataknobsConversationStorage,
        )
        from dataknobs_llm.llm import EchoProvider, LLMConfig
        from dataknobs_llm.prompts import (
            AsyncPromptBuilder,
            FileSystemPromptLibrary,
        )

        tmpdir = tempfile.mkdtemp()
        prompt_dir = Path(tmpdir) / "prompts"
        (prompt_dir / "system").mkdir(parents=True)
        (prompt_dir / "user").mkdir(parents=True)
        (prompt_dir / "system" / "helpful.yaml").write_text(
            _yaml.dump({"template": "You are a helpful assistant"})
        )

        llm = EchoProvider(
            LLMConfig(
                provider="echo",
                model="echo-model",
                options={"echo_prefix": ""},
            )
        )
        library = FileSystemPromptLibrary(prompt_dir)
        builder = AsyncPromptBuilder(library=library)
        storage = DataknobsConversationStorage(AsyncMemoryDatabase())

        return await ConversationManager.create(
            llm=llm, prompt_builder=builder, storage=storage
        )

    @pytest.mark.asyncio
    async def test_persist_against_pre_state_manager_writes_context_section(
        self,
    ) -> None:
        """Reproducing pin for the latent ``AttributeError`` bug.

        Against HEAD, ``ContextPersister.persist`` does
        ``manager.metadata = metadata`` against a manager whose
        ``metadata`` is a read-only ``@property``. This raises
        ``AttributeError: can't set attribute``. Test fails on HEAD;
        passes once :meth:`persist` routes through
        ``update_seed_metadata``.
        """
        from dataknobs_bots.context.accumulator import ConversationContext

        manager = await self._new_manager()
        # Pre-state: no messages added yet.
        assert manager.state is None

        context = ConversationContext(
            conversation_id=manager.conversation_id
        )
        context.add_assumption(content="user wants tutor", confidence=0.8)
        context.add_section(name="profile", content="advanced", priority=70)

        persister = ContextPersister()
        persister.persist(context, manager)  # must not raise

        # After the fix lands, the context section is readable through
        # the seed accessor pre-state...
        seeded = manager.get_seed_metadata("context")
        assert seeded is not None
        assert len(seeded["assumptions"]) == 1
        assert seeded["assumptions"][0]["content"] == "user wants tutor"
        assert len(seeded["sections"]) == 1
        assert seeded["sections"][0]["name"] == "profile"

        # ...and survives state materialization.
        await manager.add_message(role="user", content="hi")
        assert manager.state.metadata["context"]["sections"][0]["name"] == "profile"

    @pytest.mark.asyncio
    async def test_persist_against_post_state_manager_replaces_section(
        self,
    ) -> None:
        """Post-state path — also raised ``AttributeError`` before the
        fix. Also pins the original *replace* semantic of the
        ``metadata["context"] = ...`` write: calling :meth:`persist`
        again with an updated context replaces the section rather than
        merging into it.
        """
        from dataknobs_bots.context.accumulator import ConversationContext

        manager = await self._new_manager()
        await manager.add_message(role="user", content="hi")
        assert manager.state is not None

        persister = ContextPersister()

        first = ConversationContext(conversation_id=manager.conversation_id)
        first.add_assumption(content="first", confidence=0.5)
        persister.persist(first, manager)
        assert manager.state.metadata["context"]["assumptions"][0]["content"] == "first"

        # Second call with a different context — must replace, not merge.
        second = ConversationContext(conversation_id=manager.conversation_id)
        second.add_section(name="custom", content="payload", priority=60)
        persister.persist(second, manager)

        section = manager.state.metadata["context"]
        assert section["assumptions"] == []
        assert len(section["sections"]) == 1
        assert section["sections"][0]["name"] == "custom"

    @pytest.mark.asyncio
    async def test_persist_preserves_other_metadata_keys(self) -> None:
        """Persisting context must not clobber other metadata keys.

        Pins the "wizard / other top-level keys survive" semantic the
        legacy mock-based test asserted, but against a real manager so
        the write path is the production one.
        """
        from dataknobs_bots.context.accumulator import ConversationContext

        manager = await self._new_manager()
        manager.update_seed_metadata(
            {"wizard": {"progress": 0.5}, "other": "data"}
        )

        context = ConversationContext(conversation_id=manager.conversation_id)
        context.add_assumption(content="new")

        persister = ContextPersister()
        persister.persist(context, manager)

        # Pre-state: read via the seed-aware accessor.
        assert manager.get_seed_metadata("wizard") == {"progress": 0.5}
        assert manager.get_seed_metadata("other") == "data"
        assert manager.get_seed_metadata("context") is not None

        # Post-state: the same keys land on state.metadata after the
        # first message materializes state.
        await manager.add_message(role="user", content="hi")
        assert manager.state.metadata["wizard"] == {"progress": 0.5}
        assert manager.state.metadata["other"] == "data"
        assert "context" in manager.state.metadata
