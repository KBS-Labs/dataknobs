"""Tests for context builder.

Uses real :class:`ConversationManager`, :class:`ToolRegistry`, and
:class:`ArtifactRegistry` instances throughout — no ``MagicMock``. The
pre-existing :class:`ContextPersister` bug (read-only-property
assignment) survived in this module precisely because the original
mock-based tests silently accepted attribute writes the production type
rejects; running every test through real types closes that escape hatch
and keeps it closed.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest
import yaml

from dataknobs_bots.artifacts.registry import ArtifactRegistry
from dataknobs_bots.context.builder import ContextBuilder, ContextPersister
from dataknobs_data.backends.memory import AsyncMemoryDatabase
from dataknobs_llm import ToolRegistry
from dataknobs_llm.conversations import (
    ConversationManager,
    DataknobsConversationStorage,
)
from dataknobs_llm.llm import EchoProvider, LLMConfig
from dataknobs_llm.prompts import (
    AsyncPromptBuilder,
    FileSystemPromptLibrary,
)
from dataknobs_llm.tools.base import Tool


class _NoOpTool(Tool):
    """Minimal real :class:`Tool` for exercising :meth:`ToolRegistry.execute_tool`.

    The registry's real recording path runs through ``execute_tool`` —
    using a real subclass over a fake keeps the test exercising the
    production code path that surfaces a record.
    """

    def __init__(self, name: str = "search") -> None:
        super().__init__(name=name, description="no-op test tool")

    @property
    def schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs: Any) -> str:
        return "ok"


class _RaisingToolRegistry(ToolRegistry):
    """A :class:`ToolRegistry` whose history query raises.

    Tests the consumer-side error path in
    :meth:`ContextBuilder._extract_tool_history` through real types —
    overriding the public method to raise is the same fault-injection
    shape a mock's ``side_effect`` would model, but the override lives
    on the real type so the consumer reaches it through the production
    method-resolution path.
    """

    def get_execution_history(self, **kwargs: Any) -> list[Any]:  # type: ignore[override]
        raise RuntimeError("simulated registry error")


@pytest.fixture
async def make_manager(tmp_path: Path):
    """Factory fixture: build real :class:`ConversationManager` instances.

    The factory accepts ``metadata`` (an optional dict to seed) and a
    ``materialize`` flag (default ``True``). When ``materialize=True``
    the manager has an initial user message added, so ``state`` is
    non-``None`` and ``state.metadata`` carries the supplied metadata —
    matching the shape :meth:`ContextBuilder.build` reads through the
    ``manager.metadata`` read-only property. When ``False``, the
    metadata lands on the seed bucket via
    :meth:`ConversationManager.update_seed_metadata` and ``state`` is
    ``None``.

    A single prompt-library directory is set up under pytest's
    ``tmp_path``; each factory call gets its own :class:`EchoProvider`
    and storage so test instances cannot share state. All providers are
    closed on teardown.
    """
    prompt_dir = tmp_path / "prompts"
    (prompt_dir / "system").mkdir(parents=True)
    (prompt_dir / "system" / "helpful.yaml").write_text(
        yaml.dump({"template": "You are a helpful assistant"})
    )
    library = FileSystemPromptLibrary(prompt_dir)

    providers: list[EchoProvider] = []

    async def factory(
        *,
        metadata: dict[str, Any] | None = None,
        conversation_id: str = "conv_123",
        materialize: bool = True,
    ) -> ConversationManager:
        llm = EchoProvider(
            LLMConfig(
                provider="echo",
                model="echo-model",
                options={"echo_prefix": ""},
            )
        )
        providers.append(llm)
        builder = AsyncPromptBuilder(library=library)
        storage = DataknobsConversationStorage(AsyncMemoryDatabase())
        manager = await ConversationManager.create(
            llm=llm,
            prompt_builder=builder,
            storage=storage,
            conversation_id=conversation_id,
        )
        if materialize:
            await manager.add_message(role="user", content="hi")
            if metadata:
                manager.update_metadata(metadata)
        elif metadata:
            manager.update_seed_metadata(metadata)
        return manager

    yield factory

    for llm in providers:
        await llm.close()


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
        db = AsyncMemoryDatabase()
        artifact_registry = ArtifactRegistry(db=db)
        builder = ContextBuilder(artifact_registry=artifact_registry)
        assert builder._artifact_registry == artifact_registry


class TestContextBuilderBuild:
    """Tests for building context from manager."""

    @pytest.mark.asyncio
    async def test_build_empty_manager(self, make_manager) -> None:
        """Build context from a manager with no domain metadata."""
        builder = ContextBuilder()
        manager = await make_manager()

        context = await builder.build(manager)

        assert context.conversation_id == "conv_123"
        assert context.wizard_stage is None
        assert context.artifacts == []
        assert context.assumptions == []

    @pytest.mark.asyncio
    async def test_build_with_wizard_state(self, make_manager) -> None:
        """Build context from a manager carrying wizard FSM state."""
        builder = ContextBuilder()
        manager = await make_manager(
            metadata={
                "wizard": {
                    "progress": 0.5,
                    "fsm_state": {
                        "current_stage": "collect_info",
                        "data": {"name": "test"},
                        "tasks": {
                            "tasks": [{"id": "t1", "status": "pending"}]
                        },
                        "transitions": [
                            {"from_stage": "welcome", "to_stage": "collect_info"}
                        ],
                    },
                }
            }
        )

        context = await builder.build(manager)

        assert context.wizard_stage == "collect_info"
        assert context.wizard_data == {"name": "test"}
        assert context.wizard_progress == 0.5
        assert len(context.wizard_tasks) == 1
        assert len(context.transitions) == 1

    @pytest.mark.asyncio
    async def test_build_with_artifact_registry(self, make_manager) -> None:
        """Artifact registry feeds context.artifacts when present."""
        db = AsyncMemoryDatabase()
        registry = ArtifactRegistry(db=db)
        await registry.create("content", "Test 1", {"v": 1})
        await registry.create("content", "Test 2", {"v": 2})

        builder = ContextBuilder(artifact_registry=registry)
        manager = await make_manager()

        context = await builder.build(manager)

        assert len(context.artifacts) == 2

    @pytest.mark.asyncio
    async def test_build_with_artifacts_in_metadata(self, make_manager) -> None:
        """Fall back to metadata['artifacts'] when no registry is configured."""
        builder = ContextBuilder()
        manager = await make_manager(
            metadata={
                "artifacts": [
                    {"id": "a1", "name": "Test", "status": "draft"}
                ]
            }
        )

        context = await builder.build(manager)

        assert len(context.artifacts) == 1
        assert context.artifacts[0]["id"] == "a1"

    @pytest.mark.asyncio
    async def test_build_with_assumptions(self, make_manager) -> None:
        """Build context with assumptions from metadata['context']."""
        builder = ContextBuilder()
        manager = await make_manager(
            metadata={
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
        )

        context = await builder.build(manager)

        assert len(context.assumptions) == 1
        assert context.assumptions[0].content == "Test assumption"
        assert context.assumptions[0].confidence == 0.7

    @pytest.mark.asyncio
    async def test_build_with_tool_history(self, make_manager) -> None:
        """Build context with tool history from metadata['tool_history']."""
        builder = ContextBuilder()
        manager = await make_manager(
            metadata={
                "tool_history": [
                    {"tool_name": "search", "success": True, "duration_ms": 100}
                ]
            }
        )

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
    """Tool-registry integration tests against real :class:`ToolRegistry`."""

    @pytest.mark.asyncio
    async def test_build_with_tool_registry(self, make_manager) -> None:
        """A real registry's recorded execution is surfaced by build()."""
        tool_registry = ToolRegistry(track_executions=True)
        tool_registry.register_tool(_NoOpTool(name="search"))

        manager = await make_manager()
        # Execute through ``execute_tool`` — the same registry method
        # DynaBot's ``_execute_tools`` routes through in production, so
        # the recorded code path is the production one. ``_context=manager``
        # lets the registry pull ``conversation_id`` for the record.
        await tool_registry.execute_tool("search", _context=manager)

        builder = ContextBuilder(tool_registry=tool_registry)
        context = await builder.build(manager)

        assert len(context.tool_history) == 1
        record = context.tool_history[0]
        assert record["tool_name"] == "search"
        assert record["success"] is True
        # Duration is whatever a real execution took — bound it loosely.
        assert record["duration_ms"] >= 0
        # Timestamp is the real start time of the recorded execution.
        assert record["timestamp"] <= time.time()

    @pytest.mark.asyncio
    async def test_build_with_tool_registry_error(self, make_manager) -> None:
        """Build falls back to metadata when the registry's query raises.

        Uses a real :class:`_RaisingToolRegistry` subclass that overrides
        :meth:`get_execution_history` to raise — exercising the consumer's
        try/except branch through the production method-resolution path.
        """
        tool_registry = _RaisingToolRegistry(track_executions=True)

        builder = ContextBuilder(tool_registry=tool_registry)
        manager = await make_manager(
            metadata={
                "tool_history": [{"tool_name": "fallback", "success": True}]
            }
        )

        context = await builder.build(manager)

        assert len(context.tool_history) == 1
        assert context.tool_history[0]["tool_name"] == "fallback"


class TestContextPersisterAgainstRealManager:
    """Behavioural pins exercising :class:`ContextPersister` against a
    real :class:`ConversationManager`.

    The original :class:`TestContextPersister` used ``MagicMock`` for
    the manager, which silently accepted attribute assignment to
    ``metadata`` — so it did not catch the production bug where the
    real manager's ``metadata`` is a read-only ``@property``. These
    tests pin both paths (pre-state and post-state).
    """

    @pytest.mark.asyncio
    async def test_persist_against_pre_state_manager_writes_context_section(
        self, make_manager
    ) -> None:
        """Reproducing pin for the latent ``AttributeError`` bug.

        Against the pre-fix code, ``ContextPersister.persist`` did
        ``manager.metadata = metadata`` against a manager whose
        ``metadata`` is a read-only ``@property``. This raises
        ``AttributeError: can't set attribute``. Fails on pre-fix HEAD;
        passes once :meth:`persist` routes through
        ``update_seed_metadata``.
        """
        from dataknobs_bots.context.accumulator import ConversationContext

        manager = await make_manager(materialize=False)
        assert manager.state is None  # Pre-state.

        context = ConversationContext(conversation_id=manager.conversation_id)
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
        self, make_manager
    ) -> None:
        """Post-state path — also raised ``AttributeError`` before the
        fix. Also pins the original *replace* semantic of the
        ``metadata["context"] = ...`` write: calling :meth:`persist`
        again with an updated context replaces the section rather than
        merging into it.
        """
        from dataknobs_bots.context.accumulator import ConversationContext

        manager = await make_manager()
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
    async def test_persist_preserves_other_metadata_keys(
        self, make_manager
    ) -> None:
        """Persisting context must not clobber other metadata keys.

        Pins the "wizard / other top-level keys survive" semantic the
        legacy mock-based test asserted, but against a real manager so
        the write path is the production one.
        """
        from dataknobs_bots.context.accumulator import ConversationContext

        manager = await make_manager(materialize=False)
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
