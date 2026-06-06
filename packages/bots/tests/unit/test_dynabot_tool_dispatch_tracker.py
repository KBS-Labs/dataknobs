"""Pin DynaBot's tool dispatch through :meth:`ToolRegistry.execute_tool`.

Before this fix, :meth:`DynaBot._execute_tools` called ``tool.execute``
directly, leaving the registry's execution tracker empty in production —
so the "tool history in context" feature surfaced by
:meth:`ContextBuilder._extract_tool_history` was effectively dead code on
real bot turns. The fix routes DynaBot's dispatch through
``registry.execute_tool``, which records onto the tracker as a side
effect when ``track_executions=True``.

Reproducing pin (``test_execute_tools_populates_registry_tracker``)
fails on pre-fix HEAD: the tool runs (so the turn completes) but the
tracker stays empty because nothing went through the registry's
recording code path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml

from dataknobs_bots.bot.base import DynaBot
from dataknobs_bots.bot.context import BotContext
from dataknobs_bots.bot.turn import TurnMode, TurnState
from dataknobs_data.backends.memory import AsyncMemoryDatabase
from dataknobs_llm import ToolRegistry
from dataknobs_llm.conversations import (
    ConversationManager,
    DataknobsConversationStorage,
)
from dataknobs_llm.llm import EchoProvider, LLMConfig
from dataknobs_llm.prompts import AsyncPromptBuilder, FileSystemPromptLibrary
from dataknobs_llm.tools.base import Tool


@dataclass
class _SimpleToolCall:
    """A minimal ``(name, parameters)`` carrier for ``_execute_tools``.

    Mirrors the duck-typed contract DynaBot uses: ``.name`` and
    ``.parameters`` are the only attributes read.
    """

    name: str
    parameters: dict[str, Any]


class _NoOpTool(Tool):
    """Real :class:`Tool` whose ``execute`` returns a fixed value."""

    def __init__(self, name: str = "search") -> None:
        super().__init__(name=name, description="no-op test tool")

    @property
    def schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs: Any) -> str:
        return "ok"


@pytest.fixture
async def bot_with_tracking_registry(tmp_path: Path):
    """Build a real :class:`DynaBot` whose tool_registry tracks executions.

    Returns a tuple ``(bot, manager, tool)`` so the test can drive a
    turn through ``_execute_tools`` and inspect the tracker afterwards.
    """
    prompt_dir = tmp_path / "prompts"
    (prompt_dir / "system").mkdir(parents=True)
    (prompt_dir / "system" / "helpful.yaml").write_text(
        yaml.dump({"template": "You are a helpful assistant"})
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

    tool = _NoOpTool()
    registry = ToolRegistry(track_executions=True)
    registry.register_tool(tool)

    bot = DynaBot(
        llm=llm,
        prompt_builder=builder,
        conversation_storage=storage,
        tool_registry=registry,
    )

    manager = await ConversationManager.create(
        llm=llm,
        prompt_builder=builder,
        storage=storage,
        conversation_id="conv-tracker-test",
    )
    await manager.add_message(role="user", content="hi")

    yield bot, manager, tool

    await llm.close()


class TestDynaBotPopulatesRegistryTracker:
    """Behavioural pin: DynaBot's dispatch surfaces records on the tracker.

    The chain ``DynaBot._execute_tools → registry.execute_tool →
    tracker`` is the production code path that
    :meth:`ContextBuilder._extract_tool_history` consumes. Before the
    fix this chain was broken at the first step — DynaBot called the
    tool directly, so ``get_execution_history()`` always returned an
    empty list on real turns.
    """

    @pytest.mark.asyncio
    async def test_execute_tools_populates_registry_tracker(
        self, bot_with_tracking_registry
    ):
        """Reproducing pin — fails against pre-fix HEAD.

        Pre-fix: ``DynaBot._execute_tools`` calls ``tool.execute(...)``
        directly. The tool runs, the result lands on
        ``turn.tool_executions``, but ``registry.get_execution_history``
        still returns ``[]`` because nothing went through
        ``registry.execute_tool``'s recording path.

        Post-fix: dispatch goes through ``registry.execute_tool``; the
        tracker records the call and a consumer (e.g.
        ``ContextBuilder``) can surface it.
        """
        bot, manager, _tool = bot_with_tracking_registry

        context = BotContext(
            conversation_id=manager.conversation_id or "conv-tracker-test",
            client_id="test-client",
        )
        turn = TurnState(
            mode=TurnMode.CHAT,
            message="search",
            context=context,
            manager=manager,
        )

        await bot._execute_tools(
            turn,
            [_SimpleToolCall(name="search", parameters={"q": "foo"})],
        )

        # Pre-fix invariant: turn.tool_executions captures the call —
        # this never broke and still passes today.
        assert len(turn.tool_executions) == 1
        assert turn.tool_executions[0].tool_name == "search"
        assert turn.tool_executions[0].result == "ok"

        # The pin: the registry's tracker is now populated, so
        # production consumers of get_execution_history see the call.
        history = bot.tool_registry.get_execution_history()
        assert len(history) == 1
        assert history[0].tool_name == "search"
        assert history[0].success is True
        assert history[0].parameters == {"q": "foo"}

    @pytest.mark.asyncio
    async def test_execute_tools_records_failures_on_tracker(
        self, bot_with_tracking_registry
    ):
        """A tool that raises must still record onto the tracker as a
        failed execution — matches the registry's documented
        record-on-exception semantic.
        """
        bot, manager, _tool = bot_with_tracking_registry

        class _BoomTool(Tool):
            def __init__(self) -> None:
                super().__init__(name="boom", description="raises")

            @property
            def schema(self) -> dict[str, Any]:
                return {"type": "object", "properties": {}}

            async def execute(self, **kwargs: Any) -> Any:
                raise RuntimeError("kaboom")

        bot.tool_registry.register_tool(_BoomTool())

        context = BotContext(
            conversation_id=manager.conversation_id or "conv-tracker-test",
            client_id="test-client",
        )
        turn = TurnState(
            mode=TurnMode.CHAT,
            message="boom",
            context=context,
            manager=manager,
        )

        await bot._execute_tools(
            turn,
            [_SimpleToolCall(name="boom", parameters={})],
        )

        # DynaBot's error handling still appends a ToolExecution carrying
        # the error string.
        assert len(turn.tool_executions) == 1
        assert turn.tool_executions[0].error is not None

        # And the tracker records the failure.
        history = bot.tool_registry.get_execution_history(failed_only=True)
        assert len(history) == 1
        assert history[0].tool_name == "boom"
        assert history[0].success is False
        assert "kaboom" in (history[0].error or "")
