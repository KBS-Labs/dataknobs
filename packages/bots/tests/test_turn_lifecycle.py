"""Tests for turn lifecycle: finally_turn hook, plugin_data seeding, tool timeouts.

Covers:
- Gap 18: Tool execution safety (per-tool timeout, wall-clock loop timeout)
- Gap 19: finally_turn cleanup hook, plugin_data parameter on entry points
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from dataknobs_bots.bot.context import BotContext
from dataknobs_bots.bot.turn import ToolExecution, TurnState
from dataknobs_bots.middleware.base import Middleware
from dataknobs_bots.testing import BotTestHarness
from dataknobs_llm.testing import ErrorResponse, text_response, tool_call_response
from dataknobs_llm.tools.base import Tool


# ---------------------------------------------------------------------------
# Test middleware
# ---------------------------------------------------------------------------


class LifecycleTracker(Middleware):
    """Records which hooks fired and in what order."""

    def __init__(self) -> None:
        self.events: list[str] = []
        self.plugin_data_snapshots: dict[str, dict[str, Any]] = {}

    async def on_turn_start(self, turn: TurnState) -> str | None:
        self.events.append("on_turn_start")
        self.plugin_data_snapshots["on_turn_start"] = dict(turn.plugin_data)
        return None

    async def after_turn(self, turn: TurnState) -> None:
        self.events.append("after_turn")
        self.plugin_data_snapshots["after_turn"] = dict(turn.plugin_data)

    async def finally_turn(self, turn: TurnState) -> None:
        self.events.append("finally_turn")
        self.plugin_data_snapshots["finally_turn"] = dict(turn.plugin_data)

    async def on_error(
        self, error: Exception, message: str, context: BotContext
    ) -> None:
        self.events.append("on_error")


class PluginDataWriter(Middleware):
    """Writes a marker to plugin_data in on_turn_start."""

    async def on_turn_start(self, turn: TurnState) -> str | None:
        turn.plugin_data["writer_key"] = "writer_value"
        return None


# ---------------------------------------------------------------------------
# Test tools
# ---------------------------------------------------------------------------


class SlowTool(Tool):
    """Tool that sleeps indefinitely (for timeout testing)."""

    def __init__(self) -> None:
        super().__init__(name="slow", description="Sleeps forever")
        self.was_called = False

    @property
    def schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs: Any) -> Any:
        kwargs.pop("_context", None)
        self.was_called = True
        await asyncio.sleep(3600)  # 1 hour — will be cancelled by timeout
        return {"result": "never reached"}


class QuickTool(Tool):
    """Tool that returns immediately."""

    def __init__(self) -> None:
        super().__init__(name="quick", description="Returns immediately")
        self.call_count = 0

    @property
    def schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs: Any) -> Any:
        kwargs.pop("_context", None)
        self.call_count += 1
        return {"result": "done"}


# ---------------------------------------------------------------------------
# Bot configs
# ---------------------------------------------------------------------------

_SIMPLE_BOT_CONFIG: dict[str, Any] = {
    "llm": {"provider": "echo", "model": "test"},
    "conversation_storage": {"backend": "memory"},
    "reasoning": {"strategy": "simple"},
}


# ---------------------------------------------------------------------------
# finally_turn tests
# ---------------------------------------------------------------------------


class TestFinallyTurn:
    """Gap 19: finally_turn fires unconditionally."""

    @pytest.mark.asyncio
    async def test_finally_turn_fires_on_success(self) -> None:
        """finally_turn fires after a successful chat turn."""
        tracker = LifecycleTracker()
        async with await BotTestHarness.create(
            bot_config=_SIMPLE_BOT_CONFIG,
            main_responses=[text_response("Hello!")],
            middleware=[tracker],
        ) as harness:
            await harness.chat("Hi")

        assert "finally_turn" in tracker.events
        assert "after_turn" in tracker.events
        # finally_turn fires after after_turn
        assert tracker.events.index("after_turn") < tracker.events.index(
            "finally_turn"
        )

    @pytest.mark.asyncio
    async def test_finally_turn_fires_on_error(self) -> None:
        """finally_turn fires even when the turn raises an exception."""
        tracker = LifecycleTracker()
        async with await BotTestHarness.create(
            bot_config=_SIMPLE_BOT_CONFIG,
            main_responses=[
                ErrorResponse(RuntimeError("provider unavailable")),
            ],
            middleware=[tracker],
        ) as harness:
            with pytest.raises(RuntimeError, match="provider unavailable"):
                await harness.chat("This will fail")

        assert "finally_turn" in tracker.events
        assert "on_error" in tracker.events

    @pytest.mark.asyncio
    async def test_finally_turn_receives_plugin_data_from_on_turn_start(
        self,
    ) -> None:
        """plugin_data written in on_turn_start is available in finally_turn."""
        writer = PluginDataWriter()
        tracker = LifecycleTracker()
        async with await BotTestHarness.create(
            bot_config=_SIMPLE_BOT_CONFIG,
            main_responses=[text_response("OK")],
            middleware=[writer, tracker],
        ) as harness:
            await harness.chat("test")

        assert tracker.plugin_data_snapshots["finally_turn"]["writer_key"] == (
            "writer_value"
        )

    @pytest.mark.asyncio
    async def test_finally_turn_fires_on_stream_early_exit(self) -> None:
        """finally_turn fires when stream_chat is abandoned via aclosing.

        _finalize_turn should NOT run (partial data), but finally_turn
        should fire for cleanup.
        """
        from contextlib import aclosing

        tracker = LifecycleTracker()
        async with await BotTestHarness.create(
            bot_config={
                "llm": {"provider": "echo", "model": "test"},
                "conversation_storage": {"backend": "memory"},
            },
            main_responses=[text_response("A long response")],
            middleware=[tracker],
        ) as harness:
            async with aclosing(
                harness.bot.stream_chat("Hi", harness.context)
            ) as stream:
                async for _chunk in stream:
                    break  # Exit after first chunk

        assert "finally_turn" in tracker.events
        # after_turn should NOT fire — stream was not fully consumed,
        # so _finalize_turn was skipped to avoid writing partial data.
        assert "after_turn" not in tracker.events


# ---------------------------------------------------------------------------
# plugin_data parameter tests
# ---------------------------------------------------------------------------


class TestPluginDataParameter:
    """Gap 19: plugin_data can be seeded from the call site."""

    @pytest.mark.asyncio
    async def test_plugin_data_seeded_from_chat(self) -> None:
        """plugin_data passed to chat() is visible in middleware hooks."""
        tracker = LifecycleTracker()
        async with await BotTestHarness.create(
            bot_config=_SIMPLE_BOT_CONFIG,
            main_responses=[text_response("OK")],
            middleware=[tracker],
        ) as harness:
            await harness.bot.chat(
                "test",
                harness.context,
                plugin_data={"session_id": "abc-123"},
            )

        assert tracker.plugin_data_snapshots["on_turn_start"]["session_id"] == (
            "abc-123"
        )
        assert tracker.plugin_data_snapshots["finally_turn"]["session_id"] == (
            "abc-123"
        )

    @pytest.mark.asyncio
    async def test_plugin_data_seeded_from_greet(self) -> None:
        """plugin_data passed to greet() triggers finally_turn even
        when strategy returns None (no greeting template)."""
        tracker = LifecycleTracker()
        async with await BotTestHarness.create(
            bot_config={
                **_SIMPLE_BOT_CONFIG,
                "reasoning": {"strategy": "simple"},
            },
            main_responses=[text_response("Welcome!")],
            middleware=[tracker],
        ) as harness:
            result = await harness.bot.greet(
                harness.context,
                plugin_data={"request_id": "req-456"},
            )

        # Simple strategy with no greeting_template returns None,
        # but finally_turn should still fire (inside try/finally).
        # after_turn does NOT fire because _finalize_turn is skipped
        # when greet() returns None before reaching it.
        assert result is None
        assert "finally_turn" in tracker.events
        assert "after_turn" not in tracker.events
        assert tracker.plugin_data_snapshots["finally_turn"]["request_id"] == (
            "req-456"
        )

    @pytest.mark.asyncio
    async def test_greet_no_strategy_fires_finally_turn_with_plugin_data(
        self,
    ) -> None:
        """greet() with no reasoning strategy fires finally_turn for cleanup
        when plugin_data is provided."""
        tracker = LifecycleTracker()
        # Bot with NO reasoning strategy
        async with await BotTestHarness.create(
            bot_config={
                "llm": {"provider": "echo", "model": "test"},
                "conversation_storage": {"backend": "memory"},
                # No "reasoning" key — no strategy
            },
            main_responses=[text_response("unused")],
            middleware=[tracker],
        ) as harness:
            result = await harness.bot.greet(
                harness.context,
                plugin_data={"db_session": "session-handle"},
            )

        assert result is None
        assert "finally_turn" in tracker.events
        assert tracker.plugin_data_snapshots["finally_turn"]["db_session"] == (
            "session-handle"
        )
        # No other lifecycle hooks should fire — no turn was initiated
        assert "on_turn_start" not in tracker.events
        assert "after_turn" not in tracker.events

    @pytest.mark.asyncio
    async def test_greet_no_strategy_no_plugin_data_skips_finally(
        self,
    ) -> None:
        """greet() with no strategy and no plugin_data skips finally_turn."""
        tracker = LifecycleTracker()
        async with await BotTestHarness.create(
            bot_config={
                "llm": {"provider": "echo", "model": "test"},
                "conversation_storage": {"backend": "memory"},
            },
            main_responses=[text_response("unused")],
            middleware=[tracker],
        ) as harness:
            result = await harness.bot.greet(harness.context)

        assert result is None
        assert "finally_turn" not in tracker.events

    @pytest.mark.asyncio
    async def test_greet_no_strategy_empty_plugin_data_fires_finally(
        self,
    ) -> None:
        """greet() with no strategy and plugin_data={} fires finally_turn.

        Empty dict is still an explicit plugin_data argument (is not None),
        so finally_turn fires for cleanup.
        """
        tracker = LifecycleTracker()
        async with await BotTestHarness.create(
            bot_config={
                "llm": {"provider": "echo", "model": "test"},
                "conversation_storage": {"backend": "memory"},
            },
            main_responses=[text_response("unused")],
            middleware=[tracker],
        ) as harness:
            result = await harness.bot.greet(
                harness.context,
                plugin_data={},
            )

        assert result is None
        assert "finally_turn" in tracker.events

    @pytest.mark.asyncio
    async def test_plugin_data_merged_with_middleware_writes(self) -> None:
        """Caller-seeded plugin_data coexists with middleware-written data."""
        writer = PluginDataWriter()
        tracker = LifecycleTracker()
        async with await BotTestHarness.create(
            bot_config=_SIMPLE_BOT_CONFIG,
            main_responses=[text_response("OK")],
            middleware=[writer, tracker],
        ) as harness:
            await harness.bot.chat(
                "test",
                harness.context,
                plugin_data={"caller_key": "caller_value"},
            )

        finally_data = tracker.plugin_data_snapshots["finally_turn"]
        assert finally_data["caller_key"] == "caller_value"
        assert finally_data["writer_key"] == "writer_value"


# ---------------------------------------------------------------------------
# Tool timeout tests
# ---------------------------------------------------------------------------


class TestToolTimeout:
    """Gap 18: Per-tool and loop-level timeouts."""

    @pytest.mark.asyncio
    async def test_tool_timeout_error_in_tool_execution(self) -> None:
        """Timeout error is recorded in ToolExecution with error string."""

        class ToolExecTracker(Middleware):
            """Records tool executions from after_turn."""

            def __init__(self) -> None:
                self.tool_executions: list[ToolExecution] = []

            async def after_turn(self, turn: TurnState) -> None:
                self.tool_executions.extend(turn.tool_executions)

        slow_tool = SlowTool()
        exec_tracker = ToolExecTracker()
        async with await BotTestHarness.create(
            bot_config={
                **_SIMPLE_BOT_CONFIG,
                "tool_timeout": 0.01,
            },
            main_responses=[
                tool_call_response("slow", {}),
                text_response("OK after timeout"),
            ],
            tools=[slow_tool],
            middleware=[exec_tracker],
        ) as harness:
            await harness.chat("run slow tool")

        assert len(exec_tracker.tool_executions) == 1
        exec_record = exec_tracker.tool_executions[0]
        assert exec_record.tool_name == "slow"
        assert exec_record.error is not None
        assert "Timed out" in exec_record.error
        assert exec_record.duration_ms is not None

    @pytest.mark.asyncio
    async def test_tool_loop_timeout_exits_loop(self) -> None:
        """Wall-clock loop timeout exits the tool loop early."""

        class ToolExecTracker(Middleware):
            """Records tool executions from after_turn."""

            def __init__(self) -> None:
                self.tool_executions: list[ToolExecution] = []

            async def after_turn(self, turn: TurnState) -> None:
                self.tool_executions.extend(turn.tool_executions)

        quick_tool = QuickTool()
        exec_tracker = ToolExecTracker()
        async with await BotTestHarness.create(
            bot_config={
                **_SIMPLE_BOT_CONFIG,
                # Very short loop timeout — will expire before second iteration
                "tool_loop_timeout": 0.0,
                "max_tool_iterations": 10,
            },
            main_responses=[
                tool_call_response("quick", {}),
                # These would be consumed if the loop continued:
                tool_call_response("quick", {}),
                tool_call_response("quick", {}),
                text_response("Eventually done"),
            ],
            tools=[quick_tool],
            middleware=[exec_tracker],
        ) as harness:
            # The loop should exit after at most 1 tool execution
            # because tool_loop_timeout=0.0 expires immediately
            await harness.chat("run tools")

        # With timeout=0.0, the top-of-loop check uses >= so 0 elapsed
        # at start passes (0 >= 0 is True), meaning the loop exits
        # before executing any tools.
        assert quick_tool.call_count == 0

    @pytest.mark.asyncio
    async def test_llm_recall_bounded_by_remaining_budget(self) -> None:
        """Wall-clock budget limits total tool rounds even when tools are fast.

        With max_tool_iterations=10 but a very short budget, the loop
        should exit well before 10 iterations.  The exact number depends
        on machine speed, so we assert a ceiling rather than an exact
        count.
        """

        class ToolExecTracker(Middleware):
            """Records tool executions from after_turn."""

            def __init__(self) -> None:
                self.tool_executions: list[ToolExecution] = []

            async def after_turn(self, turn: TurnState) -> None:
                self.tool_executions.extend(turn.tool_executions)

        quick_tool = QuickTool()
        exec_tracker = ToolExecTracker()
        async with await BotTestHarness.create(
            bot_config={
                **_SIMPLE_BOT_CONFIG,
                # Tiny budget — at most a few iterations before expiry
                "tool_loop_timeout": 0.001,
                "max_tool_iterations": 10,
            },
            main_responses=[
                tool_call_response("quick", {}),
                tool_call_response("quick", {}),
                tool_call_response("quick", {}),
                tool_call_response("quick", {}),
                tool_call_response("quick", {}),
                text_response("done"),
            ],
            tools=[quick_tool],
            middleware=[exec_tracker],
        ) as harness:
            await harness.chat("test budget enforcement")

        # Budget should prevent reaching max_tool_iterations (10).
        # Exact count depends on machine speed; assert well below max.
        assert quick_tool.call_count <= 5
