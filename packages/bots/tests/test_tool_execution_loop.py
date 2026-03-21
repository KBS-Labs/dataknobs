"""Tests for DynaBot-level tool execution loop (Gap 13).

Verifies that DynaBot executes tool_calls returned by strategies that
don't handle tools internally (e.g. SimpleReasoning), for both
``chat()`` and ``stream_chat()`` paths.
"""

from __future__ import annotations

from typing import Any

import pytest

from dataknobs_bots.bot.base import DynaBot
from dataknobs_bots.bot.context import BotContext
from dataknobs_bots.bot.turn import ToolExecution, TurnState
from dataknobs_bots.middleware.base import Middleware
from dataknobs_bots.testing import BotTestHarness
from dataknobs_llm.testing import (
    multi_tool_response,
    text_response,
    tool_call_response,
)
from dataknobs_llm.tools.base import Tool


# ---------------------------------------------------------------------------
# Test tools and middleware
# ---------------------------------------------------------------------------


class EchoTool(Tool):
    """Simple tool that returns its input for testing."""

    def __init__(self) -> None:
        super().__init__(name="echo", description="Echoes the input back")
        self.calls: list[dict[str, Any]] = []

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to echo"},
            },
            "required": ["text"],
        }

    async def execute(self, **kwargs: Any) -> Any:
        ctx = kwargs.pop("_context", None)
        self.calls.append({"kwargs": kwargs, "context": ctx})
        return {"echoed": kwargs.get("text", "")}


class FailingTool(Tool):
    """Tool that raises an error."""

    def __init__(self) -> None:
        super().__init__(name="fail", description="Always fails")

    @property
    def schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs: Any) -> Any:
        kwargs.pop("_context", None)
        msg = "Tool failure"
        raise RuntimeError(msg)


class ToolExecutionTracker(Middleware):
    """Middleware that records on_tool_executed calls."""

    def __init__(self) -> None:
        self.executions: list[ToolExecution] = []

    async def on_tool_executed(
        self, execution: ToolExecution, context: BotContext
    ) -> None:
        self.executions.append(execution)


class TurnTracker(Middleware):
    """Middleware that records after_turn calls."""

    def __init__(self) -> None:
        self.turns: list[TurnState] = []

    async def after_turn(self, turn: TurnState) -> None:
        self.turns.append(turn)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Non-wizard bot config — SimpleReasoning strategy with tools.
_SIMPLE_BOT_CONFIG: dict[str, Any] = {
    "llm": {"provider": "echo", "model": "test"},
    "conversation_storage": {"backend": "memory"},
    "reasoning": {"strategy": "simple"},
}

# No-strategy bot config — direct LLM calls.
_NO_STRATEGY_BOT_CONFIG: dict[str, Any] = {
    "llm": {"provider": "echo", "model": "test"},
    "conversation_storage": {"backend": "memory"},
}


# ---------------------------------------------------------------------------
# chat() tool execution tests
# ---------------------------------------------------------------------------


class TestChatToolExecution:
    """DynaBot.chat() executes tool_calls from non-ReAct strategies."""

    @pytest.mark.asyncio
    async def test_single_tool_call_executed(self) -> None:
        """Tool call is executed and follow-up response returned."""
        echo_tool = EchoTool()
        async with await BotTestHarness.create(
            bot_config=_SIMPLE_BOT_CONFIG,
            main_responses=[
                tool_call_response("echo", {"text": "hello"}),
                text_response("The echo said: hello"),
            ],
            tools=[echo_tool],
        ) as harness:
            result = await harness.chat("echo hello")

        assert result.response == "The echo said: hello"
        assert len(echo_tool.calls) == 1
        assert echo_tool.calls[0]["kwargs"] == {"text": "hello"}

    @pytest.mark.asyncio
    async def test_multi_round_tool_calls(self) -> None:
        """Multiple rounds of tool calls are executed sequentially."""
        echo_tool = EchoTool()
        async with await BotTestHarness.create(
            bot_config=_SIMPLE_BOT_CONFIG,
            main_responses=[
                tool_call_response("echo", {"text": "first"}),
                tool_call_response("echo", {"text": "second"}),
                text_response("Done with both"),
            ],
            tools=[echo_tool],
        ) as harness:
            result = await harness.chat("do two things")

        assert result.response == "Done with both"
        assert len(echo_tool.calls) == 2
        assert echo_tool.calls[0]["kwargs"] == {"text": "first"}
        assert echo_tool.calls[1]["kwargs"] == {"text": "second"}

    @pytest.mark.asyncio
    async def test_max_iterations_prevents_infinite_loop(self) -> None:
        """Tool loop stops after max_tool_iterations."""
        echo_tool = EchoTool()
        responses = [
            tool_call_response("echo", {"text": f"iter{i}"})
            for i in range(DynaBot._DEFAULT_MAX_TOOL_ITERATIONS + 2)
        ]
        async with await BotTestHarness.create(
            bot_config=_SIMPLE_BOT_CONFIG,
            main_responses=responses,
            tools=[echo_tool],
        ) as harness:
            await harness.chat("loop forever")
            max_iters = harness.bot._max_tool_iterations

        assert len(echo_tool.calls) == max_iters

    @pytest.mark.asyncio
    async def test_no_tools_registered_skips_loop(self) -> None:
        """When no tools are registered, tool_calls in response are ignored."""
        async with await BotTestHarness.create(
            bot_config=_SIMPLE_BOT_CONFIG,
            main_responses=[
                tool_call_response("echo", {"text": "hello"}),
            ],
        ) as harness:
            result = await harness.chat("hi")

        # tool_call_response has content="" by default
        assert result.response == ""

    @pytest.mark.asyncio
    async def test_tool_not_found_records_error(self) -> None:
        """Unknown tool name records error and continues to follow-up."""
        tracker = ToolExecutionTracker()
        async with await BotTestHarness.create(
            bot_config=_SIMPLE_BOT_CONFIG,
            main_responses=[
                tool_call_response("nonexistent_tool", {"x": 1}),
                text_response("Recovered from missing tool"),
            ],
            tools=[EchoTool()],  # Only "echo" is registered
            middleware=[tracker],
        ) as harness:
            result = await harness.chat("use missing tool")

        assert result.response == "Recovered from missing tool"
        assert len(tracker.executions) == 1
        assert tracker.executions[0].tool_name == "nonexistent_tool"
        assert tracker.executions[0].error == "Tool not found"

    @pytest.mark.asyncio
    async def test_tool_execution_error_recorded(self) -> None:
        """Tool that raises records error and adds error observation."""
        tracker = ToolExecutionTracker()
        async with await BotTestHarness.create(
            bot_config=_SIMPLE_BOT_CONFIG,
            main_responses=[
                tool_call_response("fail", {}),
                text_response("Recovered from tool error"),
            ],
            tools=[FailingTool()],
            middleware=[tracker],
        ) as harness:
            result = await harness.chat("use failing tool")

        assert result.response == "Recovered from tool error"
        assert len(tracker.executions) == 1
        assert tracker.executions[0].tool_name == "fail"
        assert "Tool failure" in (tracker.executions[0].error or "")

    @pytest.mark.asyncio
    async def test_on_tool_executed_middleware_fires(self) -> None:
        """on_tool_executed middleware hook fires for DynaBot-executed tools."""
        echo_tool = EchoTool()
        tracker = ToolExecutionTracker()
        async with await BotTestHarness.create(
            bot_config=_SIMPLE_BOT_CONFIG,
            main_responses=[
                tool_call_response("echo", {"text": "hi"}),
                text_response("done"),
            ],
            tools=[echo_tool],
            middleware=[tracker],
        ) as harness:
            await harness.chat("hi")

        assert len(tracker.executions) == 1
        assert tracker.executions[0].tool_name == "echo"
        assert tracker.executions[0].result == {"echoed": "hi"}
        assert tracker.executions[0].duration_ms is not None
        assert tracker.executions[0].duration_ms >= 0

    @pytest.mark.asyncio
    async def test_tool_execution_context_has_turn_data(self) -> None:
        """Tool receives turn_data from plugin_data bridge."""
        echo_tool = EchoTool()

        class PluginWriter(Middleware):
            async def on_turn_start(self, turn: TurnState) -> str | None:
                turn.plugin_data["test_key"] = "test_value"
                return None

        async with await BotTestHarness.create(
            bot_config=_SIMPLE_BOT_CONFIG,
            main_responses=[
                tool_call_response("echo", {"text": "hi"}),
                text_response("done"),
            ],
            tools=[echo_tool],
            middleware=[PluginWriter()],
        ) as harness:
            await harness.chat("hi")

        assert len(echo_tool.calls) == 1
        tool_ctx = echo_tool.calls[0]["context"]
        assert tool_ctx is not None
        assert tool_ctx.extra.get("turn_data", {}).get("test_key") == "test_value"

    @pytest.mark.asyncio
    async def test_tool_executions_in_after_turn(self) -> None:
        """after_turn receives all tool executions from the turn."""
        echo_tool = EchoTool()
        turn_tracker = TurnTracker()
        async with await BotTestHarness.create(
            bot_config=_SIMPLE_BOT_CONFIG,
            main_responses=[
                tool_call_response("echo", {"text": "a"}),
                tool_call_response("echo", {"text": "b"}),
                text_response("done"),
            ],
            tools=[echo_tool],
            middleware=[turn_tracker],
        ) as harness:
            await harness.chat("hi")

        assert len(turn_tracker.turns) == 1
        turn = turn_tracker.turns[0]
        assert len(turn.tool_executions) == 2
        assert turn.tool_executions[0].tool_name == "echo"
        assert turn.tool_executions[1].tool_name == "echo"

    @pytest.mark.asyncio
    async def test_no_tool_calls_in_response_skips_loop(self) -> None:
        """Normal response without tool_calls passes through unchanged."""
        echo_tool = EchoTool()
        async with await BotTestHarness.create(
            bot_config=_SIMPLE_BOT_CONFIG,
            main_responses=[text_response("just text")],
            tools=[echo_tool],
        ) as harness:
            result = await harness.chat("hi")

        assert result.response == "just text"
        assert len(echo_tool.calls) == 0


# ---------------------------------------------------------------------------
# stream_chat() tool execution tests
# ---------------------------------------------------------------------------


class TestStreamChatToolExecution:
    """DynaBot.stream_chat() executes tool_calls from streaming responses."""

    @pytest.mark.asyncio
    async def test_stream_tool_call_executed(self) -> None:
        """Tool call in streaming response triggers execution and re-stream."""
        echo_tool = EchoTool()
        async with await BotTestHarness.create(
            bot_config=_SIMPLE_BOT_CONFIG,
            main_responses=[
                tool_call_response("echo", {"text": "hello"}),
                text_response("The echo said: hello"),
            ],
            tools=[echo_tool],
        ) as harness:
            chunks = []
            async for chunk in harness.bot.stream_chat(
                "echo hello", harness.context
            ):
                chunks.append(chunk)

        full_text = "".join(c.delta for c in chunks)
        assert "The echo said: hello" in full_text
        assert len(echo_tool.calls) == 1

    @pytest.mark.asyncio
    async def test_stream_is_final_suppressed_on_intermediate(self) -> None:
        """Intermediate rounds suppress is_final; final chunk is from re-gen."""
        echo_tool = EchoTool()
        async with await BotTestHarness.create(
            bot_config=_SIMPLE_BOT_CONFIG,
            main_responses=[
                tool_call_response("echo", {"text": "hi"}),
                text_response("done"),
            ],
            tools=[echo_tool],
        ) as harness:
            chunks = []
            async for chunk in harness.bot.stream_chat("hi", harness.context):
                chunks.append(chunk)

        is_final_chunks = [c for c in chunks if c.is_final]
        assert len(is_final_chunks) == 1
        # The full streamed text must include the post-tool re-generation
        full_text = "".join(c.delta for c in chunks)
        assert "done" in full_text

    @pytest.mark.asyncio
    async def test_stream_multi_round(self) -> None:
        """Multiple streaming tool rounds work correctly."""
        echo_tool = EchoTool()
        async with await BotTestHarness.create(
            bot_config=_SIMPLE_BOT_CONFIG,
            main_responses=[
                tool_call_response("echo", {"text": "first"}),
                tool_call_response("echo", {"text": "second"}),
                text_response("All done"),
            ],
            tools=[echo_tool],
        ) as harness:
            chunks = []
            async for chunk in harness.bot.stream_chat(
                "do stuff", harness.context
            ):
                chunks.append(chunk)

        full_text = "".join(c.delta for c in chunks)
        assert "All done" in full_text
        assert len(echo_tool.calls) == 2

    @pytest.mark.asyncio
    async def test_stream_no_tools_skips_loop(self) -> None:
        """Streaming without tool_calls passes through normally."""
        echo_tool = EchoTool()
        async with await BotTestHarness.create(
            bot_config=_SIMPLE_BOT_CONFIG,
            main_responses=[text_response("just text")],
            tools=[echo_tool],
        ) as harness:
            chunks = []
            async for chunk in harness.bot.stream_chat("hi", harness.context):
                chunks.append(chunk)

        full_text = "".join(c.delta for c in chunks)
        assert full_text == "just text"
        assert len(echo_tool.calls) == 0

    @pytest.mark.asyncio
    async def test_stream_on_tool_executed_fires(self) -> None:
        """on_tool_executed fires for tools executed in streaming path."""
        echo_tool = EchoTool()
        tracker = ToolExecutionTracker()
        async with await BotTestHarness.create(
            bot_config=_SIMPLE_BOT_CONFIG,
            main_responses=[
                tool_call_response("echo", {"text": "hi"}),
                text_response("done"),
            ],
            tools=[echo_tool],
            middleware=[tracker],
        ) as harness:
            async for _ in harness.bot.stream_chat("hi", harness.context):
                pass

        assert len(tracker.executions) == 1
        assert tracker.executions[0].tool_name == "echo"

    @pytest.mark.asyncio
    async def test_stream_max_iterations_prevents_infinite_loop(self) -> None:
        """Streaming tool loop stops after max_tool_iterations."""
        echo_tool = EchoTool()
        responses = [
            tool_call_response("echo", {"text": f"iter{i}"})
            for i in range(DynaBot._DEFAULT_MAX_TOOL_ITERATIONS + 2)
        ]
        async with await BotTestHarness.create(
            bot_config=_SIMPLE_BOT_CONFIG,
            main_responses=responses,
            tools=[echo_tool],
        ) as harness:
            chunks = []
            async for chunk in harness.bot.stream_chat(
                "loop forever", harness.context
            ):
                chunks.append(chunk)
            max_iters = harness.bot._max_tool_iterations

        assert len(echo_tool.calls) == max_iters

    @pytest.mark.asyncio
    async def test_stream_no_tools_registered_ignores_tool_calls(self) -> None:
        """Streaming with tool_calls but no registry passes through as-is."""
        async with await BotTestHarness.create(
            bot_config=_SIMPLE_BOT_CONFIG,
            main_responses=[
                tool_call_response("echo", {"text": "hello"}),
            ],
        ) as harness:
            chunks = []
            async for chunk in harness.bot.stream_chat("hi", harness.context):
                chunks.append(chunk)

        # Response passes through without tool execution
        is_final_chunks = [c for c in chunks if c.is_final]
        assert len(is_final_chunks) >= 1


# ---------------------------------------------------------------------------
# Multi-tool call tests (both paths)
# ---------------------------------------------------------------------------


class TestMultiToolExecution:
    """Multiple tool calls in a single LLM response."""

    @pytest.mark.asyncio
    async def test_chat_multi_tool_single_response(self) -> None:
        """Multiple tools called in one response are all executed."""
        echo_tool = EchoTool()
        failing_tool = FailingTool()
        tracker = ToolExecutionTracker()
        async with await BotTestHarness.create(
            bot_config=_SIMPLE_BOT_CONFIG,
            main_responses=[
                multi_tool_response([
                    ("echo", {"text": "first"}),
                    ("echo", {"text": "second"}),
                ]),
                text_response("both done"),
            ],
            tools=[echo_tool, failing_tool],
            middleware=[tracker],
        ) as harness:
            result = await harness.chat("do two things at once")

        assert result.response == "both done"
        assert len(echo_tool.calls) == 2
        assert echo_tool.calls[0]["kwargs"] == {"text": "first"}
        assert echo_tool.calls[1]["kwargs"] == {"text": "second"}
        assert len(tracker.executions) == 2

    @pytest.mark.asyncio
    async def test_stream_multi_tool_single_response(self) -> None:
        """Multiple tools in one streaming response are all executed."""
        echo_tool = EchoTool()
        async with await BotTestHarness.create(
            bot_config=_SIMPLE_BOT_CONFIG,
            main_responses=[
                multi_tool_response([
                    ("echo", {"text": "a"}),
                    ("echo", {"text": "b"}),
                ]),
                text_response("multi done"),
            ],
            tools=[echo_tool],
        ) as harness:
            chunks = []
            async for chunk in harness.bot.stream_chat(
                "do two at once", harness.context
            ):
                chunks.append(chunk)

        full_text = "".join(c.delta for c in chunks)
        assert "multi done" in full_text
        assert len(echo_tool.calls) == 2


# ---------------------------------------------------------------------------
# No-strategy path tests
# ---------------------------------------------------------------------------


class TestNoStrategyToolExecution:
    """Tool execution works when no reasoning strategy is configured."""

    @pytest.mark.asyncio
    async def test_chat_no_strategy(self) -> None:
        """chat() without strategy executes tools."""
        echo_tool = EchoTool()
        async with await BotTestHarness.create(
            bot_config=_NO_STRATEGY_BOT_CONFIG,
            main_responses=[
                tool_call_response("echo", {"text": "hi"}),
                text_response("done"),
            ],
            tools=[echo_tool],
        ) as harness:
            result = await harness.chat("hi")

        assert result.response == "done"
        assert len(echo_tool.calls) == 1

    @pytest.mark.asyncio
    async def test_stream_no_strategy(self) -> None:
        """stream_chat() without strategy executes tools."""
        echo_tool = EchoTool()
        async with await BotTestHarness.create(
            bot_config=_NO_STRATEGY_BOT_CONFIG,
            main_responses=[
                tool_call_response("echo", {"text": "hi"}),
                text_response("done"),
            ],
            tools=[echo_tool],
        ) as harness:
            chunks = []
            async for chunk in harness.bot.stream_chat("hi", harness.context):
                chunks.append(chunk)

        full_text = "".join(c.delta for c in chunks)
        assert "done" in full_text
        assert len(echo_tool.calls) == 1
