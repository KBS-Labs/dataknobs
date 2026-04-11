"""Tests for ReAct phased reasoning protocol (item 81).

Verifies:
- ReActReasoning implements PhasedReasoningProtocol
- ReActTurnHandle construction and field defaults
- begin_turn: setup, extra_context, no-tools fast path
- process_input: final answer, tool_calls with iterate, duplicate
  detection, max iterations, ToolsNotSupportedError
- finalize_turn: stored response vs. synthesis call
- End-to-end via BotTestHarness: DynaBot routes ReAct through phased
  flow, middleware fires per-tool
- Streaming iterative loop exercised via StreamingPhasedProtocol
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from dataknobs_bots.reasoning.base import (
    PhasedReasoningProtocol,
    ProcessResult,
    TurnHandle,
)
from dataknobs_bots.reasoning.react import ReActReasoning, ReActTurnHandle
from dataknobs_bots.testing import BotTestHarness
from dataknobs_data.backends.memory import AsyncMemoryDatabase
from dataknobs_llm import LLMConfig
from dataknobs_llm.conversations import ConversationManager
from dataknobs_llm.conversations.storage import DataknobsConversationStorage
from dataknobs_llm.llm.providers.echo import EchoProvider
from dataknobs_llm.prompts import ConfigPromptLibrary
from dataknobs_llm.prompts.builders import AsyncPromptBuilder
from dataknobs_llm.testing import text_response, tool_call_response
from dataknobs_llm.tools.base import Tool


# ---------------------------------------------------------------------------
# Test tool
# ---------------------------------------------------------------------------


class EchoTool(Tool):
    """Simple tool that returns its input for testing."""

    def __init__(self) -> None:
        super().__init__(name="echo_tool", description="Echoes input back")
        self.call_count = 0

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "message": {"type": "string"},
            },
        }

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        self.call_count += 1
        return {"echoed": kwargs.get("message", ""), "call": self.call_count}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_provider() -> EchoProvider:
    """Create an EchoProvider with minimal config."""
    config = LLMConfig(
        provider="echo",
        model="echo-test",
        options={"echo_prefix": ""},
    )
    return EchoProvider(config)


async def _make_manager(provider: EchoProvider) -> ConversationManager:
    """Create a ConversationManager for unit tests."""
    library = ConfigPromptLibrary({
        "system": {
            "assistant": {"template": "You are a test bot."},
        },
    })
    builder = AsyncPromptBuilder(library=library)
    storage = DataknobsConversationStorage(AsyncMemoryDatabase())
    mgr = await ConversationManager.create(
        llm=provider,
        prompt_builder=builder,
        storage=storage,
        system_prompt_name="assistant",
    )
    await mgr.add_message(role="user", content="test input")
    return mgr


# =========================================================================
# Protocol detection
# =========================================================================


class TestReActProtocolDetection:
    """Verify ReActReasoning satisfies PhasedReasoningProtocol."""

    def test_react_satisfies_protocol(self) -> None:
        """ReActReasoning satisfies PhasedReasoningProtocol isinstance check."""
        strategy = ReActReasoning()
        assert isinstance(strategy, PhasedReasoningProtocol)

    def test_react_turn_handle_extends_turn_handle(self) -> None:
        """ReActTurnHandle is a subclass of TurnHandle."""
        assert issubclass(ReActTurnHandle, TurnHandle)


# =========================================================================
# ReActTurnHandle construction
# =========================================================================


class TestReActTurnHandle:
    """Verify ReActTurnHandle field defaults."""

    def test_defaults(self) -> None:
        handle = ReActTurnHandle(manager=None, llm=None)
        assert handle.iteration == 0
        assert handle.max_iterations == 5
        assert handle.prev_tool_calls is None
        assert handle.trace is None
        assert handle.final_response is None
        assert handle.store_trace is False
        assert handle.verbose is False
        # Inherited from TurnHandle
        assert handle.early_response is None
        assert handle.tool_extra_context == {}

    def test_custom_values(self) -> None:
        handle = ReActTurnHandle(
            manager="mgr",
            llm="llm",
            tools=["t1"],
            max_iterations=10,
            trace=[],
            store_trace=True,
            verbose=True,
        )
        assert handle.max_iterations == 10
        assert handle.trace == []
        assert handle.store_trace is True


# =========================================================================
# begin_turn
# =========================================================================


class TestReActBeginTurn:
    """Verify begin_turn setup and early return paths."""

    @pytest.mark.asyncio
    async def test_no_tools_returns_early_response(self) -> None:
        """begin_turn with no tools sets early_response from LLM call."""
        provider = _make_provider()
        provider.set_responses([text_response("No tools response")])
        manager = await _make_manager(provider)
        strategy = ReActReasoning()

        handle = await strategy.begin_turn(manager, provider, tools=None)

        assert handle.early_response is not None
        assert handle.early_response.content == "No tools response"
        assert isinstance(handle, ReActTurnHandle)

    @pytest.mark.asyncio
    async def test_with_tools_no_early_response(self) -> None:
        """begin_turn with tools initializes handle without early_response."""
        provider = _make_provider()
        manager = await _make_manager(provider)
        tool = EchoTool()
        strategy = ReActReasoning(max_iterations=3, store_trace=True)

        handle = await strategy.begin_turn(
            manager, provider, tools=[tool], temperature=0.5
        )

        assert handle.early_response is None
        assert isinstance(handle, ReActTurnHandle)
        assert handle.max_iterations == 3
        assert handle.trace == []
        assert handle.iteration == 0
        assert handle.kwargs == {"temperature": 0.5}

    @pytest.mark.asyncio
    async def test_extra_context_populated(self) -> None:
        """begin_turn populates tool_extra_context from strategy fields."""
        provider = _make_provider()
        manager = await _make_manager(provider)
        tool = EchoTool()

        strategy = ReActReasoning(
            artifact_registry="mock_registry",
            review_executor="mock_executor",
            extra_context={"custom_key": "custom_value"},
        )

        handle = await strategy.begin_turn(manager, provider, tools=[tool])

        assert handle.tool_extra_context["artifact_registry"] == "mock_registry"
        assert handle.tool_extra_context["review_executor"] == "mock_executor"
        assert handle.tool_extra_context["custom_key"] == "custom_value"


# =========================================================================
# process_input
# =========================================================================


class TestReActProcessInput:
    """Verify process_input iteration logic."""

    @pytest.mark.asyncio
    async def test_final_answer_no_tool_calls(self) -> None:
        """LLM returns no tool_calls -> action='final_answer'."""
        provider = _make_provider()
        provider.set_responses([text_response("Final answer")])
        manager = await _make_manager(provider)
        tool = EchoTool()
        strategy = ReActReasoning()

        handle = await strategy.begin_turn(manager, provider, tools=[tool])
        result = await strategy.process_input(handle)

        assert result.action == "final_answer"
        assert result.needs_tool_execution is False
        assert result.iterate is False
        assert handle.final_response is not None
        assert handle.final_response.content == "Final answer"

    @pytest.mark.asyncio
    async def test_tool_calls_returns_iterate(self) -> None:
        """LLM returns tool_calls -> iterate=True, needs_tool_execution=True."""
        provider = _make_provider()
        provider.set_responses([
            tool_call_response("echo_tool", {"message": "hello"}),
        ])
        manager = await _make_manager(provider)
        tool = EchoTool()
        strategy = ReActReasoning()

        handle = await strategy.begin_turn(manager, provider, tools=[tool])
        result = await strategy.process_input(handle)

        assert result.action == "tool_calls"
        assert result.needs_tool_execution is True
        assert result.iterate is True
        assert len(result.pending_tool_calls) == 1
        assert result.pending_tool_calls[0].name == "echo_tool"
        assert handle.iteration == 1

    @pytest.mark.asyncio
    async def test_duplicate_detection(self) -> None:
        """Same tool calls twice in a row -> action='duplicate_break'."""
        provider = _make_provider()
        provider.set_responses([
            tool_call_response("echo_tool", {"message": "same"}),
            tool_call_response("echo_tool", {"message": "same"}),
        ])
        manager = await _make_manager(provider)
        tool = EchoTool()
        strategy = ReActReasoning()

        handle = await strategy.begin_turn(manager, provider, tools=[tool])

        # First call: returns tool_calls
        result1 = await strategy.process_input(handle)
        assert result1.action == "tool_calls"

        # Simulate DynaBot executing tools (add observation to history)
        await manager.add_message(
            content="Observation from echo_tool: echoed",
            role="tool",
            name="echo_tool",
        )

        # Second call with same tool calls: duplicate detection
        result2 = await strategy.process_input(handle)
        assert result2.action == "duplicate_break"
        assert result2.needs_tool_execution is False
        assert handle.final_response is None  # finalize_turn does synthesis

    @pytest.mark.asyncio
    async def test_max_iterations(self) -> None:
        """After max_iterations -> action='max_iterations'."""
        provider = _make_provider()
        manager = await _make_manager(provider)
        tool = EchoTool()
        strategy = ReActReasoning(max_iterations=2)

        handle = await strategy.begin_turn(manager, provider, tools=[tool])
        # Manually set iteration to max
        handle.iteration = 2

        result = await strategy.process_input(handle)

        assert result.action == "max_iterations"
        assert result.needs_tool_execution is False

    @pytest.mark.asyncio
    async def test_type_error_on_wrong_handle(self) -> None:
        """process_input raises TypeError for non-ReActTurnHandle."""
        strategy = ReActReasoning()
        wrong_handle = TurnHandle(manager=None, llm=None)

        with pytest.raises(TypeError, match="Expected ReActTurnHandle"):
            await strategy.process_input(wrong_handle)

    @pytest.mark.asyncio
    async def test_tools_not_supported_returns_early_response(self) -> None:
        """ToolsNotSupportedError returns early_response with user message."""
        from dataknobs_llm.exceptions import ToolsNotSupportedError

        provider = _make_provider()
        manager = await _make_manager(provider)

        # Make complete() raise ToolsNotSupportedError
        async def raise_tools_error(**kwargs: Any) -> Any:
            raise ToolsNotSupportedError(model="test-model")

        manager.complete = raise_tools_error  # type: ignore[assignment]

        tool = EchoTool()
        strategy = ReActReasoning()

        handle = await strategy.begin_turn(manager, provider, tools=[tool])
        result = await strategy.process_input(handle)

        assert result.early_response is not None
        assert "doesn't support tool calling" in result.early_response.content
        assert result.action == "tools_not_supported"


# =========================================================================
# finalize_turn
# =========================================================================


class TestReActFinalizeTurn:
    """Verify finalize_turn behavior."""

    @pytest.mark.asyncio
    async def test_returns_stored_response(self) -> None:
        """finalize_turn returns handle.final_response when set."""
        provider = _make_provider()
        provider.set_responses([text_response("Final answer")])
        manager = await _make_manager(provider)
        tool = EchoTool()
        strategy = ReActReasoning()

        handle = await strategy.begin_turn(manager, provider, tools=[tool])
        # Simulate process_input storing final_response
        await strategy.process_input(handle)

        response = await strategy.finalize_turn(handle)
        assert response.content == "Final answer"

    @pytest.mark.asyncio
    async def test_synthesis_call_when_no_stored_response(self) -> None:
        """finalize_turn calls manager.complete() when no final_response."""
        provider = _make_provider()
        provider.set_responses([
            tool_call_response("echo_tool", {"message": "same"}),
            tool_call_response("echo_tool", {"message": "same"}),
            text_response("Synthesized response"),
        ])
        manager = await _make_manager(provider)
        tool = EchoTool()
        strategy = ReActReasoning()

        handle = await strategy.begin_turn(manager, provider, tools=[tool])

        # First iteration: tool calls
        await strategy.process_input(handle)
        await manager.add_message(
            content="Observation", role="tool", name="echo_tool"
        )
        # Second iteration: duplicate -> handle.final_response = None
        result = await strategy.process_input(handle)
        assert result.action == "duplicate_break"

        # finalize_turn should do a synthesis call
        response = await strategy.finalize_turn(handle)
        assert response.content == "Synthesized response"

    @pytest.mark.asyncio
    async def test_type_error_on_wrong_handle(self) -> None:
        """finalize_turn raises TypeError for non-ReActTurnHandle."""
        strategy = ReActReasoning()
        wrong_handle = TurnHandle(manager=None, llm=None)

        with pytest.raises(TypeError, match="Expected ReActTurnHandle"):
            await strategy.finalize_turn(wrong_handle)


# =========================================================================
# ProcessResult.iterate field
# =========================================================================


class TestProcessResultIterate:
    """Verify ProcessResult iterate field behavior."""

    def test_iterate_defaults_to_false(self) -> None:
        """iterate defaults to False (wizard behavior)."""
        result = ProcessResult()
        assert result.iterate is False

    def test_iterate_set_to_true(self) -> None:
        result = ProcessResult(iterate=True)
        assert result.iterate is True


# =========================================================================
# TurnHandle.tool_extra_context field
# =========================================================================


class TestTurnHandleExtraContext:
    """Verify TurnHandle tool_extra_context field."""

    def test_defaults_to_empty_dict(self) -> None:
        handle = TurnHandle(manager=None, llm=None)
        assert handle.tool_extra_context == {}

    def test_set_extra_context(self) -> None:
        handle = TurnHandle(manager=None, llm=None)
        handle.tool_extra_context = {"key": "value"}
        assert handle.tool_extra_context["key"] == "value"


# =========================================================================
# End-to-end via BotTestHarness
# =========================================================================


class TestReActPhasedEndToEnd:
    """End-to-end: DynaBot routes ReAct through phased flow."""

    @pytest.mark.asyncio
    async def test_react_phased_tool_execution(self) -> None:
        """ReAct tool calls execute via DynaBot's phased loop."""
        tool = EchoTool()

        async with await BotTestHarness.create(
            bot_config={
                "llm": {"provider": "echo", "model": "test"},
                "conversation_storage": {"backend": "memory"},
                "reasoning": {"strategy": "react"},
            },
            main_responses=[
                tool_call_response("echo_tool", {"message": "hello"}),
                text_response("The tool said hello back"),
            ],
            tools=[tool],
        ) as harness:
            result = await harness.chat("Use the echo tool")

        assert result.response == "The tool said hello back"
        assert tool.call_count == 1

    @pytest.mark.asyncio
    async def test_react_phased_no_tools_fast_path(self) -> None:
        """ReAct with no tools registered uses simple generation."""
        async with await BotTestHarness.create(
            bot_config={
                "llm": {"provider": "echo", "model": "test"},
                "conversation_storage": {"backend": "memory"},
                "reasoning": {"strategy": "react"},
            },
            main_responses=[
                text_response("Simple response"),
            ],
        ) as harness:
            result = await harness.chat("Hello")

        assert result.response == "Simple response"

    @pytest.mark.asyncio
    async def test_react_phased_multi_iteration(self) -> None:
        """ReAct iterates multiple times through phased loop."""
        tool = EchoTool()

        async with await BotTestHarness.create(
            bot_config={
                "llm": {"provider": "echo", "model": "test"},
                "conversation_storage": {"backend": "memory"},
                "reasoning": {"strategy": "react"},
            },
            main_responses=[
                tool_call_response("echo_tool", {"message": "first"}),
                tool_call_response("echo_tool", {"message": "second"}),
                text_response("Done after two tool calls"),
            ],
            tools=[tool],
        ) as harness:
            result = await harness.chat("Do two things")

        assert result.response == "Done after two tool calls"
        assert tool.call_count == 2

    @pytest.mark.asyncio
    async def test_react_phased_middleware_fires_per_tool(self) -> None:
        """on_tool_executed middleware fires for each tool call."""
        from dataknobs_bots.bot.turn import ToolExecution
        from dataknobs_bots.middleware.base import Middleware

        class ToolTracker(Middleware):
            def __init__(self) -> None:
                self.executions: list[ToolExecution] = []

            async def on_tool_executed(
                self, execution: ToolExecution, context: Any
            ) -> None:
                self.executions.append(execution)

        tool = EchoTool()
        tracker = ToolTracker()

        async with await BotTestHarness.create(
            bot_config={
                "llm": {"provider": "echo", "model": "test"},
                "conversation_storage": {"backend": "memory"},
                "reasoning": {"strategy": "react"},
            },
            main_responses=[
                tool_call_response("echo_tool", {"message": "one"}),
                tool_call_response("echo_tool", {"message": "two"}),
                text_response("All done"),
            ],
            tools=[tool],
            middleware=[tracker],
        ) as harness:
            await harness.chat("Do things")

        assert len(tracker.executions) == 2
        assert tracker.executions[0].tool_name == "echo_tool"
        assert tracker.executions[1].tool_name == "echo_tool"


# =========================================================================
# Streaming iterative phased loop (StreamingPhasedProtocol + iterate=True)
# =========================================================================


class TestStreamingIterativePhasedLoop:
    """Exercise the streaming iterative loop in stream_chat.

    The StreamingPhasedProtocol branch in stream_chat has an iterative
    process_input loop that supports iterate=True, but no built-in
    strategy currently uses both StreamingPhasedProtocol and iterate=True.
    This test exercises that code path using a minimal inline strategy.
    """

    @pytest.mark.asyncio
    async def test_streaming_iterate_executes_tools_and_streams(self) -> None:
        """StreamingPhasedProtocol with iterate=True: DynaBot loops
        process_input, executes tools, then streams finalize_turn.
        """
        from dataclasses import dataclass

        from dataknobs_llm import LLMResponse, LLMStreamResponse

        from dataknobs_bots.bot.turn import ToolExecution
        from dataknobs_bots.reasoning.base import (
            ReasoningStrategy,
            ToolCallSpec,
        )

        @dataclass
        class _IterHandle(TurnHandle):
            """Turn handle for the test strategy."""
            call_count: int = 0

        class _IterStreamStrategy(ReasoningStrategy):
            """Minimal strategy: iterate once with a tool call, then stream."""

            async def begin_turn(
                self, manager: Any, llm: Any, tools: Any = None, **kw: Any,
            ) -> _IterHandle:
                return _IterHandle(
                    manager=manager, llm=llm, tools=tools, kwargs=kw,
                )

            async def process_input(self, handle: TurnHandle) -> ProcessResult:
                assert isinstance(handle, _IterHandle)
                handle.call_count += 1
                if handle.call_count == 1:
                    # First call: request tool execution and iterate
                    return ProcessResult(
                        needs_tool_execution=True,
                        iterate=True,
                        pending_tool_calls=[
                            ToolCallSpec(
                                name="echo_tool", parameters={"message": "hi"},
                            ),
                        ],
                        action="tool_calls",
                    )
                # Second call: no more tools
                return ProcessResult(action="final")

            async def finalize_turn(
                self,
                handle: TurnHandle,
                tool_results: list[ToolExecution] | None = None,
            ) -> Any:
                return LLMResponse(
                    content="streamed done",
                    model="test",
                    finish_reason="stop",
                )

            def stream_finalize_turn(
                self,
                handle: TurnHandle,
                tool_results: list[ToolExecution] | None = None,
            ) -> AsyncIterator[Any]:
                return self._stream(handle)

            async def _stream(self, handle: TurnHandle) -> AsyncIterator[Any]:
                yield LLMStreamResponse(delta="streamed ", is_final=False)
                yield LLMStreamResponse(
                    delta="done", is_final=True, finish_reason="stop",
                )

            async def generate(
                self, manager: Any, llm: Any, tools: Any = None, **kw: Any,
            ) -> Any:
                return LLMResponse(
                    content="fallback",
                    model="test",
                    finish_reason="stop",
                )

        from dataknobs_bots.bot.base import DynaBot
        from dataknobs_bots.bot.context import BotContext

        bot = await DynaBot.from_config({
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
            "reasoning": {"strategy": "simple"},
        })
        # Replace strategy with our test strategy
        strategy = _IterStreamStrategy()
        bot.reasoning_strategy = strategy

        tool = EchoTool()
        bot.tool_registry.register_tool(tool)

        context = BotContext(
            conversation_id="stream-iter-test", client_id="test"
        )

        chunks: list[str] = []
        async for chunk in bot.stream_chat("test", context):
            chunks.append(chunk.delta)

        assert "".join(chunks) == "streamed done"
        assert tool.call_count == 1
