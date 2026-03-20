"""Tests for TurnState pipeline, after_turn middleware, and on_tool_executed.

Verifies:
- TurnState dataclass behavior
- after_turn middleware fires for chat, stream, and greet paths
- after_turn receives real usage data (Gap 1 fix)
- on_tool_executed fires for ReAct tool executions (Gap 9 fix)
- CostTrackingMiddleware.after_turn uses real tokens (not estimates)
"""

from typing import Any

import pytest

from dataknobs_bots.bot.context import BotContext
from dataknobs_bots.bot.turn import ToolExecution, TurnMode, TurnState
from dataknobs_bots.middleware.base import Middleware


# ---------------------------------------------------------------------------
# Test middleware that records after_turn and on_tool_executed calls
# ---------------------------------------------------------------------------
class TurnTrackingMiddleware(Middleware):
    """Middleware that captures after_turn and on_tool_executed calls."""

    def __init__(self) -> None:
        self.turns: list[TurnState] = []
        self.tool_executions: list[ToolExecution] = []

    async def after_turn(self, turn: TurnState) -> None:
        self.turns.append(turn)

    async def on_tool_executed(
        self, execution: ToolExecution, context: BotContext
    ) -> None:
        self.tool_executions.append(execution)


# ---------------------------------------------------------------------------
# TurnState dataclass tests
# ---------------------------------------------------------------------------
class TestTurnStateDataclass:
    """TurnState holds per-turn pipeline state."""

    def test_defaults(self) -> None:
        ctx = BotContext(conversation_id="c1", client_id="t1")
        turn = TurnState(mode=TurnMode.CHAT, message="hi", context=ctx)
        assert turn.is_streaming is False
        assert turn.is_greet is False
        assert turn.response_content == ""
        assert turn.tool_executions == []
        assert turn.stream_chunks == []
        assert turn.usage is None

    def test_stream_mode(self) -> None:
        ctx = BotContext(conversation_id="c1", client_id="t1")
        turn = TurnState(mode=TurnMode.STREAM, message="hi", context=ctx)
        assert turn.is_streaming is True
        assert turn.is_greet is False

    def test_greet_mode(self) -> None:
        ctx = BotContext(conversation_id="c1", client_id="t1")
        turn = TurnState(mode=TurnMode.GREET, message="", context=ctx)
        assert turn.is_greet is True
        assert turn.is_streaming is False

    def test_middleware_kwargs(self) -> None:
        ctx = BotContext(conversation_id="c1", client_id="t1")
        turn = TurnState(mode=TurnMode.CHAT, message="hi", context=ctx)
        turn.usage = {"prompt_tokens": 10, "completion_tokens": 20}
        turn.model = "test-model"
        turn.provider_name = "echo"
        kwargs = turn.middleware_kwargs()
        assert kwargs["tokens_used"] == {"prompt_tokens": 10, "completion_tokens": 20}
        assert kwargs["model"] == "test-model"
        assert kwargs["provider"] == "echo"

    def test_middleware_kwargs_empty_when_no_data(self) -> None:
        ctx = BotContext(conversation_id="c1", client_id="t1")
        turn = TurnState(mode=TurnMode.CHAT, message="hi", context=ctx)
        assert turn.middleware_kwargs() == {}

    def test_populate_from_response(self) -> None:
        from dataknobs_llm import LLMResponse

        ctx = BotContext(conversation_id="c1", client_id="t1")
        turn = TurnState(mode=TurnMode.CHAT, message="hi", context=ctx)
        response = LLMResponse(
            content="hello",
            model="test-model",
            finish_reason="stop",
            usage={"prompt_tokens": 5, "completion_tokens": 10},
        )

        class FakeProvider:
            provider_name = "test-provider"

        turn.populate_from_response(response, FakeProvider())
        assert turn.usage == {"prompt_tokens": 5, "completion_tokens": 10}
        assert turn.model == "test-model"
        assert turn.provider_name == "test-provider"


# ---------------------------------------------------------------------------
# ToolExecution dataclass tests
# ---------------------------------------------------------------------------
class TestToolExecution:
    """ToolExecution records a single tool invocation."""

    def test_success(self) -> None:
        ex = ToolExecution(
            tool_name="search", parameters={"q": "test"}, result="found it"
        )
        assert ex.tool_name == "search"
        assert ex.error is None
        assert ex.duration_ms is None

    def test_failure(self) -> None:
        ex = ToolExecution(
            tool_name="search", parameters={"q": "test"}, error="not found"
        )
        assert ex.error == "not found"
        assert ex.result is None


# ---------------------------------------------------------------------------
# after_turn middleware dispatch tests
# ---------------------------------------------------------------------------
class TestAfterTurnMiddleware:
    """after_turn fires for all turn types with TurnState."""

    @pytest.mark.asyncio
    async def test_chat_fires_after_turn(self) -> None:
        from dataknobs_bots.testing import BotTestHarness

        tracker = TurnTrackingMiddleware()

        async with await BotTestHarness.create(
            bot_config={
                "llm": {"provider": "echo", "model": "test"},
                "conversation_storage": {"backend": "memory"},
            },
            middleware=[tracker],
        ) as harness:
            await harness.chat("Hello")

        assert len(tracker.turns) == 1
        turn = tracker.turns[0]
        assert turn.mode == TurnMode.CHAT
        assert turn.message == "Hello"
        assert turn.response_content != ""

    @pytest.mark.asyncio
    async def test_stream_fires_after_turn(self) -> None:
        from dataknobs_bots.testing import BotTestHarness

        tracker = TurnTrackingMiddleware()

        async with await BotTestHarness.create(
            bot_config={
                "llm": {"provider": "echo", "model": "test"},
                "conversation_storage": {"backend": "memory"},
            },
            middleware=[tracker],
        ) as harness:
            chunks = []
            async for chunk in harness.bot.stream_chat(
                "Hello", harness.context
            ):
                chunks.append(chunk.delta)

        assert len(tracker.turns) == 1
        turn = tracker.turns[0]
        assert turn.mode == TurnMode.STREAM
        assert turn.is_streaming is True
        assert turn.response_content == "".join(chunks)

    @pytest.mark.asyncio
    async def test_greet_fires_after_turn(self) -> None:
        from dataknobs_bots.testing import BotTestHarness

        tracker = TurnTrackingMiddleware()

        async with await BotTestHarness.create(
            bot_config={
                "llm": {"provider": "echo", "model": "test"},
                "conversation_storage": {"backend": "memory"},
                "reasoning": {
                    "strategy": "simple",
                    "greeting_template": "Welcome!",
                },
            },
            middleware=[tracker],
        ) as harness:
            result = await harness.greet()
            assert result.response == "Welcome!"

        assert len(tracker.turns) == 1
        turn = tracker.turns[0]
        assert turn.mode == TurnMode.GREET
        assert turn.is_greet is True
        assert turn.response_content == "Welcome!"


# ---------------------------------------------------------------------------
# after_turn usage data tests (Gap 1 fix)
# ---------------------------------------------------------------------------
class TestAfterTurnUsageData:
    """after_turn receives real usage data for all turn types."""

    @pytest.mark.asyncio
    async def test_chat_turn_has_provider_name(self) -> None:
        """Chat turns populate provider_name from the LLM."""
        from dataknobs_bots.testing import BotTestHarness

        tracker = TurnTrackingMiddleware()

        async with await BotTestHarness.create(
            bot_config={
                "llm": {"provider": "echo", "model": "test-model"},
                "conversation_storage": {"backend": "memory"},
            },
            middleware=[tracker],
        ) as harness:
            await harness.chat("Hello")

        turn = tracker.turns[0]
        # EchoProvider sets provider_name
        assert turn.provider_name is not None
        assert turn.model is not None


# ---------------------------------------------------------------------------
# on_tool_executed tests (Gap 9 fix)
# ---------------------------------------------------------------------------
class TestOnToolExecuted:
    """on_tool_executed fires for each tool execution in a turn."""

    @pytest.mark.asyncio
    async def test_react_tool_execution_fires_hook(self) -> None:
        """ReAct strategy populates tool_executions, DynaBot fires hooks."""
        from dataknobs_llm import Tool
        from dataknobs_llm.testing import text_response, tool_call_response

        from dataknobs_bots.testing import BotTestHarness

        class GreetTool(Tool):
            def __init__(self) -> None:
                super().__init__(
                    name="greet_tool", description="Greets a person"
                )

            @property
            def schema(self) -> dict[str, Any]:
                return {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                }

            async def execute(self, **kwargs: Any) -> str:
                name = kwargs.get("name", "World")
                return f"Greeting sent to {name}"

        tracker = TurnTrackingMiddleware()

        async with await BotTestHarness.create(
            bot_config={
                "llm": {"provider": "echo", "model": "test"},
                "conversation_storage": {"backend": "memory"},
                "reasoning": {"strategy": "react"},
            },
            main_responses=[
                tool_call_response("greet_tool", {"name": "Alice"}),
                text_response("Hello Alice!"),
            ],
            tools=[GreetTool()],
            middleware=[tracker],
        ) as harness:
            await harness.chat("Greet Alice")

        # Verify on_tool_executed was called
        assert len(tracker.tool_executions) == 1
        exec_record = tracker.tool_executions[0]
        assert exec_record.tool_name == "greet_tool"
        assert exec_record.parameters == {"name": "Alice"}
        assert exec_record.result == "Greeting sent to Alice"
        assert exec_record.error is None
        assert exec_record.duration_ms is not None
        assert exec_record.duration_ms >= 0

        # Also verify turn has the tool executions
        turn = tracker.turns[0]
        assert len(turn.tool_executions) == 1
        assert turn.tool_executions[0].tool_name == "greet_tool"


# ---------------------------------------------------------------------------
# CostTrackingMiddleware after_turn integration
# ---------------------------------------------------------------------------
class TestCostTrackingAfterTurn:
    """CostTrackingMiddleware.after_turn uses real usage data."""

    @pytest.mark.asyncio
    async def test_cost_tracking_after_turn_skips_without_usage(self) -> None:
        """after_turn is a no-op when turn has no usage data."""
        from dataknobs_bots.middleware.cost import CostTrackingMiddleware

        mw = CostTrackingMiddleware()
        ctx = BotContext(conversation_id="c1", client_id="t1")
        turn = TurnState(mode=TurnMode.CHAT, message="hi", context=ctx)
        # No usage data
        await mw.after_turn(turn)
        # Should not record anything — legacy hooks handle it
        totals = mw.get_total_tokens()
        assert totals["total"] == 0

    @pytest.mark.asyncio
    async def test_cost_tracking_after_turn_records_real_usage(self) -> None:
        """after_turn records real token counts when available."""
        from dataknobs_bots.middleware.cost import CostTrackingMiddleware

        mw = CostTrackingMiddleware()
        ctx = BotContext(conversation_id="c1", client_id="t1")
        turn = TurnState(mode=TurnMode.STREAM, message="hi", context=ctx)
        turn.usage = {"prompt_tokens": 100, "completion_tokens": 200}
        turn.provider_name = "ollama"
        turn.model = "llama3.2"

        await mw.after_turn(turn)

        stats = mw.get_client_stats("t1")
        assert stats is not None
        assert stats["total_input_tokens"] == 100
        assert stats["total_output_tokens"] == 200
        # Streaming path should increment post_stream_calls
        assert stats["post_stream_calls"] == 1


# ---------------------------------------------------------------------------
# Default no-op middleware hooks don't break existing middleware
# ---------------------------------------------------------------------------
class TestMiddlewareBackwardCompatibility:
    """Existing middleware subclasses inherit no-op after_turn/on_tool_executed."""

    @pytest.mark.asyncio
    async def test_existing_middleware_inherits_noop(self) -> None:
        """Middleware subclasses that only override legacy hooks still work.

        Legacy middleware that only implements before_message/after_message etc.
        inherits no-op unified hooks — no breakage.
        """

        class LegacyMiddleware(Middleware):
            """Middleware using only legacy hooks (still supported)."""

            async def before_message(
                self, message: str, context: BotContext
            ) -> None:
                pass

            async def after_message(
                self, response: str, context: BotContext, **kwargs: Any
            ) -> None:
                pass

        mw = LegacyMiddleware()
        ctx = BotContext(conversation_id="c1", client_id="t1")
        turn = TurnState(mode=TurnMode.CHAT, message="hi", context=ctx)

        # Unified hooks should be inherited no-ops
        await mw.after_turn(turn)
        await mw.on_tool_executed(
            ToolExecution(tool_name="test", parameters={}), ctx
        )
        result = await mw.on_turn_start(turn)
        assert result is None

        # Legacy hooks it implements should also work
        await mw.before_message("hi", ctx)
        await mw.after_message("hello", ctx)

        # Legacy hooks it doesn't implement should be inherited no-ops
        await mw.post_stream("hi", "hello", ctx)
        await mw.on_error(ValueError("e"), "hi", ctx)
        await mw.on_hook_error("hook", ValueError("e"), ctx)


# ---------------------------------------------------------------------------
# Plugin data middleware for testing
# ---------------------------------------------------------------------------
class PluginDataMiddleware(Middleware):
    """Middleware that writes to plugin_data in on_turn_start and reads in after_turn."""

    def __init__(self, key: str = "test_plugin", value: Any = "written") -> None:
        self.key = key
        self.value = value
        self.read_back: Any = None

    async def on_turn_start(self, turn: TurnState) -> str | None:
        turn.plugin_data[self.key] = self.value
        return None

    async def after_turn(self, turn: TurnState) -> None:
        self.read_back = turn.plugin_data.get(self.key)


class MessageTransformMiddleware(Middleware):
    """Middleware that transforms messages via on_turn_start."""

    def __init__(self, transform_fn: Any) -> None:
        self._transform_fn = transform_fn

    async def on_turn_start(self, turn: TurnState) -> str | None:
        return self._transform_fn(turn.message)


# ---------------------------------------------------------------------------
# Plugin data round-trip tests
# ---------------------------------------------------------------------------
class TestPluginData:
    """Plugin data flows from on_turn_start through to after_turn."""

    @pytest.mark.asyncio
    async def test_plugin_data_round_trip(self) -> None:
        """on_turn_start writes → after_turn reads same data."""
        from dataknobs_bots.testing import BotTestHarness

        plugin_mw = PluginDataMiddleware(key="session_id", value="abc-123")

        async with await BotTestHarness.create(
            bot_config={
                "llm": {"provider": "echo", "model": "test"},
                "conversation_storage": {"backend": "memory"},
            },
            middleware=[plugin_mw],
        ) as harness:
            await harness.chat("Hello")

        assert plugin_mw.read_back == "abc-123"

    @pytest.mark.asyncio
    async def test_plugin_data_available_in_after_turn(self) -> None:
        """after_turn sees plugin_data written by on_turn_start."""
        from dataknobs_bots.testing import BotTestHarness

        tracker = TurnTrackingMiddleware()
        plugin_mw = PluginDataMiddleware(key="flag", value=True)

        async with await BotTestHarness.create(
            bot_config={
                "llm": {"provider": "echo", "model": "test"},
                "conversation_storage": {"backend": "memory"},
            },
            middleware=[plugin_mw, tracker],
        ) as harness:
            await harness.chat("Hello")

        turn = tracker.turns[0]
        assert turn.plugin_data["flag"] is True

    @pytest.mark.asyncio
    async def test_plugin_data_cleared_between_turns(self) -> None:
        """Plugin data does not leak between turns."""
        from dataknobs_bots.testing import BotTestHarness

        tracker = TurnTrackingMiddleware()
        # Only writes on first turn
        call_count = 0

        class OnceMiddleware(Middleware):
            async def on_turn_start(self, turn: TurnState) -> str | None:
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    turn.plugin_data["first_only"] = True
                return None

        async with await BotTestHarness.create(
            bot_config={
                "llm": {"provider": "echo", "model": "test"},
                "conversation_storage": {"backend": "memory"},
            },
            main_responses=["response 1", "response 2"],
            middleware=[OnceMiddleware(), tracker],
        ) as harness:
            await harness.chat("Turn 1")
            await harness.chat("Turn 2")

        # First turn has the data
        assert tracker.turns[0].plugin_data.get("first_only") is True
        # Second turn should NOT have it (plugin_data is fresh per turn)
        assert "first_only" not in tracker.turns[1].plugin_data


# ---------------------------------------------------------------------------
# Message transform tests
# ---------------------------------------------------------------------------
class TestMessageTransform:
    """on_turn_start can transform the user message before it reaches the LLM."""

    @pytest.mark.asyncio
    async def test_message_transform_applied(self) -> None:
        """Transformed message is what the LLM sees."""
        from dataknobs_bots.testing import BotTestHarness

        transform = MessageTransformMiddleware(
            transform_fn=lambda msg: msg.upper()
        )
        tracker = TurnTrackingMiddleware()

        async with await BotTestHarness.create(
            bot_config={
                "llm": {"provider": "echo", "model": "test"},
                "conversation_storage": {"backend": "memory"},
            },
            middleware=[transform, tracker],
        ) as harness:
            await harness.chat("hello world")

        turn = tracker.turns[0]
        # The message on the turn should be the transformed version
        assert turn.message == "HELLO WORLD"

    @pytest.mark.asyncio
    async def test_chained_transforms(self) -> None:
        """Multiple transforms chain: each receives the previous result."""
        from dataknobs_bots.testing import BotTestHarness

        upper = MessageTransformMiddleware(transform_fn=lambda msg: msg.upper())
        exclaim = MessageTransformMiddleware(transform_fn=lambda msg: msg + "!")
        tracker = TurnTrackingMiddleware()

        async with await BotTestHarness.create(
            bot_config={
                "llm": {"provider": "echo", "model": "test"},
                "conversation_storage": {"backend": "memory"},
            },
            middleware=[upper, exclaim, tracker],
        ) as harness:
            await harness.chat("hello")

        turn = tracker.turns[0]
        # upper runs first → "HELLO", then exclaim → "HELLO!"
        assert turn.message == "HELLO!"


# ---------------------------------------------------------------------------
# LLM middleware bridge tests
# ---------------------------------------------------------------------------
class TestLLMMiddlewareBridge:
    """Plugin data bridges to ConversationMiddleware via turn_data."""

    @pytest.mark.asyncio
    async def test_conversation_middleware_sees_plugin_data(self) -> None:
        """ConversationMiddleware can read plugin_data via state.turn_data."""
        from dataknobs_llm.conversations.middleware import ConversationMiddleware

        from dataknobs_bots.testing import BotTestHarness

        # ConversationMiddleware that reads turn_data
        seen_turn_data: dict[str, Any] = {}

        class SpyConvMiddleware(ConversationMiddleware):
            async def process_request(self, messages: Any, state: Any) -> Any:
                seen_turn_data.update(state.turn_data)
                return messages

            async def process_response(self, response: Any, state: Any) -> Any:
                return response

        # Bot middleware that writes plugin_data
        plugin_mw = PluginDataMiddleware(key="bridge_test", value="visible")

        async with await BotTestHarness.create(
            bot_config={
                "llm": {"provider": "echo", "model": "test"},
                "conversation_storage": {"backend": "memory"},
            },
            middleware=[plugin_mw],
        ) as harness:
            # Inject ConversationMiddleware onto the manager
            # We need to do a chat first to create the manager, then add MW
            # Actually, let's add it to the bot's conversation storage config
            # Better: do a chat, get the manager, add middleware, do another chat
            await harness.chat("Setup turn")

            # Get the cached manager and add our spy middleware
            manager = harness.bot._conversation_managers[
                harness.context.conversation_id
            ]
            manager.middleware.append(SpyConvMiddleware())

            # Second chat — spy should see plugin_data
            await harness.chat("Bridge test")

        assert seen_turn_data.get("bridge_test") == "visible"

    @pytest.mark.asyncio
    async def test_conversation_middleware_can_write_to_turn_data(self) -> None:
        """ConversationMiddleware writes to turn_data → visible in after_turn."""
        from dataknobs_llm.conversations.middleware import ConversationMiddleware

        from dataknobs_bots.testing import BotTestHarness

        class WriterConvMiddleware(ConversationMiddleware):
            async def process_request(self, messages: Any, state: Any) -> Any:
                state.turn_data["llm_mw_wrote"] = "from_llm_layer"
                return messages

            async def process_response(self, response: Any, state: Any) -> Any:
                return response

        tracker = TurnTrackingMiddleware()

        async with await BotTestHarness.create(
            bot_config={
                "llm": {"provider": "echo", "model": "test"},
                "conversation_storage": {"backend": "memory"},
            },
            middleware=[tracker],
        ) as harness:
            # Setup manager
            await harness.chat("Setup turn")
            manager = harness.bot._conversation_managers[
                harness.context.conversation_id
            ]
            manager.middleware.append(WriterConvMiddleware())

            await harness.chat("Write test")

        # The second turn should have the data written by ConversationMiddleware
        turn = tracker.turns[1]
        assert turn.plugin_data.get("llm_mw_wrote") == "from_llm_layer"


# ---------------------------------------------------------------------------
# Tool bridge tests
# ---------------------------------------------------------------------------
class TestToolBridge:
    """Plugin data reaches tools via ToolExecutionContext.extra['turn_data']."""

    @pytest.mark.asyncio
    async def test_tool_sees_plugin_data(self) -> None:
        """Tool can read plugin_data via _context.extra['turn_data']."""
        from dataknobs_llm import Tool
        from dataknobs_llm.testing import text_response, tool_call_response

        from dataknobs_bots.testing import BotTestHarness

        seen_turn_data: dict[str, Any] = {}

        class SpyTool(Tool):
            def __init__(self) -> None:
                super().__init__(name="spy_tool", description="Reads turn_data")

            @property
            def schema(self) -> dict[str, Any]:
                return {"type": "object", "properties": {}}

            async def execute(self, **kwargs: Any) -> str:
                ctx = kwargs.get("_context")
                if ctx and "turn_data" in ctx.extra:
                    seen_turn_data.update(ctx.extra["turn_data"])
                return "spied"

        plugin_mw = PluginDataMiddleware(key="tool_visible", value=42)

        async with await BotTestHarness.create(
            bot_config={
                "llm": {"provider": "echo", "model": "test"},
                "conversation_storage": {"backend": "memory"},
                "reasoning": {"strategy": "react"},
            },
            main_responses=[
                tool_call_response("spy_tool", {}),
                text_response("Done spying"),
            ],
            tools=[SpyTool()],
            middleware=[plugin_mw],
        ) as harness:
            await harness.chat("Spy on data")

        assert seen_turn_data.get("tool_visible") == 42
