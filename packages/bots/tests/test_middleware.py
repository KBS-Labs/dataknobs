"""Tests for middleware implementations."""

import pytest

from dataknobs_bots.bot.context import BotContext
from dataknobs_bots.bot.turn import TurnMode, TurnState
from dataknobs_bots.middleware import (
    CostTrackingMiddleware,
    LoggingMiddleware,
    Middleware,
)


@pytest.fixture
def bot_context() -> BotContext:
    """Create a test bot context."""
    return BotContext(
        conversation_id="test-conv-001",
        client_id="test-client",
        user_id="test-user",
    )


def _make_turn(
    context: BotContext,
    *,
    mode: TurnMode = TurnMode.CHAT,
    message: str = "Hello",
    response_content: str = "Hi there!",
    usage: dict | None = None,
    provider_name: str | None = None,
    model: str | None = None,
) -> TurnState:
    """Create a TurnState for testing middleware hooks."""
    turn = TurnState(mode=mode, message=message, context=context)
    turn.response_content = response_content
    turn.usage = usage
    turn.provider_name = provider_name
    turn.model = model
    return turn


class TestMiddlewareBase:
    """Tests for Middleware base class."""

    def test_middleware_is_concrete(self):
        """Middleware is a concrete base class with no-op hooks."""
        mw = Middleware()
        assert isinstance(mw, Middleware)

    @pytest.mark.asyncio
    async def test_all_hooks_are_noops(self, bot_context: BotContext):
        """All hooks on the base class run without error."""
        mw = Middleware()
        turn = _make_turn(bot_context)

        # Legacy hooks
        await mw.before_message("msg", bot_context)
        await mw.after_message("resp", bot_context)
        await mw.post_stream("msg", "resp", bot_context)
        await mw.on_error(ValueError("e"), "msg", bot_context)
        await mw.on_hook_error("hook", ValueError("e"), bot_context)

        # Unified hooks
        result = await mw.on_turn_start(turn)
        assert result is None
        await mw.after_turn(turn)
        from dataknobs_bots.bot.turn import ToolExecution
        await mw.on_tool_executed(
            ToolExecution(tool_name="t", parameters={}), bot_context
        )


class TestCostTrackingMiddleware:
    """Tests for CostTrackingMiddleware."""

    @pytest.mark.asyncio
    async def test_on_turn_start_runs_without_error(
        self, bot_context: BotContext
    ):
        """on_turn_start logs estimated tokens without error."""
        middleware = CostTrackingMiddleware()
        turn = _make_turn(bot_context, message="Hello, world!")
        result = await middleware.on_turn_start(turn)
        assert result is None  # No message transform

    @pytest.mark.asyncio
    async def test_after_turn_tracks_usage_with_real_data(
        self, bot_context: BotContext
    ):
        """after_turn tracks token usage from real provider data."""
        middleware = CostTrackingMiddleware()

        turn = _make_turn(
            bot_context,
            response_content="Hello! How can I help you?",
            usage={"input": 10, "output": 15},
            provider_name="openai",
            model="gpt-4o",
        )
        await middleware.after_turn(turn)

        stats = middleware.get_client_stats("test-client")
        assert stats is not None
        assert stats["total_requests"] == 1
        assert stats["total_input_tokens"] == 10
        assert stats["total_output_tokens"] == 15
        assert stats["total_cost_usd"] > 0

    @pytest.mark.asyncio
    async def test_after_turn_estimates_when_no_usage(
        self, bot_context: BotContext
    ):
        """after_turn estimates tokens from text when provider has no usage data."""
        middleware = CostTrackingMiddleware()

        turn = _make_turn(
            bot_context,
            message="Hello, world!",  # 13 chars → 3 estimated tokens
            response_content="Hi there! How can I help you?",  # 29 chars → 7 tokens
            usage=None,
        )
        await middleware.after_turn(turn)

        stats = middleware.get_client_stats("test-client")
        assert stats is not None
        assert stats["total_requests"] == 1
        assert stats["total_input_tokens"] > 0
        assert stats["total_output_tokens"] > 0
        assert stats["chat_turns"] == 1

    @pytest.mark.asyncio
    async def test_after_turn_streaming_increments_stream_counter(
        self, bot_context: BotContext
    ):
        """Streaming turns increment the stream_turns stat counter."""
        middleware = CostTrackingMiddleware()

        turn = _make_turn(
            bot_context,
            mode=TurnMode.STREAM,
            message="Hello",
            response_content="Streamed response here",
            usage=None,
        )
        await middleware.after_turn(turn)

        stats = middleware.get_client_stats("test-client")
        assert stats is not None
        assert stats["stream_turns"] == 1
        assert stats["chat_turns"] == 0

    @pytest.mark.asyncio
    async def test_after_turn_with_disabled_tracking(
        self, bot_context: BotContext
    ):
        """Disabled tracking doesn't record stats."""
        middleware = CostTrackingMiddleware(track_tokens=False)

        turn = _make_turn(
            bot_context,
            usage={"input": 10, "output": 15},
            provider_name="openai",
            model="gpt-4o",
        )
        await middleware.after_turn(turn)

        stats = middleware.get_client_stats("test-client")
        assert stats is None

    @pytest.mark.asyncio
    async def test_on_error_tracks_call_count(self, bot_context: BotContext):
        """on_error increments error counter on a fresh client."""
        middleware = CostTrackingMiddleware()
        error = ValueError("Test error")
        await middleware.on_error(error, "Test message", bot_context)

        stats = middleware.get_client_stats("test-client")
        assert stats is not None
        assert stats["on_error_calls"] == 1
        assert stats["chat_turns"] == 0
        assert stats["stream_turns"] == 0

    @pytest.mark.asyncio
    async def test_counters_accumulate_across_turn_types(
        self, bot_context: BotContext
    ):
        """Counters track chat vs stream requests and errors."""
        middleware = CostTrackingMiddleware()

        # Chat turn
        chat_turn = _make_turn(
            bot_context,
            usage={"input": 10, "output": 15},
            provider_name="ollama",
            model="llama3.2",
        )
        await middleware.after_turn(chat_turn)

        # Stream turn
        stream_turn = _make_turn(
            bot_context,
            mode=TurnMode.STREAM,
            message="Hello",
            response_content="Streamed",
            usage=None,
        )
        await middleware.after_turn(stream_turn)

        # Error
        await middleware.on_error(ValueError("err"), "msg", bot_context)

        stats = middleware.get_client_stats("test-client")
        assert stats is not None
        assert stats["chat_turns"] == 1
        assert stats["stream_turns"] == 1
        assert stats["on_error_calls"] == 1
        assert stats["total_requests"] == 2  # chat + stream

    @pytest.mark.asyncio
    async def test_after_turn_by_model_tracking(self, bot_context: BotContext):
        """after_turn tracks by_provider and by_model correctly."""
        middleware = CostTrackingMiddleware()

        turn = _make_turn(
            bot_context,
            usage={"input": 100, "output": 200},
            provider_name="openai",
            model="gpt-4o",
        )
        await middleware.after_turn(turn)

        stats = middleware.get_client_stats("test-client")
        assert stats is not None

        assert "openai" in stats["by_provider"]
        provider_stats = stats["by_provider"]["openai"]
        assert "by_model" in provider_stats
        assert "gpt-4o" in provider_stats["by_model"]

        model_stats = provider_stats["by_model"]["gpt-4o"]
        assert model_stats["requests"] == 1
        assert model_stats["input_tokens"] == 100
        assert model_stats["output_tokens"] == 200
        assert model_stats["cost_usd"] > 0

    @pytest.mark.asyncio
    async def test_on_error(self, bot_context: BotContext):
        """on_error hook runs without error."""
        middleware = CostTrackingMiddleware()
        error = ValueError("Test error")
        await middleware.on_error(error, "Test message", bot_context)

    @pytest.mark.asyncio
    async def test_custom_cost_rates(self, bot_context: BotContext):
        """Custom cost rates are applied."""
        custom_rates = {
            "custom_provider": {
                "custom_model": {"input": 0.01, "output": 0.02},
            }
        }
        middleware = CostTrackingMiddleware(cost_rates=custom_rates)

        turn = _make_turn(
            bot_context,
            usage={"input": 1000, "output": 1000},
            provider_name="custom_provider",
            model="custom_model",
        )
        await middleware.after_turn(turn)

        stats = middleware.get_client_stats("test-client")
        assert stats is not None
        # Cost should be: (1000/1000 * 0.01) + (1000/1000 * 0.02) = 0.03
        assert abs(stats["total_cost_usd"] - 0.03) < 0.001

    @pytest.mark.asyncio
    async def test_ollama_is_free(self, bot_context: BotContext):
        """Ollama models have zero cost."""
        middleware = CostTrackingMiddleware()

        turn = _make_turn(
            bot_context,
            usage={"input": 1000, "output": 1000},
            provider_name="ollama",
            model="llama3.1:8b",
        )
        await middleware.after_turn(turn)

        stats = middleware.get_client_stats("test-client")
        assert stats is not None
        assert stats["total_cost_usd"] == 0.0

    def test_get_total_cost(self):
        """get_total_cost aggregates across clients."""
        middleware = CostTrackingMiddleware()

        middleware._usage_stats["client-1"] = {
            "total_cost_usd": 0.05,
            "total_requests": 1,
            "total_input_tokens": 100,
            "total_output_tokens": 100,
        }
        middleware._usage_stats["client-2"] = {
            "total_cost_usd": 0.10,
            "total_requests": 2,
            "total_input_tokens": 200,
            "total_output_tokens": 200,
        }

        total = middleware.get_total_cost()
        assert abs(total - 0.15) < 0.001

    def test_get_total_tokens(self):
        """get_total_tokens aggregates across clients."""
        middleware = CostTrackingMiddleware()

        middleware._usage_stats["client-1"] = {
            "total_cost_usd": 0.0,
            "total_requests": 1,
            "total_input_tokens": 100,
            "total_output_tokens": 150,
        }
        middleware._usage_stats["client-2"] = {
            "total_cost_usd": 0.0,
            "total_requests": 1,
            "total_input_tokens": 200,
            "total_output_tokens": 250,
        }

        tokens = middleware.get_total_tokens()
        assert tokens["input"] == 300
        assert tokens["output"] == 400
        assert tokens["total"] == 700

    def test_clear_stats_single_client(self):
        """Clearing stats for a single client."""
        middleware = CostTrackingMiddleware()

        middleware._usage_stats["client-1"] = {"total_cost_usd": 0.05}
        middleware._usage_stats["client-2"] = {"total_cost_usd": 0.10}

        middleware.clear_stats("client-1")

        assert middleware.get_client_stats("client-1") is None
        assert middleware.get_client_stats("client-2") is not None

    def test_clear_stats_all(self):
        """Clearing all stats."""
        middleware = CostTrackingMiddleware()

        middleware._usage_stats["client-1"] = {"total_cost_usd": 0.05}
        middleware._usage_stats["client-2"] = {"total_cost_usd": 0.10}

        middleware.clear_stats()

        assert middleware.get_client_stats("client-1") is None
        assert middleware.get_client_stats("client-2") is None

    def test_export_stats_json(self):
        """Exporting stats as JSON."""
        middleware = CostTrackingMiddleware()

        middleware._usage_stats["client-1"] = {
            "client_id": "client-1",
            "total_cost_usd": 0.05,
            "total_requests": 1,
            "total_input_tokens": 100,
            "total_output_tokens": 100,
            "by_provider": {},
        }

        json_str = middleware.export_stats_json()
        assert "client-1" in json_str
        assert "0.05" in json_str

    def test_export_stats_csv(self):
        """Exporting stats as CSV."""
        middleware = CostTrackingMiddleware()

        middleware._usage_stats["client-1"] = {
            "total_requests": 1,
            "total_input_tokens": 100,
            "total_output_tokens": 150,
            "total_cost_usd": 0.05,
        }

        csv_str = middleware.export_stats_csv()
        lines = csv_str.strip().split("\n")
        assert len(lines) == 2  # Header + 1 data row
        assert "client_id" in lines[0]
        assert "client-1" in lines[1]


class TestLoggingMiddleware:
    """Tests for LoggingMiddleware."""

    @pytest.mark.asyncio
    async def test_on_turn_start(self, bot_context: BotContext):
        """on_turn_start logs user message without error."""
        middleware = LoggingMiddleware()
        turn = _make_turn(bot_context, message="Hello, world!")
        result = await middleware.on_turn_start(turn)
        assert result is None  # No message transform

    @pytest.mark.asyncio
    async def test_after_turn(self, bot_context: BotContext):
        """after_turn logs turn completion without error."""
        middleware = LoggingMiddleware()
        turn = _make_turn(
            bot_context,
            response_content="Hi there!",
            usage={"input": 10, "output": 15},
            provider_name="echo",
            model="test",
        )
        await middleware.after_turn(turn)

    @pytest.mark.asyncio
    async def test_on_error(self, bot_context: BotContext):
        """on_error logs without error."""
        middleware = LoggingMiddleware()
        error = ValueError("Test error")
        await middleware.on_error(error, "Test message", bot_context)

    def test_json_format_option(self):
        """JSON format option is set correctly."""
        middleware = LoggingMiddleware(json_format=True)
        assert middleware.json_format is True

        middleware = LoggingMiddleware(json_format=False)
        assert middleware.json_format is False

    def test_include_metadata_option(self):
        """include_metadata option is set correctly."""
        middleware = LoggingMiddleware(include_metadata=True)
        assert middleware.include_metadata is True

        middleware = LoggingMiddleware(include_metadata=False)
        assert middleware.include_metadata is False

    def test_log_level_option(self):
        """log_level option is set correctly."""
        middleware = LoggingMiddleware(log_level="DEBUG")
        assert middleware.log_level == "DEBUG"

        middleware = LoggingMiddleware(log_level="WARNING")
        assert middleware.log_level == "WARNING"
