"""Tests for middleware implementations."""

import pytest

from dataknobs_bots.bot.context import BotContext
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


class TestMiddlewareBase:
    """Tests for Middleware base class."""

    def test_middleware_is_abstract(self):
        """Test that Middleware cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Middleware()  # type: ignore


class TestCostTrackingMiddleware:
    """Tests for CostTrackingMiddleware."""

    @pytest.mark.asyncio
    async def test_before_message(self, bot_context: BotContext):
        """Test before_message hook runs without error."""
        middleware = CostTrackingMiddleware()
        await middleware.before_message("Hello, world!", bot_context)
        # Should complete without error

    @pytest.mark.asyncio
    async def test_after_message_tracks_usage(self, bot_context: BotContext):
        """Test that after_message tracks token usage."""
        middleware = CostTrackingMiddleware()

        await middleware.after_message(
            "Hello! How can I help you?",
            bot_context,
            tokens_used={"input": 10, "output": 15},
            provider="openai",
            model="gpt-4o",
        )

        stats = middleware.get_client_stats("test-client")
        assert stats is not None
        assert stats["total_requests"] == 1
        assert stats["total_input_tokens"] == 10
        assert stats["total_output_tokens"] == 15
        assert stats["total_cost_usd"] > 0

    @pytest.mark.asyncio
    async def test_after_message_with_disabled_tracking(self, bot_context: BotContext):
        """Test that disabled tracking doesn't record stats."""
        middleware = CostTrackingMiddleware(track_tokens=False)

        await middleware.after_message(
            "Response",
            bot_context,
            tokens_used={"input": 10, "output": 15},
            provider="openai",
            model="gpt-4o",
        )

        stats = middleware.get_client_stats("test-client")
        assert stats is None

    @pytest.mark.asyncio
    async def test_on_error(self, bot_context: BotContext):
        """Test on_error hook runs without error."""
        middleware = CostTrackingMiddleware()
        error = ValueError("Test error")
        await middleware.on_error(error, "Test message", bot_context)
        # Should complete without error

    @pytest.mark.asyncio
    async def test_custom_cost_rates(self, bot_context: BotContext):
        """Test custom cost rates are applied."""
        custom_rates = {
            "custom_provider": {
                "custom_model": {"input": 0.01, "output": 0.02},
            }
        }
        middleware = CostTrackingMiddleware(cost_rates=custom_rates)

        await middleware.after_message(
            "Response",
            bot_context,
            tokens_used={"input": 1000, "output": 1000},
            provider="custom_provider",
            model="custom_model",
        )

        stats = middleware.get_client_stats("test-client")
        assert stats is not None
        # Cost should be: (1000/1000 * 0.01) + (1000/1000 * 0.02) = 0.03
        assert abs(stats["total_cost_usd"] - 0.03) < 0.001

    @pytest.mark.asyncio
    async def test_ollama_is_free(self, bot_context: BotContext):
        """Test that Ollama models have zero cost."""
        middleware = CostTrackingMiddleware()

        await middleware.after_message(
            "Response",
            bot_context,
            tokens_used={"input": 1000, "output": 1000},
            provider="ollama",
            model="llama3.1:8b",
        )

        stats = middleware.get_client_stats("test-client")
        assert stats is not None
        assert stats["total_cost_usd"] == 0.0

    def test_get_total_cost(self):
        """Test get_total_cost aggregates across clients."""
        middleware = CostTrackingMiddleware()

        # Manually set up stats for testing
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
        """Test get_total_tokens aggregates across clients."""
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
        """Test clearing stats for a single client."""
        middleware = CostTrackingMiddleware()

        middleware._usage_stats["client-1"] = {"total_cost_usd": 0.05}
        middleware._usage_stats["client-2"] = {"total_cost_usd": 0.10}

        middleware.clear_stats("client-1")

        assert middleware.get_client_stats("client-1") is None
        assert middleware.get_client_stats("client-2") is not None

    def test_clear_stats_all(self):
        """Test clearing all stats."""
        middleware = CostTrackingMiddleware()

        middleware._usage_stats["client-1"] = {"total_cost_usd": 0.05}
        middleware._usage_stats["client-2"] = {"total_cost_usd": 0.10}

        middleware.clear_stats()

        assert middleware.get_client_stats("client-1") is None
        assert middleware.get_client_stats("client-2") is None

    def test_export_stats_json(self):
        """Test exporting stats as JSON."""
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
        """Test exporting stats as CSV."""
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
    async def test_before_message(self, bot_context: BotContext):
        """Test before_message logs without error."""
        middleware = LoggingMiddleware()
        await middleware.before_message("Hello, world!", bot_context)
        # Should complete without error

    @pytest.mark.asyncio
    async def test_after_message(self, bot_context: BotContext):
        """Test after_message logs without error."""
        middleware = LoggingMiddleware()
        await middleware.after_message(
            "Hi there!",
            bot_context,
            tokens_used={"input": 10, "output": 15},
            response_time_ms=150,
        )
        # Should complete without error

    @pytest.mark.asyncio
    async def test_on_error(self, bot_context: BotContext):
        """Test on_error logs without error."""
        middleware = LoggingMiddleware()
        error = ValueError("Test error")
        await middleware.on_error(error, "Test message", bot_context)
        # Should complete without error

    def test_json_format_option(self):
        """Test JSON format option is set correctly."""
        middleware = LoggingMiddleware(json_format=True)
        assert middleware.json_format is True

        middleware = LoggingMiddleware(json_format=False)
        assert middleware.json_format is False

    def test_include_metadata_option(self):
        """Test include_metadata option is set correctly."""
        middleware = LoggingMiddleware(include_metadata=True)
        assert middleware.include_metadata is True

        middleware = LoggingMiddleware(include_metadata=False)
        assert middleware.include_metadata is False

    def test_log_level_option(self):
        """Test log_level option is set correctly."""
        middleware = LoggingMiddleware(log_level="DEBUG")
        assert middleware.log_level == "DEBUG"

        middleware = LoggingMiddleware(log_level="WARNING")
        assert middleware.log_level == "WARNING"
