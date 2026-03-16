"""Tests for DynaBot on_error middleware dispatch across all entry points.

Verifies that chat(), greet(), and stream_chat() all call on_error middleware
when exceptions occur during message processing.

Bug: chat() and greet() had no error handling — exceptions propagated directly
to the caller without calling on_error middleware. Only stream_chat() called
on_error. This test file reproduces the bug and verifies the fix.
"""

from typing import Any

import pytest
from dataknobs_bots.bot.base import DynaBot
from dataknobs_bots.bot.context import BotContext
from dataknobs_bots.middleware.base import Middleware
from dataknobs_bots.reasoning.base import ReasoningStrategy
from dataknobs_bots.testing import ErrorRaisingStrategy
from dataknobs_data.backends.memory import AsyncMemoryDatabase
from dataknobs_llm import EchoProvider, LLMConfig
from dataknobs_llm.conversations.storage import DataknobsConversationStorage
from dataknobs_llm.llm.providers.echo import ErrorResponse
from dataknobs_llm.prompts import ConfigPromptLibrary
from dataknobs_llm.prompts.builders import AsyncPromptBuilder


class ErrorTrackingMiddleware(Middleware):
    """Test middleware that records on_error calls."""

    def __init__(self) -> None:
        self.errors: list[tuple[Exception, str, BotContext]] = []
        self.after_message_calls: int = 0
        self.post_stream_calls: int = 0

    async def before_message(
        self, message: str, context: BotContext, **kwargs: Any
    ) -> None:
        pass

    async def after_message(
        self, response: str, context: BotContext, **kwargs: Any
    ) -> None:
        self.after_message_calls += 1

    async def post_stream(
        self, message: str, response: str, context: BotContext
    ) -> None:
        self.post_stream_calls += 1

    async def on_error(
        self, error: Exception, message: str, context: BotContext
    ) -> None:
        self.errors.append((error, message, context))


class HookErrorMiddleware(Middleware):
    """Test middleware that raises in a configurable hook.

    Args:
        fail_on: Which hook should raise (before_message, after_message,
            post_stream, on_error).
    """

    def __init__(self, fail_on: str) -> None:
        self._fail_on = fail_on

    async def before_message(
        self, message: str, context: BotContext, **kwargs: Any
    ) -> None:
        if self._fail_on == "before_message":
            raise RuntimeError("before_message failed")

    async def after_message(
        self, response: str, context: BotContext, **kwargs: Any
    ) -> None:
        if self._fail_on == "after_message":
            raise RuntimeError("after_message failed")

    async def post_stream(
        self, message: str, response: str, context: BotContext
    ) -> None:
        if self._fail_on == "post_stream":
            raise RuntimeError("post_stream failed")

    async def on_error(
        self, error: Exception, message: str, context: BotContext
    ) -> None:
        if self._fail_on == "on_error":
            raise RuntimeError("on_error failed")



def _make_provider() -> EchoProvider:
    """Create an EchoProvider with standard test config."""
    config = LLMConfig(
        provider="echo",
        model="echo-test",
        options={"echo_prefix": ""},
    )
    return EchoProvider(config)


def _make_prompt_builder() -> AsyncPromptBuilder:
    """Create a real AsyncPromptBuilder with a minimal prompt library."""
    library = ConfigPromptLibrary({
        "system": {
            "assistant": {
                "template": "You are a helpful assistant.",
            },
        },
    })
    return AsyncPromptBuilder(library=library)


def _make_bot(
    provider: EchoProvider,
    middleware: ErrorTrackingMiddleware | list[Middleware],
    *,
    reasoning_strategy: ReasoningStrategy | None = None,
) -> DynaBot:
    """Create a minimal DynaBot with real constructs."""
    mw_list = middleware if isinstance(middleware, list) else [middleware]
    storage = DataknobsConversationStorage(AsyncMemoryDatabase())
    bot = DynaBot(
        llm=provider,
        prompt_builder=_make_prompt_builder(),
        conversation_storage=storage,
        middleware=mw_list,
    )
    if reasoning_strategy is not None:
        bot.reasoning_strategy = reasoning_strategy
    return bot


def _make_error_provider() -> EchoProvider:
    """Create an EchoProvider that raises on complete()."""
    provider = _make_provider()
    provider.set_responses([ErrorResponse(ValueError("test LLM error"))])
    return provider


@pytest.fixture
def bot_context() -> BotContext:
    return BotContext(
        conversation_id="test-conv",
        client_id="test-client",
        user_id="test-user",
    )


@pytest.fixture
def middleware() -> ErrorTrackingMiddleware:
    return ErrorTrackingMiddleware()


@pytest.mark.asyncio
class TestChatOnError:
    """Tests for on_error middleware dispatch in chat()."""

    async def test_chat_calls_on_error_middleware_on_exception(
        self,
        bot_context: BotContext,
        middleware: ErrorTrackingMiddleware,
    ) -> None:
        """chat() must call on_error on all middleware when generation fails."""
        provider = _make_error_provider()
        bot = _make_bot(provider, middleware)

        with pytest.raises(ValueError, match="test LLM error"):
            await bot.chat("hello", bot_context)

        assert len(middleware.errors) == 1
        error, message, context = middleware.errors[0]
        assert isinstance(error, ValueError)
        assert str(error) == "test LLM error"
        assert message == "hello"
        assert context is bot_context

    async def test_chat_no_on_error_on_success(
        self,
        bot_context: BotContext,
        middleware: ErrorTrackingMiddleware,
    ) -> None:
        """on_error must NOT be called when chat() succeeds."""
        provider = _make_provider()
        bot = _make_bot(provider, middleware)

        result = await bot.chat("hello", bot_context)

        assert result  # Got a response
        assert len(middleware.errors) == 0
        assert middleware.after_message_calls == 1


@pytest.mark.asyncio
class TestGreetOnError:
    """Tests for on_error middleware dispatch in greet()."""

    async def test_greet_calls_on_error_middleware_on_exception(
        self,
        bot_context: BotContext,
        middleware: ErrorTrackingMiddleware,
    ) -> None:
        """greet() must call on_error on all middleware when strategy fails."""
        provider = _make_provider()
        greet_error = ValueError("test greet error")
        strategy = ErrorRaisingStrategy(greet_error)
        bot = _make_bot(provider, middleware, reasoning_strategy=strategy)

        with pytest.raises(ValueError, match="test greet error"):
            await bot.greet(bot_context)

        assert len(middleware.errors) == 1
        error, message, context = middleware.errors[0]
        assert error is greet_error
        assert message == ""  # greet has no user message
        assert context is bot_context

    async def test_greet_no_on_error_on_success(
        self,
        bot_context: BotContext,
        middleware: ErrorTrackingMiddleware,
    ) -> None:
        """on_error must NOT be called when greet() succeeds."""
        provider = _make_provider()
        bot = _make_bot(provider, middleware)

        # No reasoning strategy → returns None, no error
        result = await bot.greet(bot_context)

        assert result is None
        assert len(middleware.errors) == 0


@pytest.mark.asyncio
class TestStreamChatOnError:
    """Regression test: stream_chat() must continue to call on_error."""

    async def test_stream_chat_on_error_still_works(
        self,
        bot_context: BotContext,
        middleware: ErrorTrackingMiddleware,
    ) -> None:
        """stream_chat() must still call on_error after refactor to shared helper."""
        provider = _make_error_provider()
        bot = _make_bot(provider, middleware)

        with pytest.raises(ValueError, match="test LLM error"):
            async for _chunk in bot.stream_chat("hello", bot_context):
                pass  # pragma: no cover

        assert len(middleware.errors) == 1
        error, message, context = middleware.errors[0]
        assert isinstance(error, ValueError)
        assert message == "hello"
        assert context is bot_context


@pytest.mark.asyncio
class TestPreparationPhaseOnError:
    """Tests that on_error fires for errors during message preparation."""

    async def test_chat_on_error_fires_for_before_message_failure(
        self,
        bot_context: BotContext,
    ) -> None:
        """chat() calls on_error when before_message middleware raises."""
        tracker = ErrorTrackingMiddleware()
        failing_mw = HookErrorMiddleware("before_message")
        provider = _make_provider()
        bot = _make_bot(provider, [failing_mw, tracker])

        with pytest.raises(RuntimeError, match="before_message failed"):
            await bot.chat("hello", bot_context)

        assert len(tracker.errors) == 1
        error, message, _ = tracker.errors[0]
        assert isinstance(error, RuntimeError)
        assert message == "hello"

    async def test_stream_chat_on_error_fires_for_before_message_failure(
        self,
        bot_context: BotContext,
    ) -> None:
        """stream_chat() calls on_error when before_message middleware raises."""
        tracker = ErrorTrackingMiddleware()
        failing_mw = HookErrorMiddleware("before_message")
        provider = _make_provider()
        bot = _make_bot(provider, [failing_mw, tracker])

        with pytest.raises(RuntimeError, match="before_message failed"):
            async for _chunk in bot.stream_chat("hello", bot_context):
                pass  # pragma: no cover

        assert len(tracker.errors) == 1
        error, message, _ = tracker.errors[0]
        assert isinstance(error, RuntimeError)
        assert message == "hello"

    async def test_greet_on_error_fires_for_before_message_failure(
        self,
        bot_context: BotContext,
    ) -> None:
        """greet() calls on_error when before_message middleware raises."""
        tracker = ErrorTrackingMiddleware()
        failing_mw = HookErrorMiddleware("before_message")
        provider = _make_provider()
        strategy = ErrorRaisingStrategy()
        bot = _make_bot(
            provider, [failing_mw, tracker], reasoning_strategy=strategy
        )

        with pytest.raises(RuntimeError, match="before_message failed"):
            await bot.greet(bot_context)

        assert len(tracker.errors) == 1
        error, message, _ = tracker.errors[0]
        assert isinstance(error, RuntimeError)
        assert message == ""  # greet has no user message


@pytest.mark.asyncio
class TestMiddlewareIsolation:
    """Tests that one failing middleware doesn't prevent others from running."""

    async def test_on_error_continues_after_middleware_failure(
        self,
        bot_context: BotContext,
    ) -> None:
        """If one middleware's on_error raises, others still get called."""
        failing_mw = HookErrorMiddleware("on_error")
        tracker = ErrorTrackingMiddleware()
        provider = _make_error_provider()
        # Failing middleware is first — tracker should still get notified
        bot = _make_bot(provider, [failing_mw, tracker])

        with pytest.raises(ValueError, match="test LLM error"):
            await bot.chat("hello", bot_context)

        assert len(tracker.errors) == 1

    async def test_after_message_continues_after_middleware_failure(
        self,
        bot_context: BotContext,
    ) -> None:
        """If one middleware's after_message raises, others still get called."""
        failing_mw = HookErrorMiddleware("after_message")
        tracker = ErrorTrackingMiddleware()
        provider = _make_provider()
        bot = _make_bot(provider, [failing_mw, tracker])

        result = await bot.chat("hello", bot_context)

        assert result  # Response still returned despite middleware failure
        assert tracker.after_message_calls == 1
        assert len(tracker.errors) == 0  # No on_error — this isn't a request error

    async def test_post_stream_continues_after_middleware_failure(
        self,
        bot_context: BotContext,
    ) -> None:
        """If one middleware's post_stream raises, others still get called."""
        failing_mw = HookErrorMiddleware("post_stream")
        tracker = ErrorTrackingMiddleware()
        provider = _make_provider()
        bot = _make_bot(provider, [failing_mw, tracker])

        chunks: list[str] = []
        async for chunk in bot.stream_chat("hello", bot_context):
            chunks.append(chunk.delta)

        assert "".join(chunks)  # Response still streamed
        assert tracker.post_stream_calls == 1
        assert len(tracker.errors) == 0

    async def test_before_message_continues_after_middleware_failure(
        self,
        bot_context: BotContext,
    ) -> None:
        """If one middleware's before_message raises, others still get called.

        Two ErrorTrackingMiddleware instances sandwich a failing middleware.
        Both trackers should have their before_message called (verified via
        on_error — if before_message was skipped, the tracker wouldn't be
        in a valid state, but the error would still propagate).
        """
        tracker_before = ErrorTrackingMiddleware()
        failing_mw = HookErrorMiddleware("before_message")
        tracker_after = ErrorTrackingMiddleware()
        provider = _make_provider()
        bot = _make_bot(provider, [tracker_before, failing_mw, tracker_after])

        with pytest.raises(RuntimeError, match="before_message failed"):
            await bot.chat("hello", bot_context)

        # Both trackers get on_error — proving the error propagated
        # and all middleware were notified
        assert len(tracker_before.errors) == 1
        assert len(tracker_after.errors) == 1
