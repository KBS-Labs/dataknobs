"""Tests for DynaBot on_error middleware dispatch across all entry points.

Verifies that chat(), greet(), and stream_chat() all call on_error middleware
when exceptions occur during message processing.

Bug: chat() and greet() had no error handling — exceptions propagated directly
to the caller without calling on_error middleware. Only stream_chat() called
on_error. This test file reproduces the bug and verifies the fix.
"""

from collections.abc import AsyncIterator
from typing import Any

import pytest
from dataknobs_bots.bot.base import DynaBot
from dataknobs_bots.bot.context import BotContext
from dataknobs_bots.middleware.base import Middleware
from dataknobs_bots.reasoning.base import ReasoningManagerProtocol, ReasoningStrategy
from dataknobs_data.backends.memory import AsyncMemoryDatabase
from dataknobs_llm import EchoProvider, LLMConfig, LLMResponse
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


class ErrorRaisingStrategy(ReasoningStrategy):
    """Test strategy that raises on generate() and greet()."""

    async def generate(
        self,
        manager: ReasoningManagerProtocol,
        llm: Any,
        tools: list[Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        raise ValueError("test strategy error")

    async def stream_generate(
        self,
        manager: ReasoningManagerProtocol,
        llm: Any,
        tools: list[Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[LLMResponse]:
        raise ValueError("test strategy error")
        yield  # Make it an async generator  # pragma: no cover

    async def greet(
        self,
        manager: ReasoningManagerProtocol,
        llm: Any,
        *,
        initial_context: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any | None:
        raise ValueError("test greet error")


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
    middleware: ErrorTrackingMiddleware,
    *,
    reasoning_strategy: ReasoningStrategy | None = None,
) -> DynaBot:
    """Create a minimal DynaBot with real constructs."""
    storage = DataknobsConversationStorage(AsyncMemoryDatabase())
    bot = DynaBot(
        llm=provider,
        prompt_builder=_make_prompt_builder(),
        conversation_storage=storage,
        middleware=[middleware],
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

    async def test_chat_reraises_after_on_error(
        self,
        bot_context: BotContext,
        middleware: ErrorTrackingMiddleware,
    ) -> None:
        """chat() must re-raise the exception after calling on_error."""
        provider = _make_error_provider()
        bot = _make_bot(provider, middleware)

        with pytest.raises(ValueError, match="test LLM error"):
            await bot.chat("hello", bot_context)

        # on_error was called AND the exception propagated
        assert len(middleware.errors) == 1

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
        strategy = ErrorRaisingStrategy()
        bot = _make_bot(provider, middleware, reasoning_strategy=strategy)

        with pytest.raises(ValueError, match="test greet error"):
            await bot.greet(bot_context)

        assert len(middleware.errors) == 1
        error, message, context = middleware.errors[0]
        assert isinstance(error, ValueError)
        assert str(error) == "test greet error"
        assert message == ""  # greet has no user message
        assert context is bot_context

    async def test_greet_reraises_after_on_error(
        self,
        bot_context: BotContext,
        middleware: ErrorTrackingMiddleware,
    ) -> None:
        """greet() must re-raise the exception after calling on_error."""
        provider = _make_provider()
        strategy = ErrorRaisingStrategy()
        bot = _make_bot(provider, middleware, reasoning_strategy=strategy)

        with pytest.raises(ValueError, match="test greet error"):
            await bot.greet(bot_context)

        assert len(middleware.errors) == 1

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
