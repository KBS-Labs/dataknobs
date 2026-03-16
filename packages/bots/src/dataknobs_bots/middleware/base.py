"""Base middleware interface for bot request/response lifecycle."""

from abc import ABC, abstractmethod
from typing import Any

from dataknobs_bots.bot.context import BotContext


class Middleware(ABC):
    """Abstract base class for bot middleware.

    Middleware provides hooks into the bot request/response lifecycle:
    - before_message: Called before processing user message
    - after_message: Called after generating bot response (non-streaming)
    - post_stream: Called after streaming response completes
    - on_error: Called when a request-level error occurs (preparation or generation failed)
    - on_hook_error: Called when a middleware hook itself fails (after_message, post_stream, etc.)

    Error semantics:
        ``on_error`` fires when the bot request fails — the caller does NOT
        receive a response.  ``on_hook_error`` fires when a middleware's own
        hook raises after the request already succeeded — the caller DID
        receive a response, but a middleware could not complete its
        post-processing.  This distinction lets middleware differentiate
        between "the request failed" and "observability/post-processing
        broke."

    Example:
        ```python
        class MyMiddleware(Middleware):
            async def before_message(self, message: str, context: BotContext) -> None:
                ...

            async def after_message(
                self, response: str, context: BotContext, **kwargs: Any
            ) -> None:
                ...

            async def post_stream(
                self, message: str, response: str, context: BotContext
            ) -> None:
                ...

            async def on_error(
                self, error: Exception, message: str, context: BotContext
            ) -> None:
                ...  # Request failed

            async def on_hook_error(
                self, hook_name: str, error: Exception, context: BotContext
            ) -> None:
                ...  # A middleware hook failed
        ```
    """

    @abstractmethod
    async def before_message(self, message: str, context: BotContext) -> None:
        """Called before processing user message.

        Args:
            message: User's input message
            context: Bot context with conversation and user info
        """
        ...

    @abstractmethod
    async def after_message(
        self, response: str, context: BotContext, **kwargs: Any
    ) -> None:
        """Called after generating bot response (non-streaming).

        Args:
            response: Bot's generated response
            context: Bot context
            **kwargs: Additional data (e.g., tokens_used, response_time_ms, provider, model)
        """
        ...

    @abstractmethod
    async def post_stream(
        self, message: str, response: str, context: BotContext
    ) -> None:
        """Called after streaming response completes.

        This hook is called after stream_chat() finishes streaming all chunks.
        It provides both the original user message and the complete accumulated
        response, useful for logging, analytics, or post-processing.

        Args:
            message: Original user message that triggered the stream
            response: Complete accumulated response from streaming
            context: Bot context
        """
        ...

    @abstractmethod
    async def on_error(
        self, error: Exception, message: str, context: BotContext
    ) -> None:
        """Called when a request-level error occurs during message processing.

        This hook fires when the bot request fails (preparation, generation,
        or memory/middleware post-processing in ``before_message``). The
        caller does NOT receive a response.

        Args:
            error: The exception that occurred
            message: User message that caused the error
            context: Bot context
        """
        ...

    @abstractmethod
    async def on_hook_error(
        self, hook_name: str, error: Exception, context: BotContext
    ) -> None:
        """Called when a middleware hook itself raises an exception.

        This fires when a middleware's ``after_message``, ``post_stream``,
        or ``on_error`` hook raises.  Unlike ``on_error``, this does NOT
        mean the request failed — the response was already delivered to the
        caller.  It means a middleware could not complete its own
        post-processing (e.g., a logging sink was unreachable, a metrics
        backend timed out).

        Args:
            hook_name: Name of the hook that failed (e.g. ``"after_message"``,
                ``"post_stream"``, ``"on_error"``)
            error: The exception raised by the middleware hook
            context: Bot context
        """
        ...
