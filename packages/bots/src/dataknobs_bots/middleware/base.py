"""Base middleware interface for bot request/response lifecycle."""

from abc import ABC, abstractmethod
from typing import Any

from dataknobs_bots.bot.context import BotContext


class Middleware(ABC):
    """Abstract base class for bot middleware.

    Middleware provides hooks into the bot request/response lifecycle:
    - before_message: Called before processing user message
    - after_message: Called after generating bot response
    - on_error: Called when an error occurs

    Example:
        ```python
        class MyMiddleware(Middleware):
            async def before_message(self, message: str, context: BotContext) -> None:
                print(f"Processing: {message}")

            async def after_message(
                self, response: str, context: BotContext, **kwargs: Any
            ) -> None:
                print(f"Response: {response}")

            async def on_error(
                self, error: Exception, message: str, context: BotContext
            ) -> None:
                print(f"Error: {error}")
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
        """Called after generating bot response.

        Args:
            response: Bot's generated response
            context: Bot context
            **kwargs: Additional data (e.g., tokens_used, response_time_ms, provider, model)
        """
        ...

    @abstractmethod
    async def on_error(
        self, error: Exception, message: str, context: BotContext
    ) -> None:
        """Called when an error occurs during message processing.

        Args:
            error: The exception that occurred
            message: User message that caused the error
            context: Bot context
        """
        ...
