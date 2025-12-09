"""Logging middleware for conversation tracking."""

import json
import logging
from datetime import datetime, timezone
from typing import Any

from dataknobs_bots.bot.context import BotContext

from .base import Middleware

logger = logging.getLogger(__name__)


class LoggingMiddleware(Middleware):
    """Middleware for tracking conversation interactions.

    Logs all user messages and bot responses with context
    for monitoring, debugging, and analytics.

    Attributes:
        log_level: Logging level to use (default: INFO)
        include_metadata: Whether to include full context metadata
        json_format: Whether to output logs in JSON format

    Example:
        ```python
        # Basic usage
        middleware = LoggingMiddleware()

        # With JSON format for log aggregation
        middleware = LoggingMiddleware(
            log_level="INFO",
            include_metadata=True,
            json_format=True
        )
        ```
    """

    def __init__(
        self,
        log_level: str = "INFO",
        include_metadata: bool = True,
        json_format: bool = False,
    ):
        """Initialize logging middleware.

        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
            include_metadata: Whether to log full context metadata
            json_format: Whether to output in JSON format
        """
        self.log_level = log_level
        self.include_metadata = include_metadata
        self.json_format = json_format
        self._logger = logging.getLogger(f"{__name__}.ConversationLogger")
        self._logger.setLevel(getattr(logging, log_level.upper()))

    async def before_message(self, message: str, context: BotContext) -> None:
        """Called before processing user message.

        Args:
            message: User's input message
            context: Bot context with conversation and user info
        """
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": "user_message",
            "client_id": context.client_id,
            "user_id": context.user_id,
            "conversation_id": context.conversation_id,
            "message_length": len(message),
        }

        if self.include_metadata:
            log_data["session_metadata"] = context.session_metadata
            log_data["request_metadata"] = context.request_metadata

        if self.json_format:
            self._logger.info(json.dumps(log_data))
        else:
            self._logger.info(f"User message: {log_data}")

        # Log content at DEBUG level (first 200 chars)
        self._logger.debug(f"Message content: {message[:200]}...")

    async def after_message(
        self, response: str, context: BotContext, **kwargs: Any
    ) -> None:
        """Called after generating bot response.

        Args:
            response: Bot's generated response
            context: Bot context
            **kwargs: Additional data (e.g., tokens_used, response_time_ms)
        """
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": "bot_response",
            "client_id": context.client_id,
            "user_id": context.user_id,
            "conversation_id": context.conversation_id,
            "response_length": len(response),
        }

        # Add optional metrics
        if "tokens_used" in kwargs:
            log_data["tokens_used"] = kwargs["tokens_used"]
        if "response_time_ms" in kwargs:
            log_data["response_time_ms"] = kwargs["response_time_ms"]
        if "provider" in kwargs:
            log_data["provider"] = kwargs["provider"]
        if "model" in kwargs:
            log_data["model"] = kwargs["model"]

        if self.include_metadata:
            log_data["session_metadata"] = context.session_metadata
            log_data["request_metadata"] = context.request_metadata

        if self.json_format:
            self._logger.info(json.dumps(log_data))
        else:
            self._logger.info(f"Bot response: {log_data}")

        # Log content at DEBUG level (first 200 chars)
        self._logger.debug(f"Response content: {response[:200]}...")

    async def on_error(
        self, error: Exception, message: str, context: BotContext
    ) -> None:
        """Called when an error occurs during message processing.

        Args:
            error: The exception that occurred
            message: User message that caused the error
            context: Bot context
        """
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": "error",
            "client_id": context.client_id,
            "user_id": context.user_id,
            "conversation_id": context.conversation_id,
            "error_type": type(error).__name__,
            "error_message": str(error),
        }

        if self.json_format:
            self._logger.error(json.dumps(log_data), exc_info=error)
        else:
            self._logger.error(f"Error processing message: {log_data}", exc_info=error)
