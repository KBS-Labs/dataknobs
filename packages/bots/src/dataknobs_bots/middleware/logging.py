"""Logging middleware for conversation tracking."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from .base import Middleware

if TYPE_CHECKING:
    from dataknobs_bots.bot.context import BotContext
    from dataknobs_bots.bot.turn import TurnState

logger = logging.getLogger(__name__)


class LoggingMiddleware(Middleware):
    """Middleware for tracking conversation interactions.

    Logs all user messages and bot responses with context
    for monitoring, debugging, and analytics.

    Uses the unified TurnState hooks:

    - ``on_turn_start`` — logs incoming user message
    - ``after_turn`` — logs turn completion with response, usage, tools

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

    async def on_turn_start(self, turn: TurnState) -> str | None:
        """Log incoming user message at the start of a turn.

        Args:
            turn: Turn state at the start of the pipeline.

        Returns:
            None (no message transform).
        """
        log_data: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": "user_message",
            "mode": turn.mode.value,
            "client_id": turn.context.client_id,
            "user_id": turn.context.user_id,
            "conversation_id": turn.context.conversation_id,
            "message_length": len(turn.message),
        }

        if self.include_metadata:
            log_data["session_metadata"] = turn.context.session_metadata
            log_data["request_metadata"] = turn.context.request_metadata

        if self.json_format:
            self._logger.info(json.dumps(log_data))
        else:
            self._logger.info("User message: %s", log_data)

        # Log content at DEBUG level (first 200 chars)
        self._logger.debug("Message content: %.200s...", turn.message)
        return None

    async def after_turn(self, turn: TurnState) -> None:
        """Log turn completion with unified data for all turn types.

        Args:
            turn: Completed turn state.
        """
        log_data: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": "turn_complete",
            "mode": turn.mode.value,
            "client_id": turn.context.client_id,
            "user_id": turn.context.user_id,
            "conversation_id": turn.context.conversation_id,
            "response_length": len(turn.response_content),
        }

        if turn.usage:
            log_data["tokens_used"] = turn.usage
        if turn.provider_name:
            log_data["provider"] = turn.provider_name
        if turn.model:
            log_data["model"] = turn.model
        if turn.tool_executions:
            log_data["tool_executions"] = len(turn.tool_executions)

        if self.include_metadata:
            log_data["session_metadata"] = turn.context.session_metadata
            log_data["request_metadata"] = turn.context.request_metadata

        if self.json_format:
            self._logger.info(json.dumps(log_data))
        else:
            self._logger.info("Turn complete: %s", log_data)

        # Log content at DEBUG level (first 200 chars)
        self._logger.debug("Response content: %.200s...", turn.response_content)

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
            self._logger.error(
                "Error processing message: %s", log_data, exc_info=error
            )

    async def on_hook_error(
        self, hook_name: str, error: Exception, context: BotContext
    ) -> None:
        """Called when a middleware hook itself raises.

        Args:
            hook_name: Name of the hook that failed
            error: The exception raised by the middleware hook
            context: Bot context
        """
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": "hook_error",
            "hook_name": hook_name,
            "client_id": context.client_id,
            "user_id": context.user_id,
            "conversation_id": context.conversation_id,
            "error_type": type(error).__name__,
            "error_message": str(error),
        }

        if self.json_format:
            self._logger.warning(json.dumps(log_data), exc_info=error)
        else:
            self._logger.warning(
                "Middleware hook %s failed: %s", hook_name, log_data, exc_info=error
            )
