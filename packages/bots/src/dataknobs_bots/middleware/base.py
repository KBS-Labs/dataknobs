"""Base middleware interface for bot request/response lifecycle."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dataknobs_bots.bot.context import BotContext

if TYPE_CHECKING:
    from dataknobs_bots.bot.turn import ToolExecution, TurnState


class Middleware:
    """Base class for bot middleware.

    Middleware provides hooks into the bot request/response lifecycle.
    All hooks are concrete no-ops — subclasses override only the hooks
    they need.

    **Preferred hooks** (receive full ``TurnState``):

    - ``on_turn_start(turn)`` — before processing; can write
      ``plugin_data`` and optionally transform the message.
    - ``after_turn(turn)`` — after any turn completes (chat, stream,
      greet); unified successor to ``after_message`` and ``post_stream``.
    - ``on_tool_executed(execution, context)`` — after each tool call.

    **Legacy hooks** (kept for backward compatibility):

    - ``before_message(message, context)`` — use ``on_turn_start``
      instead.
    - ``after_message(response, context, **kwargs)`` — use
      ``after_turn`` instead.
    - ``post_stream(message, response, context)`` — use ``after_turn``
      instead.

    **Error hooks** (no TurnState equivalent — still primary):

    - ``on_error(error, message, context)`` — request failed.
    - ``on_hook_error(hook_name, error, context)`` — a hook failed.

    Error semantics:
        ``on_error`` fires when the bot request fails — the caller does NOT
        receive a response.  ``on_hook_error`` fires when a middleware's own
        hook raises after the request already succeeded — the caller DID
        receive a response, but a middleware could not complete its
        post-processing.

    Example:
        ```python
        class MyMiddleware(Middleware):
            async def on_turn_start(self, turn):
                turn.plugin_data["started"] = True
                return None  # or return transformed message

            async def after_turn(self, turn):
                log.info("Turn %s done", turn.mode.value)

            async def on_error(self, error, message, context):
                log.error("Request failed: %s", error)
        ```
    """

    # --- Legacy hooks (concrete no-ops, kept for backward compat) ---

    async def before_message(
        self, message: str, context: BotContext
    ) -> None:
        """Called before processing user message.

        .. deprecated::
            Use ``on_turn_start`` instead, which provides the full
            ``TurnState`` including ``plugin_data`` for cross-middleware
            communication and supports message transforms.

        Args:
            message: User's input message
            context: Bot context with conversation and user info
        """

    async def after_message(
        self, response: str, context: BotContext, **kwargs: Any
    ) -> None:
        """Called after generating bot response (non-streaming).

        .. deprecated::
            Use ``after_turn`` instead, which fires for all turn types
            (chat, stream, greet) and provides the full ``TurnState``
            with usage data, tool executions, and plugin data.

        Args:
            response: Bot's generated response
            context: Bot context
            **kwargs: Additional data (e.g., tokens_used, response_time_ms, provider, model)
        """

    async def post_stream(
        self, message: str, response: str, context: BotContext
    ) -> None:
        """Called after streaming response completes.

        .. deprecated::
            Use ``after_turn`` instead, which fires for all turn types
            and provides real token usage data from the provider.

        Args:
            message: Original user message that triggered the stream
            response: Complete accumulated response from streaming
            context: Bot context
        """

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

    # --- Unified hooks (preferred) ---

    async def on_turn_start(
        self, turn: TurnState
    ) -> str | None:
        """Called at the start of every turn, before message processing.

        Receives the full ``TurnState`` including ``plugin_data`` for
        cross-middleware communication. Middleware can:

        - Write to ``turn.plugin_data`` to share data with downstream
          pipeline participants (LLM middleware, tools, ``after_turn``).
        - Return a transformed message string to replace ``turn.message``
          before it reaches the LLM (e.g., PII stripping, attack
          sanitization). Transforms chain: each middleware receives the
          message as modified by the previous one.
        - Return ``None`` to leave the message unchanged.

        Args:
            turn: Turn state at the start of the pipeline.

        Returns:
            Transformed message string, or ``None`` to keep the original.
        """
        return None

    async def after_turn(self, turn: TurnState) -> None:
        """Called after any turn completes (chat, stream, or greet).

        Provides the full ``TurnState`` with usage data, tool executions,
        and response content regardless of how the turn was initiated.
        This is the unified successor to ``after_message`` and
        ``post_stream`` — implement this for uniform post-turn handling.

        The legacy hooks (``after_message`` / ``post_stream``) continue
        to fire as well, so existing middleware is unaffected.

        Args:
            turn: Complete turn state with all pipeline data.
        """

    async def on_tool_executed(
        self, execution: ToolExecution, context: BotContext
    ) -> None:
        """Called after each tool execution within a turn.

        Fired once per tool invocation, before ``after_turn``. Useful
        for tool usage auditing, cost tracking, or logging.

        Args:
            execution: Record of the tool execution (name, params, result,
                error, duration).
            context: Bot context for the current turn.
        """
