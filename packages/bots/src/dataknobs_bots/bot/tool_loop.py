"""Delivery seam for the shared monolithic tool-execution loop.

``DynaBot`` runs the *same* cap / wall-clock-timeout / execute / budget /
LLM-re-call / cap-warning lifecycle in two non-phased delivery modes:
buffered (``chat``) and streaming (``stream_chat``).  Historically each mode
carried its own hand-written copy of that skeleton, so a loop-control change
had to be made — and could silently drift — in two places.

This module factors the skeleton into a single async-generator core
(``DynaBot._run_monolithic_tool_loop``) plus the ``_ToolLoopDelivery`` seam
defined here.  The core owns the control flow; a delivery object owns *only*
the axes on which the two modes genuinely differ:

* where the pending tool_calls come from (``has_pending`` / ``pending_calls``);
* how intermediate usage is accumulated (buffered vs streaming accounting);
* whether pending is cleared *before* the budget gate (streaming only);
* how the LLM is re-invoked (a buffered ``complete`` with a per-call deadline
  vs a streaming ``stream_complete`` whose chunks are yielded through);
* the exact warning strings emitted per mode.

Two asymmetries are **load-bearing behavior** and are preserved deliberately,
not tidied away:

* Streaming clears pending *before* the budget check, so a budget-break leaves
  ``has_pending()`` false and flags no orphan; buffered never clears, so its
  budget-break flags an orphan.  ``clear_pending_after_execute`` placement in
  the core encodes this.
* The buffered re-call wraps ``complete`` in ``asyncio.wait_for`` and breaks on
  a per-call timeout; the streaming re-stream has no per-call deadline (only the
  pre-stream budget gate).  ``_StreamingDelivery.recall`` ignores ``remaining``.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from dataknobs_llm import LLMStreamResponse

if TYPE_CHECKING:
    from .turn import TurnState

logger = logging.getLogger(__name__)


class _ToolLoopDelivery(ABC):
    """The per-mode seam driven by ``DynaBot._run_monolithic_tool_loop``.

    Subclasses supply the mode-specific warning strings (verbatim copies of the
    historical per-mode messages so log output is unchanged) and the divergent
    per-iteration behavior.  Everything else — the cap loop, the wall-clock
    guard, ``_execute_tools``, the budget gate, and the cap-hit warning — lives
    in the shared core.
    """

    #: Emitted when the wall-clock timeout trips before an iteration's work.
    MSG_TIMEOUT: str
    #: Emitted when the remaining budget is exhausted before the LLM re-call.
    MSG_BUDGET: str
    #: Emitted when the iteration cap is reached with pending tool_calls.
    MSG_CAP: str

    #: Buffered sets this ``True`` when its re-call exceeds the per-call
    #: deadline, signalling the core to break; streaming never sets it.
    broke: bool = False

    @abstractmethod
    def has_pending(self) -> bool:
        """Whether there are pending tool_calls to execute this iteration."""

    @abstractmethod
    def pending_calls(self) -> list[Any] | None:
        """The pending tool_calls to hand to ``_execute_tools``."""

    @abstractmethod
    def accumulate_usage(self, turn: TurnState) -> None:
        """Fold the just-completed round's token usage into ``turn``."""

    def clear_pending_after_execute(self) -> None:
        """Clear pending *before* the budget gate.

        Default is a no-op (buffered — pending is reassigned by ``recall``).
        Streaming overrides this to ``None`` its pending so a budget-break
        flags no orphan.
        """
        return

    @abstractmethod
    async def recall(
        self, turn: TurnState, remaining: float
    ) -> AsyncIterator[LLMStreamResponse] | None:
        """Re-invoke the LLM after tools ran.

        Returns an async iterator of chunks to yield through (streaming), or
        ``None`` when the re-call produced nothing to yield (buffered, whose
        ``complete`` result is captured on the delivery instead).
        """


class _BufferedDelivery(_ToolLoopDelivery):
    """Buffered (``chat``) delivery — one ``complete`` per re-call, deadlined."""

    MSG_TIMEOUT = "Tool execution loop exceeded wall-clock timeout (%.1fs)"
    MSG_BUDGET = "Tool loop budget exhausted before LLM re-call (%.1fs budget)"
    MSG_CAP = (
        "Tool execution loop reached max iterations (%d) with pending tool_calls"
    )
    MSG_RECALL_TIMEOUT = (
        "LLM re-call exceeded remaining tool loop budget "
        "(%.1fs remaining of %.1fs)"
    )

    def __init__(
        self,
        response: Any,
        *,
        recall_kwargs: dict[str, Any],
        turn_timeout: float,
    ) -> None:
        self.response = response
        self._recall_kwargs = recall_kwargs
        self._turn_timeout = turn_timeout

    def has_pending(self) -> bool:
        return bool(getattr(self.response, "tool_calls", None))

    def pending_calls(self) -> list[Any] | None:
        return self.response.tool_calls

    def accumulate_usage(self, turn: TurnState) -> None:
        turn.accumulate_usage(self.response)

    async def recall(
        self, turn: TurnState, remaining: float
    ) -> AsyncIterator[LLMStreamResponse] | None:
        try:
            self.response = await asyncio.wait_for(
                turn.manager.complete(**self._recall_kwargs),
                timeout=remaining,
            )
        except (TimeoutError, asyncio.TimeoutError):
            logger.warning(
                self.MSG_RECALL_TIMEOUT,
                remaining,
                self._turn_timeout,
                extra={
                    "conversation_id": getattr(
                        turn.manager, "conversation_id", None
                    ),
                },
            )
            self.broke = True
        return None


class _StreamingDelivery(_ToolLoopDelivery):
    """Streaming (``stream_chat``) delivery — chunks re-streamed through."""

    MSG_TIMEOUT = (
        "Streaming tool execution loop exceeded wall-clock timeout (%.1fs)"
    )
    MSG_BUDGET = (
        "Streaming tool loop budget exhausted before LLM re-stream "
        "(%.1fs budget)"
    )
    MSG_CAP = (
        "Streaming tool execution loop reached max iterations (%d) "
        "with pending tool_calls"
    )

    def __init__(
        self,
        pending: list[Any] | None,
        *,
        provider: Any,
        has_tools: bool,
        recall_kwargs: dict[str, Any],
    ) -> None:
        self.pending = pending
        self._provider = provider
        # ``has_tools`` snapshots ``bool(bot.tool_registry)`` at construction.
        # The tool registry is turn-stable (never mutated mid-turn), so this
        # is equivalent to the historical live ``self.tool_registry`` read in
        # the inline re-stream.
        self._has_tools = has_tools
        self._recall_kwargs = recall_kwargs

    def has_pending(self) -> bool:
        return bool(self.pending)

    def pending_calls(self) -> list[Any] | None:
        return self.pending

    def accumulate_usage(self, turn: TurnState) -> None:
        turn.accumulate_usage_from_stream()

    def clear_pending_after_execute(self) -> None:
        self.pending = None

    async def recall(
        self, turn: TurnState, remaining: float
    ) -> AsyncIterator[LLMStreamResponse] | None:
        # ``remaining`` is intentionally unused: the streaming re-stream has no
        # per-call deadline, only the pre-stream budget gate in the core.
        return self._restream(turn)

    async def _restream(
        self, turn: TurnState
    ) -> AsyncIterator[LLMStreamResponse]:
        async for chunk in turn.manager.stream_complete(**self._recall_kwargs):
            turn.stream_chunks.append(chunk.delta)
            if chunk.is_final or chunk.usage:
                turn.populate_from_final_stream_chunk(chunk, self._provider)
            if chunk.tool_calls and self._has_tools:
                self.pending = chunk.tool_calls
                yield LLMStreamResponse(
                    delta=chunk.delta,
                    is_final=False,
                    usage=chunk.usage,
                    model=chunk.model,
                )
            else:
                yield chunk
