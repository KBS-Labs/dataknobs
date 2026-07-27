"""Behavioral guards for the unified monolithic tool-execution loop.

``chat()`` (buffered) and ``stream_chat()`` (streaming) share one
cap / wall-clock-timeout / execute / budget / re-call / cap-warning lifecycle,
driven by ``DynaBot._run_monolithic_tool_loop`` with a per-mode delivery
(``bot/tool_loop.py``).  The 24 tests in ``test_tool_execution_loop.py`` pin the
happy-path behavior of both modes and must pass unchanged; the tests here pin
the properties that make the *unification* safe and lock the deliberate
per-mode asymmetries so a future "cleanup" cannot silently erase them.

Real constructs only — ``BotTestHarness`` drives a real ``DynaBot`` +
``EchoProvider``; ``set_response_delay`` simulates provider latency. No mocks.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any

import pytest

from dataknobs_bots.bot.turn import TurnState
from dataknobs_bots.middleware.base import Middleware
from dataknobs_bots.testing import BotTestHarness
from dataknobs_llm.llm.base import LLMMessage
from dataknobs_llm.testing import text_response, tool_call_response
from dataknobs_llm.tools.base import Tool

pytestmark = pytest.mark.asyncio


_SIMPLE_BOT_CONFIG: dict[str, Any] = {
    "llm": {"provider": "echo", "model": "test"},
    "conversation_storage": {"backend": "memory"},
    "reasoning": {"strategy": "simple"},
}


def _config(**overrides: Any) -> dict[str, Any]:
    return {**_SIMPLE_BOT_CONFIG, **overrides}


class _TurnTracker(Middleware):
    """Captures the finalized ``TurnState`` (fires in ``_finalize_turn``)."""

    def __init__(self) -> None:
        self.turns: list[TurnState] = []

    async def after_turn(self, turn: TurnState) -> None:
        self.turns.append(turn)


class _EchoTool(Tool):
    """Echo tool with an optional per-execute sleep to burn loop budget."""

    def __init__(self, *, sleep: float = 0.0) -> None:
        super().__init__(name="echo", description="Echoes the input back")
        self._sleep = sleep
        self.calls = 0

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"text": {"type": "string"}},
        }

    async def execute(self, **kwargs: Any) -> Any:
        kwargs.pop("_context", None)
        self.calls += 1
        if self._sleep:
            await asyncio.sleep(self._sleep)
        return {"echoed": kwargs.get("text", "")}


class _CacheDropTool(Tool):
    """Tool that evicts its own conversation from the manager cache mid-loop.

    Simulates a bounded-cache eviction (``max_cached_conversations``) or an
    explicit ``clear_conversation`` landing while a turn is in flight. The
    unified loop drives the *held* ``turn.manager`` reference and never
    re-fetches it from the cache, so the turn must survive the eviction.
    """

    def __init__(self) -> None:
        super().__init__(name="drop", description="drops the conv cache")
        self.bot: Any = None
        self.conv_id: str | None = None
        self.captured_manager: Any = None
        self.dropped = False

    @property
    def schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs: Any) -> Any:
        kwargs.pop("_context", None)
        if self.bot is not None and self.conv_id is not None and not self.dropped:
            # Capture the cached manager, then evict it out from under the
            # in-flight turn.
            self.captured_manager = self.bot._conversation_managers.get(
                self.conv_id
            )
            self.bot._drop_conversation_cache(self.conv_id)
            self.dropped = True
        return {"ok": True}


def _tool_history_delay(delay: float) -> Callable[[list[LLMMessage]], float]:
    """Delay only the re-call/re-stream (calls whose history has tool usage).

    The initial generation sees only the user message; the post-tool re-call
    carries an assistant ``tool_calls`` message (and a ``role == "tool"``
    result), so keying on either marker hits the re-invocation and nothing
    else.
    """

    def _fn(messages: list[LLMMessage]) -> float:
        has_tool_history = any(
            getattr(m, "tool_calls", None) or getattr(m, "role", None) == "tool"
            for m in messages
        )
        return delay if has_tool_history else 0.0

    return _fn


# ---------------------------------------------------------------------------
# T-EVICT — the held ``turn.manager`` reference survives cache eviction.
# ---------------------------------------------------------------------------


class TestHeldManagerSurvivesEviction:
    """The loop must not re-fetch the manager from the cache each iteration.

    Guards the property FU-cache-bounding relies on: an eviction reclaims a
    *cache slot*, never pulls the manager out from under an in-flight turn.
    Passes today (the loop holds ``turn.manager``); would fail if a future edit
    re-fetched the manager from ``_conversation_managers`` mid-loop.
    """

    async def test_buffered_completes_after_midloop_eviction(self) -> None:
        tool = _CacheDropTool()
        tracker = _TurnTracker()
        async with await BotTestHarness.create(
            bot_config=_SIMPLE_BOT_CONFIG,
            main_responses=[
                tool_call_response("drop", {}),
                text_response("done"),
            ],
            tools=[tool],
            middleware=[tracker],
        ) as harness:
            tool.bot = harness.bot
            tool.conv_id = harness.context.conversation_id

            result = await harness.chat("evict me")

        # (a) turn completed without KeyError/AttributeError.
        assert result.response == "done"
        # eviction really landed mid-loop.
        assert tool.dropped is True
        assert tool.conv_id not in harness.bot._conversation_managers
        # (b) the re-call targeted the same held manager instance.
        assert tracker.turns[0].manager is tool.captured_manager
        assert tool.captured_manager is not None

    async def test_streaming_completes_after_midloop_eviction(self) -> None:
        tool = _CacheDropTool()
        tracker = _TurnTracker()
        async with await BotTestHarness.create(
            bot_config=_SIMPLE_BOT_CONFIG,
            main_responses=[
                tool_call_response("drop", {}),
                text_response("done"),
            ],
            tools=[tool],
            middleware=[tracker],
        ) as harness:
            tool.bot = harness.bot
            tool.conv_id = harness.context.conversation_id

            result = await harness.stream_chat("evict me")

        assert "done" in result.response
        assert tool.dropped is True
        assert tool.conv_id not in harness.bot._conversation_managers
        assert tracker.turns[0].manager is tool.captured_manager
        assert tool.captured_manager is not None


# ---------------------------------------------------------------------------
# T-PARITY-BUDGET — budget-break orphan flag differs by mode (load-bearing).
# ---------------------------------------------------------------------------


class TestBudgetBreakOrphanFlagAsymmetry:
    """Buffered budget-break flags an orphan; streaming does not.

    Streaming clears pending *before* the budget gate, so a budget-break leaves
    no pending → ``tool_loop_left_pending_call`` False. Buffered never clears
    → True. This asymmetry is behavior, not mess; lock it so a future
    unification cannot silently converge the two.
    """

    async def test_buffered_budget_break_flags_pending(self) -> None:
        tracker = _TurnTracker()
        async with await BotTestHarness.create(
            # Tiny budget + a tool that sleeps past it → the re-call budget
            # gate trips after the first execute (before any re-call).
            bot_config=_config(tool_loop_timeout=0.02),
            main_responses=[
                tool_call_response("echo", {"text": "hi"}),
                text_response("unreached"),
            ],
            tools=[_EchoTool(sleep=0.1)],
            middleware=[tracker],
        ) as harness:
            await harness.chat("burn the budget")

        assert tracker.turns[0].tool_loop_left_pending_call is True

    async def test_streaming_budget_break_flags_no_pending(self) -> None:
        tracker = _TurnTracker()
        async with await BotTestHarness.create(
            bot_config=_config(tool_loop_timeout=0.02),
            main_responses=[
                tool_call_response("echo", {"text": "hi"}),
                text_response("unreached"),
            ],
            tools=[_EchoTool(sleep=0.1)],
            middleware=[tracker],
        ) as harness:
            await harness.stream_chat("burn the budget")

        assert tracker.turns[0].tool_loop_left_pending_call is False


# ---------------------------------------------------------------------------
# T-PARITY-TIMEOUT — buffered re-call is deadlined; streaming re-stream is not.
# ---------------------------------------------------------------------------


class TestRecallDeadlineAsymmetry:
    """Buffered wraps the re-call in ``wait_for``; streaming re-stream is not.

    The buffered re-call has a per-call wall-clock deadline (``remaining``) and
    breaks on timeout; the streaming re-stream has only the pre-stream budget
    gate (no per-call deadline). Lock both halves so the deliberate asymmetry
    is visible and preserved.
    """

    # The injected re-call delay sits comfortably above the loop budget so the
    # buffered ``wait_for`` trips; the budget (0.3s) is in turn far above the
    # near-zero time a no-op echo-tool execute consumes, so the pre-recall
    # budget gate can never spuriously pre-empt the intended path even under
    # heavy CI load.
    _DELAY = 0.6
    _BUDGET = 0.3

    async def test_buffered_recall_timeout_is_bounded(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        tracker = _TurnTracker()
        async with await BotTestHarness.create(
            # Budget < injected re-call delay → the re-call's wait_for trips
            # and the loop breaks with the pre-recall response retained.
            bot_config=_config(tool_loop_timeout=self._BUDGET),
            main_responses=[
                tool_call_response("echo", {"text": "hi"}),
                text_response("real answer"),
            ],
            tools=[_EchoTool()],
            middleware=[tracker],
        ) as harness:
            harness.bot.llm.set_response_delay(_tool_history_delay(self._DELAY))
            with caplog.at_level(logging.WARNING):
                await harness.chat("use the tool")

        # The buffered re-call deadline fired.
        assert "exceeded remaining tool loop budget" in caplog.text
        # Pre-recall (tool_call) response retained → flagged as an orphan.
        assert tracker.turns[0].tool_loop_left_pending_call is True

    async def test_streaming_restream_has_no_deadline(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        async with await BotTestHarness.create(
            bot_config=_config(tool_loop_timeout=self._BUDGET),
            main_responses=[
                tool_call_response("echo", {"text": "hi"}),
                text_response("real answer"),
            ],
            tools=[_EchoTool()],
        ) as harness:
            harness.bot.llm.set_response_delay(_tool_history_delay(self._DELAY))
            with caplog.at_level(logging.WARNING):
                result = await harness.stream_chat("use the tool")

        # No per-call deadline: the slow re-stream runs to completion and the
        # real answer is delivered — the recall-timeout warning never fires.
        assert "real answer" in result.response
        assert "exceeded remaining tool loop budget" not in caplog.text


# ---------------------------------------------------------------------------
# T-CAP-WARN — both modes emit their exact (verbatim) cap-hit warning.
# ---------------------------------------------------------------------------


class TestCapHitWarnings:
    """Verifies the per-mode cap-hit strings survived the verbatim copy."""

    async def test_buffered_cap_hit_warns(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        async with await BotTestHarness.create(
            bot_config=_config(max_tool_iterations=2),
            main_responses=[
                tool_call_response("echo", {"text": f"iter{i}"})
                for i in range(4)
            ],
            tools=[_EchoTool()],
        ) as harness:
            with caplog.at_level(logging.WARNING):
                await harness.chat("loop forever")

        assert (
            "Tool execution loop reached max iterations" in caplog.text
        )

    async def test_streaming_cap_hit_warns(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        async with await BotTestHarness.create(
            bot_config=_config(max_tool_iterations=2),
            main_responses=[
                tool_call_response("echo", {"text": f"iter{i}"})
                for i in range(4)
            ],
            tools=[_EchoTool()],
        ) as harness:
            with caplog.at_level(logging.WARNING):
                await harness.stream_chat("loop forever")

        assert (
            "Streaming tool execution loop reached max iterations"
            in caplog.text
        )
