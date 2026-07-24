"""Bound the phased terminal synthesis by the remaining tool-loop budget.

A phased reasoning strategy (ReAct) that terminates its tool loop abnormally
(max iterations / duplicate break / loop-timeout break) falls through to a
terminal synthesis call (``finalize_turn`` / ``stream_finalize_turn``). That
synthesis was historically **unbounded**: the turn's wall-clock became
``tool_loop_timeout`` + an unbounded synthesis. These tests bound the synthesis
by the budget the loop left unspent (floored at ``_MIN_FINALIZE_BUDGET``) and
degrade gracefully on timeout.

Reproduce-first: each timing test FAILS on the pre-fix code (turn runs for the
full injected synthesis delay) and PASSES after the fix (turn ≈ the floored
budget, graceful fallback surfaced). Real constructs only — ``BotTestHarness``
drives a real ``DynaBot`` and ``EchoProvider`` simulates a slow provider via
``set_response_delay``; no mocks.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from typing import Any

import pytest

from dataknobs_bots.bot.base import (
    _FINALIZE_TIMEOUT_REASON,
    _MIN_FINALIZE_BUDGET,
)
from dataknobs_bots.bot.config import _DEFAULT_TOOL_LOOP_TIMEOUT_MESSAGE
from dataknobs_bots.bot.turn import TurnMode, TurnState
from dataknobs_bots.testing import BotTestHarness
from dataknobs_llm.llm.base import LLMMessage
from dataknobs_llm.testing import text_response, tool_call_response
from dataknobs_llm.tools import Tool

pytestmark = pytest.mark.asyncio


# A synthesis delay long enough that, if the finalize were unbounded, the turn
# would obviously exceed the floored budget. Small enough to keep the suite fast
# once bounded (the turn stops at ~_MIN_FINALIZE_BUDGET, not this value).
_SLOW_SYNTHESIS_DELAY = 5.0

# A generous ceiling: the bounded turn should finish well under this. Comfortably
# below _SLOW_SYNTHESIS_DELAY so a regression (unbounded finalize) trips it.
_BOUNDED_CEILING = _MIN_FINALIZE_BUDGET + 2.0


class _EchoTool(Tool):
    """Trivial tool so ReAct enters its phased tool loop."""

    def __init__(self) -> None:
        super().__init__(name="echo_tool", description="echoes its input")

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"text": {"type": "string"}},
        }

    async def execute(self, text: str = "", **kwargs: Any) -> str:
        return f"echoed:{text}"


def _synthesis_delay(delay: float) -> Callable[[list[LLMMessage]], float]:
    """Return a delay predicate that sleeps only on the synthesis call.

    At ``tool_loop_timeout == 0`` the phased loop breaks at the wall-clock guard
    *before* executing the tool, leaving an orphan assistant ``tool_use`` in
    history; ReAct's ``finalize_turn`` pairs it with a synthetic ``tool_result``
    for the synthesis re-call. So the synthesis call's messages carry an
    assistant message with ``tool_calls`` (and, once paired, a ``role ==
    "tool"`` message), whereas the loop's first ``process_input`` call sees only
    the user message. Keying on that history — an OR over either marker — makes
    the injected latency hit the synthesis and nothing else.
    """

    def _fn(messages: list[LLMMessage]) -> float:
        is_synthesis = any(
            getattr(m, "tool_calls", None)
            or getattr(m, "role", None) == "tool"
            for m in messages
        )
        return delay if is_synthesis else 0.0

    return _fn


def _react_config(
    *,
    tool_loop_timeout: float,
    tool_loop_timeout_message: str | None = None,
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "llm": {"provider": "echo", "model": "echo-model"},
        "conversation_storage": {"backend": "memory"},
        "reasoning": {"strategy": "react", "max_iterations": 3},
        "tool_loop_timeout": tool_loop_timeout,
    }
    if tool_loop_timeout_message is not None:
        config["tool_loop_timeout_message"] = tool_loop_timeout_message
    return config


async def test_phased_finalize_bounded_buffered(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Buffered chat(): a slow synthesis is bounded, not run to completion."""
    async with await BotTestHarness.create(
        bot_config=_react_config(tool_loop_timeout=0.0),
        main_responses=[
            tool_call_response("echo_tool", {"text": "hi"}),
            text_response("real synthesized answer"),
        ],
        tools=[_EchoTool()],
    ) as harness:
        harness.bot.llm.set_response_delay(
            _synthesis_delay(_SLOW_SYNTHESIS_DELAY)
        )

        with caplog.at_level(logging.WARNING):
            start = time.monotonic()
            result = await harness.chat("please use the tool")
            elapsed = time.monotonic() - start

    # Bounded: nowhere near the 5s synthesis delay.
    assert elapsed < _BOUNDED_CEILING, (
        f"finalize was not bounded: {elapsed:.2f}s"
    )
    # Degraded to the fallback, not the real (slow) synthesis text.
    assert result.response == _DEFAULT_TOOL_LOOP_TIMEOUT_MESSAGE
    assert "exceeded remaining tool loop budget" in caplog.text


async def test_phased_finalize_bounded_streaming(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Streaming stream_chat(): a slow synthesis stream is deadline-bounded."""
    async with await BotTestHarness.create(
        bot_config=_react_config(tool_loop_timeout=0.0),
        main_responses=[
            tool_call_response("echo_tool", {"text": "hi"}),
            text_response("real synthesized answer"),
        ],
        tools=[_EchoTool()],
    ) as harness:
        harness.bot.llm.set_response_delay(
            _synthesis_delay(_SLOW_SYNTHESIS_DELAY)
        )

        with caplog.at_level(logging.WARNING):
            start = time.monotonic()
            result = await harness.stream_chat("please use the tool")
            elapsed = time.monotonic() - start

    assert elapsed < _BOUNDED_CEILING, (
        f"streaming finalize was not bounded: {elapsed:.2f}s"
    )
    # Only the graceful fallback chunk is yielded (the source never produced
    # one — it was cancelled at the deadline).
    assert result.chunks == [_DEFAULT_TOOL_LOOP_TIMEOUT_MESSAGE]
    assert result.response == _DEFAULT_TOOL_LOOP_TIMEOUT_MESSAGE
    assert "exceeded remaining tool loop budget" in caplog.text


async def test_phased_finalize_fast_path_unchanged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A normal (fast) synthesis returns the real response, no fallback."""
    async with await BotTestHarness.create(
        bot_config=_react_config(tool_loop_timeout=0.0),
        main_responses=[
            tool_call_response("echo_tool", {"text": "hi"}),
            text_response("real synthesized answer"),
        ],
        tools=[_EchoTool()],
    ) as harness:
        # No delay injected — the synthesis completes instantly.
        with caplog.at_level(logging.WARNING):
            result = await harness.chat("please use the tool")

    assert result.response == "real synthesized answer"
    assert "exceeded remaining tool loop budget" not in caplog.text


async def test_phased_finalize_stored_response_no_synthesis(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A stored final_response (no tool call) returns instantly — no network.

    Even with a large synthesis delay configured, the stored-response path
    performs no LLM re-call, so the delay predicate never fires and the turn
    is instant. Proves the wait_for wrapper is a no-op on the hot path.
    """
    async with await BotTestHarness.create(
        bot_config=_react_config(tool_loop_timeout=120.0),
        main_responses=[text_response("direct answer, no tools")],
        tools=[_EchoTool()],
    ) as harness:
        harness.bot.llm.set_response_delay(
            _synthesis_delay(_SLOW_SYNTHESIS_DELAY)
        )
        with caplog.at_level(logging.WARNING):
            start = time.monotonic()
            result = await harness.chat("just answer directly")
            elapsed = time.monotonic() - start

    assert result.response == "direct answer, no tools"
    assert elapsed < _BOUNDED_CEILING
    assert "exceeded remaining tool loop budget" not in caplog.text


async def test_finalize_budget_floor_grants_real_attempt() -> None:
    """At zero remaining budget the finalize still gets the floor, so a quick
    synthesis (under the floor) completes rather than being killed at 0."""
    quick = _MIN_FINALIZE_BUDGET / 2.0
    async with await BotTestHarness.create(
        bot_config=_react_config(tool_loop_timeout=0.0),
        main_responses=[
            tool_call_response("echo_tool", {"text": "hi"}),
            text_response("real synthesized answer"),
        ],
        tools=[_EchoTool()],
    ) as harness:
        # Synthesis sleeps less than the floor → completes within budget.
        harness.bot.llm.set_response_delay(_synthesis_delay(quick))
        result = await harness.chat("please use the tool")

    assert result.response == "real synthesized answer"


async def test_finalize_budget_tracks_remaining() -> None:
    """``_finalize_budget`` returns the remaining budget when it exceeds the
    floor, and the floor when it does not."""
    async with await BotTestHarness.create(
        bot_config=_react_config(tool_loop_timeout=10.0),
        main_responses=[text_response("noop")],
    ) as harness:
        bot = harness.bot
        now = time.monotonic()

        # 5s already elapsed of a 10s budget → ~5s remaining (> floor).
        budget = bot._finalize_budget(now - 5.0)
        assert 4.0 < budget < 6.0

        # 9.95s elapsed → ~0.05s remaining (< floor) → floored.
        floored = bot._finalize_budget(now - 9.95)
        assert floored == _MIN_FINALIZE_BUDGET

        # Fully exhausted / overrun → still floored, never negative.
        assert bot._finalize_budget(now - 20.0) == _MIN_FINALIZE_BUDGET


async def test_finalize_timeout_builders_markers() -> None:
    """The fallback response/chunk carry the canonical finish_reason plus the
    precise reason in metadata (no minted finish_reason value)."""
    async with await BotTestHarness.create(
        bot_config=_react_config(tool_loop_timeout=0.0),
        main_responses=[text_response("noop")],
    ) as harness:
        bot = harness.bot

        resp = bot._finalize_timeout_response()
        assert resp.content == _DEFAULT_TOOL_LOOP_TIMEOUT_MESSAGE
        assert resp.model == bot.llm.config.model
        assert resp.finish_reason == "length"
        # A wall-clock finalize timeout is NOT a token-budget cutoff, so
        # ``truncated`` stays False (its documented meaning); the timeout cause
        # is carried by finish_reason + termination_reason instead.
        assert resp.truncated is False
        assert resp.metadata["termination_reason"] == _FINALIZE_TIMEOUT_REASON

        chunk = bot._finalize_timeout_chunk()
        assert chunk.delta == _DEFAULT_TOOL_LOOP_TIMEOUT_MESSAGE
        assert chunk.is_final is True
        assert chunk.finish_reason == "length"
        assert chunk.truncated is False
        assert chunk.metadata["termination_reason"] == _FINALIZE_TIMEOUT_REASON


async def test_custom_tool_loop_timeout_message_surfaces(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """D4: a custom ``tool_loop_timeout_message`` surfaces in the degraded
    response for both buffered and streaming paths.

    Separate harnesses per path: a timed-out synthesis is cancelled *before*
    it consumes its queued response (the latency is applied ahead of response
    resolution), so sharing one ``EchoProvider`` queue across two turns would
    misalign it. Isolated harnesses keep each turn's script clean.
    """
    custom = "We ran out of time — please try again."

    async with await BotTestHarness.create(
        bot_config=_react_config(
            tool_loop_timeout=0.0, tool_loop_timeout_message=custom
        ),
        main_responses=[
            tool_call_response("echo_tool", {"text": "hi"}),
            text_response("real synthesized answer"),
        ],
        tools=[_EchoTool()],
    ) as harness:
        harness.bot.llm.set_response_delay(
            _synthesis_delay(_SLOW_SYNTHESIS_DELAY)
        )
        buffered = await harness.chat("please use the tool")
    assert buffered.response == custom

    async with await BotTestHarness.create(
        bot_config=_react_config(
            tool_loop_timeout=0.0, tool_loop_timeout_message=custom
        ),
        main_responses=[
            tool_call_response("echo_tool", {"text": "hi"}),
            text_response("real synthesized answer"),
        ],
        tools=[_EchoTool()],
    ) as harness:
        harness.bot.llm.set_response_delay(
            _synthesis_delay(_SLOW_SYNTHESIS_DELAY)
        )
        streaming = await harness.stream_chat("please use the tool")
    assert streaming.chunks == [custom]


class _HangingCloseStream:
    """A finalize source whose ``aclose()`` hangs — exercises teardown bounding.

    The source never produces a chunk within the budget (so
    ``_bounded_finalize_stream`` hits its deadline and truncates), and its
    ``aclose()`` then blocks far past the turn budget. Against the pre-fix
    unbounded ``await aclose()`` the whole consumption hangs (the outer
    ``wait_for`` below trips → the test fails); once the teardown is bounded it
    returns in ≈ the close ceiling.
    """

    def __init__(self) -> None:
        self.closed = False

    def __aiter__(self) -> "_HangingCloseStream":
        return self

    async def __anext__(self) -> Any:
        await asyncio.sleep(30)  # never resolves within the finalize budget
        raise StopAsyncIteration

    async def aclose(self) -> None:
        self.closed = True
        await asyncio.sleep(30)  # a teardown that hangs on a slow connection


async def test_bounded_finalize_stream_bounds_source_teardown() -> None:
    """The streaming teardown is itself bounded — a hung ``aclose()`` cannot
    reintroduce the unbounded wall-clock this path exists to eliminate.

    Reproduce-first: the outer ``wait_for`` trips (test fails) against the
    pre-fix unbounded ``await aclose()``; with the bounded teardown the
    consumption returns in ≈ ``_FINALIZE_SOURCE_CLOSE_TIMEOUT`` (1.0s).
    """
    async with await BotTestHarness.create(
        bot_config=_react_config(tool_loop_timeout=0.0),
        main_responses=[text_response("noop")],
    ) as harness:
        bot = harness.bot
        source = _HangingCloseStream()
        turn = TurnState(
            mode=TurnMode.STREAM, message="hi", context=harness.context
        )

        chunks: list[Any] = []

        async def _consume() -> None:
            async for chunk in bot._bounded_finalize_stream(source, 0.05, turn):
                chunks.append(chunk)

        start = time.monotonic()
        # Bound the whole consumption so a regression (unbounded aclose) fails
        # loudly here instead of hanging the suite.
        await asyncio.wait_for(_consume(), timeout=5.0)
        elapsed = time.monotonic() - start

    assert source.closed is True  # teardown was attempted, not skipped
    # Returned in ≈ the close ceiling, nowhere near the 30s hang.
    assert elapsed < _BOUNDED_CEILING, f"teardown not bounded: {elapsed:.2f}s"
    # Exactly the graceful fallback chunk (the source produced none in time).
    assert len(chunks) == 1
    assert chunks[0].metadata["termination_reason"] == _FINALIZE_TIMEOUT_REASON
