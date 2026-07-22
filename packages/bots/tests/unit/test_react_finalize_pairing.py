"""Reproduce-first tests for the ReAct finalize tool_use/tool_result pairing.

Background
----------
When a ReAct turn ends *abnormally* — a duplicate-tool-call break, or a
DynaBot-level tool-loop timeout — the loop can leave an assistant
``tool_use`` in conversation history with no following ``tool_result``.
The synthesis LLM call then re-sends that history to the provider.  On
Anthropic's Messages API a dangling ``tool_use`` is a hard 400; other
providers tolerate it.  Historically the strategy appended a
``role="system"`` "loop ended, use existing results" notice intending it to
sit inline after the ``tool_use`` — but adapters that lift system messages
to a top-level ``system`` param (Anthropic) hoist that notice *out of the
message array*, so the ``tool_use`` is left dangling.

The fix pairs every orphan ``tool_use`` with a synthetic ``role="tool"``
result at the synthesis chokepoint
(:func:`dataknobs_bots.reasoning.react._pair_orphan_tool_calls`), and removes
the now-redundant ``role="system"`` notices.

Assertion strategy
------------------
The defect is a conversation-state invariant, so the assertion is
structural and ties the property to the *real* provider contract without a
live API call: capture the messages the finalize synthesis received
(``EchoProvider.get_last_call()``) and run them through
``AnthropicAdapter.adapt_messages(...)`` — the exact conversion the API 400
checks — asserting every ``tool_use`` block is paired with a
``tool_result``.

Route coverage
--------------
The orphan-producing routes (these tests FAIL against unfixed code):
- phased duplicate-break (``process_input`` → ``finalize_turn``),
- DynaBot ``tool_loop_timeout`` guard,
- monolithic ``generate()`` duplicate-break,
- streaming duplicate-break (``_stream_finalize``).

The ``max_iterations`` routes do NOT produce an orphan today — the last
iteration's tool calls are executed and paired *before* the cap fires — so
those, plus the happy path, are covered as no-false-positive guards (the
pairing helper is a correct no-op there and must not append spurious
messages).
"""

from __future__ import annotations

from typing import Any

import pytest

from dataknobs_bots.reasoning.react import (
    _UNEXECUTED_TOOL_RESULT,
    ReActReasoning,
)
from dataknobs_bots.testing import BotTestHarness
from dataknobs_data.backends.memory import AsyncMemoryDatabase
from dataknobs_llm import LLMConfig, LLMMessage
from dataknobs_llm.conversations import ConversationManager
from dataknobs_llm.conversations.storage import DataknobsConversationStorage
from dataknobs_llm.llm.providers.anthropic import AnthropicAdapter
from dataknobs_llm.llm.providers.echo import EchoProvider
from dataknobs_llm.prompts import ConfigPromptLibrary
from dataknobs_llm.prompts.builders import AsyncPromptBuilder
from dataknobs_llm.testing import text_response, tool_call_response
from dataknobs_llm.tools.base import Tool


# ---------------------------------------------------------------------------
# Test tool
# ---------------------------------------------------------------------------


class EchoTool(Tool):
    """Simple tool that returns its input for testing."""

    def __init__(self) -> None:
        super().__init__(name="echo_tool", description="Echoes input back")
        self.call_count = 0

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"message": {"type": "string"}},
        }

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        self.call_count += 1
        return {"echoed": kwargs.get("message", ""), "call": self.call_count}


# ---------------------------------------------------------------------------
# Structural assertion helpers
# ---------------------------------------------------------------------------


def _adapt(messages: list[LLMMessage]) -> list[dict[str, Any]]:
    """Adapt an LLMMessage history to Anthropic message blocks."""
    _system, anthropic_messages = AnthropicAdapter().adapt_messages(messages)
    return anthropic_messages


def _assert_no_dangling_tool_use(messages: list[LLMMessage]) -> None:
    """Assert every ``tool_use`` block pairs with a ``tool_result``.

    Runs ``messages`` through the Anthropic adapter (the exact conversion
    the API 400 validates) and asserts no ``tool_use`` id is left without a
    matching ``tool_result`` — the precise dangling-``tool_use`` condition.
    """
    anthropic_messages = _adapt(messages)
    tool_use_ids: list[str] = []
    tool_result_ids: set[str] = set()
    for m in anthropic_messages:
        content = m.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if block.get("type") == "tool_use":
                tool_use_ids.append(block["id"])
            elif block.get("type") == "tool_result":
                tool_result_ids.add(block["tool_use_id"])

    unpaired = [tid for tid in tool_use_ids if tid not in tool_result_ids]
    assert not unpaired, (
        "Dangling tool_use blocks after adaptation (the exact Anthropic 400 "
        f"condition): {unpaired}. Adapted messages: {anthropic_messages}"
    )


def _synthesis_messages(provider: EchoProvider) -> list[LLMMessage]:
    """Return the message list the most recent completion received."""
    last = provider.get_last_call()
    assert last is not None, "expected at least one provider call"
    return list(last["messages"])


def _has_synthetic_pairing(messages: list[LLMMessage]) -> bool:
    """Whether a synthetic unexecuted-tool ``tool_result`` was appended."""
    return any(
        m.role == "tool" and m.content == _UNEXECUTED_TOOL_RESULT
        for m in messages
    )


def _system_notice_present(messages: list[LLMMessage]) -> bool:
    """Whether a removed mid-conversation ``role="system"`` notice leaked.

    The system prompt itself is legitimate; this looks specifically for the
    old duplicate/timeout notices ("System notice: ...").
    """
    return any(
        m.role == "system" and m.content and "System notice" in m.content
        for m in messages
    )


# ---------------------------------------------------------------------------
# Direct-manager helpers (for the monolithic generate() route)
# ---------------------------------------------------------------------------


def _make_provider(responses: list[Any]) -> EchoProvider:
    provider = EchoProvider(
        LLMConfig(provider="echo", model="echo-test", options={"echo_prefix": ""})
    )
    provider.set_responses(responses)
    return provider


async def _make_manager(provider: EchoProvider) -> ConversationManager:
    library = ConfigPromptLibrary(
        {"system": {"assistant": {"template": "You are a test bot."}}}
    )
    builder = AsyncPromptBuilder(library=library)
    storage = DataknobsConversationStorage(AsyncMemoryDatabase())
    mgr = await ConversationManager.create(
        llm=provider,
        prompt_builder=builder,
        storage=storage,
        system_prompt_name="assistant",
    )
    await mgr.add_message(role="user", content="test input")
    return mgr


# =========================================================================
# T1 — phased duplicate-break (finalize_turn synthesis)
# =========================================================================


class TestPhasedDuplicateBreak:
    """Duplicate tool calls end the phased loop with an unexecuted call."""

    @pytest.mark.asyncio
    async def test_finalize_pairs_orphan_tool_use(self) -> None:
        tool = EchoTool()

        async with await BotTestHarness.create(
            bot_config={
                "llm": {"provider": "echo", "model": "test"},
                "conversation_storage": {"backend": "memory"},
                "reasoning": {"strategy": "react"},
            },
            main_responses=[
                tool_call_response("echo_tool", {"message": "same"}),
                tool_call_response("echo_tool", {"message": "same"}),
                text_response("Synthesized answer"),
            ],
            tools=[tool],
        ) as harness:
            result = await harness.chat("Use the echo tool")

        # The turn completes without error and returns the synthesis text.
        assert result.response == "Synthesized answer"

        messages = _synthesis_messages(harness.provider)
        # Structural: the synthesis history adapts to a paired Anthropic
        # sequence (no dangling tool_use → no 400).
        _assert_no_dangling_tool_use(messages)
        # The abandoned (duplicate) tool_use is paired via a synthetic
        # tool_result carrying the guidance, not a hoisted-away system notice.
        assert _has_synthetic_pairing(messages)
        assert not _system_notice_present(messages)


# =========================================================================
# T2 — DynaBot tool_loop_timeout guard (cross-package route)
# =========================================================================


class TestToolLoopTimeout:
    """A wall-clock timeout breaks the loop with a pending tool call."""

    @pytest.mark.asyncio
    async def test_finalize_pairs_orphan_after_timeout(self) -> None:
        tool = EchoTool()

        async with await BotTestHarness.create(
            bot_config={
                "llm": {"provider": "echo", "model": "test"},
                "conversation_storage": {"backend": "memory"},
                "reasoning": {"strategy": "react"},
                # 0.0 budget → the loop times out immediately after the
                # first process_input returns a pending tool call, before
                # it is executed.
                "tool_loop_timeout": 0.0,
            },
            main_responses=[
                tool_call_response("echo_tool", {"message": "hi"}),
                text_response("Synthesized after timeout"),
            ],
            tools=[tool],
        ) as harness:
            result = await harness.chat("Use the echo tool")

        assert result.response == "Synthesized after timeout"
        # The tool never executed (timeout fired first).
        assert tool.call_count == 0

        messages = _synthesis_messages(harness.provider)
        _assert_no_dangling_tool_use(messages)
        assert _has_synthetic_pairing(messages)
        assert not _system_notice_present(messages)


# =========================================================================
# T4 — monolithic generate() duplicate-break (HybridReasoning route)
# =========================================================================


class TestMonolithicGenerateDuplicateBreak:
    """The monolithic generate() path (used by HybridReasoning) also
    leaves an orphan on a duplicate break."""

    @pytest.mark.asyncio
    async def test_generate_pairs_orphan_tool_use(self) -> None:
        provider = _make_provider(
            [
                tool_call_response("echo_tool", {"message": "same"}),
                tool_call_response("echo_tool", {"message": "same"}),
                text_response("Synthesized answer"),
            ]
        )
        manager = await _make_manager(provider)
        tool = EchoTool()
        strategy = ReActReasoning()

        response = await strategy.generate(manager, provider, tools=[tool])

        assert response.content == "Synthesized answer"
        # First call executed, duplicate (second) did not.
        assert tool.call_count == 1

        messages = _synthesis_messages(provider)
        _assert_no_dangling_tool_use(messages)
        assert _has_synthetic_pairing(messages)
        assert not _system_notice_present(messages)


# =========================================================================
# T5 — streaming finalize duplicate-break (_stream_finalize synthesis)
# =========================================================================


class TestStreamingDuplicateBreak:
    """The streaming finalize path pairs orphans too."""

    @pytest.mark.asyncio
    async def test_stream_finalize_pairs_orphan_tool_use(self) -> None:
        tool = EchoTool()

        async with await BotTestHarness.create(
            bot_config={
                "llm": {"provider": "echo", "model": "test"},
                "conversation_storage": {"backend": "memory"},
                "reasoning": {"strategy": "react"},
            },
            main_responses=[
                tool_call_response("echo_tool", {"message": "same"}),
                tool_call_response("echo_tool", {"message": "same"}),
                text_response("Streamed synthesis"),
            ],
            tools=[tool],
        ) as harness:
            result = await harness.stream_chat("Use the echo tool")

        assert result.response == "Streamed synthesis"

        messages = _synthesis_messages(harness.provider)
        _assert_no_dangling_tool_use(messages)
        assert _has_synthetic_pairing(messages)
        assert not _system_notice_present(messages)


# =========================================================================
# T3 — max_iterations (no-false-positive guard)
# =========================================================================


class TestMaxIterationsNoFalsePositive:
    """max_iterations ends the loop with the last call already paired, so
    the helper must be a no-op (no spurious synthetic tool_result)."""

    @pytest.mark.asyncio
    async def test_max_iterations_no_spurious_pairing(self) -> None:
        tool = EchoTool()

        async with await BotTestHarness.create(
            bot_config={
                "llm": {"provider": "echo", "model": "test"},
                "conversation_storage": {"backend": "memory"},
                # Distinct params each turn so the cap (not duplicate
                # detection) ends the loop.
                "reasoning": {"strategy": "react", "max_iterations": 2},
            },
            main_responses=[
                tool_call_response("echo_tool", {"message": "a"}),
                tool_call_response("echo_tool", {"message": "b"}),
                text_response("Synthesized after cap"),
            ],
            tools=[tool],
        ) as harness:
            result = await harness.chat("Do two distinct things")

        assert result.response == "Synthesized after cap"
        assert tool.call_count == 2

        messages = _synthesis_messages(harness.provider)
        # Already paired — no dangling tool_use, and no synthetic append.
        _assert_no_dangling_tool_use(messages)
        assert not _has_synthetic_pairing(messages)
        assert not _system_notice_present(messages)


# =========================================================================
# T6 — happy path (no abnormal termination)
# =========================================================================


class TestHappyPathUnchanged:
    """A normal turn ending in a real final answer is unchanged: the
    finalize helper never runs (stored final_response short-circuits)."""

    @pytest.mark.asyncio
    async def test_final_answer_no_synthetic_pairing(self) -> None:
        tool = EchoTool()

        async with await BotTestHarness.create(
            bot_config={
                "llm": {"provider": "echo", "model": "test"},
                "conversation_storage": {"backend": "memory"},
                "reasoning": {"strategy": "react"},
            },
            main_responses=[
                tool_call_response("echo_tool", {"message": "go"}),
                text_response("The tool answered"),
            ],
            tools=[tool],
        ) as harness:
            result = await harness.chat("Use the echo tool")

        assert result.response == "The tool answered"
        assert tool.call_count == 1

        messages = _synthesis_messages(harness.provider)
        _assert_no_dangling_tool_use(messages)
        assert not _has_synthetic_pairing(messages)
        assert not _system_notice_present(messages)
