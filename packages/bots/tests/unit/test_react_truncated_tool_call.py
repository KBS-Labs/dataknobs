"""Reproduce-first tests for the ReAct reaction to a truncated tool call.

Background
----------
When a provider cuts generation off at the token budget *during a tool-call
turn* (Anthropic ``stop_reason == "max_tokens"``, OpenAI
``finish_reason == "length"``), it returns a partial ``tool_use`` whose
arguments may be missing or malformed, with :attr:`LLMResponse.truncated`
set.  Historically the ReAct loop treated that turn like any other tool-call
turn and *executed the incomplete call* — which surfaces downstream as a
masked "argument required" error, and the model then retries the identical
oversized call until the duplicate-breaker fires.

The fix: a truncated tool-call turn is **terminal, not executed**.  It is
abandoned exactly like a duplicate-tool-call break — the orphan ``tool_use``
is paired at the synthesis chokepoint and a final answer is synthesized
without tools.  The tool is never called with partial arguments.

Discriminating assertion
------------------------
``tool.call_count == 0`` is the assertion that FAILS against unfixed code
(which executes the truncated call, ``call_count == 1``) and PASSES after.
The structural no-dangling-``tool_use`` assertions mirror the sibling
finalize-pairing tests: they tie the terminal handling to the real Anthropic
message contract without a live API call.

Route coverage
--------------
- phased ``chat`` (``process_input`` → ``finalize_turn``),
- phased ``stream_chat`` (``process_input`` → ``_stream_finalize``),
- monolithic ``generate()`` (HybridReasoning route).

No-false-positive guards (pass before and after the fix):
- a truncated *text* turn (no tool calls) is returned as-is, not intercepted,
- a normal (non-truncated) tool-call turn still executes the tool.
"""

from __future__ import annotations

from typing import Any

import pytest

from dataknobs_bots.reasoning.react import ReActReasoning
from dataknobs_bots.testing import BotTestHarness
from dataknobs_data.backends.memory import AsyncMemoryDatabase
from dataknobs_llm import LLMConfig, LLMMessage
from dataknobs_llm.conversations import ConversationManager
from dataknobs_llm.conversations.storage import DataknobsConversationStorage
from dataknobs_llm.llm.message_sequence import _UNEXECUTED_TOOL_RESULT
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
    """Simple tool that records whether it was called."""

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

    Runs ``messages`` through the Anthropic adapter (the exact conversion the
    API 400 validates) and asserts no ``tool_use`` id is left without a
    matching ``tool_result``.
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


#: Both synthetic guidance strings share this marker prefix.
_SYNTHETIC_PREFIX = "[Tool result unavailable:"


def _synthetic_pairing_contents(messages: list[LLMMessage]) -> list[str]:
    """Contents of the synthetic pairing ``tool_result`` messages, if any."""
    return [
        m.content
        for m in messages
        if m.role == "tool"
        and isinstance(m.content, str)
        and m.content.startswith(_SYNTHETIC_PREFIX)
    ]


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
# Phased chat route (process_input → finalize_turn)
# =========================================================================


class TestTruncatedToolCallPhased:
    """A truncated tool-call turn is abandoned, not executed (chat)."""

    @pytest.mark.asyncio
    async def test_truncated_tool_call_not_executed(self) -> None:
        tool = EchoTool()

        async with await BotTestHarness.create(
            bot_config={
                "llm": {"provider": "echo", "model": "test"},
                "conversation_storage": {"backend": "memory"},
                "reasoning": {"strategy": "react"},
            },
            main_responses=[
                # Truncated mid-tool-call: incomplete arguments.
                tool_call_response("echo_tool", {"message": "part"}, truncated=True),
                # Synthesis after the abandoned call.
                text_response("Synthesized answer"),
            ],
            tools=[tool],
        ) as harness:
            result = await harness.chat("Use the echo tool")

        # The incomplete call is never fed to the tool.
        assert tool.call_count == 0
        # The turn ends with a synthesized answer.
        assert result.response == "Synthesized answer"

        messages = _synthesis_messages(harness.provider)
        # Terminal handling still pairs the orphan tool_use — no 400.
        _assert_no_dangling_tool_use(messages)
        # The call was never reached, so it carries the generic
        # "loop ended before execution" guidance (not the duplicate nuance).
        assert _UNEXECUTED_TOOL_RESULT in _synthetic_pairing_contents(messages)


# =========================================================================
# Phased streaming route (process_input → _stream_finalize)
# =========================================================================


class TestTruncatedToolCallStreaming:
    """The streaming finalize path abandons a truncated call too."""

    @pytest.mark.asyncio
    async def test_truncated_tool_call_not_executed_streaming(self) -> None:
        tool = EchoTool()

        async with await BotTestHarness.create(
            bot_config={
                "llm": {"provider": "echo", "model": "test"},
                "conversation_storage": {"backend": "memory"},
                "reasoning": {"strategy": "react"},
            },
            main_responses=[
                tool_call_response("echo_tool", {"message": "part"}, truncated=True),
                text_response("Streamed synthesis"),
            ],
            tools=[tool],
        ) as harness:
            result = await harness.stream_chat("Use the echo tool")

        assert tool.call_count == 0
        assert result.response == "Streamed synthesis"

        messages = _synthesis_messages(harness.provider)
        _assert_no_dangling_tool_use(messages)
        assert _UNEXECUTED_TOOL_RESULT in _synthetic_pairing_contents(messages)


# =========================================================================
# Monolithic generate() route (HybridReasoning)
# =========================================================================


class TestTruncatedToolCallMonolithic:
    """The monolithic generate() path abandons a truncated call too."""

    @pytest.mark.asyncio
    async def test_generate_abandons_truncated_tool_call(self) -> None:
        provider = _make_provider(
            [
                tool_call_response("echo_tool", {"message": "part"}, truncated=True),
                text_response("Synthesized answer"),
            ]
        )
        manager = await _make_manager(provider)
        tool = EchoTool()
        strategy = ReActReasoning()

        response = await strategy.generate(manager, provider, tools=[tool])

        assert tool.call_count == 0
        assert response.content == "Synthesized answer"

        messages = _synthesis_messages(provider)
        _assert_no_dangling_tool_use(messages)
        assert _UNEXECUTED_TOOL_RESULT in _synthetic_pairing_contents(messages)


# =========================================================================
# Truncation on a NON-first iteration
# =========================================================================


class TestTruncatedToolCallAfterSuccessfulCall:
    """A truncated call on a *later* iteration abandons only that call — the
    earlier successful tool result survives into the tool-less synthesis.

    Extends the first-iteration coverage: proves (a) an earlier well-formed
    call still executes and its real observation reaches synthesis, and (b) the
    truncated orphan carries the *generic* ``_UNEXECUTED_TOOL_RESULT`` guidance
    (its incomplete args give it a different ``(name, params)`` signature from
    the answered call, so it is not misclassified as a duplicate-break orphan).
    """

    @pytest.mark.asyncio
    async def test_truncated_on_second_iteration(self) -> None:
        tool = EchoTool()

        async with await BotTestHarness.create(
            bot_config={
                "llm": {"provider": "echo", "model": "test"},
                "conversation_storage": {"backend": "memory"},
                "reasoning": {"strategy": "react"},
            },
            main_responses=[
                # iter0: normal call — executes.
                tool_call_response("echo_tool", {"message": "first"}),
                # iter1: truncated mid-call, different (incomplete) args —
                # abandoned, not executed.
                tool_call_response(
                    "echo_tool", {"message": "second"}, truncated=True
                ),
                # synthesis after the abandoned second call.
                text_response("Synthesized answer"),
            ],
            tools=[tool],
        ) as harness:
            result = await harness.chat("Use the echo tool")

        # Only the first (complete) call ran.
        assert tool.call_count == 1
        assert result.response == "Synthesized answer"

        messages = _synthesis_messages(harness.provider)
        _assert_no_dangling_tool_use(messages)

        # The first call's REAL observation reaches synthesis (a role="tool"
        # message that is not a synthetic pairing).
        real_obs = [
            m
            for m in messages
            if m.role == "tool"
            and isinstance(m.content, str)
            and not m.content.startswith(_SYNTHETIC_PREFIX)
        ]
        assert real_obs, "first tool's real observation must reach synthesis"

        # The truncated orphan carries GENERIC (not duplicate) guidance — its
        # signature differs from the answered call.
        assert _UNEXECUTED_TOOL_RESULT in _synthetic_pairing_contents(messages)


# =========================================================================
# Multi-block partial truncation (all-or-nothing abandonment)
# =========================================================================


class TestTruncatedMultiBlockAbandonsAll:
    """A truncated turn with multiple ``tool_use`` blocks abandons *every*
    block, not just the incomplete one — ``truncated`` is response-level, so a
    truncated turn's block completeness cannot be trusted.
    """

    @pytest.mark.asyncio
    async def test_truncated_multiblock_none_executed(self) -> None:
        tool = EchoTool()

        async with await BotTestHarness.create(
            bot_config={
                "llm": {"provider": "echo", "model": "test"},
                "conversation_storage": {"backend": "memory"},
                "reasoning": {"strategy": "react"},
            },
            main_responses=[
                tool_call_response(
                    "echo_tool",
                    {"message": "a"},
                    truncated=True,
                    additional_tools=[("echo_tool", {"message": "b"})],
                ),
                text_response("Synthesized answer"),
            ],
            tools=[tool],
        ) as harness:
            result = await harness.chat("Use the echo tool")

        # Neither block executed — the second (complete) block is abandoned too.
        assert tool.call_count == 0
        assert result.response == "Synthesized answer"

        messages = _synthesis_messages(harness.provider)
        _assert_no_dangling_tool_use(messages)

        # Both blocks paired with generic guidance (distinct ids → 2 results).
        pairings = _synthetic_pairing_contents(messages)
        assert len(pairings) == 2
        assert all(c == _UNEXECUTED_TOOL_RESULT for c in pairings)


# =========================================================================
# No-false-positive guards
# =========================================================================


class TestTruncatedTextNotIntercepted:
    """A truncated *text* turn (no tool calls) is already terminal — the
    loop returns it as-is and does not fabricate a synthesis pass.
    """

    @pytest.mark.asyncio
    async def test_truncated_text_returned_directly(self) -> None:
        tool = EchoTool()

        async with await BotTestHarness.create(
            bot_config={
                "llm": {"provider": "echo", "model": "test"},
                "conversation_storage": {"backend": "memory"},
                "reasoning": {"strategy": "react"},
            },
            main_responses=[text_response("Partial answer", truncated=True)],
            tools=[tool],
        ) as harness:
            result = await harness.chat("Just answer")

        assert tool.call_count == 0
        assert result.response == "Partial answer"


class TestNonTruncatedToolCallExecutes:
    """A normal (non-truncated) tool-call turn still executes the tool —
    the truncation guard must not intercept well-formed calls.
    """

    @pytest.mark.asyncio
    async def test_normal_tool_call_still_runs(self) -> None:
        tool = EchoTool()

        async with await BotTestHarness.create(
            bot_config={
                "llm": {"provider": "echo", "model": "test"},
                "conversation_storage": {"backend": "memory"},
                "reasoning": {"strategy": "react"},
            },
            main_responses=[
                tool_call_response("echo_tool", {"message": "hi"}),
                text_response("Done"),
            ],
            tools=[tool],
        ) as harness:
            result = await harness.chat("Use the echo tool")

        assert tool.call_count == 1
        assert result.response == "Done"
