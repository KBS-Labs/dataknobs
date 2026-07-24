"""Reproduce-first tests for the opt-in adaptive-budget truncation retry.

Background
----------
When a provider cuts generation off at the token budget *during a tool-call
turn* (Anthropic ``stop_reason == "max_tokens"``, OpenAI
``finish_reason == "length"``), it returns a partial ``tool_use`` with
:attr:`LLMResponse.truncated` set.  The safe default is to *abandon* that turn
and synthesize a final answer without tools (the sibling
``test_react_truncated_tool_call`` file pins that behavior).

But for a task whose work *was* the oversized call, "abandon" means
"discarded."  This module covers the opt-in recovery: when
``ReActReasoningConfig.truncation_retry_max_tokens`` is set, a truncated
tool-call turn is retried **once** at the larger budget before being
abandoned.  A clean retry proceeds through the normal loop; a still-truncated
retry falls back to the terminal synthesis path — one attempt, no loop.

Discriminating assertions
-------------------------
- ``retry succeeds`` — the real tool executes (``call_count == 1``).  With the
  retry disabled the turn abandons and the tool is never called, so this
  assertion is what the feature buys.
- ``default off unchanged`` — exactly one tool-call ``complete()`` and one
  synthesis ``complete()`` (no retry): pins byte-identical behavior to the
  abandon-only default.
- ``branches off the truncated node`` — the retry request carries no dangling
  ``tool_use`` (the incomplete node is off the active path): pins that the
  retry does not reintroduce the message-sequence hazard.

Route coverage
--------------
- phased ``chat`` (``process_input`` → ``finalize_turn``),
- monolithic ``generate()`` — the retry lives in one shared helper invoked at
  both ``complete()`` sites, so both paths are exercised.
"""

from __future__ import annotations

from typing import Any

import pytest

from dataknobs_bots.reasoning.react import ReActReasoning
from dataknobs_bots.reasoning.react_config import ReActReasoningConfig
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

#: A generous retry budget; the provider clamps it to the model ceiling.
RETRY_BUDGET = 4096


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
# Structural assertion helper (mirrors the sibling truncation-terminal tests)
# ---------------------------------------------------------------------------


def _assert_no_dangling_tool_use(messages: list[LLMMessage]) -> None:
    """Assert every ``tool_use`` block pairs with a ``tool_result``.

    Runs ``messages`` through the Anthropic adapter (the exact conversion the
    API 400 validates) and asserts no ``tool_use`` id is left unpaired.
    """
    _system, anthropic_messages = AnthropicAdapter().adapt_messages(messages)
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
        "Dangling tool_use after adaptation (the exact Anthropic 400 "
        f"condition): {unpaired}. Adapted messages: {anthropic_messages}"
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
# Phased chat route (process_input → finalize_turn)
# =========================================================================


class TestTruncationRetrySucceeds:
    """A clean retry recovers the turn — the real tool executes."""

    @pytest.mark.asyncio
    async def test_truncated_tool_call_retry_succeeds(self) -> None:
        tool = EchoTool()

        async with await BotTestHarness.create(
            bot_config={
                "llm": {"provider": "echo", "model": "test"},
                "conversation_storage": {"backend": "memory"},
                "reasoning": {
                    "strategy": "react",
                    "truncation_retry_max_tokens": RETRY_BUDGET,
                },
            },
            main_responses=[
                # iter0: truncated mid-tool-call — triggers the retry.
                tool_call_response(
                    "echo_tool", {"message": "part"}, truncated=True
                ),
                # retry at the larger budget: a clean, complete tool call.
                tool_call_response("echo_tool", {"message": "whole"}),
                # next iteration returns the final answer.
                text_response("Recovered answer"),
            ],
            tools=[tool],
        ) as harness:
            result = await harness.chat("Use the echo tool")

        # The retry produced a real, executable call — impossible under the
        # abandon-only default (which never reaches the tool).
        assert tool.call_count == 1
        assert result.response == "Recovered answer"
        # truncated + retry + final = three completions (one retry issued).
        assert harness.provider.call_count == 3


class TestTruncationRetryStillTruncatedTerminal:
    """A still-truncated retry falls back to terminal synthesis — no loop."""

    @pytest.mark.asyncio
    async def test_retry_still_truncated_is_terminal(self) -> None:
        tool = EchoTool()

        async with await BotTestHarness.create(
            bot_config={
                "llm": {"provider": "echo", "model": "test"},
                "conversation_storage": {"backend": "memory"},
                "reasoning": {
                    "strategy": "react",
                    "truncation_retry_max_tokens": RETRY_BUDGET,
                },
            },
            main_responses=[
                # iter0: truncated — triggers the retry.
                tool_call_response(
                    "echo_tool", {"message": "part"}, truncated=True
                ),
                # retry: still truncated — abandon, no second retry.
                tool_call_response(
                    "echo_tool", {"message": "part"}, truncated=True
                ),
                # terminal synthesis without tools.
                text_response("Synthesized answer"),
            ],
            tools=[tool],
        ) as harness:
            result = await harness.chat("Use the echo tool")

        # Never executed, terminal synthesis wins.
        assert tool.call_count == 0
        assert result.response == "Synthesized answer"
        # truncated + one retry + synthesis = three completions (exactly one
        # retry — a loop would consume more).
        assert harness.provider.call_count == 3


class TestTruncationRetryDefaultOffUnchanged:
    """Without the opt-in config, behavior is byte-identical to abandon-only."""

    @pytest.mark.asyncio
    async def test_default_off_no_retry(self) -> None:
        tool = EchoTool()

        async with await BotTestHarness.create(
            bot_config={
                "llm": {"provider": "echo", "model": "test"},
                "conversation_storage": {"backend": "memory"},
                # No truncation_retry_max_tokens → retry disabled.
                "reasoning": {"strategy": "react"},
            },
            main_responses=[
                tool_call_response(
                    "echo_tool", {"message": "part"}, truncated=True
                ),
                text_response("Synthesized answer"),
            ],
            tools=[tool],
        ) as harness:
            result = await harness.chat("Use the echo tool")

        assert tool.call_count == 0
        assert result.response == "Synthesized answer"
        # truncated tool-call + synthesis = two completions, NO retry.
        assert harness.provider.call_count == 2


class TestTruncationRetryBranchesOffTruncatedNode:
    """The retry branches off the truncated node — no orphan ``tool_use``."""

    @pytest.mark.asyncio
    async def test_retry_request_has_no_dangling_tool_use(self) -> None:
        tool = EchoTool()

        async with await BotTestHarness.create(
            bot_config={
                "llm": {"provider": "echo", "model": "test"},
                "conversation_storage": {"backend": "memory"},
                "reasoning": {
                    "strategy": "react",
                    "truncation_retry_max_tokens": RETRY_BUDGET,
                },
            },
            main_responses=[
                tool_call_response(
                    "echo_tool", {"message": "part"}, truncated=True
                ),
                tool_call_response("echo_tool", {"message": "whole"}),
                text_response("Recovered answer"),
            ],
            tools=[tool],
        ) as harness:
            await harness.chat("Use the echo tool")

            # The retry is the SECOND completion.  Because the loop branched
            # off the truncated node, its request history excludes the
            # incomplete tool_use — no dangling tool_use reaches the provider.
            retry_call = harness.provider.calls[1]
            _assert_no_dangling_tool_use(list(retry_call["messages"]))


# =========================================================================
# Monolithic generate() route (shared helper — D4)
# =========================================================================


class TestTruncationRetryMonolithic:
    """The monolithic ``generate()`` path retries through the same helper."""

    @pytest.mark.asyncio
    async def test_generate_retry_succeeds(self) -> None:
        provider = _make_provider(
            [
                tool_call_response(
                    "echo_tool", {"message": "part"}, truncated=True
                ),
                tool_call_response("echo_tool", {"message": "whole"}),
                text_response("Recovered answer"),
            ]
        )
        manager = await _make_manager(provider)
        tool = EchoTool()
        strategy = ReActReasoning(
            ReActReasoningConfig(truncation_retry_max_tokens=RETRY_BUDGET)
        )

        response = await strategy.generate(manager, provider, tools=[tool])

        assert tool.call_count == 1
        assert response.content == "Recovered answer"
        assert provider.call_count == 3

    @pytest.mark.asyncio
    async def test_generate_default_off_no_retry(self) -> None:
        provider = _make_provider(
            [
                tool_call_response(
                    "echo_tool", {"message": "part"}, truncated=True
                ),
                text_response("Synthesized answer"),
            ]
        )
        manager = await _make_manager(provider)
        tool = EchoTool()
        strategy = ReActReasoning()  # default config → retry disabled

        response = await strategy.generate(manager, provider, tools=[tool])

        assert tool.call_count == 0
        assert response.content == "Synthesized answer"
        assert provider.call_count == 2
