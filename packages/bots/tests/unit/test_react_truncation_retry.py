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
  ``tool_use`` (the incomplete node is off the active path) AND is pinned to be
  the retry (carries the merged budget), so the assertion discriminates the
  ``branch_from`` mechanic rather than passing on the abandon-only synthesis.
- ``retry error degrades`` — a retry ``complete()`` that raises falls back to
  terminal synthesis instead of aborting the turn: the retry is strictly
  additive to the abandon-and-synthesize contract.
- ``config validation`` — a non-positive ``truncation_retry_max_tokens`` is
  rejected at construction (direct and ``from_dict``): the runtime guard is
  ``budget is None``, so ``0`` would otherwise enable the retry with an
  impossible budget.
- ``merge not override`` — a caller ``llm_config_overrides`` survives the budget
  merge (``max_tokens`` set to the retry budget, other overrides preserved, the
  caller's dict not mutated).

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
from dataknobs_llm.testing import (
    ErrorResponse,
    text_response,
    tool_call_response,
)
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


def _assert_no_dangling_tool_use(
    messages: list[LLMMessage], *, require_tool_use: bool = False
) -> None:
    """Assert every ``tool_use`` block pairs with a ``tool_result``.

    Runs ``messages`` through the Anthropic adapter (the exact conversion the
    API 400 validates) and asserts no ``tool_use`` id is left unpaired.

    With ``require_tool_use=True`` additionally asserts at least one
    ``tool_use`` block is present — used where the point is that a truncated
    tool attempt was *retained* (and paired) in history, not merely absent.
    A plain no-dangling check would pass vacuously on a history that dropped
    the attempt entirely, so it cannot distinguish "retained + paired" from
    "dropped".
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
    if require_tool_use:
        assert tool_use_ids, (
            "expected a tool_use block in history but found none — the "
            "truncated attempt was dropped, not retained. Adapted messages: "
            f"{anthropic_messages}"
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

            # The retry is the SECOND completion.  Pin that calls[1] is the
            # retry (carries the merged budget), not the terminal synthesis —
            # otherwise the no-dangling assertion below would also hold of the
            # abandon-only path's synthesis call and wouldn't discriminate the
            # branch_from mechanic.
            retry_call = harness.provider.calls[1]
            assert (retry_call["config_overrides"] or {}).get(
                "max_tokens"
            ) == RETRY_BUDGET
            # Because the loop branched off the truncated node, the retry's
            # request history excludes the incomplete tool_use — no dangling
            # tool_use reaches the provider.  Removing branch_from would leave
            # the orphan here and trip this assertion.
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


# =========================================================================
# Config validation (a non-positive budget is a misconfiguration, not "on")
# =========================================================================


class TestTruncationRetryConfigValidation:
    """A non-positive ``truncation_retry_max_tokens`` is rejected at build.

    Reproduce-first: ``0`` and negatives were previously accepted (the runtime
    guard is ``budget is None``), enabling the retry with an impossible budget
    — a hard provider error or a guaranteed re-truncation.  ``__post_init__``
    now fails loud at construction, on both the direct and ``from_dict`` paths.
    """

    def test_zero_budget_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be a positive integer"):
            ReActReasoningConfig(truncation_retry_max_tokens=0)

    def test_negative_budget_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be a positive integer"):
            ReActReasoningConfig(truncation_retry_max_tokens=-1)

    def test_from_dict_zero_budget_rejected(self) -> None:
        # The config-load path (StructuredConfig.from_dict) constructs the
        # dataclass, so __post_init__ fires there too — a YAML typo is caught
        # at parse time, not at the first truncated turn.
        with pytest.raises(ValueError, match="must be a positive integer"):
            ReActReasoningConfig.from_dict(
                {"truncation_retry_max_tokens": 0}
            )

    def test_positive_budget_accepted(self) -> None:
        cfg = ReActReasoningConfig(truncation_retry_max_tokens=RETRY_BUDGET)
        assert cfg.truncation_retry_max_tokens == RETRY_BUDGET

    def test_none_budget_accepted(self) -> None:
        # The default (disabled) must remain constructible.
        cfg = ReActReasoningConfig()
        assert cfg.truncation_retry_max_tokens is None


# =========================================================================
# Retry error → degrade (the retry is additive to the abandon contract)
# =========================================================================


class TestTruncationRetryErrorDegrades:
    """A retry ``complete()`` that raises degrades to terminal synthesis.

    Reproduce-first: without the try/except in ``_maybe_retry_truncated_tool_call``
    the retry's exception propagates out of the turn and ``chat`` raises,
    converting a graceful abandon into a hard failure.  The guard returns the
    original truncated response so the caller's terminal branch synthesizes,
    exactly as the disabled default would.
    """

    @pytest.mark.asyncio
    async def test_retry_provider_error_falls_back_to_synthesis(self) -> None:
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
                # retry: the provider raises (rate-limit / network / bad call).
                ErrorResponse(RuntimeError("provider unavailable")),
                # terminal synthesis still runs — the turn degrades, not aborts.
                text_response("Degraded answer"),
            ],
            tools=[tool],
        ) as harness:
            # Must NOT raise: the retry error is absorbed into the degrade path.
            result = await harness.chat("Use the echo tool")

            assert tool.call_count == 0
            assert result.response == "Degraded answer"
            # truncated + failed retry (recorded) + synthesis = three completions.
            assert harness.provider.call_count == 3
            # Parity with the disabled-default abandon: the error path restores
            # the truncated node, so its tool_use is RETAINED in the synthesis
            # request AND paired with a synthetic tool_result.  ``require_tool_use``
            # makes this discriminating — without the node restore the synthesis
            # history would carry no tool_use at all (a plain no-dangling check
            # would pass vacuously and hide the dropped attempt).
            _assert_no_dangling_tool_use(
                list(harness.provider.calls[2]["messages"]),
                require_tool_use=True,
            )


# =========================================================================
# Merge-not-override contract (caller overrides survive the budget merge)
# =========================================================================


class TestTruncationRetryMergesOverrides:
    """The retry merges the larger budget into caller ``llm_config_overrides``.

    Characterization test for the stated merge contract: a per-call override
    threaded through ``kwargs`` must survive, with ``max_tokens`` set to the
    retry budget.  Uses the monolithic route so the caller override is passed
    directly and the recorded retry call can be inspected.
    """

    @pytest.mark.asyncio
    async def test_retry_call_carries_caller_override_and_budget(self) -> None:
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

        await strategy.generate(
            manager,
            provider,
            tools=[tool],
            llm_config_overrides={"temperature": 0.9},
        )

        # calls[0] is the initial completion; calls[1] is the retry.
        retry_overrides = provider.calls[1]["config_overrides"]
        # Caller's override preserved AND the budget merged in (budget wins for
        # max_tokens; the caller's temperature is untouched).
        assert retry_overrides["temperature"] == 0.9
        assert retry_overrides["max_tokens"] == RETRY_BUDGET
        # The caller's own dict must not have been mutated by the merge.
        assert provider.calls[0]["config_overrides"] == {"temperature": 0.9}
