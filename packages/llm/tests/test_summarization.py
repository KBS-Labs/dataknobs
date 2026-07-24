"""Tests for the shared conversation-summarization seam.

The seam (``summarize_messages`` + ``format_messages_for_summary`` + the
``Summarizer`` protocol / ``LLMSummarizer`` default) is the one place the
prompt-fill + ``llm.complete`` summarization pattern lives — consumed by both
``ConversationManager.compact_history`` and ``dataknobs_bots``' ``SummaryMemory``.
Real ``EchoProvider`` (echoes the prompt back), no mocks.
"""

from __future__ import annotations

from dataknobs_llm import (
    DEFAULT_SUMMARIZATION_PROMPT,
    LLMSummarizer,
    Summarizer,
    format_messages_for_summary,
    summarize_messages,
)
from dataknobs_llm.llm import EchoProvider, LLMConfig
from dataknobs_llm.llm.base import LLMMessage


def _echo() -> EchoProvider:
    return EchoProvider(LLMConfig(provider="echo", model="echo-model"))


def test_format_messages_renders_role_content_lines() -> None:
    rendered = format_messages_for_summary(
        [
            LLMMessage(role="user", content="hi"),
            LLMMessage(role="assistant", content=None),  # tool-only turn
            LLMMessage(role="tool", content="obs"),
        ]
    )
    assert rendered == "user: hi\nassistant: \ntool: obs"


async def test_summarize_messages_fills_prompt_and_completes() -> None:
    messages = [
        LLMMessage(role="user", content="what is the capital of france"),
        LLMMessage(role="assistant", content="Paris"),
    ]
    result = await summarize_messages(_echo(), messages)
    # EchoProvider echoes the prompt, so the filled template (with the formatted
    # messages and the default "(none)" existing summary) round-trips.
    assert "user: what is the capital of france" in result
    assert "assistant: Paris" in result
    assert "(none)" in result


async def test_summarize_messages_carries_existing_summary() -> None:
    result = await summarize_messages(
        _echo(),
        [LLMMessage(role="user", content="more")],
        existing_summary="prior summary text",
    )
    assert "prior summary text" in result


async def test_summarize_messages_honors_custom_prompt() -> None:
    result = await summarize_messages(
        _echo(),
        [LLMMessage(role="user", content="x")],
        prompt="CUSTOM {existing_summary} :: {new_messages}",
    )
    assert "CUSTOM (none) :: user: x" in result
    assert "You are a conversation summarizer" not in result


async def test_llm_summarizer_satisfies_protocol_and_delegates() -> None:
    summarizer = LLMSummarizer(_echo())
    assert isinstance(summarizer, Summarizer)
    result = await summarizer.summarize([LLMMessage(role="user", content="hello")])
    assert "user: hello" in result


def test_default_prompt_has_expected_placeholders() -> None:
    assert "{existing_summary}" in DEFAULT_SUMMARIZATION_PROMPT
    assert "{new_messages}" in DEFAULT_SUMMARIZATION_PROMPT
