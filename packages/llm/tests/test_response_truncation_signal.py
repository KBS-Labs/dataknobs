"""Tests for the cross-provider response-truncation signal (S2).

Bug: when a provider cuts generation off at the token budget (Anthropic
``stop_reason == "max_tokens"``, OpenAI/Ollama ``length``, Bedrock
``stopReason == "max_tokens"``), the truncated response — most dangerously a
truncated ``tool_calls`` turn carrying partial/invalid arguments — was returned
indistinguishable from a clean completion. No flag, no log; the consumer had to
know each provider's stop-reason vocabulary and correlate it with a tool-call
turn. Diagnosing it from the consumer side was a production forensics session.

Fix: a shared ``LLMResponse.truncated`` / ``LLMStreamResponse.truncated`` flag
that every provider populates from its own truncation stop reason, plus a base
``_warn_if_truncated`` hook that logs loudly on a truncated tool-call turn. The
Claude-family adapters (Anthropic, Bedrock) additionally normalize
``finish_reason`` onto the canonical vocabulary its docstring advertises
(``max_tokens`` → ``length``, ``tool_use`` → ``tool_calls``, ``end_turn`` →
``stop``) through one shared ``normalize_claude_stop_reason`` helper, preserving
the raw value on ``metadata['raw_finish_reason']``.

These reproduce-first tests FAIL against HEAD (no ``truncated`` field, no
normalization, no warning) and pass after the fix. Provider responses are built
with sanctioned SDK stand-ins (no dataknobs construct returns a real
provider-truncated response); the base-warning and serialization tests use real
dataknobs code paths.
"""

from __future__ import annotations

import logging
import types
from typing import Any, Self

import pytest

from dataknobs_llm import EchoProvider
from dataknobs_llm.llm.base import (
    LLMConfig,
    LLMMessage,
    LLMResponse,
    LLMStreamResponse,
    ToolCall,
)
from dataknobs_llm.llm.providers.anthropic import AnthropicAdapter, AnthropicProvider
from dataknobs_llm.llm.providers.bedrock import BedrockConverseAdapter
from dataknobs_llm.llm.providers.ollama import OllamaAdapter, OllamaProvider
from dataknobs_llm.llm.providers.openai import OpenAIAdapter
from dataknobs_llm.testing import (
    llm_response_from_dict,
    llm_response_to_dict,
    text_response,
    tool_call_response,
)

from test_anthropic_param_handling import make_anthropic_response


# ---------------------------------------------------------------------------
# Fake provider responses (sanctioned SDK stand-ins)
# ---------------------------------------------------------------------------


def make_openai_response(
    *,
    content: str = "",
    finish_reason: str = "stop",
    model: str = "gpt-4",
) -> object:
    """Build a minimal object with the OpenAI response attribute shape."""
    message = types.SimpleNamespace(content=content, function_call=None)
    choice = types.SimpleNamespace(message=message, finish_reason=finish_reason)
    usage = types.SimpleNamespace(
        prompt_tokens=5, completion_tokens=7, total_tokens=12
    )
    return types.SimpleNamespace(choices=[choice], model=model, usage=usage)


# ---------------------------------------------------------------------------
# Field defaults
# ---------------------------------------------------------------------------


class TestResponseFields:
    """The truncation flag defaults to False on both response types."""

    def test_llm_response_default_not_truncated(self) -> None:
        assert LLMResponse(content="x", model="m").truncated is False

    def test_stream_response_default_not_truncated(self) -> None:
        assert LLMStreamResponse(delta="x").truncated is False


# ---------------------------------------------------------------------------
# Anthropic — adapt_response detection + finish_reason normalization
# ---------------------------------------------------------------------------


class TestAnthropicAdaptResponse:
    """Anthropic surfaces max_tokens truncation and normalizes finish_reason."""

    def test_max_tokens_tool_call_is_truncated(self) -> None:
        """The landmine: max_tokens on a tool_use turn → truncated + length."""
        response = make_anthropic_response(
            [{"type": "tool_use", "id": "t1", "name": "submit", "input": {}}],
            stop_reason="max_tokens",
        )
        parsed = AnthropicAdapter().adapt_response(response)
        assert parsed.truncated is True
        assert parsed.tool_calls is not None
        # Normalized onto the canonical vocabulary, raw preserved.
        assert parsed.finish_reason == "length"
        assert parsed.metadata["raw_finish_reason"] == "max_tokens"

    def test_clean_end_turn_not_truncated(self) -> None:
        response = make_anthropic_response(
            [{"type": "text", "text": "done"}], stop_reason="end_turn"
        )
        parsed = AnthropicAdapter().adapt_response(response)
        assert parsed.truncated is False
        assert parsed.finish_reason == "stop"

    def test_tool_use_stop_reason_normalized_not_truncated(self) -> None:
        response = make_anthropic_response(
            [{"type": "tool_use", "id": "t1", "name": "search", "input": {}}],
            stop_reason="tool_use",
        )
        parsed = AnthropicAdapter().adapt_response(response)
        assert parsed.truncated is False
        assert parsed.finish_reason == "tool_calls"

    def test_unmapped_stop_reason_passes_through(self) -> None:
        response = make_anthropic_response(
            [{"type": "text", "text": "hi"}], stop_reason="refusal"
        )
        parsed = AnthropicAdapter().adapt_response(response)
        assert parsed.finish_reason == "refusal"
        # No normalization happened → no raw stashed.
        assert "raw_finish_reason" not in parsed.metadata


class TestAnthropicStreamTruncation:
    """The Anthropic streaming final chunk carries the truncation flag."""

    async def test_stream_final_chunk_truncated(self) -> None:
        provider = AnthropicProvider(
            LLMConfig(provider="anthropic", model="claude-3-sonnet")
        )

        class _StreamCtx:
            async def __aenter__(self) -> Self:
                return self

            async def __aexit__(self, *exc: object) -> None:
                return None

            def __aiter__(self) -> Self:
                return self

            async def __anext__(self) -> object:
                raise StopAsyncIteration

            async def get_final_message(self) -> object:
                return make_anthropic_response(
                    [{"type": "tool_use", "id": "t1", "name": "s", "input": {}}],
                    stop_reason="max_tokens",
                )

        provider._client = types.SimpleNamespace(
            messages=types.SimpleNamespace(stream=lambda **_kw: _StreamCtx())
        )
        provider._is_initialized = True

        chunks = [c async for c in provider.stream_complete("hi")]
        final = chunks[-1]
        assert final.is_final is True
        assert final.truncated is True


# ---------------------------------------------------------------------------
# OpenAI parity
# ---------------------------------------------------------------------------


class TestOpenAIAdaptResponse:
    """OpenAI ``finish_reason == 'length'`` sets the truncation flag."""

    def test_length_is_truncated(self) -> None:
        parsed = OpenAIAdapter().adapt_response(
            make_openai_response(content="cut off", finish_reason="length")
        )
        assert parsed.truncated is True
        assert parsed.finish_reason == "length"

    def test_stop_not_truncated(self) -> None:
        parsed = OpenAIAdapter().adapt_response(
            make_openai_response(content="all good", finish_reason="stop")
        )
        assert parsed.truncated is False

    def test_tool_calls_not_truncated(self) -> None:
        parsed = OpenAIAdapter().adapt_response(
            make_openai_response(finish_reason="tool_calls")
        )
        assert parsed.truncated is False


# ---------------------------------------------------------------------------
# Ollama parity
# ---------------------------------------------------------------------------


class TestOllamaAdaptResponse:
    """Ollama ``done_reason == 'length'`` sets the truncation flag."""

    def test_done_reason_length_is_truncated(self) -> None:
        parsed = OllamaAdapter().adapt_response(
            {
                "message": {"content": "cut off"},
                "model": "llama3.2",
                "done": True,
                "done_reason": "length",
            }
        )
        assert parsed.truncated is True
        assert parsed.finish_reason == "length"

    def test_done_reason_stop_not_truncated(self) -> None:
        parsed = OllamaAdapter().adapt_response(
            {
                "message": {"content": "all good"},
                "model": "llama3.2",
                "done": True,
                "done_reason": "stop",
            }
        )
        assert parsed.truncated is False
        assert parsed.finish_reason == "stop"

    def test_missing_done_reason_not_truncated(self) -> None:
        """Back-compat: a payload without done_reason is a clean finish."""
        parsed = OllamaAdapter().adapt_response(
            {"message": {"content": "ok"}, "model": "llama3.2", "done": True}
        )
        assert parsed.truncated is False
        assert parsed.finish_reason == "stop"


# ---------------------------------------------------------------------------
# Bedrock parity (Claude-on-Bedrock has the same max_tokens hazard)
# ---------------------------------------------------------------------------


class TestBedrockAdaptResponse:
    """Bedrock ``stopReason == 'max_tokens'`` sets the truncation flag.

    Bedrock runs Claude, so it shares the native Anthropic stop-reason
    vocabulary verbatim and normalizes ``finish_reason`` through the same shared
    ``normalize_claude_stop_reason`` helper — so ``finish_reason`` reads
    identically whether a Claude model is served via the native Anthropic API or
    Bedrock Converse (the raw ``stopReason`` is preserved on
    ``metadata['raw_finish_reason']``). Reproduce-first for the normalization:
    FAILS against the pre-fix adapter (raw ``max_tokens`` passthrough).
    """

    def test_max_tokens_is_truncated_and_normalized(self) -> None:
        parsed = BedrockConverseAdapter().adapt_response(
            {
                "output": {"message": {"content": [{"text": "cut"}]}},
                "stopReason": "max_tokens",
            },
            model="anthropic.claude-3-haiku-20240307-v1:0",
        )
        assert parsed.truncated is True
        # Normalized onto the canonical vocabulary, raw stopReason preserved —
        # identical to the native Anthropic provider for the same event.
        assert parsed.finish_reason == "length"
        assert parsed.metadata["raw_finish_reason"] == "max_tokens"

    def test_end_turn_normalized_to_stop_not_truncated(self) -> None:
        parsed = BedrockConverseAdapter().adapt_response(
            {
                "output": {"message": {"content": [{"text": "done"}]}},
                "stopReason": "end_turn",
            },
            model="anthropic.claude-3-haiku-20240307-v1:0",
        )
        assert parsed.truncated is False
        assert parsed.finish_reason == "stop"

    def test_tool_use_normalized_to_tool_calls(self) -> None:
        parsed = BedrockConverseAdapter().adapt_response(
            {
                "output": {
                    "message": {
                        "content": [
                            {"toolUse": {"toolUseId": "t1", "name": "s", "input": {}}}
                        ]
                    }
                },
                "stopReason": "tool_use",
            },
            model="anthropic.claude-3-haiku-20240307-v1:0",
        )
        assert parsed.truncated is False
        assert parsed.finish_reason == "tool_calls"


# ---------------------------------------------------------------------------
# Rebuild / deprecated-function_call paths preserve the flag
#
# The buffered/streaming complete paths route the response through
# _analyze_response, but two paths rebuild the response and could silently drop
# a newly-added field (or skip the warn hook): Ollama's <think>-tag extraction,
# and the deprecated function_call() entry point that surfaces a legacy dict.
# ---------------------------------------------------------------------------


class TestOllamaThinkTagReconstruction:
    """Ollama's ``<think>``-extraction rebuild must not drop the flag.

    A reasoning model (DeepSeek-R1, Qwen3) emits ``<think>...</think>`` then
    starts its visible answer and hits ``max_tokens`` mid-answer:
    ``done_reason == "length"`` → the adapter sets ``truncated=True``, and the
    ``<think>`` extraction then rebuilds the response to strip the thinking
    block. Reproduce-first: FAILS when the rebuild lists fields explicitly and
    omits ``truncated`` (silently False, warning lost), passes when it copies
    every field via ``dataclasses.replace``.
    """

    def _provider(self) -> OllamaProvider:
        return OllamaProvider(LLMConfig(provider="ollama", model="deepseek-r1"))

    def test_truncated_survives_think_extraction(self) -> None:
        response = LLMResponse(
            content="<think>reasoning</think>partial ans",
            model="deepseek-r1",
            finish_reason="length",
            truncated=True,
        )
        result = self._provider()._analyze_response(response)
        assert result.content == "partial ans"
        assert result.metadata["thinking"] == "reasoning"
        # The whole point of the flag: it must survive the rebuild.
        assert result.truncated is True

    def test_truncated_tool_call_survives_and_warns(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        response = LLMResponse(
            content="<think>plan</think>",
            model="deepseek-r1",
            finish_reason="length",
            truncated=True,
            tool_calls=[ToolCall(name="submit", parameters={"x": 1})],
        )
        with caplog.at_level(logging.WARNING, logger="dataknobs_llm.llm.base"):
            result = self._provider()._analyze_response(response)
        assert result.truncated is True
        assert any(
            "mid tool-call" in r.message and r.levelno == logging.WARNING
            for r in caplog.records
        )


class TestAnthropicFunctionCallTruncation:
    """The deprecated ``function_call()`` path preserves truncation + warns.

    ``function_call()`` parses via ``adapt_response`` (which sets ``truncated``
    and ``metadata['raw_finish_reason']``) then surfaces a legacy
    ``function_call`` dict. Reproduce-first: FAILS when it rebuilds a fresh
    ``LLMResponse`` dropping ``truncated`` / ``metadata`` and skipping the warn
    hook; passes when it routes through the shared ``_analyze_response`` choke
    point.
    """

    def _provider_returning(self, response: object) -> AnthropicProvider:
        provider = AnthropicProvider(
            LLMConfig(provider="anthropic", model="claude-3-sonnet")
        )

        class _Client:
            def __init__(self) -> None:
                self.messages = self

            async def create(self, **_kw: Any) -> object:
                return response

        provider._client = _Client()
        provider._is_initialized = True
        return provider

    async def test_truncated_tool_call_preserved_and_warns(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        provider = self._provider_returning(
            make_anthropic_response(
                [{"type": "tool_use", "id": "t1", "name": "submit", "input": {}}],
                stop_reason="max_tokens",
            )
        )
        with caplog.at_level(logging.WARNING, logger="dataknobs_llm.llm.base"):
            with pytest.warns(DeprecationWarning):
                result = await provider.function_call(
                    [LLMMessage(role="user", content="go")],
                    [{"name": "submit", "description": "d", "parameters": {}}],
                )
        assert result.truncated is True
        assert result.finish_reason == "length"
        assert result.metadata["raw_finish_reason"] == "max_tokens"
        # Legacy dict still surfaced for backward compatibility.
        assert result.function_call == {"name": "submit", "arguments": {}}
        # The dangerous truncated tool-call turn warned.
        assert any(
            "mid tool-call" in r.message and r.levelno == logging.WARNING
            for r in caplog.records
        )


# ---------------------------------------------------------------------------
# Base warning — shared _warn_if_truncated / _analyze_response
# ---------------------------------------------------------------------------


class TestTruncationWarning:
    """The base hook warns loudly on a truncated tool-call turn."""

    def _provider(self) -> EchoProvider:
        return EchoProvider(LLMConfig(provider="echo", model="echo-model"))

    def test_truncated_tool_call_warns(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        resp = tool_call_response("submit", {"x": 1}, truncated=True)
        with caplog.at_level(logging.WARNING, logger="dataknobs_llm.llm.base"):
            self._provider()._analyze_response(resp)
        assert any(
            "mid tool-call" in r.message and r.levelno == logging.WARNING
            for r in caplog.records
        )

    def test_truncated_text_only_is_info_not_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        resp = text_response("half a sen", truncated=True, finish_reason="length")
        with caplog.at_level(logging.INFO, logger="dataknobs_llm.llm.base"):
            self._provider()._analyze_response(resp)
        assert not any(r.levelno == logging.WARNING for r in caplog.records)
        assert any(
            "truncated" in r.message.lower() and r.levelno == logging.INFO
            for r in caplog.records
        )

    def test_not_truncated_no_log(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        resp = tool_call_response("submit", {"x": 1})
        with caplog.at_level(logging.INFO, logger="dataknobs_llm.llm.base"):
            self._provider()._analyze_response(resp)
        assert not any(
            "truncated" in r.message.lower() for r in caplog.records
        )


# ---------------------------------------------------------------------------
# Capture/replay serialization
# ---------------------------------------------------------------------------


class TestSerializationRoundTrip:
    """The flag round-trips; capture fixtures stay byte-identical when False."""

    def test_truncated_round_trips(self) -> None:
        resp = LLMResponse(content="x", model="m", truncated=True)
        restored = llm_response_from_dict(llm_response_to_dict(resp))
        assert restored.truncated is True

    def test_falsy_truncated_omitted_from_dict(self) -> None:
        d = llm_response_to_dict(LLMResponse(content="x", model="m"))
        assert "truncated" not in d

    def test_old_capture_without_key_loads_false(self) -> None:
        # A capture recorded before the field existed has no "truncated" key.
        restored = llm_response_from_dict({"content": "x", "model": "m"})
        assert restored.truncated is False


# ---------------------------------------------------------------------------
# Testing factories expose the flag
# ---------------------------------------------------------------------------


class TestTestingFactories:
    """The scripted-response factories can produce truncated responses."""

    def test_text_response_truncated(self) -> None:
        assert text_response("x", truncated=True).truncated is True

    def test_tool_call_response_truncated(self) -> None:
        resp = tool_call_response("submit", {"x": 1}, truncated=True)
        assert resp.truncated is True

    def test_defaults_not_truncated(self) -> None:
        assert text_response("x").truncated is False
        assert tool_call_response("submit").truncated is False
