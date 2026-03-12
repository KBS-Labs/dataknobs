"""Tests for Anthropic provider parameter handling.

Bug: Anthropic provider unconditionally sends both temperature and top_p.
The Anthropic API rejects requests with both:
  400 Bad Request: temperature and top_p cannot both be specified

Root cause: LLMConfig defaults temperature=0.7 and top_p=1.0 (non-None),
making "not set" indistinguishable from "explicitly set."
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dataknobs_llm.llm.base import LLMConfig


# ---------------------------------------------------------------------------
# LLMConfig.generation_params() tests
# ---------------------------------------------------------------------------


class TestGenerationParams:
    """Test LLMConfig.generation_params() returns only explicitly-set values."""

    def test_default_config_returns_empty(self):
        """Default LLMConfig should return no generation params."""
        config = LLMConfig(provider="anthropic", model="claude-3-haiku")
        params = config.generation_params()
        assert "temperature" not in params
        assert "top_p" not in params
        assert "frequency_penalty" not in params
        assert "presence_penalty" not in params

    def test_only_temperature_set(self):
        """Only temperature should appear when explicitly set."""
        config = LLMConfig(provider="anthropic", model="claude-3-haiku", temperature=0.5)
        params = config.generation_params()
        assert params["temperature"] == 0.5
        assert "top_p" not in params

    def test_only_top_p_set(self):
        """Only top_p should appear when explicitly set."""
        config = LLMConfig(provider="anthropic", model="claude-3-haiku", top_p=0.9)
        params = config.generation_params()
        assert params["top_p"] == 0.9
        assert "temperature" not in params

    def test_both_set(self):
        """Both should appear when both explicitly set."""
        config = LLMConfig(
            provider="anthropic", model="claude-3-haiku",
            temperature=0.5, top_p=0.9,
        )
        params = config.generation_params()
        assert params["temperature"] == 0.5
        assert params["top_p"] == 0.9

    def test_max_tokens_included(self):
        """max_tokens should appear when set."""
        config = LLMConfig(provider="anthropic", model="claude-3-haiku", max_tokens=1000)
        params = config.generation_params()
        assert params["max_tokens"] == 1000

    def test_stop_sequences_included(self):
        """stop_sequences should appear when set."""
        config = LLMConfig(
            provider="anthropic", model="claude-3-haiku",
            stop_sequences=["STOP"],
        )
        params = config.generation_params()
        assert params["stop_sequences"] == ["STOP"]

    def test_seed_included(self):
        """seed should appear when set."""
        config = LLMConfig(provider="anthropic", model="claude-3-haiku", seed=42)
        params = config.generation_params()
        assert params["seed"] == 42


# ---------------------------------------------------------------------------
# Anthropic provider _build_api_params() tests
# ---------------------------------------------------------------------------


class TestAnthropicBuildApiParams:
    """Test that Anthropic provider builds correct API params.

    Mock justification: We need to verify what parameters are sent to the
    Anthropic API client. This tests request construction, not behavior.
    EchoProvider cannot be used here.
    """

    def _make_provider(self, **config_kwargs: Any):
        """Create an AnthropicProvider with a mocked client."""
        from dataknobs_llm.llm.providers.anthropic import AnthropicProvider

        config = LLMConfig(provider="anthropic", **config_kwargs)
        provider = AnthropicProvider(config)
        provider._client = MagicMock()
        provider._is_initialized = True
        return provider

    @pytest.mark.asyncio
    async def test_default_config_sends_neither_temp_nor_top_p(self):
        """Default config should not send temperature or top_p."""
        provider = self._make_provider(model="claude-3-haiku-20240307")
        provider._client.messages.create = AsyncMock(return_value=MagicMock(
            content=[MagicMock(text="response", type="text")],
            model="claude-3-haiku-20240307",
            stop_reason="end_turn",
            usage=MagicMock(input_tokens=10, output_tokens=5),
        ))

        await provider.complete("Hello")

        call_kwargs = provider._client.messages.create.call_args
        # Neither temperature nor top_p should be in the kwargs
        assert "temperature" not in call_kwargs.kwargs and (
            not call_kwargs.args or "temperature" not in str(call_kwargs)
        ), f"temperature should not be sent. Got: {call_kwargs}"

    @pytest.mark.asyncio
    async def test_only_temperature_sent_when_set(self):
        """Only temperature should be sent when explicitly set."""
        provider = self._make_provider(
            model="claude-3-haiku-20240307", temperature=0.5,
        )
        provider._client.messages.create = AsyncMock(return_value=MagicMock(
            content=[MagicMock(text="response", type="text")],
            model="claude-3-haiku-20240307",
            stop_reason="end_turn",
            usage=MagicMock(input_tokens=10, output_tokens=5),
        ))

        await provider.complete("Hello")

        kwargs = provider._client.messages.create.call_args.kwargs
        assert kwargs.get("temperature") == 0.5
        assert "top_p" not in kwargs

    @pytest.mark.asyncio
    async def test_only_top_p_sent_when_set(self):
        """Only top_p should be sent when explicitly set."""
        provider = self._make_provider(
            model="claude-3-haiku-20240307", top_p=0.9,
        )
        provider._client.messages.create = AsyncMock(return_value=MagicMock(
            content=[MagicMock(text="response", type="text")],
            model="claude-3-haiku-20240307",
            stop_reason="end_turn",
            usage=MagicMock(input_tokens=10, output_tokens=5),
        ))

        await provider.complete("Hello")

        kwargs = provider._client.messages.create.call_args.kwargs
        assert kwargs.get("top_p") == 0.9
        assert "temperature" not in kwargs

    @pytest.mark.asyncio
    async def test_both_sent_when_both_explicitly_set(self):
        """Both should be sent when both explicitly set."""
        provider = self._make_provider(
            model="claude-3-haiku-20240307", temperature=0.5, top_p=0.9,
        )
        provider._client.messages.create = AsyncMock(return_value=MagicMock(
            content=[MagicMock(text="response", type="text")],
            model="claude-3-haiku-20240307",
            stop_reason="end_turn",
            usage=MagicMock(input_tokens=10, output_tokens=5),
        ))

        await provider.complete("Hello")

        kwargs = provider._client.messages.create.call_args.kwargs
        assert kwargs.get("temperature") == 0.5
        assert kwargs.get("top_p") == 0.9
