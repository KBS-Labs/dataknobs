"""Tests for Anthropic provider parameter handling.

Bug: Anthropic provider unconditionally sends both temperature and top_p.
The Anthropic API rejects requests with both:
  400 Bad Request: temperature and top_p cannot both be specified

Root cause: LLMConfig defaults temperature=0.7 and top_p=1.0 (non-None),
making "not set" indistinguishable from "explicitly set."
"""

from __future__ import annotations

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

    _build_api_params() is a pure method (LLMConfig -> dict) that does not
    require an Anthropic client, so we call it directly on an uninitialised
    provider instance — no mocks needed.
    """

    def _make_provider(self, **config_kwargs):
        """Create an AnthropicProvider without initialising the client."""
        from dataknobs_llm.llm.providers.anthropic import AnthropicProvider

        config = LLMConfig(provider="anthropic", **config_kwargs)
        return AnthropicProvider(config)

    def test_default_config_sends_neither_temp_nor_top_p(self):
        """Default config should not send temperature or top_p."""
        provider = self._make_provider(model="claude-3-haiku-20240307")
        params = provider._build_api_params(provider.config)
        assert "temperature" not in params
        assert "top_p" not in params

    def test_only_temperature_sent_when_set(self):
        """Only temperature should be sent when explicitly set."""
        provider = self._make_provider(
            model="claude-3-haiku-20240307", temperature=0.5,
        )
        params = provider._build_api_params(provider.config)
        assert params["temperature"] == 0.5
        assert "top_p" not in params

    def test_only_top_p_sent_when_set(self):
        """Only top_p should be sent when explicitly set."""
        provider = self._make_provider(
            model="claude-3-haiku-20240307", top_p=0.9,
        )
        params = provider._build_api_params(provider.config)
        assert params["top_p"] == 0.9
        assert "temperature" not in params

    def test_both_sent_when_both_explicitly_set(self):
        """Both should be sent when both explicitly set."""
        provider = self._make_provider(
            model="claude-3-haiku-20240307", temperature=0.5, top_p=0.9,
        )
        params = provider._build_api_params(provider.config)
        assert params["temperature"] == 0.5
        assert params["top_p"] == 0.9

    def test_max_tokens_defaults_to_1024(self):
        """max_tokens should default to 1024 when not set."""
        provider = self._make_provider(model="claude-3-haiku-20240307")
        params = provider._build_api_params(provider.config)
        assert params["max_tokens"] == 1024

    def test_max_tokens_uses_explicit_value(self):
        """max_tokens should use explicitly set value."""
        provider = self._make_provider(
            model="claude-3-haiku-20240307", max_tokens=2048,
        )
        params = provider._build_api_params(provider.config)
        assert params["max_tokens"] == 2048

    def test_stop_sequences_included(self):
        """stop_sequences should be forwarded to API params."""
        provider = self._make_provider(
            model="claude-3-haiku-20240307", stop_sequences=["STOP", "END"],
        )
        params = provider._build_api_params(provider.config)
        assert params["stop_sequences"] == ["STOP", "END"]

    def test_model_always_included(self):
        """model should always be in API params."""
        provider = self._make_provider(model="claude-3-haiku-20240307")
        params = provider._build_api_params(provider.config)
        assert params["model"] == "claude-3-haiku-20240307"
