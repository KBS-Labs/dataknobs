"""Tests for model-family request-shape constraints (S1).

Bug: ``AnthropicProvider`` forwards ``temperature`` (and other sampling
params) to any Anthropic model, but the Claude 5 model family rejects
``temperature`` with a hard 400. There was no model-family awareness — a
Claude-5-family bot config carrying ``temperature:`` produced a provider 400
the moment it ran.

Fix: a config-overridable ``ModelConstraints`` surface. A provider
auto-detects the family's request-shape rules (Claude 5 → rejects
``temperature``) and the base overlays any ``LLMConfig.constraints`` override,
so a consumer can declare/withdraw a rule at runtime without a dataknobs
release. The provider drops rejected params before the call — drop-and-warn,
never silently.

The end-to-end reproduce-first tests capture the kwargs the provider passes to
``messages.create`` via a minimal stand-in for ``anthropic.AsyncAnthropic``
(the sanctioned narrow case: no dataknobs testing construct produces a real
Anthropic request, and the stand-in exercises the real provider wiring end to
end). They FAIL against HEAD (``temperature`` forwarded for Claude 5) and pass
after the fix.
"""

from __future__ import annotations

import logging
from typing import Any

from dataknobs_llm.llm.base import LLMConfig, ModelConstraints
from dataknobs_llm.llm.providers.anthropic import AnthropicProvider

from test_anthropic_param_handling import make_anthropic_response


# ---------------------------------------------------------------------------
# Reusable stand-in for the Anthropic SDK client
# ---------------------------------------------------------------------------


class _CaptureAnthropicClient:
    """Records the kwargs passed to ``messages.create``.

    Minimal stand-in for ``anthropic.AsyncAnthropic`` — a sanctioned SDK
    stand-in (no dataknobs testing construct returns a real Anthropic
    request/response). Exercises the real ``AnthropicProvider.complete``
    wiring (``adapt_messages`` → ``_build_api_kwargs`` → ``messages.create``
    → ``adapt_response``) without a live API or the ``anthropic`` package.
    """

    def __init__(self) -> None:
        self.captured_kwargs: dict[str, Any] = {}
        # ``provider._client.messages.create`` → this object's ``create``.
        self.messages = self

    async def create(self, **kwargs: Any) -> object:
        self.captured_kwargs = kwargs
        return make_anthropic_response([{"type": "text", "text": "ok"}])


def _provider_with_capture(
    model: str, **config_kwargs: Any
) -> tuple[AnthropicProvider, _CaptureAnthropicClient]:
    """Build an initialised ``AnthropicProvider`` backed by a capture client."""
    provider = AnthropicProvider(
        LLMConfig(provider="anthropic", model=model, **config_kwargs)
    )
    client = _CaptureAnthropicClient()
    provider._client = client
    provider._is_initialized = True
    return provider, client


# ---------------------------------------------------------------------------
# End-to-end reproduce-first: rejected params are dropped before the API call
# ---------------------------------------------------------------------------


class TestModelFamilyParamRejection:
    """The provider drops family-rejected params before ``messages.create``."""

    async def test_temperature_dropped_for_claude_5(self) -> None:
        """Claude 5 rejects ``temperature`` → it must not reach the API."""
        provider, client = _provider_with_capture(
            "claude-sonnet-5", temperature=0.3
        )
        await provider.complete("hi")
        assert "temperature" not in client.captured_kwargs

    async def test_temperature_kept_for_claude_4_5(self) -> None:
        """Claude 4.5 still accepts ``temperature`` → forwarded unchanged."""
        provider, client = _provider_with_capture(
            "claude-haiku-4-5-20251001", temperature=0.3
        )
        await provider.complete("hi")
        assert client.captured_kwargs["temperature"] == 0.3

    async def test_temperature_kept_for_claude_opus_4_8(self) -> None:
        """Opus 4.8 is not Claude 5 → ``temperature`` forwarded."""
        provider, client = _provider_with_capture(
            "claude-opus-4-8", temperature=0.5
        )
        await provider.complete("hi")
        assert client.captured_kwargs["temperature"] == 0.5

    async def test_stream_drops_temperature_for_claude_5(self) -> None:
        """The streaming path shares the same choke point."""
        provider, _ = _provider_with_capture("claude-sonnet-5", temperature=0.3)

        captured: dict[str, Any] = {}

        class _StreamCtx:
            async def __aenter__(self) -> "_StreamCtx":
                return self

            async def __aexit__(self, *exc: object) -> None:
                return None

            def __aiter__(self) -> "_StreamCtx":
                return self

            async def __anext__(self) -> object:
                raise StopAsyncIteration

            async def get_final_message(self) -> object:
                return make_anthropic_response([{"type": "text", "text": "ok"}])

        def _stream(**kwargs: Any) -> _StreamCtx:
            captured.update(kwargs)
            return _StreamCtx()

        provider._client.messages.stream = _stream  # type: ignore[attr-defined]

        async for _ in provider.stream_complete("hi"):
            pass

        assert "temperature" not in captured

    def test_drop_emits_warning(self, caplog) -> None:
        """A dropped param must be logged, never silent."""
        provider = AnthropicProvider(
            LLMConfig(provider="anthropic", model="claude-sonnet-5", temperature=0.3)
        )
        with caplog.at_level(logging.WARNING):
            params = provider._build_api_kwargs(provider.config)
        assert "temperature" not in params
        assert any(
            "temperature" in rec.message and "claude-sonnet-5" in rec.message
            for rec in caplog.records
        )

    def test_no_warning_when_nothing_dropped(self, caplog) -> None:
        """No spurious warning for a family with no rejected params."""
        provider = AnthropicProvider(
            LLMConfig(
                provider="anthropic",
                model="claude-haiku-4-5-20251001",
                temperature=0.3,
            )
        )
        with caplog.at_level(logging.WARNING):
            params = provider._build_api_kwargs(provider.config)
        assert params["temperature"] == 0.3
        assert not caplog.records


# ---------------------------------------------------------------------------
# S1 surface: ModelConstraints detection + config override
# ---------------------------------------------------------------------------


class TestModelConstraintsResolution:
    """The resolved ``ModelConstraints`` surface and its config override."""

    def test_claude_5_detected_rejects_temperature(self) -> None:
        provider = AnthropicProvider(
            LLMConfig(provider="anthropic", model="claude-sonnet-5")
        )
        constraints = provider.get_constraints()
        assert "temperature" in constraints.rejected_params

    def test_claude_4_5_detected_rejects_nothing(self) -> None:
        provider = AnthropicProvider(
            LLMConfig(provider="anthropic", model="claude-haiku-4-5-20251001")
        )
        constraints = provider.get_constraints()
        assert constraints.rejected_params == frozenset()

    def test_anthropic_never_accepts_inline_system(self) -> None:
        """Every Anthropic model hoists system messages (read by #187)."""
        for model in ("claude-sonnet-5", "claude-haiku-4-5-20251001", "claude-3-opus"):
            provider = AnthropicProvider(
                LLMConfig(provider="anthropic", model=model)
            )
            assert provider.get_constraints().accepts_inline_system is False

    def test_config_override_adds_rejected_param(self) -> None:
        """A consumer can declare an extra rejected param without a release."""
        provider = AnthropicProvider(
            LLMConfig(
                provider="anthropic",
                model="claude-haiku-4-5-20251001",
                constraints={"rejected_params": ["top_p"]},
            )
        )
        constraints = provider.get_constraints()
        assert constraints.rejected_params == frozenset({"top_p"})

    def test_config_override_withdraws_stale_rule(self) -> None:
        """Passing an empty list withdraws the auto-detected rejection."""
        provider = AnthropicProvider(
            LLMConfig(
                provider="anthropic",
                model="claude-sonnet-5",
                constraints={"rejected_params": []},
            )
        )
        constraints = provider.get_constraints()
        assert constraints.rejected_params == frozenset()

    async def test_config_override_withdrawal_forwards_param(self) -> None:
        """Withdrawing the rule means the param reaches the API again."""
        provider, client = _provider_with_capture(
            "claude-sonnet-5",
            temperature=0.3,
            constraints={"rejected_params": []},
        )
        await provider.complete("hi")
        assert client.captured_kwargs["temperature"] == 0.3


class TestModelConstraintsDataclass:
    """Unit tests for the ``ModelConstraints`` value type."""

    def test_defaults_are_permissive(self) -> None:
        c = ModelConstraints()
        assert c.rejected_params == frozenset()
        assert c.accepts_inline_system is True
        assert c.max_tokens_ceiling is None

    def test_with_overrides_is_pure(self) -> None:
        base = ModelConstraints(rejected_params=frozenset({"temperature"}))
        overridden = base.with_overrides({"rejected_params": ["top_p"]})
        # Original is unchanged (frozen, pure overlay).
        assert base.rejected_params == frozenset({"temperature"})
        assert overridden.rejected_params == frozenset({"top_p"})

    def test_with_overrides_none_rejected_params_clears(self) -> None:
        base = ModelConstraints(rejected_params=frozenset({"temperature"}))
        assert base.with_overrides(
            {"rejected_params": None}
        ).rejected_params == frozenset()

    def test_with_overrides_absent_key_preserved(self) -> None:
        base = ModelConstraints(
            rejected_params=frozenset({"temperature"}),
            accepts_inline_system=False,
        )
        overridden = base.with_overrides({"max_tokens_ceiling": 8192})
        assert overridden.rejected_params == frozenset({"temperature"})
        assert overridden.accepts_inline_system is False
        assert overridden.max_tokens_ceiling == 8192
