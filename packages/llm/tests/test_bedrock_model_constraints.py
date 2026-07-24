"""Model-family request-shape constraints for the Bedrock (Claude) provider.

Bedrock Converse runs the *same* Claude models as the native Anthropic
provider, so the same per-model ``max_tokens`` output ceilings and the same
Claude-5 ``temperature`` rejection apply — a Claude model's output ceiling and
its ``temperature`` support are properties of the *model*, not the endpoint.

Before this change the clamp/drop lived only in the Anthropic provider's
``_build_api_kwargs``; Bedrock built its Converse request straight from
``adapt_config`` with no constraint awareness, so a Claude-on-Bedrock request
that asked for more output tokens than the model grants was sent unclamped
(risking truncation / a 400), and a Claude-5-on-Bedrock request carried
``temperature`` the model rejects.

The fix moves the clamp/drop into the shared, provider-agnostic
``LLMProvider._apply_request_constraints`` (canonical config space) and gives
Bedrock a ``_detect_constraints`` that sources Claude ceilings + the Claude-5
temperature rule from the shared ``_claude_shared`` helper. These reproduce-first
tests drive the real ``BedrockProvider.complete`` wiring through the sanctioned
boundary stub (``session.client("bedrock-runtime")``) and capture the Converse
request kwargs. Each FAILS against the pre-change provider (no clamp / no drop
on Bedrock) and passes after.
"""

from __future__ import annotations

import logging
from typing import Any

from dataknobs_llm.llm.base import LLMConfig
from dataknobs_llm.llm.providers.bedrock import BedrockProvider

from test_bedrock_provider import _StubBedrockClient, _stub_provider


def _converse_client() -> _StubBedrockClient:
    """A Converse stub returning a minimal, valid completion."""
    return _StubBedrockClient(
        converse_response={
            "output": {"message": {"content": [{"text": "ok"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
        }
    )


def _inference_config(client: _StubBedrockClient) -> dict[str, Any]:
    """The ``inferenceConfig`` block from the single captured Converse call."""
    assert client.converse_calls, "no converse call was captured"
    return client.converse_calls[0].get("inferenceConfig", {})


# ---------------------------------------------------------------------------
# max_tokens clamp on Bedrock (reproduce-first)
# ---------------------------------------------------------------------------


class TestBedrockMaxTokensClamp:
    """``max_tokens`` clamps to the Claude family ceiling on Bedrock too."""

    async def test_clamp_via_config_override(self) -> None:
        """A ceiling below the request clamps ``maxTokens`` in the Converse call.

        Seed-independent (ceiling from the config override). Reproduce-first:
        FAILS on the pre-change provider (Bedrock never clamped → ``maxTokens``
        stays 500).
        """
        client = _converse_client()
        provider = _stub_provider(
            LLMConfig(
                provider="bedrock",
                model="anthropic.claude-sonnet-5-v1:0",
                max_tokens=500,
                constraints={"max_tokens_ceiling": 100},
            ),
            client,
        )
        await provider.complete("hi")
        assert _inference_config(client)["maxTokens"] == 100

    async def test_clamp_from_bundled_resource_via_region_prefixed_id(
        self,
    ) -> None:
        """The bundled Claude ceiling resolves through a region-prefixed id.

        ``us.anthropic.claude-opus-4-8-...`` canonicalizes + substring-matches
        the family key ``claude-opus-4-8`` (128000) in the bundled resource, so
        an over-ceiling request clamps down — proving the shared resource
        resolution serves Bedrock ids, not only native Anthropic ids.
        """
        client = _converse_client()
        provider = _stub_provider(
            LLMConfig(
                provider="bedrock",
                model="us.anthropic.claude-opus-4-8-20260101-v1:0",
                max_tokens=500_000,
            ),
            client,
        )
        await provider.complete("hi")
        assert _inference_config(client)["maxTokens"] == 128000

    async def test_no_clamp_for_non_claude_model(self) -> None:
        """A non-Claude Bedrock model has no ceiling data → no clamp."""
        client = _converse_client()
        provider = _stub_provider(
            LLMConfig(
                provider="bedrock",
                model="meta.llama3-1-70b-instruct-v1:0",
                max_tokens=500_000,
            ),
            client,
        )
        await provider.complete("hi")
        assert _inference_config(client)["maxTokens"] == 500_000

    async def test_no_clamp_under_ceiling(self) -> None:
        """A request at/below the ceiling passes through untouched."""
        client = _converse_client()
        provider = _stub_provider(
            LLMConfig(
                provider="bedrock",
                model="anthropic.claude-opus-4-8-v1:0",
                max_tokens=1000,
            ),
            client,
        )
        await provider.complete("hi")
        assert _inference_config(client)["maxTokens"] == 1000

    def test_clamp_emits_warning(self, caplog) -> None:
        """The clamp is warn-not-silent, naming the model."""
        client = _converse_client()
        provider = _stub_provider(
            LLMConfig(
                provider="bedrock",
                model="anthropic.claude-sonnet-5-v1:0",
                max_tokens=500,
                constraints={"max_tokens_ceiling": 100},
            ),
            client,
        )
        with caplog.at_level(logging.WARNING):
            provider._apply_request_constraints(provider.config)
        assert any(
            "max_tokens" in rec.message and "claude-sonnet-5" in rec.message
            for rec in caplog.records
        )


# ---------------------------------------------------------------------------
# Claude-5 temperature rejection on Bedrock (reproduce-first)
# ---------------------------------------------------------------------------


class TestBedrockTemperatureRejection:
    """Claude 5 rejects ``temperature`` on Bedrock exactly as on the native API."""

    async def test_temperature_dropped_for_claude_5(self) -> None:
        """Claude 5 does not support ``temperature`` → it must not be sent.

        Reproduce-first: FAILS on the pre-change provider (Bedrock forwarded
        ``temperature`` for every model).
        """
        client = _converse_client()
        provider = _stub_provider(
            LLMConfig(
                provider="bedrock",
                model="anthropic.claude-sonnet-5-v1:0",
                temperature=0.3,
            ),
            client,
        )
        await provider.complete("hi")
        assert "temperature" not in _inference_config(client)

    async def test_temperature_kept_for_claude_3(self) -> None:
        """A non-Claude-5 model still forwards ``temperature`` unchanged."""
        client = _converse_client()
        provider = _stub_provider(
            LLMConfig(
                provider="bedrock",
                model="anthropic.claude-3-haiku-20240307-v1:0",
                temperature=0.3,
            ),
            client,
        )
        await provider.complete("hi")
        assert _inference_config(client)["temperature"] == 0.3


# ---------------------------------------------------------------------------
# The constraints surface is config-overridable on Bedrock
# ---------------------------------------------------------------------------


class TestBedrockConstraintsSurface:
    def test_detect_claude_5_rejects_temperature(self) -> None:
        provider = BedrockProvider(
            LLMConfig(provider="bedrock", model="anthropic.claude-opus-5-v1:0")
        )
        assert "temperature" in provider.get_constraints().rejected_params

    def test_detect_non_claude_is_permissive(self) -> None:
        provider = BedrockProvider(
            LLMConfig(provider="bedrock", model="amazon.nova-pro-v1:0")
        )
        constraints = provider.get_constraints()
        assert constraints.rejected_params == frozenset()
        assert constraints.max_tokens_ceiling is None

    def test_config_override_withdraws_temperature_rule(self) -> None:
        """A consumer can withdraw the auto-detected rejection at runtime."""
        provider = BedrockProvider(
            LLMConfig(
                provider="bedrock",
                model="anthropic.claude-sonnet-5-v1:0",
                constraints={"rejected_params": []},
            )
        )
        assert provider.get_constraints().rejected_params == frozenset()
