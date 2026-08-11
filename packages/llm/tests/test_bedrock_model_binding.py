"""Tests for the Amazon Bedrock model-metadata binding.

The Bedrock provider hand-maintained capability / ceiling / price / validation
facts as scattered Python literals: a ``_KNOWN_MODEL_PREFIXES`` whitelist for
``validate_model``, a capability substring list that mis-detected vision (Nova
lite/pro, Llama-3.2 vision, and Pixtral got none), constraints that populated
only the Claude output ceiling (``max_input_tokens`` was never set → the input
budget was dead for Bedrock), and a per-1K price table wired to ``cost_usd``
only through the adapter.

This binding composes the model-metadata substrate — a bundled
``bedrock_models.yaml`` resource (full non-Claude profiles + Claude-on-Bedrock
pricing/availability) + the SHARED Claude family sources (ceilings +
capabilities, reused with the native Anthropic provider, no duplication) + a
corrected last-resort heuristic + config-override — into a per-provider
resolver, after which capability / constraint / pricing / validation are
one-line profile reads. It also wires ``cost_usd`` off the resolved per-Mtok
``ModelPricing`` (buffered AND streaming) and ships an opt-in
``ListFoundationModels`` live-availability source.

Every reproduce-first test below FAILS against HEAD (the stale literals) and
passes after the binding. Request-shaping / cost / live-availability tests drive
the real provider wiring through the sanctioned ``session.client(...)`` boundary
stub (no faithful local Bedrock emulator exists).
"""

from __future__ import annotations

from typing import Any

from dataknobs_llm.llm.base import LLMConfig, LLMResponse, ModelCapability
from dataknobs_llm.llm.model_profile import BundledResourceSource
from dataknobs_llm.llm.providers.bedrock import (
    BedrockProvider,
    _bedrock_availability_extractor,
)
from dataknobs_llm.tooling import model_limits

from _bedrock_stubs import _StubBedrockClient, _StubSession, _stub_provider


def _provider(model: str, **config_kwargs: Any) -> BedrockProvider:
    """A construction-only provider (no session) for pure detect-path reads."""
    return BedrockProvider(LLMConfig(provider="bedrock", model=model, **config_kwargs))


def _converse_client() -> _StubBedrockClient:
    """A Converse stub returning a minimal, valid completion with usage."""
    return _StubBedrockClient(
        converse_response={
            "output": {"message": {"content": [{"text": "ok"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
        }
    )


def _inference_config(client: _StubBedrockClient) -> dict[str, Any]:
    assert client.converse_calls, "no converse call was captured"
    return client.converse_calls[0].get("inferenceConfig", {})


# ---------------------------------------------------------------------------
# Capabilities — resolved off the profile (resource + shared Claude, corrected)
# ---------------------------------------------------------------------------


class TestCapabilities:
    """Capability detection is a one-line profile read, corrected for vision."""

    def test_nova_pro_has_vision(self) -> None:
        assert ModelCapability.VISION in _provider("amazon.nova-pro-v1:0").get_capabilities()

    def test_llama_3_2_90b_has_vision(self) -> None:
        """FAILS on HEAD: the old substring list gated vision on Claude/Nova, so
        the multimodal Llama-3.2 vision model resolved no VISION.
        """
        caps = _provider("meta.llama3-2-90b-instruct-v1:0").get_capabilities()
        assert ModelCapability.VISION in caps

    def test_pixtral_has_vision(self) -> None:
        """FAILS on HEAD: Pixtral (multimodal Mistral) resolved no VISION."""
        caps = _provider("mistral.pixtral-large-2502-v1:0").get_capabilities()
        assert ModelCapability.VISION in caps

    def test_nova_micro_has_no_vision(self) -> None:
        """Nova micro is text-only → the resource corrects the old over-detection
        that gave every ``nova`` id VISION.
        """
        caps = _provider("amazon.nova-micro-v1:0").get_capabilities()
        assert ModelCapability.VISION not in caps

    def test_embedding_model_disjoint(self) -> None:
        assert _provider("amazon.titan-embed-text-v2:0").get_capabilities() == [
            ModelCapability.EMBEDDINGS
        ]

    def test_claude_on_bedrock_reuses_shared_caps(self) -> None:
        """A Claude-on-Bedrock id resolves the SHARED Claude capability set
        (proving the reuse composition, not a Bedrock-local copy).
        """
        caps = _provider("anthropic.claude-3-5-sonnet-20240620-v1:0").get_capabilities()
        assert ModelCapability.VISION in caps
        assert ModelCapability.FUNCTION_CALLING in caps
        assert ModelCapability.JSON_MODE in caps

    def test_unknown_future_nova_gets_heuristic_base(self) -> None:
        """An unlisted future ``amazon.nova-*`` falls to the heuristic base set."""
        caps = _provider("amazon.nova-ultra-v9:0").get_capabilities()
        assert ModelCapability.CHAT in caps
        assert ModelCapability.FUNCTION_CALLING in caps


# ---------------------------------------------------------------------------
# Constraints — ceilings, max_input_tokens (input budget), rejected params
# ---------------------------------------------------------------------------


class TestConstraints:
    """Request-shape constraints resolve off the profile."""

    def test_claude_on_bedrock_output_ceiling(self) -> None:
        c = _provider("anthropic.claude-opus-4-8-v1:0").get_constraints()
        assert c.max_tokens_ceiling == 128000

    def test_claude_on_bedrock_input_window_now_populated(self) -> None:
        """FAILS on HEAD: Bedrock never set ``max_input_tokens`` → the input
        budget was dead. The shared Claude ceiling resource now supplies it.
        """
        c = _provider("anthropic.claude-opus-4-8-v1:0").get_constraints()
        assert c.max_input_tokens == 1000000

    def test_non_claude_ceiling_and_context(self) -> None:
        """FAILS on HEAD: non-Claude models had no ceiling data at all."""
        c = _provider("amazon.nova-pro-v1:0").get_constraints()
        assert c.max_tokens_ceiling == 5120
        assert c.max_input_tokens == 300000

    def test_claude_5_rejects_temperature(self) -> None:
        c = _provider("anthropic.claude-sonnet-5-v1:0").get_constraints()
        assert "temperature" in c.rejected_params

    def test_cross_region_resolves_same_family(self) -> None:
        base = _provider("anthropic.claude-opus-4-8-20260101-v1:0").get_constraints()
        regioned = _provider("us.anthropic.claude-opus-4-8-20260101-v1:0").get_constraints()
        assert regioned.max_tokens_ceiling == base.max_tokens_ceiling
        assert regioned.max_tokens_ceiling == 128000

    def test_config_override_wins_on_ceiling(self) -> None:
        c = _provider(
            "amazon.nova-pro-v1:0",
            model_profile_overrides={"max_output_tokens": 999},
        ).get_constraints()
        assert c.max_tokens_ceiling == 999


# ---------------------------------------------------------------------------
# validate_model — data-sourced availability (default, no control-plane call)
# ---------------------------------------------------------------------------


class TestValidateModel:
    """Default validate_model reads the data-sourced ``available`` facet."""

    async def test_listed_model_valid(self) -> None:
        assert await _provider("meta.llama3-1-70b-instruct-v1:0").validate_model() is True

    async def test_vendor_prefix_valid(self) -> None:
        """An unlisted id under a recognised vendor prefix validates via the
        heuristic ``available`` (the old permissive-prefix behavior, data-sourced).
        """
        assert await _provider("amazon.nova-ultra-v9:0").validate_model() is True

    async def test_region_prefixed_claude_valid(self) -> None:
        assert (
            await _provider("us.anthropic.claude-3-5-sonnet-20240620-v1:0").validate_model() is True
        )

    async def test_unknown_vendor_invalid(self) -> None:
        assert await _provider("gpt-4").validate_model() is False


# ---------------------------------------------------------------------------
# Request shaping — per-call kwargs routed through the choke point
# ---------------------------------------------------------------------------


class TestRequestShaping:
    """``complete`` routes per-call ``**kwargs`` through the shaping choke point."""

    async def test_max_tokens_kwarg_clamped_and_shaped(self) -> None:
        """FAILS on HEAD: a raw ``max_tokens=`` kwarg landed as an un-clamped
        top-level Converse key (a ValidationException) instead of the clamped
        ``inferenceConfig.maxTokens``.
        """
        client = _converse_client()
        provider = _stub_provider(
            LLMConfig(provider="bedrock", model="amazon.nova-pro-v1:0"),
            client,
        )
        await provider.complete("hi", max_tokens=500_000)
        assert _inference_config(client)["maxTokens"] == 5120
        assert "max_tokens" not in client.converse_calls[0]

    async def test_temperature_kwarg_dropped_for_claude_5(self) -> None:
        """A per-call ``temperature`` kwarg for Claude 5 is dropped, not sent."""
        client = _converse_client()
        provider = _stub_provider(
            LLMConfig(provider="bedrock", model="anthropic.claude-sonnet-5-v1:0"),
            client,
        )
        await provider.complete("hi", temperature=0.9)
        assert "temperature" not in _inference_config(client)

    async def test_wire_only_kwarg_passes_through(self) -> None:
        """A genuine wire-only Converse param (not a config field) is untouched.

        Regression guard mirroring the OpenAI ``response_format``-dict case: the
        fold must only route *shaped config-field* kwargs through shaping; a
        Converse-only param reaches the request verbatim.
        """
        client = _converse_client()
        provider = _stub_provider(
            LLMConfig(provider="bedrock", model="amazon.nova-pro-v1:0"),
            client,
        )
        extra = {"guardrailConfig": {"guardrailIdentifier": "g", "guardrailVersion": "1"}}
        await provider.complete("hi", additionalModelRequestFields={"x": 1}, **extra)
        sent = client.converse_calls[0]
        assert sent["additionalModelRequestFields"] == {"x": 1}
        assert sent["guardrailConfig"] == extra["guardrailConfig"]


# ---------------------------------------------------------------------------
# Pricing + cost — get_pricing (facts) + provider-stamped cost_usd
# ---------------------------------------------------------------------------


class TestPricingAndCost:
    """Profile-sourced pricing + provider-stamped cost (buffered and streaming)."""

    def test_get_pricing_reads_profile(self) -> None:
        """FAILS on HEAD: Bedrock had no ``_detect_pricing`` → get_pricing None."""
        pricing = _provider("amazon.nova-pro-v1:0").get_pricing()
        assert pricing is not None
        assert pricing.input_per_mtok == 0.8
        assert pricing.output_per_mtok == 3.2

    def test_get_pricing_unknown_none(self) -> None:
        assert _provider("acme.mystery-model-v9:0").get_pricing() is None

    def test_cost_back_compat_value(self) -> None:
        """The migrated per-Mtok price reproduces the old per-1K cost exactly."""
        provider = _provider("anthropic.claude-3-haiku-20240307-v1:0")
        response = LLMResponse(
            content="x",
            model="anthropic.claude-3-haiku-20240307-v1:0",
            usage={"prompt_tokens": 10, "completion_tokens": 5},
        )
        cost = provider.estimate_cost(response)
        assert cost is not None
        assert abs(cost - ((10 / 1000) * 0.00025 + (5 / 1000) * 0.00125)) < 1e-12

    async def test_complete_stamps_cost(self) -> None:
        """The buffered path stamps cost_usd from the resolved per-Mtok pricing."""
        client = _converse_client()
        provider = _stub_provider(
            LLMConfig(provider="bedrock", model="amazon.nova-pro-v1:0"),
            client,
        )
        result = await provider.complete("hi")
        # 10 input @ $0.8/Mtok + 5 output @ $3.2/Mtok
        expected = (10 / 1_000_000) * 0.8 + (5 / 1_000_000) * 3.2
        assert result.cost_usd is not None
        assert abs(result.cost_usd - expected) < 1e-12

    async def test_complete_unpriced_model_cost_none(self) -> None:
        client = _converse_client()
        provider = _stub_provider(
            LLMConfig(provider="bedrock", model="acme.mystery-model-v9:0"),
            client,
        )
        result = await provider.complete("hi")
        assert result.cost_usd is None

    async def test_streaming_final_chunk_carries_cost(self) -> None:
        """FAILS on HEAD: the stream path computed no cost at all."""
        client = _StubBedrockClient(
            stream_events=[
                {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "hi"}}},
                {"messageStop": {"stopReason": "end_turn"}},
                {"metadata": {"usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15}}},
            ]
        )
        provider = _stub_provider(
            LLMConfig(provider="bedrock", model="amazon.nova-pro-v1:0"),
            client,
        )
        chunks = [c async for c in provider.stream_complete("hi")]
        final = chunks[-1]
        assert final.is_final
        expected = (10 / 1_000_000) * 0.8 + (5 / 1_000_000) * 3.2
        assert final.cost_usd is not None
        assert abs(final.cost_usd - expected) < 1e-12


# ---------------------------------------------------------------------------
# Opt-in ListFoundationModels live availability
# ---------------------------------------------------------------------------


class _CatalogClient(_StubBedrockClient):
    """Adds a control-plane ``list_foundation_models`` to the boundary stub."""

    def __init__(self, model_ids: list[str]) -> None:
        super().__init__()
        self._summaries = [{"modelId": m} for m in model_ids]
        self.list_calls = 0

    async def list_foundation_models(self, **_kwargs: Any) -> dict[str, Any]:
        self.list_calls += 1
        return {"modelSummaries": self._summaries}


def _live_provider(model: str, catalog: list[str]) -> BedrockProvider:
    provider = BedrockProvider(
        LLMConfig(
            provider="bedrock",
            model=model,
            options={"model_availability_live": True},
        )
    )
    provider._session = _StubSession(_CatalogClient(catalog))
    provider._is_initialized = True
    return provider


class TestLiveAvailability:
    """Opt-in live availability resolves against the account catalog."""

    def test_extractor_marks_available(self) -> None:
        assert _bedrock_availability_extractor({"modelId": "x"}).available is True

    async def test_present_model_valid(self) -> None:
        provider = _live_provider(
            "amazon.nova-pro-v1:0", ["amazon.nova-pro-v1:0", "meta.llama3-8b-instruct-v1:0"]
        )
        assert await provider.validate_model() is True

    async def test_absent_model_invalid(self) -> None:
        """A model absent from the account catalog is False — even though it is a
        recognised vendor id the *maintained* resource would call available.
        """
        provider = _live_provider("amazon.nova-pro-v1:0", ["meta.llama3-8b-instruct-v1:0"])
        assert await provider.validate_model() is False

    async def test_default_off_is_maintained_availability(self) -> None:
        """Without the opt-in, validate_model never calls the control-plane and
        uses the maintained ``available`` facet (byte-identical to the default).
        """
        provider = _provider("amazon.nova-pro-v1:0")
        assert provider._availability_source is None
        assert await provider.validate_model() is True


# ---------------------------------------------------------------------------
# Resource-load unit
# ---------------------------------------------------------------------------


class TestResource:
    """The bundled bedrock_models.yaml loads and resolves through the substrate."""

    def test_non_claude_full_profile(self) -> None:
        source = BundledResourceSource.from_resource(
            "dataknobs_llm.llm.providers", "data/bedrock_models.yaml"
        )
        profile = source.resolve("amazon.nova-pro-v1:0")
        assert profile.max_output_tokens == 5120
        assert profile.context_window == 300000
        assert profile.pricing is not None
        assert ModelCapability.VISION in (profile.capabilities or frozenset())
        assert profile.available is True

    def test_claude_entry_carries_only_bedrock_facets(self) -> None:
        """Claude-on-Bedrock entries carry ONLY pricing + availability — caps and
        ceilings come from the shared Claude sources (no duplication).
        """
        source = BundledResourceSource.from_resource(
            "dataknobs_llm.llm.providers", "data/bedrock_models.yaml"
        )
        profile = source.resolve("anthropic.claude-3-haiku-20240307-v1:0")
        assert profile.pricing is not None
        assert profile.available is True
        assert profile.capabilities is None
        assert profile.max_output_tokens is None
        assert profile.context_window is None


# ---------------------------------------------------------------------------
# tooling/model_limits.py --provider bedrock (availability/modality drift)
# ---------------------------------------------------------------------------


class _CatalogControlPlaneClient:
    """Stand-in for a ``bedrock`` control-plane client (sync list, like boto3)."""

    def __init__(self, summaries: list[dict[str, Any]]) -> None:
        self._summaries = summaries

    def list_foundation_models(self, **_kwargs: Any) -> dict[str, Any]:
        return {"modelSummaries": self._summaries}


class _AsyncCatalogClient:
    """Stand-in whose ``list_foundation_models`` is awaitable (aioboto3 shape)."""

    def __init__(self, summaries: list[dict[str, Any]]) -> None:
        self._summaries = summaries

    async def list_foundation_models(self, **_kwargs: Any) -> dict[str, Any]:
        return {"modelSummaries": self._summaries}


def _bedrock_resource(tmp_path: Any, body: str) -> Any:
    path = tmp_path / "bedrock.yaml"
    path.write_text(body, encoding="utf-8")
    return path


class TestTooling:
    """``--provider bedrock`` diffs the catalog's availability + modalities."""

    async def test_fetch_handles_sync_and_async_clients(self) -> None:
        summaries = [
            {
                "modelId": "amazon.nova-pro-v1:0",
                "inputModalities": ["TEXT", "IMAGE"],
                "responseStreamingSupported": True,
            }
        ]
        sync_facts = await model_limits.fetch_bedrock_facts(_CatalogControlPlaneClient(summaries))
        async_facts = await model_limits.fetch_bedrock_facts(_AsyncCatalogClient(summaries))
        assert sync_facts == async_facts
        assert sync_facts["amazon.nova-pro-v1:0"] == {"vision": True, "streaming": True}

    def test_check_ok_when_matching(self, tmp_path: Any) -> None:
        path = _bedrock_resource(
            tmp_path,
            "models:\n  amazon.nova-pro:\n"
            "    capabilities: [chat, vision, streaming]\n    available: true\n",
        )
        client = _CatalogControlPlaneClient(
            [
                {
                    "modelId": "amazon.nova-pro-v1:0",
                    "inputModalities": ["TEXT", "IMAGE"],
                    "responseStreamingSupported": True,
                }
            ]
        )
        rc = model_limits.main(
            ["--provider", "bedrock", "--check"], client=client, resource_path=path
        )
        assert rc == 0

    def test_check_flags_uncovered_model(self, tmp_path: Any) -> None:
        """A model in the catalog with no resource family is drift (AWS added it)."""
        path = _bedrock_resource(
            tmp_path,
            "models:\n  amazon.nova-pro:\n"
            "    capabilities: [chat, vision, streaming]\n    available: true\n",
        )
        client = _CatalogControlPlaneClient(
            [
                {
                    "modelId": "amazon.nova-pro-v1:0",
                    "inputModalities": ["TEXT", "IMAGE"],
                    "responseStreamingSupported": True,
                },
                {
                    "modelId": "acme.brand-new-v1:0",
                    "inputModalities": ["TEXT"],
                    "responseStreamingSupported": True,
                },
            ]
        )
        rc = model_limits.main(
            ["--provider", "bedrock", "--check"], client=client, resource_path=path
        )
        assert rc == 1

    def test_check_flags_modality_drift(self, tmp_path: Any) -> None:
        """A model that gained vision in the catalog but not the resource is drift."""
        path = _bedrock_resource(
            tmp_path,
            "models:\n  amazon.nova-pro:\n"
            "    capabilities: [chat, streaming]\n    available: true\n",
        )
        client = _CatalogControlPlaneClient(
            [
                {
                    "modelId": "amazon.nova-pro-v1:0",
                    "inputModalities": ["TEXT", "IMAGE"],
                    "responseStreamingSupported": True,
                }
            ]
        )
        rc = model_limits.main(
            ["--provider", "bedrock", "--check"], client=client, resource_path=path
        )
        assert rc == 1

    def test_claude_live_ids_are_skipped(self, tmp_path: Any) -> None:
        """A Claude live id is skipped — its modalities come from the shared
        Claude source, not the modality-less Bedrock Claude entry, so no drift.
        """
        path = _bedrock_resource(
            tmp_path,
            "models:\n  anthropic.claude-3-haiku:\n    available: true\n",
        )
        client = _CatalogControlPlaneClient(
            [
                {
                    "modelId": "anthropic.claude-3-haiku-20240307-v1:0",
                    "inputModalities": ["TEXT", "IMAGE"],
                    "responseStreamingSupported": True,
                }
            ]
        )
        rc = model_limits.main(
            ["--provider", "bedrock", "--check"], client=client, resource_path=path
        )
        assert rc == 0

    def test_update_unsupported_for_bedrock(self, tmp_path: Any) -> None:
        """--update is rejected for bedrock (ceilings/pricing aren't live-sourced)."""
        path = _bedrock_resource(tmp_path, "models: {}\n")
        client = _CatalogControlPlaneClient([])
        rc = model_limits.main(
            ["--provider", "bedrock", "--update"], client=client, resource_path=path
        )
        assert rc == 2

    def test_anthropic_remains_default(self, tmp_path: Any) -> None:
        """No --provider still targets anthropic (byte-identical dispatch)."""
        path = tmp_path / "limits.yaml"
        path.write_text("models:\n  claude-a: 100\n", encoding="utf-8")

        class _M:
            def __init__(self, mid: str, mt: int) -> None:
                self.id, self.max_tokens = mid, mt

        class _Models:
            def list(self, **_k: Any) -> Any:
                async def _gen() -> Any:
                    yield _M("claude-a", 100)

                return _gen()

        class _Client:
            models = _Models()

        rc = model_limits.main(["--check"], client=_Client(), resource_path=path)
        assert rc == 0
