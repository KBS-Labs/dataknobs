"""Tests for the OpenAI model-metadata binding.

The OpenAI provider hand-maintained capability/ceiling/pricing facts as scattered
Python literals: capability substring lists that miss current families (``gpt-5``
gets no function-calling / JSON / vision), no request-shape constraints at all
(``max_tokens`` never clamped to the model ceiling, the reasoning-family
``temperature`` rejection and the ``max_tokens`` → ``max_completion_tokens`` rename
unhandled), and a stale per-1K pricing table that is not provider-wired.

This binding composes the model-metadata substrate (a bundled
``openai_models.yaml`` resource + a corrected last-resort heuristic +
config-override) into a per-provider resolver, after which capability /
constraint / pricing detection are one-line profile reads. It also lights up the
shared request-shaping choke point for OpenAI (clamp ``max_tokens``, drop rejected
sampling params, apply ``param_remaps``) that was previously dead for this
provider.

Every reproduce-first test below FAILS against HEAD (the stale literals) and
passes after the binding. The request-shaping tests capture the kwargs the
provider passes to ``chat.completions.create`` via a minimal stand-in for
``openai.AsyncOpenAI`` — the sanctioned narrow SDK stand-in (no dataknobs testing
construct produces a real OpenAI request), exercising the real provider wiring
end to end without a live API.
"""

from __future__ import annotations

import types
from typing import Any

from dataknobs_llm.llm.base import LLMConfig, LLMResponse, ModelCapability
from dataknobs_llm.llm.model_profile import BundledResourceSource, ModelPricing
from dataknobs_llm.llm.providers.openai import OpenAIProvider
from dataknobs_llm.llm.utils import CostCalculator


# ---------------------------------------------------------------------------
# Minimal stand-in for the OpenAI SDK client
# ---------------------------------------------------------------------------


class _StubMessage:
    def __init__(self, content: str) -> None:
        self.content = content
        self.function_call = None


class _StubChoice:
    def __init__(self, content: str, finish_reason: str = "stop") -> None:
        self.message = _StubMessage(content)
        self.finish_reason = finish_reason


class _StubUsage:
    def __init__(self) -> None:
        self.prompt_tokens = 10
        self.completion_tokens = 5
        self.total_tokens = 15


class _StubChatResponse:
    def __init__(self, model: str) -> None:
        self.choices = [_StubChoice("ok")]
        self.model = model
        self.usage = _StubUsage()


class _StubCompletions:
    def __init__(self, outer: _CaptureOpenAIClient) -> None:
        self._outer = outer

    async def create(self, **kwargs: Any) -> _StubChatResponse:
        self._outer.captured_kwargs = kwargs
        return _StubChatResponse(str(kwargs.get("model", "?")))


class _StubModelList:
    def __init__(self, ids: list[str]) -> None:
        self.data = [types.SimpleNamespace(id=i) for i in ids]


class _StubModels:
    def __init__(self, ids: list[str]) -> None:
        self._ids = ids

    async def list(self, **_kwargs: Any) -> _StubModelList:
        return _StubModelList(self._ids)


class _CaptureOpenAIClient:
    """Records kwargs passed to ``chat.completions.create``.

    Minimal ``openai.AsyncOpenAI`` stand-in exercising the real
    ``OpenAIProvider.complete`` wiring (``adapt_messages`` → ``_build_api_kwargs``
    → ``chat.completions.create`` → ``adapt_response``) without a live API.
    """

    def __init__(self, model_ids: tuple[str, ...] = ()) -> None:
        self.captured_kwargs: dict[str, Any] = {}
        self._completions = _StubCompletions(self)
        self.chat = types.SimpleNamespace(completions=self._completions)
        self.models = _StubModels(list(model_ids))

    async def close(self) -> None:  # pragma: no cover - lifecycle no-op
        pass


def _provider(model: str, **config_kwargs: Any) -> OpenAIProvider:
    """Build an initialised ``OpenAIProvider`` backed by a capture client."""
    provider = OpenAIProvider(
        LLMConfig(provider="openai", model=model, **config_kwargs)
    )
    provider._client = _CaptureOpenAIClient()
    provider._is_initialized = True
    return provider


# ---------------------------------------------------------------------------
# Capabilities — resolved off the profile (resource-primary)
# ---------------------------------------------------------------------------


class TestCapabilities:
    """Capability detection is a one-line profile read, corrected for new families."""

    def test_gpt5_has_tools_json_vision(self) -> None:
        """gpt-5 was missed by the stale substring lists (no tools/json/vision).

        FAILS on HEAD: ``gpt-5`` is not in the ``tool_capable`` list and vision
        is gated on ``gpt-4o``, so gpt-5 resolved to text/chat/streaming only.
        """
        caps = _provider("gpt-5").get_capabilities()
        assert ModelCapability.FUNCTION_CALLING in caps
        assert ModelCapability.JSON_MODE in caps
        assert ModelCapability.VISION in caps

    def test_gpt4o_still_has_capabilities(self) -> None:
        caps = _provider("gpt-4o").get_capabilities()
        assert ModelCapability.VISION in caps
        assert ModelCapability.FUNCTION_CALLING in caps

    def test_embedding_model_capability(self) -> None:
        caps = _provider("text-embedding-3-large").get_capabilities()
        assert ModelCapability.EMBEDDINGS in caps

    def test_unknown_future_model_gets_heuristic_caps(self) -> None:
        """An unlisted gpt-family model falls to the corrected heuristic."""
        caps = _provider("gpt-6-turbo").get_capabilities()
        assert ModelCapability.CHAT in caps
        assert ModelCapability.FUNCTION_CALLING in caps


# ---------------------------------------------------------------------------
# Constraints — ceilings, rejected params, param_remaps
# ---------------------------------------------------------------------------


class TestConstraints:
    """Request-shape constraints resolve off the profile (were absent on HEAD)."""

    def test_gpt5_has_output_ceiling(self) -> None:
        """FAILS on HEAD: OpenAI had no ``_detect_constraints`` → ceiling None."""
        c = _provider("gpt-5").get_constraints()
        assert c.max_tokens_ceiling == 128000

    def test_gpt5_has_input_window(self) -> None:
        c = _provider("gpt-5").get_constraints()
        assert c.max_input_tokens == 400000

    def test_o1_rejects_temperature(self) -> None:
        c = _provider("o1").get_constraints()
        assert "temperature" in c.rejected_params

    def test_o1_param_remap_present(self) -> None:
        """The reasoning family remaps ``max_tokens`` → ``max_completion_tokens``.

        FAILS on HEAD: ``ModelConstraints`` had no ``param_remaps`` field.
        """
        c = _provider("o1").get_constraints()
        assert c.param_remaps.get("max_tokens") == "max_completion_tokens"

    def test_non_reasoning_model_has_no_remap(self) -> None:
        c = _provider("gpt-4o").get_constraints()
        assert not c.param_remaps

    def test_config_override_wins_on_ceiling(self) -> None:
        c = _provider(
            "gpt-4o",
            model_profile_overrides={"max_output_tokens": 999},
        ).get_constraints()
        assert c.max_tokens_ceiling == 999


# ---------------------------------------------------------------------------
# Request shaping — the shared clamp/drop/remap choke point, now live for OpenAI
# ---------------------------------------------------------------------------


class TestRequestShaping:
    """``complete`` routes through the shared request-shaping choke point."""

    async def test_max_tokens_clamped_to_ceiling(self) -> None:
        """FAILS on HEAD: OpenAI never clamped ``max_tokens`` (no ceiling)."""
        provider = _provider("gpt-4o", max_tokens=50000)
        await provider.complete("hi")
        assert provider._client.captured_kwargs["max_tokens"] == 16384

    async def test_reasoning_temperature_dropped(self) -> None:
        """o1 rejects ``temperature`` → it must not reach the API."""
        provider = _provider("o1", temperature=0.5, max_tokens=1000)
        await provider.complete("hi")
        assert "temperature" not in provider._client.captured_kwargs

    async def test_reasoning_max_tokens_remapped(self) -> None:
        """o1: ``max_tokens`` must be renamed to ``max_completion_tokens``.

        FAILS on HEAD: no remap mechanism → ``max_tokens`` sent (a 400).
        """
        provider = _provider("o1", max_tokens=1000)
        await provider.complete("hi")
        kwargs = provider._client.captured_kwargs
        assert kwargs.get("max_completion_tokens") == 1000
        assert "max_tokens" not in kwargs

    async def test_unknown_model_left_permissive(self) -> None:
        """Regression guard: an unknown model is shaped exactly as before.

        An unknown model resolves an all-None profile → no ceiling, no rejected
        params, no remap → ``max_tokens`` passes through untouched (the historical
        OpenAI behavior for every model).
        """
        provider = _provider("mystery-model-9", max_tokens=99999)
        await provider.complete("hi")
        assert provider._client.captured_kwargs["max_tokens"] == 99999

    async def test_reasoning_max_tokens_kwarg_does_not_collide(self) -> None:
        """A per-call ``max_tokens`` kwarg must fold into the single renamed key.

        Reproduces the double-key 400: the pre-fix ``params.update(kwargs)`` at
        each call site re-added ``max_tokens`` AFTER ``_build_api_kwargs`` had
        already produced ``max_completion_tokens``, so the request carried BOTH
        keys (an OpenAI 400) and the kwarg silently lost to the config value.
        A kwarg naming a config field now routes through the full drop/clamp/
        remap shaping (like ``config_overrides``), so it wins as the single
        ``max_completion_tokens``.
        """
        provider = _provider("o1", max_tokens=1000)
        await provider.complete("hi", max_tokens=500)
        kwargs = provider._client.captured_kwargs
        assert kwargs.get("max_completion_tokens") == 500
        assert "max_tokens" not in kwargs

    async def test_reasoning_temperature_kwarg_dropped(self) -> None:
        """A per-call ``temperature`` kwarg for a reasoning model must be dropped.

        Reproduces the drop-bypass: the pre-fix ``params.update(kwargs)`` re-added
        a caller ``temperature`` AFTER the rejected-param drop, so it reached the
        API (a 400 for the o-series). A kwarg naming a config field now flows
        through the rejected-param drop.
        """
        provider = _provider("o1")
        await provider.complete("hi", temperature=0.9)
        assert "temperature" not in provider._client.captured_kwargs

    async def test_max_tokens_kwarg_clamped_to_ceiling(self) -> None:
        """A per-call ``max_tokens`` kwarg is clamped like a config value.

        Guards that folding a config-field kwarg into the config subjects it to
        the ceiling clamp, not just the remap — a kwarg is no longer an escape
        hatch around the shaping the PR exists to apply.
        """
        provider = _provider("gpt-4o")
        await provider.complete("hi", max_tokens=50000)
        assert provider._client.captured_kwargs["max_tokens"] == 16384

    async def test_non_field_kwarg_passes_through(self) -> None:
        """A genuine wire-only kwarg (not a config field) is still passed through.

        Regression guard: the fix must only route *config-field* kwargs through
        shaping — a provider-specific wire param the caller supplies (here a
        ``user`` tracking id) must reach the API untouched.
        """
        provider = _provider("gpt-4o")
        await provider.complete("hi", user="acct-123")
        assert provider._client.captured_kwargs["user"] == "acct-123"

    async def test_response_format_dict_kwarg_preserved(self) -> None:
        """A ``response_format`` dict kwarg (the real OpenAI wire form) survives.

        Regression guard: ``response_format`` is an ``LLMConfig`` field whose
        canonical value is a narrow string (``'json'``/``'text'``), but the
        OpenAI wire accepts the richer dict form (``{"type": "json_object"}`` /
        ``{"type": "json_schema", ...}``). A caller passing that dict as a
        per-call kwarg (as ``dataknobs_bots.review.executor`` does for JSON-mode
        enforcement) must reach the API untouched. Routing *every* field-named
        kwarg through ``config.clone`` would fold the dict into the config, where
        ``adapt_config``'s strict ``response_format == 'json'`` check drops it —
        silently disabling JSON mode. Only *shaped* fields (dropped/clamped/
        remapped sampling params) belong in that fold; ``response_format`` is a
        wire-only passthrough.
        """
        provider = _provider("gpt-4o")
        await provider.complete("hi", response_format={"type": "json_object"})
        assert provider._client.captured_kwargs["response_format"] == {
            "type": "json_object"
        }


# ---------------------------------------------------------------------------
# Pricing — get_pricing (facts) + estimate_cost (convenience) + CostCalculator
# ---------------------------------------------------------------------------


class TestPricing:
    """Profile-sourced pricing reachable via the provider + CostCalculator math."""

    def test_get_pricing_reads_profile(self) -> None:
        """FAILS on HEAD: ``LLMProvider.get_pricing`` did not exist."""
        pricing = _provider("gpt-4o").get_pricing()
        assert pricing is not None
        assert pricing.output_per_mtok == 10.0

    def test_get_pricing_for_other_model(self) -> None:
        pricing = _provider("gpt-4o").get_pricing("gpt-4o-mini")
        assert pricing is not None
        assert pricing.input_per_mtok == 0.15

    def test_get_pricing_unknown_model_none(self) -> None:
        assert _provider("mystery-model-9").get_pricing() is None

    def test_estimate_cost_through_profile(self) -> None:
        """FAILS on HEAD: ``LLMProvider.estimate_cost`` did not exist."""
        provider = _provider("gpt-4o")
        response = LLMResponse(
            content="hi",
            model="gpt-4o",
            usage={"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000},
        )
        cost = provider.estimate_cost(response)
        # 1M input @ $2.50/Mtok + 1M output @ $10.00/Mtok = $12.50
        assert cost is not None
        assert abs(cost - 12.5) < 1e-6

    def test_estimate_cost_unknown_model_none(self) -> None:
        provider = _provider("mystery-model-9")
        response = LLMResponse(
            content="hi",
            model="mystery-model-9",
            usage={"prompt_tokens": 100, "completion_tokens": 50},
        )
        assert provider.estimate_cost(response) is None


# ---------------------------------------------------------------------------
# CostCalculator rewire — explicit ModelPricing + migrated fallback
# ---------------------------------------------------------------------------


class TestCostCalculatorRewire:
    """CostCalculator computes from an explicit ``ModelPricing``; fallback preserved."""

    def test_calculate_cost_with_explicit_pricing(self) -> None:
        response = LLMResponse(
            content="hi",
            model="whatever",
            usage={"prompt_tokens": 1_000_000, "completion_tokens": 0},
        )
        pricing = ModelPricing(input_per_mtok=3.0, output_per_mtok=15.0)
        cost = CostCalculator.calculate_cost(response, pricing=pricing)
        assert cost is not None
        assert abs(cost - 3.0) < 1e-6

    def test_legacy_fallback_still_works(self) -> None:
        """The migrated per-Mtok fallback reproduces the historical gpt-4 cost."""
        response = LLMResponse(
            content="hi",
            model="gpt-4",
            usage={"prompt_tokens": 100, "completion_tokens": 50},
        )
        cost = CostCalculator.calculate_cost(response)
        assert cost is not None
        assert abs(cost - 0.006) < 1e-4


# ---------------------------------------------------------------------------
# Resource-load unit
# ---------------------------------------------------------------------------


class TestResource:
    """The bundled openai_models.yaml loads and resolves through the substrate."""

    def test_resource_loads_and_resolves(self) -> None:
        source = BundledResourceSource.from_resource(
            "dataknobs_llm.llm.providers", "data/openai_models.yaml"
        )
        profile = source.resolve("gpt-4o")
        assert profile.max_output_tokens == 16384
        assert profile.context_window == 128000
        assert profile.pricing is not None
        assert ModelCapability.VISION in (profile.capabilities or frozenset())

    def test_resource_dated_alias_resolves(self) -> None:
        """A dated snapshot id resolves against the bare family key."""
        source = BundledResourceSource.from_resource(
            "dataknobs_llm.llm.providers", "data/openai_models.yaml"
        )
        assert source.resolve("gpt-4o-2024-11-20").max_output_tokens == 16384
