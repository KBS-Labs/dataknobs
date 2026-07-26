"""Structural recurrence guard for the shared profile-detection mixin.

The model-metadata ``_detect_*`` trio (``_detect_capabilities`` /
``_detect_constraints`` / ``_detect_pricing``) was reproduced once per
substrate-bound provider. Extracting it to :class:`ProfileDetectionMixin` made
each bound provider *inherit* the trio; these tests pin that inheritance by MRO
identity so a future provider (or a maintainer) that hand-copies a ``_detect_*``
instead of inheriting fails here — the duplication cannot silently return.

The behavioral equivalence of the extraction (byte-identical capability /
constraint / pricing outputs per model) is covered by the per-provider binding
suites and the golden-master suites, which run unchanged. This module adds only
the *structural* checks the binding suites cannot express, plus the one
byte-identical-risk pin (Anthropic pricing stays ``None`` — it gained the mixin
``_detect_pricing`` where before it inherited the base ``None`` default).
"""

from __future__ import annotations

import pytest

from dataknobs_llm.llm.base import LLMConfig
from dataknobs_llm.llm.profile_detection import ProfileDetectionMixin
from dataknobs_llm.llm.providers.anthropic import AnthropicProvider
from dataknobs_llm.llm.providers.bedrock import BedrockProvider
from dataknobs_llm.llm.providers.ollama import OllamaProvider
from dataknobs_llm.llm.providers.openai import OpenAIProvider

# Providers that inherit the ENTIRE trio unchanged and keep the default
# lookup key (their resolve key is ``config.model``).
FULLY_INHERITING = [OpenAIProvider, OllamaProvider]


class TestTrioInheritedNotCopied:
    """Each bound provider inherits the trio from the mixin, not a local copy."""

    @pytest.mark.parametrize("provider_cls", FULLY_INHERITING)
    def test_openai_ollama_inherit_all_three(self, provider_cls: type) -> None:
        assert (
            provider_cls._detect_capabilities
            is ProfileDetectionMixin._detect_capabilities
        )
        assert (
            provider_cls._detect_constraints
            is ProfileDetectionMixin._detect_constraints
        )
        assert provider_cls._detect_pricing is ProfileDetectionMixin._detect_pricing

    @pytest.mark.parametrize("provider_cls", FULLY_INHERITING)
    def test_openai_ollama_keep_default_lookup_key(self, provider_cls: type) -> None:
        assert (
            provider_cls._profile_lookup_key
            is ProfileDetectionMixin._profile_lookup_key
        )

    def test_bedrock_inherits_trio_overrides_lookup_key(self) -> None:
        # Bedrock's only variance is the resolve key, so it inherits all three
        # detection methods and overrides the key seam instead of re-copying.
        assert (
            BedrockProvider._detect_capabilities
            is ProfileDetectionMixin._detect_capabilities
        )
        assert (
            BedrockProvider._detect_constraints
            is ProfileDetectionMixin._detect_constraints
        )
        assert BedrockProvider._detect_pricing is ProfileDetectionMixin._detect_pricing
        assert (
            BedrockProvider._profile_lookup_key
            is not ProfileDetectionMixin._profile_lookup_key
        )

    def test_anthropic_inherits_caps_and_pricing_overrides_only_constraints(
        self,
    ) -> None:
        # Anthropic adds two constraint rules (no inline system, discovered
        # rejected-param union), so it overrides ONLY _detect_constraints and
        # inherits capabilities + pricing + the default lookup key.
        assert (
            AnthropicProvider._detect_capabilities
            is ProfileDetectionMixin._detect_capabilities
        )
        assert AnthropicProvider._detect_pricing is ProfileDetectionMixin._detect_pricing
        assert (
            AnthropicProvider._detect_constraints
            is not ProfileDetectionMixin._detect_constraints
        )
        assert (
            AnthropicProvider._profile_lookup_key
            is ProfileDetectionMixin._profile_lookup_key
        )


class TestBedrockLookupKeyCanonicalizes:
    """The Bedrock lookup-key override strips the cross-region prefix."""

    def test_region_prefixed_id_canonicalizes_to_base_family(self) -> None:
        provider = BedrockProvider(
            LLMConfig(
                provider="bedrock",
                model="us.anthropic.claude-opus-4-8-20260101-v1:0",
            )
        )
        # The mixin resolves under this key; it must drop the ``us.`` prefix so
        # the cross-region id resolves the same family as its base id.
        key = provider._profile_lookup_key(provider.config)
        assert key == "anthropic.claude-opus-4-8-20260101-v1:0"

    def test_plain_id_is_unchanged(self) -> None:
        provider = BedrockProvider(
            LLMConfig(provider="bedrock", model="amazon.nova-pro-v1:0")
        )
        assert provider._profile_lookup_key(provider.config) == "amazon.nova-pro-v1:0"


class TestAnthropicPricingStaysNone:
    """D-EXTRACT-PRICING pin: inheriting the mixin _detect_pricing is byte-identical.

    Anthropic had no _detect_pricing override (base default ``None``); it now
    inherits the mixin's profile read. Anthropic's sources set no pricing facet,
    so the read returns ``None`` for every model — proving the extraction
    changed nothing observable.
    """

    @pytest.mark.parametrize(
        "model",
        [
            "claude-opus-4-8",
            "claude-sonnet-5",
            "claude-haiku-4-5-20251001",
            "claude-3-5-sonnet-20241022",
        ],
    )
    def test_get_pricing_is_none_across_model_set(self, model: str) -> None:
        provider = AnthropicProvider(LLMConfig(provider="anthropic", model=model))
        assert provider.get_pricing(model) is None
        assert provider._detect_pricing(provider.config) is None
