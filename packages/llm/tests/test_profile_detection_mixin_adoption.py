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
the *structural* checks the binding suites cannot express:

- The explicit per-provider adoption-shape assertions (each named provider
  inherits exactly the trio it should and overrides exactly the seams it should).
- A **maintenance-free** enumeration guard that discovers *every* shipped
  substrate-bound provider (any provider supplying a concrete ``_profile_resolver``)
  and asserts it inherits the trio rather than re-copying it — so a *future* fifth
  provider that hand-copies is caught without anyone remembering to extend a list.
- The Anthropic pricing pins. Anthropic had no ``_detect_pricing`` override
  (base default ``None``, which silently dropped a consumer pricing override); it
  now inherits the mixin read. Two pins: shipped/default pricing stays ``None``
  for every model (byte-identical to before), **and** a
  ``model_profile_overrides.pricing`` now flows through — the deliberate
  consumer-extensibility alignment that makes Anthropic honor pricing overrides
  like its three siblings (OpenAI / Bedrock / Ollama), which always did.
"""

from __future__ import annotations

import inspect

import pytest

import dataknobs_llm.llm.providers  # noqa: F401  (force every provider module to load)
from dataknobs_llm.llm.base import LLMConfig, LLMProvider
from dataknobs_llm.llm.profile_detection import ProfileDetectionMixin
from dataknobs_llm.llm.providers.anthropic import AnthropicProvider
from dataknobs_llm.llm.providers.bedrock import BedrockProvider
from dataknobs_llm.llm.providers.huggingface import HuggingFaceProvider
from dataknobs_llm.llm.providers.ollama import OllamaProvider
from dataknobs_llm.llm.providers.openai import OpenAIProvider

# Providers that inherit the ENTIRE trio unchanged and keep the default
# lookup key (their resolve key is ``config.model``). HuggingFace is the first
# *clean* adopter of the extraction — it supplies only ``_profile_resolver`` and
# inherits capabilities, constraints, pricing, AND the default lookup key.
FULLY_INHERITING = [OpenAIProvider, OllamaProvider, HuggingFaceProvider]


def _all_subclasses(root: type) -> set[type]:
    """Every transitive subclass of *root* currently loaded."""
    seen: set[type] = set()
    stack = list(root.__subclasses__())
    while stack:
        cls = stack.pop()
        if cls in seen:
            continue
        seen.add(cls)
        stack.extend(cls.__subclasses__())
    return seen


def _is_substrate_bound(cls: type) -> bool:
    """True when *cls* supplies a concrete ``_profile_resolver`` (binds the substrate).

    Resolved through the MRO, so a subclass that inherits a concrete resolver from
    a bound parent counts too; the mixin's own abstract hook does not.
    """
    resolver = getattr(cls, "_profile_resolver", None)
    return resolver is not None and not getattr(resolver, "__isabstractmethod__", False)


def _shipped_substrate_bound_providers() -> list[type]:
    """Discover every shipped, concrete, substrate-bound provider class.

    Scoped to the shipped ``providers`` package (excludes test doubles and
    consumer subclasses) — this is the recurrence surface the guard defends.
    """
    return sorted(
        (
            cls
            for cls in _all_subclasses(LLMProvider)
            if cls.__module__.startswith("dataknobs_llm.llm.providers")
            and not inspect.isabstract(cls)
            and _is_substrate_bound(cls)
        ),
        key=lambda c: c.__name__,
    )


# Discovered at import time (all provider modules are loaded above). A new bound
# provider added to the package is picked up here automatically.
_BOUND_PROVIDERS = _shipped_substrate_bound_providers()


class TestTrioInheritedNotCopied:
    """Each bound provider inherits the trio from the mixin, not a local copy."""

    @pytest.mark.parametrize("provider_cls", FULLY_INHERITING)
    def test_openai_ollama_inherit_all_three(self, provider_cls: type) -> None:
        assert provider_cls._detect_capabilities is ProfileDetectionMixin._detect_capabilities
        assert provider_cls._detect_constraints is ProfileDetectionMixin._detect_constraints
        assert provider_cls._detect_pricing is ProfileDetectionMixin._detect_pricing

    @pytest.mark.parametrize("provider_cls", FULLY_INHERITING)
    def test_openai_ollama_keep_default_lookup_key(self, provider_cls: type) -> None:
        assert provider_cls._profile_lookup_key is ProfileDetectionMixin._profile_lookup_key

    def test_bedrock_inherits_trio_overrides_lookup_key(self) -> None:
        # Bedrock's only variance is the resolve key, so it inherits all three
        # detection methods and overrides the key seam instead of re-copying.
        assert BedrockProvider._detect_capabilities is ProfileDetectionMixin._detect_capabilities
        assert BedrockProvider._detect_constraints is ProfileDetectionMixin._detect_constraints
        assert BedrockProvider._detect_pricing is ProfileDetectionMixin._detect_pricing
        assert BedrockProvider._profile_lookup_key is not ProfileDetectionMixin._profile_lookup_key

    def test_anthropic_inherits_caps_and_pricing_overrides_only_constraints(
        self,
    ) -> None:
        # Anthropic adds two constraint rules (no inline system, discovered
        # rejected-param union), so it overrides ONLY _detect_constraints and
        # inherits capabilities + pricing + the default lookup key.
        assert AnthropicProvider._detect_capabilities is ProfileDetectionMixin._detect_capabilities
        assert AnthropicProvider._detect_pricing is ProfileDetectionMixin._detect_pricing
        assert (
            AnthropicProvider._detect_constraints is not ProfileDetectionMixin._detect_constraints
        )
        assert AnthropicProvider._profile_lookup_key is ProfileDetectionMixin._profile_lookup_key


class TestNoSubstrateBoundProviderReCopiesTrio:
    """Maintenance-free recurrence guard over *every* shipped bound provider.

    Unlike :class:`TestTrioInheritedNotCopied` (a fixed named list), this walks
    the live subclass graph, so a *future* fifth substrate-bound provider is
    checked automatically. It fails only when such a provider re-copies caps or
    pricing instead of inheriting, or hand-inlines the resolver in an overriding
    ``_detect_constraints`` instead of routing through the shared
    ``_resolve_profile`` helper.
    """

    def test_enumeration_finds_the_known_adopters(self) -> None:
        # Guard against the guard silently degrading to an empty set (which would
        # make every parametrized check below vacuously pass). The discovered set
        # must at least contain the five current adopters; a new one may be added.
        assert set(_BOUND_PROVIDERS) >= {
            OpenAIProvider,
            AnthropicProvider,
            BedrockProvider,
            OllamaProvider,
            HuggingFaceProvider,
        }

    @pytest.mark.parametrize("provider_cls", _BOUND_PROVIDERS, ids=lambda c: c.__name__)
    def test_bound_provider_adopts_mixin(self, provider_cls: type) -> None:
        assert issubclass(provider_cls, ProfileDetectionMixin), (
            f"{provider_cls.__name__} supplies a concrete _profile_resolver but "
            "does not adopt ProfileDetectionMixin — it must inherit the trio, not "
            "re-copy it."
        )

    @pytest.mark.parametrize("provider_cls", _BOUND_PROVIDERS, ids=lambda c: c.__name__)
    def test_bound_provider_inherits_caps_and_pricing(self, provider_cls: type) -> None:
        # A re-copied method is a distinct function object, so identity fails.
        assert provider_cls._detect_capabilities is ProfileDetectionMixin._detect_capabilities, (
            f"{provider_cls.__name__} re-copies _detect_capabilities"
        )
        assert provider_cls._detect_pricing is ProfileDetectionMixin._detect_pricing, (
            f"{provider_cls.__name__} re-copies _detect_pricing"
        )

    @pytest.mark.parametrize("provider_cls", _BOUND_PROVIDERS, ids=lambda c: c.__name__)
    def test_overriding_constraints_routes_through_shared_helper(self, provider_cls: type) -> None:
        # A bound provider may override _detect_constraints (Anthropic does), but
        # only by reusing the shared _resolve_profile — never by re-inlining the
        # resolver composition + lookup key, which is the duplication we killed.
        if provider_cls._detect_constraints is ProfileDetectionMixin._detect_constraints:
            return  # inherited verbatim — nothing to re-copy
        source = inspect.getsource(provider_cls._detect_constraints)
        assert "_resolve_profile" in source, (
            f"{provider_cls.__name__} overrides _detect_constraints without "
            "routing through the shared _resolve_profile helper — it re-inlines "
            "the resolver composition the mixin exists to share."
        )


class TestValidateModelPinTemplate:
    """Every bound provider resolves an unpinned ``validate_model`` to a probe.

    The ``model_profile_overrides.available`` pin prologue lives once in
    :meth:`ProfileDetectionMixin.validate_model`, which honors the pin then calls
    :meth:`_probe_model_available`. Two adoption shapes are valid:

    - *Probe-style* (OpenAI, HuggingFace): inherit the mixin ``validate_model``
      and override ``_probe_model_available`` with the network probe.
    - *Facet-resolved* (Ollama, Bedrock, Anthropic): override ``validate_model``
      directly (availability is read from a resolved facet / offered-set), and
      never touch ``_probe_model_available``.

    The footgun the guard closes: a *future* probe-style adopter that inherits the
    mixin ``validate_model`` but forgets the probe override raises
    ``NotImplementedError`` only when ``validate_model`` is first called without a
    pin — a runtime failure, not an import/definition one. This pins the invariant
    structurally instead: a provider that inherits the mixin ``validate_model``
    MUST override ``_probe_model_available``.
    """

    def test_current_adopters_split_across_both_shapes(self) -> None:
        # Guard against the invariant becoming vacuous: pin that both shapes are
        # actually exercised by the shipped providers.
        inherits_validate = {
            c for c in _BOUND_PROVIDERS if c.validate_model is ProfileDetectionMixin.validate_model
        }
        overrides_validate = set(_BOUND_PROVIDERS) - inherits_validate
        assert inherits_validate >= {OpenAIProvider, HuggingFaceProvider}
        assert overrides_validate >= {
            OllamaProvider,
            BedrockProvider,
            AnthropicProvider,
        }

    @pytest.mark.parametrize("provider_cls", _BOUND_PROVIDERS, ids=lambda c: c.__name__)
    def test_inherited_validate_model_has_a_probe_override(self, provider_cls: type) -> None:
        # If the provider inherits the mixin's pin-honoring validate_model, it must
        # supply the probe the template calls — otherwise an unpinned validate_model
        # raises NotImplementedError at call time.
        if provider_cls.validate_model is not ProfileDetectionMixin.validate_model:
            return  # facet-resolved: overrides validate_model, no probe needed
        assert (
            provider_cls._probe_model_available is not ProfileDetectionMixin._probe_model_available
        ), (
            f"{provider_cls.__name__} inherits ProfileDetectionMixin.validate_model "
            "but does not override _probe_model_available — an unpinned "
            "validate_model() would raise NotImplementedError at call time."
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
        provider = BedrockProvider(LLMConfig(provider="bedrock", model="amazon.nova-pro-v1:0"))
        assert provider._profile_lookup_key(provider.config) == "amazon.nova-pro-v1:0"


class TestAnthropicPricing:
    """D-EXTRACT-PRICING pins for Anthropic's move from base-``None`` to the mixin.

    Anthropic had no ``_detect_pricing`` override, so it used the base default
    that hard-returned ``None`` **regardless of any consumer override**. It now
    inherits the mixin's profile read. Two contracts:

    - *Shipped/default is byte-identical.* Anthropic's own sources set no pricing
      facet, so with no override the read resolves ``None`` for every model.
    - *A consumer pricing override now flows.* Previously an Anthropic
      ``model_profile_overrides.pricing`` was silently dropped (base ``None``);
      it is now honored — the deliberate consumer-extensibility alignment that
      makes Anthropic behave like OpenAI / Bedrock / Ollama, all of which always
      read the profile (and therefore honored pricing overrides) before this PR.
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

    def test_pricing_override_now_flows(self) -> None:
        # The behavior this PR intentionally changed: before the extraction this
        # override reached base-``None`` and was dropped; now it flows through the
        # inherited mixin read, matching every sibling provider.
        provider = AnthropicProvider(
            LLMConfig(
                provider="anthropic",
                model="claude-opus-4-8",
                model_profile_overrides={
                    "pricing": {"input_per_mtok": 15.0, "output_per_mtok": 75.0}
                },
            )
        )
        pricing = provider.get_pricing()
        assert pricing is not None
        assert pricing.input_per_mtok == 15.0
        assert pricing.output_per_mtok == 75.0
