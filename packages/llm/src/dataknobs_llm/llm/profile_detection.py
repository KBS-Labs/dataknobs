"""Shared model-metadata ``_detect_*`` trio for substrate-bound providers.

A provider bound to the model-metadata substrate
(:mod:`dataknobs_llm.llm.model_profile`) ends its binding with the same three
"read a facet off the resolved :class:`~.model_profile.ModelProfile`" methods:
:meth:`~.base.LLMProvider._detect_capabilities`,
:meth:`~.base.LLMProvider._detect_constraints`, and
:meth:`~.base.LLMProvider._detect_pricing`. :class:`ProfileDetectionMixin`
implements that trio once, in terms of two per-provider hooks — the existing
:meth:`_profile_resolver` and the new :meth:`_profile_lookup_key` — so a bound
provider inherits the trio instead of hand-copying it.

This lives in a dedicated module rather than in :mod:`~.base` deliberately:
:mod:`~.model_profile` imports from :mod:`~.base` (for ``ModelCapability`` /
``CAPABILITY_NAMES``), so :mod:`~.base` cannot runtime-import the resolver /
``CAPABILITY_ORDER`` symbols without a circular import (:mod:`~.base` keeps its
own ``ModelPricing`` reference under ``TYPE_CHECKING`` for the same reason). A
downstream module that imports from *both* :mod:`~.base` and
:mod:`~.model_profile` — and that neither imports back — is the cycle-free home.
"""

from __future__ import annotations

from abc import abstractmethod

from .base import LLMConfig, LLMProvider, ModelCapability, ModelConstraints
from .model_profile import (
    CAPABILITY_ORDER,
    LayeredModelProfileResolver,
    ModelPricing,
    ModelProfile,
)

__all__ = ["ProfileDetectionMixin"]


class ProfileDetectionMixin(LLMProvider):
    """Implements the model-metadata ``_detect_*`` trio for a bound provider.

    A provider adopting the substrate lists this mixin **first** among its bases
    (``class OpenAIProvider(ProfileDetectionMixin, AsyncLLMProvider)``) and
    supplies :meth:`_profile_resolver`; it then inherits
    :meth:`_detect_capabilities`, :meth:`_detect_constraints`, and
    :meth:`_detect_pricing` with no per-provider copy.

    Two extension points cover the only real per-provider variance:

    - Override :meth:`_profile_lookup_key` when the resolve key is not
      ``config.model`` (Bedrock normalizes a region-prefixed inference-profile id
      to its base id before the catalog lookup).
    - Override :meth:`_detect_constraints` — calling :meth:`_resolve_profile` so
      the lookup-key logic stays shared — when the family adds constraint rules
      beyond the four profile facets (Anthropic's inline-system ban + its
      process-local runtime-discovered rejected-param overlay).

    The mixin subclasses :class:`~.base.LLMProvider` so ``self.config`` and the
    template methods are typed; it stays abstract on :meth:`_profile_resolver`,
    so an *unbound* provider (``EchoProvider``, a not-yet-bound
    ``HuggingFaceProvider``, any consumer provider) that does not adopt the mixin
    still satisfies the base's own abstract ``_detect_capabilities`` itself — the
    abstract-method safety net is preserved.
    """

    @abstractmethod
    def _profile_resolver(self, config: LLMConfig) -> LayeredModelProfileResolver:
        """The provider's layered resolver (highest-precedence source first)."""

    def _profile_lookup_key(self, config: LLMConfig) -> str:
        """The model id passed to ``resolve()``. Default: the configured model.

        Overridden by a provider whose id space needs normalization before the
        catalog lookup (Bedrock strips a cross-region inference-profile prefix
        via ``_canonical_model_id`` so the profile resolves the same family as
        the base id).
        """
        return config.model

    def _resolve_profile(self, config: LLMConfig) -> ModelProfile:
        """Resolve *config*'s profile through the provider resolver + lookup key.

        The single point every ``_detect_*`` method (and an overriding
        :meth:`_detect_constraints`) routes through, so the resolver composition
        and the lookup-key normalization are defined exactly once per provider.
        """
        return self._profile_resolver(config).resolve(self._profile_lookup_key(config))

    def _detect_capabilities(self) -> list[ModelCapability]:
        """Read the ``capabilities`` facet off the resolved profile.

        Projected back to the historical ordered list via
        :data:`~.model_profile.CAPABILITY_ORDER` (an absent facet resolves to the
        empty set → no capabilities).
        """
        capabilities = self._resolve_profile(self.config).capabilities or frozenset()
        return [c for c in CAPABILITY_ORDER if c in capabilities]

    def _detect_constraints(self, config: LLMConfig) -> ModelConstraints:
        """Read the four request-shape facets off the resolved profile.

        ``max_tokens_ceiling`` from ``max_output_tokens`` (the base clamps an
        over-budget ``max_tokens`` down to it), ``max_input_tokens`` from
        ``context_window`` (informational), ``rejected_params`` (dropped before
        the call), and ``param_remaps`` (wire renames applied after
        ``adapt_config``). ``accepts_inline_system`` stays the permissive default;
        a family that forbids an inline system role (Anthropic) overrides this
        method. All overridable per request via ``LLMConfig.constraints``.
        """
        profile = self._resolve_profile(config)
        return ModelConstraints(
            rejected_params=frozenset(profile.rejected_params or ()),
            max_tokens_ceiling=profile.max_output_tokens,
            max_input_tokens=profile.context_window,
            param_remaps=dict(profile.param_remaps or {}),
        )

    def _detect_pricing(self, config: LLMConfig) -> ModelPricing | None:
        """Read the ``pricing`` facet off the resolved profile (else ``None``)."""
        return self._resolve_profile(config).pricing
