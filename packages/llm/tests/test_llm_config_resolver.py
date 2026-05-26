"""Tests for the ``llm`` config resolver.

``dataknobs-llm`` registers a resolver into the shared ``config_registries``
so ``StructuredConfig.validate`` can check a raw ``llm`` section (e.g. a bot
config's provider section) without constructing a provider. ``LLMConfig`` is a
single config class keyed by ``provider`` — there are no per-provider config
subclasses — so the resolver returns ``LLMConfig`` for any *known* provider
and ``None`` for an unknown/missing one. These pin:

- The resolver is registered eagerly on import.
- Every provider the construction factory knows resolves to ``LLMConfig``
  (the no-drift guarantee — both consult the same provider registry).
- An unknown / missing provider resolves to ``None`` (so ``validate`` raises).
- Provider keys are canonicalized (the registry uses ``canonicalize_keys``).

These construct config internals directly (not provider flows), so no
provider instance is created.
"""

from __future__ import annotations

# Required side-effect import: importing the package registers the "llm"
# resolver in config_registries. Do NOT remove as "unused".
import dataknobs_llm  # noqa: F401
from dataknobs_llm.llm.base import LLMConfig
from dataknobs_llm.llm.providers import _provider_registry

from dataknobs_common.structured_config import config_registries


def _resolver():
    return config_registries.get("llm")


def test_resolver_registered_on_import() -> None:
    assert config_registries.has("llm")


def test_resolver_returns_llm_config_for_every_known_provider() -> None:
    # Drift guard: the resolver delegates to the same provider registry the
    # factory uses, so every constructible provider validates against
    # LLMConfig — validation and construction can never disagree.
    resolver = _resolver()
    keys = _provider_registry.list_keys()
    assert keys, "expected the built-in providers to be registered"
    for key in keys:
        assert resolver({"provider": key}) is LLMConfig, f"drift for {key!r}"


def test_resolver_canonicalizes_provider_key() -> None:
    # The provider registry canonicalizes keys, so a differently-cased
    # provider still resolves.
    assert _resolver()({"provider": "OpenAI"}) is LLMConfig


def test_resolver_returns_none_for_unknown_provider() -> None:
    assert _resolver()({"provider": "definitely-not-a-provider"}) is None


def test_resolver_returns_none_for_missing_provider() -> None:
    assert _resolver()({}) is None
    assert _resolver()({"provider": ""}) is None
    assert _resolver()({"provider": None}) is None
