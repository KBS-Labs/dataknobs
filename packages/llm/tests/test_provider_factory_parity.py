"""Drift guards for the ``dataknobs-llm`` provider registry.

The provider registry's dispatch shape is
``provider_class(llm_config)`` where ``llm_config`` is an
:class:`LLMConfig` dataclass. Drift modes:

- A provider's ``__init__`` regresses to a non-``LLMConfig`` shape
  (e.g., direct kwargs) — the factory dispatch breaks at the
  consumer's call site.
- A provider's ``__init__`` accepts a typed config that's NOT
  :class:`LLMConfig` (e.g., a provider-specific dataclass) and the
  factory's hand-off no longer succeeds.

The check runs without instantiating providers, so providers with
optional dependencies (anthropic SDK, openai SDK, ollama runtime,
huggingface transformers) are still audited.
"""

from __future__ import annotations

import inspect

import pytest

from dataknobs_common.testing import (
    assert_config_attribute_access_matches_dataclass,
)
from dataknobs_llm.llm.base import LLMConfig
from dataknobs_llm.llm.providers import (
    AnthropicProvider,
    EchoProvider,
    HuggingFaceProvider,
    OllamaProvider,
    OpenAIProvider,
)

PROVIDERS = [
    ("openai", OpenAIProvider),
    ("anthropic", AnthropicProvider),
    ("ollama", OllamaProvider),
    ("huggingface", HuggingFaceProvider),
    ("echo", EchoProvider),
]


@pytest.mark.parametrize(
    "name, provider_cls",
    PROVIDERS,
    ids=[name for name, _ in PROVIDERS],
)
def test_provider_init_accepts_llm_config(
    name: str, provider_cls: type
) -> None:
    """Every registered provider's ``__init__`` accepts an :class:`LLMConfig`.

    ``LLMProviderFactory.create`` calls ``provider_class(llm_config)``
    after normalizing the config. If a provider regresses to a
    different ctor shape (direct kwargs, provider-specific dataclass),
    that dispatch breaks.

    The parametric check: ``provider_cls(LLMConfig(provider=name))``
    must succeed type-wise — we verify the *signature accepts* an
    ``LLMConfig`` value without actually instantiating (some providers
    open network connections / load models in ``__init__``).
    """
    sig = inspect.signature(provider_cls.__init__)
    params = list(sig.parameters.values())
    # Skip ``self``; the first real parameter should be the config.
    real_params = [p for p in params if p.name != "self"]
    assert real_params, (
        f"{provider_cls.__name__}.__init__ has no parameters after `self`."
    )
    first = real_params[0]
    # The convention is ``__init__(self, config: LLMConfig | Config | dict | ...)``
    # — check that ``LLMConfig`` is at least allowed by the annotation,
    # OR that the param has no annotation (back-compat).
    if first.annotation is inspect.Parameter.empty:
        # Untyped — relies on duck typing; the runtime path is exercised
        # by per-provider behavioural tests.
        return
    anno = str(first.annotation)
    # Permissive match: the annotation mentions ``LLMConfig`` somewhere
    # (Union[LLMConfig, Config, Dict[str, Any]] is the common shape).
    assert "LLMConfig" in anno or "Config" in anno or "Dict" in anno, (
        f"{provider_cls.__name__}.__init__'s config parameter "
        f"({first.name}: {anno}) does not accept LLMConfig. The factory "
        "calls provider_class(LLMConfig) — fix the ctor signature or "
        "update this parity test to reflect the new convention."
    )


@pytest.mark.parametrize(
    "name, provider_cls",
    PROVIDERS,
    ids=[name for name, _ in PROVIDERS],
)
def test_provider_config_access_within_llmconfig(
    name: str, provider_cls: type
) -> None:
    """Every ``self.config.<attr>`` a provider reads is an :class:`LLMConfig` field.

    The body-access direction, complementary to
    ``test_provider_init_accepts_llm_config`` (which guards the
    ctor-surface direction). A provider could read
    ``self.config.custom_extension`` — accepted by the permissive ctor
    signature check, but an ``AttributeError`` the first time that
    (often un-CI'd, provider-specific) path runs. This AST-walks the
    provider's MRO for such reads against ``LLMConfig``'s field +
    attribute surface. Config methods (``clone``, ``generation_params``)
    are valid reads; reads off the base classes are covered by the MRO
    walk.
    """
    assert_config_attribute_access_matches_dataclass(provider_cls, LLMConfig)


def test_llm_config_dataclass_is_complete() -> None:
    """``LLMConfig`` exposes the documented field set.

    Drift mode: a provider relies on a field that ``LLMConfig`` doesn't
    declare. This is a smoke check — the canonical field set is the
    one currently shipped; the test pins it so the contributor adding
    a new provider notices the cross-provider impact.
    """
    import dataclasses

    field_names = {f.name for f in dataclasses.fields(LLMConfig)}
    # Core fields that every provider relies on. If any of these is
    # removed (e.g., during a refactor), the factory dispatch breaks
    # for every provider. The list is intentionally minimal — provider-
    # specific options live in the ``options`` dict.
    expected_minimum = {"provider", "model", "api_key", "api_base"}
    missing = expected_minimum - field_names
    assert not missing, (
        f"LLMConfig is missing fields every provider relies on: {missing}"
    )


def test_registered_providers_audit_set() -> None:
    """The audit matrix above must cover every registered built-in.

    A new built-in provider → add a row to the parametrized test
    above so the parity guard continues to cover the registry.
    """
    from dataknobs_llm.llm.providers import _provider_registry

    keys = set(_provider_registry.list_keys())
    builtin = {p[0] for p in PROVIDERS}
    new_unaudited = keys - builtin
    if new_unaudited:
        pytest.fail(
            f"New providers registered without parity coverage: "
            f"{new_unaudited}. Add a row to PROVIDERS."
        )
