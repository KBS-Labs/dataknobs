"""Contract tests for the two provider-identity accessors.

``LLMProvider`` exposes two deliberately distinct names, and conflating them
is the defect this module exists to prevent:

* ``provider_name`` — the canonical **family** key (closed set, lower-cased,
  matches what the provider registry resolved on). Key lookup tables on this.
* ``impl_name`` — the concrete **class** serving the call (open set, includes
  wrappers). Diagnostics only.

The canonicalization test is the regression guard for a fix that, written
without ``.lower()``, would have re-created the original bug for any config
author who capitalized the provider name.
"""

from __future__ import annotations

import pytest

from dataknobs_llm import EchoProvider, LLMConfig, LLMProviderFactory
from dataknobs_llm.llm.base import AsyncLLMProvider
from dataknobs_llm.llm.providers import _provider_registry
from dataknobs_llm.llm.providers.caching import (
    CachingEmbedProvider,
    MemoryEmbeddingCache,
)
from dataknobs_llm.testing import CapturingProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _registered_families() -> list[str]:
    """Every family the provider registry knows.

    Driven off the registry rather than a hand-written list so a seventh
    provider cannot land without a ``provider_name``.
    """
    return sorted(_provider_registry.list_keys())


def _echo(provider: str = "echo") -> EchoProvider:
    return EchoProvider(LLMConfig(provider=provider, model="test-model"))


# ---------------------------------------------------------------------------
# provider_name — the family axis
# ---------------------------------------------------------------------------


#: What each built-in provider *class* must report, independent of any config.
#: The registry-driven test below feeds the key in and asserts it comes back,
#: which ``config.provider.lower()`` satisfies by construction — it proves the
#: accessor exists on every registered class but cannot catch one that
#: overrides it wrongly. Pinning the built-ins against a written-down mapping
#: is what closes that: a provider that starts reporting someone else's family
#: fails here, and the registry-driven test keeps a *new* provider from landing
#: without an accessor at all. Neither test subsumes the other.
BUILTIN_FAMILIES = {
    "OpenAIProvider": "openai",
    "AnthropicProvider": "anthropic",
    "BedrockProvider": "bedrock",
    "OllamaProvider": "ollama",
    "HuggingFaceProvider": "huggingface",
    "EchoProvider": "echo",
}


class TestProviderName:
    """``provider_name`` reports the canonical family key."""

    @pytest.mark.parametrize("family", _registered_families())
    def test_every_registered_provider_reports_its_family(self, family: str) -> None:
        """Built through the factory, each provider reports the key it resolved on."""
        factory = LLMProviderFactory(is_async=True)
        provider = factory.create({"provider": family, "model": "test-model"})

        assert provider.provider_name == family

    def test_every_builtin_class_maps_to_its_expected_family(self) -> None:
        """The written-down mapping, checked class-by-class.

        Also guards the mapping itself against going stale: it must name every
        registered built-in, so adding a provider without a row here fails.
        """
        factory = LLMProviderFactory(is_async=True)
        actual = {}
        for family in _registered_families():
            provider = factory.create({"provider": family, "model": "test-model"})
            actual[type(provider).__name__] = provider.provider_name

        assert actual == BUILTIN_FAMILIES

    def test_canonicalizes_a_capitalized_config(self) -> None:
        """``provider: OpenAI`` and ``provider: openai`` report the same family.

        Regression guard: a ``provider_name`` returning ``config.provider``
        verbatim passes every lowercase test and leaves the capitalized-config
        author with the original exact-match miss.
        """
        assert _echo("Echo").provider_name == "echo"
        assert _echo("ECHO").provider_name == "echo"
        assert _echo("echo").provider_name == "echo"

    def test_canonicalizes_when_built_through_the_factory(self) -> None:
        """The factory resolves the class case-insensitively but stores verbatim.

        ``PluginRegistry(canonicalize_keys=True)`` lower-cases only the
        *lookup*; the config keeps whatever the author typed. So the property
        — not the factory — is what makes the family key canonical.
        """
        factory = LLMProviderFactory(is_async=True)
        provider = factory.create({"provider": "Echo", "model": "test-model"})

        assert provider.config.provider == "Echo"  # stored verbatim
        assert provider.provider_name == "echo"  # reported canonically

    def test_verbatim_configured_string_stays_reachable(self) -> None:
        """A consumer wanting what the author typed still has ``config.provider``."""
        assert _echo("OpenAI").config.provider == "OpenAI"


# ---------------------------------------------------------------------------
# impl_name — the implementation axis
# ---------------------------------------------------------------------------


class TestImplName:
    """``impl_name`` reports the concrete class."""

    @pytest.mark.parametrize("family", _registered_families())
    def test_every_registered_provider_reports_its_class(self, family: str) -> None:
        factory = LLMProviderFactory(is_async=True)
        provider = factory.create({"provider": family, "model": "test-model"})

        assert provider.impl_name == type(provider).__name__

    def test_is_unaffected_by_config_spelling(self) -> None:
        """The class name is a property of the object, not of the config."""
        assert _echo("ECHO").impl_name == "EchoProvider"


# ---------------------------------------------------------------------------
# The two axes are distinct — this is why there are two
# ---------------------------------------------------------------------------


class TestTheTwoAxesAreDistinct:
    """Wrappers are where the family and the implementation diverge.

    For a bare provider the two accessors differ only in spelling, so a
    single-accessor implementation would pass any test written against one.
    A wrapper is the case that forces them apart: it serves the call itself
    while being billed as the family it wraps.
    """

    def test_caching_wrapper_reports_wrapped_family_and_own_class(self) -> None:
        inner = _echo("Echo")
        wrapper = CachingEmbedProvider(inner, MemoryEmbeddingCache())

        assert wrapper.provider_name == "echo"
        assert wrapper.impl_name == "CachingEmbedProvider"
        assert wrapper.provider_name != wrapper.impl_name

    def test_capturing_wrapper_reports_wrapped_family_and_own_class(self) -> None:
        inner = _echo("Echo")
        wrapper = CapturingProvider(inner, role="main")

        assert wrapper.provider_name == "echo"
        assert wrapper.impl_name == "CapturingProvider"
        assert wrapper.provider_name != wrapper.impl_name

    def test_wrapper_family_matches_the_provider_it_wraps(self) -> None:
        """The wrapper is billed as its inner — that is the whole point."""
        inner = _echo("Echo")
        wrapper = CachingEmbedProvider(inner, MemoryEmbeddingCache())

        assert wrapper.provider_name == inner.provider_name
        assert wrapper.impl_name != inner.impl_name


# ---------------------------------------------------------------------------
# Neither accessor is DK-class-specific
# ---------------------------------------------------------------------------


class TestConsumerRegisteredProvider:
    """A provider DK has never heard of gets both accessors for free."""

    def test_reports_its_registered_family_and_own_class(self) -> None:
        class AcmeProvider(EchoProvider):
            """A consumer's provider, registered under its own family key."""

        factory = LLMProviderFactory(is_async=True)
        factory.register_provider("acme", AcmeProvider)
        try:
            provider = factory.create({"provider": "acme", "model": "test-model"})

            assert provider.provider_name == "acme"
            assert provider.impl_name == "AcmeProvider"
        finally:
            _provider_registry.unregister("acme")

    def test_family_is_canonical_for_a_consumer_provider_too(self) -> None:
        class AcmeProvider(EchoProvider):
            pass

        provider = AcmeProvider(LLMConfig(provider="ACME", model="test-model"))

        assert provider.provider_name == "acme"


# ---------------------------------------------------------------------------
# First in-package consumer: SchemaExtractor
# ---------------------------------------------------------------------------


class TestSchemaExtractorUsesTheContract:
    """Extraction records are attributed to the family, not to a class name.

    ``SchemaExtractor`` predates the contract and had reinvented it by hand:
    it reached for a private ``_provider_name`` that no provider sets, then
    fell back to munging ``type(p).__name__``. That is right for the six
    concrete providers only by naming coincidence, and wrong for any wrapper.
    """

    @pytest.fixture
    def schema(self) -> dict:
        return {"type": "object", "properties": {"name": {"type": "string"}}}

    async def _extract_and_get_recorded_provider(
        self, provider: object, schema: dict
    ) -> str | None:
        from dataknobs_llm.extraction.observability import ExtractionTracker
        from dataknobs_llm.extraction.schema_extractor import SchemaExtractor

        tracker = ExtractionTracker()
        extractor = SchemaExtractor(provider=provider)
        await extractor.extract("some text", schema, tracker=tracker)

        return tracker.query()[0].provider

    @pytest.mark.asyncio
    async def test_bare_provider_is_attributed_to_its_family(self, schema: dict) -> None:
        inner = EchoProvider({"provider": "echo", "model": "test", "options": {"echo_prefix": ""}})

        recorded = await self._extract_and_get_recorded_provider(inner, schema)

        assert recorded == "echo"

    @pytest.mark.asyncio
    async def test_wrapped_provider_is_attributed_to_the_wrapped_family(self, schema: dict) -> None:
        """The regression guard for the hand-rolled munging.

        Fails against the old implementation, which recorded
        ``'cachingembed'`` — a string that is not a family, matches no rate
        table, and names a wrapper rather than the thing being billed.
        """
        inner = EchoProvider({"provider": "echo", "model": "test", "options": {"echo_prefix": ""}})
        wrapper = CachingEmbedProvider(inner, MemoryEmbeddingCache())

        recorded = await self._extract_and_get_recorded_provider(wrapper, schema)

        assert recorded == "echo"

    @pytest.mark.asyncio
    async def test_non_provider_double_still_records_something_usable(self, schema: dict) -> None:
        """``SchemaExtractor`` accepts any object with ``complete()``.

        The ``getattr`` fallback is deliberate — a test double that is not an
        ``LLMProvider`` has no ``provider_name``, and must not crash the
        extraction it is standing in for.
        """
        from dataknobs_llm import LLMResponse

        class AcmeDouble:
            async def complete(self, *args: object, **kwargs: object) -> LLMResponse:
                return LLMResponse(content="{}", model="test-model")

        recorded = await self._extract_and_get_recorded_provider(AcmeDouble(), schema)

        assert recorded == "AcmeDouble"

    @pytest.mark.asyncio
    async def test_class_name_fallback_routes_through_impl_name(self, schema: dict) -> None:
        """Even the fallback asks the object rather than re-deriving its class.

        Narrow by construction — every ``LLMProvider`` answers
        ``provider_name``, so this branch is reached only by a duck-typed
        object. It is still the right shape: re-deriving ``type(p).__name__``
        inline is the pattern this contract exists to end, and a double that
        declares ``impl_name`` has stated what it wants to be called.
        """
        from dataknobs_llm import LLMResponse

        class AcmeDouble:
            impl_name = "AcmeGateway"

            async def complete(self, *args: object, **kwargs: object) -> LLMResponse:
                return LLMResponse(content="{}", model="test-model")

        recorded = await self._extract_and_get_recorded_provider(AcmeDouble(), schema)

        assert recorded == "AcmeGateway"

    @pytest.mark.asyncio
    async def test_records_the_model_the_provider_is_configured_with(self, schema: dict) -> None:
        """``model_used`` must come from the config, not a private attribute.

        The original read ``provider._model``, which **no provider sets** — so
        every extraction that did not pass ``model=`` explicitly recorded
        ``None``. That is the same defect as the class-name munging one line
        below it: an observability field silently degraded by reaching for a
        private attribute instead of the public contract.
        """
        from dataknobs_llm.extraction.observability import ExtractionTracker
        from dataknobs_llm.extraction.schema_extractor import SchemaExtractor

        provider = EchoProvider(
            {
                "provider": "echo",
                "model": "llama3.2:3b",
                "options": {"echo_prefix": ""},
            }
        )
        tracker = ExtractionTracker()
        await SchemaExtractor(provider=provider).extract("some text", schema, tracker=tracker)

        assert tracker.query()[0].model_used == "llama3.2:3b"


# ---------------------------------------------------------------------------
# The sync half of the contract
# ---------------------------------------------------------------------------


class TestSyncProviderAdapter:
    """The only sync provider object DK actually ships must honor the contract.

    There are no ``SyncLLMProvider`` subclasses in-tree, so
    ``create_llm_provider(..., is_async=False)`` returns a
    ``SyncProviderAdapter`` wrapping an async provider. Declaring the accessors
    on ``LLMProvider`` therefore does *not* reach the sync path on its own,
    despite that being exactly what the docs promise — and an adapter with no
    ``provider_name`` degrades to the class name at every consumer, which is
    the original defect surviving on the sync half.
    """

    def _sync_provider(self, spelling: str = "echo") -> object:
        from dataknobs_llm.llm.providers import create_llm_provider

        return create_llm_provider({"provider": spelling, "model": "test-model"}, is_async=False)

    def test_reports_the_wrapped_family(self) -> None:
        assert self._sync_provider().provider_name == "echo"

    def test_canonicalizes_a_capitalized_config(self) -> None:
        assert self._sync_provider("Echo").provider_name == "echo"

    def test_reports_its_own_class_as_the_implementation(self) -> None:
        """It *is* the object serving the call, so it names itself.

        Same split as every other wrapper: billed as the family it wraps,
        diagnosed as the class that actually ran.
        """
        provider = self._sync_provider()

        assert provider.impl_name == "SyncProviderAdapter"
        assert provider.provider_name != provider.impl_name


# ---------------------------------------------------------------------------
# The family key stays consumer-overridable
# ---------------------------------------------------------------------------


class TestProviderNameOverride:
    """A consumer provider can still declare its own family key.

    ``TurnState`` has read ``getattr(provider, "provider_name", None)`` since
    before the accessor existed — an open invitation for a consumer's provider
    to set ``self.provider_name`` in ``__init__`` and get correct attribution.
    Turning that name into a read-only property revokes the invitation with an
    ``AttributeError`` at construction, so the setter is what keeps the
    pre-existing extension point working.
    """

    def test_a_provider_can_assign_its_family_key(self) -> None:
        class AcmeProvider(EchoProvider):
            def __init__(self, config: object) -> None:
                super().__init__(config)
                self.provider_name = "acme"

        provider = AcmeProvider(LLMConfig(provider="openai-compatible", model="m"))

        assert provider.provider_name == "acme"

    def test_an_override_is_canonicalized_like_a_configured_one(self) -> None:
        """The override is a family key, so it obeys the same closed-set rule."""
        provider = _echo("echo")
        provider.provider_name = "ACME"

        assert provider.provider_name == "acme"

    def test_clearing_the_override_restores_the_configured_family(self) -> None:
        provider = _echo("openai")
        provider.provider_name = "acme"
        provider.provider_name = None

        assert provider.provider_name == "openai"

    def test_an_override_does_not_disturb_the_implementation_axis(self) -> None:
        provider = _echo("echo")
        provider.provider_name = "acme"

        assert provider.impl_name == "EchoProvider"


# ---------------------------------------------------------------------------
# The contract lives on the shared base, not on the async half
# ---------------------------------------------------------------------------


def test_contract_is_declared_on_the_common_base() -> None:
    """Both accessors must reach ``SyncLLMProvider`` as well.

    Declaring them on ``AsyncLLMProvider`` would silently miss the sync half.
    """
    from dataknobs_llm.llm.base import LLMProvider

    assert isinstance(LLMProvider.__dict__.get("provider_name"), property)
    assert isinstance(LLMProvider.__dict__.get("impl_name"), property)
    assert "provider_name" not in AsyncLLMProvider.__dict__
    assert "impl_name" not in AsyncLLMProvider.__dict__
