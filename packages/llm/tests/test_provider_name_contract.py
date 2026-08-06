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


class TestProviderName:
    """``provider_name`` reports the canonical family key."""

    @pytest.mark.parametrize("family", _registered_families())
    def test_every_registered_provider_reports_its_family(
        self, family: str
    ) -> None:
        """Built through the factory, each provider reports the key it resolved on."""
        factory = LLMProviderFactory(is_async=True)
        provider = factory.create({"provider": family, "model": "test-model"})

        assert provider.provider_name == family

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
    def test_every_registered_provider_reports_its_class(
        self, family: str
    ) -> None:
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
    async def test_bare_provider_is_attributed_to_its_family(
        self, schema: dict
    ) -> None:
        inner = EchoProvider(
            {"provider": "echo", "model": "test", "options": {"echo_prefix": ""}}
        )

        recorded = await self._extract_and_get_recorded_provider(inner, schema)

        assert recorded == "echo"

    @pytest.mark.asyncio
    async def test_wrapped_provider_is_attributed_to_the_wrapped_family(
        self, schema: dict
    ) -> None:
        """The regression guard for the hand-rolled munging.

        Fails against the old implementation, which recorded
        ``'cachingembed'`` — a string that is not a family, matches no rate
        table, and names a wrapper rather than the thing being billed.
        """
        inner = EchoProvider(
            {"provider": "echo", "model": "test", "options": {"echo_prefix": ""}}
        )
        wrapper = CachingEmbedProvider(inner, MemoryEmbeddingCache())

        recorded = await self._extract_and_get_recorded_provider(wrapper, schema)

        assert recorded == "echo"

    @pytest.mark.asyncio
    async def test_non_provider_double_still_records_something_usable(
        self, schema: dict
    ) -> None:
        """``SchemaExtractor`` accepts any object with ``complete()``.

        The ``getattr`` fallback is deliberate — a test double that is not an
        ``LLMProvider`` has no ``provider_name``, and must not crash the
        extraction it is standing in for.
        """
        from dataknobs_llm import LLMResponse

        class AcmeDouble:
            async def complete(self, *args: object, **kwargs: object) -> LLMResponse:
                return LLMResponse(content="{}", model="test-model")

        recorded = await self._extract_and_get_recorded_provider(
            AcmeDouble(), schema
        )

        assert recorded == "AcmeDouble"


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
