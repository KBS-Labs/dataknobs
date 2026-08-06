"""The config schema's provider list is the provider registry, not a copy of it.

``llm.provider`` was pinned to a hand-written enum of five names. A registry
that already exists, is authoritative, and is **consumer-extensible** had been
transcribed into a literal, and the copy immediately diverged from it:

* ``bedrock`` is a family DK ships and prices; the enum rejected it.
* ``provider: OpenAI`` resolves fine at runtime — the registry canonicalizes
  its lookups — and the enum rejected it as an invalid value.
* A consumer's own provider, registered through the documented
  ``register_provider`` extension point, could never appear in a literal
  written before it existed, so its config was rejected outright.

The third is the one a closed enum can never fix. A validator that rejects
what the runtime accepts is worse than no validator: it blocks a working
configuration and points at the wrong thing.
"""

from __future__ import annotations

import logging

import pytest

from dataknobs_bots.config.schema import DynaBotConfigSchema
from dataknobs_bots.config.validation import resolve_enum_options
from dataknobs_llm import EchoProvider, LLMProviderFactory
from dataknobs_llm.llm.providers import _provider_registry


def _errors(provider: str) -> list[str]:
    """Schema errors mentioning ``provider`` for a minimal llm section.

    Driven through ``DynaBotConfigSchema.validate`` rather than
    ``ConfigValidator.validate``: the latter runs schema checks only when a
    schema was injected, so a bare ``ConfigValidator()`` reports no schema
    errors at all and would make every assertion here vacuously true.
    """
    result = DynaBotConfigSchema().validate(
        {"llm": {"provider": provider, "model": "test-model"}}
    )
    return [e for e in result.errors if "provider" in e]


def test_the_harness_reaches_the_enum_check() -> None:
    """Non-vacuity guard for ``_errors``.

    Every acceptance assertion in this module is an *absence* of errors, which
    a harness that validates nothing satisfies perfectly. This pins that the
    path really does reject something.
    """
    assert _errors("definitely-not-a-provider")


class TestShippedFamiliesValidate:
    """Every family the registry knows must pass the validator."""

    @pytest.mark.parametrize(
        "family", sorted(_provider_registry.list_keys())
    )
    def test_registered_family_is_accepted(self, family: str) -> None:
        """Driven off the registry, so a seventh provider cannot drift again."""
        assert _errors(family) == []

    def test_bedrock_is_accepted(self) -> None:
        """Named explicitly: it is the family the copy had already lost."""
        assert _errors("bedrock") == []


class TestSpellingIsCanonicalized:
    """The runtime resolves case-insensitively; the validator must agree.

    ``PluginRegistry(canonicalize_keys=True)`` lower-cases the lookup, so
    ``provider: OpenAI`` builds an ``OpenAIProvider`` without complaint. A
    validator that rejects it contradicts the code it is validating.
    """

    @pytest.mark.parametrize("spelling", ["OpenAI", "OPENAI", "openai"])
    def test_any_spelling_of_a_known_family_is_accepted(
        self, spelling: str
    ) -> None:
        assert _errors(spelling) == []


class TestConsumerRegisteredProvider:
    """The extension point the closed enum silently disabled."""

    def test_a_consumer_provider_is_accepted_once_registered(self) -> None:
        """Registered through the documented public entry point.

        No literal written inside DK can contain this name, which is why the
        validator has to ask the registry rather than carry a list.
        """

        class AcmeProvider(EchoProvider):
            pass

        LLMProviderFactory.register_provider("acme", AcmeProvider)
        try:
            assert _errors("acme") == []
        finally:
            _provider_registry.unregister("acme")

    def test_an_unregistered_provider_is_still_rejected(self) -> None:
        """Opening the set to the registry must not open it to typos.

        ``openia`` is registered nowhere, so it fails here exactly as it would
        fail at construction — the point is that the two now agree.
        """
        errors = _errors("nonesuch-provider")

        assert errors
        assert "nonesuch-provider" in errors[0]

    def test_the_rejection_names_what_is_available(self) -> None:
        """A rejection has to be actionable, and the list is now live."""
        message = _errors("nonesuch-provider")[0]

        assert "bedrock" in message


class TestUnknownRegistryFailsOpenLoudly:
    """A registry name this build does not have leaves the field unchecked.

    That is deliberate: a consumer schema may name a registry supplied by a
    newer DK, and declining to constrain one field beats rejecting every value
    for it. But the outcome — a validator switching itself off — is
    indistinguishable from a field that passed, so it is pinned here rather
    than left to be rediscovered and "fixed" as a bug, and it has to be
    audible.
    """

    def test_an_unknown_registry_leaves_the_field_unconstrained(self) -> None:
        assert resolve_enum_options({"enum_registry": "no-such-registry"}) is None

    def test_a_field_naming_an_unknown_registry_accepts_anything(self) -> None:
        """Through the real validator, not just the resolver."""
        schema = DynaBotConfigSchema()
        schema.register_component(
            "acme_thing",
            {"properties": {"kind": {"enum_registry": "no-such-registry"}}},
        )

        result = schema.validate({"acme_thing": {"kind": "anything-at-all"}})

        assert [e for e in result.errors if "kind" in e] == []

    def test_the_miss_is_logged_at_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """DEBUG would make a misspelled ``enum_registry`` silent.

        The failure guarded against is a typo that disables checking for a
        field with no signal at any level an operator actually runs at.
        """
        with caplog.at_level(logging.WARNING):
            resolve_enum_options({"enum_registry": "no-such-registry"})

        assert any(
            "no-such-registry" in record.getMessage()
            for record in caplog.records
            if record.levelno >= logging.WARNING
        )

    def test_a_literal_enum_is_unaffected_by_the_registry_path(self) -> None:
        """Only ``enum_registry`` reaches the registry lookup."""
        assert resolve_enum_options({"enum": ["a", "b"]}) == ["a", "b"]
        assert resolve_enum_options({}) is None


class TestSchemaQuerySeesTheRegistry:
    """``get_valid_options`` backs the config builder and the wizard tools."""

    def test_provider_options_come_from_the_registry(self) -> None:
        options = DynaBotConfigSchema().get_valid_options("llm", "provider")

        assert "bedrock" in options
        assert set(options) == set(_provider_registry.list_keys())

    def test_a_closed_enum_field_is_unaffected(self) -> None:
        """Only the registry-backed field changes behavior.

        ``memory.type`` is a genuinely closed set implemented in this package,
        so it keeps a literal enum — the change is not "stop declaring enums",
        it is "stop copying a registry into one".
        """
        options = DynaBotConfigSchema().get_valid_options("memory", "type")

        assert options == ["buffer", "composite", "summary", "vector"]
