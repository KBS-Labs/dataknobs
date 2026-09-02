"""Unit tests for the unified model-metadata substrate (``model_profile``).

Covers the genuinely-new pieces built for the substrate:

- :func:`merge_partials` — the per-facet, first-non-``None``-wins merge (D-MERGE),
  including the load-bearing ``None``-vs-empty-collection distinction.
- :class:`LayeredModelProfileResolver` — source precedence composition.
- :func:`profile_from_loose` — loose config/resource → ``ModelProfile`` parsing.
- :func:`match_family_key` — the exact / family-alias / bare-alias matcher.
- The built-in sources (:class:`CallableModelMetadataSource`,
  :class:`ConfigOverrideSource`, :class:`BundledResourceSource`) and the
  consumer-extensible :data:`model_metadata_sources` registry.

These are behavioral unit tests of new code, exercised through the real public
surface (no mocks).
"""

from __future__ import annotations

import pytest

from dataknobs_llm.llm.base import ModelCapability
from dataknobs_llm.llm.model_profile import (
    CAPABILITY_ORDER,
    BundledResourceSource,
    CallableModelMetadataSource,
    ConfigOverrideSource,
    LayeredModelProfileResolver,
    ModelPricing,
    ModelProfile,
    PartialModelProfile,
    match_family_key,
    merge_partials,
    model_metadata_sources,
    profile_from_loose,
)


# ---------------------------------------------------------------------------
# merge_partials — D-MERGE (per facet, first non-None wins; override not union)
# ---------------------------------------------------------------------------


class TestMergePartials:
    def test_empty_input_is_all_none(self) -> None:
        assert merge_partials([]) == ModelProfile()

    def test_first_non_none_wins_per_facet(self) -> None:
        high = ModelProfile(max_output_tokens=100)
        low = ModelProfile(max_output_tokens=999, context_window=200_000)
        merged = merge_partials([high, low])
        assert merged.max_output_tokens == 100  # high precedence wins
        assert merged.context_window == 200_000  # only low knew it

    def test_none_is_skipped_lower_layer_fills(self) -> None:
        high = ModelProfile(max_output_tokens=None, context_window=8000)
        low = ModelProfile(max_output_tokens=4096)
        merged = merge_partials([high, low])
        assert merged.max_output_tokens == 4096  # high was None -> low fills
        assert merged.context_window == 8000

    def test_empty_frozenset_is_authoritative_and_wins(self) -> None:
        """A present empty collection = 'known none' and wins over a lower guess."""
        high = ModelProfile(rejected_params=frozenset())
        low = ModelProfile(rejected_params=frozenset({"temperature"}))
        merged = merge_partials([high, low])
        assert merged.rejected_params == frozenset()  # override, NOT union

    def test_empty_mapping_facet_is_authoritative(self) -> None:
        high = ModelProfile(param_remaps={})
        low = ModelProfile(param_remaps={"max_tokens": "max_completion_tokens"})
        assert merge_partials([high, low]).param_remaps == {}

    def test_override_not_union_for_capabilities(self) -> None:
        high = ModelProfile(capabilities=frozenset({ModelCapability.CHAT}))
        low = ModelProfile(capabilities=frozenset({ModelCapability.VISION}))
        merged = merge_partials([high, low])
        assert merged.capabilities == frozenset({ModelCapability.CHAT})

    def test_partial_alias_is_model_profile(self) -> None:
        assert PartialModelProfile is ModelProfile


# ---------------------------------------------------------------------------
# match_family_key — exact / family-alias / bare-alias
# ---------------------------------------------------------------------------


class TestMatchFamilyKey:
    def test_exact_match(self) -> None:
        assert match_family_key("claude-sonnet-5", ["claude-sonnet-5"]) == ("claude-sonnet-5")

    def test_family_alias_short_key_covers_dated_request(self) -> None:
        # resource-style: short family key is a substring of the dated request
        assert (
            match_family_key("claude-sonnet-5-20260514", ["claude-sonnet-5", "claude-opus-5"])
            == "claude-sonnet-5"
        )

    def test_bare_alias_request_is_substring_of_dated_key(self) -> None:
        # dynamic-style: bare request resolves against a longer dated cache key
        assert (
            match_family_key("claude-sonnet-5", ["claude-sonnet-5-20260514"])
            == "claude-sonnet-5-20260514"
        )

    def test_longest_overlap_wins(self) -> None:
        keys = ["claude", "claude-sonnet-5"]
        assert match_family_key("claude-sonnet-5-20260514", keys) == ("claude-sonnet-5")

    def test_no_match_returns_none(self) -> None:
        assert match_family_key("gpt-4", ["claude", "gemini"]) is None


# ---------------------------------------------------------------------------
# profile_from_loose — loose parsing / coercions
# ---------------------------------------------------------------------------


class TestProfileFromLoose:
    def test_empty_mapping_is_all_none(self) -> None:
        assert profile_from_loose({}) == ModelProfile()

    def test_capabilities_parsed_unknown_dropped(self) -> None:
        prof = profile_from_loose({"capabilities": ["vision", "function_calling", "bogus"]})
        assert prof.capabilities == frozenset(
            {ModelCapability.VISION, ModelCapability.FUNCTION_CALLING}
        )

    def test_empty_capability_list_is_authoritative_empty(self) -> None:
        assert profile_from_loose({"capabilities": []}).capabilities == frozenset()

    def test_rejected_params_and_aliases(self) -> None:
        prof = profile_from_loose({"rejected_params": ["temperature"], "aliases": ["a", "b"]})
        assert prof.rejected_params == frozenset({"temperature"})
        assert prof.aliases == ("a", "b")

    def test_param_remaps(self) -> None:
        prof = profile_from_loose({"param_remaps": {"max_tokens": "max_completion_tokens"}})
        assert prof.param_remaps == {"max_tokens": "max_completion_tokens"}

    def test_pricing_dict_becomes_model_pricing(self) -> None:
        prof = profile_from_loose({"pricing": {"input_per_mtok": 3.0, "output_per_mtok": 15.0}})
        assert prof.pricing == ModelPricing(input_per_mtok=3.0, output_per_mtok=15.0)

    def test_pricing_instance_passes_through(self) -> None:
        pricing = ModelPricing(input_per_mtok=1.0)
        assert profile_from_loose({"pricing": pricing}).pricing is pricing

    def test_ints_and_bool(self) -> None:
        prof = profile_from_loose(
            {"context_window": 200000, "max_output_tokens": 8192, "available": True}
        )
        assert prof.context_window == 200000
        assert prof.max_output_tokens == 8192
        assert prof.available is True


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------


class TestCallableSource:
    def test_wraps_callable(self) -> None:
        src = CallableModelMetadataSource("x", lambda m: ModelProfile(max_output_tokens=len(m)))
        assert src.name == "x"
        assert src.resolve("abcd").max_output_tokens == 4


class TestConfigOverrideSource:
    def test_absent_override_is_empty(self) -> None:
        assert ConfigOverrideSource(None).resolve("m") == ModelProfile()

    def test_flat_mapping_applies_to_any_model(self) -> None:
        src = ConfigOverrideSource({"max_output_tokens": 4096})
        assert src.resolve("anything").max_output_tokens == 4096

    def test_per_model_mapping_matches_by_family(self) -> None:
        src = ConfigOverrideSource({"claude-opus-5": {"max_output_tokens": 4096}})
        # family-alias: dated request resolves to the short model key
        assert src.resolve("claude-opus-5-20260101").max_output_tokens == 4096
        # a different model is untouched
        assert src.resolve("gpt-4").max_output_tokens is None


class TestBundledResourceSource:
    def test_dict_constructor_family_match(self) -> None:
        src = BundledResourceSource(
            {
                "claude-sonnet-5": ModelProfile(max_output_tokens=128000),
                "claude-haiku-4-5": ModelProfile(max_output_tokens=64000),
            }
        )
        assert src.resolve("claude-sonnet-5-20260514").max_output_tokens == 128000
        assert src.resolve("claude-haiku-4-5-20251001").max_output_tokens == 64000

    def test_unknown_model_is_empty(self) -> None:
        src = BundledResourceSource({"claude-sonnet-5": ModelProfile(max_output_tokens=1)})
        assert src.resolve("gpt-4") == ModelProfile()

    def test_missing_resource_degrades_to_empty(self) -> None:
        # A bogus package/resource must not raise — it degrades to an empty source.
        src = BundledResourceSource.from_resource(
            "dataknobs_llm.llm.providers", "data/does_not_exist.yaml"
        )
        assert src.resolve("claude-sonnet-5") == ModelProfile()


# ---------------------------------------------------------------------------
# LayeredModelProfileResolver — precedence composition
# ---------------------------------------------------------------------------


class TestLayeredResolver:
    def _src(self, name: str, **facets: object) -> CallableModelMetadataSource:
        return CallableModelMetadataSource(name, lambda _m, f=facets: ModelProfile(**f))

    def test_precedence_config_over_live_over_resource(self) -> None:
        resolver = LayeredModelProfileResolver(
            [
                self._src("config", max_output_tokens=100),
                self._src("live", max_output_tokens=200, context_window=8000),
                self._src("resource", max_output_tokens=300, context_window=9000),
            ]
        )
        prof = resolver.resolve("m")
        assert prof.max_output_tokens == 100  # config wins
        assert prof.context_window == 8000  # config silent -> live fills

    def test_sources_property_preserves_order(self) -> None:
        a = self._src("a")
        b = self._src("b")
        resolver = LayeredModelProfileResolver([a, b])
        assert [s.name for s in resolver.sources] == ["a", "b"]


# ---------------------------------------------------------------------------
# model_metadata_sources registry — consumer extension point
# ---------------------------------------------------------------------------


class TestSourceRegistry:
    def test_register_create_and_unknown(self) -> None:
        name = "_test_gateway_source"

        def factory(config: dict) -> CallableModelMetadataSource:
            return CallableModelMetadataSource(name, lambda _m: ModelProfile(available=True))

        model_metadata_sources.register(name, factory)
        try:
            src = model_metadata_sources.create(key=name, config={})
            assert src.resolve("m").available is True
        finally:
            model_metadata_sources.unregister(name)

        with pytest.raises(ValueError, match="model metadata source"):
            model_metadata_sources.create(key="_never_registered", config={})


# ---------------------------------------------------------------------------
# CAPABILITY_ORDER is a projection, so a member missing from it is dropped
# ---------------------------------------------------------------------------


def test_capability_order_covers_the_enum() -> None:
    """Every ``ModelCapability`` must be listed, and nothing else may be.

    ``ProfileDetectionMixin._detect_capabilities`` projects the resolved
    frozenset through this tuple --- ``[c for c in CAPABILITY_ORDER if c in
    capabilities]`` --- so a member left out is not merely unordered. It is
    dropped from every provider that resolves capabilities through a profile,
    while the source and the bundled resource both still report it, and
    nothing anywhere raises.

    That is not hypothetical: ``EMBEDDING_DIMENSIONS`` was added to the enum,
    declared in two bundled resources and returned by both OpenAI sources, and
    ``get_capabilities()`` answered ``['embeddings']`` for every model until
    this tuple learned the name. A guard is cheaper than finding it twice.
    """
    assert set(CAPABILITY_ORDER) == set(ModelCapability)
    assert len(CAPABILITY_ORDER) == len(set(CAPABILITY_ORDER))
