"""A ``$resource`` reference declares what happens when its resource is absent.

Two resolvers read the same reference format and disagree about a missing
resource: :meth:`EnvironmentAwareConfig.resolve_for_build` warns and degrades
to the reference's inline defaults, while :meth:`ConfigBindingResolver.resolve`
raises. Neither *chose* — the shared primitive
(:meth:`EnvironmentConfig.get_resource`) overloads one parameter with two
meanings, so a caller holding defaults inherits leniency and a caller holding
none inherits strictness.

This file pins the vocabulary that lets a reference say which it wants, and
the precedence chain that decides when it does not say.

Three of these reproduce defects rather than covering a feature, and they were
written to fail first:

* a non-empty ``$requires`` on an *absent* resource resolved silently — the
  weaker failure (present but under-capable) aborted the build while the total
  failure proceeded;
* a ``$``-prefixed key outside the marker set was promoted to an inline
  default and handed to a factory as a keyword argument;
* which is why ``$requred: true`` read as *not required* — one character from
  the marker meant to close this class, at the exact site meant to close it.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from dataknobs_config.environment_aware import EnvironmentAwareConfig
from dataknobs_config.environment_config import (
    EnvironmentConfig,
    ResourceNotFoundError,
)
from dataknobs_config.exceptions import ConfigError


@pytest.fixture
def environment() -> EnvironmentConfig:
    """An environment defining one resource, so "absent" means absent."""
    return EnvironmentConfig(
        name="repro",
        resources={
            "vector_stores": {
                "present": {
                    "backend": "pgvector",
                    "capabilities": ["similarity_search"],
                },
            },
        },
    )


def _resolve(environment: EnvironmentConfig, reference: dict[str, Any]) -> dict[str, Any]:
    """Resolve a single reference under ``vector_store``, declaring no policy.

    Deliberately touches none of the new vocabulary, so the three
    reproduce-first cases below fail against unmodified code *for the reason
    the defect exists* rather than on an unknown keyword argument.
    """
    config = EnvironmentAwareConfig(
        config={"vector_store": reference},
        environment=environment,
    )
    resolved = config.resolve_for_build(resolve_env_vars=False)
    return dict(resolved["vector_store"])


class TestRequiresOnAMissingResource:
    """``$requires`` is a stronger claim than presence, so absence fails it."""

    def test_requires_on_missing_resource_raises(self, environment):
        """A capability a missing resource cannot possibly satisfy.

        Reproduces the severity inversion: a resource that exists but lacks a
        declared capability raised, while a resource that does not exist at
        all resolved to ``{}`` and was handed to a factory.
        """
        with pytest.raises(ResourceNotFoundError) as excinfo:
            _resolve(
                environment,
                {
                    "$resource": "absent",
                    "type": "vector_stores",
                    "$requires": ["persistence"],
                },
            )

        message = str(excinfo.value)
        assert "absent" in message
        assert "persistence" in message, "the unmet requirement must be named"

    def test_present_but_under_capable_still_raises(self, environment):
        """The pre-existing half of the pair is unchanged."""
        with pytest.raises(ConfigError, match="persistence"):
            _resolve(
                environment,
                {
                    "$resource": "present",
                    "type": "vector_stores",
                    "$requires": ["persistence"],
                },
            )

    def test_requires_with_explicit_opt_out_degrades(self, environment, caplog):
        """``$required: false`` beside ``$requires`` is coherent, not contradictory.

        The author said "if it is there it must do X; it may be absent."
        """
        with caplog.at_level(logging.WARNING):
            resolved = _resolve(
                environment,
                {
                    "$resource": "absent",
                    "type": "vector_stores",
                    "$requires": ["persistence"],
                    "$required": False,
                    "metric": "cosine",
                },
            )

        assert resolved == {"metric": "cosine"}
        assert any(record.levelno == logging.WARNING for record in caplog.records)


class TestMalformedMarkers:
    """A ``$``-prefixed key is a marker or it is a mistake — never a default."""

    def test_unknown_dollar_marker_rejected(self, environment):
        """``$reqiures`` reached the factory as a keyword argument.

        The marker set is closed and the defaults comprehension takes
        everything else, so a misspelling was not rejected — it was promoted.
        """
        with pytest.raises(ConfigError) as excinfo:
            _resolve(
                environment,
                {
                    "$resource": "present",
                    "type": "vector_stores",
                    "$reqiures": ["persistence"],
                },
            )

        message = str(excinfo.value)
        assert "$reqiures" in message
        assert "$requires" in message, "the valid markers must be listed"

    def test_misspelled_required_marker_is_rejected(self, environment):
        """``$requred: true`` must not silently mean *not required*.

        This is the reason the guard is a precondition for the policy rather
        than an adjacent nicety: the new marker lands one character from an
        existing one, in the same block, and the failure mode of a typo is
        exactly the silent degrade the policy exists to close.
        """
        with pytest.raises(ConfigError, match=r"\$requred"):
            _resolve(
                environment,
                {
                    "$resource": "absent",
                    "type": "vector_stores",
                    "$requred": True,
                },
            )

    def test_valid_markers_are_not_rejected(self, environment):
        """The guard must not fire on the vocabulary it is guarding."""
        resolved = _resolve(
            environment,
            {
                "$resource": "present",
                "type": "vector_stores",
                "$requires": ["similarity_search"],
                "$required": True,
            },
        )
        assert resolved["backend"] == "pgvector"

    def test_non_dollar_keys_are_still_inline_defaults(self, environment):
        """Only ``$``-prefixed keys are markers; the rest are data."""
        resolved = _resolve(
            environment,
            {
                "$resource": "absent",
                "type": "vector_stores",
                "metric": "cosine",
                "dimensions": 1536,
            },
        )
        assert resolved == {"metric": "cosine", "dimensions": 1536}


def _resolve_with_policy(
    environment: EnvironmentConfig,
    reference: dict[str, Any],
    *,
    instance: bool | None = None,
    call: bool | None = None,
) -> dict[str, Any]:
    """Resolve a reference with the two code levels set independently."""
    config = EnvironmentAwareConfig(
        config={"vector_store": reference},
        environment=environment,
        strict_resources=instance,
    )
    resolved = config.resolve_for_build(resolve_env_vars=False, strict_resources=call)
    return dict(resolved["vector_store"])


def _absent(**extra: Any) -> dict[str, Any]:
    """A reference to a resource the fixture environment does not define."""
    return {"$resource": "absent", "type": "vector_stores", **extra}


class TestOutcomeMatrix:
    """Every row of the contract, including the ones that do not change."""

    def test_found_resolves(self, environment):
        assert (
            _resolve(environment, {"$resource": "present", "type": "vector_stores"})["backend"]
            == "pgvector"
        )

    def test_missing_lenient_with_defaults_warns_and_degrades(self, environment, caplog):
        with caplog.at_level(logging.WARNING):
            resolved = _resolve(environment, _absent(metric="cosine"))

        assert resolved == {"metric": "cosine"}
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warnings, "the degrade is silent otherwise -- the warning is the only signal"
        assert "inline defaults" in warnings[0].getMessage()

    def test_missing_lenient_without_defaults_warns_and_empties(self, environment, caplog):
        with caplog.at_level(logging.WARNING):
            resolved = _resolve(environment, _absent())

        assert resolved == {}
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warnings
        assert "empty config" in warnings[0].getMessage(), (
            "the two degradations must stay distinguishable: one still builds, "
            "the other is a factory about to be called with no arguments"
        )

    def test_missing_and_required_raises(self, environment):
        with pytest.raises(ResourceNotFoundError, match=r"\$required"):
            _resolve(environment, _absent(**{"$required": True}))

    def test_required_beats_inline_defaults(self, environment):
        """Declaring defaults does not soften an explicit ``$required``."""
        with pytest.raises(ResourceNotFoundError):
            _resolve(environment, _absent(metric="cosine", **{"$required": True}))

    @pytest.mark.parametrize("value", ["yes", "on", 1, 0, None, "  ", "truthy"])
    def test_unparseable_required_value_raises(self, environment, value):
        """Fail closed. A value that silently read as False is the whole defect."""
        with pytest.raises(ConfigError, match="boolean"):
            _resolve(environment, _absent(**{"$required": value}))

    @pytest.mark.parametrize(
        ("value", "expect_raise"),
        [(True, True), (False, False), ("true", True), ("false", False), ("TRUE", True)],
    )
    def test_required_accepts_bools_and_their_string_spellings(
        self, environment, value, expect_raise
    ):
        """``$required: ${STRICT}`` arrives as text -- substitution does not coerce."""
        if expect_raise:
            with pytest.raises(ResourceNotFoundError):
                _resolve(environment, _absent(**{"$required": value}))
        else:
            assert _resolve(environment, _absent(**{"$required": value})) == {}

    def test_malformed_required_raises_even_when_the_resource_is_present(self, environment):
        """Otherwise the error first appears in the deployment that lacks the resource."""
        with pytest.raises(ConfigError, match="boolean"):
            _resolve(
                environment,
                {"$resource": "present", "type": "vector_stores", "$required": "maybe"},
            )

    def test_failure_message_names_the_lever(self, environment):
        """An operator has to know which level made it strict to respond to it."""
        with pytest.raises(ResourceNotFoundError) as excinfo:
            _resolve_with_policy(environment, _absent(), call=True)

        assert "strict_resources=True" in str(excinfo.value)


class TestPrecedenceChain:
    """Four levels, four owners; each adjacent pair proves its direction."""

    def test_environment_setting_makes_a_plain_reference_strict(self):
        """The operator's level -- and the only one a generated reference can reach."""
        environment = EnvironmentConfig(
            name="prod", settings={"strict_resources": True}, resources={}
        )
        with pytest.raises(ResourceNotFoundError, match="strict_resources"):
            _resolve(environment, _absent())

    def test_environment_setting_accepts_its_string_spelling(self):
        """``strict_resources: ${STRICT}`` reaches the setting as text too."""
        environment = EnvironmentConfig(
            name="prod", settings={"strict_resources": "true"}, resources={}
        )
        with pytest.raises(ResourceNotFoundError):
            _resolve(environment, _absent())

    def test_unparseable_environment_setting_raises(self):
        """It must not silently read as lenient -- same rule as the marker."""
        environment = EnvironmentConfig(
            name="prod", settings={"strict_resources": "sometimes"}, resources={}
        )
        with pytest.raises(ConfigError, match="strict_resources"):
            _resolve(environment, _absent())

    def test_instance_level_beats_the_environment_setting(self):
        environment = EnvironmentConfig(
            name="prod", settings={"strict_resources": True}, resources={}
        )
        assert _resolve_with_policy(environment, _absent(), instance=False) == {}

    def test_call_level_beats_the_instance_level(self, environment):
        assert _resolve_with_policy(environment, _absent(), instance=True, call=False) == {}

    def test_call_level_strict_beats_a_lenient_instance(self, environment):
        with pytest.raises(ResourceNotFoundError):
            _resolve_with_policy(environment, _absent(), instance=False, call=True)

    def test_reference_marker_beats_every_code_level(self, environment):
        assert (
            _resolve_with_policy(
                environment, _absent(**{"$required": False}), instance=True, call=True
            )
            == {}
        )

    def test_requires_beats_a_lenient_code_level(self, environment):
        """A resolver-wide ``False`` speaks for references that said nothing.

        This one said something.
        """
        with pytest.raises(ResourceNotFoundError, match="persistence"):
            _resolve_with_policy(
                environment,
                _absent(**{"$requires": ["persistence"]}),
                instance=False,
                call=False,
            )

    def test_default_is_unchanged_when_nothing_declares_a_policy(self, environment):
        assert _resolve(environment, _absent(metric="cosine")) == {"metric": "cosine"}

    def test_instance_policy_is_readable(self, environment):
        config = EnvironmentAwareConfig(config={}, environment=environment, strict_resources=True)
        assert config.strict_resources is True
        assert EnvironmentAwareConfig(config={}, environment=environment).strict_resources is None


class TestPolicyPropagation:
    """A missed forward is silent, so each carrier gets its own test."""

    @pytest.fixture
    def nesting_environment(self) -> EnvironmentConfig:
        """A resource that itself carries a reference to an absent resource."""
        return EnvironmentConfig(
            name="nest",
            resources={
                "vector_stores": {
                    "outer": {
                        "backend": "pgvector",
                        "embedder": {"$resource": "absent", "type": "embedders"},
                    },
                },
            },
        )

    def test_strict_reaches_a_reference_nested_inside_a_resolved_resource(
        self, nesting_environment
    ):
        with pytest.raises(ResourceNotFoundError, match="absent"):
            _resolve_with_policy(
                nesting_environment,
                {"$resource": "outer", "type": "vector_stores"},
                call=True,
            )

    def test_strict_reaches_a_reference_nested_in_an_inline_default(self, environment):
        """The outer reference resolves, so its surviving defaults are walked."""
        reference = {
            "$resource": "present",
            "type": "vector_stores",
            "embedder": {"$resource": "also-absent", "type": "embedders"},
        }
        with pytest.raises(ResourceNotFoundError, match="also-absent"):
            _resolve_with_policy(environment, reference, call=True)

    def test_strict_reaches_a_reference_inside_a_list(self, environment):
        config = EnvironmentAwareConfig(
            config={"stores": [{"$resource": "absent", "type": "vector_stores"}]},
            environment=environment,
            strict_resources=True,
        )
        with pytest.raises(ResourceNotFoundError, match="absent"):
            config.resolve_for_build(resolve_env_vars=False)

    def test_with_environment_preserves_the_policy(self, environment):
        """The common path for a caller supplying config and environment apart.

        Dropping the flag here would revert strict mode to lenient exactly
        where a second environment enters.
        """
        original = EnvironmentAwareConfig(
            config={"vector_store": _absent()},
            environment=environment,
            strict_resources=True,
        )
        moved = original.with_environment(EnvironmentConfig(name="other", resources={}))

        assert moved.strict_resources is True
        with pytest.raises(ResourceNotFoundError):
            moved.resolve_for_build(resolve_env_vars=False)

    def test_from_dict_forwards_the_policy(self, tmp_path):
        config = EnvironmentAwareConfig.from_dict(
            {"vector_store": _absent()},
            environment="nonexistent",
            env_dir=tmp_path,
            strict_resources=True,
        )
        assert config.strict_resources is True
        with pytest.raises(ResourceNotFoundError):
            config.resolve_for_build(resolve_env_vars=False)

    def test_load_app_forwards_the_policy(self, tmp_path):
        import yaml

        app_dir = tmp_path / "apps"
        app_dir.mkdir()
        (app_dir / "demo.yaml").write_text(yaml.safe_dump({"vector_store": _absent()}))

        config = EnvironmentAwareConfig.load_app(
            "demo",
            app_dir=app_dir,
            env_dir=tmp_path / "environments",
            environment="dev",
            strict_resources=True,
        )
        assert config.strict_resources is True
        with pytest.raises(ResourceNotFoundError):
            config.resolve_for_build(resolve_env_vars=False)


class TestFindUnresolvedResources:
    """Raise-on-first is right for a build and wrong for a preflight."""

    @pytest.fixture
    def survey_config(self, environment) -> EnvironmentAwareConfig:
        return EnvironmentAwareConfig(
            config={
                "bot": {
                    "knowledge_base": {
                        "vector_store": _absent(metric="cosine"),
                        "embedder": {"$resource": "missing-embedder", "type": "embedders"},
                    },
                    "store": {"$resource": "present", "type": "vector_stores"},
                },
                "extras": [
                    {"$resource": "in-a-list", "type": "databases", "$required": True},
                ],
            },
            environment=environment,
        )

    def test_reports_every_unresolvable_reference_in_one_pass(self, survey_config):
        found = survey_config.find_unresolved_resources()
        assert {ref.resource_name for ref in found} == {
            "absent",
            "missing-embedder",
            "in-a-list",
        }

    def test_paths_are_dotted_with_list_indices(self, survey_config):
        by_name = {ref.resource_name: ref for ref in survey_config.find_unresolved_resources()}
        assert by_name["absent"].path == "bot.knowledge_base.vector_store"
        assert by_name["missing-embedder"].path == "bot.knowledge_base.embedder"
        assert by_name["in-a-list"].path == "extras[0]"

    def test_reports_the_effective_policy_per_reference(self, survey_config):
        by_name = {ref.resource_name: ref for ref in survey_config.find_unresolved_resources()}
        assert by_name["in-a-list"].required is True
        assert by_name["absent"].required is False
        assert by_name["absent"].has_inline_defaults is True
        assert by_name["missing-embedder"].has_inline_defaults is False

    def test_policy_argument_changes_required_not_membership(self, survey_config):
        strict = survey_config.find_unresolved_resources(strict_resources=True)
        assert len(strict) == 3
        assert all(ref.required for ref in strict)

    def test_raises_nothing_and_builds_nothing(self, survey_config):
        """Even the reference that declares ``$required: true`` is reported, not raised."""
        assert survey_config.find_unresolved_resources()

    def test_empty_when_everything_resolves(self, environment):
        config = EnvironmentAwareConfig(
            config={"store": {"$resource": "present", "type": "vector_stores"}},
            environment=environment,
        )
        assert config.find_unresolved_resources() == []

    def test_resource_types_and_names_are_reported(self, survey_config):
        by_name = {ref.resource_name: ref for ref in survey_config.find_unresolved_resources()}
        assert by_name["absent"].resource_type == "vector_stores"
        assert by_name["in-a-list"].resource_type == "databases"

    def test_a_variable_selected_reference_is_reported_under_its_resolved_name(
        self, environment, monkeypatch
    ):
        """The raw ``${VAR}`` text would be a finding nobody can act on."""
        monkeypatch.setenv("LLM_BINDING", "resolved-name")
        config = EnvironmentAwareConfig(
            config={"llm": {"$resource": "${LLM_BINDING}", "type": "llm_providers"}},
            environment=environment,
        )
        found = config.find_unresolved_resources()
        assert [ref.resource_name for ref in found] == ["resolved-name"]

    def test_a_nested_variable_selected_reference_is_also_resolved(self, environment, monkeypatch):
        """The survey must name what the build would look up, at every depth.

        A reference's inline defaults are held back by the entry pass and
        expanded later at their own splice, so descending into them without
        repeating that expansion reports a raw ``${VAR}`` while the build
        reports the name it expands to.
        """
        monkeypatch.setenv("INNER", "inner-name")
        config = EnvironmentAwareConfig(
            config={
                "store": {
                    "$resource": "present",
                    "type": "vector_stores",
                    "fallback": {"$resource": "${INNER}", "type": "embedders"},
                }
            },
            environment=environment,
        )

        surveyed = [ref.resource_name for ref in config.find_unresolved_resources()]

        with pytest.raises(ResourceNotFoundError) as excinfo:
            config.resolve_for_build(strict_resources=True)

        assert surveyed == ["inner-name"]
        assert "inner-name" in str(excinfo.value), "the two must agree on the name"

    def test_surveys_a_reference_nested_inside_a_resolved_resource(self):
        environment = EnvironmentConfig(
            name="nest",
            resources={
                "vector_stores": {
                    "outer": {"embedder": {"$resource": "absent", "type": "embedders"}},
                },
            },
        )
        config = EnvironmentAwareConfig(
            config={"store": {"$resource": "outer", "type": "vector_stores"}},
            environment=environment,
        )
        found = config.find_unresolved_resources()
        assert [ref.resource_name for ref in found] == ["absent"]

    def test_a_self_referential_resource_is_visited_once(self):
        """A survey must not hang where a build would."""
        environment = EnvironmentConfig(
            name="cycle",
            resources={
                "vector_stores": {
                    "loop": {"inner": {"$resource": "loop", "type": "vector_stores"}},
                },
            },
        )
        config = EnvironmentAwareConfig(
            config={"store": {"$resource": "loop", "type": "vector_stores"}},
            environment=environment,
        )
        assert config.find_unresolved_resources() == []

    def test_a_malformed_reference_raises_rather_than_being_skipped(self, environment):
        """A survey reporting a tree sound while the build raises is worse than none."""
        config = EnvironmentAwareConfig(
            config={"store": _absent(**{"$reqiures": ["x"]})},
            environment=environment,
        )
        with pytest.raises(ConfigError, match=r"\$reqiures"):
            config.find_unresolved_resources()

    def test_config_key_scopes_the_survey(self, survey_config):
        found = survey_config.find_unresolved_resources(config_key="extras")
        assert [ref.resource_name for ref in found] == ["in-a-list"]


class TestBothResolversAgree:
    """The recurrence guard.

    The defect was two resolvers reading one format and disagreeing about a
    missing resource, with neither having chosen. These pin the agreement, so
    a future change that quietly makes one of them lenient again fails here
    rather than in a deployment.
    """

    def test_the_reference_resolver_never_returns_for_a_required_missing_resource(
        self, environment
    ):
        with pytest.raises(ResourceNotFoundError):
            _resolve(environment, _absent(**{"$required": True}))
        with pytest.raises(ResourceNotFoundError):
            _resolve_with_policy(environment, _absent(), call=True)

    def test_the_binding_resolver_never_returns_for_a_missing_resource(self, environment):
        """It has no reference to read a policy off, so it can only be strict."""
        from dataknobs_config import ConfigBindingResolver

        resolver = ConfigBindingResolver(environment)
        resolver.register_factory("vector_stores", lambda **config: config)

        with pytest.raises(ResourceNotFoundError):
            resolver.resolve("vector_stores", "absent")

    def test_both_resolvers_raise_the_same_exception_type(self, environment):
        """The asymmetry the item closes was in the type as much as the policy."""
        from dataknobs_config import ConfigBindingResolver

        resolver = ConfigBindingResolver(environment)
        resolver.register_factory("vector_stores", lambda **config: config)

        with pytest.raises(ResourceNotFoundError) as from_binding:
            resolver.resolve("vector_stores", "absent")
        with pytest.raises(ResourceNotFoundError) as from_reference:
            _resolve(environment, _absent(**{"$required": True}))

        assert type(from_binding.value) is type(from_reference.value)

    def test_a_malformed_reference_is_a_config_error_not_a_missing_resource(self, environment):
        """The other half of the exception-type rule: malformed is not absent."""
        with pytest.raises(ConfigError) as excinfo:
            _resolve(environment, _absent(**{"$nonsense": 1}))
        assert not isinstance(excinfo.value, ResourceNotFoundError)
