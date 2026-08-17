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

Several of these reproduce defects rather than covering a feature, and were
written to fail first.

In the vocabulary itself:

* a non-empty ``$requires`` on an *absent* resource resolved silently — the
  weaker failure (present but under-capable) aborted the build while the total
  failure proceeded;
* a ``$``-prefixed key outside the marker set was promoted to an inline
  default and handed to a factory as a keyword argument;
* which is why ``$requred: true`` read as *not required* — one character from
  the marker meant to close this class, at the exact site meant to close it;
* ``$requires: persistence`` was iterated character by character, because only
  its sibling's *value* was ever validated, not its own;
* and ``$resorce:`` — the one misspelling a guard that fires on ``$resource``
  cannot fire on — left an ordinary dict that resolved to itself.

In the machinery that reads it, where the recurring shape is **two readers of
one format**:

* the survey walked the tree separately from the build and disagreed with it
  in both directions, so a preflight could report a tree sound that the build
  could not resolve;
* neither walk guarded against a resource reaching itself on the build side,
  so a cycle was a ``RecursionError``;
* the binding resolver stopped at the resource it looked up, handing a
  reference nested inside one to a factory as a literal dict;
* and the environment's ``strict_resources`` setting was parsed on the branch
  taken only when a resource is absent, so a malformed flag hid until one was.
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

    Deliberately touches none of the new vocabulary, so the reproduce-first
    cases below fail against unmodified code *for the reason the defect
    exists* rather than on an unknown keyword argument.
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
        """It must not silently read as lenient -- same rule as the marker.

        Rejected where it is written rather than where it is read, so the
        deployment that meets it is not the one that happens to be missing a
        resource. See :class:`TestEnvironmentSettingIsValidatedWhereItIsWritten`.
        """
        with pytest.raises(ConfigError, match="strict_resources"):
            EnvironmentConfig(name="prod", settings={"strict_resources": "sometimes"}, resources={})

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

    def test_the_binding_resolver_resolves_a_reference_nested_in_a_resource(self):
        """A resource may carry a reference, and both resolvers must follow it.

        Reproduces the second half of the disagreement. The policy was
        unified while the *reach* was not: ``ConfigBindingResolver`` looked a
        resource up and handed it straight to the factory, so a nested
        reference arrived as a literal ``{"$resource": ...}`` keyword
        argument -- the same silent-degrade shape the marker guard closes,
        one layer down.
        """
        from dataknobs_config import ConfigBindingResolver

        environment = EnvironmentConfig(
            name="nest",
            resources={
                "vector_stores": {
                    "outer": {
                        "backend": "pgvector",
                        "embedder": {"$resource": "emb", "type": "embedders"},
                    },
                },
                "embedders": {"emb": {"model": "minilm"}},
            },
        )
        resolver = ConfigBindingResolver(environment)
        resolver.register_factory("vector_stores", lambda **config: config)

        created = resolver.resolve("vector_stores", "outer")

        assert created["embedder"] == {"model": "minilm"}

    def test_the_binding_resolver_raises_on_a_nested_missing_resource(self):
        """Strictness reaches the nested reference too, or it stops one level down."""
        from dataknobs_config import ConfigBindingResolver

        environment = EnvironmentConfig(
            name="nest",
            resources={
                "vector_stores": {
                    "outer": {"embedder": {"$resource": "gone", "type": "embedders"}},
                },
            },
        )
        resolver = ConfigBindingResolver(environment)
        resolver.register_factory("vector_stores", lambda **config: config)

        with pytest.raises(ResourceNotFoundError, match="gone"):
            resolver.resolve("vector_stores", "outer")

    def test_a_nested_reference_may_still_opt_out(self):
        """Strict is the binding resolver's default, not its only answer.

        It has no reference of its own to read a policy off, which is why it
        is strict. A reference *nested* in the resource it resolves does have
        one, and it is read there like anywhere else.
        """
        from dataknobs_config import ConfigBindingResolver

        environment = EnvironmentConfig(
            name="nest",
            resources={
                "vector_stores": {
                    "outer": {
                        "embedder": {
                            "$resource": "gone",
                            "type": "embedders",
                            "$required": False,
                            "model": "fallback",
                        },
                    },
                },
            },
        )
        resolver = ConfigBindingResolver(environment)
        resolver.register_factory("vector_stores", lambda **config: config)

        created = resolver.resolve("vector_stores", "outer")

        assert created["embedder"] == {"model": "fallback"}


class TestTheSurveyAndTheBuildAgree:
    """The survey exists to predict the build, so a disagreement is the defect.

    Both walked the tree, and they walked it differently -- in both
    directions. The survey descended where the build discards, and stopped
    where the build descends. Each of these pins one direction against the
    build itself rather than against an expected list, so the two cannot
    drift apart again without failing here.
    """

    def test_a_second_reference_to_one_resource_still_surveys_its_own_defaults(self):
        """A visited-set keyed on the resource swallowed the reference's own defaults.

        The set is shared-subtree bookkeeping for the *resource*; a
        reference's inline defaults belong to the call site, not to the
        resource it names. Returning early for the second reference dropped
        them -- and because the guard is a dict-iteration-order artefact,
        swapping the two keys hid it.
        """
        environment = EnvironmentConfig(
            name="repeat",
            resources={"vector_stores": {"present": {"backend": "pgvector"}}},
        )
        config = EnvironmentAwareConfig(
            config={
                "first": {"$resource": "present", "type": "vector_stores"},
                "second": {
                    "$resource": "present",
                    "type": "vector_stores",
                    "embedder": {"$resource": "gone", "type": "embedders"},
                },
            },
            environment=environment,
        )

        surveyed = [ref.resource_name for ref in config.find_unresolved_resources()]

        with pytest.raises(ResourceNotFoundError, match="gone"):
            config.resolve_for_build(resolve_env_vars=False, strict_resources=True)

        assert surveyed == ["gone"], "the build reaches it, so the survey must report it"

    def test_a_default_the_environment_overrides_is_not_reported(self):
        """The splice discards it, so a reference among it is unreachable.

        The build merges an inline default only where the environment did not
        supply the key. Surveying every default regardless reports a finding
        an operator cannot act on -- the build never looks at it.
        """
        environment = EnvironmentConfig(
            name="override",
            resources={
                "vector_stores": {
                    "present": {"backend": "pgvector", "embedder": {"model": "from-env"}},
                },
            },
        )
        config = EnvironmentAwareConfig(
            config={
                "store": {
                    "$resource": "present",
                    "type": "vector_stores",
                    "embedder": {"$resource": "gone", "type": "embedders"},
                },
            },
            environment=environment,
        )

        surveyed = config.find_unresolved_resources()
        built = config.resolve_for_build(resolve_env_vars=False, strict_resources=True)

        assert surveyed == [], "the environment supplied `embedder`, so the default is discarded"
        assert built["store"]["embedder"] == {"model": "from-env"}

    def test_a_capability_failure_is_raised_rather_than_certified_sound(self, environment):
        """A survey that reports a tree sound while the build raises is worse than none.

        Presence is the question the survey answers by listing. Every other
        way a reference can fail -- malformed, cyclic, or naming a resource
        that does not declare a capability it ``$requires`` -- raises here,
        for the same reason a malformed reference always did.
        """
        config = EnvironmentAwareConfig(
            config={
                "store": {
                    "$resource": "present",
                    "type": "vector_stores",
                    "$requires": ["persistence"],
                },
            },
            environment=environment,
        )
        with pytest.raises(ConfigError, match="persistence"):
            config.find_unresolved_resources()

    def test_a_root_level_reference_is_reported_with_an_empty_path(self, environment):
        """A dotted path of zero segments. Rendered ``<root>`` where a message needs a name."""
        config = EnvironmentAwareConfig(config=_absent(), environment=environment)
        found = config.find_unresolved_resources()
        assert [(ref.path, ref.resource_name) for ref in found] == [("", "absent")]


class TestReferenceCycles:
    """A resource that refers to itself terminates, and says so."""

    @pytest.fixture
    def cyclic(self) -> EnvironmentAwareConfig:
        environment = EnvironmentConfig(
            name="cycle",
            resources={
                "vector_stores": {
                    "loop": {"inner": {"$resource": "loop", "type": "vector_stores"}},
                },
            },
        )
        return EnvironmentAwareConfig(
            config={"store": {"$resource": "loop", "type": "vector_stores"}},
            environment=environment,
        )

    def test_the_build_reports_the_cycle_rather_than_recursing(self, cyclic):
        """It raised ``RecursionError`` -- a stack trace, not a diagnosis."""
        with pytest.raises(ConfigError, match="cycle") as excinfo:
            cyclic.resolve_for_build(resolve_env_vars=False)
        assert "loop" in str(excinfo.value), "the cycle must name the resource it closes on"

    def test_the_survey_reports_the_cycle_the_same_way(self, cyclic):
        """The survey guarded itself and left the build to crash.

        A guard on one side of a build/survey pair is the divergence this
        file exists to prevent: the survey returned ``[]`` -- *sound* -- for
        a config the build could not resolve at all.
        """
        with pytest.raises(ConfigError, match="cycle"):
            cyclic.find_unresolved_resources()

    def test_two_references_to_one_resource_are_not_a_cycle(self, environment):
        """A shared subtree is not a loop, and must not be reported as one."""
        config = EnvironmentAwareConfig(
            config={
                "a": {"$resource": "present", "type": "vector_stores"},
                "b": {"$resource": "present", "type": "vector_stores"},
            },
            environment=environment,
        )
        built = config.resolve_for_build(resolve_env_vars=False, strict_resources=True)
        assert built["a"]["backend"] == built["b"]["backend"] == "pgvector"

    def test_a_default_naming_its_own_reference_is_not_a_cycle(self, environment):
        """The default is spliced after the resource is finished, not inside it."""
        built = _resolve(
            environment,
            {
                "$resource": "present",
                "type": "vector_stores",
                "fallback": {"$resource": "present", "type": "vector_stores"},
            },
        )
        assert built["fallback"]["backend"] == "pgvector"


class TestMarkerShape:
    """The marker set is closed; so is what each marker may hold."""

    def test_requires_must_be_a_list(self, environment):
        """``$requires: persistence`` iterated as characters.

        A scalar is the natural mistake -- its sibling ``$required`` takes
        one. Unvalidated it was truthy, so an absent resource failed with
        ``$requires: ['p', 'e', 'r', ...]`` and a present one reported eight
        missing single-character capabilities.
        """
        with pytest.raises(ConfigError, match=r"\$requires"):
            _resolve(environment, _absent(**{"$requires": "persistence"}))

    def test_requires_members_must_be_strings(self, environment):
        with pytest.raises(ConfigError, match=r"\$requires"):
            _resolve(environment, _absent(**{"$requires": ["ok", 3]}))

    def test_an_empty_requires_list_is_not_a_claim(self, environment):
        """It declares nothing, so it defers like an absent marker."""
        assert _resolve(environment, _absent(**{"$requires": []})) == {}

    def test_a_policy_marker_without_a_resource_marker_is_rejected(self, environment):
        """``$resorce`` is the one misspelling the closed set could not catch.

        The guard fires on a block that already contains ``$resource``, so a
        typo in that key produced an ordinary dict that resolved to itself.
        The leftover ``$required`` is what gives it away.
        """
        config = EnvironmentAwareConfig(
            config={
                "store": {
                    "$resorce": "present",
                    "type": "vector_stores",
                    "$required": True,
                },
            },
            environment=environment,
        )
        with pytest.raises(ConfigError, match=r"\$resource"):
            config.resolve_for_build(resolve_env_vars=False)

    def test_a_malformed_marker_inside_a_list_is_rejected(self, environment):
        config = EnvironmentAwareConfig(
            config={"extras": [_absent(**{"$requred": True})]},
            environment=environment,
        )
        with pytest.raises(ConfigError, match=r"\$requred"):
            config.resolve_for_build(resolve_env_vars=False)

    def test_a_malformed_marker_inside_inline_defaults_is_rejected(self, environment):
        config = EnvironmentAwareConfig(
            config={
                "store": {
                    "$resource": "present",
                    "type": "vector_stores",
                    "fallback": _absent(**{"$requred": True}),
                },
            },
            environment=environment,
        )
        with pytest.raises(ConfigError, match=r"\$requred"):
            config.resolve_for_build(resolve_env_vars=False)


class TestEnvironmentSettingIsValidatedWhereItIsWritten:
    """A malformed setting is malformed before any resource goes missing."""

    def test_a_malformed_setting_is_rejected_even_when_every_resource_resolves(self):
        """It was parsed on the missing-resource branch, so it hid until one was.

        The same argument the reference marker already got: a malformed value
        is malformed in every environment, and deferring the parse surfaces
        it first in whichever deployment lacks the resource -- the one least
        equipped to read a message about a setting.
        """
        with pytest.raises(ConfigError, match="strict_resources"):
            EnvironmentConfig(name="prod", settings={"strict_resources": "yes"})

    def test_from_dict_rejects_it_too(self):
        with pytest.raises(ConfigError, match="strict_resources"):
            EnvironmentConfig.from_dict({"name": "prod", "settings": {"strict_resources": 1}})

    def test_a_setting_still_spelled_as_a_template_is_not_parsed_early(self):
        """``strict_resources: ${STRICT}`` is a template, not a value yet.

        Loading with ``substitute_vars=False`` is a supported way to hold a
        config raw. Parsing there would reject every deployment that spells
        the flag as a variable.
        """
        environment = EnvironmentConfig.from_dict(
            {"name": "prod", "settings": {"strict_resources": "${STRICT}"}},
            substitute_vars=False,
        )
        assert environment.get_setting("strict_resources") == "${STRICT}"

    def test_a_valid_setting_survives_construction(self):
        for value in (True, False, "true", "FALSE"):
            EnvironmentConfig(name="prod", settings={"strict_resources": value})

    def test_reference_opt_out_still_beats_the_environment_setting(self):
        """The level pair the matrix left untested: reference over operator."""
        environment = EnvironmentConfig(
            name="prod", settings={"strict_resources": True}, resources={}
        )
        assert _resolve(environment, _absent(**{"$required": False})) == {}


class TestCallContract:
    """A preflight that checked nothing must not report green."""

    def test_strict_without_resource_resolution_is_refused(self, environment):
        """The flag is only read where references are resolved.

        ``resolve_for_build(strict_resources=True)`` is documented as *the*
        startup preflight, so the combination that silently skips every
        check it promises cannot be a no-op.
        """
        config = EnvironmentAwareConfig(config={"store": _absent()}, environment=environment)
        with pytest.raises(ValueError, match="resolve_resources"):
            config.resolve_for_build(resolve_resources=False, strict_resources=True)

    def test_an_instance_policy_with_resources_off_is_not_refused(self, environment):
        """A standing policy is not a per-call assertion about this call."""
        config = EnvironmentAwareConfig(
            config={"store": _absent()}, environment=environment, strict_resources=True
        )
        assert config.resolve_for_build(resolve_resources=False, resolve_env_vars=False) == {
            "store": _absent()
        }


class TestFailureMessages:
    """An operator reads these under time pressure, in a log."""

    def test_the_failure_names_the_config_path(self, environment):
        """Three references to ``default`` are indistinguishable without it."""
        config = EnvironmentAwareConfig(
            config={"bot": {"knowledge_base": {"vector_store": _absent(**{"$required": True})}}},
            environment=environment,
        )
        with pytest.raises(ResourceNotFoundError) as excinfo:
            config.resolve_for_build(resolve_env_vars=False)

        assert "bot.knowledge_base.vector_store" in str(excinfo.value)

    def test_the_message_survives_the_keyerror_base(self, environment):
        """``KeyError.__str__`` is ``repr(args[0])``, which quotes and escapes.

        The type deliberately subclasses both, so callers written against
        either keep working. The cost landed on the message -- the one part
        of this that is read by a person.
        """
        with pytest.raises(ResourceNotFoundError) as excinfo:
            _resolve(environment, _absent(**{"$required": True}))

        message = str(excinfo.value)
        assert message.startswith("Resource 'absent'"), message
        assert "\\'" not in message, "the repr escaping makes the quoted names unreadable"


class TestDirectResourceAccess:
    """``EnvironmentAwareConfig.get_resource`` forwards the policy it is given."""

    def test_required_true_raises_even_with_defaults(self, environment):
        config = EnvironmentAwareConfig(config={}, environment=environment)
        with pytest.raises(ResourceNotFoundError):
            config.get_resource("vector_stores", "absent", {"metric": "cosine"}, required=True)

    def test_required_false_returns_defaults_when_none_were_given(self, environment):
        config = EnvironmentAwareConfig(config={}, environment=environment)
        assert config.get_resource("vector_stores", "absent", required=False) == {}

    def test_required_none_preserves_the_historical_coupling(self, environment):
        config = EnvironmentAwareConfig(config={}, environment=environment)
        with pytest.raises(ResourceNotFoundError):
            config.get_resource("vector_stores", "absent")
        assert config.get_resource("vector_stores", "absent", {"metric": "cosine"}) == {
            "metric": "cosine"
        }

    def test_defaults_fill_gaps_in_a_found_resource(self, environment):
        config = EnvironmentAwareConfig(config={}, environment=environment)
        resolved = config.get_resource(
            "vector_stores", "present", {"backend": "ignored", "metric": "cosine"}
        )
        assert resolved["backend"] == "pgvector"
        assert resolved["metric"] == "cosine"
