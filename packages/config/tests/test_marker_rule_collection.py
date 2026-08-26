"""The marker rule, collected rather than raised.

The rule that says which ``$``-prefixed keys a config may carry lived only
behind two raising functions, private to the resolver. A caller that reports a
verdict rather than building -- a validator, an editor -- could reach the
marker *set* but not the *rule*, so the one that tried transcribed a clause of
it and the transcription covered one rule of two at one depth of N.

These tests pin the collecting entry point that makes the rule callable, and
pin that the two raising wrappers still raise exactly what they raised.
"""

from __future__ import annotations

import pytest
import yaml

from dataknobs_config import MarkerViolation, collect_marker_violations
from dataknobs_config.environment_aware import (
    _validate_orphaned_markers,
    _validate_reference_markers,
    resolve_resource_references,
)
from dataknobs_config.environment_config import EnvironmentConfig
from dataknobs_config.exceptions import ConfigError


class TestCollectingTheRule:
    """One rule, applied at every depth, reporting every offender."""

    def test_a_top_level_reference_marker_is_collected(self) -> None:
        violations = collect_marker_violations(
            {"$resource": "default", "type": "llm_providers", "$requred": True}
        )

        assert len(violations) == 1
        assert violations[0].path == ""
        assert "$requred" in violations[0].message

    def test_a_nested_reference_marker_is_collected(self) -> None:
        """The depth-1 transcription in ``bots`` stopped exactly here."""
        violations = collect_marker_violations(
            {"bot": {"llm": {"$resource": "default", "type": "llm_providers", "$requred": True}}}
        )

        assert [v.path for v in violations] == ["bot.llm"]

    def test_an_orphaned_policy_marker_is_collected(self) -> None:
        """A misspelled ``$resource`` leaves its qualifiers behind."""
        violations = collect_marker_violations(
            {"bot": {"llm": {"$resorce": "default", "$required": True}}}
        )

        assert [v.path for v in violations] == ["bot.llm"]
        assert "$required" in violations[0].message

    def test_a_marker_inside_a_list_is_collected_with_its_index(self) -> None:
        violations = collect_marker_violations(
            {"tools": [{"name": "ok"}, {"$resource": "t", "$requres": ["streaming"]}]}
        )

        assert [v.path for v in violations] == ["tools[1]"]

    def test_every_violation_is_collected_not_just_the_first(self) -> None:
        """Collecting is the whole difference from the raising wrappers."""
        violations = collect_marker_violations(
            {
                "a": {"$resource": "one", "$requred": True},
                "b": {"$resorce": "two", "$requires": ["streaming"]},
            }
        )

        assert sorted(v.path for v in violations) == ["a", "b"]

    def test_a_clean_config_collects_nothing(self) -> None:
        """The anti-vacuity half: a rule that always fires reports nothing."""
        assert (
            collect_marker_violations(
                {
                    "bot": {
                        "llm": {
                            "$resource": "default",
                            "type": "llm_providers",
                            "$required": True,
                            "$requires": ["streaming"],
                        },
                        "prompt": {"$custom": "not a policy marker, left alone"},
                    }
                }
            )
            == []
        )

    def test_a_violation_carries_the_path_it_was_found_at(self) -> None:
        violations = collect_marker_violations(
            {"llm": {"$resource": "d", "$requred": True}}, path="bot"
        )

        assert violations == [
            MarkerViolation(path="bot.llm", message=violations[0].message),
        ]


class TestTheDivergenceFromTheResolver:
    """The collector descends where the resolver deliberately does not.

    ``_splice_found_resource`` expands an inline default only once it is known
    to survive, so a default the environment overrides is never walked and a
    malformed reference inside it is never checked. That is correct for a
    build -- expanding a value nothing will read is work for nothing -- and
    wrong for a validator, whose subject is the authored tree rather than one
    environment's view of it.
    """

    @pytest.fixture
    def env(self) -> EnvironmentConfig:
        return EnvironmentConfig.from_dict(
            {
                "name": "test",
                "resources": {"databases": {"main": {"backend": "postgres", "pool": {}}}},
            }
        )

    def test_the_resolver_does_not_reach_an_overridden_default(self, env) -> None:
        """Red-half of the pair: this is the gap, and it is intended."""
        config = {
            "db": {
                "$resource": "main",
                "type": "databases",
                # `pool` is supplied by the resource above, so this default is
                # discarded -- and never walked, so its marker is never seen.
                "pool": {"$resource": "other", "$requred": True},
            }
        }

        resolved = resolve_resource_references(config, env)

        assert resolved["db"]["pool"] == {}

    def test_the_collector_reaches_it(self, env) -> None:
        violations = collect_marker_violations(
            {
                "db": {
                    "$resource": "main",
                    "type": "databases",
                    "pool": {"$resource": "other", "$requred": True},
                }
            }
        )

        assert [v.path for v in violations] == ["db.pool"]


class TestTheRaisingWrappersAreUnchanged:
    """The refactor moves the rule; it must not move the behaviour."""

    def test_reference_markers_still_raise_config_error(self) -> None:
        with pytest.raises(ConfigError) as excinfo:
            _validate_reference_markers({"$resource": "default", "$requred": True}, path="bot.llm")

        message = str(excinfo.value)
        assert "$requred" in message
        assert "bot.llm" in message
        assert "inline default" in message

    def test_reference_markers_report_every_offender_in_sorted_order(self) -> None:
        with pytest.raises(ConfigError) as excinfo:
            _validate_reference_markers({"$resource": "d", "$zeta": 1, "$alpha": 2}, path="bot.llm")

        assert "['$alpha', '$zeta']" in str(excinfo.value)

    def test_reference_markers_accept_every_marker(self) -> None:
        _validate_reference_markers(
            {"$resource": "d", "type": "t", "$required": True, "$requires": ["s"]},
            path="bot.llm",
        )

    def test_orphaned_markers_still_raise_config_error(self) -> None:
        with pytest.raises(ConfigError) as excinfo:
            _validate_orphaned_markers({"$required": True, "backend": "memory"}, path="bot.db")

        message = str(excinfo.value)
        assert "$required" in message
        assert "bot.db" in message
        assert "misspelled" in message

    def test_orphaned_markers_leave_a_non_policy_dollar_key_alone(self) -> None:
        _validate_orphaned_markers({"$custom": "passed through"}, path="bot.db")

    def test_the_root_path_is_named_in_a_message(self) -> None:
        """``<root>`` rather than ``''`` -- a sentence needs a noun."""
        with pytest.raises(ConfigError) as excinfo:
            _validate_orphaned_markers({"$required": True}, path="")

        assert "<root>" in str(excinfo.value)


class TestAConfigThatContainsItself:
    """A structural cycle is reported, not followed round.

    YAML anchors build one: ``a: &x`` with ``b: *x`` under it produces a dict
    that contains itself, and ``yaml.safe_load`` does it without complaint.
    Both readers of the format descended it until the stack ran out.

    `c7b6a24e` settled the policy for the other cycle a config can carry -- a
    resource that reaches itself -- and settled it for every entry point at
    once, because guarding one walk *"left the build to exhaust the stack on
    the same input, and left the survey certifying that input as sound."*
    That is this cycle exactly, in the dimension that commit did not cover.
    """

    @pytest.fixture
    def env(self) -> EnvironmentConfig:
        return EnvironmentConfig.from_dict({"name": "test", "resources": {}})

    @pytest.fixture
    def cyclic_dict(self) -> dict:
        return yaml.safe_load("bot:\n  llm: &x\n    nested: *x\n")

    @pytest.fixture
    def cyclic_list(self) -> dict:
        return yaml.safe_load("tools: &t\n  - *t\n")

    def test_the_collector_reports_a_cyclic_dict(self, cyclic_dict) -> None:
        with pytest.raises(ConfigError) as excinfo:
            collect_marker_violations(cyclic_dict)

        message = str(excinfo.value)
        assert "cycle" in message
        assert "bot.llm" in message

    def test_the_resolver_reports_a_cyclic_dict(self, cyclic_dict, env) -> None:
        with pytest.raises(ConfigError) as excinfo:
            resolve_resource_references(cyclic_dict, env)

        assert "cycle" in str(excinfo.value)

    def test_the_collector_reports_a_cyclic_list(self, cyclic_list) -> None:
        with pytest.raises(ConfigError) as excinfo:
            collect_marker_violations(cyclic_list)

        assert "cycle" in str(excinfo.value)

    def test_the_resolver_reports_a_cyclic_list(self, cyclic_list, env) -> None:
        with pytest.raises(ConfigError) as excinfo:
            resolve_resource_references(cyclic_list, env)

        assert "cycle" in str(excinfo.value)

    def test_the_message_names_both_ends_of_the_cycle(self, cyclic_dict) -> None:
        """Where it closed and where that block was entered.

        One path alone locates half of a cycle, and the half it locates is
        the one the reader can already see.
        """
        with pytest.raises(ConfigError) as excinfo:
            collect_marker_violations(cyclic_dict)

        message = str(excinfo.value)
        # Quoted, so the entered path is not merely a prefix of the closing one.
        assert "'bot.llm.nested'" in message
        assert "'bot.llm'" in message

    def test_a_shared_anchor_that_is_not_a_cycle_still_resolves(self, env) -> None:
        """The anti-vacuity half, and the one a visited-set would break.

        An anchor reused for its ordinary purpose -- not repeating a block --
        puts the *same object* at two paths without either containing the
        other. A guard that refused to re-enter any object it had seen would
        reject this, so the guard tracks what the descent is currently inside
        rather than everything it has ever been.
        """
        shared = yaml.safe_load("defaults: &d\n  timeout: 5\na: *d\nb: *d\n")
        assert shared["a"] is shared["b"]

        assert collect_marker_violations(shared) == []
        assert resolve_resource_references(shared, env) == {
            "defaults": {"timeout": 5},
            "a": {"timeout": 5},
            "b": {"timeout": 5},
        }

    def test_a_sibling_repeated_at_one_level_is_not_a_cycle(self, env) -> None:
        """The pop must happen, or the second sibling reads as a repeat."""
        block = {"timeout": 5}
        assert collect_marker_violations({"a": block, "b": block}) == []


class TestTheResourceCycleStillReportsItself:
    """The other cycle a config can carry, unchanged.

    Two different things reach themselves in a config tree and they are
    detected differently: a resource by its identity, a block by object
    identity. Sharing the descent's bookkeeping must not merge the verdicts.
    """

    def test_a_resource_that_reaches_itself_names_the_chain(self) -> None:
        env = EnvironmentConfig.from_dict(
            {
                "name": "test",
                "resources": {"databases": {"a": {"$resource": "a", "type": "databases"}}},
            }
        )

        with pytest.raises(ConfigError) as excinfo:
            resolve_resource_references({"db": {"$resource": "a", "type": "databases"}}, env)

        message = str(excinfo.value)
        assert "Resource reference cycle" in message
        assert "databases/a" in message

    def test_a_second_reference_to_one_resource_is_not_a_cycle(self) -> None:
        """The guard is popped at the splice, so two references are two."""
        env = EnvironmentConfig.from_dict(
            {"name": "test", "resources": {"databases": {"main": {"backend": "postgres"}}}}
        )

        resolved = resolve_resource_references(
            {
                "primary": {"$resource": "main", "type": "databases"},
                "replica": {"$resource": "main", "type": "databases"},
            },
            env,
        )

        assert resolved["primary"] == {"backend": "postgres"}
        assert resolved["replica"] == {"backend": "postgres"}
