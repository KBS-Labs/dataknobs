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
