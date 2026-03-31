"""Tests for wizard field derivation recovery.

When an extraction model captures one field but misses a deterministically
related field (e.g., ``domain_id`` but not ``domain_name``), derivation
rules fill in the missing value without an additional LLM call.

Integration tests exercise the full DynaBot.from_config() → bot.chat() path
via ``BotTestHarness``.  Unit tests verify the derivation engine directly.
"""

import re
from typing import Any

import pytest

from dataknobs_bots.reasoning.wizard_derivations import (
    BUILTIN_TRANSFORMS,
    PARAMETERIZED_TRANSFORMS,
    DerivationRule,
    FieldTransform,
    _SKIP,
    apply_field_derivations,
    parse_derivation_rules,
    _lower_hyphen,
    _lower_underscore,
    _title_case,
    _equals,
    _not_equals,
    _constant,
    _map_transform,
    _boolean,
    _one_of,
    _contains,
    _first,
    _last,
    _join,
    _split,
    _length,
    _regex_match,
    _regex_extract,
    _regex_replace,
    _execute_expression,
)
from dataknobs_bots.testing import BotTestHarness, WizardConfigBuilder


# ---------------------------------------------------------------------------
# Unit tests: built-in transforms
# ---------------------------------------------------------------------------


class TestBuiltinTransforms:
    """Verify each built-in transform function."""

    def test_title_case_hyphen(self) -> None:
        assert _title_case("chess-champ", {}) == "Chess Champ"

    def test_title_case_underscore(self) -> None:
        assert _title_case("hello_world", {}) == "Hello World"

    def test_title_case_mixed(self) -> None:
        assert _title_case("my-cool_app", {}) == "My Cool App"

    def test_lower_hyphen_basic(self) -> None:
        assert _lower_hyphen("Chess Champ", {}) == "chess-champ"

    def test_lower_hyphen_underscores(self) -> None:
        assert _lower_hyphen("Hello_World", {}) == "hello-world"

    def test_lower_hyphen_mixed_whitespace(self) -> None:
        assert _lower_hyphen("My Cool  App", {}) == "my-cool-app"

    def test_lower_underscore_basic(self) -> None:
        assert _lower_underscore("Chess Champ", {}) == "chess_champ"

    def test_lower_underscore_hyphens(self) -> None:
        assert _lower_underscore("hello-world", {}) == "hello_world"


# ---------------------------------------------------------------------------
# Unit tests: parse_derivation_rules
# ---------------------------------------------------------------------------


class TestParsing:
    """Verify derivation rule parsing from config dicts."""

    def test_parse_basic_rule(self) -> None:
        rules = parse_derivation_rules([
            {"source": "a", "target": "b", "transform": "copy"},
        ])
        assert len(rules) == 1
        assert rules[0].source == "a"
        assert rules[0].target == "b"
        assert rules[0].transform_name == "copy"
        assert rules[0].when == "target_missing"

    def test_parse_with_when_condition(self) -> None:
        rules = parse_derivation_rules([
            {"source": "a", "target": "b", "transform": "copy",
             "when": "always"},
        ])
        assert rules[0].when == "always"

    def test_parse_skips_missing_source(self) -> None:
        rules = parse_derivation_rules([
            {"target": "b", "transform": "copy"},
        ])
        assert len(rules) == 0

    def test_parse_skips_missing_target(self) -> None:
        rules = parse_derivation_rules([
            {"source": "a", "transform": "copy"},
        ])
        assert len(rules) == 0

    def test_parse_unknown_when_defaults(self) -> None:
        rules = parse_derivation_rules([
            {"source": "a", "target": "b", "transform": "copy",
             "when": "bogus"},
        ])
        assert len(rules) == 1
        assert rules[0].when == "target_missing"

    def test_parse_unknown_transform_skips(self) -> None:
        rules = parse_derivation_rules([
            {"source": "a", "target": "b", "transform": "nonexistent"},
        ])
        assert len(rules) == 0

    def test_parse_template_rule(self) -> None:
        rules = parse_derivation_rules([
            {"source": "intent", "target": "desc",
             "transform": "template",
             "template": "A {{ intent }} bot"},
        ])
        assert len(rules) == 1
        assert rules[0].template == "A {{ intent }} bot"

    def test_parse_custom_missing_class_skips(self) -> None:
        rules = parse_derivation_rules([
            {"source": "a", "target": "b", "transform": "custom"},
        ])
        assert len(rules) == 0

    def test_parse_default_transform_is_copy(self) -> None:
        rules = parse_derivation_rules([
            {"source": "a", "target": "b"},
        ])
        assert rules[0].transform_name == "copy"


# ---------------------------------------------------------------------------
# Unit tests: apply_field_derivations
# ---------------------------------------------------------------------------


class TestApplyDerivations:
    """Verify the derivation engine directly."""

    def test_basic_copy(self) -> None:
        rules = [DerivationRule(source="a", target="b",
                                transform_name="copy")]
        data: dict = {"a": "value"}
        derived = apply_field_derivations(rules, data)
        assert derived == {"b"}
        assert data["b"] == "value"

    def test_title_case_derivation(self) -> None:
        rules = [DerivationRule(source="id", target="name",
                                transform_name="title_case")]
        data: dict = {"id": "chess-champ"}
        derived = apply_field_derivations(rules, data)
        assert derived == {"name"}
        assert data["name"] == "Chess Champ"

    def test_no_overwrite_when_target_missing(self) -> None:
        """target_missing guard: skip when target already present."""
        rules = [DerivationRule(source="id", target="name",
                                transform_name="title_case")]
        data: dict = {"id": "chess-champ", "name": "Custom Name"}
        derived = apply_field_derivations(rules, data)
        assert derived == set()
        assert data["name"] == "Custom Name"

    def test_always_guard_overwrites(self) -> None:
        rules = [DerivationRule(source="id", target="name",
                                transform_name="title_case",
                                when="always")]
        data: dict = {"id": "chess-champ", "name": "Old Name"}
        derived = apply_field_derivations(rules, data)
        assert derived == {"name"}
        assert data["name"] == "Chess Champ"

    def test_target_empty_guard(self) -> None:
        rules = [DerivationRule(source="id", target="name",
                                transform_name="title_case",
                                when="target_empty")]
        # Empty string — should derive
        data: dict = {"id": "chess-champ", "name": ""}
        derived = apply_field_derivations(rules, data)
        assert derived == {"name"}
        assert data["name"] == "Chess Champ"

    def test_target_empty_guard_skips_nonempty(self) -> None:
        rules = [DerivationRule(source="id", target="name",
                                transform_name="title_case",
                                when="target_empty")]
        data: dict = {"id": "chess-champ", "name": "Existing"}
        derived = apply_field_derivations(rules, data)
        assert derived == set()

    def test_source_missing_skips(self) -> None:
        rules = [DerivationRule(source="id", target="name",
                                transform_name="title_case")]
        data: dict = {"other": "value"}
        derived = apply_field_derivations(rules, data)
        assert derived == set()
        assert "name" not in data

    def test_bidirectional_first_wins(self) -> None:
        """When both directions are configured, the first matching rule wins."""
        rules = [
            DerivationRule(source="id", target="name",
                           transform_name="title_case"),
            DerivationRule(source="name", target="id",
                           transform_name="lower_hyphen"),
        ]
        data: dict = {"id": "chess-champ"}
        derived = apply_field_derivations(rules, data)
        assert derived == {"name"}
        assert data["name"] == "Chess Champ"
        # Second rule should NOT fire because name is now present
        assert data["id"] == "chess-champ"

    def test_template_derivation(self) -> None:
        rules = [DerivationRule(
            source="intent", target="desc",
            transform_name="template",
            template="A {{ intent }} bot for {{ subject }}",
        )]
        data: dict = {"intent": "tutoring", "subject": "chess"}
        derived = apply_field_derivations(rules, data)
        assert derived == {"desc"}
        assert data["desc"] == "A tutoring bot for chess"

    def test_template_empty_render_skips(self) -> None:
        """Template rendering to empty string should not set the field."""
        rules = [DerivationRule(
            source="intent", target="desc",
            transform_name="template",
            template="{{ missing_var }}",
        )]
        data: dict = {"intent": "tutoring"}
        derived = apply_field_derivations(rules, data)
        assert derived == set()
        assert "desc" not in data

    def test_template_partial_render_skips(self) -> None:
        """Template with some variables missing must not produce a partial value.

        Bug: with jinja2.Undefined, "A {{ intent }} bot for {{ subject }}"
        rendered as "A tutoring bot for" when subject was missing — a
        non-empty garbage string that passed the strip() guard.
        """
        rules = [DerivationRule(
            source="intent", target="desc",
            transform_name="template",
            template="A {{ intent }} bot for {{ subject }}",
        )]
        data: dict = {"intent": "tutoring"}  # subject missing
        derived = apply_field_derivations(rules, data)
        assert derived == set()
        assert "desc" not in data

    def test_custom_field_is_present(self) -> None:
        """Custom field_is_present function is respected."""
        rules = [DerivationRule(source="a", target="b",
                                transform_name="copy")]
        data: dict = {"a": 0}  # 0 is falsy but not None
        # Default: is not None → present
        derived = apply_field_derivations(rules, data)
        assert derived == {"b"}

    def test_custom_field_is_present_strict(self) -> None:
        """Strict field_is_present that rejects falsy values."""
        rules = [DerivationRule(source="a", target="b",
                                transform_name="copy")]
        data: dict = {"a": 0}
        derived = apply_field_derivations(
            rules, data, field_is_present=lambda v: bool(v),
        )
        assert derived == set()

    def test_empty_rules_returns_empty(self) -> None:
        data: dict = {"a": "value"}
        derived = apply_field_derivations([], data)
        assert derived == set()

    def test_multiple_rules_chain(self) -> None:
        """Derivations can chain: A→B, then B→C in same pass."""
        rules = [
            DerivationRule(source="a", target="b",
                           transform_name="copy"),
            DerivationRule(source="b", target="c",
                           transform_name="title_case"),
        ]
        data: dict = {"a": "hello-world"}
        derived = apply_field_derivations(rules, data)
        assert derived == {"b", "c"}
        assert data["b"] == "hello-world"
        assert data["c"] == "Hello World"


# ---------------------------------------------------------------------------
# Unit tests: custom transform
# ---------------------------------------------------------------------------


class _UpperTransform:
    """Test custom transform that uppercases."""

    def transform(self, value: Any, wizard_data: dict) -> Any:
        return str(value).upper()


class TestCustomTransform:
    """Verify custom FieldTransform protocol support."""

    def test_custom_transform_applies(self) -> None:
        custom = _UpperTransform()
        assert isinstance(custom, FieldTransform)
        rules = [DerivationRule(
            source="a", target="b",
            transform_name="custom",
            custom_transform=custom,
        )]
        data: dict = {"a": "hello"}
        derived = apply_field_derivations(rules, data)
        assert derived == {"b"}
        assert data["b"] == "HELLO"


# ---------------------------------------------------------------------------
# Unit tests: conditional/logical transforms
# ---------------------------------------------------------------------------


class TestConditionalTransforms:
    """Verify conditional/logical transform functions."""

    def test_equals_match(self) -> None:
        assert _equals("research_assistant", {}, transform_value="research_assistant") is True

    def test_equals_no_match(self) -> None:
        assert _equals("tutor", {}, transform_value="research_assistant") is False

    def test_equals_type_coercion(self) -> None:
        assert _equals(42, {}, transform_value="42") is True

    def test_not_equals_match(self) -> None:
        assert _not_equals("tutor", {}, transform_value="research_assistant") is True

    def test_not_equals_no_match(self) -> None:
        assert _not_equals("research_assistant", {}, transform_value="research_assistant") is False

    def test_constant_returns_value(self) -> None:
        assert _constant("anything", {}, transform_value="fixed") == "fixed"

    def test_constant_returns_boolean(self) -> None:
        assert _constant("anything", {}, transform_value=True) is True

    def test_constant_returns_none(self) -> None:
        assert _constant("anything", {}, transform_value=None) is None

    def test_map_found(self) -> None:
        assert _map_transform("quiz", {}, transform_map={"quiz": True}) is True

    def test_map_not_found_no_default(self) -> None:
        assert _map_transform("unknown", {}, transform_map={"quiz": True}) is _SKIP

    def test_map_not_found_with_default(self) -> None:
        result = _map_transform(
            "unknown", {},
            transform_map={"quiz": True},
            transform_default="other",
        )
        assert result == "other"

    def test_map_value_is_false(self) -> None:
        result = _map_transform("quiz", {}, transform_map={"quiz": False})
        assert result is False

    def test_map_key_coercion(self) -> None:
        result = _map_transform(42, {}, transform_map={"42": "matched"})
        assert result == "matched"

    def test_boolean_truthy(self) -> None:
        assert _boolean("yes", {}) is True

    def test_boolean_falsy_empty_string(self) -> None:
        assert _boolean("", {}) is False

    def test_boolean_falsy_none(self) -> None:
        assert _boolean(None, {}) is False

    def test_boolean_falsy_zero(self) -> None:
        assert _boolean(0, {}) is False

    def test_one_of_match(self) -> None:
        result = _one_of(
            "research_assistant", {},
            transform_values=["research_assistant", "domain_expert"],
        )
        assert result is True

    def test_one_of_no_match(self) -> None:
        result = _one_of(
            "tutor", {},
            transform_values=["research_assistant", "domain_expert"],
        )
        assert result is False

    def test_contains_match(self) -> None:
        assert _contains("deep research paper", {}, transform_value="research") is True

    def test_contains_no_match(self) -> None:
        assert _contains("quiz creation", {}, transform_value="research") is False

    def test_contains_case_insensitive(self) -> None:
        assert _contains("Research Paper", {}, transform_value="research") is True


# ---------------------------------------------------------------------------
# Unit tests: collection transforms
# ---------------------------------------------------------------------------


class TestCollectionTransforms:
    """Verify collection transform functions."""

    def test_first_list(self) -> None:
        assert _first(["a", "b", "c"], {}) == "a"

    def test_first_empty(self) -> None:
        assert _first([], {}) is _SKIP

    def test_first_string(self) -> None:
        assert _first("abc", {}) == "a"

    def test_last_list(self) -> None:
        assert _last(["a", "b", "c"], {}) == "c"

    def test_last_empty(self) -> None:
        assert _last([], {}) is _SKIP

    def test_join_default_separator(self) -> None:
        assert _join(["a", "b", "c"], {}) == "a, b, c"

    def test_join_custom_separator(self) -> None:
        assert _join(["a", "b"], {}, transform_value=" - ") == "a - b"

    def test_join_non_strings(self) -> None:
        assert _join([1, 2, 3], {}) == "1, 2, 3"

    def test_split_default_separator(self) -> None:
        assert _split("a,b,c", {}) == ["a", "b", "c"]

    def test_split_custom_separator(self) -> None:
        assert _split("a - b - c", {}, transform_value=" - ") == ["a", "b", "c"]

    def test_split_strips_whitespace(self) -> None:
        assert _split("a, b, c", {}) == ["a", "b", "c"]

    def test_length_list(self) -> None:
        assert _length(["a", "b"], {}) == 2

    def test_length_string(self) -> None:
        assert _length("hello", {}) == 5

    def test_length_dict(self) -> None:
        assert _length({"a": 1}, {}) == 1

    def test_length_non_measurable(self) -> None:
        assert _length(42, {}) is _SKIP


# ---------------------------------------------------------------------------
# Unit tests: regex transforms
# ---------------------------------------------------------------------------


class TestRegexTransforms:
    """Verify regex transform functions."""

    def test_regex_match_true(self) -> None:
        pat = re.compile(r"@\w+\.\w+")
        assert _regex_match("user@example.com", {}, compiled_regex=pat) is True

    def test_regex_match_false(self) -> None:
        pat = re.compile(r"@\w+\.\w+")
        assert _regex_match("not-an-email", {}, compiled_regex=pat) is False

    def test_regex_extract_found(self) -> None:
        pat = re.compile(r"@([\w.]+)")
        assert _regex_extract("user@example.com", {}, compiled_regex=pat) == "example.com"

    def test_regex_extract_not_found(self) -> None:
        pat = re.compile(r"@([\w.]+)")
        assert _regex_extract("no-match", {}, compiled_regex=pat) is _SKIP

    def test_regex_replace(self) -> None:
        pat = re.compile(r"\s+")
        result = _regex_replace(
            "hello   world", {},
            compiled_regex=pat,
            transform_replacement=" ",
        )
        assert result == "hello world"


# ---------------------------------------------------------------------------
# Unit tests: expression transform
# ---------------------------------------------------------------------------


class TestExpressionTransform:
    """Verify expression transform via _execute_expression."""

    def test_expression_equality(self) -> None:
        rule = DerivationRule(
            source="intent", target="flag",
            transform_name="expression",
            expression="value == 'ra'",
        )
        assert _execute_expression(rule, "ra", {}) is True

    def test_expression_ternary(self) -> None:
        rule = DerivationRule(
            source="intent", target="max_q",
            transform_name="expression",
            expression="10 if value == 'quiz' else 5",
        )
        assert _execute_expression(rule, "quiz", {}) == 10

    def test_expression_data_access(self) -> None:
        rule = DerivationRule(
            source="x", target="y",
            transform_name="expression",
            expression="int(data.get('count', 0)) * 2",
        )
        assert _execute_expression(rule, "x", {"count": 3}) == 6

    def test_expression_has_helper(self) -> None:
        rule = DerivationRule(
            source="x", target="y",
            transform_name="expression",
            expression="has('name')",
        )
        assert _execute_expression(rule, "x", {"name": "test"}) is True

    def test_expression_list_comprehension(self) -> None:
        rule = DerivationRule(
            source="topics", target="lower_topics",
            transform_name="expression",
            expression="[x.lower() for x in value]",
        )
        assert _execute_expression(rule, ["A", "B"], {}) == ["a", "b"]

    def test_expression_dict_lookup(self) -> None:
        rule = DerivationRule(
            source="diff", target="time",
            transform_name="expression",
            expression="{'easy': 30, 'hard': 120}.get(value, 60)",
        )
        assert _execute_expression(rule, "hard", {}) == 120

    def test_expression_returns_none(self) -> None:
        rule = DerivationRule(
            source="x", target="y",
            transform_name="expression",
            expression="None",
        )
        assert _execute_expression(rule, "x", {}) is None

    def test_expression_error_returns_skip(self) -> None:
        rule = DerivationRule(
            source="x", target="y",
            transform_name="expression",
            expression="1/0",
        )
        assert _execute_expression(rule, "x", {}) is _SKIP


# ---------------------------------------------------------------------------
# Unit tests: expression security
# ---------------------------------------------------------------------------


class TestExpressionSecurity:
    """Verify expression transform blocks dangerous operations."""

    def test_expression_no_import(self) -> None:
        rule = DerivationRule(
            source="x", target="y",
            transform_name="expression",
            expression="__import__('os')",
        )
        assert _execute_expression(rule, "x", {}) is _SKIP

    def test_expression_no_open(self) -> None:
        rule = DerivationRule(
            source="x", target="y",
            transform_name="expression",
            expression="open('/etc/passwd')",
        )
        assert _execute_expression(rule, "x", {}) is _SKIP

    def test_expression_no_exec(self) -> None:
        rule = DerivationRule(
            source="x", target="y",
            transform_name="expression",
            expression="exec('import os')",
        )
        assert _execute_expression(rule, "x", {}) is _SKIP

    def test_expression_no_eval(self) -> None:
        rule = DerivationRule(
            source="x", target="y",
            transform_name="expression",
            expression="eval('1+1')",
        )
        assert _execute_expression(rule, "x", {}) is _SKIP

    def test_expression_no_getattr(self) -> None:
        rule = DerivationRule(
            source="x", target="y",
            transform_name="expression",
            expression="getattr(data, '__class__')",
        )
        assert _execute_expression(rule, "x", {}) is _SKIP

    def test_expression_data_is_snapshot(self) -> None:
        """Data dict mutations in expression don't affect original."""
        original = {"key": "value"}
        rule = DerivationRule(
            source="x", target="y",
            transform_name="expression",
            expression="data.update({'injected': True}) or 'done'",
        )
        _execute_expression(rule, "x", original)
        assert "injected" not in original


# ---------------------------------------------------------------------------
# Unit tests: parsing validation for new transforms
# ---------------------------------------------------------------------------


class TestNewTransformParsing:
    """Verify parsing validation for parameterized transforms."""

    def test_parse_equals_rule(self) -> None:
        rules = parse_derivation_rules([
            {"source": "intent", "target": "flag",
             "transform": "equals", "transform_value": "research"},
        ])
        assert len(rules) == 1
        assert rules[0].transform_name == "equals"
        assert rules[0].transform_value == "research"

    def test_parse_map_rule(self) -> None:
        rules = parse_derivation_rules([
            {"source": "intent", "target": "style",
             "transform": "map",
             "transform_map": {"tutor": "socratic"},
             "transform_default": "structured"},
        ])
        assert len(rules) == 1
        assert rules[0].transform_map == {"tutor": "socratic"}
        assert rules[0].transform_default == "structured"

    def test_parse_one_of_rule(self) -> None:
        rules = parse_derivation_rules([
            {"source": "intent", "target": "flag",
             "transform": "one_of",
             "transform_values": ["a", "b"]},
        ])
        assert len(rules) == 1
        assert rules[0].transform_values == ["a", "b"]

    def test_parse_regex_precompiles(self) -> None:
        rules = parse_derivation_rules([
            {"source": "email", "target": "valid",
             "transform": "regex_match",
             "transform_value": r"\d+"},
        ])
        assert len(rules) == 1
        assert rules[0].compiled_regex is not None
        assert rules[0].compiled_regex.pattern == r"\d+"

    def test_parse_invalid_regex_skips(self) -> None:
        rules = parse_derivation_rules([
            {"source": "email", "target": "valid",
             "transform": "regex_match",
             "transform_value": "[invalid"},
        ])
        assert len(rules) == 0

    def test_parse_missing_transform_value(self) -> None:
        rules = parse_derivation_rules([
            {"source": "a", "target": "b", "transform": "equals"},
        ])
        assert len(rules) == 0

    def test_parse_missing_transform_map(self) -> None:
        rules = parse_derivation_rules([
            {"source": "a", "target": "b", "transform": "map"},
        ])
        assert len(rules) == 0

    def test_parse_missing_expression(self) -> None:
        rules = parse_derivation_rules([
            {"source": "a", "target": "b", "transform": "expression"},
        ])
        assert len(rules) == 0

    def test_parse_expression_rule(self) -> None:
        rules = parse_derivation_rules([
            {"source": "a", "target": "b",
             "transform": "expression",
             "expression": "value == 'x'"},
        ])
        assert len(rules) == 1
        assert rules[0].expression == "value == 'x'"

    def test_parse_regex_replace_missing_replacement(self) -> None:
        rules = parse_derivation_rules([
            {"source": "a", "target": "b",
             "transform": "regex_replace",
             "transform_value": r"\s+"},
        ])
        assert len(rules) == 0

    def test_parse_contains_missing_value(self) -> None:
        rules = parse_derivation_rules([
            {"source": "a", "target": "b", "transform": "contains"},
        ])
        assert len(rules) == 0

    def test_parse_one_of_not_list(self) -> None:
        rules = parse_derivation_rules([
            {"source": "a", "target": "b",
             "transform": "one_of",
             "transform_values": "not-a-list"},
        ])
        assert len(rules) == 0


# ---------------------------------------------------------------------------
# Unit tests: engine integration via apply_field_derivations
# ---------------------------------------------------------------------------


class TestNewTransformIntegration:
    """Verify new transforms through the full derivation engine."""

    def test_equals_derivation_end_to_end(self) -> None:
        rules = parse_derivation_rules([
            {"source": "intent", "target": "kb_enabled",
             "transform": "equals",
             "transform_value": "research_assistant"},
        ])
        data: dict[str, Any] = {"intent": "research_assistant"}
        derived = apply_field_derivations(rules, data)
        assert derived == {"kb_enabled"}
        assert data["kb_enabled"] is True

    def test_map_derivation_end_to_end(self) -> None:
        rules = parse_derivation_rules([
            {"source": "intent", "target": "style",
             "transform": "map",
             "transform_map": {
                 "research_assistant": "conversational",
                 "tutor": "socratic",
             },
             "transform_default": "structured"},
        ])
        data: dict[str, Any] = {"intent": "tutor"}
        derived = apply_field_derivations(rules, data)
        assert derived == {"style"}
        assert data["style"] == "socratic"

    def test_map_no_default_unmatched_key_skips(self) -> None:
        """map with no transform_default skips when key not found."""
        rules = parse_derivation_rules([
            {"source": "intent", "target": "style",
             "transform": "map",
             "transform_map": {"tutor": "socratic"}},
            # No transform_default — unmatched key should skip
        ])
        data: dict[str, Any] = {"intent": "research_assistant"}
        derived = apply_field_derivations(rules, data)
        assert derived == set()
        assert "style" not in data

    def test_map_explicit_null_default_stores_none(self) -> None:
        """map with transform_default: null stores None explicitly."""
        rules = parse_derivation_rules([
            {"source": "intent", "target": "style",
             "transform": "map",
             "transform_map": {"tutor": "socratic"},
             "transform_default": None},
        ])
        data: dict[str, Any] = {"intent": "research_assistant"}
        derived = apply_field_derivations(rules, data)
        assert derived == {"style"}
        assert data["style"] is None

    def test_first_empty_list_skips(self) -> None:
        """first on empty list skips instead of storing None."""
        rules = parse_derivation_rules([
            {"source": "topics", "target": "primary",
             "transform": "first"},
        ])
        data: dict[str, Any] = {"topics": []}
        derived = apply_field_derivations(rules, data)
        assert derived == set()
        assert "primary" not in data

    def test_regex_extract_no_match_skips(self) -> None:
        """regex_extract with no match skips instead of storing None."""
        rules = parse_derivation_rules([
            {"source": "email", "target": "domain",
             "transform": "regex_extract",
             "transform_value": r"@([\w.]+)"},
        ])
        data: dict[str, Any] = {"email": "not-an-email"}
        derived = apply_field_derivations(rules, data)
        assert derived == set()
        assert "domain" not in data

    def test_expression_derivation_end_to_end(self) -> None:
        rules = parse_derivation_rules([
            {"source": "intent", "target": "max_q",
             "transform": "expression",
             "expression": "10 if value == 'quiz_maker' else 5"},
        ])
        data: dict[str, Any] = {"intent": "quiz_maker"}
        derived = apply_field_derivations(rules, data)
        assert derived == {"max_q"}
        assert data["max_q"] == 10

    def test_conditional_with_guard(self) -> None:
        """when: target_missing respects existing target value."""
        rules = parse_derivation_rules([
            {"source": "intent", "target": "kb_enabled",
             "transform": "equals",
             "transform_value": "research_assistant",
             "when": "target_missing"},
        ])
        data: dict[str, Any] = {"intent": "research_assistant", "kb_enabled": False}
        derived = apply_field_derivations(rules, data)
        assert derived == set()
        assert data["kb_enabled"] is False

    def test_boolean_derivation_return_type(self) -> None:
        rules = parse_derivation_rules([
            {"source": "path", "target": "configured",
             "transform": "boolean"},
        ])
        data: dict[str, Any] = {"path": "/some/path"}
        apply_field_derivations(rules, data)
        assert data["configured"] is True
        assert isinstance(data["configured"], bool)

    def test_constant_none_sets_field(self) -> None:
        """constant with transform_value: null sets the field to None."""
        rules = parse_derivation_rules([
            {"source": "intent", "target": "cleared",
             "transform": "constant",
             "transform_value": None},
        ])
        data: dict[str, Any] = {"intent": "anything"}
        derived = apply_field_derivations(rules, data)
        assert derived == {"cleared"}
        assert data["cleared"] is None

    def test_expression_none_sets_field(self) -> None:
        """expression evaluating to None sets the field to None."""
        rules = parse_derivation_rules([
            {"source": "intent", "target": "cleared",
             "transform": "expression",
             "expression": "None"},
        ])
        data: dict[str, Any] = {"intent": "anything"}
        derived = apply_field_derivations(rules, data)
        assert derived == {"cleared"}
        assert data["cleared"] is None

    def test_mixed_transforms_in_pipeline(self) -> None:
        rules = parse_derivation_rules([
            {"source": "intent", "target": "kb_enabled",
             "transform": "equals",
             "transform_value": "research_assistant"},
            {"source": "intent", "target": "style",
             "transform": "map",
             "transform_map": {"research_assistant": "conversational"},
             "transform_default": "structured"},
            {"source": "domain_id", "target": "domain_name",
             "transform": "title_case"},
        ])
        data: dict[str, Any] = {
            "intent": "research_assistant",
            "domain_id": "chess-champ",
        }
        derived = apply_field_derivations(rules, data)
        assert derived == {"kb_enabled", "style", "domain_name"}
        assert data["kb_enabled"] is True
        assert data["style"] == "conversational"
        assert data["domain_name"] == "Chess Champ"


# ---------------------------------------------------------------------------
# Integration tests: BotTestHarness
# ---------------------------------------------------------------------------


def _wizard_config_with_derivations(
    derivations: list[dict],
    **extra_settings: Any,
) -> dict:
    """Build a wizard config with derivation rules."""
    return (
        WizardConfigBuilder("derivation-test")
        .stage(
            "gather",
            is_start=True,
            prompt="Tell me your domain ID and name.",
        )
        .field("domain_id", field_type="string", required=True)
        .field("domain_name", field_type="string", required=True)
        .transition(
            "done",
            "data.get('domain_id') and data.get('domain_name')",
        )
        .stage("done", is_end=True, prompt="All done!")
        .settings(derivations=derivations, **extra_settings)
        .build()
    )


class TestFieldDerivationIntegration:
    """Integration tests exercising derivation through bot.chat()."""

    @pytest.mark.asyncio
    async def test_title_case_derivation_enables_auto_advance(self) -> None:
        """domain_id extracted → domain_name derived → auto-advance fires."""
        config = _wizard_config_with_derivations(
            derivations=[
                {"source": "domain_id", "target": "domain_name",
                 "transform": "title_case"},
            ],
            auto_advance_filled_stages=True,
        )

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["Got it!"],
            extraction_results=[
                [{"domain_id": "chess-champ"}],  # domain_name omitted
            ],
        ) as harness:
            await harness.chat("ID is chess-champ")
            assert harness.wizard_data["domain_id"] == "chess-champ"
            assert harness.wizard_data["domain_name"] == "Chess Champ"
            assert harness.wizard_stage == "done"

    @pytest.mark.asyncio
    async def test_lower_hyphen_derivation(self) -> None:
        """domain_name extracted → domain_id derived via lower_hyphen."""
        config = _wizard_config_with_derivations(
            derivations=[
                {"source": "domain_name", "target": "domain_id",
                 "transform": "lower_hyphen"},
            ],
            auto_advance_filled_stages=True,
        )

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["Got it!"],
            extraction_results=[
                [{"domain_name": "Chess Champ"}],  # domain_id omitted
            ],
        ) as harness:
            await harness.chat("Call it Chess Champ")
            assert harness.wizard_data["domain_name"] == "Chess Champ"
            assert harness.wizard_data["domain_id"] == "chess-champ"
            assert harness.wizard_stage == "done"

    @pytest.mark.asyncio
    async def test_derivation_does_not_overwrite_extracted(self) -> None:
        """When both fields are extracted, derivation does not overwrite."""
        config = _wizard_config_with_derivations(
            derivations=[
                {"source": "domain_id", "target": "domain_name",
                 "transform": "title_case"},
            ],
            auto_advance_filled_stages=True,
        )

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["Got it!"],
            extraction_results=[
                [{"domain_id": "chess-champ",
                  "domain_name": "My Custom Name"}],
            ],
        ) as harness:
            await harness.chat("ID chess-champ, name My Custom Name")
            assert harness.wizard_data["domain_name"] == "My Custom Name"
            assert harness.wizard_stage == "done"

    @pytest.mark.asyncio
    async def test_bidirectional_derivation(self) -> None:
        """Both directions configured; only the needed one fires."""
        config = _wizard_config_with_derivations(
            derivations=[
                {"source": "domain_id", "target": "domain_name",
                 "transform": "title_case"},
                {"source": "domain_name", "target": "domain_id",
                 "transform": "lower_hyphen"},
            ],
            auto_advance_filled_stages=True,
        )

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["Got it!"],
            extraction_results=[
                [{"domain_id": "my-bot"}],
            ],
        ) as harness:
            await harness.chat("ID is my-bot")
            assert harness.wizard_data["domain_id"] == "my-bot"
            assert harness.wizard_data["domain_name"] == "My Bot"
            assert harness.wizard_stage == "done"

    @pytest.mark.asyncio
    async def test_no_derivation_when_source_missing(self) -> None:
        """Derivation does not fire when source field is absent."""
        config = _wizard_config_with_derivations(
            derivations=[
                {"source": "domain_id", "target": "domain_name",
                 "transform": "title_case"},
            ],
        )

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["Tell me the ID too", "Got it!"],
            extraction_results=[
                # Only domain_name extracted — source for derivation
                # (domain_id) is missing
                [{"domain_name": "Chess Champ"}],
                [{"domain_id": "chess-champ"}],
            ],
        ) as harness:
            await harness.chat("Name is Chess Champ")
            # domain_id derivation rule has source=domain_id which
            # is not yet present, so no derivation
            assert harness.wizard_data.get("domain_name") == "Chess Champ"
            assert "domain_id" not in harness.wizard_data or not harness.wizard_data.get("domain_id")

    @pytest.mark.asyncio
    async def test_per_stage_derivation_disabled(self) -> None:
        """derivation_enabled: false on a stage suppresses derivation."""
        config = (
            WizardConfigBuilder("derivation-disable-test")
            .stage(
                "gather",
                is_start=True,
                prompt="Tell me your domain ID.",
                extraction_scope="current_message",
                derivation_enabled=False,
            )
            .field("domain_id", field_type="string", required=True)
            .field("domain_name", field_type="string", required=True)
            .transition(
                "done",
                "data.get('domain_id') and data.get('domain_name')",
            )
            .stage("done", is_end=True, prompt="All done!")
            .settings(derivations=[
                {"source": "domain_id", "target": "domain_name",
                 "transform": "title_case"},
            ])
            .build()
        )

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["Need the name too", "Got it!"],
            extraction_results=[
                [{"domain_id": "chess-champ"}],
                [{"domain_id": "chess-champ", "domain_name": "Chess Champ"}],
            ],
        ) as harness:
            await harness.chat("ID is chess-champ")
            # Derivation disabled — domain_name NOT derived
            assert harness.wizard_data.get("domain_id") == "chess-champ"
            assert not harness.wizard_data.get("domain_name")

    @pytest.mark.asyncio
    async def test_template_derivation_integration(self) -> None:
        """Template derivation using multiple wizard data fields."""
        config = (
            WizardConfigBuilder("template-derivation-test")
            .stage(
                "gather",
                is_start=True,
                prompt="Tell me intent and subject.",
            )
            .field("intent", field_type="string", required=True)
            .field("subject", field_type="string", required=True)
            .field("description", field_type="string", required=True)
            .transition(
                "done",
                "data.get('intent') and data.get('subject') and data.get('description')",
            )
            .stage("done", is_end=True, prompt="All done!")
            .settings(
                derivations=[
                    {"source": "intent", "target": "description",
                     "transform": "template",
                     "template": "A {{ intent }} bot for {{ subject }}"},
                ],
                auto_advance_filled_stages=True,
            )
            .build()
        )

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["Got it!"],
            extraction_results=[
                [{"intent": "tutoring", "subject": "chess"}],
            ],
        ) as harness:
            await harness.chat("I want a tutoring bot for chess")
            assert harness.wizard_data["intent"] == "tutoring"
            assert harness.wizard_data["subject"] == "chess"
            assert harness.wizard_data["description"] == "A tutoring bot for chess"
            assert harness.wizard_stage == "done"

    @pytest.mark.asyncio
    async def test_wizard_conditional_derivation(self) -> None:
        """Wizard extracts intent, equals derivation auto-sets kb_enabled."""
        config = (
            WizardConfigBuilder("conditional-derivation-test")
            .stage(
                "gather",
                is_start=True,
                prompt="What kind of bot?",
            )
            .field("intent", field_type="string", required=True)
            .field("kb_enabled", field_type="boolean", required=True)
            .transition(
                "done",
                "data.get('intent') and data.get('kb_enabled') is not None",
            )
            .stage("done", is_end=True, prompt="All done!")
            .settings(
                derivations=[
                    {"source": "intent", "target": "kb_enabled",
                     "transform": "equals",
                     "transform_value": "research_assistant"},
                ],
                auto_advance_filled_stages=True,
            )
            .build()
        )

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["Got it!"],
            extraction_results=[
                [{"intent": "research_assistant"}],
            ],
        ) as harness:
            await harness.chat("I want a research assistant")
            assert harness.wizard_data["intent"] == "research_assistant"
            assert harness.wizard_data["kb_enabled"] is True
            assert harness.wizard_stage == "done"

    @pytest.mark.asyncio
    async def test_wizard_expression_derivation(self) -> None:
        """Expression derivation produces typed result in wizard data."""
        config = (
            WizardConfigBuilder("expression-derivation-test")
            .stage(
                "gather",
                is_start=True,
                prompt="What kind of bot?",
            )
            .field("intent", field_type="string", required=True)
            .field("max_questions", field_type="integer", required=True)
            .transition(
                "done",
                "data.get('intent') and data.get('max_questions') is not None",
            )
            .stage("done", is_end=True, prompt="All done!")
            .settings(
                derivations=[
                    {"source": "intent", "target": "max_questions",
                     "transform": "expression",
                     "expression": "10 if value == 'quiz_maker' else 5"},
                ],
                auto_advance_filled_stages=True,
            )
            .build()
        )

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["Got it!"],
            extraction_results=[
                [{"intent": "quiz_maker"}],
            ],
        ) as harness:
            await harness.chat("I want a quiz maker")
            assert harness.wizard_data["intent"] == "quiz_maker"
            assert harness.wizard_data["max_questions"] == 10
            assert harness.wizard_stage == "done"


# ---------------------------------------------------------------------------
# Post-extraction derivation pass tests
# ---------------------------------------------------------------------------


def _wizard_config_optional_derivation(
    derivations: list[dict],
    **extra_settings: Any,
) -> dict:
    """Wizard where all required fields satisfy extraction.

    ``kb_enabled`` is optional, derived from required ``intent``.
    When all required fields are extracted, the recovery pipeline
    is skipped — derivations must run unconditionally to fill
    ``kb_enabled``.
    """
    return (
        WizardConfigBuilder("post-extract-derivation-test")
        .stage(
            "gather",
            is_start=True,
            prompt="Tell me your intent.",
        )
        .field("intent", field_type="string", required=True)
        .field("kb_enabled", field_type="boolean", required=False)
        .transition(
            "done",
            "data.get('intent')",
        )
        .stage("done", is_end=True, prompt="All done!")
        .settings(derivations=derivations, **extra_settings)
        .build()
    )


class TestPostExtractionDerivation:
    """Derivations run unconditionally after merge + defaults.

    Prior to this feature, derivations only ran inside the recovery
    pipeline, which is gated on missing required fields.  These tests
    verify that optional fields are derived even when all required
    fields are present.
    """

    @pytest.mark.asyncio
    async def test_optional_field_derived_from_required(self) -> None:
        """Optional kb_enabled derived from required intent via equals."""
        config = _wizard_config_optional_derivation(
            derivations=[
                {
                    "source": "intent",
                    "target": "kb_enabled",
                    "transform": "equals",
                    "transform_value": "research_assistant",
                },
            ],
        )

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["Got it!"],
            extraction_results=[
                [{"intent": "research_assistant"}],
            ],
        ) as harness:
            await harness.chat("I want a research assistant")
            assert harness.wizard_data["intent"] == "research_assistant"
            assert harness.wizard_data["kb_enabled"] is True
            assert harness.wizard_stage == "done"

    @pytest.mark.asyncio
    async def test_optional_derivation_false_when_no_match(self) -> None:
        """equals returns False when source doesn't match."""
        config = _wizard_config_optional_derivation(
            derivations=[
                {
                    "source": "intent",
                    "target": "kb_enabled",
                    "transform": "equals",
                    "transform_value": "research_assistant",
                },
            ],
        )

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["Got it!"],
            extraction_results=[
                [{"intent": "quiz_maker"}],
            ],
        ) as harness:
            await harness.chat("I want a quiz maker")
            assert harness.wizard_data["intent"] == "quiz_maker"
            assert harness.wizard_data["kb_enabled"] is False
            assert harness.wizard_stage == "done"

    @pytest.mark.asyncio
    async def test_derivation_does_not_overwrite_extracted_optional(
        self,
    ) -> None:
        """Extraction-provided optional value not overwritten by derivation."""
        config = _wizard_config_optional_derivation(
            derivations=[
                {
                    "source": "intent",
                    "target": "kb_enabled",
                    "transform": "equals",
                    "transform_value": "research_assistant",
                },
            ],
        )

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["Got it!"],
            extraction_results=[
                # User explicitly set kb_enabled=False despite matching intent
                [{"intent": "research_assistant", "kb_enabled": False}],
            ],
        ) as harness:
            await harness.chat("research assistant but no KB")
            assert harness.wizard_data["intent"] == "research_assistant"
            # target_missing guard: extraction value preserved
            assert harness.wizard_data["kb_enabled"] is False

    @pytest.mark.asyncio
    async def test_derivation_suppressed_by_stage_flag(self) -> None:
        """derivation_enabled: false suppresses post-extraction pass."""
        config = (
            WizardConfigBuilder("suppressed-derivation-test")
            .stage(
                "gather",
                is_start=True,
                prompt="Tell me your intent.",
                derivation_enabled=False,
            )
            .field("intent", field_type="string", required=True)
            .field("kb_enabled", field_type="boolean", required=False)
            .transition("done", "data.get('intent')")
            .stage("done", is_end=True, prompt="All done!")
            .settings(
                derivations=[
                    {
                        "source": "intent",
                        "target": "kb_enabled",
                        "transform": "equals",
                        "transform_value": "research_assistant",
                    },
                ],
            )
            .build()
        )

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["Got it!"],
            extraction_results=[
                [{"intent": "research_assistant"}],
            ],
        ) as harness:
            await harness.chat("I want a research assistant")
            assert harness.wizard_data["intent"] == "research_assistant"
            # Derivation suppressed — kb_enabled stays None
            assert harness.wizard_data.get("kb_enabled") is None

    @pytest.mark.asyncio
    async def test_recovery_derivation_noop_after_post_extraction(
        self,
    ) -> None:
        """Recovery pipeline derivation is a no-op when post-extraction pass already filled.

        Scenario: required field ``name`` is missing, recovery pipeline
        runs.  But the derivation target (``kb_enabled``) was already
        filled by the post-extraction pass, so the recovery derivation
        step adds nothing new for that field.
        """
        config = (
            WizardConfigBuilder("recovery-noop-test")
            .stage(
                "gather",
                is_start=True,
                prompt="Tell me everything.",
            )
            .field("intent", field_type="string", required=True)
            .field("name", field_type="string", required=True)
            .field("kb_enabled", field_type="boolean", required=False)
            .transition(
                "done",
                "data.get('intent') and data.get('name')",
            )
            .stage("done", is_end=True, prompt="All done!")
            .settings(
                derivations=[
                    {
                        "source": "intent",
                        "target": "kb_enabled",
                        "transform": "equals",
                        "transform_value": "research_assistant",
                    },
                ],
                recovery_pipeline=["derivation"],
            )
            .build()
        )

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["What's your name?", "Got it!"],
            extraction_results=[
                # Turn 1: intent extracted, name missing → recovery runs
                [{"intent": "research_assistant"}],
                # Turn 2: name extracted
                [{"name": "Alice"}],
            ],
        ) as harness:
            await harness.chat("I want a research assistant")
            # After turn 1: post-extraction derivation fills kb_enabled,
            # recovery pipeline runs for missing 'name' but derivation
            # step is no-op for kb_enabled (already filled)
            assert harness.wizard_data["intent"] == "research_assistant"
            assert harness.wizard_data["kb_enabled"] is True
            assert harness.wizard_data.get("name") is None

            await harness.chat("My name is Alice")
            assert harness.wizard_data["name"] == "Alice"
            assert harness.wizard_stage == "done"
