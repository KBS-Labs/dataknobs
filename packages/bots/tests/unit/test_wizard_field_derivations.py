"""Tests for wizard field derivation recovery.

When an extraction model captures one field but misses a deterministically
related field (e.g., ``domain_id`` but not ``domain_name``), derivation
rules fill in the missing value without an additional LLM call.

Integration tests exercise the full DynaBot.from_config() → bot.chat() path
via ``BotTestHarness``.  Unit tests verify the derivation engine directly.
"""

from typing import Any

import pytest

from dataknobs_bots.reasoning.wizard_derivations import (
    BUILTIN_TRANSFORMS,
    DerivationRule,
    FieldTransform,
    apply_field_derivations,
    parse_derivation_rules,
    _lower_hyphen,
    _lower_underscore,
    _title_case,
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
