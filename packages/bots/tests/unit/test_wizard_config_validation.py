"""Tests for wizard configuration validation warnings.

Tests that the WizardConfigLoader and WizardConfigBuilder properly
warn about common config issues:
- Unrecognized stage fields (e.g. ``extracts``)
- Missing schema + response_template on non-end stages
- English-language conditions instead of Python
- Python str.format() syntax instead of Jinja2
"""

from __future__ import annotations

import logging

import pytest

from dataknobs_bots.config.wizard_builder import WizardConfigBuilder
from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_config(**overrides: object) -> dict:
    """Build a minimal valid wizard config with optional stage overrides."""
    stage = {
        "name": "welcome",
        "is_start": True,
        "is_end": True,
        "prompt": "Hello!",
    }
    stage.update(overrides)  # type: ignore[arg-type]
    return {
        "name": "test",
        "stages": [stage],
    }


def _two_stage_config(
    first_overrides: dict | None = None,
    second_overrides: dict | None = None,
) -> dict:
    """Build a two-stage wizard config for transition tests."""
    first = {
        "name": "step1",
        "is_start": True,
        "prompt": "First step",
        "schema": {
            "type": "object",
            "properties": {"val": {"type": "string"}},
        },
        "transitions": [
            {"target": "step2", "condition": "data.get('val')"},
        ],
    }
    second = {
        "name": "step2",
        "is_end": True,
        "prompt": "Done!",
    }
    if first_overrides:
        first.update(first_overrides)
    if second_overrides:
        second.update(second_overrides)
    return {
        "name": "test",
        "stages": [first, second],
    }


# ---------------------------------------------------------------------------
# WizardConfigLoader._validate_config tests
# ---------------------------------------------------------------------------


class TestLoaderValidation:
    """Tests for WizardConfigLoader config validation warnings."""

    def test_unrecognized_field_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        """``extracts`` field generates a warning."""
        config = _minimal_config(extracts=[{"field": "name"}])
        loader = WizardConfigLoader()
        with caplog.at_level(logging.WARNING):
            loader.load_from_dict(config)
        assert any("unrecognized field 'extracts'" in r.message for r in caplog.records)

    def test_missing_schema_and_template_warns(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Non-end stage with no schema and no response_template warns."""
        config = _two_stage_config(
            first_overrides={"schema": None},
        )
        # Remove schema from first stage
        del config["stages"][0]["schema"]
        loader = WizardConfigLoader()
        with caplog.at_level(logging.WARNING):
            loader.load_from_dict(config)
        assert any(
            "no 'schema' and no 'response_template'" in r.message
            for r in caplog.records
        )

    def test_english_condition_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        """English-language condition generates a warning."""
        config = _two_stage_config(
            first_overrides={
                "transitions": [
                    {"target": "step2", "condition": "user_name is provided"},
                ],
            },
        )
        loader = WizardConfigLoader()
        with caplog.at_level(logging.WARNING):
            loader.load_from_dict(config)
        assert any(
            "appears to be natural language" in r.message for r in caplog.records
        )

    def test_python_format_syntax_warns(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Python ``{name}`` template syntax generates a warning."""
        config = _minimal_config(
            prompt="Hello {user_name}, welcome!",
        )
        loader = WizardConfigLoader()
        with caplog.at_level(logging.WARNING):
            loader.load_from_dict(config)
        assert any(
            "Python format syntax" in r.message for r in caplog.records
        )

    def test_python_format_in_response_template_warns(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Python format syntax in response_template warns."""
        config = _minimal_config(
            response_template="Hi {name}!",
        )
        loader = WizardConfigLoader()
        with caplog.at_level(logging.WARNING):
            loader.load_from_dict(config)
        assert any(
            "Python format syntax" in r.message for r in caplog.records
        )

    def test_valid_config_no_warnings(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A well-formed config produces no validation warnings."""
        config = _two_stage_config(
            first_overrides={
                "response_template": "Hello {{ name }}!",
                "schema": {
                    "type": "object",
                    "properties": {"val": {"type": "string"}},
                },
                "transitions": [
                    {"target": "step2", "condition": "data.get('val')"},
                ],
            },
        )
        loader = WizardConfigLoader()
        with caplog.at_level(logging.WARNING):
            loader.load_from_dict(config)
        validation_warnings = [
            r for r in caplog.records
            if r.levelno == logging.WARNING
            and (
                "unrecognized field" in r.message
                or "no 'schema'" in r.message
                or "natural language" in r.message
                or "Python format syntax" in r.message
            )
        ]
        assert validation_warnings == []

    def test_jinja2_syntax_not_flagged(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Proper Jinja2 ``{{ name }}`` is not flagged as Python format."""
        config = _minimal_config(
            response_template="Hello {{ user_name }}!",
        )
        loader = WizardConfigLoader()
        with caplog.at_level(logging.WARNING):
            loader.load_from_dict(config)
        assert not any(
            "Python format syntax" in r.message for r in caplog.records
        )

    def test_conversation_mode_no_schema_no_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Conversation-mode stages don't warn about missing schema."""
        config = _minimal_config(
            mode="conversation",
            is_end=False,
            transitions=[{"target": "welcome"}],
        )
        loader = WizardConfigLoader()
        with caplog.at_level(logging.WARNING):
            loader.load_from_dict(config)
        assert not any(
            "no 'schema' and no 'response_template'" in r.message
            for r in caplog.records
        )


# ---------------------------------------------------------------------------
# WizardConfigBuilder.validate() tests
# ---------------------------------------------------------------------------


class TestBuilderValidation:
    """Tests for WizardConfigBuilder validation warnings."""

    def test_pure_llm_stage_warns(self) -> None:
        """Non-end stage with no schema and no template produces a warning."""
        builder = (
            WizardConfigBuilder("test")
            .add_structured_stage("step1", "Hello", is_start=True)
            .add_end_stage("step2", "Done")
            .add_transition("step1", "step2")
        )
        result = builder.validate()
        assert any(
            "no schema and no response_template" in w for w in result.warnings
        )

    def test_python_format_template_warns(self) -> None:
        """response_template with ``{name}`` syntax produces a warning."""
        builder = (
            WizardConfigBuilder("test")
            .add_structured_stage(
                "step1",
                "Hello",
                is_start=True,
                is_end=True,
                response_template="Hi {name}!",
            )
        )
        result = builder.validate()
        assert any("Python format syntax" in w for w in result.warnings)

    def test_jinja2_template_no_warning(self) -> None:
        """response_template with Jinja2 syntax produces no format warning."""
        builder = (
            WizardConfigBuilder("test")
            .add_structured_stage(
                "step1",
                "Hello",
                is_start=True,
                is_end=True,
                response_template="Hi {{ name }}!",
                schema={"type": "object", "properties": {"name": {"type": "string"}}},
            )
        )
        result = builder.validate()
        assert not any("Python format syntax" in w for w in result.warnings)

    def test_valid_config_no_extra_warnings(self) -> None:
        """Well-formed builder config produces no new validation warnings."""
        builder = (
            WizardConfigBuilder("test")
            .add_structured_stage(
                "step1",
                "Enter name",
                is_start=True,
                schema={"type": "object", "properties": {"name": {"type": "string"}}},
                response_template="Hello {{ name }}!",
            )
            .add_end_stage("done", "Finished!")
            .add_transition("step1", "done", condition="data.get('name')")
        )
        result = builder.validate()
        assert result.valid
        # Filter for only our new warnings
        relevant = [
            w for w in result.warnings
            if "no schema and no response_template" in w
            or "Python format syntax" in w
        ]
        assert relevant == []
