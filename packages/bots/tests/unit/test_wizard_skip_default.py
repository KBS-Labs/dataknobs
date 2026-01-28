"""Tests for wizard skip_default functionality.

Tests for applying default values when users skip stages.
"""

from typing import Any

import pytest

from dataknobs_bots.reasoning.wizard import WizardReasoning, WizardState
from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader

from .conftest import WizardTestManager


class TestSkipDefault:
    """Tests for skip_default stage configuration."""

    def test_skip_default_loaded_in_metadata(self) -> None:
        """skip_default should be included in stage metadata."""
        wizard_config: dict[str, Any] = {
            "name": "test-wizard",
            "stages": [
                {
                    "name": "configure_llm",
                    "is_start": True,
                    "is_end": True,
                    "prompt": "Choose your LLM",
                    "can_skip": True,
                    "skip_default": {"provider": "anthropic", "model": "claude-3"},
                }
            ],
        }

        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(wizard_config)

        metadata = wizard_fsm.current_metadata
        assert metadata["skip_default"] == {"provider": "anthropic", "model": "claude-3"}

    def test_skip_default_none_when_not_configured(self) -> None:
        """skip_default should be None when not configured."""
        wizard_config: dict[str, Any] = {
            "name": "test-wizard",
            "stages": [
                {
                    "name": "configure",
                    "is_start": True,
                    "is_end": True,
                    "prompt": "Configure",
                    "can_skip": True,
                    # No skip_default
                }
            ],
        }

        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(wizard_config)

        metadata = wizard_fsm.current_metadata
        assert metadata.get("skip_default") is None


class TestSkipWithDefaults:
    """Tests for applying skip_default when user skips."""

    @pytest.mark.asyncio
    async def test_skip_applies_default_values(self) -> None:
        """Skipping a stage should apply skip_default values to state data."""
        wizard_config: dict[str, Any] = {
            "name": "test-wizard",
            "stages": [
                {
                    "name": "configure_llm",
                    "is_start": True,
                    "prompt": "Choose your LLM",
                    "can_skip": True,
                    "skip_default": {"provider": "anthropic", "model": "claude-3"},
                    "transitions": [{"target": "done"}],
                },
                {
                    "name": "done",
                    "is_end": True,
                    "prompt": "Complete!",
                },
            ],
        }

        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(wizard_config)
        reasoning = WizardReasoning(wizard_fsm=wizard_fsm, strict_validation=False)

        manager = WizardTestManager()
        manager.messages = [{"role": "user", "content": "skip"}]
        manager.metadata = {}

        # Script response for after skip
        manager.echo_provider.set_responses(["Skipped! Moving on."])

        state = WizardState(current_stage="configure_llm")

        # Handle navigation should apply skip_default
        # Args: message, state, manager, llm
        result = await reasoning._handle_navigation("skip", state, manager, None)

        # Should return None (continue to normal flow)
        assert result is None

        # State should have skip marker and default values
        assert state.data["_skipped_configure_llm"] is True
        assert state.data["provider"] == "anthropic"
        assert state.data["model"] == "claude-3"

    @pytest.mark.asyncio
    async def test_skip_without_default_no_values_added(self) -> None:
        """Skipping without skip_default should only set skip marker."""
        wizard_config: dict[str, Any] = {
            "name": "test-wizard",
            "stages": [
                {
                    "name": "optional_step",
                    "is_start": True,
                    "prompt": "Optional configuration",
                    "can_skip": True,
                    # No skip_default
                    "transitions": [{"target": "done"}],
                },
                {
                    "name": "done",
                    "is_end": True,
                    "prompt": "Complete!",
                },
            ],
        }

        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(wizard_config)
        reasoning = WizardReasoning(wizard_fsm=wizard_fsm, strict_validation=False)

        manager = WizardTestManager()
        state = WizardState(current_stage="optional_step")

        await reasoning._handle_navigation("skip", state, manager, None)

        # Should only have skip marker
        assert state.data["_skipped_optional_step"] is True
        assert len(state.data) == 1  # Only the skip marker

    @pytest.mark.asyncio
    async def test_use_defaults_triggers_skip(self) -> None:
        """'use defaults' should trigger skip with defaults."""
        wizard_config: dict[str, Any] = {
            "name": "test-wizard",
            "stages": [
                {
                    "name": "settings",
                    "is_start": True,
                    "is_end": True,
                    "prompt": "Configure settings",
                    "can_skip": True,
                    "skip_default": {"theme": "dark", "language": "en"},
                }
            ],
        }

        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(wizard_config)
        reasoning = WizardReasoning(wizard_fsm=wizard_fsm, strict_validation=False)

        manager = WizardTestManager()
        state = WizardState(current_stage="settings")

        await reasoning._handle_navigation("use defaults", state, manager, None)

        assert state.data["theme"] == "dark"
        assert state.data["language"] == "en"

    @pytest.mark.asyncio
    async def test_use_default_singular_triggers_skip(self) -> None:
        """'use default' (singular) should trigger skip with defaults."""
        wizard_config: dict[str, Any] = {
            "name": "test-wizard",
            "stages": [
                {
                    "name": "settings",
                    "is_start": True,
                    "is_end": True,
                    "prompt": "Configure settings",
                    "can_skip": True,
                    "skip_default": {"value": 42},
                }
            ],
        }

        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(wizard_config)
        reasoning = WizardReasoning(wizard_fsm=wizard_fsm, strict_validation=False)

        manager = WizardTestManager()
        state = WizardState(current_stage="settings")

        await reasoning._handle_navigation("use default", state, manager, None)

        assert state.data["value"] == 42

    @pytest.mark.asyncio
    async def test_skip_non_skippable_stage_rejected(self) -> None:
        """Skipping a non-skippable stage should be rejected."""
        wizard_config: dict[str, Any] = {
            "name": "test-wizard",
            "stages": [
                {
                    "name": "required_step",
                    "is_start": True,
                    "is_end": True,
                    "prompt": "Required information",
                    "can_skip": False,  # Not skippable
                    "skip_default": {"value": "ignored"},
                }
            ],
        }

        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(wizard_config)
        reasoning = WizardReasoning(wizard_fsm=wizard_fsm, strict_validation=False)

        manager = WizardTestManager()
        manager.echo_provider.set_responses(["This step cannot be skipped."])
        state = WizardState(current_stage="required_step")

        result = await reasoning._handle_navigation("skip", state, manager, None)

        # Should return a response (rejection message)
        assert result is not None
        # State should not have skip marker or defaults
        assert "_skipped_required_step" not in state.data
        assert "value" not in state.data

    @pytest.mark.asyncio
    async def test_skip_default_merges_with_existing_data(self) -> None:
        """skip_default should merge with, not replace, existing state data."""
        wizard_config: dict[str, Any] = {
            "name": "test-wizard",
            "stages": [
                {
                    "name": "configure",
                    "is_start": True,
                    "is_end": True,
                    "prompt": "Configure",
                    "can_skip": True,
                    "skip_default": {"new_field": "default_value"},
                }
            ],
        }

        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(wizard_config)
        reasoning = WizardReasoning(wizard_fsm=wizard_fsm, strict_validation=False)

        manager = WizardTestManager()
        # Pre-existing data from earlier stages
        state = WizardState(
            current_stage="configure",
            data={"existing_field": "existing_value"},
        )

        await reasoning._handle_navigation("skip", state, manager, None)

        # Both existing and default values should be present
        assert state.data["existing_field"] == "existing_value"
        assert state.data["new_field"] == "default_value"
        assert state.data["_skipped_configure"] is True

    @pytest.mark.asyncio
    async def test_skip_default_with_nested_values(self) -> None:
        """skip_default should work with nested dict values."""
        wizard_config: dict[str, Any] = {
            "name": "test-wizard",
            "stages": [
                {
                    "name": "configure",
                    "is_start": True,
                    "is_end": True,
                    "prompt": "Configure",
                    "can_skip": True,
                    "skip_default": {
                        "llm": {"provider": "anthropic", "model": "claude-3"},
                        "features": ["chat", "tools"],
                    },
                }
            ],
        }

        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(wizard_config)
        reasoning = WizardReasoning(wizard_fsm=wizard_fsm, strict_validation=False)

        manager = WizardTestManager()
        state = WizardState(current_stage="configure")

        await reasoning._handle_navigation("skip", state, manager, None)

        assert state.data["llm"] == {"provider": "anthropic", "model": "claude-3"}
        assert state.data["features"] == ["chat", "tools"]
