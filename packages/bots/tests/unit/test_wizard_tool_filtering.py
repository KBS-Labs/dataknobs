"""Tests for wizard stage tool filtering.

Tests for the _filter_tools_for_stage method in WizardReasoning,
ensuring safe defaults (no tools) when stages don't specify tools.
"""

from typing import Any

import pytest

from dataknobs_bots.reasoning.wizard import WizardReasoning
from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader


class SimpleTool:
    """Simple callable tool for testing.

    Real tools have a 'name' attribute and are callable. This is a minimal
    implementation for testing tool filtering logic.
    """

    def __init__(self, name: str):
        self.name = name

    def __call__(self, *args: Any, **kwargs: Any) -> str:
        return f"Called {self.name}"


@pytest.fixture
def sample_tools() -> list[SimpleTool]:
    """Create sample tools for testing."""
    return [SimpleTool("tool_a"), SimpleTool("tool_b"), SimpleTool("tool_c")]


@pytest.fixture
def minimal_wizard_config() -> dict[str, Any]:
    """Create minimal wizard config for testing."""
    return {
        "name": "test-wizard",
        "stages": [
            {
                "name": "start",
                "is_start": True,
                "is_end": True,
                "prompt": "Test",
            }
        ],
    }


@pytest.fixture
def wizard_reasoning(minimal_wizard_config: dict[str, Any]) -> WizardReasoning:
    """Create WizardReasoning instance for testing."""
    loader = WizardConfigLoader()
    wizard_fsm = loader.load_from_dict(minimal_wizard_config)
    return WizardReasoning(wizard_fsm=wizard_fsm, strict_validation=False)


class TestFilterToolsForStage:
    """Tests for _filter_tools_for_stage method."""

    def test_no_tools_key_returns_none(
        self, wizard_reasoning: WizardReasoning, sample_tools: list[SimpleTool]
    ) -> None:
        """Stage without 'tools' key should return None (no tools)."""
        stage: dict[str, Any] = {"name": "configure_identity", "prompt": "Enter name"}

        result = wizard_reasoning._filter_tools_for_stage(stage, sample_tools)

        assert result is None

    def test_empty_tools_list_returns_none(
        self, wizard_reasoning: WizardReasoning, sample_tools: list[SimpleTool]
    ) -> None:
        """Stage with empty 'tools' list should return None."""
        stage: dict[str, Any] = {"name": "some_stage", "tools": []}

        result = wizard_reasoning._filter_tools_for_stage(stage, sample_tools)

        assert result is None

    def test_specific_tools_returns_filtered(
        self, wizard_reasoning: WizardReasoning, sample_tools: list[SimpleTool]
    ) -> None:
        """Stage with specific tools should return only those tools."""
        stage: dict[str, Any] = {"name": "review", "tools": ["tool_a", "tool_c"]}

        result = wizard_reasoning._filter_tools_for_stage(stage, sample_tools)

        assert result is not None
        assert len(result) == 2
        assert {t.name for t in result} == {"tool_a", "tool_c"}

    def test_no_tools_passed_returns_none(
        self, wizard_reasoning: WizardReasoning
    ) -> None:
        """When no tools are passed, return None regardless of stage config."""
        stage: dict[str, Any] = {"name": "review", "tools": ["tool_a"]}

        result = wizard_reasoning._filter_tools_for_stage(stage, None)

        assert result is None

    def test_empty_tools_passed_returns_none(
        self, wizard_reasoning: WizardReasoning
    ) -> None:
        """When empty tools list is passed, return None."""
        stage: dict[str, Any] = {"name": "review", "tools": ["tool_a"]}

        result = wizard_reasoning._filter_tools_for_stage(stage, [])

        assert result is None

    def test_nonexistent_tool_name_ignored(
        self, wizard_reasoning: WizardReasoning
    ) -> None:
        """Tool names in stage config that don't match any tool are ignored."""
        tools = [SimpleTool("tool_a")]
        stage: dict[str, Any] = {"name": "review", "tools": ["tool_a", "nonexistent"]}

        result = wizard_reasoning._filter_tools_for_stage(stage, tools)

        assert result is not None
        assert len(result) == 1
        assert result[0].name == "tool_a"

    def test_all_nonexistent_returns_none(
        self, wizard_reasoning: WizardReasoning
    ) -> None:
        """If no tools match the stage config, return None."""
        tools = [SimpleTool("tool_a")]
        stage: dict[str, Any] = {
            "name": "review",
            "tools": ["nonexistent1", "nonexistent2"],
        }

        result = wizard_reasoning._filter_tools_for_stage(stage, tools)

        assert result is None

    def test_preserves_tool_order(
        self, wizard_reasoning: WizardReasoning, sample_tools: list[SimpleTool]
    ) -> None:
        """Tools are returned in the order they appear in the input list."""
        # Request tools in reverse order (c, a)
        stage: dict[str, Any] = {"name": "review", "tools": ["tool_c", "tool_a"]}

        result = wizard_reasoning._filter_tools_for_stage(stage, sample_tools)

        assert result is not None
        # Should be in input order (a, c), not request order (c, a)
        assert [t.name for t in result] == ["tool_a", "tool_c"]

    def test_single_tool_match(
        self, wizard_reasoning: WizardReasoning, sample_tools: list[SimpleTool]
    ) -> None:
        """Stage requesting single tool returns just that tool."""
        stage: dict[str, Any] = {"name": "save", "tools": ["tool_b"]}

        result = wizard_reasoning._filter_tools_for_stage(stage, sample_tools)

        assert result is not None
        assert len(result) == 1
        assert result[0].name == "tool_b"

    def test_all_tools_match(
        self, wizard_reasoning: WizardReasoning, sample_tools: list[SimpleTool]
    ) -> None:
        """Stage requesting all available tools returns all."""
        stage: dict[str, Any] = {
            "name": "full_access",
            "tools": ["tool_a", "tool_b", "tool_c"],
        }

        result = wizard_reasoning._filter_tools_for_stage(stage, sample_tools)

        assert result is not None
        assert len(result) == 3
        assert {t.name for t in result} == {"tool_a", "tool_b", "tool_c"}


class TestToolFilteringIntegration:
    """Integration tests for tool filtering in wizard stages."""

    def test_data_collection_stage_no_tools(self) -> None:
        """Data collection stages should not receive tools."""
        wizard_config = {
            "name": "test-wizard",
            "stages": [
                {
                    "name": "welcome",
                    "prompt": "What subject?",
                    "is_start": True,
                    "schema": {
                        "type": "object",
                        "properties": {"subject": {"type": "string"}},
                    },
                    # Note: no 'tools' key - should get NO tools
                    "transitions": [
                        {"target": "done", "condition": "data.get('subject')"}
                    ],
                },
                {"name": "done", "prompt": "Complete!", "is_end": True},
            ],
        }

        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(wizard_config)
        reasoning = WizardReasoning(wizard_fsm=wizard_fsm, strict_validation=False)

        # Get the current stage metadata
        stage = reasoning._fsm.current_metadata

        # Even if tools are registered at bot level, this stage should get none
        all_tools = [SimpleTool("preview"), SimpleTool("save")]
        filtered = reasoning._filter_tools_for_stage(stage, all_tools)

        assert filtered is None, "Data collection stages should not receive tools"

    def test_tool_using_stage_gets_specified_tools(self) -> None:
        """Stages with explicit tools: key get only those tools."""
        wizard_config = {
            "name": "test-wizard",
            "stages": [
                {
                    "name": "review",
                    "is_start": True,
                    "is_end": True,
                    "prompt": "Review your config",
                    "tools": ["preview_config"],  # Explicit tool list
                }
            ],
        }

        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(wizard_config)
        reasoning = WizardReasoning(wizard_fsm=wizard_fsm, strict_validation=False)

        stage = reasoning._fsm.current_metadata

        all_tools = [
            SimpleTool("preview_config"),
            SimpleTool("save_config"),
            SimpleTool("validate_config"),
        ]
        filtered = reasoning._filter_tools_for_stage(stage, all_tools)

        assert filtered is not None
        assert len(filtered) == 1
        assert filtered[0].name == "preview_config"

    def test_multi_stage_wizard_tool_isolation(self) -> None:
        """Each stage should only get its specified tools."""
        wizard_config = {
            "name": "multi-stage-wizard",
            "stages": [
                {
                    "name": "configure",
                    "is_start": True,
                    "prompt": "Configure your settings",
                    # No tools key - data collection stage
                    "transitions": [{"target": "review"}],
                },
                {
                    "name": "review",
                    "prompt": "Review your config",
                    "tools": ["preview_config", "validate_config"],
                    "transitions": [{"target": "save"}],
                },
                {
                    "name": "save",
                    "is_end": True,
                    "prompt": "Save your config",
                    "tools": ["save_config"],
                },
            ],
        }

        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(wizard_config)
        reasoning = WizardReasoning(wizard_fsm=wizard_fsm, strict_validation=False)

        all_tools = [
            SimpleTool("preview_config"),
            SimpleTool("validate_config"),
            SimpleTool("save_config"),
        ]

        # Test each stage's tool access
        stages = wizard_fsm._stage_metadata

        # Configure stage: no tools key -> no tools
        configure_stage = stages["configure"]
        result = reasoning._filter_tools_for_stage(configure_stage, all_tools)
        assert result is None, "Configure stage should have no tools"

        # Review stage: 2 specific tools
        review_stage = stages["review"]
        result = reasoning._filter_tools_for_stage(review_stage, all_tools)
        assert result is not None
        assert len(result) == 2
        assert {t.name for t in result} == {"preview_config", "validate_config"}

        # Save stage: 1 specific tool
        save_stage = stages["save"]
        result = reasoning._filter_tools_for_stage(save_stage, all_tools)
        assert result is not None
        assert len(result) == 1
        assert result[0].name == "save_config"
