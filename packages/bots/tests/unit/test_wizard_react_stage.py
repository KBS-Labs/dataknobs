"""Tests for Wizard + ReAct composition (Phase 4).

This module tests the ReAct-style tool iteration within wizard stages,
enabling the LLM to make multiple tool calls within a single wizard turn.
"""

from typing import Any

import pytest

from dataknobs_bots.reasoning.wizard import WizardReasoning, WizardState
from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader
from dataknobs_llm.testing import ResponseSequenceBuilder, text_response, tool_call_response
from dataknobs_llm.tools.base import Tool

from .conftest import WizardTestManager


# -----------------------------------------------------------------------------
# Test Tools
# -----------------------------------------------------------------------------


class PreviewConfigTool(Tool):
    """Test tool that previews configuration."""

    def __init__(self) -> None:
        super().__init__(
            name="preview_config",
            description="Preview the current configuration",
        )
        self.call_count = 0
        self.last_call_args: dict[str, Any] = {}

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "format": {
                    "type": "string",
                    "enum": ["yaml", "json"],
                    "default": "yaml",
                }
            },
        }

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        self.call_count += 1
        self.last_call_args = kwargs
        return {
            "preview": {
                "name": "Test Bot",
                "provider": "anthropic",
            }
        }


class ValidateConfigTool(Tool):
    """Test tool that validates configuration."""

    def __init__(self, errors: list[str] | None = None) -> None:
        super().__init__(
            name="validate_config",
            description="Validate the current configuration",
        )
        self.call_count = 0
        self.errors = errors or []

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "strict": {"type": "boolean", "default": False}
            },
        }

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        self.call_count += 1
        return {
            "valid": len(self.errors) == 0,
            "errors": self.errors,
        }


class FailingTool(Tool):
    """Test tool that raises an exception."""

    def __init__(self) -> None:
        super().__init__(
            name="failing_tool",
            description="A tool that always fails",
        )
        self.call_count = 0

    @property
    def schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        self.call_count += 1
        raise ValueError("Tool execution failed intentionally")


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def react_wizard_config() -> dict[str, Any]:
    """Create a wizard config with ReAct-enabled stage."""
    return {
        "name": "react-test-wizard",
        "version": "1.0",
        "settings": {
            "tool_reasoning": "single",  # Default
            "max_tool_iterations": 3,
        },
        "stages": [
            {
                "name": "review",
                "is_start": True,
                "prompt": "Review your configuration",
                "reasoning": "react",
                "max_iterations": 2,
                "tools": ["preview_config", "validate_config"],
                "transitions": [{"target": "done"}],
            },
            {
                "name": "done",
                "is_end": True,
                "prompt": "Complete!",
            },
        ],
    }


@pytest.fixture
def default_react_wizard_config() -> dict[str, Any]:
    """Wizard config where tool_reasoning defaults to 'react'."""
    return {
        "name": "default-react-wizard",
        "version": "1.0",
        "settings": {
            "tool_reasoning": "react",
            "max_tool_iterations": 5,
        },
        "stages": [
            {
                "name": "review",
                "is_start": True,
                "prompt": "Review your configuration",
                "tools": ["preview_config"],
                "transitions": [{"target": "done"}],
            },
            {
                "name": "done",
                "is_end": True,
                "prompt": "Complete!",
            },
        ],
    }


@pytest.fixture
def preview_tool() -> PreviewConfigTool:
    """Create preview config tool."""
    return PreviewConfigTool()


@pytest.fixture
def validate_tool() -> ValidateConfigTool:
    """Create validate config tool."""
    return ValidateConfigTool()


@pytest.fixture
def failing_tool() -> FailingTool:
    """Create failing tool."""
    return FailingTool()


# -----------------------------------------------------------------------------
# TestUseReactForStage
# -----------------------------------------------------------------------------


class TestUseReactForStage:
    """Tests for _use_react_for_stage method."""

    def test_stage_with_react_reasoning_uses_react(
        self, react_wizard_config: dict[str, Any]
    ) -> None:
        """Stage with reasoning: react uses ReAct."""
        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(react_wizard_config)
        reasoning = WizardReasoning(wizard_fsm=wizard_fsm)

        stage = {"name": "review", "reasoning": "react"}
        assert reasoning._use_react_for_stage(stage) is True

    def test_stage_with_single_reasoning_uses_single(
        self, react_wizard_config: dict[str, Any]
    ) -> None:
        """Stage with reasoning: single uses single call."""
        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(react_wizard_config)
        reasoning = WizardReasoning(wizard_fsm=wizard_fsm)

        stage = {"name": "welcome", "reasoning": "single"}
        assert reasoning._use_react_for_stage(stage) is False

    def test_default_tool_reasoning_applied_when_no_stage_setting(
        self, default_react_wizard_config: dict[str, Any]
    ) -> None:
        """Default tool_reasoning setting applies when stage doesn't specify."""
        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(default_react_wizard_config)
        reasoning = WizardReasoning(
            wizard_fsm=wizard_fsm,
            default_tool_reasoning="react",
        )

        stage = {"name": "review"}  # No explicit reasoning
        assert reasoning._use_react_for_stage(stage) is True

    def test_case_insensitive_reasoning_value(
        self, react_wizard_config: dict[str, Any]
    ) -> None:
        """Reasoning value is case-insensitive."""
        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(react_wizard_config)
        reasoning = WizardReasoning(wizard_fsm=wizard_fsm)

        assert reasoning._use_react_for_stage({"reasoning": "REACT"}) is True
        assert reasoning._use_react_for_stage({"reasoning": "React"}) is True
        assert reasoning._use_react_for_stage({"reasoning": "Single"}) is False
        assert reasoning._use_react_for_stage({"reasoning": "SINGLE"}) is False


# -----------------------------------------------------------------------------
# TestGetMaxIterations
# -----------------------------------------------------------------------------


class TestGetMaxIterations:
    """Tests for _get_max_iterations method."""

    def test_get_max_iterations_from_stage(
        self, react_wizard_config: dict[str, Any]
    ) -> None:
        """Max iterations from stage config."""
        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(react_wizard_config)
        reasoning = WizardReasoning(
            wizard_fsm=wizard_fsm,
            default_max_iterations=5,
        )

        stage = {"name": "review", "max_iterations": 2}
        assert reasoning._get_max_iterations(stage) == 2

    def test_get_max_iterations_default_fallback(
        self, react_wizard_config: dict[str, Any]
    ) -> None:
        """Max iterations falls back to default when not set."""
        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(react_wizard_config)
        reasoning = WizardReasoning(
            wizard_fsm=wizard_fsm,
            default_max_iterations=5,
        )

        stage = {"name": "save"}  # No max_iterations
        assert reasoning._get_max_iterations(stage) == 5

    def test_get_max_iterations_zero_uses_default(
        self, react_wizard_config: dict[str, Any]
    ) -> None:
        """max_iterations of 0 falls back to default (falsy value check)."""
        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(react_wizard_config)
        reasoning = WizardReasoning(
            wizard_fsm=wizard_fsm,
            default_max_iterations=3,
        )

        stage = {"name": "review", "max_iterations": 0}
        assert reasoning._get_max_iterations(stage) == 3


# -----------------------------------------------------------------------------
# TestFindTool
# -----------------------------------------------------------------------------


class TestFindTool:
    """Tests for _find_tool method."""

    def test_find_existing_tool(
        self,
        react_wizard_config: dict[str, Any],
        preview_tool: PreviewConfigTool,
        validate_tool: ValidateConfigTool,
    ) -> None:
        """Find tool by name."""
        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(react_wizard_config)
        reasoning = WizardReasoning(wizard_fsm=wizard_fsm)

        tools = [preview_tool, validate_tool]

        found = reasoning._find_tool("preview_config", tools)
        assert found is preview_tool

        found = reasoning._find_tool("validate_config", tools)
        assert found is validate_tool

    def test_find_nonexistent_tool_returns_none(
        self,
        react_wizard_config: dict[str, Any],
        preview_tool: PreviewConfigTool,
    ) -> None:
        """Finding nonexistent tool returns None."""
        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(react_wizard_config)
        reasoning = WizardReasoning(wizard_fsm=wizard_fsm)

        tools = [preview_tool]

        found = reasoning._find_tool("unknown_tool", tools)
        assert found is None

    def test_find_tool_empty_list(
        self, react_wizard_config: dict[str, Any]
    ) -> None:
        """Finding tool in empty list returns None."""
        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(react_wizard_config)
        reasoning = WizardReasoning(wizard_fsm=wizard_fsm)

        found = reasoning._find_tool("any_tool", [])
        assert found is None


# -----------------------------------------------------------------------------
# TestExecuteReactToolCall
# -----------------------------------------------------------------------------


class TestExecuteReactToolCall:
    """Tests for _execute_react_tool_call method."""

    @pytest.mark.asyncio
    async def test_execute_tool_successfully(
        self,
        react_wizard_config: dict[str, Any],
        preview_tool: PreviewConfigTool,
    ) -> None:
        """Execute tool and get result."""
        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(react_wizard_config)
        reasoning = WizardReasoning(wizard_fsm=wizard_fsm)
        state = WizardState(current_stage="review", data={})

        # Create a mock tool call
        class MockToolCall:
            name = "preview_config"
            parameters: dict[str, Any] = {"format": "yaml"}

        tool_call = MockToolCall()
        tools = [preview_tool]

        # Mock tool context (we don't need real context for this test)
        tool_context = None

        result = await reasoning._execute_react_tool_call(
            tool_call, tools, state, tool_context
        )

        assert preview_tool.call_count == 1
        assert "preview" in result
        assert result["preview"]["name"] == "Test Bot"

    @pytest.mark.asyncio
    async def test_execute_unknown_tool_returns_error(
        self,
        react_wizard_config: dict[str, Any],
        preview_tool: PreviewConfigTool,
    ) -> None:
        """Executing unknown tool returns error dict."""
        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(react_wizard_config)
        reasoning = WizardReasoning(wizard_fsm=wizard_fsm)
        state = WizardState(current_stage="review", data={})

        class MockToolCall:
            name = "unknown_tool"
            parameters: dict[str, Any] = {}

        tool_call = MockToolCall()
        tools = [preview_tool]
        tool_context = None

        result = await reasoning._execute_react_tool_call(
            tool_call, tools, state, tool_context
        )

        assert "error" in result
        assert "not found" in result["error"]
        assert preview_tool.call_count == 0

    @pytest.mark.asyncio
    async def test_execute_failing_tool_returns_error(
        self,
        react_wizard_config: dict[str, Any],
        failing_tool: FailingTool,
    ) -> None:
        """Executing failing tool returns error dict."""
        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(react_wizard_config)
        reasoning = WizardReasoning(wizard_fsm=wizard_fsm)
        state = WizardState(current_stage="review", data={})

        class MockToolCall:
            name = "failing_tool"
            parameters: dict[str, Any] = {}

        tool_call = MockToolCall()
        tools = [failing_tool]
        tool_context = None

        result = await reasoning._execute_react_tool_call(
            tool_call, tools, state, tool_context
        )

        assert "error" in result
        assert "failed" in result["error"].lower()
        assert failing_tool.call_count == 1


# -----------------------------------------------------------------------------
# TestReactStageResponse
# -----------------------------------------------------------------------------


class TestReactStageResponse:
    """Tests for _react_stage_response method (full ReAct loop)."""

    @pytest.mark.asyncio
    async def test_react_loop_no_tool_calls_returns_immediately(
        self,
        react_wizard_config: dict[str, Any],
        preview_tool: PreviewConfigTool,
    ) -> None:
        """ReAct loop returns immediately when LLM doesn't request tools."""
        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(react_wizard_config)
        reasoning = WizardReasoning(wizard_fsm=wizard_fsm)
        state = WizardState(current_stage="review", data={})

        manager = WizardTestManager()
        manager.add_user_message("Show me the config")

        # Script the provider to return a text response (no tool calls)
        manager.echo_provider.set_responses([
            text_response("Here's your config summary.")
        ])

        stage = {"name": "review", "max_iterations": 3}
        tools = [preview_tool]

        response = await reasoning._react_stage_response(
            manager, "Test prompt", stage, state, tools
        )

        assert response.content == "Here's your config summary."
        assert preview_tool.call_count == 0  # Tool was not called

    @pytest.mark.asyncio
    async def test_react_loop_executes_tool_and_returns_final_response(
        self,
        react_wizard_config: dict[str, Any],
        preview_tool: PreviewConfigTool,
    ) -> None:
        """ReAct loop executes tool, then returns final response."""
        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(react_wizard_config)
        reasoning = WizardReasoning(wizard_fsm=wizard_fsm)
        state = WizardState(current_stage="review", data={})

        manager = WizardTestManager()
        manager.add_user_message("Preview the config")

        # Script responses: first a tool call, then a text response
        manager.echo_provider.set_responses([
            tool_call_response("preview_config", {"format": "yaml"}),
            text_response("I've previewed your config. It looks good!"),
        ])

        stage = {"name": "review", "max_iterations": 3}
        tools = [preview_tool]

        response = await reasoning._react_stage_response(
            manager, "Test prompt", stage, state, tools
        )

        assert preview_tool.call_count == 1
        assert response.content == "I've previewed your config. It looks good!"

    @pytest.mark.asyncio
    async def test_react_loop_multiple_tool_calls(
        self,
        react_wizard_config: dict[str, Any],
        preview_tool: PreviewConfigTool,
        validate_tool: ValidateConfigTool,
    ) -> None:
        """ReAct loop can make multiple sequential tool calls."""
        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(react_wizard_config)
        reasoning = WizardReasoning(wizard_fsm=wizard_fsm)
        state = WizardState(current_stage="review", data={})

        manager = WizardTestManager()
        manager.add_user_message("Preview and validate the config")

        # Script responses: preview, validate, then text
        manager.echo_provider.set_responses([
            tool_call_response("preview_config", {}),
            tool_call_response("validate_config", {}),
            text_response("Config previewed and validated!"),
        ])

        stage = {"name": "review", "max_iterations": 5}
        tools = [preview_tool, validate_tool]

        response = await reasoning._react_stage_response(
            manager, "Test prompt", stage, state, tools
        )

        assert preview_tool.call_count == 1
        assert validate_tool.call_count == 1
        assert response.content == "Config previewed and validated!"

    @pytest.mark.asyncio
    async def test_react_loop_max_iterations_forces_text_response(
        self,
        react_wizard_config: dict[str, Any],
        preview_tool: PreviewConfigTool,
    ) -> None:
        """Max iterations reached forces final text response."""
        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(react_wizard_config)
        reasoning = WizardReasoning(wizard_fsm=wizard_fsm)
        state = WizardState(current_stage="review", data={})

        manager = WizardTestManager()
        manager.add_user_message("Preview endlessly")

        # Script: 2 tool calls (hitting max_iterations=2), then final text
        manager.echo_provider.set_responses([
            tool_call_response("preview_config", {}),  # Iteration 1
            tool_call_response("preview_config", {}),  # Iteration 2 (max)
            text_response("Final response after max iterations"),  # Forced
        ])

        stage = {"name": "review", "max_iterations": 2}
        tools = [preview_tool]

        response = await reasoning._react_stage_response(
            manager, "Test prompt", stage, state, tools
        )

        # Tool was called twice (once per iteration)
        assert preview_tool.call_count == 2
        assert response.content == "Final response after max iterations"


# -----------------------------------------------------------------------------
# TestReactConfigLoading
# -----------------------------------------------------------------------------


class TestReactConfigLoading:
    """Tests for loading ReAct config from wizard YAML."""

    def test_stage_reasoning_loaded_from_config(
        self, react_wizard_config: dict[str, Any]
    ) -> None:
        """Stage reasoning setting is loaded from config."""
        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(react_wizard_config)

        # Check stage metadata (current_metadata is a property)
        stage = wizard_fsm.current_metadata
        assert stage.get("reasoning") == "react"
        assert stage.get("max_iterations") == 2

    def test_wizard_level_defaults_loaded(
        self, default_react_wizard_config: dict[str, Any]
    ) -> None:
        """Wizard-level default settings are loaded."""
        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(default_react_wizard_config)

        reasoning = WizardReasoning(
            wizard_fsm=wizard_fsm,
            default_tool_reasoning=wizard_fsm.settings.get("tool_reasoning", "single"),
            default_max_iterations=wizard_fsm.settings.get("max_tool_iterations", 3),
        )

        assert reasoning._default_tool_reasoning == "react"
        assert reasoning._default_max_iterations == 5

    def test_from_config_loads_react_settings(self) -> None:
        """WizardReasoning.from_config loads ReAct settings."""
        # This requires a file, so we test via load_from_dict path
        config = {
            "name": "config-test",
            "version": "1.0",
            "settings": {
                "tool_reasoning": "react",
                "max_tool_iterations": 7,
            },
            "stages": [
                {"name": "start", "is_start": True, "is_end": True, "prompt": "Done"}
            ],
        }

        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(config)

        assert wizard_fsm.settings.get("tool_reasoning") == "react"
        assert wizard_fsm.settings.get("max_tool_iterations") == 7
