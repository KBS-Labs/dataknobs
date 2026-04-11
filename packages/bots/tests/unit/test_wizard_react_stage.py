"""Tests for Wizard + strategy composition (Phase 4).

This module tests per-state strategy injection in wizard stages,
including ReAct-style tool iteration within a single wizard turn.
"""

import logging
from typing import Any

import pytest

from dataknobs_bots.reasoning.wizard import WizardReasoning, WizardState
from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader
from dataknobs_llm.conversations import ConversationManager
from dataknobs_llm.llm.providers.echo import EchoProvider
from dataknobs_llm.testing import text_response, tool_call_response
from dataknobs_llm.tools.base import Tool


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


# -----------------------------------------------------------------------------
# TestUseReactForStage
# -----------------------------------------------------------------------------


class TestResolveStageStrategy:
    """Tests for _resolve_stage_strategy method."""

    def test_stage_with_react_reasoning_resolves_react(
        self, react_wizard_config: dict[str, Any]
    ) -> None:
        """Stage with reasoning: react resolves to a ReActReasoning."""
        from dataknobs_bots.reasoning.react import ReActReasoning

        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(react_wizard_config)
        reasoning = WizardReasoning(wizard_fsm=wizard_fsm)

        stage = {"name": "review", "reasoning": "react"}
        strategy = reasoning._resolve_stage_strategy(stage)
        assert isinstance(strategy, ReActReasoning)

    def test_stage_with_single_reasoning_resolves_none(
        self, react_wizard_config: dict[str, Any]
    ) -> None:
        """Stage with reasoning: single resolves to None (direct call)."""
        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(react_wizard_config)
        reasoning = WizardReasoning(wizard_fsm=wizard_fsm)

        stage = {"name": "welcome", "reasoning": "single"}
        assert reasoning._resolve_stage_strategy(stage) is None

    def test_default_tool_reasoning_applied_when_no_stage_setting(
        self, default_react_wizard_config: dict[str, Any]
    ) -> None:
        """Default tool_reasoning setting applies when stage doesn't specify."""
        from dataknobs_bots.reasoning.react import ReActReasoning

        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(default_react_wizard_config)
        reasoning = WizardReasoning(
            wizard_fsm=wizard_fsm,
            default_tool_reasoning="react",
        )

        stage = {"name": "review"}  # No explicit reasoning
        strategy = reasoning._resolve_stage_strategy(stage)
        assert isinstance(strategy, ReActReasoning)

    def test_case_insensitive_reasoning_value(
        self, react_wizard_config: dict[str, Any]
    ) -> None:
        """Reasoning value is case-insensitive."""
        from dataknobs_bots.reasoning.react import ReActReasoning

        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(react_wizard_config)
        reasoning = WizardReasoning(wizard_fsm=wizard_fsm)

        assert isinstance(
            reasoning._resolve_stage_strategy({"reasoning": "REACT"}),
            ReActReasoning,
        )
        assert isinstance(
            reasoning._resolve_stage_strategy({"reasoning": "React"}),
            ReActReasoning,
        )
        assert reasoning._resolve_stage_strategy({"reasoning": "Single"}) is None
        assert reasoning._resolve_stage_strategy({"reasoning": "SINGLE"}) is None


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
# TestReactStageResponse
# -----------------------------------------------------------------------------


class TestStrategyStageResponse:
    """Tests for _strategy_stage_response method (full ReAct loop via registry)."""

    async def _run_strategy_response(
        self,
        reasoning: WizardReasoning,
        manager: Any,
        stage: dict[str, Any],
        tools: list[Any],
    ) -> Any:
        """Helper: resolve strategy and delegate to _strategy_stage_response."""
        strategy = reasoning._resolve_stage_strategy(stage)
        assert strategy is not None, "Expected a non-single strategy"
        state = WizardState(current_stage=stage.get("name", "test"), data={})
        response, *_ = await reasoning._strategy_stage_response(
            strategy, manager, "Test prompt", stage, state, tools,
        )
        return response

    @pytest.mark.asyncio
    async def test_react_loop_no_tool_calls_returns_immediately(
        self,
        react_wizard_config: dict[str, Any],
        preview_tool: PreviewConfigTool,
        conversation_manager_pair: tuple[ConversationManager, EchoProvider],
    ) -> None:
        """ReAct loop returns immediately when LLM doesn't request tools."""
        manager, provider = conversation_manager_pair

        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(react_wizard_config)
        reasoning = WizardReasoning(wizard_fsm=wizard_fsm)

        await manager.add_message(role="user", content="Show me the config")

        provider.set_responses([
            text_response("Here's your config summary.")
        ])

        stage = {"name": "review", "reasoning": "react", "max_iterations": 3}
        response = await self._run_strategy_response(
            reasoning, manager, stage, [preview_tool],
        )

        assert response.content == "Here's your config summary."
        assert preview_tool.call_count == 0

    @pytest.mark.asyncio
    async def test_react_loop_executes_tool_and_returns_final_response(
        self,
        react_wizard_config: dict[str, Any],
        preview_tool: PreviewConfigTool,
        conversation_manager_pair: tuple[ConversationManager, EchoProvider],
    ) -> None:
        """ReAct loop executes tool, then returns final response."""
        manager, provider = conversation_manager_pair

        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(react_wizard_config)
        reasoning = WizardReasoning(wizard_fsm=wizard_fsm)

        await manager.add_message(role="user", content="Preview the config")

        provider.set_responses([
            tool_call_response("preview_config", {"format": "yaml"}),
            text_response("I've previewed your config. It looks good!"),
        ])

        stage = {"name": "review", "reasoning": "react", "max_iterations": 3}
        response = await self._run_strategy_response(
            reasoning, manager, stage, [preview_tool],
        )

        assert preview_tool.call_count == 1
        assert response.content == "I've previewed your config. It looks good!"

    @pytest.mark.asyncio
    async def test_react_loop_multiple_tool_calls(
        self,
        react_wizard_config: dict[str, Any],
        preview_tool: PreviewConfigTool,
        validate_tool: ValidateConfigTool,
        conversation_manager_pair: tuple[ConversationManager, EchoProvider],
    ) -> None:
        """ReAct loop can make multiple sequential tool calls."""
        manager, provider = conversation_manager_pair

        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(react_wizard_config)
        reasoning = WizardReasoning(wizard_fsm=wizard_fsm)

        await manager.add_message(role="user", content="Preview and validate")

        provider.set_responses([
            tool_call_response("preview_config", {}),
            tool_call_response("validate_config", {}),
            text_response("Config previewed and validated!"),
        ])

        stage = {"name": "review", "reasoning": "react", "max_iterations": 5}
        response = await self._run_strategy_response(
            reasoning, manager, stage, [preview_tool, validate_tool],
        )

        assert preview_tool.call_count == 1
        assert validate_tool.call_count == 1
        assert response.content == "Config previewed and validated!"

    @pytest.mark.asyncio
    async def test_react_loop_max_iterations_forces_text_response(
        self,
        react_wizard_config: dict[str, Any],
        preview_tool: PreviewConfigTool,
        conversation_manager_pair: tuple[ConversationManager, EchoProvider],
    ) -> None:
        """Max iterations reached forces final text response."""
        manager, provider = conversation_manager_pair

        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(react_wizard_config)
        reasoning = WizardReasoning(wizard_fsm=wizard_fsm)

        await manager.add_message(role="user", content="Preview endlessly")

        provider.set_responses([
            tool_call_response("preview_config", {"format": "yaml"}),
            tool_call_response("preview_config", {"format": "json"}),
            text_response("Final response after max iterations"),
        ])

        stage = {"name": "review", "reasoning": "react", "max_iterations": 2}
        response = await self._run_strategy_response(
            reasoning, manager, stage, [preview_tool],
        )

        assert preview_tool.call_count == 2
        assert response.content == "Final response after max iterations"

    @pytest.mark.asyncio
    async def test_react_loop_duplicate_tool_calls_break_early(
        self,
        react_wizard_config: dict[str, Any],
        preview_tool: PreviewConfigTool,
        conversation_manager_pair: tuple[ConversationManager, EchoProvider],
    ) -> None:
        """Identical consecutive tool calls trigger early break."""
        manager, provider = conversation_manager_pair

        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(react_wizard_config)
        reasoning = WizardReasoning(wizard_fsm=wizard_fsm)

        await manager.add_message(role="user", content="Preview config")

        provider.set_responses([
            tool_call_response("preview_config", {"format": "yaml"}),
            tool_call_response("preview_config", {"format": "yaml"}),
            text_response("Done after duplicate detection"),
        ])

        stage = {"name": "review", "reasoning": "react", "max_iterations": 5}
        response = await self._run_strategy_response(
            reasoning, manager, stage, [preview_tool],
        )

        assert preview_tool.call_count == 1
        assert response.content == "Done after duplicate detection"

    @pytest.mark.asyncio
    async def test_react_stage_stores_reasoning_trace(
        self,
        react_wizard_config: dict[str, Any],
        preview_tool: PreviewConfigTool,
        conversation_manager_pair: tuple[ConversationManager, EchoProvider],
    ) -> None:
        """Wizard ReAct stage stores reasoning trace when store_trace enabled."""
        manager, provider = conversation_manager_pair

        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(react_wizard_config)
        reasoning = WizardReasoning(
            wizard_fsm=wizard_fsm, default_store_trace=True,
        )

        await manager.add_message(role="user", content="Preview the config")

        provider.set_responses([
            tool_call_response("preview_config", {"format": "yaml"}),
            text_response("Here is the preview."),
        ])

        stage = {"name": "review", "reasoning": "react", "max_iterations": 3}
        await self._run_strategy_response(
            reasoning, manager, stage, [preview_tool],
        )

        trace = manager.metadata.get("reasoning_trace")
        assert trace is not None, "reasoning_trace not stored in metadata"
        assert len(trace) >= 1, "trace should have at least one iteration"

        tool_names = [
            tc["name"]
            for step in trace
            for tc in step.get("tool_calls", [])
        ]
        assert "preview_config" in tool_names

    @pytest.mark.asyncio
    async def test_react_stage_no_trace_by_default(
        self,
        react_wizard_config: dict[str, Any],
        preview_tool: PreviewConfigTool,
        conversation_manager_pair: tuple[ConversationManager, EchoProvider],
    ) -> None:
        """Wizard ReAct stage does NOT store trace when store_trace is off."""
        manager, provider = conversation_manager_pair

        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(react_wizard_config)
        reasoning = WizardReasoning(wizard_fsm=wizard_fsm)

        await manager.add_message(role="user", content="Preview the config")

        provider.set_responses([
            tool_call_response("preview_config", {"format": "yaml"}),
            text_response("Here is the preview."),
        ])

        stage = {"name": "review", "reasoning": "react", "max_iterations": 3}
        await self._run_strategy_response(
            reasoning, manager, stage, [preview_tool],
        )

        trace = manager.metadata.get("reasoning_trace")
        assert trace is None

    @pytest.mark.asyncio
    async def test_react_stage_store_trace_per_stage_override(
        self,
        react_wizard_config: dict[str, Any],
        preview_tool: PreviewConfigTool,
        conversation_manager_pair: tuple[ConversationManager, EchoProvider],
    ) -> None:
        """Per-stage store_trace overrides wizard-level setting."""
        manager, provider = conversation_manager_pair

        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(react_wizard_config)
        reasoning = WizardReasoning(wizard_fsm=wizard_fsm)

        await manager.add_message(role="user", content="Preview the config")

        provider.set_responses([
            tool_call_response("preview_config", {"format": "yaml"}),
            text_response("Here is the preview."),
        ])

        stage = {
            "name": "review", "reasoning": "react",
            "max_iterations": 3, "store_trace": True,
        }
        await self._run_strategy_response(
            reasoning, manager, stage, [preview_tool],
        )

        trace = manager.metadata.get("reasoning_trace")
        assert trace is not None, "stage-level store_trace should override wizard default"

    @pytest.mark.asyncio
    async def test_react_stage_verbose_from_wizard_settings(
        self,
        react_wizard_config: dict[str, Any],
        preview_tool: PreviewConfigTool,
        conversation_manager_pair: tuple[ConversationManager, EchoProvider],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Wizard-level verbose setting is forwarded to ReAct stage."""
        manager, provider = conversation_manager_pair

        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(react_wizard_config)
        reasoning = WizardReasoning(
            wizard_fsm=wizard_fsm, default_verbose=True,
        )

        await manager.add_message(role="user", content="Preview the config")

        provider.set_responses([
            tool_call_response("preview_config", {"format": "yaml"}),
            text_response("Done."),
        ])

        stage = {"name": "review", "reasoning": "react", "max_iterations": 3}

        react_logger = "dataknobs_bots.reasoning.react"
        with caplog.at_level(logging.DEBUG, logger=react_logger):
            response = await self._run_strategy_response(
                reasoning, manager, stage, [preview_tool],
            )

        assert response is not None
        react_debug_msgs = [
            r for r in caplog.records
            if r.name == react_logger and r.levelno == logging.DEBUG
        ]
        assert len(react_debug_msgs) > 0, (
            "verbose=True should produce DEBUG-level logs from ReAct"
        )

    @pytest.mark.asyncio
    async def test_react_stage_verbose_per_stage_override(
        self,
        react_wizard_config: dict[str, Any],
        preview_tool: PreviewConfigTool,
        conversation_manager_pair: tuple[ConversationManager, EchoProvider],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Per-stage verbose overrides wizard-level setting."""
        manager, provider = conversation_manager_pair

        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(react_wizard_config)
        reasoning = WizardReasoning(wizard_fsm=wizard_fsm)

        await manager.add_message(role="user", content="Preview the config")

        provider.set_responses([
            tool_call_response("preview_config", {"format": "yaml"}),
            text_response("Done."),
        ])

        stage = {
            "name": "review", "reasoning": "react",
            "max_iterations": 3, "verbose": True,
        }

        react_logger = "dataknobs_bots.reasoning.react"
        with caplog.at_level(logging.DEBUG, logger=react_logger):
            response = await self._run_strategy_response(
                reasoning, manager, stage, [preview_tool],
            )

        assert response is not None
        react_debug_msgs = [
            r for r in caplog.records
            if r.name == react_logger and r.levelno == logging.DEBUG
        ]
        assert len(react_debug_msgs) > 0, (
            "stage-level verbose=True should produce DEBUG-level logs"
        )

    @pytest.mark.asyncio
    async def test_react_stage_verbose_no_debug_by_default(
        self,
        react_wizard_config: dict[str, Any],
        preview_tool: PreviewConfigTool,
        conversation_manager_pair: tuple[ConversationManager, EchoProvider],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Wizard ReAct stage does NOT log at DEBUG level by default."""
        manager, provider = conversation_manager_pair

        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(react_wizard_config)
        reasoning = WizardReasoning(wizard_fsm=wizard_fsm)

        await manager.add_message(role="user", content="Preview the config")

        provider.set_responses([
            tool_call_response("preview_config", {"format": "yaml"}),
            text_response("Done."),
        ])

        stage = {"name": "review", "reasoning": "react", "max_iterations": 3}

        react_logger = "dataknobs_bots.reasoning.react"
        with caplog.at_level(logging.DEBUG, logger=react_logger):
            await self._run_strategy_response(
                reasoning, manager, stage, [preview_tool],
            )

        react_debug_msgs = [
            r for r in caplog.records
            if r.name == react_logger and r.levelno == logging.DEBUG
        ]
        assert len(react_debug_msgs) == 0, (
            "verbose=False (default) should not produce DEBUG-level logs"
        )


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

        assert reasoning._response._default_tool_reasoning == "react"
        assert reasoning._response._default_max_iterations == 5

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
