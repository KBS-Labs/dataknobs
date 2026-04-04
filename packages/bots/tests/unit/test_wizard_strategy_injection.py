"""Tests for per-state strategy injection in wizard stages (Item 64A).

Validates that wizard stages can reference any registered strategy by
name, not just the hardcoded single/react binary.
"""

from typing import Any

import pytest

from dataknobs_bots.reasoning.react import ReActReasoning
from dataknobs_bots.reasoning.simple import SimpleReasoning
from dataknobs_bots.reasoning.wizard import WizardReasoning, WizardState
from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader
from dataknobs_llm.conversations import ConversationManager
from dataknobs_llm.llm.providers.echo import EchoProvider
from dataknobs_llm.testing import text_response, tool_call_response
from dataknobs_llm.tools.base import Tool


# ---------------------------------------------------------------------------
# Test tool
# ---------------------------------------------------------------------------


class EchoTool(Tool):
    """Simple tool that echoes its input."""

    def __init__(self) -> None:
        super().__init__(name="echo_tool", description="Echo input")
        self.call_count = 0

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"message": {"type": "string"}},
        }

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        self.call_count += 1
        return {"echoed": kwargs.get("message", "")}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def echo_tool() -> EchoTool:
    return EchoTool()


# conversation_manager_pair provided by unit/conftest.py


def _make_wizard_config(
    *,
    tool_reasoning: str = "single",
    stages: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a minimal wizard config dict."""
    return {
        "name": "strategy-injection-test",
        "version": "1.0",
        "settings": {"tool_reasoning": tool_reasoning},
        "stages": stages or [
            {
                "name": "start",
                "is_start": True,
                "is_end": True,
                "prompt": "Done",
            },
        ],
    }


# ---------------------------------------------------------------------------
# TestResolveStageStrategy — strategy resolution logic
# ---------------------------------------------------------------------------


class TestResolveStageStrategy:
    """Test _resolve_stage_strategy with various strategy names."""

    def _make_reasoning(
        self, config: dict[str, Any],
    ) -> WizardReasoning:
        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(config)
        tool_reasoning = wizard_fsm.settings.get("tool_reasoning", "single")
        return WizardReasoning(
            wizard_fsm=wizard_fsm,
            default_tool_reasoning=tool_reasoning,
        )

    def test_single_resolves_to_none(self) -> None:
        """reasoning: single → None (direct manager.complete path)."""
        reasoning = self._make_reasoning(_make_wizard_config())
        assert reasoning._resolve_stage_strategy({"reasoning": "single"}) is None

    def test_default_single_resolves_to_none(self) -> None:
        """No per-stage reasoning + default single → None."""
        reasoning = self._make_reasoning(_make_wizard_config())
        assert reasoning._resolve_stage_strategy({}) is None

    def test_react_resolves_to_react_reasoning(self) -> None:
        """reasoning: react → ReActReasoning instance."""
        reasoning = self._make_reasoning(_make_wizard_config())
        strategy = reasoning._resolve_stage_strategy({"reasoning": "react"})
        assert isinstance(strategy, ReActReasoning)

    def test_simple_explicit_resolves_to_simple_reasoning(self) -> None:
        """reasoning: simple → SimpleReasoning instance.

        'simple' is a registered strategy distinct from 'single'.
        'single' means the direct manager.complete() fast path (None).
        'simple' means route through SimpleReasoning.generate().
        """
        reasoning = self._make_reasoning(
            _make_wizard_config(tool_reasoning="react"),
        )
        strategy = reasoning._resolve_stage_strategy({"reasoning": "simple"})
        assert isinstance(strategy, SimpleReasoning)

    def test_default_react_applied_to_stages_without_override(self) -> None:
        """Wizard-level default_tool_reasoning=react applies to stages."""
        reasoning = self._make_reasoning(
            _make_wizard_config(tool_reasoning="react"),
        )
        strategy = reasoning._resolve_stage_strategy({})
        assert isinstance(strategy, ReActReasoning)

    def test_per_stage_override_beats_default(self) -> None:
        """Per-stage reasoning overrides wizard-level default."""
        reasoning = self._make_reasoning(
            _make_wizard_config(tool_reasoning="react"),
        )
        # Stage says single → None, even though default is react
        assert reasoning._resolve_stage_strategy({"reasoning": "single"}) is None

    def test_case_insensitive(self) -> None:
        """Strategy names are case-insensitive."""
        reasoning = self._make_reasoning(_make_wizard_config())
        assert isinstance(
            reasoning._resolve_stage_strategy({"reasoning": "REACT"}),
            ReActReasoning,
        )
        assert reasoning._resolve_stage_strategy({"reasoning": "SINGLE"}) is None

    def test_reasoning_config_forwarded(self) -> None:
        """reasoning_config dict is forwarded to strategy from_config."""
        reasoning = self._make_reasoning(_make_wizard_config())
        strategy = reasoning._resolve_stage_strategy({
            "reasoning": "react",
            "reasoning_config": {"max_iterations": 7},
        })
        assert isinstance(strategy, ReActReasoning)
        assert strategy.max_iterations == 7

    def test_unknown_strategy_raises(self) -> None:
        """Unknown strategy name raises ConfigurationError at resolution."""
        from dataknobs_common.exceptions import ConfigurationError

        reasoning = self._make_reasoning(_make_wizard_config())
        with pytest.raises(ConfigurationError, match="nonexistent"):
            reasoning._resolve_stage_strategy({"reasoning": "nonexistent"})

    def test_wizard_strategy_in_wizard_stage_raises(self) -> None:
        """Using 'reasoning: wizard' inside a wizard stage raises."""
        from dataknobs_common.exceptions import ConfigurationError

        reasoning = self._make_reasoning(_make_wizard_config())
        with pytest.raises(ConfigurationError, match="subflows"):
            reasoning._resolve_stage_strategy({"reasoning": "wizard"})


# ---------------------------------------------------------------------------
# TestStrategyStageResponse — full strategy dispatch in wizard context
# ---------------------------------------------------------------------------


class TestStrategyStageResponse:
    """Test _strategy_stage_response delegates correctly."""

    async def _run(
        self,
        reasoning: WizardReasoning,
        manager: Any,
        stage: dict[str, Any],
        tools: list[Any],
    ) -> Any:
        strategy = reasoning._resolve_stage_strategy(stage)
        assert strategy is not None
        state = WizardState(
            current_stage=stage.get("name", "test"), data={},
        )
        return await reasoning._strategy_stage_response(
            strategy, manager, "Test prompt", stage, state, tools,
        )

    @pytest.mark.asyncio
    async def test_react_via_registry_executes_tool(
        self,
        echo_tool: EchoTool,
        conversation_manager_pair: tuple[ConversationManager, EchoProvider],
    ) -> None:
        """ReAct strategy resolved from registry executes tools."""
        manager, provider = conversation_manager_pair
        await manager.add_message(role="user", content="echo hello")

        provider.set_responses([
            tool_call_response("echo_tool", {"message": "hello"}),
            text_response("Echoed: hello"),
        ])

        config = _make_wizard_config()
        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(config)
        reasoning = WizardReasoning(wizard_fsm=wizard_fsm)

        stage = {"name": "test", "reasoning": "react", "max_iterations": 3}
        response = await self._run(reasoning, manager, stage, [echo_tool])

        assert echo_tool.call_count == 1
        assert response.content == "Echoed: hello"

    @pytest.mark.asyncio
    async def test_react_without_tools_still_dispatches(
        self,
        conversation_manager_pair: tuple[ConversationManager, EchoProvider],
    ) -> None:
        """Strategy dispatches even without tools.

        ReAct handles the no-tools case internally by falling back to
        simple completion — the wizard dispatch does not gate on tools.
        """
        manager, provider = conversation_manager_pair
        await manager.add_message(role="user", content="hello")

        provider.set_responses([text_response("Hi there")])

        config = _make_wizard_config()
        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(config)
        reasoning = WizardReasoning(wizard_fsm=wizard_fsm)

        stage = {"name": "greet", "reasoning": "react"}
        strategy = reasoning._resolve_stage_strategy(stage)
        assert strategy is not None

        # Dispatch with no tools — strategy still runs, ReAct falls
        # back to simple completion internally
        state = WizardState(current_stage="greet", data={})
        response = await reasoning._strategy_stage_response(
            strategy, manager, "Test prompt", stage, state, tools=[],
        )
        assert response.content == "Hi there"


# ---------------------------------------------------------------------------
# TestStrategyNameValidation — config-time warnings
# ---------------------------------------------------------------------------


class TestStrategyNameValidation:
    """Test that unknown strategy names produce warnings."""

    def test_valid_names_no_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Valid strategy names produce no warnings."""
        import logging

        config = _make_wizard_config(
            tool_reasoning="react",
            stages=[
                {
                    "name": "start",
                    "is_start": True,
                    "is_end": True,
                    "prompt": "Done",
                    "reasoning": "simple",
                },
            ],
        )
        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(config)

        with caplog.at_level(logging.WARNING):
            WizardReasoning(
                wizard_fsm=wizard_fsm,
                default_tool_reasoning="react",
            )

        warning_msgs = [
            r for r in caplog.records
            if "Unknown reasoning strategy" in r.message
        ]
        assert len(warning_msgs) == 0

    def test_unknown_default_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        """Unknown default_tool_reasoning produces a warning."""
        import logging

        config = _make_wizard_config(tool_reasoning="nonexistent")
        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(config)

        with caplog.at_level(logging.WARNING):
            WizardReasoning(
                wizard_fsm=wizard_fsm,
                default_tool_reasoning="nonexistent",
            )

        warning_msgs = [
            r for r in caplog.records
            if "Unknown reasoning strategy" in r.message
        ]
        assert len(warning_msgs) >= 1

    def test_unknown_stage_reasoning_warns(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Unknown per-stage reasoning value produces a warning."""
        import logging

        config = _make_wizard_config(stages=[
            {
                "name": "bad",
                "is_start": True,
                "is_end": True,
                "prompt": "Done",
                "reasoning": "totally_fake",
            },
        ])
        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(config)

        with caplog.at_level(logging.WARNING):
            WizardReasoning(wizard_fsm=wizard_fsm)

        warning_msgs = [
            r for r in caplog.records
            if "Unknown reasoning strategy" in r.message
        ]
        assert len(warning_msgs) >= 1
        assert "totally_fake" in warning_msgs[0].message
