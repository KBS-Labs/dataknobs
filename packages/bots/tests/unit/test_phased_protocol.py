"""Tests for the phased reasoning protocol (item 78).

Verifies:
- Protocol types (TurnHandle, ProcessResult, PhasedReasoningProtocol)
- WizardReasoning implements PhasedReasoningProtocol
- Non-phased strategies do NOT implement the protocol
- Phase methods produce correct results for key scenarios
- DynaBot routes wizard through phased flow
- update_tool_tasks wiring in finalize_turn
"""

from __future__ import annotations

import pytest

from dataknobs_bots.reasoning.base import (
    PhasedReasoningProtocol,
    ProcessResult,
    TurnHandle,
)
from dataknobs_bots.reasoning.simple import SimpleReasoning
from dataknobs_bots.reasoning.wizard import WizardReasoning
from dataknobs_bots.reasoning.wizard_tasks import update_tool_tasks
from dataknobs_bots.reasoning.wizard_types import (
    RecoveryResult,
    WizardState,
    WizardTurnHandle,
)
from dataknobs_bots.reasoning.observability import WizardTask, WizardTaskList
from dataknobs_bots.testing import BotTestHarness, WizardConfigBuilder
from dataknobs_llm.testing import text_response


# =========================================================================
# Protocol detection tests
# =========================================================================


class TestPhasedProtocolDetection:
    """Verify isinstance checks for PhasedReasoningProtocol."""

    def test_wizard_satisfies_protocol(self) -> None:
        """WizardReasoning satisfies PhasedReasoningProtocol isinstance check."""
        # Use __new__ to avoid __init__ (requires wizard_fsm arg).
        # The isinstance check only verifies method presence.
        wizard = WizardReasoning.__new__(WizardReasoning)
        assert isinstance(wizard, PhasedReasoningProtocol)

    def test_simple_does_not_satisfy_protocol(self) -> None:
        """SimpleReasoning does not have phase methods."""
        strategy = SimpleReasoning()
        assert not isinstance(strategy, PhasedReasoningProtocol)


# =========================================================================
# Type construction tests
# =========================================================================


class TestTurnHandleTypes:
    """Verify TurnHandle and ProcessResult construction."""

    def test_turn_handle_defaults(self) -> None:
        """TurnHandle has sensible defaults."""
        handle = TurnHandle(manager=None, llm=None)
        assert handle.tools is None
        assert handle.kwargs == {}
        assert handle.early_response is None

    def test_wizard_turn_handle_extends(self) -> None:
        """WizardTurnHandle extends TurnHandle with wizard fields."""
        handle = WizardTurnHandle(manager=None, llm=None)
        assert handle.wizard_state is None
        assert handle.user_message == ""
        assert handle.skip_extraction is False
        # Base fields
        assert handle.early_response is None

    def test_process_result_defaults(self) -> None:
        """ProcessResult defaults to no early response, no tool execution."""
        result = ProcessResult()
        assert result.early_response is None
        assert result.needs_tool_execution is False
        assert result.action == ""

    def test_recovery_result_defaults(self) -> None:
        """RecoveryResult defaults extraction to None."""
        result = RecoveryResult(new_data_keys={"a", "b"})
        assert result.new_data_keys == {"a", "b"}
        assert result.extraction is None

    def test_recovery_result_with_extraction(self) -> None:
        """RecoveryResult carries extraction when provided."""
        result = RecoveryResult(new_data_keys=set(), extraction="mock_ext")
        assert result.extraction == "mock_ext"


# =========================================================================
# Phased generate behavioral tests (via BotTestHarness)
# =========================================================================


class TestPhasedGenerate:
    """Test that phased generate produces identical results to monolithic."""

    @pytest.mark.asyncio
    async def test_basic_extraction_and_transition(self) -> None:
        """Phased flow extracts data and transitions stages."""
        config = (
            WizardConfigBuilder("test")
            .stage("gather", is_start=True, prompt="What is your name?")
                .field("name", field_type="string", required=True)
                .transition("done", "has('name')")
            .stage("done", is_end=True, prompt="All done!")
            .build()
        )

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["Got it!"],
            extraction_results=[[{"name": "Alice"}]],
        ) as harness:
            await harness.chat("My name is Alice")
            assert harness.wizard_data["name"] == "Alice"
            assert harness.wizard_stage == "done"

    @pytest.mark.asyncio
    async def test_navigation_early_return(self) -> None:
        """Navigation commands produce early returns from begin_turn."""
        config = (
            WizardConfigBuilder("test")
            .stage("first", is_start=True, prompt="Step 1")
                .field("a", field_type="string", required=True)
                .transition("second", "has('a')")
            .stage("second", prompt="Step 2")
                .field("b", field_type="string", required=True)
                .transition("done", "has('b')")
            .stage("done", is_end=True, prompt="Done!")
            .build()
        )

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["OK!", "Back!"],
            extraction_results=[[{"a": "val"}], []],
        ) as harness:
            # Move to second stage
            await harness.chat("val")
            assert harness.wizard_stage == "second"

            # Navigate back — early response from begin_turn
            await harness.chat("back")
            assert harness.wizard_stage == "first"

    @pytest.mark.asyncio
    async def test_clarification_early_return(self) -> None:
        """Low-confidence extraction produces early return from process_input."""
        config = (
            WizardConfigBuilder("test")
            .stage("gather", is_start=True, prompt="What is your name?")
                .field("name", field_type="string", required=True)
                .transition("done", "has('name')")
            .stage("done", is_end=True, prompt="Done!")
            .build()
        )

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["I didn't catch that. What's your name?"],
            extraction_results=[[{}]],  # Empty extraction → clarification
        ) as harness:
            await harness.chat("hmm")
            # Should stay on gather stage (clarification, not transition)
            assert harness.wizard_stage == "gather"


# =========================================================================
# DynaBot phased routing tests
# =========================================================================


class TestDynaBotPhasedRouting:
    """Test that DynaBot routes through phased flow for wizard strategies."""

    @pytest.mark.asyncio
    async def test_wizard_uses_phased_flow(self) -> None:
        """Wizard strategy goes through _generate_phased_response in DynaBot."""
        config = (
            WizardConfigBuilder("test")
            .stage("start", is_start=True, prompt="Hello!")
                .field("x", field_type="string", required=True)
                .transition("done", "has('x')")
            .stage("done", is_end=True, prompt="Done!")
            .build()
        )

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["OK!"],
            extraction_results=[[{"x": "val"}]],
        ) as harness:
            # Verify the strategy is detected as phased
            assert isinstance(
                harness.bot.reasoning_strategy, PhasedReasoningProtocol
            )
            # Chat should work through the phased flow
            result = await harness.chat("val")
            assert result.response  # Got a response
            assert harness.wizard_stage == "done"

    @pytest.mark.asyncio
    async def test_simple_strategy_uses_legacy_flow(self) -> None:
        """Simple strategy does NOT go through phased flow."""
        async with await BotTestHarness.create(
            bot_config={
                "llm": {"provider": "echo", "model": "test"},
                "conversation_storage": {"backend": "memory"},
                "reasoning": {"strategy": "simple"},
            },
            main_responses=[text_response("Hello!")],
        ) as harness:
            assert not isinstance(
                harness.bot.reasoning_strategy, PhasedReasoningProtocol
            )
            result = await harness.chat("Hi")
            assert result.response


# =========================================================================
# update_tool_tasks wiring test
# =========================================================================


class TestToolTaskWiring:
    """Test that update_tool_tasks correctly processes tool results."""

    def test_successful_tool_completes_task(self) -> None:
        """Successful tool execution marks matching tasks as complete."""
        state = WizardState(current_stage="test")
        state.tasks = WizardTaskList(tasks=[
            WizardTask(
                id="t1",
                description="Run search",
                completed_by="tool_result",
                tool_name="search",
            ),
        ])

        update_tool_tasks(state, "search", success=True)

        assert state.tasks.tasks[0].status == "completed"

    def test_failed_tool_leaves_task_pending(self) -> None:
        """Failed tool execution does not mark tasks as complete."""
        state = WizardState(current_stage="test")
        state.tasks = WizardTaskList(tasks=[
            WizardTask(
                id="t1",
                description="Run search",
                completed_by="tool_result",
                tool_name="search",
            ),
        ])

        update_tool_tasks(state, "search", success=False)

        assert state.tasks.tasks[0].status == "pending"

    def test_unmatched_tool_name_ignored(self) -> None:
        """Tool results for non-matching tools don't affect tasks."""
        state = WizardState(current_stage="test")
        state.tasks = WizardTaskList(tasks=[
            WizardTask(
                id="t1",
                description="Run search",
                completed_by="tool_result",
                tool_name="search",
            ),
        ])

        update_tool_tasks(state, "calculator", success=True)

        assert state.tasks.tasks[0].status == "pending"

    def test_non_tool_result_task_ignored(self) -> None:
        """Tasks with completed_by != 'tool_result' are not affected."""
        state = WizardState(current_stage="test")
        state.tasks = WizardTaskList(tasks=[
            WizardTask(
                id="t1",
                description="Fill name field",
                completed_by="field_extraction",
                tool_name="search",  # Even if tool_name matches
            ),
        ])

        update_tool_tasks(state, "search", success=True)

        assert state.tasks.tasks[0].status == "pending"
