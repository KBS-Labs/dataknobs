"""Tests verifying that AdvancedFSM.step() returns StepResult."""

import pytest

from dataknobs_fsm.api.advanced import (
    AdvancedFSM,
    ExecutionMode,
    StepResult,
    create_advanced_fsm,
)
from dataknobs_fsm.config.builder import FSMBuilder
from dataknobs_fsm.config.loader import ConfigLoader
from dataknobs_fsm.core.data_modes import DataHandlingMode


def _build_simple_fsm():
    """Build a simple 3-state FSM."""
    config = {
        "name": "step_result_fsm",
        "main_network": "main",
        "networks": [
            {
                "name": "main",
                "states": [
                    {"name": "start", "is_start": True},
                    {"name": "middle"},
                    {"name": "end", "is_end": True},
                ],
                "arcs": [
                    {"from": "start", "to": "middle", "name": "proceed"},
                    {"from": "middle", "to": "end", "name": "finish"},
                ],
            }
        ],
    }
    loader = ConfigLoader()
    fsm_config = loader.load_from_dict(config)
    builder = FSMBuilder()
    return builder.build(fsm_config)


@pytest.fixture
def simple_fsm():
    return _build_simple_fsm()


@pytest.fixture
def advanced(simple_fsm):
    return AdvancedFSM(simple_fsm, ExecutionMode.STEP_BY_STEP)


class TestStepReturnsStepResult:
    """Verify step() returns StepResult, not StateInstance|None."""

    @pytest.mark.asyncio
    async def test_step_returns_step_result_type(self, advanced):
        async with advanced.execution_context({"key": "value"}) as ctx:
            result = await advanced.step(ctx)
            assert isinstance(result, StepResult)

    @pytest.mark.asyncio
    async def test_step_success_fields(self, advanced):
        async with advanced.execution_context({"key": "value"}) as ctx:
            result = await advanced.step(ctx)

            assert result.success is True
            assert result.from_state == "start"
            assert result.to_state == "middle"
            assert result.transition != "none"
            assert result.duration >= 0.0

    @pytest.mark.asyncio
    async def test_step_no_transition_at_end(self, advanced):
        """At end state, step returns transition='none'."""
        async with advanced.execution_context({"key": "value"}) as ctx:
            # Walk to end
            while True:
                result = await advanced.step(ctx)
                if result.is_complete or result.transition == "none":
                    break

            # One more step at end state
            result = await advanced.step(ctx)
            assert result.transition == "none"
            assert result.success is True

    @pytest.mark.asyncio
    async def test_step_is_complete(self, advanced):
        """is_complete is True when arriving at an end state."""
        async with advanced.execution_context({"key": "value"}) as ctx:
            results = []
            for _ in range(10):
                result = await advanced.step(ctx)
                results.append(result)
                if result.is_complete or result.transition == "none":
                    break

            # At least one result should mark completion
            assert any(r.is_complete for r in results)


class TestStepBreakpoint:
    """Test at_breakpoint field in StepResult."""

    @pytest.mark.asyncio
    async def test_step_at_breakpoint(self, advanced):
        advanced.add_breakpoint("middle")

        async with advanced.execution_context({"key": "value"}) as ctx:
            result = await advanced.step(ctx)

            # Should have transitioned to middle and flagged breakpoint
            assert result.to_state == "middle"
            assert result.at_breakpoint is True


class TestTraceAndProfileExecution:
    """Verify trace_execution and profile_execution still work."""

    @pytest.mark.asyncio
    async def test_trace_execution_works(self, simple_fsm):
        advanced = AdvancedFSM(simple_fsm, ExecutionMode.TRACE)
        trace = await advanced.trace_execution({"key": "value"})
        assert isinstance(trace, list)

    @pytest.mark.asyncio
    async def test_profile_execution_works(self, simple_fsm):
        advanced = AdvancedFSM(simple_fsm, ExecutionMode.PROFILE)
        profile = await advanced.profile_execution({"key": "value"})
        assert isinstance(profile, dict)
        assert "total_time" in profile
        assert "transitions" in profile


class TestRunUntilBreakpoint:
    """Test run_until_breakpoint returns StepResult."""

    @pytest.mark.asyncio
    async def test_run_until_breakpoint_returns_step_result(self, advanced):
        advanced.add_breakpoint("middle")

        async with advanced.execution_context({"key": "value"}) as ctx:
            result = await advanced.run_until_breakpoint(ctx)
            assert result is not None
            assert isinstance(result, StepResult)
            assert result.at_breakpoint is True
