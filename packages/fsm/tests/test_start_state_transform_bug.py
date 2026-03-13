"""Tests for start state transform execution bug.

Bug: Start state transforms are silently skipped because
ContextFactory.create_context() pre-sets current_state, and execution
methods guard transforms behind `if not context.current_state:` which
is always false.

Affected paths:
- AsyncExecutionEngine.execute()
- ExecutionEngine.execute()
- AdvancedFSM.execute_step_sync()
- AdvancedFSM.execute_step_async()
"""

import pytest

from dataknobs_fsm.api.advanced import AdvancedFSM
from dataknobs_fsm.api.async_simple import AsyncSimpleFSM
from dataknobs_fsm.api.simple import SimpleFSM


def _start_transform(state):
    """Transform that marks data as computed."""
    return {**state.data, "computed": True}


def _check_computed(data, context):
    """Pre-test that requires computed == True."""
    return data.get("computed") is True


def _counting_start_transform(state):
    """Transform that counts invocations via a list in data."""
    count = state.data.get("transform_count", 0) + 1
    return {**state.data, "computed": True, "transform_count": count}


FSM_CONFIG = {
    "name": "start_transform_test",
    "main_network": "main",
    "networks": [
        {
            "name": "main",
            "states": [
                {
                    "name": "start",
                    "is_start": True,
                    "functions": {"transform": "start_transform"},
                },
                {"name": "end", "is_end": True},
            ],
            "arcs": [
                {
                    "from": "start",
                    "to": "end",
                    "name": "to_end",
                    "pre_test": "check_computed",
                }
            ],
        }
    ],
}

CUSTOM_FUNCTIONS = {
    "start_transform": _start_transform,
    "check_computed": _check_computed,
}


class TestStartStateTransformBug:
    """Start state transforms must execute even when ContextFactory pre-sets state."""

    def test_start_transform_via_simple_fsm(self):
        """SimpleFSM should execute start state transform."""
        fsm = SimpleFSM(FSM_CONFIG, custom_functions=CUSTOM_FUNCTIONS)
        result = fsm.process({"input": "data"})
        assert result["success"] is True, (
            "FSM should reach end state — start transform should set computed=True"
        )

    @pytest.mark.asyncio
    async def test_start_transform_via_async_simple_fsm(self):
        """AsyncSimpleFSM should execute start state transform."""
        fsm = AsyncSimpleFSM(FSM_CONFIG, custom_functions=CUSTOM_FUNCTIONS)
        result = await fsm.process({"input": "data"})
        assert result["success"] is True, (
            "FSM should reach end state — start transform should set computed=True"
        )

    @pytest.mark.asyncio
    async def test_start_transform_via_advanced_fsm_step_async(self):
        """AdvancedFSM.execute_step_async() should execute start state transform."""
        advanced = AdvancedFSM(FSM_CONFIG, custom_functions=CUSTOM_FUNCTIONS)
        context = advanced.create_context({"input": "data"})
        result = await advanced.execute_step_async(context)

        assert result.success is True, (
            f"Step should succeed — start transform should set computed=True. "
            f"Error: {result.error}"
        )
        # Should have transitioned from start to end
        assert result.to_state == "end"

    def test_start_transform_via_advanced_fsm_step_sync(self):
        """AdvancedFSM.execute_step_sync() should execute start state transform."""
        advanced = AdvancedFSM(FSM_CONFIG, custom_functions=CUSTOM_FUNCTIONS)
        context = advanced.create_context({"input": "data"})
        result = advanced.execute_step_sync(context)

        assert result.success is True, (
            f"Step should succeed — start transform should set computed=True. "
            f"Error: {result.error}"
        )
        assert result.to_state == "end"


class TestStartTransformNotDuplicated:
    """Start state transform must execute exactly once."""

    @pytest.mark.asyncio
    async def test_transform_executes_exactly_once(self):
        """Counting transform on start state should be called exactly once."""
        counting_config = {
            "name": "counting_test",
            "main_network": "main",
            "networks": [
                {
                    "name": "main",
                    "states": [
                        {
                            "name": "start",
                            "is_start": True,
                            "functions": {"transform": "counting_transform"},
                        },
                        {"name": "end", "is_end": True},
                    ],
                    "arcs": [
                        {"from": "start", "to": "end", "name": "to_end"},
                    ],
                }
            ],
        }
        functions = {
            "counting_transform": _counting_start_transform,
        }
        fsm = AsyncSimpleFSM(counting_config, custom_functions=functions)
        result = await fsm.process({"input": "data"})
        assert result["success"] is True
        assert result["data"].get("transform_count") == 1, (
            f"Start transform should execute exactly once, got count="
            f"{result['data'].get('transform_count')}"
        )


class TestIntermediateTransformsStillWork:
    """Regression: transforms on intermediate states must still work."""

    @pytest.mark.asyncio
    async def test_intermediate_transform(self):
        """Transform on non-start state should still execute."""
        call_count = 0

        def intermediate_transform(state):
            nonlocal call_count
            call_count += 1
            return {**state.data, "processed": True}

        config = {
            "name": "intermediate_test",
            "main_network": "main",
            "networks": [
                {
                    "name": "main",
                    "states": [
                        {"name": "start", "is_start": True},
                        {
                            "name": "middle",
                            "functions": {"transform": "mid_transform"},
                        },
                        {"name": "end", "is_end": True},
                    ],
                    "arcs": [
                        {"from": "start", "to": "middle", "name": "to_mid"},
                        {"from": "middle", "to": "end", "name": "to_end"},
                    ],
                }
            ],
        }
        fsm = AsyncSimpleFSM(
            config, custom_functions={"mid_transform": intermediate_transform}
        )
        result = await fsm.process({"input": "data"})
        assert result["success"] is True
        assert call_count == 1
        assert result["data"].get("processed") is True
