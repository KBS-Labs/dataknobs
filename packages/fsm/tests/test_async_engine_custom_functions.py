"""Tests for AsyncExecutionEngine custom_functions support."""

import asyncio

import pytest

from dataknobs_fsm.config.builder import FSMBuilder
from dataknobs_fsm.config.loader import ConfigLoader
from dataknobs_fsm.execution.async_engine import AsyncExecutionEngine
from dataknobs_fsm.execution.context import ExecutionContext


def _build_fsm_with_transform(transform_name: str, transform_func):
    """Build an FSM that references a transform registered via builder."""
    config = {
        "name": "custom_func_fsm",
        "main_network": "main",
        "networks": [
            {
                "name": "main",
                "states": [
                    {"name": "start", "is_start": True},
                    {"name": "end", "is_end": True},
                ],
                "arcs": [
                    {
                        "from": "start",
                        "to": "end",
                        "name": "go",
                        "transform": {
                            "type": "registered",
                            "name": transform_name,
                        },
                    }
                ],
            }
        ],
    }
    loader = ConfigLoader()
    fsm_config = loader.load_from_dict(config)
    builder = FSMBuilder()
    builder.register_function(transform_name, transform_func)
    return builder.build(fsm_config)


def _build_fsm_with_pretest(pretest_lambda: str):
    """Build an FSM with an arc gated by an inline lambda pre-test."""
    config = {
        "name": "pretest_fsm",
        "main_network": "main",
        "networks": [
            {
                "name": "main",
                "states": [
                    {"name": "start", "is_start": True},
                    {"name": "end", "is_end": True},
                ],
                "arcs": [
                    {
                        "from": "start",
                        "to": "end",
                        "name": "go",
                        "pre_test": {"test": pretest_lambda},
                    }
                ],
            }
        ],
    }
    loader = ConfigLoader()
    fsm_config = loader.load_from_dict(config)
    builder = FSMBuilder()
    return builder.build(fsm_config)


class TestCustomTransformExecutes:
    """Test that custom_functions transforms are visible to the engine."""

    @pytest.mark.asyncio
    async def test_custom_transform_executes(self):
        """Custom transform passed via custom_functions runs during transition."""

        def double_value(data, context):
            result = dict(data) if isinstance(data, dict) else {}
            result["value"] = result.get("value", 0) * 2
            return result

        # Register the function with the builder so the config validates,
        # AND pass it as a custom_function to the engine.
        fsm = _build_fsm_with_transform("double_value", double_value)
        engine = AsyncExecutionEngine(
            fsm, custom_functions={"double_value": double_value}
        )

        ctx = ExecutionContext()
        ctx.data = {"value": 5}
        success, _ = await engine.execute(ctx)

        assert success
        assert ctx.data.get("value") == 10

    @pytest.mark.asyncio
    async def test_async_custom_transform(self):
        """Async custom transform is properly awaited."""

        async def async_transform(data, context):
            await asyncio.sleep(0)
            result = dict(data) if isinstance(data, dict) else {}
            result["transformed"] = True
            return result

        fsm = _build_fsm_with_transform("async_transform", async_transform)
        engine = AsyncExecutionEngine(
            fsm, custom_functions={"async_transform": async_transform}
        )

        ctx = ExecutionContext()
        ctx.data = {"input": "test"}
        success, _ = await engine.execute(ctx)

        assert success
        assert ctx.data.get("transformed") is True


class TestCustomPretestGatesTransition:
    """Test that custom pre-test functions gate arc selection."""

    @pytest.mark.asyncio
    async def test_custom_pretest_gates_transition(self):
        """An inline lambda pre-test that returns True allows transition."""
        fsm = _build_fsm_with_pretest("lambda state: True")
        engine = AsyncExecutionEngine(fsm)

        ctx = ExecutionContext()
        ctx.data = {"key": "value"}
        success, _ = await engine.execute(ctx)

        assert success
        assert ctx.current_state == "end"

    @pytest.mark.asyncio
    async def test_custom_pretest_blocks_transition(self):
        """An inline lambda pre-test that returns False blocks transition."""
        fsm = _build_fsm_with_pretest("lambda state: False")
        engine = AsyncExecutionEngine(fsm)

        ctx = ExecutionContext()
        ctx.data = {"key": "value"}
        success, _ = await engine.execute(ctx)

        # Transition should not happen (blocked by pre-test)
        assert not success
        assert ctx.current_state == "start"


class TestNoCustomFunctionsWorks:
    """Test that engine works normally without custom functions."""

    @pytest.mark.asyncio
    async def test_no_custom_functions_works(self):
        config = {
            "name": "plain_fsm",
            "main_network": "main",
            "networks": [
                {
                    "name": "main",
                    "states": [
                        {"name": "start", "is_start": True},
                        {"name": "end", "is_end": True},
                    ],
                    "arcs": [{"from": "start", "to": "end", "name": "go"}],
                }
            ],
        }
        loader = ConfigLoader()
        fsm_config = loader.load_from_dict(config)
        builder = FSMBuilder()
        fsm = builder.build(fsm_config)

        engine = AsyncExecutionEngine(fsm)

        ctx = ExecutionContext()
        ctx.data = {"key": "value"}
        success, _ = await engine.execute(ctx)

        assert success
        assert ctx.current_state == "end"
