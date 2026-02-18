"""Tests for AsyncExecutionEngine hook support."""

import asyncio

import pytest

from dataknobs_fsm.config.builder import FSMBuilder
from dataknobs_fsm.config.loader import ConfigLoader
from dataknobs_fsm.execution.async_engine import AsyncExecutionEngine
from dataknobs_fsm.execution.context import ExecutionContext


def _build_simple_fsm():
    """Build a simple 3-state FSM for hook testing."""
    config = {
        "name": "hook_test_fsm",
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


class TestPreTransitionHook:
    """Test pre-transition hooks fire before transitions."""

    @pytest.mark.asyncio
    async def test_pre_transition_hook_fires(self, simple_fsm):
        calls: list[str] = []

        def pre_hook(context, arc):
            calls.append(f"pre:{arc.target_state}")

        engine = AsyncExecutionEngine(simple_fsm)
        engine.add_pre_transition_hook(pre_hook)

        ctx = ExecutionContext()
        await engine.execute(ctx)

        assert len(calls) >= 1
        assert calls[0] == "pre:middle"

    @pytest.mark.asyncio
    async def test_async_pre_hook_awaited(self, simple_fsm):
        calls: list[str] = []

        async def async_pre_hook(context, arc):
            await asyncio.sleep(0)
            calls.append(f"async_pre:{arc.target_state}")

        engine = AsyncExecutionEngine(simple_fsm)
        engine.add_pre_transition_hook(async_pre_hook)

        ctx = ExecutionContext()
        await engine.execute(ctx)

        assert len(calls) >= 1
        assert calls[0] == "async_pre:middle"


class TestPostTransitionHook:
    """Test post-transition hooks fire after transitions."""

    @pytest.mark.asyncio
    async def test_post_transition_hook_fires(self, simple_fsm):
        calls: list[str] = []

        def post_hook(context, arc):
            calls.append(f"post:{arc.target_state}")

        engine = AsyncExecutionEngine(simple_fsm)
        engine.add_post_transition_hook(post_hook)

        ctx = ExecutionContext()
        await engine.execute(ctx)

        assert len(calls) >= 1
        assert calls[0] == "post:middle"

    @pytest.mark.asyncio
    async def test_async_post_hook_awaited(self, simple_fsm):
        calls: list[str] = []

        async def async_post_hook(context, arc):
            await asyncio.sleep(0)
            calls.append(f"async_post:{arc.target_state}")

        engine = AsyncExecutionEngine(simple_fsm)
        engine.add_post_transition_hook(async_post_hook)

        ctx = ExecutionContext()
        await engine.execute(ctx)

        assert len(calls) >= 1
        assert calls[0] == "async_post:middle"


class TestErrorHook:
    """Test error hooks fire on transition failures."""

    @pytest.mark.asyncio
    async def test_error_hook_fires_on_failure(self):
        """Error hook is called when a transform raises."""
        calls: list[str] = []

        def error_hook(context, arc, exc):
            calls.append(f"error:{type(exc).__name__}")

        def bad_transform(data, context):
            raise RuntimeError("boom")

        config = {
            "name": "error_hook_fsm",
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
                                "name": "bad_transform",
                            },
                        }
                    ],
                }
            ],
        }
        loader = ConfigLoader()
        fsm_config = loader.load_from_dict(config)
        builder = FSMBuilder()
        builder.register_function("bad_transform", bad_transform)
        fsm = builder.build(fsm_config)

        engine = AsyncExecutionEngine(fsm)
        engine.add_error_hook(error_hook)

        ctx = ExecutionContext()
        success, _ = await engine.execute(ctx)

        assert not success
        assert len(calls) >= 1
        assert calls[0] == "error:RuntimeError"


class TestHooksDisabled:
    """Test that enable_hooks=False suppresses all hooks."""

    @pytest.mark.asyncio
    async def test_hooks_disabled(self, simple_fsm):
        calls: list[str] = []

        def pre_hook(context, arc):
            calls.append("pre")

        def post_hook(context, arc):
            calls.append("post")

        engine = AsyncExecutionEngine(simple_fsm, enable_hooks=False)
        engine.add_pre_transition_hook(pre_hook)
        engine.add_post_transition_hook(post_hook)

        ctx = ExecutionContext()
        await engine.execute(ctx)

        assert len(calls) == 0


class TestMultipleHooksOrdering:
    """Test that multiple hooks fire in registration order."""

    @pytest.mark.asyncio
    async def test_multiple_hooks_fire_in_order(self, simple_fsm):
        calls: list[int] = []

        engine = AsyncExecutionEngine(simple_fsm)
        for i in range(3):
            engine.add_pre_transition_hook(lambda ctx, arc, idx=i: calls.append(idx))

        ctx = ExecutionContext()
        await engine.execute(ctx)

        # First three calls should preserve order from first transition
        assert calls[:3] == [0, 1, 2]


class TestHookErrorDoesNotBreakExecution:
    """Test that a raising hook does not stop FSM execution."""

    @pytest.mark.asyncio
    async def test_hook_error_does_not_break_execution(self, simple_fsm):
        def bad_hook(context, arc):
            raise ValueError("hook error")

        engine = AsyncExecutionEngine(simple_fsm)
        engine.add_pre_transition_hook(bad_hook)

        ctx = ExecutionContext()
        success, _ = await engine.execute(ctx)

        # Execution should succeed despite hook error
        assert success


class TestStatisticsIncludeHooks:
    """Test get_statistics reports hooks_enabled."""

    def test_statistics_hooks_enabled(self, simple_fsm):
        engine = AsyncExecutionEngine(simple_fsm, enable_hooks=True)
        stats = engine.get_statistics()
        assert stats["hooks_enabled"] is True

    def test_statistics_hooks_disabled(self, simple_fsm):
        engine = AsyncExecutionEngine(simple_fsm, enable_hooks=False)
        stats = engine.get_statistics()
        assert stats["hooks_enabled"] is False
