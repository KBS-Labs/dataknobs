"""Tests for AdvancedFSM.execute_step_async().

Validates that the async step path produces correct StepResult objects
and properly awaits async pre-tests, transforms, and hooks.
"""

import pytest

from dataknobs_fsm.api.advanced import (
    AdvancedFSM,
    ExecutionHook,
    StepResult,
    create_advanced_fsm,
)


def _simple_config(
    *,
    arc_condition: dict | None = None,
    arc_transform: dict | None = None,
) -> dict:
    """Build a minimal 3-state FSM config (start -> middle -> done)."""
    arcs = [
        {
            "from": "start",
            "to": "middle",
            **({"condition": arc_condition} if arc_condition else {}),
            **({"transform": arc_transform} if arc_transform else {}),
        },
        {"from": "middle", "to": "done"},
    ]
    return {
        "name": "TestFSM",
        "main_network": "main",
        "networks": [
            {
                "name": "main",
                "states": [
                    {"name": "start", "is_start": True},
                    {"name": "middle"},
                    {"name": "done", "is_end": True},
                ],
                "arcs": arcs,
            }
        ],
    }


class TestExecuteStepAsyncBasic:
    """Basic execute_step_async produces correct StepResult."""

    @pytest.mark.asyncio
    async def test_first_step_transitions(self) -> None:
        config = _simple_config()
        advanced = create_advanced_fsm(config)
        context = advanced.create_context({"key": "value"})

        result = await advanced.execute_step_async(context)

        assert isinstance(result, StepResult)
        assert result.success is True
        assert result.to_state == "middle"
        assert result.is_complete is False

    @pytest.mark.asyncio
    async def test_second_step_reaches_end(self) -> None:
        config = _simple_config()
        advanced = create_advanced_fsm(config)
        context = advanced.create_context({"key": "value"})

        await advanced.execute_step_async(context)
        result = await advanced.execute_step_async(context)

        assert result.success is True
        assert result.to_state == "done"
        assert result.is_complete is True


class TestExecuteStepAsyncWithAsyncTransform:
    """Async transforms are awaited and data is transformed."""

    @pytest.mark.asyncio
    async def test_async_transform_applied(self) -> None:
        async def enrich(data: dict, context: object) -> dict:
            result = dict(data)
            result["enriched"] = True
            return result

        config = _simple_config(
            arc_transform={"type": "registered", "name": "enrich"},
        )
        advanced = create_advanced_fsm(config, custom_functions={"enrich": enrich})
        context = advanced.create_context({"value": 1})

        result = await advanced.execute_step_async(context)

        assert result.success is True
        assert result.data_after.get("enriched") is True

    @pytest.mark.asyncio
    async def test_sync_transform_also_works(self) -> None:
        def stamp(data: dict, context: object) -> dict:
            result = dict(data)
            result["stamped"] = True
            return result

        config = _simple_config(
            arc_transform={"type": "registered", "name": "stamp"},
        )
        advanced = create_advanced_fsm(config, custom_functions={"stamp": stamp})
        context = advanced.create_context({"value": 1})

        result = await advanced.execute_step_async(context)

        assert result.success is True
        assert result.data_after.get("stamped") is True


class TestExecuteStepAsyncWithAsyncPretest:
    """Async pre-tests gate transitions correctly."""

    @pytest.mark.asyncio
    async def test_async_pretest_allows(self) -> None:
        async def allow(data: dict, context: object) -> bool:
            return True

        config = _simple_config(
            arc_condition={"type": "registered", "name": "allow"},
        )
        advanced = create_advanced_fsm(config, custom_functions={"allow": allow})
        context = advanced.create_context({})

        result = await advanced.execute_step_async(context)

        assert result.success is True
        assert result.to_state == "middle"

    @pytest.mark.asyncio
    async def test_async_pretest_blocks(self) -> None:
        async def deny(data: dict, context: object) -> bool:
            return False

        config = _simple_config(
            arc_condition={"type": "registered", "name": "deny"},
        )
        advanced = create_advanced_fsm(config, custom_functions={"deny": deny})
        context = advanced.create_context({})

        result = await advanced.execute_step_async(context)

        # No transition occurred — both arcs from start have pre-tests that fail
        # Actually only the first arc has the pretest; the "middle->done" arc
        # is unreachable.  So we stay at start.
        assert result.success is True
        assert result.to_state == "start"
        assert result.transition == "none"


class TestExecuteStepAsyncHooksAwaited:
    """Async hooks are properly awaited."""

    @pytest.mark.asyncio
    async def test_async_hooks_fire(self) -> None:
        events: list[str] = []

        async def on_enter(state_name: str) -> None:
            events.append(f"enter:{state_name}")

        async def on_exit(state_name: str) -> None:
            events.append(f"exit:{state_name}")

        config = _simple_config()
        advanced = create_advanced_fsm(config)
        advanced.set_hooks(ExecutionHook(
            on_state_enter=on_enter,
            on_state_exit=on_exit,
        ))
        context = advanced.create_context({})

        await advanced.execute_step_async(context)

        assert "exit:start" in events
        assert "enter:middle" in events

    @pytest.mark.asyncio
    async def test_sync_hooks_also_work(self) -> None:
        events: list[str] = []

        def on_enter(state_name: str) -> None:
            events.append(f"enter:{state_name}")

        config = _simple_config()
        advanced = create_advanced_fsm(config)
        advanced.set_hooks(ExecutionHook(on_state_enter=on_enter))
        context = advanced.create_context({})

        await advanced.execute_step_async(context)

        assert "enter:middle" in events


class TestExecuteStepAsyncEndState:
    """is_complete=True when reaching an end state."""

    @pytest.mark.asyncio
    async def test_end_state_detected(self) -> None:
        config = _simple_config()
        advanced = create_advanced_fsm(config)
        context = advanced.create_context({})

        # Step 1: start -> middle
        r1 = await advanced.execute_step_async(context)
        assert r1.is_complete is False

        # Step 2: middle -> done (end)
        r2 = await advanced.execute_step_async(context)
        assert r2.is_complete is True


class TestExecuteStepAsyncErrorHandling:
    """Errors produce success=False in StepResult."""

    @pytest.mark.asyncio
    async def test_transform_error_returns_failure(self) -> None:
        async def boom(data: dict, context: object) -> dict:
            raise RuntimeError("transform exploded")

        config = _simple_config(
            arc_transform={"type": "registered", "name": "boom"},
        )
        advanced = create_advanced_fsm(config, custom_functions={"boom": boom})
        context = advanced.create_context({})

        result = await advanced.execute_step_async(context)

        assert result.success is False
        assert "transform exploded" in (result.error or "")
