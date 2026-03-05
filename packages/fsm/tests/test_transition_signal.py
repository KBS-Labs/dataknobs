"""Tests for the _transition_signal mechanism in AdvancedFSM.

Validates that transforms can set ``_transition_signal`` in context.data
to pre-select the next arc, bypassing normal condition evaluation.
"""

from __future__ import annotations

import pytest

from dataknobs_fsm.api.advanced import AdvancedFSM, create_advanced_fsm
from dataknobs_fsm.functions.base import FunctionContext


def _make_config(
    *,
    arcs: list[dict] | None = None,
    states: list[dict] | None = None,
) -> dict:
    """Build a minimal FSM config.

    Defaults to three states (start, middle, end) with two arcs
    (start->middle, start->end) so that transition selection matters.
    """
    if states is None:
        states = [
            {"name": "start", "is_start": True},
            {"name": "middle"},
            {"name": "end", "is_end": True},
        ]
    if arcs is None:
        arcs = [
            {"from": "start", "to": "middle"},
            {"from": "start", "to": "end"},
            {"from": "middle", "to": "end"},
        ]
    return {
        "name": "test_signal",
        "version": "1.0",
        "main_network": "main",
        "networks": [
            {
                "name": "main",
                "states": states,
                "arcs": arcs,
            }
        ],
    }


class TestTransitionSignalSync:
    """Sync execution path honors _transition_signal."""

    def test_no_signal_uses_normal_evaluation(self) -> None:
        """Without a signal, the first matching arc is selected (default)."""
        config = _make_config()
        fsm = create_advanced_fsm(config)
        context = fsm.create_context({"value": 1})

        result = fsm.execute_step_sync(context)

        assert result.success
        # First arc (start->middle) is selected by default
        assert result.to_state == "middle"

    def test_signal_selects_target_arc(self) -> None:
        """_transition_signal causes the matching arc to be selected."""
        config = _make_config()
        fsm = create_advanced_fsm(config)
        context = fsm.create_context({"value": 1, "_transition_signal": "end"})

        result = fsm.execute_step_sync(context)

        assert result.success
        assert result.to_state == "end"

    def test_signal_consumed_after_use(self) -> None:
        """_transition_signal is popped from data after being used."""
        config = _make_config()
        fsm = create_advanced_fsm(config)
        context = fsm.create_context({"value": 1, "_transition_signal": "end"})

        fsm.execute_step_sync(context)

        assert "_transition_signal" not in context.data

    def test_unmatched_signal_falls_through(self) -> None:
        """Signal that doesn't match any arc falls through to normal eval."""
        config = _make_config()
        fsm = create_advanced_fsm(config)
        context = fsm.create_context({
            "value": 1,
            "_transition_signal": "nonexistent_state",
        })

        result = fsm.execute_step_sync(context)

        assert result.success
        # Falls through to normal evaluation → first arc (start->middle)
        assert result.to_state == "middle"
        # Signal was consumed (popped) even though it didn't match
        assert "_transition_signal" not in context.data

    def test_transform_sets_signal_for_next_step(self) -> None:
        """Transform that sets _transition_signal affects the NEXT step."""

        def set_signal(data: dict, ctx: FunctionContext) -> dict:
            result = dict(data)
            result["_transition_signal"] = "end"
            return result

        config = _make_config(
            arcs=[
                {
                    "from": "start",
                    "to": "middle",
                    "transform": {"type": "registered", "name": "set_signal"},
                },
                {"from": "middle", "to": "end"},
                # Add a second arc from middle to verify signal picks the right one
            ],
            states=[
                {"name": "start", "is_start": True},
                {"name": "middle"},
                {"name": "other"},
                {"name": "end", "is_end": True},
            ],
        )
        # Add arc from middle->other so there are two choices
        config["networks"][0]["arcs"].append({"from": "middle", "to": "other"})

        fsm = create_advanced_fsm(
            config,
            custom_functions={"set_signal": set_signal},
        )
        context = fsm.create_context({"value": 1})

        # Step 1: start -> middle (transform sets signal)
        result1 = fsm.execute_step_sync(context)
        assert result1.success
        assert result1.to_state == "middle"

        # Step 2: middle -> end (signal selects "end" arc)
        result2 = fsm.execute_step_sync(context)
        assert result2.success
        assert result2.to_state == "end"


class TestTransitionSignalAsync:
    """Async execution path honors _transition_signal."""

    @pytest.mark.asyncio
    async def test_signal_selects_target_arc_async(self) -> None:
        """_transition_signal works through execute_step_async."""
        config = _make_config()
        fsm = create_advanced_fsm(config)
        context = fsm.create_context({"value": 1, "_transition_signal": "end"})

        result = await fsm.execute_step_async(context)

        assert result.success
        assert result.to_state == "end"

    @pytest.mark.asyncio
    async def test_signal_consumed_async(self) -> None:
        """Signal is consumed in async path."""
        config = _make_config()
        fsm = create_advanced_fsm(config)
        context = fsm.create_context({"value": 1, "_transition_signal": "end"})

        await fsm.execute_step_async(context)

        assert "_transition_signal" not in context.data

    @pytest.mark.asyncio
    async def test_unmatched_signal_falls_through_async(self) -> None:
        """Unmatched signal falls through to normal eval in async path."""
        config = _make_config()
        fsm = create_advanced_fsm(config)
        context = fsm.create_context({
            "value": 1,
            "_transition_signal": "nonexistent_state",
        })

        result = await fsm.execute_step_async(context)

        assert result.success
        assert result.to_state == "middle"
        assert "_transition_signal" not in context.data

    @pytest.mark.asyncio
    async def test_transform_sets_signal_for_next_step_async(self) -> None:
        """Async: transform sets signal, next step uses it."""

        async def set_signal(data: dict, ctx: FunctionContext) -> dict:
            result = dict(data)
            result["_transition_signal"] = "end"
            return result

        config = _make_config(
            arcs=[
                {
                    "from": "start",
                    "to": "middle",
                    "transform": {"type": "registered", "name": "set_signal"},
                },
                {"from": "middle", "to": "end"},
            ],
            states=[
                {"name": "start", "is_start": True},
                {"name": "middle"},
                {"name": "other"},
                {"name": "end", "is_end": True},
            ],
        )
        config["networks"][0]["arcs"].append({"from": "middle", "to": "other"})

        fsm = create_advanced_fsm(
            config,
            custom_functions={"set_signal": set_signal},
        )
        context = fsm.create_context({"value": 1})

        # Step 1: start -> middle (transform sets signal)
        result1 = await fsm.execute_step_async(context)
        assert result1.success
        assert result1.to_state == "middle"

        # Step 2: middle -> end (signal selects "end" arc)
        result2 = await fsm.execute_step_async(context)
        assert result2.success
        assert result2.to_state == "end"


class TestTransitionSignalSkipsConditions:
    """Signal bypasses pre-test condition evaluation."""

    def test_signal_bypasses_failing_pretest(self) -> None:
        """Signal selects arc even when pre-tests would reject it."""

        def always_false(data: dict, ctx: object) -> bool:
            return False

        config = _make_config(
            arcs=[
                {
                    "from": "start",
                    "to": "end",
                    "pre_test": "block",
                },
            ],
        )
        fsm = create_advanced_fsm(
            config,
            custom_functions={"block": always_false},
        )

        # Without signal: no transitions available (pre-test blocks)
        context_no_signal = fsm.create_context({"value": 1})
        result = fsm.execute_step_sync(context_no_signal)
        assert result.to_state == "start"  # Stayed in start

        # With signal: arc selected despite failing pre-test
        context_with_signal = fsm.create_context({
            "value": 1,
            "_transition_signal": "end",
        })
        result = fsm.execute_step_sync(context_with_signal)
        assert result.success
        assert result.to_state == "end"
