"""Reproduce-first: AdvancedFSM step API must surface state-transform failures.

The execution engines were fixed so a record whose *state* transform raises is
reported as ``success=False`` (``BaseExecutionEngine.finalize_single_result``).
The step-driver API (``AdvancedFSM.execute_step_sync`` / ``execute_step_async``)
shares the same ``handle_transform_error`` mechanism — entering a state runs its
transforms and a raise is recorded in ``context.failed_states`` — but the step
methods hard-coded ``success=True`` and never consulted it. So a step that
entered a state whose transform raised reported a fully successful step: the
same "silent success at a final state" defect class, still live in the parallel
step-driver implementation.

These tests drive a state whose registered transform raises. They FAIL against
the hard-coded ``success=True`` behavior and PASS once the step reports
``success=False`` and surfaces the offending state in ``StepResult.failed_states``.

Note: this is distinct from an *arc* transform raising (covered by
``test_advanced_fsm_async_step.py``), which already failed the step via
``_execute_arc_transform`` returning ``(False, ...)``.
"""

from __future__ import annotations

import pytest

from dataknobs_fsm.api.advanced import StepResult, create_advanced_fsm


def _boom_state(*_args, **_kwargs):
    """A state transform that always raises, however it is invoked."""
    raise RuntimeError("state transform exploded")


def _state_fail_config() -> dict:
    """start -> middle (state transform raises) -> done."""
    return {
        "name": "StateFailFSM",
        "main_network": "main",
        "networks": [
            {
                "name": "main",
                "states": [
                    {"name": "start", "is_start": True},
                    {
                        "name": "middle",
                        "functions": {
                            "transform": {"type": "registered", "name": "boom_state"}
                        },
                    },
                    {"name": "done", "is_end": True},
                ],
                "arcs": [
                    {"from": "start", "to": "middle"},
                    {"from": "middle", "to": "done"},
                ],
            }
        ],
    }


@pytest.mark.asyncio
async def test_async_step_surfaces_state_transform_failure() -> None:
    advanced = create_advanced_fsm(
        _state_fail_config(), custom_functions={"boom_state": _boom_state}
    )
    context = advanced.create_context({"id": "1"})

    # Step 1: start -> middle, whose state transform raises.
    result = await advanced.execute_step_async(context)

    assert isinstance(result, StepResult)
    assert result.to_state == "middle"
    assert result.success is False, (
        "a state transform that raised was reported as a successful step"
    )
    assert result.failed_states == ["middle"]
    assert "middle" in (result.error or "")


def test_sync_step_surfaces_state_transform_failure() -> None:
    advanced = create_advanced_fsm(
        _state_fail_config(), custom_functions={"boom_state": _boom_state}
    )
    context = advanced.create_context({"id": "1"})

    result = advanced.execute_step_sync(context)

    assert result.to_state == "middle"
    assert result.success is False, (
        "a state transform that raised was reported as a successful step"
    )
    assert result.failed_states == ["middle"]
    assert "middle" in (result.error or "")


@pytest.mark.asyncio
async def test_run_until_breakpoint_stops_on_state_transform_failure() -> None:
    """The aggregating driver stops and returns the failing step."""
    advanced = create_advanced_fsm(
        _state_fail_config(), custom_functions={"boom_state": _boom_state}
    )
    context = advanced.create_context({"id": "1"})

    result = await advanced.run_until_breakpoint(context)

    assert result is not None
    assert result.success is False
    assert result.failed_states == ["middle"]


def test_step_run_on_failure_state_runs_and_surfaces_accumulated_failure() -> None:
    """Step-driver: a run_on_failure state runs despite a prior failure, and the
    step reports success=True while still carrying the accumulated failed_states.

    This locks two parts of the contract on the step path specifically:

    * the ``run_on_failure`` exemption is honored by ``execute_step_sync`` (the
      step path has its own guard, separate from the engine drive loop); and
    * the documented manual-stepping trap — a step that enters a *clean*
      downstream state after an upstream failure reports ``success=True`` (that
      state did not itself fail) but still surfaces the accumulated
      ``failed_states``, so a consumer must check ``failed_states``, not
      ``success`` alone.
    """
    calls: list[str] = []

    def _spy_cleanup(*args, **_kwargs):
        calls.append("cleanup")
        state = args[0]
        data = getattr(state, "data", state)
        return dict(data) if isinstance(data, dict) else {}

    config = {
        "name": "StepRecoveryFSM",
        "main_network": "main",
        "networks": [
            {
                "name": "main",
                "states": [
                    {"name": "start", "is_start": True},
                    {
                        "name": "fail",
                        "functions": {
                            "transform": {"type": "registered", "name": "boom_state"}
                        },
                    },
                    {
                        "name": "cleanup",
                        "run_on_failure": True,
                        "functions": {
                            "transform": {"type": "registered", "name": "spy_cleanup"}
                        },
                    },
                    {"name": "done", "is_end": True},
                ],
                "arcs": [
                    {"from": "start", "to": "fail"},
                    {"from": "fail", "to": "cleanup"},
                    {"from": "cleanup", "to": "done"},
                ],
            }
        ],
    }
    advanced = create_advanced_fsm(
        config,
        custom_functions={"boom_state": _boom_state, "spy_cleanup": _spy_cleanup},
    )
    context = advanced.create_context({"id": "1"})

    # Step 1: start -> fail, whose transform raises.
    step1 = advanced.execute_step_sync(context)
    assert step1.to_state == "fail"
    assert step1.success is False
    assert step1.failed_states == ["fail"]

    # Step 2: fail -> cleanup. cleanup is run_on_failure, so its transform RUNS
    # despite the prior failure; the step succeeds (cleanup did not itself fail)
    # but still surfaces the accumulated failure.
    step2 = advanced.execute_step_sync(context)
    assert step2.to_state == "cleanup"
    assert "cleanup" in calls, (
        "a run_on_failure state's transform must run in the step-driver path too"
    )
    assert step2.success is True, (
        "the step into a clean run_on_failure state should report success=True"
    )
    assert step2.failed_states == ["fail"], (
        "the accumulated upstream failure must still surface on a success=True step"
    )


def test_clean_state_transform_still_succeeds() -> None:
    """Guard: a non-raising state transform keeps success=True / no failed_states."""

    def _stamp(*args, **_kwargs):
        # state_obj-first call form: single positional arg with .data access.
        state = args[0]
        data = getattr(state, "data", state)
        return {**data, "stamped": True}

    config = _state_fail_config()
    config["networks"][0]["states"][1]["functions"] = {
        "transform": {"type": "registered", "name": "stamp"}
    }
    advanced = create_advanced_fsm(config, custom_functions={"stamp": _stamp})
    context = advanced.create_context({"id": "1"})

    result = advanced.execute_step_sync(context)

    assert result.success is True
    assert result.failed_states is None
    assert result.to_state == "middle"
