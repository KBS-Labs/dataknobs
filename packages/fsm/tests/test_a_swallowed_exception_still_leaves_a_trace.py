# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs

# SPDX-License-Identifier: Apache-2.0

"""An exception the engine declines to raise is still written down.

Three handlers in the execution path catch every exception and continue, on
purpose: a validator is optional, and a monitoring hook must not be able to
stop a run. The policy is right. What was missing is the record --- all three
were ``except Exception: pass``, so the exception reached no log, no result
and no attribute, and the run finished reporting exactly what it would have
reported had nothing gone wrong.

=====================================================  ==========================
handler                                                 what survived it
=====================================================  ==========================
``AsyncExecutionEngine`` validator loop                 nothing
``AsyncExecutionEngine._run_pre_validators``            nothing
``BaseExecutionEngine.evaluate_arc_condition_common``   nothing
``AsyncExecutionEngine._fire_hooks``                    nothing
``AdvancedFSM._call_hook_async``                        nothing
=====================================================  ==========================

Two of the five answer ``False`` rather than continuing, and those are the
ones that change an outcome: a pre-validator that *crashed* is reported as a
pre-validator that *rejected the record*, and an arc condition that crashed is
reported as an arc that declined. The record is turned away and the bug that
turned it away leaves no trace. The package already has the right shape for
this one --- ``_evaluate_arc_pre_test`` logs with the traceback and re-raises,
on the argument that an infrastructure outage must not be reported as a
data-quality drop --- and these two disagree with it. **They keep their
outcome here**, because changing what happens to a record is not the same
decision as writing down why; what changes is that the reason exists.

The validator one had already been noticed and left half-done: its comment
reads ``# Log but don't fail - validators are optional`` above a bare
``pass``. It is also the most consequential of the three. A validator that
raises is indistinguishable from a validator that passed, so a record the
consumer wrote a validator to reject flows on as a clean record --- and the
validator's own bug is the reason.

This does not change what any of the three handlers *do*. Every exception is
still caught, no run fails that did not fail before, and the two hook
handlers keep the guarantee their comments claim. They now log at ``WARNING``
with the traceback attached.

Real constructs only: a real :class:`FSM`, the real engine, the real
``AdvancedFSM`` hook surface.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import pytest

from dataknobs_fsm.api.advanced import AdvancedFSM, ExecutionHook
from dataknobs_fsm.config.builder import build_fsm
from dataknobs_fsm.core.context_factory import ContextFactory
from dataknobs_fsm.core.modes import ProcessingMode
from dataknobs_fsm.execution.async_engine import AsyncExecutionEngine

CONFIG: Dict[str, Any] = {
    "name": "swallow_probe",
    "data_mode": "copy",
    "resources": [],
    "states": [
        {"name": "start", "is_start": True, "resources": []},
        {"name": "work", "resources": []},
        {"name": "done", "is_end": True},
    ],
    "arcs": [
        {"from": "start", "to": "work", "name": "go"},
        {"from": "work", "to": "done", "name": "fin"},
    ],
}

REASON = "the validator's own regex was malformed"
GATE_REASON = "the gate's reference table was not loaded"


class BrokenValidator:
    """A consumer validator with a bug in it, not a record it wants to reject."""

    def validate(self, state: Any) -> Dict[str, Any]:
        raise ValueError(REASON)


def _fsm_with_validator() -> Any:
    fsm = build_fsm(dict(CONFIG))
    fsm.states["work"].validation_functions.append(BrokenValidator())
    return fsm


async def test_a_validator_that_raises_is_written_down(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Its comment said "Log but don't fail"; it did neither half of that.

    The log half is what this test is for, and it still is. The other half
    has since been settled the other way: a validator that raises now refuses
    the record, as ``pre_validation_functions`` always has --- the two lists
    sit five lines apart in the same state entry and gave opposite answers to
    the same event. A record its gate could not check is not a checked
    record, so the traceback below is the reason for a refusal rather than a
    note beside a success.
    """
    fsm = _fsm_with_validator()
    engine = AsyncExecutionEngine(fsm)
    context = ContextFactory.create_context(fsm, {"a": 1}, data_mode=ProcessingMode.SINGLE)

    with caplog.at_level(logging.WARNING, logger="dataknobs_fsm"):
        success, _ = await engine.execute(context, {"a": 1})

    assert success is False, "a record the gate could not check is refused"
    carrying = [r for r in caplog.records if r.exc_info]
    assert carrying, [r.getMessage() for r in caplog.records]
    assert any(REASON in str(r.exc_info[1]) for r in carrying if r.exc_info)
    assert any("work" in r.getMessage() for r in carrying)


async def test_an_engine_hook_that_raises_is_written_down(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """ "Hooks must not break execution" is the policy, not a reason to say nothing."""
    fsm = build_fsm(dict(CONFIG))
    engine = AsyncExecutionEngine(fsm)

    def broken(context: Any, arc: Any) -> None:
        raise RuntimeError("the metrics sink was closed")

    engine.add_pre_transition_hook(broken)
    context = ContextFactory.create_context(fsm, {"a": 1}, data_mode=ProcessingMode.SINGLE)

    with caplog.at_level(logging.WARNING, logger="dataknobs_fsm"):
        success, _ = await engine.execute(context, {"a": 1})

    assert success is True, "a hook must not be able to fail the run"
    carrying = [r for r in caplog.records if r.exc_info]
    assert carrying, [r.getMessage() for r in caplog.records]
    assert any("the metrics sink was closed" in str(r.exc_info[1]) for r in carrying if r.exc_info)


async def test_an_advanced_api_hook_that_raises_is_written_down(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The same policy, in the other copy of it, on the published hook surface."""
    fsm = build_fsm(dict(CONFIG))
    advanced = AdvancedFSM(fsm)

    def broken(state_name: Any) -> None:
        raise RuntimeError("the tracing span was already closed")

    advanced.set_hooks(ExecutionHook(on_state_enter=broken))
    context = ContextFactory.create_context(fsm, {"a": 1}, data_mode=ProcessingMode.SINGLE)

    with caplog.at_level(logging.WARNING, logger="dataknobs_fsm"):
        step = await advanced.execute_step_async(context)

    assert step.success is True, "a hook must not be able to fail the step"
    carrying = [r for r in caplog.records if r.exc_info]
    assert carrying, [r.getMessage() for r in caplog.records]
    assert any(
        "the tracing span was already closed" in str(r.exc_info[1]) for r in carrying if r.exc_info
    )
    assert any("on_state_enter" in r.getMessage() for r in carrying)


class BrokenGate:
    """A pre-validator with a bug in it, not a record it means to reject."""

    __name__ = "broken_gate"

    def __call__(self, record: Dict[str, Any], func_context: Any) -> bool:
        raise ValueError(GATE_REASON)


async def test_a_pre_validator_that_raises_is_written_down(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A crash in a gate reads as a refusal, so the record is simply turned away."""
    fsm = build_fsm(dict(CONFIG))
    fsm.states["work"].pre_validation_functions.append(BrokenGate())
    engine = AsyncExecutionEngine(fsm)
    context = ContextFactory.create_context(fsm, {"a": 1}, data_mode=ProcessingMode.SINGLE)

    with caplog.at_level(logging.WARNING, logger="dataknobs_fsm"):
        entered = await engine.enter_state(context, "work")

    assert entered is False, "the outcome is unchanged: entry still fails"
    carrying = [r for r in caplog.records if r.exc_info]
    assert carrying, [r.getMessage() for r in caplog.records]
    assert any(GATE_REASON in str(r.exc_info[1]) for r in carrying if r.exc_info)
    assert any("work" in r.getMessage() for r in carrying)


async def test_an_arc_condition_that_raises_is_written_down(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The same shape one layer out: a broken condition reads as a declined arc."""
    fsm = build_fsm(dict(CONFIG))
    engine = AsyncExecutionEngine(fsm)
    context = ContextFactory.create_context(fsm, {"a": 1}, data_mode=ProcessingMode.SINGLE)

    def broken(record: Any, func_context: Any) -> bool:
        raise ValueError(GATE_REASON)

    arc = next(iter(fsm.main_network.arcs.values()))
    arc.condition = broken

    with caplog.at_level(logging.WARNING, logger="dataknobs_fsm"):
        allowed = engine.evaluate_arc_condition_common(arc, context)

    assert allowed is False, "the outcome is unchanged: the arc is still declined"
    carrying = [r for r in caplog.records if r.exc_info]
    assert carrying, [r.getMessage() for r in caplog.records]
    assert any(GATE_REASON in str(r.exc_info[1]) for r in carrying if r.exc_info)


async def test_a_passing_gate_logs_nothing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The control for the two that answer False: a clean refusal is not an error."""
    fsm = build_fsm(dict(CONFIG))

    def rejects(record: Dict[str, Any], func_context: Any) -> bool:
        return False

    fsm.states["work"].pre_validation_functions.append(rejects)
    engine = AsyncExecutionEngine(fsm)
    context = ContextFactory.create_context(fsm, {"a": 1}, data_mode=ProcessingMode.SINGLE)

    with caplog.at_level(logging.WARNING, logger="dataknobs_fsm"):
        entered = await engine.enter_state(context, "work")

    assert entered is False, "a validator that says no is still a no"
    assert [r for r in caplog.records if r.exc_info] == [], "and it is not an error"


async def test_a_working_validator_logs_nothing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The control: a clean run stays quiet, so the log means something."""
    fsm = build_fsm(dict(CONFIG))

    class Fine:
        def validate(self, state: Any) -> Dict[str, Any]:
            return {"checked": True}

    fsm.states["work"].validation_functions.append(Fine())
    engine = AsyncExecutionEngine(fsm)
    context = ContextFactory.create_context(fsm, {"a": 1}, data_mode=ProcessingMode.SINGLE)

    with caplog.at_level(logging.WARNING, logger="dataknobs_fsm"):
        success, _ = await engine.execute(context, {"a": 1})

    assert success is True
    assert [r for r in caplog.records if r.exc_info] == []
