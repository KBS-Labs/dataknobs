# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""A transform that fails says what went wrong, not only where.

``BaseExecutionEngine.handle_transform_error`` is the single sink every state
transform failure reaches --- from the async engine's transform loop, from
``AdvancedFSM``'s stepped transform runner, and from
``process_transform_result`` when a transform *returns*
``ExecutionResult.failure_result(...)``. It took the exception as its first
argument and recorded the state name::

    def handle_transform_error(self, error, context, state_name):
        context.failed_states.add(state_name)

``error`` was never read. It was not stored, not attached to the record, and
not logged --- measured at ``DEBUG`` over the whole workspace logger, zero
records mention it and zero carry ``exc_info``. So every surface the failure
reaches carried the same sentence and nothing else:

========================================  =========================================
surface                                    what a caller saw
========================================  =========================================
``SimpleFSM.process``                      ``State transform failed in: work``
``AsyncSimpleFSM.process``                 ``State transform failed in: work``
``AsyncSimpleFSM.process_batch``           ``State transform failed in: work``
``AsyncBatchExecutor`` metadata            ``failed_states: ['work']``
``AdvancedFSM`` ``StepResult.error``       ``State transform failed in: work``
logs                                       nothing
========================================  =========================================

The record still reaches a final state and is correctly reported as a
failure --- that contract is older than this change and is unaltered. What
was missing is the reason, and the state name is the one part of it the
caller already knew. A ``KeyError`` naming the column that was absent, a
``ConnectionError`` from the load step, a ``ValueError`` a consumer raised on
purpose with a message written for exactly this moment: all of them were
discarded at the sink, with their tracebacks.

The narrowest case is the second door. A transform that returns
``ExecutionResult.failure_result("row 42 has no key column")`` has *stated* its
reason, in a string, deliberately --- and ``process_transform_result`` wrapped
it in ``Exception(result.error)`` and handed it to the sink that dropped it.

The exception is now kept on ``context.transform_errors``, keyed by state, and
its type and message are part of the message every surface above already
carried. ``failed_states`` is unchanged: same type, same contents, same
lifecycle, still the authority on *which* states failed.

Real constructs only: real configs through the real facades, a real
:class:`FSM` and real engines.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import pytest

from dataknobs_fsm.api.advanced import AdvancedFSM
from dataknobs_fsm.api.async_simple import AsyncSimpleFSM
from dataknobs_fsm.api.simple import SimpleFSM
from dataknobs_fsm.config.builder import build_fsm
from dataknobs_fsm.core.context_factory import ContextFactory
from dataknobs_fsm.core.data_modes import DataHandlingMode
from dataknobs_fsm.core.modes import ProcessingMode
from dataknobs_fsm.execution.async_engine import AsyncExecutionEngine
from dataknobs_fsm.execution.context import ExecutionContext
from dataknobs_fsm.functions.base import ExecutionResult

REASON = "the upstream row had no 'name' column"


def _config(*, recovery: bool = False) -> Dict[str, Any]:
    """``start -> work -> done``, optionally with a recovery state after work.

    A ``run_on_failure`` state is the only way a second state's transform runs
    after the first has failed, so it is how a two-reason message is reached.
    """
    states: List[Dict[str, Any]] = [
        {"name": "start", "is_start": True, "resources": []},
        {
            "name": "work",
            "resources": [],
            "functions": {"transform": {"type": "registered", "name": "boom"}},
        },
    ]
    arcs: List[Dict[str, Any]] = [{"from": "start", "to": "work", "name": "go"}]

    if recovery:
        states.append(
            {
                "name": "cleanup",
                "resources": [],
                "run_on_failure": True,
                "functions": {"transform": {"type": "registered", "name": "boom2"}},
            }
        )
        arcs.append({"from": "work", "to": "cleanup", "name": "recover"})
        arcs.append({"from": "cleanup", "to": "done", "name": "fin"})
    else:
        arcs.append({"from": "work", "to": "done", "name": "fin"})

    states.append({"name": "done", "is_end": True})
    return {
        "name": "reason_probe",
        "data_mode": DataHandlingMode.COPY.value,
        "resources": [],
        "states": states,
        "arcs": arcs,
    }


def _plain_config() -> Dict[str, Any]:
    """The same shape with no configured transform, for the pre-built-FSM door."""
    config = _config()
    for state in config["states"]:
        state.pop("functions", None)
    return config


def _boom(state: Any) -> Dict[str, Any]:
    """A consumer transform that raises with a message written to be read."""
    raise ValueError(REASON)


def _boom2(state: Any) -> Dict[str, Any]:
    raise ConnectionError("the dead-letter queue refused the record")


def _states_its_own_reason(state: Any) -> ExecutionResult:
    """The second door: a transform that reports failure instead of raising."""
    return ExecutionResult.failure_result("row 42 has no key column")


# --------------------------------------------------------------------------- #
# What a caller is told
# --------------------------------------------------------------------------- #


async def test_the_async_facade_reports_the_reason() -> None:
    """``result['error']`` named the state and nothing else."""
    fsm = AsyncSimpleFSM(
        _config(), data_mode=DataHandlingMode.COPY, custom_functions={"boom": _boom}
    )
    result = await fsm.process({"a": 1})

    assert result["success"] is False
    assert "work" in result["error"]
    assert REASON in result["error"], result["error"]
    assert "ValueError" in result["error"], result["error"]


def test_the_sync_facade_reports_the_reason() -> None:
    """The synchronous door reaches the same sink through the same base engine."""
    with SimpleFSM(
        _config(), data_mode=DataHandlingMode.COPY, custom_functions={"boom": _boom}
    ) as fsm:
        result = fsm.process({"a": 1})

    assert result["success"] is False
    assert REASON in result["error"], result["error"]


async def test_a_transform_that_states_its_reason_keeps_it() -> None:
    """``ExecutionResult.failure_result`` carried a message that was dropped.

    This path never raises: the transform returns, and
    ``process_transform_result`` builds the exception itself. The message it
    wraps is the consumer's own sentence.
    """
    fsm = AsyncSimpleFSM(
        _config(),
        data_mode=DataHandlingMode.COPY,
        custom_functions={"boom": _states_its_own_reason},
    )
    result = await fsm.process({"a": 1})

    assert result["success"] is False
    assert "row 42 has no key column" in result["error"], result["error"]


async def test_a_batch_record_reports_its_own_reason() -> None:
    """Each failing record in a batch carries the reason it failed."""
    fsm = AsyncSimpleFSM(
        _config(), data_mode=DataHandlingMode.COPY, custom_functions={"boom": _boom}
    )
    results = await fsm.process_batch([{"a": 1}, {"a": 2}])

    assert [r["success"] for r in results] == [False, False], results
    for item in results:
        assert REASON in item["error"], item["error"]


async def test_each_failed_state_reports_its_own_reason() -> None:
    """A ``run_on_failure`` state that also fails adds its reason, not just its name."""
    fsm = AsyncSimpleFSM(
        _config(recovery=True),
        data_mode=DataHandlingMode.COPY,
        custom_functions={"boom": _boom, "boom2": _boom2},
    )
    result = await fsm.process({"a": 1})

    assert result["success"] is False
    assert REASON in result["error"], result["error"]
    assert "the dead-letter queue refused the record" in result["error"], result["error"]


async def test_a_step_reports_the_reason() -> None:
    """``StepResult.error`` built the same sentence in its own copy of the code."""
    fsm = build_fsm(_plain_config())
    fsm.states["work"].transform_functions.append(_boom)
    advanced = AdvancedFSM(fsm)
    context = ContextFactory.create_context(fsm, {"a": 1}, data_mode=ProcessingMode.SINGLE)

    step = await advanced.execute_step_async(context)
    while step.success and not step.is_complete:
        step = await advanced.execute_step_async(context)

    assert step.success is False, step
    assert step.failed_states == ["work"], step
    assert REASON in (step.error or ""), step.error


# --------------------------------------------------------------------------- #
# What is kept, and where
# --------------------------------------------------------------------------- #


async def _run_engine(transform: Any) -> ExecutionContext:
    """Drive the engine over a pre-built FSM, so the context survives the run.

    ``StateDefinition.transform_functions`` takes a plain callable, which is
    the shape a consumer building an FSM programmatically produces and the one
    the facades hide.
    """
    fsm = build_fsm(_plain_config())
    fsm.states["work"].transform_functions.append(transform)
    engine = AsyncExecutionEngine(fsm)
    context = ContextFactory.create_context(fsm, {"a": 1}, data_mode=ProcessingMode.SINGLE)
    await engine.execute(context, {"a": 1})
    return context


async def test_the_exception_itself_is_kept_on_the_context() -> None:
    """Not the message --- the exception, so a caller can branch on its type."""
    context = await _run_engine(_boom)

    assert context.failed_states == {"work"}
    error = context.transform_errors["work"]
    assert isinstance(error, ValueError)
    assert str(error) == REASON
    assert error.__traceback__ is not None, "the traceback is the diagnostic"


async def test_the_failure_is_logged_with_its_traceback(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Nothing logged it. A run that fails leaves no trace to find later."""
    with caplog.at_level(logging.ERROR, logger="dataknobs_fsm"):
        await _run_engine(_boom)

    carrying = [r for r in caplog.records if r.exc_info]
    assert carrying, [r.getMessage() for r in caplog.records]
    assert any("work" in r.getMessage() for r in carrying)
    assert any(REASON in str(r.exc_info[1]) for r in carrying if r.exc_info)


# --------------------------------------------------------------------------- #
# The two collections travel together
# --------------------------------------------------------------------------- #


def test_a_clone_starts_with_no_reasons() -> None:
    """``failed_states`` is not cloned, so the reasons must not be either."""
    context = ExecutionContext()
    context.failed_states.add("work")
    context.transform_errors["work"] = ValueError(REASON)

    clone = context.clone()

    assert clone.failed_states == set()
    assert clone.transform_errors == {}


def test_a_parallel_child_starts_with_no_reasons() -> None:
    """Same rule at the other door that builds a fresh sub-path."""
    context = ExecutionContext()
    context.failed_states.add("work")
    context.transform_errors["work"] = ValueError(REASON)

    child = context.create_child_context("p1")

    assert child.failed_states == set()
    assert child.transform_errors == {}


def test_a_child_paths_reason_reaches_the_parent() -> None:
    """A merge that carried the name but not the reason would lose the diagnostic."""
    parent = ExecutionContext()
    child = parent.create_child_context("p1")
    child.failed_states.add("child_transform")
    child.transform_errors["child_transform"] = ValueError(REASON)

    assert parent.merge_child_context("p1") is True
    assert parent.failed_states == {"child_transform"}
    assert str(parent.transform_errors["child_transform"]) == REASON


def test_the_two_collections_are_carried_by_the_same_three_methods() -> None:
    """A recurrence guard, because they are two collections with one lifecycle.

    ``failed_states`` was once dropped by ``merge_child_context`` and the
    parent reported success; adding a second collection beside it adds a
    second chance to make that mistake, once per method. This asserts the
    three answers agree rather than trusting three separate edits.
    """
    seeded = ExecutionContext()
    seeded.failed_states.add("work")
    seeded.transform_errors["work"] = ValueError(REASON)

    assert bool(seeded.clone().failed_states) is bool(seeded.clone().transform_errors)
    child = seeded.create_child_context("p1")
    assert bool(child.failed_states) is bool(child.transform_errors)

    child.failed_states.add("inner")
    child.transform_errors["inner"] = ValueError("inner")
    seeded.merge_child_context("p1")
    assert set(seeded.transform_errors) == seeded.failed_states


# --------------------------------------------------------------------------- #
# Controls
# --------------------------------------------------------------------------- #


async def test_a_clean_run_records_no_reason() -> None:
    """The control: nothing failed, so nothing is reported and nothing is kept."""
    context = await _run_engine(lambda state: {"ok": True})

    assert context.failed_states == set()
    assert context.transform_errors == {}


async def test_a_clean_run_still_succeeds_through_the_facade() -> None:
    """And the caller still gets its data, with no error."""
    fsm = AsyncSimpleFSM(
        _config(),
        data_mode=DataHandlingMode.COPY,
        custom_functions={"boom": lambda state: {"ok": True}},
    )
    result = await fsm.process({"a": 1})

    assert result["success"] is True, result
    assert result["error"] is None, result
