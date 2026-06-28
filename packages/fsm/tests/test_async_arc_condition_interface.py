"""Arc-condition ``IStateTestFunction`` dispatch + initial-state error fidelity.

Reproduce-first coverage for the async-engine consolidation tail (170-FU8b /
170-FU8c-3):

- **FU8b** — a bare :class:`IStateTestFunction` instance reaching the engine's
  arc-condition path is not callable, so it must be dispatched via its ``.test``
  method. The manager (``FunctionWrapper``), the config builder, and the
  ``SimpleFSM`` build path all normalize such an instance to its bound method;
  the one path that does **not** is the low-level
  ``AsyncExecutionEngine(custom_functions=...)`` merge, which stores injected
  functions raw. These tests drive that exact path (and the public
  ``ArcExecution.can_execute_async``) with a real interface instance.

- **FU8c-3** — a rejecting *initial-state* pre-validator must surface the
  specific reason ("Pre-validation failed for state 'X'"), not the generic
  "Failed to enter initial state 'X'".

Real constructs only: real interface instances from ``tests.custom_fns_fixture``
(``HasField`` / ``AsyncHasField``), the real builder/loader, and the real async
engine. No mocks.
"""

from __future__ import annotations

import pytest

from dataknobs_fsm.config.builder import FSMBuilder
from dataknobs_fsm.config.loader import ConfigLoader
from dataknobs_fsm.core.arc import ArcDefinition, ArcExecution
from dataknobs_fsm.execution.async_engine import AsyncExecutionEngine
from dataknobs_fsm.execution.context import ExecutionContext

from tests.custom_fns_fixture import AsyncHasField, HasField


def _always_pass(data, context=None):
    """Plain registered placeholder predicate so the config arc validates.

    The engine's ``custom_functions`` then overrides this name with the bare
    interface instance under test, reproducing the un-normalized merge path.
    """
    return True


def _build_fsm_with_registered_condition(condition_name: str):
    """Two-state FSM whose start→end arc is gated by a registered condition."""
    config = {
        "name": "arc_condition_fsm",
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
                        "condition": {
                            "type": "registered",
                            "name": condition_name,
                        },
                    }
                ],
            }
        ],
    }
    loader = ConfigLoader()
    fsm_config = loader.load_from_dict(config)
    builder = FSMBuilder()
    # Register the placeholder so the registered-condition reference resolves at
    # build time; arc.pre_test becomes ``condition_name``.
    builder.register_function(condition_name, _always_pass)
    return builder.build(fsm_config)


class TestBareStateTestInstanceArcCondition:
    """FU8b: a bare IStateTestFunction injected via engine custom_functions."""

    @pytest.mark.asyncio
    async def test_bare_instance_gates_transition(self):
        """A bare ``IStateTestFunction`` custom_function gates the arc.

        Without dispatch, ``functions['gate']`` is the non-callable instance and
        ``_evaluate_arc`` calls ``instance(data, ctx)`` → ``TypeError`` →
        surfaced as a record error (``success`` False) for *every* record. With
        dispatch, the record carrying the field passes and the one without is
        gated out.
        """
        fsm = _build_fsm_with_registered_condition("gate")
        engine = AsyncExecutionEngine(
            fsm, custom_functions={"gate": HasField("flag")}
        )

        ctx_ok = ExecutionContext()
        ctx_ok.data = {"flag": 1}
        success, _ = await engine.execute(ctx_ok)
        assert success, "record satisfying the bare-instance arc condition should pass"
        assert ctx_ok.current_state == "end"

        fsm2 = _build_fsm_with_registered_condition("gate")
        engine2 = AsyncExecutionEngine(
            fsm2, custom_functions={"gate": HasField("flag")}
        )
        ctx_blocked = ExecutionContext()
        ctx_blocked.data = {"other": 1}
        success2, _ = await engine2.execute(ctx_blocked)
        assert not success2, "record failing the bare-instance arc condition is gated"
        assert ctx_blocked.current_state == "start"

    @pytest.mark.asyncio
    async def test_bare_async_instance_is_awaited(self):
        """A bare ``async def test`` instance is awaited and gates the arc.

        An un-awaited coroutine is always truthy, so a "called but not awaited"
        regression would pass *every* record; the blocked record proves the
        coroutine's ``(passed, reason)`` is actually awaited and honored.
        """
        fsm = _build_fsm_with_registered_condition("gate")
        engine = AsyncExecutionEngine(
            fsm, custom_functions={"gate": AsyncHasField("flag")}
        )
        ctx_ok = ExecutionContext()
        ctx_ok.data = {"flag": 1}
        success, _ = await engine.execute(ctx_ok)
        assert success
        assert ctx_ok.current_state == "end"

        fsm2 = _build_fsm_with_registered_condition("gate")
        engine2 = AsyncExecutionEngine(
            fsm2, custom_functions={"gate": AsyncHasField("flag")}
        )
        ctx_blocked = ExecutionContext()
        ctx_blocked.data = {"other": 1}
        success2, _ = await engine2.execute(ctx_blocked)
        assert not success2
        assert ctx_blocked.current_state == "start"

    @pytest.mark.asyncio
    async def test_arc_execution_can_execute_async_dispatches_instance(self):
        """Public ``ArcExecution.can_execute_async`` dispatches a bare instance.

        ``can_execute_async`` is the sibling arc-condition path on the exported
        ``ArcExecution`` class; it carries the same defect and the same fix
        (neighboring-code audit). A ``dict`` function registry holding a bare
        instance is the minimal reproduction.
        """
        arc_def = ArcDefinition(target_state="end", pre_test="gate")
        arc_exec = ArcExecution(arc_def, "start", {"gate": HasField("flag")})

        ctx = ExecutionContext()
        ctx.current_state = "start"

        assert await arc_exec.can_execute_async(ctx, {"flag": 1}) is True
        assert await arc_exec.can_execute_async(ctx, {"other": 1}) is False

    @pytest.mark.asyncio
    async def test_arc_execution_can_execute_async_awaits_async_instance(self):
        """``ArcExecution.can_execute_async`` awaits a bare ``async def test``.

        The async branch (``inspect.isawaitable(result)`` → ``await``) on the
        public ``ArcExecution`` API is otherwise only exercised transitively via
        ``_evaluate_arc``. ``AsyncHasField`` forces a real suspension point, so a
        "dispatched but not awaited" regression — an un-awaited coroutine is
        always truthy — would wrongly pass the blocked record and fail this test.
        """
        arc_def = ArcDefinition(target_state="end", pre_test="gate")
        arc_exec = ArcExecution(arc_def, "start", {"gate": AsyncHasField("flag")})

        ctx = ExecutionContext()
        ctx.current_state = "start"

        assert await arc_exec.can_execute_async(ctx, {"flag": 1}) is True
        assert await arc_exec.can_execute_async(ctx, {"other": 1}) is False


class TestInitialStatePreValidationErrorFidelity:
    """FU8c-3: rejecting initial-state pre-validator surfaces the specific reason."""

    @pytest.mark.asyncio
    async def test_initial_prevalidation_failure_surfaces_specific_reason(self):
        """A failing initial-state pre-validator reports *why*, not just *that*.

        Before the fix, ``_enter_initial_state`` returned the generic
        "Failed to enter initial state 'X'" — the message ``execute()`` surfaces
        to the caller — discarding the specific "Pre-validation failed for state
        'X'" recorded on ``context.last_error``. The assertion is on the
        *returned* error (the consumer-visible channel), not ``last_error``
        (which already held the specific reason).
        """
        config = {
            "name": "reject_initial_fsm",
            "main_network": "main",
            "networks": [
                {
                    "name": "main",
                    "states": [
                        {
                            "name": "start",
                            "is_start": True,
                            "pre_validators": [
                                {
                                    "type": "inline",
                                    "code": "lambda data: 'required' in data",
                                }
                            ],
                        },
                        {"name": "end", "is_end": True},
                    ],
                    "arcs": [{"from": "start", "to": "end", "name": "go"}],
                }
            ],
        }
        loader = ConfigLoader()
        fsm_config = loader.load_from_dict(config)
        fsm = FSMBuilder().build(fsm_config)
        engine = AsyncExecutionEngine(fsm)

        ctx = ExecutionContext()
        ctx.data = {"other": 1}
        success, error = await engine.execute(ctx)

        assert not success
        error_text = str(error or "")
        assert "Pre-validation failed" in error_text, (
            "initial-state pre-validation rejection should surface the specific "
            f"reason to the caller, got: {error_text!r}"
        )

    @pytest.mark.asyncio
    async def test_stale_last_error_does_not_mask_current_reason(self):
        """A reused context's stale ``last_error`` must not surface as the reason.

        ``_initial_entry_error`` prefers ``context.last_error`` to report the
        specific rejection reason. But ``_establish_state`` only records that
        reason when ``last_error`` is unset (the don't-clobber gate), so a stale
        value left on a reused SINGLE-mode context would survive and be returned
        in place of this run's actual reason — making the "specific" message
        *less* accurate than the generic one. ``_enter_initial_state`` clears
        ``last_error`` before the entry attempt to scope the contract to the
        current run; without that clear this asserts the stale string.
        """
        config = {
            "name": "reject_initial_fsm",
            "main_network": "main",
            "networks": [
                {
                    "name": "main",
                    "states": [
                        {
                            "name": "start",
                            "is_start": True,
                            "pre_validators": [
                                {
                                    "type": "inline",
                                    "code": "lambda data: 'required' in data",
                                }
                            ],
                        },
                        {"name": "end", "is_end": True},
                    ],
                    "arcs": [{"from": "start", "to": "end", "name": "go"}],
                }
            ],
        }
        loader = ConfigLoader()
        fsm_config = loader.load_from_dict(config)
        fsm = FSMBuilder().build(fsm_config)
        engine = AsyncExecutionEngine(fsm)

        ctx = ExecutionContext()
        # Simulate a reused context carrying a reason from a prior turn.
        ctx.last_error = "stale reason from a prior run"
        ctx.data = {"other": 1}

        success, error = await engine.execute(ctx)

        assert not success
        error_text = str(error or "")
        assert "stale reason from a prior run" not in error_text, (
            "a stale last_error from a prior run must not surface as this run's "
            f"initial-entry reason, got: {error_text!r}"
        )
        assert "Pre-validation failed" in error_text, (
            "the current run's specific rejection reason should surface, got: "
            f"{error_text!r}"
        )
