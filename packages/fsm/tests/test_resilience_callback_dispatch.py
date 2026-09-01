"""The resilience wrappers judge the callable they were handed, or they lie.

Every class here exists to say something *about* an execution --- it succeeded,
it failed, the circuit is open, the compensation ran. Each of them decides
whether to await the callable with ``asyncio.iscoroutinefunction``, which
answers for *functions* and reports a callable **object** whose ``__call__``
is an ``async def`` as synchronous. That shape is not exotic here: a client
holding a session, a retryable operation holding a connection, an API caller
holding a token are all written that way, and this whole module is about
wrapping exactly those.

The consequence is worse than a dropped call, because these classes report.
A discarded coroutine raises nothing and returns a truthy object, so the
wrapper takes the success path: the circuit breaker counts a success it never
executed, the bulkhead's ``executed`` metric counts work that never ran, and
a compensation action that was supposed to undo a half-finished operation
returns a coroutine instead of undoing anything. The caller is told the
opposite of what happened, which is the one failure mode a resilience
component must not have.

The deadline strategy is broken in a second, louder way that no callable
shape reaches: its synchronous branch passes the *return value* of the
function to ``asyncio.create_task``, which requires a coroutine. Every
synchronous function raises ``TypeError`` there --- so the branch written to
handle synchronous functions is the one that cannot run them.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from dataknobs_fsm.patterns.api_orchestration import CircuitBreaker as APICircuitBreaker
from dataknobs_fsm.patterns.error_recovery import (
    Bulkhead,
    BulkheadConfig,
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerState,
    CompensationConfig,
    ErrorRecoveryConfig,
    ErrorRecoveryWorkflow,
    FallbackConfig,
    RecoveryStrategy,
)


class AsyncCallableObject:
    """A stateful async operation --- the shape ``iscoroutinefunction`` misreads.

    Written as an object because it holds state across calls, which is the
    ordinary reason a caller reaches for one and the reason it turns up in
    front of a circuit breaker rather than behind it.
    """

    def __init__(self, *, raises: BaseException | None = None, returns: Any = "ran") -> None:
        self.calls = 0
        self._raises = raises
        self._returns = returns

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        if self._raises is not None:
            raise self._raises
        return self._returns


class RecordingAction:
    """An async compensation action whose whole point is its side effect."""

    def __init__(self) -> None:
        self.ran = False

    async def __call__(self, saved_state: Any) -> None:
        self.ran = True


# --------------------------------------------------------------------- #
# The circuit breakers
# --------------------------------------------------------------------- #


async def test_circuit_breaker_runs_an_async_callable_object() -> None:
    breaker = CircuitBreaker(CircuitBreakerConfig())
    operation = AsyncCallableObject(returns="value")

    result = await breaker.call(operation)

    assert operation.calls == 1
    assert result == "value", "the coroutine was returned in place of the value"


async def test_circuit_breaker_opens_on_failures_it_can_actually_see() -> None:
    """The consequential half: a breaker that cannot see a failure never trips.

    Each call raises. With the coroutine discarded rather than awaited, none
    of those exceptions reaches the breaker, so every failing call is recorded
    as a success and the circuit stays closed over an operation that has never
    once worked --- fast-failing exactly nothing.
    """
    config = CircuitBreakerConfig(failure_threshold=2)
    breaker = CircuitBreaker(config)
    operation = AsyncCallableObject(raises=RuntimeError("upstream down"))

    for _ in range(config.failure_threshold):
        with pytest.raises(RuntimeError, match="upstream down"):
            await breaker.call(operation)

    assert breaker.failure_count >= config.failure_threshold
    assert breaker.state is CircuitBreakerState.OPEN, (
        "the breaker saw no failures because it never awaited the call"
    )


async def test_api_circuit_breaker_runs_an_async_callable_object() -> None:
    """The second copy of the same class, in the same package, same defect."""
    breaker = APICircuitBreaker(threshold=2, timeout=60.0)
    operation = AsyncCallableObject(returns="payload")

    result = await breaker.call(operation)

    assert operation.calls == 1
    assert result == "payload"


async def test_api_circuit_breaker_counts_a_failure_it_awaited() -> None:
    breaker = APICircuitBreaker(threshold=2, timeout=60.0)
    operation = AsyncCallableObject(raises=RuntimeError("upstream down"))

    with pytest.raises(RuntimeError, match="upstream down"):
        await breaker.call(operation)

    assert breaker.failure_count == 1


# --------------------------------------------------------------------- #
# The bulkhead
# --------------------------------------------------------------------- #


async def test_bulkhead_runs_an_async_callable_object() -> None:
    bulkhead = Bulkhead(BulkheadConfig())
    operation = AsyncCallableObject(returns="isolated")

    result = await bulkhead.execute(operation)

    assert operation.calls == 1
    assert result == "isolated"


async def test_bulkhead_does_not_count_work_it_never_ran() -> None:
    """``executed`` is a metric a caller reads to size a pool."""
    bulkhead = Bulkhead(BulkheadConfig(track_metrics=True))
    operation = AsyncCallableObject(raises=RuntimeError("boom"))

    with pytest.raises(RuntimeError, match="boom"):
        await bulkhead.execute(operation)

    assert bulkhead.metrics is not None
    assert bulkhead.metrics["executed"] == 0, "a failed call was counted as executed"


async def test_bulkhead_releases_its_slot_after_an_async_callable_object() -> None:
    """A slot held by a call that was never awaited is a slot never returned."""
    bulkhead = Bulkhead(BulkheadConfig(max_concurrent=1, queue_timeout=1.0))
    operation = AsyncCallableObject(returns="ok")

    assert await bulkhead.execute(operation) == "ok"
    assert await bulkhead.execute(operation) == "ok"
    assert operation.calls == 2


# --------------------------------------------------------------------- #
# The workflow strategies
# --------------------------------------------------------------------- #


async def test_fallback_strategy_runs_an_async_callable_object() -> None:
    workflow = ErrorRecoveryWorkflow(
        ErrorRecoveryConfig(
            primary_strategy=RecoveryStrategy.FALLBACK,
            fallback_config=FallbackConfig(fallback_value="fallback"),
        )
    )
    operation = AsyncCallableObject(returns="primary")

    result = await workflow.execute(operation)

    assert operation.calls == 1
    assert result == "primary"


async def test_fallback_strategy_falls_back_on_a_failure_it_awaited() -> None:
    """Without the await the primary never fails, so the fallback never runs."""
    workflow = ErrorRecoveryWorkflow(
        ErrorRecoveryConfig(
            primary_strategy=RecoveryStrategy.FALLBACK,
            fallback_config=FallbackConfig(fallback_value="fallback"),
        )
    )
    operation = AsyncCallableObject(raises=RuntimeError("primary failed"))

    assert await workflow.execute(operation) == "fallback"


async def test_compensation_strategy_runs_an_async_callable_object() -> None:
    workflow = ErrorRecoveryWorkflow(
        ErrorRecoveryConfig(
            primary_strategy=RecoveryStrategy.COMPENSATE,
            compensation_config=CompensationConfig(),
        )
    )
    operation = AsyncCallableObject(returns="committed")

    assert await workflow.execute(operation) == "committed"
    assert operation.calls == 1


async def test_an_async_compensation_action_actually_compensates() -> None:
    """The action's side effect is the entire contract.

    A compensation action returning an un-awaited coroutine leaves the
    half-finished operation un-undone while the workflow's ``compensations``
    metric records that it was handled.
    """
    action = RecordingAction()
    workflow = ErrorRecoveryWorkflow(
        ErrorRecoveryConfig(
            primary_strategy=RecoveryStrategy.COMPENSATE,
            compensation_config=CompensationConfig(compensation_actions=[action]),
        )
    )

    with pytest.raises(RuntimeError, match="mid-flight"):
        await workflow.execute(AsyncCallableObject(raises=RuntimeError("mid-flight")))

    assert action.ran, "the compensation action returned a coroutine instead of running"


async def test_deadline_strategy_accepts_a_synchronous_function() -> None:
    """The branch written for synchronous functions cannot run one.

    ``asyncio.create_task`` requires a coroutine and is handed the function's
    *return value*, so every synchronous function raises ``TypeError`` --- the
    one input this branch exists to serve.
    """
    workflow = ErrorRecoveryWorkflow(
        ErrorRecoveryConfig(
            primary_strategy=RecoveryStrategy.DEADLINE,
            global_timeout=5.0,
        )
    )

    def operation() -> str:
        return "done"

    assert await workflow.execute(operation) == "done"


async def test_deadline_strategy_runs_an_async_callable_object() -> None:
    workflow = ErrorRecoveryWorkflow(
        ErrorRecoveryConfig(
            primary_strategy=RecoveryStrategy.DEADLINE,
            global_timeout=5.0,
        )
    )
    operation = AsyncCallableObject(returns="in time")

    assert await workflow.execute(operation) == "in time"
    assert operation.calls == 1


async def test_direct_execution_runs_an_async_callable_object() -> None:
    """The ``else`` arm: no strategy configured beyond calling the thing."""
    workflow = ErrorRecoveryWorkflow(ErrorRecoveryConfig(primary_strategy=RecoveryStrategy.CACHE))
    operation = AsyncCallableObject(returns="direct")

    assert await workflow.execute(operation) == "direct"
    assert operation.calls == 1


async def test_a_sync_callable_object_still_works_everywhere() -> None:
    """The other half of the shape: an object whose ``__call__`` is a plain def.

    ``iscoroutinefunction`` and ``is_async_callable`` agree on this one, so it
    is a regression guard rather than a reproduction --- the fix must not
    start awaiting things that were never awaitable.
    """

    class SyncCallableObject:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, *args: Any, **kwargs: Any) -> str:
            self.calls += 1
            return "sync"

    operation = SyncCallableObject()

    assert await CircuitBreaker(CircuitBreakerConfig()).call(operation) == "sync"
    assert await Bulkhead(BulkheadConfig()).execute(operation) == "sync"
    assert (
        await ErrorRecoveryWorkflow(
            ErrorRecoveryConfig(primary_strategy=RecoveryStrategy.CACHE)
        ).execute(operation)
        == "sync"
    )
    assert operation.calls == 3


async def test_arguments_reach_the_operation_through_every_wrapper() -> None:
    """Each wrapper forwards ``*args``/``**kwargs``; the fix must keep doing so."""
    seen: list[tuple[Any, ...]] = []

    async def operation(*args: Any, **kwargs: Any) -> Any:
        seen.append((args, tuple(sorted(kwargs.items()))))
        return "ok"

    await CircuitBreaker(CircuitBreakerConfig()).call(operation, 1, key="value")
    await Bulkhead(BulkheadConfig()).execute(operation, 1, key="value")
    await APICircuitBreaker(threshold=2, timeout=60.0).call(operation, 1, key="value")

    assert seen == [((1,), (("key", "value"),))] * 3
    assert not asyncio.get_running_loop().is_closed()
