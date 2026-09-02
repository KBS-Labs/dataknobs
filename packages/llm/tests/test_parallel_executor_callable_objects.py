"""A ``DeterministicTask`` whose ``fn`` is a callable *object* must be dispatched
by what calling it produces, not by what ``iscoroutinefunction`` says about it.

``DeterministicTask.fn`` is documented as "May be sync or async", and a stateful
async callable — an object holding a model handle, a session, a counter — is
written as a class with ``async def __call__``. ``asyncio.iscoroutinefunction``
answers ``False`` for that shape, so the executor took the sync branch, ran the
object on a worker thread (where calling it only *constructs* a coroutine), and
returned that coroutine inside a ``TaskResult`` whose ``success`` flag was
``True``. Nothing raised and the callable's body never ran.

The shape is not exotic here: dataknobs publishes two Protocols declared with
``async def __call__`` — ``IngestionManagerResolver`` and ``VectorQueryFn`` — so
a consumer implementing one of our own extension surfaces writes exactly it.

Plain ``def`` and plain ``async def`` callables are covered by
``test_parallel_executor.py::test_mixed_execution`` and are deliberately not
repeated here.
"""

import inspect
import threading

import pytest

from dataknobs_llm import (
    DeterministicTask,
    EchoProvider,
    ParallelLLMExecutor,
)


@pytest.fixture
def provider() -> EchoProvider:
    """Create a fresh EchoProvider for testing."""
    return EchoProvider({"provider": "echo", "model": "test"})


class AsyncCallableObject:
    """A stateful async callable — the shape our own Protocols publish."""

    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self, x: int) -> int:
        self.calls += 1
        return x * 2


class SyncCallableObject:
    """A stateful sync callable, which must keep running off the loop."""

    def __init__(self) -> None:
        self.calls = 0
        self.thread_name: str | None = None

    def __call__(self, x: int) -> int:
        self.calls += 1
        self.thread_name = threading.current_thread().name
        return x * 2


async def _double(x: int) -> int:
    return x * 2


def returns_a_coroutine(x: int) -> object:
    """A plain ``def`` that returns a coroutine without being one.

    No inspection of the *callable* can see this; only judging the result can.
    """
    return _double(x)


def _value_of(result: object) -> object:
    """Unwrap a ``TaskResult``, failing loudly if it carries a coroutine.

    A coroutine reaching here means the callable was dispatched as sync and its
    body never ran. Close it before failing so the assertion is what the run
    reports, rather than a "never awaited" warning from the collector.
    """
    value = result.value  # type: ignore[attr-defined]
    if inspect.iscoroutine(value):
        value.close()
        pytest.fail(
            "the callable was dispatched as sync: the task returned an "
            "un-awaited coroutine instead of its result"
        )
    return value


@pytest.mark.asyncio
async def test_async_callable_object_runs_and_returns_its_value(
    provider: EchoProvider,
) -> None:
    """An ``async def __call__`` object is awaited, not shipped to a thread."""
    executor = ParallelLLMExecutor(provider, max_concurrency=5)
    obj = AsyncCallableObject()

    results = await executor.execute_mixed({"double": DeterministicTask(fn=obj, args=(21,))})

    assert results["double"].success is True
    assert _value_of(results["double"]) == 42
    # The counter is the half a value assertion cannot make: pre-fix the task
    # was reported successful while the object had never been entered.
    assert obj.calls == 1


@pytest.mark.asyncio
async def test_async_callable_object_under_a_timeout(provider: EchoProvider) -> None:
    """The same, on the ``asyncio.wait_for`` path rather than the bare await."""
    executor = ParallelLLMExecutor(provider, max_concurrency=5)
    obj = AsyncCallableObject()

    results = await executor.execute_mixed(
        {"double": DeterministicTask(fn=obj, args=(21,), timeout=10.0)}
    )

    assert results["double"].success is True
    assert _value_of(results["double"]) == 42
    assert obj.calls == 1


@pytest.mark.asyncio
async def test_sync_callable_object_still_runs_off_the_loop(
    provider: EchoProvider,
) -> None:
    """Fixing the async case must not pull sync callables onto the event loop.

    Asserted structurally rather than by return value, which is identical
    either way: the call must land on a worker thread.
    """
    executor = ParallelLLMExecutor(provider, max_concurrency=5)
    obj = SyncCallableObject()
    loop_thread = threading.current_thread().name

    results = await executor.execute_mixed({"double": DeterministicTask(fn=obj, args=(21,))})

    assert results["double"].success is True
    assert _value_of(results["double"]) == 42
    assert obj.calls == 1
    assert obj.thread_name is not None
    assert obj.thread_name != loop_thread


@pytest.mark.asyncio
async def test_plain_function_returning_a_coroutine_is_awaited(
    provider: EchoProvider,
) -> None:
    """The shape no predicate over the callable can see.

    ``returns_a_coroutine`` is an ordinary ``def``, so every inspection of the
    function itself is correct to call it sync. Only judging what the call
    produced catches it.
    """
    executor = ParallelLLMExecutor(provider, max_concurrency=5)

    results = await executor.execute_mixed(
        {"double": DeterministicTask(fn=returns_a_coroutine, args=(21,))}
    )

    assert results["double"].success is True
    assert _value_of(results["double"]) == 42
