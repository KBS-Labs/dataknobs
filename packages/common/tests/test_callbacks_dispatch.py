"""``run_callback`` and ``run_callback_off_loop`` against every callable shape.

The pair exists so that a caller dispatching a consumer's callback does not
have to spell the branch out, and so that the two halves of the judgement ---
*is it async* and *may it block* --- are made once each rather than at every
site. Both are published from ``dataknobs-common``, which every other package
depends on, so their contract is expensive to change once adopted.

Until this file they had no test of their own: they were exercised only
through the two consumers that had adopted them, which meant the shapes those
consumers happen not to pass were unproven. One of those shapes was wrong.

**The async generator function.** ``inspect.iscoroutinefunction`` answers
``False`` for it, so ``is_async_callable`` did too; the sync arm then handed
it to a worker thread, where calling it merely *constructed* an async
generator without running a line of the body. ``inspect.isawaitable`` answers
``False`` for that object, so the result-level net did not catch it either,
and the generator came back as the callback's return value. Silent, and
exactly the failure the pair was written to retire --- one shape over.

The remedy follows ``RetryExecutor.execute_sync``, which meets the same class
of problem and refuses rather than guessing: an un-run generator handed back
as a result is indistinguishable from a real value at the call site, so the
only safe answer is a loud one.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

import pytest

from dataknobs_common.callbacks import (
    is_async_callable,
    run_callback,
    run_callback_off_loop,
)


class StatefulAsyncCallable:
    """An ``async def __call__`` on an object that remembers what it saw.

    The shape ``iscoroutinefunction`` misreads, and the reason
    ``is_async_callable`` exists.
    """

    def __init__(self, returning: Any = None) -> None:
        self.seen: list[Any] = []
        self.thread: str | None = None
        self._returning = returning

    async def __call__(self, item: Any) -> Any:
        self.seen.append(item)
        self.thread = threading.current_thread().name
        return self._returning


class StatefulSyncCallable:
    """The synchronous twin, recording which thread ran it."""

    def __init__(self, returning: Any = None) -> None:
        self.seen: list[Any] = []
        self.thread: str | None = None
        self._returning = returning

    def __call__(self, item: Any) -> Any:
        self.seen.append(item)
        self.thread = threading.current_thread().name
        return self._returning


async def _agen_function(item: Any) -> Any:
    """An ``async def`` with a ``yield`` --- an async *generator* function."""
    yield item


class AsyncGeneratorCallable:
    """An object whose ``__call__`` is an async generator function.

    Both halves of the shape at once: the object wrapper that
    ``iscoroutinefunction`` misreads, and the generator body that
    ``is_async_callable`` misreads.
    """

    async def __call__(self, item: Any) -> Any:
        yield item


def _returns_an_async_generator(item: Any) -> Any:
    """A plain ``def`` that hands back an async generator.

    Not detectable by inspecting the callable --- only by looking at what
    came back, which is the same reason ``run_callback`` judges the result
    rather than the function.
    """
    return _agen_function(item)


# --------------------------------------------------------------------- #
# The async generator hole
# --------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "callback",
    [
        pytest.param(_agen_function, id="async-generator-function"),
        pytest.param(AsyncGeneratorCallable(), id="async-generator-callable-object"),
        pytest.param(_returns_an_async_generator, id="sync-def-returning-async-generator"),
    ],
)
@pytest.mark.asyncio
async def test_an_async_generator_is_refused_rather_than_handed_back(callback: Any) -> None:
    """The defect: the generator came back as if it were the result.

    Every one of these three answers ``False`` to both
    ``inspect.iscoroutinefunction`` and ``inspect.isawaitable``, so before the
    fix each fell through both nets and was returned to the caller un-run. A
    consumer reading that return value sees an ``async_generator`` object
    where it expected data, with nothing raised to say so.
    """
    with pytest.raises(TypeError, match="async generator"):
        await run_callback(callback, "payload")

    with pytest.raises(TypeError, match="async generator"):
        await run_callback_off_loop(callback, "payload")


@pytest.mark.asyncio
async def test_the_refusal_names_the_alternative() -> None:
    """A refusal a consumer cannot act on is only half a fix."""
    with pytest.raises(TypeError) as excinfo:
        await run_callback(_agen_function, "payload")

    message = str(excinfo.value)
    assert "async generator" in message
    assert "run_callback" in message, "the message should name the surface that refused"


def test_is_async_callable_does_not_claim_an_async_generator() -> None:
    """``is_async_callable`` promises an *awaitable*, which this is not.

    Calling an async generator function returns an async generator, and
    ``await``-ing one raises ``TypeError``. So the answer stays ``False``:
    widening it to ``True`` would move the failure from the dispatch helpers,
    which can refuse cleanly, into every caller that branches on the guard
    and then awaits.
    """
    assert is_async_callable(_agen_function) is False
    assert is_async_callable(AsyncGeneratorCallable()) is False


# --------------------------------------------------------------------- #
# The shapes that must keep working
# --------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_run_callback_awaits_an_async_callable_object() -> None:
    callback = StatefulAsyncCallable(returning="done")

    assert await run_callback(callback, "payload") == "done"
    assert callback.seen == ["payload"]


@pytest.mark.asyncio
async def test_run_callback_runs_a_sync_callable_on_the_loop() -> None:
    """The on-loop arm is the point of ``run_callback``, not an accident."""
    callback = StatefulSyncCallable(returning="done")
    loop_thread = threading.current_thread().name

    assert await run_callback(callback, "payload") == "done"
    assert callback.thread == loop_thread


@pytest.mark.asyncio
async def test_run_callback_off_loop_moves_a_sync_callable_to_a_worker() -> None:
    """The whole difference between the two helpers, pinned structurally.

    A thread-identity assertion rather than a blocking-detector one, so it
    holds whether or not ``blockbuster`` is installed.
    """
    callback = StatefulSyncCallable(returning="done")
    loop_thread = threading.current_thread().name

    assert await run_callback_off_loop(callback, "payload") == "done"
    assert callback.thread is not None
    assert callback.thread != loop_thread, "a sync callback must not run on the loop thread"


@pytest.mark.asyncio
async def test_run_callback_off_loop_awaits_an_async_callable_on_the_loop() -> None:
    """An async callback is already cooperative; a thread hop would buy nothing."""
    callback = StatefulAsyncCallable(returning="done")
    loop_thread = threading.current_thread().name

    assert await run_callback_off_loop(callback, "payload") == "done"
    assert callback.thread == loop_thread


@pytest.mark.asyncio
async def test_a_sync_def_returning_a_coroutine_is_awaited() -> None:
    """The shape no amount of inspecting the callable will ever catch.

    ``run_callback`` judges the *result*, which is why it catches this and a
    hand-spelled ``is_async_callable`` branch does not.
    """

    async def inner(item: Any) -> str:
        return f"ran:{item}"

    def outer(item: Any) -> Any:
        return inner(item)

    assert await run_callback(outer, "payload") == "ran:payload"
    assert await run_callback_off_loop(outer, "payload") == "ran:payload"


@pytest.mark.asyncio
async def test_arguments_and_keywords_pass_through() -> None:
    seen: dict[str, Any] = {}

    def callback(a: Any, b: Any, *, c: Any) -> str:
        seen.update({"a": a, "b": b, "c": c})
        return "ok"

    assert await run_callback(callback, 1, 2, c=3) == "ok"
    assert seen == {"a": 1, "b": 2, "c": 3}


@pytest.mark.asyncio
async def test_a_callback_taking_its_own_callback_keyword_is_not_shadowed() -> None:
    """Why the first parameter is positional-only."""

    def callback(*, callback: Any) -> Any:
        return callback

    assert await run_callback(callback, callback="inner") == "inner"
    assert await run_callback_off_loop(callback, callback="inner") == "inner"


@pytest.mark.asyncio
async def test_an_exception_propagates_from_either_arm() -> None:
    """Both arms, because the off-loop one crosses a thread boundary."""

    def sync_boom(item: Any) -> Any:
        raise ValueError(item)

    async def async_boom(item: Any) -> Any:
        raise ValueError(item)

    for helper in (run_callback, run_callback_off_loop):
        with pytest.raises(ValueError, match="payload"):
            await helper(sync_boom, "payload")
        with pytest.raises(ValueError, match="payload"):
            await helper(async_boom, "payload")


@pytest.mark.asyncio
async def test_concurrent_off_loop_dispatches_do_not_serialise_on_one_thread() -> None:
    """Two sync callbacks gathered off the loop overlap rather than queueing."""
    barrier = threading.Barrier(2, timeout=5)

    def blocks_until_its_partner_arrives(item: Any) -> Any:
        barrier.wait()
        return item

    results = await asyncio.gather(
        run_callback_off_loop(blocks_until_its_partner_arrives, "a"),
        run_callback_off_loop(blocks_until_its_partner_arrives, "b"),
    )

    assert sorted(results) == ["a", "b"]
