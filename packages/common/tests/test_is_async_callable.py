"""Asking "is this callable async?" has to survive a callable that is an object.

``inspect.iscoroutinefunction`` answers for *functions*. A callable **object**
whose ``__call__`` is an ``async def`` is an async callable that it reports as
sync --- and that shape is not exotic: it is how anything stateful gets written
(an embedder holding a model handle, a client holding a session, a callback
holding a counter).

Getting the answer wrong is silent in both directions this repo cares about:

* ``CallbackRegistry.fire`` raises ``TypeError`` when an async callback is
  registered, because ``fire`` cannot await one. Misclassified, the callback is
  *called* instead --- producing a coroutine that is dropped on the floor. No
  exception, no callback, no trace.
* A vector ``embedding_fn`` misclassified as sync is offloaded to a thread,
  which returns the coroutine rather than an embedding. That coroutine is then
  written into the record as if it were a vector.

``fire_async`` already had the robust form (``inspect.isawaitable`` on the
result). :func:`is_async_callable` is that judgement made available *before*
the call, which is what a guard needs.
"""

from __future__ import annotations

import functools
import inspect
from typing import Any

import pytest

from dataknobs_common.callbacks import CallbackRegistry, is_async_callable


async def _async_fn(payload: dict[str, Any]) -> None:
    return None


def _sync_fn(payload: dict[str, Any]) -> None:
    return None


class AsyncCallableObject:
    """The shape ``iscoroutinefunction`` gets wrong."""

    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self, payload: dict[str, Any]) -> None:
        self.calls += 1


class SyncCallableObject:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, payload: dict[str, Any]) -> None:
        self.calls += 1


class TestThePredicate:
    """Every shape, classified by whether calling it yields an awaitable."""

    @pytest.mark.parametrize(
        ("candidate", "expected"),
        [
            (_async_fn, True),
            (_sync_fn, False),
            (AsyncCallableObject(), True),
            (SyncCallableObject(), False),
            (functools.partial(_async_fn), True),
            (functools.partial(_sync_fn), False),
            (functools.partial(AsyncCallableObject()), True),
            (functools.partial(SyncCallableObject()), False),
            (functools.partial(functools.partial(AsyncCallableObject())), True),
            (AsyncCallableObject, False),
            (SyncCallableObject, False),
            (lambda payload: None, False),
        ],
        ids=[
            "async-function",
            "sync-function",
            "async-callable-object",
            "sync-callable-object",
            "partial-of-async",
            "partial-of-sync",
            "partial-of-async-object",
            "partial-of-sync-object",
            "nested-partial-of-async-object",
            "the-async-class-itself",
            "the-sync-class-itself",
            "lambda",
        ],
    )
    def test_classification(self, candidate: Any, expected: bool) -> None:
        assert is_async_callable(candidate) is expected

    def test_a_partial_of_an_async_object_really_does_return_a_coroutine(self) -> None:
        """The composition of the two shapes this predicate exists for.

        ``iscoroutinefunction`` unwraps a ``partial`` around a function and
        cannot unwrap one around an object, because ``partial.__call__`` is a C
        dispatcher rather than the wrapped object's. Each half of that was
        already covered above and the composition was not, which is how it
        stayed wrong: it answered ``False`` for a callable that genuinely
        returns a coroutine. Binding arguments onto a stateful embedder is an
        ordinary thing to do.

        The assertion runs the call rather than trusting the classification, so
        the two cannot drift apart.
        """
        bound = functools.partial(AsyncCallableObject(), {"payload": 1})
        coroutine = bound()
        try:
            assert inspect.iscoroutine(coroutine)
            assert is_async_callable(bound) is True
        finally:
            coroutine.close()

    def test_a_class_is_not_async_however_its_instances_call(self) -> None:
        """Calling a class runs ``type.__call__`` and returns an instance.

        Reading ``AsyncCallableObject.__call__`` answers for the *instances*.
        The class itself is a synchronous factory, and treating it as async
        would make a caller await an object that is not awaitable --- loud
        rather than silent, but wrong either way.
        """
        assert not inspect.isawaitable(AsyncCallableObject())
        assert is_async_callable(AsyncCallableObject) is False

    def test_a_non_callable_is_not_async(self) -> None:
        """Answering rather than raising: callers ask about arbitrary values."""
        assert is_async_callable(None) is False
        assert is_async_callable(42) is False


class TestTheGuardThatDependsOnIt:
    """``fire()`` must refuse a callback it cannot await, whatever its shape."""

    def test_an_async_callable_object_is_refused_by_fire(self) -> None:
        registry: CallbackRegistry = CallbackRegistry()
        callback = AsyncCallableObject()
        registry.register("topic", callback)

        with pytest.raises(TypeError, match="async callback"):
            registry.fire("topic", {})

        assert callback.calls == 0, "the callback was invoked and its coroutine dropped"

    def test_an_async_function_is_still_refused(self) -> None:
        """A companion: the case that already worked keeps working."""
        registry: CallbackRegistry = CallbackRegistry()
        registry.register("topic", _async_fn)

        with pytest.raises(TypeError, match="async callback"):
            registry.fire("topic", {})

    def test_a_sync_callable_object_still_fires(self) -> None:
        """A companion: the guard must not start refusing sync objects."""
        registry: CallbackRegistry = CallbackRegistry()
        callback = SyncCallableObject()
        registry.register("topic", callback)

        registry.fire("topic", {})

        assert callback.calls == 1

    async def test_fire_async_awaits_an_async_callable_object(self) -> None:
        """A companion: ``fire_async`` already handled this and must keep doing so."""
        registry: CallbackRegistry = CallbackRegistry()
        callback = AsyncCallableObject()
        registry.register("topic", callback)

        await registry.fire_async("topic", {})

        assert callback.calls == 1
