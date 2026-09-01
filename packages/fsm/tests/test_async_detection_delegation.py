"""Three places in this package answer "is this callable async?" for themselves.

The question has one right answer and it is published from
``dataknobs_common.callbacks`` as :func:`is_async_callable`. Each site here
re-derives it, and each re-derivation is missing something the shared one
handles --- which is what independently reimplemented judgement always looks
like after a while.

**A wrapper that returns a wrapper.**
``normalize_record_callable`` picks a sync or async adapter with bare
``inspect.iscoroutinefunction``, so a callable *object* gets the sync adapter,
whose ``coerce(out)`` is then applied to a coroutine. For the gate that means
``bool(coroutine)`` --- ``True`` unconditionally, for every record, which is a
validator that validates nothing while reporting that it did.

**A hand-rolled copy of the shared predicate.**
``FunctionWrapper._check_async`` asks ``iscoroutinefunction`` and then, for
non-function callables, asks it again about ``__call__``. That is
:func:`is_async_callable` minus its ``functools.partial`` unwrapping --- and
binding arguments onto a stateful callable is an ordinary thing to do, so
``partial(client, endpoint)`` is detected as synchronous and dispatched to
``run_in_executor``, where calling it merely builds a coroutine that nothing
awaits.

**An adapter chosen from the wrong question.** The config builder selects
between a sync and an async resolved-function adapter by asking
``iscoroutinefunction`` about the interface method it found. A custom function
class implementing that method with a callable object gets the sync adapter
and stores an un-awaited coroutine as its result.

All three are one-line delegations. The value of writing them as delegation
rather than as three corrected copies is that the next thing
``is_async_callable`` learns --- as it learned ``partial`` --- arrives here
without anyone remembering that these sites exist.
"""

from __future__ import annotations

from functools import partial
from typing import Any

from dataknobs_common.callbacks import is_async_callable

from dataknobs_fsm.functions.library._callables import normalize_record_callable
from dataknobs_fsm.functions.manager import FunctionWrapper


class AsyncRecordCallable:
    """``record -> value``, written as an object because it holds a count."""

    def __init__(self, returns: Any = True) -> None:
        self.calls = 0
        self._returns = returns

    async def __call__(self, record: dict) -> Any:
        self.calls += 1
        return self._returns


class AsyncClient:
    """The ``partial``-bound shape: a stateful callable with fixed leading args."""

    def __init__(self) -> None:
        self.seen: list[Any] = []

    async def __call__(self, endpoint: str, payload: Any) -> str:
        self.seen.append((endpoint, payload))
        return f"{endpoint}:{payload}"


# --------------------------------------------------------------------- #
# normalize_record_callable
# --------------------------------------------------------------------- #


async def test_a_callable_object_gets_the_async_adapter() -> None:
    """The adapter's own docstring promises this: async iff ``fn`` is."""
    fn = AsyncRecordCallable(returns="enriched")
    normalized = normalize_record_callable(fn)

    assert is_async_callable(normalized), (
        "a callable object was normalized into a sync adapter, so the engine's "
        "own async check will route it to the sync path"
    )
    assert await normalized({"id": 1}, None) == "enriched"
    assert fn.calls == 1


async def test_a_coerced_gate_does_not_coerce_a_coroutine() -> None:
    """``bool(coroutine)`` is ``True``, so the gate admits everything.

    The failure is invisible at the call site --- a gate returning ``True`` is
    what a passing record looks like --- and it is uniform, so no record ever
    reveals it by being rejected.
    """
    fn = AsyncRecordCallable(returns=False)
    gate = normalize_record_callable(fn, coerce=bool)

    assert await gate({"id": 1}, None) is False, (
        "the predicate said no and the gate said yes: `bool` was applied to a "
        "coroutine rather than to the predicate's answer"
    )


async def test_the_two_argument_arity_still_reaches_a_callable_object() -> None:
    """Arity detection reads the signature; ``__call__``'s is the right one."""

    class TwoArgCallable:
        def __init__(self) -> None:
            self.context_seen: Any = "not called"

        async def __call__(self, record: dict, context: Any) -> str:
            self.context_seen = context
            return "two-arg"

    fn = TwoArgCallable()
    normalized = normalize_record_callable(fn)

    assert await normalized({"id": 1}, "ctx") == "two-arg"
    assert fn.context_seen == "ctx"


async def test_a_plain_async_function_is_unchanged() -> None:
    """Regression guard: the shape that already worked must keep working."""

    async def fn(record: dict) -> str:
        return "plain"

    normalized = normalize_record_callable(fn)

    assert is_async_callable(normalized)
    assert await normalized({"id": 1}, None) == "plain"


def test_a_plain_sync_function_stays_synchronous() -> None:
    """The other regression guard: nothing becomes async that was not."""

    def fn(record: dict) -> str:
        return "plain"

    normalized = normalize_record_callable(fn)

    assert not is_async_callable(normalized)
    assert normalized({"id": 1}, None) == "plain"


# --------------------------------------------------------------------- #
# FunctionWrapper._check_async
# --------------------------------------------------------------------- #


async def test_a_partial_bound_async_callable_is_detected_as_async() -> None:
    """``partial`` around an *object* is the gap in the hand-rolled copy.

    ``iscoroutinefunction`` unwraps a partial around a *function* and cannot
    unwrap one around an object --- ``partial.__call__`` is a C dispatcher, so
    asking it about ``__call__`` answers about the wrong object. Detected as
    synchronous, the wrapper hands it to ``run_in_executor``, where calling it
    constructs a coroutine that the executor returns and nobody awaits.
    """
    client = AsyncClient()
    bound = partial(client, "/users")
    wrapper = FunctionWrapper(bound, name="fetch_users")

    assert wrapper.is_async, "a partial-bound async callable was read as synchronous"
    assert await wrapper.execute_async({"id": 1}) == "/users:{'id': 1}"
    assert client.seen == [("/users", {"id": 1})]


async def test_a_bare_async_callable_object_is_detected_as_async() -> None:
    """The shape the hand-rolled copy did already handle. Regression guard."""
    wrapper = FunctionWrapper(AsyncRecordCallable(returns="ok"), name="enrich")

    assert wrapper.is_async
    assert await wrapper.execute_async({"id": 1}) == "ok"


def test_a_partial_bound_sync_callable_stays_synchronous() -> None:
    """The fix must not start awaiting what was never awaitable."""

    class SyncClient:
        def __call__(self, endpoint: str, payload: Any) -> str:
            return f"{endpoint}:{payload}"

    wrapper = FunctionWrapper(partial(SyncClient(), "/users"), name="fetch")

    assert not wrapper.is_async
    assert wrapper.execute_sync({"id": 1}) == "/users:{'id': 1}"


def test_a_class_is_not_read_as_an_async_callable() -> None:
    """Calling a class returns an instance, never an awaitable.

    A third gap in the hand-rolled copy, and one this file was not written
    expecting: ``SomeClass.__call__`` reached through the *class* is the plain
    unbound function, which really is a coroutine function --- so the copy
    answered ``True`` for a class whose instances are async. Calling the class
    runs ``type.__call__`` and hands back an instance, so the wrapper then
    awaited a constructor. ``is_async_callable`` rules classes out by name.
    """

    class Constructed:
        async def __call__(self) -> None: ...

    assert not FunctionWrapper(Constructed, name="ctor").is_async
