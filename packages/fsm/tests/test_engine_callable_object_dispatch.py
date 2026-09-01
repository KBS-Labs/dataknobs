"""The async engine asks the async question four times, in four spellings.

Two of them ask ``asyncio.iscoroutinefunction`` bare, which reports a callable
*object* whose ``__call__`` is an ``async def`` as synchronous. Two more
compensate by asking a second time about ``__call__`` --- correctly, as far as
they go, and each in its own hand-written way. That is four maintained copies
of one judgement that is published once as
:func:`~dataknobs_common.callbacks.is_async_callable`, and copies drift: none
of the four unwraps a ``functools.partial``, which
``is_async_callable`` does, and binding a leading argument onto a stateful
transform is an ordinary thing for a consumer to do.

**The arc condition is where this costs the most.** ``_evaluate_arc`` ends in
``return bool(result)``, and a coroutine is truthy. A condition that says no is
therefore read as yes --- for every record, uniformly, with nothing raised and
nothing logged. That is not a gate that occasionally leaks; it is a gate that
is not there, wearing the shape of one.

The transform path fails more visibly but no more loudly: the coroutine object
is coalesced into ``context.data`` in place of the transformed record.

The two sites that already handled callable objects are covered here as
regression guards. Their fix is a delegation rather than a correction, and the
value of writing it that way is that the next thing ``is_async_callable``
learns arrives here without anyone remembering these four sites exist.
"""

from __future__ import annotations

from functools import partial
from typing import Any

from dataknobs_fsm.config.builder import FSMBuilder
from dataknobs_fsm.config.loader import ConfigLoader
from dataknobs_fsm.execution.async_engine import AsyncExecutionEngine
from dataknobs_fsm.execution.context import ExecutionContext


class AsyncFieldGate:
    """An arc condition written as an object, because it counts its calls.

    A consumer's real one holds a compiled rule set, a cache, or a client.
    """

    def __init__(self, field: str) -> None:
        self.field = field
        self.calls = 0

    async def __call__(self, data: Any, context: Any = None) -> bool:
        self.calls += 1
        return self.field in data


class AsyncFieldSetter:
    """A transform written as an object, for the same reason."""

    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self, data: Any, context: Any = None, *, mark: str = "seen") -> Any:
        self.calls += 1
        return {**data, mark: True}


def _fsm_with_condition(name: str):
    """Two states, one arc, gated by a registered condition."""
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
                        "condition": {"type": "registered", "name": name},
                    }
                ],
            }
        ],
    }
    builder = FSMBuilder()
    # A placeholder so the reference resolves at build time; the engine's
    # `custom_functions` then supplies the callable actually under test.
    builder.register_function(name, lambda data, context=None: True)
    return builder.build(ConfigLoader().load_from_dict(config))


def _fsm_with_transform(name: str):
    """Two states, one arc, carrying a registered transform."""
    config = {
        "name": "arc_transform_fsm",
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
                        "transform": {"type": "registered", "name": name},
                    }
                ],
            }
        ],
    }
    builder = FSMBuilder()
    builder.register_function(name, lambda data, context=None: data)
    return builder.build(ConfigLoader().load_from_dict(config))


async def _run(fsm, functions: dict[str, Any], data: dict[str, Any]):
    engine = AsyncExecutionEngine(fsm, custom_functions=functions)
    context = ExecutionContext()
    context.data = data
    success, _ = await engine.execute(context)
    return success, context


# --------------------------------------------------------------------- #
# The arc condition
# --------------------------------------------------------------------- #


async def test_an_async_callable_object_condition_can_refuse() -> None:
    """The gate that is not there: ``bool(coroutine)`` is ``True``.

    The passing half of this pair proves nothing on its own --- a condition
    read as unconditionally true also lets the matching record through. It is
    the *blocked* record that shows the coroutine was awaited.
    """
    gate = AsyncFieldGate("flag")

    success, context = await _run(_fsm_with_condition("gate"), {"gate": gate}, {"flag": 1})
    assert success
    assert context.current_state == "end"

    blocked_gate = AsyncFieldGate("flag")
    success, context = await _run(_fsm_with_condition("gate"), {"gate": blocked_gate}, {"other": 1})
    assert not success, "a condition that said no was read as yes"
    assert context.current_state == "start"


async def test_a_partial_bound_async_condition_can_refuse() -> None:
    """``partial`` is the case none of the four hand-written copies covers.

    ``iscoroutinefunction`` unwraps a partial around a *function* and cannot
    unwrap one around an object, and asking about ``partial.__call__`` answers
    about the C dispatcher rather than about the wrapped callable.
    """
    gate = partial(AsyncFieldGate("flag"))

    success, context = await _run(_fsm_with_condition("gate"), {"gate": gate}, {"other": 1})

    assert not success, "a partial-bound async condition was read as unconditionally true"
    assert context.current_state == "start"


async def test_a_synchronous_condition_still_gates() -> None:
    """Regression guard: the fix must not change the shape that worked."""

    def gate(data: Any, context: Any = None) -> bool:
        return "flag" in data

    success, _ = await _run(_fsm_with_condition("gate"), {"gate": gate}, {"flag": 1})
    assert success

    success, _ = await _run(_fsm_with_condition("gate"), {"gate": gate}, {"other": 1})
    assert not success


async def test_a_plain_async_function_condition_still_gates() -> None:
    """The other regression guard, for the shape ``iscoroutinefunction`` sees."""

    async def gate(data: Any, context: Any = None) -> bool:
        return "flag" in data

    success, _ = await _run(_fsm_with_condition("gate"), {"gate": gate}, {"flag": 1})
    assert success

    success, _ = await _run(_fsm_with_condition("gate"), {"gate": gate}, {"other": 1})
    assert not success


# --------------------------------------------------------------------- #
# The arc transform
# --------------------------------------------------------------------- #


async def test_a_partial_bound_async_transform_produces_data() -> None:
    """The transform's answer becomes ``context.data``.

    An un-awaited coroutine is coalesced in place of the transformed record,
    so what reaches the next state is not the data at all.
    """
    transform = partial(AsyncFieldSetter(), mark="marked")

    success, context = await _run(_fsm_with_transform("mark"), {"mark": transform}, {"id": 1})

    assert success
    assert context.data.get("marked") is True, (
        "the coroutine object was coalesced in place of the transformed record"
    )


async def test_an_async_callable_object_transform_produces_data() -> None:
    """Regression guard: this site already read ``__call__``, and must keep to."""
    transform = AsyncFieldSetter()

    success, context = await _run(_fsm_with_transform("mark"), {"mark": transform}, {"id": 1})

    assert success
    assert context.data.get("seen") is True
    assert transform.calls == 1


async def test_a_synchronous_transform_still_runs() -> None:
    """Regression guard for the executor-offloaded arm."""

    def transform(data: Any, context: Any = None) -> Any:
        return {**data, "sync": True}

    success, context = await _run(_fsm_with_transform("mark"), {"mark": transform}, {"id": 1})

    assert success
    assert context.data.get("sync") is True


# --------------------------------------------------------------------- #
# _invoke_state_transform, and the hint it no longer reads
# --------------------------------------------------------------------- #


async def test_a_function_wrapper_transform_is_still_awaited() -> None:
    """The ``_is_async`` hint was removed; this is what stood on it.

    ``FunctionWrapper`` announces its own async-ness with an attribute, and
    ``_invoke_state_transform`` used to read it because its detection could
    not see through the wrapper's plain ``def __call__``. Nothing reads the
    attribute now: calling the wrapper hands back a coroutine, and the
    result-level check in ``run_callback_off_loop`` catches that without the
    wrapper having to remember to advertise it.
    """
    from dataknobs_fsm.functions.manager import FunctionWrapper

    calls: list[dict[str, Any]] = []

    async def enrich(data: Any, context: Any = None) -> Any:
        calls.append(data)
        return {**data, "wrapped": True}

    wrapper = FunctionWrapper(enrich, name="enrich")
    assert wrapper.is_async, "precondition: the wrapper knows it is async"

    engine = AsyncExecutionEngine(_fsm_with_transform("mark"))
    context = ExecutionContext()
    context.data = {"id": 1}

    result = await engine._invoke_state_transform(wrapper, context, None, context.data)

    assert result == {"id": 1, "wrapped": True}, "the wrapper's coroutine was not awaited"
    assert calls == [{"id": 1}]


async def test_a_synchronous_function_wrapper_transform_still_runs() -> None:
    """The wrapper's other half, for the same reason."""
    from dataknobs_fsm.functions.manager import FunctionWrapper

    def enrich(data: Any, context: Any = None) -> Any:
        return {**data, "wrapped": True}

    wrapper = FunctionWrapper(enrich, name="enrich")
    assert not wrapper.is_async

    engine = AsyncExecutionEngine(_fsm_with_transform("mark"))
    context = ExecutionContext()
    context.data = {"id": 1}

    result = await engine._invoke_state_transform(wrapper, context, None, context.data)

    assert result == {"id": 1, "wrapped": True}
