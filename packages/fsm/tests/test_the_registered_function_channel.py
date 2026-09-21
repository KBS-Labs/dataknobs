# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""What may be handed to an FSM by name, and what it may hand back.

Three declarations described nothing that exists, and this file pins what
replaced them.

``ITransformFunction.transform`` was declared ``def ... -> ExecutionResult``.
Measured across the tree: 32 implementations, **none** returning an
``ExecutionResult`` and 14 of them ``async def``. ``git log --all -S'return
ExecutionResult' -- packages/fsm/src/dataknobs_fsm/functions/library/``
returns no commit, so no library transform has ever returned one.
``IValidationFunction.validate`` said the same thing, and its 10
implementations all return ``bool``.

``custom_functions`` was declared ``dict[str, Callable]``, but a bare
interface instance is not a ``Callable`` --- it carries its logic on
``transform`` / ``validate`` / ``test``, which is why the tree holds three
separate pieces of machinery for finding that method. Under that annotation
mypy read all four ``isinstance`` arms of
``FunctionWrapper._normalize_interface_callable`` as unreachable.

Real constructs only: real ``AsyncSimpleFSM`` / ``AdvancedFSM`` instances
over real configs.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
from typing import Any, Dict

from dataknobs_fsm.api.advanced import create_advanced_fsm
from dataknobs_fsm.api.async_simple import AsyncSimpleFSM
from dataknobs_fsm.core.data_modes import DataHandlingMode
import pytest

from dataknobs_fsm.functions.base import (
    ExecutionResult,
    TransformError,
    IStateTestFunction,
    ITransformFunction,
    IValidationFunction,
)
from dataknobs_fsm.functions.library.transformers import ChainTransformer
from dataknobs_fsm.functions.manager import FunctionWrapper, InterfaceWrapper


def _config(*, gated: bool = False) -> Dict[str, Any]:
    """One work state running the registered ``fn``, optionally gated."""
    work: Dict[str, Any] = {
        "name": "work",
        "resources": [],
        "functions": {"transform": {"type": "registered", "name": "fn"}},
    }
    if gated:
        work["pre_validators"] = [{"type": "registered", "name": "gate"}]
    return {
        "name": "channel_probe",
        "data_mode": DataHandlingMode.COPY.value,
        "resources": [],
        "states": [
            {"name": "start", "is_start": True, "resources": []},
            work,
            {"name": "done", "is_end": True},
        ],
        "arcs": [
            {"from": "start", "to": "work", "name": "go"},
            {"from": "work", "to": "done", "name": "fin"},
        ],
    }


async def _run(functions: Dict[str, Any], data: Dict[str, Any], *, gated: bool = False) -> Any:
    fsm = AsyncSimpleFSM(
        _config(gated=gated), data_mode=DataHandlingMode.COPY, custom_functions=functions
    )
    return await fsm.process(dict(data))


# --------------------------------------------------------------------------- #
# TransformOutcome --- each member, sync and async
# --------------------------------------------------------------------------- #


class _ReturnsDict(ITransformFunction):
    """The shape all 28 annotated implementations use."""

    def transform(self, data: Any, context: Any = None) -> Dict[str, Any]:
        return {**data, "seen": "dict"}

    def get_transform_description(self) -> str:
        return "returns a dict"


class _ReturnsDictAsync(ITransformFunction):
    """The same, ``async def`` --- 14 of the 32 are."""

    async def transform(self, data: Any, context: Any = None) -> Dict[str, Any]:
        return {**data, "seen": "dict-async"}

    def get_transform_description(self) -> str:
        return "returns a dict, asynchronously"


class _ReturnsExecutionResult(ITransformFunction):
    """The member nothing in the tree uses --- and the engines do honour it."""

    def transform(self, data: Any, context: Any = None) -> ExecutionResult:
        return ExecutionResult.success_result({**data, "seen": "result"})

    def get_transform_description(self) -> str:
        return "returns an ExecutionResult"


class _ReturnsNone(ITransformFunction):
    """``None`` means "I mutated the record in place", not "I produced nothing"."""

    def transform(self, data: Any, context: Any = None) -> None:
        data["seen"] = "in-place"

    def get_transform_description(self) -> str:
        return "mutates in place"


async def test_a_transform_may_return_the_record() -> None:
    """The ordinary case, and the one every shipped implementation takes."""
    result = await _run({"fn": _ReturnsDict()}, {"id": 1})

    assert result["success"], result
    assert result["data"]["seen"] == "dict", result["data"]


async def test_a_transform_may_be_async() -> None:
    """``async def`` is not a deviation; the declaration admits both flavours.

    The engines route every invocation through ``run_callback_off_loop``,
    which awaits an async implementation and offloads a sync one, so neither
    is privileged. The declaration said ``def``, and 14 shipped
    implementations disagreed with it.
    """
    result = await _run({"fn": _ReturnsDictAsync()}, {"id": 1})

    assert result["success"], result
    assert result["data"]["seen"] == "dict-async", result["data"]


async def test_a_transform_may_return_an_execution_result() -> None:
    """The member the old declaration named --- kept, because it works.

    Nothing in the tree returns one, which is why the declaration could be
    wrong for a year without anything breaking. It stays in the union
    because both engines unwrap it, so removing it would break a consumer
    who took the interface at its word.
    """
    result = await _run({"fn": _ReturnsExecutionResult()}, {"id": 1})

    assert result["success"], result
    assert result["data"]["seen"] == "result", result["data"]


async def test_a_transform_may_return_none_and_keep_the_record() -> None:
    """``None`` is the in-place signal, so the record must survive it."""
    result = await _run({"fn": _ReturnsNone()}, {"id": 1})

    assert result["success"], result
    assert result["data"]["id"] == 1, f"the record was dropped, not preserved: {result['data']}"


# --------------------------------------------------------------------------- #
# ValidationOutcome --- what a gate may say
# --------------------------------------------------------------------------- #


class _Gate(IValidationFunction):
    """A validator of the shape all 10 shipped ones have: returns ``bool``."""

    def validate(self, data: Any, context: Any = None) -> bool:
        return bool(data.get("ok"))

    def get_validation_rules(self) -> Dict[str, Any]:
        return {"requires": "ok"}


async def test_a_validator_returning_false_stops_the_record() -> None:
    """``False`` is the only value that fails a record, and it must."""
    result = await _run({"fn": _ReturnsDict(), "gate": _Gate()}, {"ok": False}, gated=True)

    assert "seen" not in result["data"], (
        f"the gate said no and the transform ran anyway: {result['data']}"
    )


async def test_a_validator_returning_true_passes_the_record() -> None:
    """The other half of the gate, so the test above cannot pass by inertia."""
    result = await _run({"fn": _ReturnsDict(), "gate": _Gate()}, {"ok": True}, gated=True)

    assert result["data"].get("seen") == "dict", result["data"]


def test_a_wrapped_validator_answers_within_the_declared_outcome() -> None:
    """Whatever route reaches a wrapped validator, the answer is a ``ValidationOutcome``.

    ``InterfaceWrapper`` wrapped every non-``None`` answer from ``validate``
    into ``ExecutionResult.success_result(...)`` --- including ``False``. An
    ``ExecutionResult`` is deliberately not a member of
    :data:`~dataknobs_fsm.functions.base.ValidationOutcome`, precisely because
    no validator path unwraps one: a wrapped ``False`` is neither ``False`` nor
    a dict, so the gate reads it as a pass. Wrapping was therefore a validator
    that could not reject, latent only because the pre-validator route reached
    the wrapper's ``__call__`` and the ``__call__`` skipped the wrapping.

    Both routes are checked here, since the defect was that they disagreed.
    """
    wrapper = InterfaceWrapper(FunctionWrapper(_Gate(), "gate"), IValidationFunction)

    for label, answer in (
        ("validate", wrapper.validate({"ok": False}, None)),
        ("__call__", wrapper({"ok": False}, None)),
    ):
        assert answer is False, f"{label} answered {answer!r}, which no gate reads as a refusal"

    for label, answer in (
        ("validate", wrapper.validate({"ok": True}, None)),
        ("__call__", wrapper({"ok": True}, None)),
    ):
        assert answer is True, f"{label} answered {answer!r}"


# --------------------------------------------------------------------------- #
# RegisteredFunction --- the channel carries instances, not only callables
# --------------------------------------------------------------------------- #


class _OpenGate(IStateTestFunction):
    """An arc condition as a bare interface instance."""

    def test(self, data: Any, context: Any = None) -> tuple[bool, str | None]:
        return True, None

    def get_test_description(self) -> str:
        return "always open"


def _open_gate_callable(data: Any, context: Any = None) -> bool:
    """The same answer as :class:`_OpenGate`, in the shape that always worked."""
    return True


_PRE_TEST_CONFIG: Dict[str, Any] = {
    "name": "pre_test_probe",
    "version": "1.0",
    "main_network": "main",
    "networks": [
        {
            "name": "main",
            "states": [
                {"name": "start", "is_start": True},
                {"name": "end", "is_end": True},
            ],
            "arcs": [{"from": "start", "to": "end", "pre_test": "gate"}],
        }
    ],
}


def _step_with_gate(gate: Any) -> str | None:
    with create_advanced_fsm(_PRE_TEST_CONFIG, custom_functions={"gate": gate}) as fsm:
        return fsm.execute_step_sync(fsm.create_context({"value": 1})).to_state


def test_a_callable_arc_pre_test_opens_the_arc() -> None:
    """The control for the test below: this shape has always worked."""
    assert _step_with_gate(_open_gate_callable) == "end"


def test_an_interface_arc_pre_test_opens_the_arc() -> None:
    """A bare ``IStateTestFunction`` is not callable, and the arc used to vanish.

    ``AdvancedFSM._resolve_test_function`` handed the registered object
    straight to its two call sites, which invoke it as ``func(data,
    context)``. The resulting ``TypeError`` was swallowed by the
    arc-skipping ``except Exception``, so a condition that said *yes* read
    as *no* and the FSM simply did not move --- with ``success`` still True,
    so nothing surfaced.

    ``as_state_test_callable`` exists for exactly this; its docstring named
    the async engine's ``custom_functions`` merge as "the one path that
    bypasses it", and this was a second.
    """
    assert _step_with_gate(_OpenGate()) == "end", (
        "the arc was dropped, so a bare IStateTestFunction condition read as False"
    )


# --------------------------------------------------------------------------- #
# A consumer of TransformOutcome has to honour all of it
# --------------------------------------------------------------------------- #


def test_a_chain_keeps_a_record_a_link_mutated_in_place() -> None:
    """``ChainTransformer`` feeds each link's answer to the next one.

    ``None`` means "I mutated the record in place", so a link that returns it
    must not blank the chain. Passing the raw ``None`` on made the *next* link
    receive nothing and the chain return nothing --- which the engines then
    read as "mutated in place" against the *original* record, silently
    discarding every link in the chain.
    """
    chain = ChainTransformer([_ReturnsNone(), _ReturnsDict()])

    result = chain.transform({"id": 1})

    assert result == {"id": 1, "seen": "dict"}, result


def test_a_chain_unwraps_a_link_that_returns_an_execution_result() -> None:
    """The other member of the union, which the engines unwrap and this did not."""
    chain = ChainTransformer([_ReturnsExecutionResult(), _ReturnsDict()])

    result = chain.transform({"id": 1})

    assert result == {"id": 1, "seen": "dict"}, result


def test_a_chain_refuses_an_async_link() -> None:
    """A synchronous chain cannot await, so it says so instead of passing it on.

    ``transform`` may be written ``async def`` --- 14 shipped implementations
    are --- and a chain that handed the coroutine to the next link produced a
    record that was a coroutine object, with nothing raised and the coroutine
    never awaited.
    """
    chain = ChainTransformer([_ReturnsDictAsync()])

    with pytest.raises(TransformError, match="async"):
        chain.transform({"id": 1})


# --------------------------------------------------------------------------- #
# A wrapper around an async function must read as one
# --------------------------------------------------------------------------- #


def test_a_wrapper_declares_the_async_flavour_it_reports() -> None:
    """``wrapper.is_async`` and the two detectors must never disagree.

    ``FunctionWrapper`` is not itself an ``async def``, so a caller asking
    ``asyncio.iscoroutinefunction(wrapper)`` gets the wrong answer unless the
    wrapper says otherwise. It used to say so by handing back
    ``asyncio.coroutines._is_coroutine`` from ``__getattr__`` --- a private
    sentinel typeshed does not declare and CPython removes in 3.14. The public
    ``inspect.markcoroutinefunction`` is read by both detectors.

    The last two cases are the load-bearing ones, and they are the two
    ``FunctionWrapper._check_async`` was itself fixed for: a wrapper over an
    ``async def`` *function* reads as async anyway, because ``__getattr__``
    forwards ``__code__`` to the wrapped function and its ``CO_COROUTINE``
    flag answers for it. An async callable *object* has no ``__code__`` to
    forward, so without the marking the wrapper reports ``is_async`` True and
    both detectors say False --- it disagrees with itself.
    """

    async def async_fn(value: int) -> int:
        return value

    def sync_fn(value: int) -> int:
        return value

    class AsyncCallable:
        async def __call__(self, value: int) -> int:
            return value

    class SyncCallable:
        def __call__(self, value: int) -> int:
            return value

    cases: list[tuple[str, Any, bool]] = [
        ("async def function", async_fn, True),
        ("sync def function", sync_fn, False),
        ("object with async __call__", AsyncCallable(), True),
        ("object with sync __call__", SyncCallable(), False),
        ("partial over an async object", functools.partial(AsyncCallable()), True),
    ]

    for label, func, expected in cases:
        wrapper = FunctionWrapper(func, label)
        assert wrapper.is_async is expected, f"{label}: is_async"
        assert asyncio.iscoroutinefunction(wrapper) is expected, (
            f"{label}: asyncio.iscoroutinefunction disagrees with is_async={wrapper.is_async}"
        )
        assert inspect.iscoroutinefunction(wrapper) is expected, (
            f"{label}: inspect.iscoroutinefunction disagrees with is_async={wrapper.is_async}"
        )
