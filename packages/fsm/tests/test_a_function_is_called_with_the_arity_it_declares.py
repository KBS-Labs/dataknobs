# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""Every door calls a function with the number of arguments it declares.

``ITransformFunction.transform`` and ``IValidationFunction.validate`` declare
``(data, context=None)``. Measured across the workspace: **42** implementations,
of which **28** omit ``context`` entirely --- ``transformers.py`` 9,
``validators.py`` 9, ``streaming.py`` 4, and ``dataknobs_llm``'s
``fsm_integration/functions.py`` 6. mypy reported every one as an ``override``
incompatibility, which is the type checker saying the shipped library does not
implement the interface it declares.

That mattered because the two doors into an FSM disagreed about what to do with
a one-argument implementation:

* the **config** door (``{"type": "builtin"}`` / ``{"type": "custom"}``) adapts
  it --- ``_ResolvedLibraryFunction`` introspects the arity and passes the
  record alone;
* the **registration** door (``custom_functions=``) did not. It ran the record
  through ``InterfaceWrapper``, whose arity heuristic reads a one-argument
  function as an *inline lambda* wanting a ``StateDataWrapper``
  (``wrap_for_lambda``), and handed the interface method that wrapper instead of
  the ``dict`` its own signature asks for.

So the same shipped ``FieldMapper`` ran through one door and failed through the
other. The tests below pin both doors, each with the opposite-answer control
that keeps it from passing by inertia, and pin the inline-lambda convention the
heuristic exists to serve so the fix cannot be bought at its expense.

Real constructs only: real ``SimpleFSM`` / ``AsyncSimpleFSM`` over real configs,
and the shipped library classes rather than stand-ins for them.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from dataknobs_fsm.api.async_simple import AsyncSimpleFSM
from dataknobs_fsm.api.simple import SimpleFSM
from dataknobs_fsm.core.data_modes import DataHandlingMode
from dataknobs_fsm.core.data_wrapper import StateDataWrapper
from dataknobs_fsm.functions.base import (
    IStateTestFunction,
    ITransformFunction,
    IValidationFunction,
)
from dataknobs_fsm.functions.library.transformers import FieldMapper
from dataknobs_fsm.functions.library.validators import RequiredFieldsValidator

MAP_A_TO_B = {"a": "b"}


def _config(
    *,
    transform: Dict[str, Any] | None = None,
    gate: bool = False,
    condition: bool = False,
) -> Dict[str, Any]:
    """One ``work`` state, optionally gated and optionally reached by condition."""
    work: Dict[str, Any] = {"name": "work", "resources": []}
    if transform is not None:
        work["functions"] = {"transform": transform}
    if gate:
        work["pre_validators"] = [{"type": "registered", "name": "gate"}]

    entry: Dict[str, Any] = {"from": "start", "to": "work", "name": "go"}
    if condition:
        entry["condition"] = {"type": "registered", "name": "cond"}

    return {
        "name": "arity_probe",
        "data_mode": DataHandlingMode.COPY.value,
        "resources": [],
        "states": [
            {"name": "start", "is_start": True, "resources": []},
            work,
            {"name": "done", "is_end": True},
        ],
        "arcs": [entry, {"from": "work", "to": "done", "name": "fin"}],
    }


REGISTERED = {"type": "registered", "name": "fn"}
#: The same ``FieldMapper``, reached through the config door instead.
BUILTIN = {
    "type": "builtin",
    "name": "transformers.map_fields",
    "params": {"mapping": MAP_A_TO_B},
}


async def _run_async(config: Dict[str, Any], functions: Dict[str, Any] | None, data: Any) -> Any:
    fsm = AsyncSimpleFSM(config, data_mode=DataHandlingMode.COPY, custom_functions=functions)
    return await fsm.process(dict(data))


def _run_sync(config: Dict[str, Any], functions: Dict[str, Any] | None, data: Any) -> Any:
    fsm = SimpleFSM(config, data_mode=DataHandlingMode.COPY, custom_functions=functions)
    return fsm.process(dict(data))


# --------------------------------------------------------------------------- #
# The shipped library, through both doors
# --------------------------------------------------------------------------- #


async def test_a_shipped_transform_runs_through_the_config_door() -> None:
    """The control: this door introspects the arity, and always has."""
    result = await _run_async(_config(transform=BUILTIN), None, {"a": 1})

    assert result["success"], result
    assert result["data"] == {"b": 1}, result["data"]


async def test_a_shipped_transform_runs_through_the_registration_door() -> None:
    """The same class, handed over by name instead of named in the config.

    ``FieldMapper.transform`` asks ``if source in data``, and the
    ``StateDataWrapper`` it was handed defines ``__getitem__`` but neither
    ``__contains__`` nor ``__iter__`` --- so ``in`` falls back to integer
    indexing and dies with ``KeyError: 0``, surfacing as a failed state.
    """
    result = await _run_async(
        _config(transform=REGISTERED), {"fn": FieldMapper(MAP_A_TO_B)}, {"a": 1}
    )

    assert result["success"], result
    assert result["data"] == {"b": 1}, result["data"]


def test_a_shipped_transform_runs_through_the_registration_door_on_the_sync_facade() -> None:
    """Both façades reach the same wrapper, so both have to be pinned."""
    result = _run_sync(_config(transform=REGISTERED), {"fn": FieldMapper(MAP_A_TO_B)}, {"a": 1})

    assert result["success"], result
    assert result["data"] == {"b": 1}, result["data"]


# --------------------------------------------------------------------------- #
# A gate has to be able to say both words
# --------------------------------------------------------------------------- #


class _Marker(ITransformFunction):
    """Records that the gate let the record through."""

    def transform(self, data: Any, context: Any = None) -> Dict[str, Any]:
        return {**data, "seen": True}

    def get_transform_description(self) -> str:
        return "marker"


async def _gated(data: Dict[str, Any]) -> bool:
    """Whether the transform behind a shipped one-argument gate ran."""
    result = await _run_async(
        _config(transform=REGISTERED, gate=True),
        {"fn": _Marker(), "gate": RequiredFieldsValidator(["a"])},
        data,
    )
    return bool(result["data"].get("seen"))


async def test_a_shipped_gate_admits_the_record_it_should() -> None:
    """A one-argument validator rejected *every* record, including valid ones.

    The crash inside the gate reads as a refusal, so the failure was not an
    error a caller could see --- it was a gate that never opened.
    """
    assert await _gated({"a": 1}), "the gate refused a record that satisfies it"


async def test_a_shipped_gate_rejects_the_record_it_should() -> None:
    """The other half, so the test above cannot pass by the gate being absent."""
    assert not await _gated({"z": 1}), "the gate admitted a record that violates it"


# --------------------------------------------------------------------------- #
# An arc condition, which reaches a second copy of the same heuristic
# --------------------------------------------------------------------------- #


class _OneArgCondition(IStateTestFunction):
    """An arc condition written to the one-argument convention."""

    def test(self, data: Any) -> Tuple[bool, str | None]:  # type: ignore[override]
        return ("go" in data, None)

    def get_test_description(self) -> str:
        return "one-arg condition"


async def _conditioned(data: Dict[str, Any]) -> bool:
    result = await _run_async(
        _config(transform=REGISTERED, condition=True),
        {"fn": _Marker(), "cond": _OneArgCondition()},
        data,
    )
    return bool(result["data"].get("seen"))


async def test_a_one_arg_arc_condition_opens_the_arc() -> None:
    """``_create_test_method`` carries its own copy of the arity heuristic.

    It also reads the *wrapper's* signature rather than the implementation's:
    a registered function is wrapped twice, and ``FunctionWrapper.__call__`` is
    ``(*args, **kwargs)``, so the heuristic saw two parameters and called the
    one-argument condition with two.
    """
    assert await _conditioned({"go": 1}), "the arc stayed shut on a condition that said yes"


async def test_a_one_arg_arc_condition_closes_the_arc() -> None:
    """The other half: the condition must still be able to say no."""
    assert not await _conditioned({"x": 1}), "the arc opened on a condition that said no"


# --------------------------------------------------------------------------- #
# A consumer's own implementation --- the population no in-tree edit can reach
# --------------------------------------------------------------------------- #


class _OneArgRecordTransform(ITransformFunction):
    """What the interface says ``data`` is: a mapping, tested and read as one."""

    received: Any = None

    def transform(self, data: Any) -> Dict[str, Any]:  # type: ignore[override]
        type(self).received = data
        return {**data, "kind": "one-arg"} if "a" in data else {"kind": "missing"}

    def get_transform_description(self) -> str:
        return "one-arg record transform"


class _TwoArgRecordTransform(ITransformFunction):
    """The conformant shape, which must keep receiving the context."""

    received_context: Any = None

    def transform(self, data: Any, context: Any = None) -> Dict[str, Any]:
        type(self).received_context = context
        return {**data, "kind": "two-arg"}

    def get_transform_description(self) -> str:
        return "two-arg record transform"


async def test_a_one_arg_interface_transform_receives_the_plain_record() -> None:
    """An interface implementation is never an inline lambda, whatever its arity.

    ``ITransformFunction.transform`` declares ``data``, not ``state`` --- so the
    arity says only whether to forward the context, never what to pass as the
    record. That is the judgement the config door has always made.
    """
    _OneArgRecordTransform.received = None

    result = await _run_async(
        _config(transform=REGISTERED), {"fn": _OneArgRecordTransform()}, {"a": 1}
    )

    assert result["success"], result
    assert result["data"]["kind"] == "one-arg", result["data"]
    assert not isinstance(_OneArgRecordTransform.received, StateDataWrapper), (
        "the implementation was handed the inline-lambda wrapper, not the record"
    )


async def test_a_two_arg_interface_transform_still_receives_the_context() -> None:
    """The context must keep arriving, or the fix has traded one gap for another."""
    _TwoArgRecordTransform.received_context = None

    result = await _run_async(
        _config(transform=REGISTERED), {"fn": _TwoArgRecordTransform()}, {"a": 1}
    )

    assert result["success"], result
    assert result["data"]["kind"] == "two-arg", result["data"]
    assert _TwoArgRecordTransform.received_context is not None, (
        "a two-argument implementation was called with the record alone"
    )


class _OneArgGate(IValidationFunction):
    """A consumer's gate, written to the one-argument convention."""

    def validate(self, data: Any) -> bool:  # type: ignore[override]
        return "a" in data

    def get_validation_rules(self) -> Dict[str, Any]:
        return {"requires": "a"}


async def _consumer_gated(data: Dict[str, Any]) -> bool:
    result = await _run_async(
        _config(transform=REGISTERED, gate=True),
        {"fn": _Marker(), "gate": _OneArgGate()},
        data,
    )
    return bool(result["data"].get("seen"))


async def test_a_one_arg_interface_gate_admits_the_record_it_should() -> None:
    """The engines reach a resolved validator two ways, and both must shape the call.

    A state's transforms are invoked through ``.transform``; its pre-validators
    are invoked by calling the resolved object itself. ``InterfaceWrapper``
    shaped only the first, and handed the second straight to the inner
    wrapper --- so the same object behaved differently depending on which of
    the two routes the engine happened to take to it.
    """
    assert await _consumer_gated({"a": 1}), "the gate refused a record that satisfies it"


async def test_a_one_arg_interface_gate_rejects_the_record_it_should() -> None:
    """The other half, so the test above cannot pass by the gate being absent."""
    assert not await _consumer_gated({"z": 1}), "the gate admitted a record that violates it"


# --------------------------------------------------------------------------- #
# The config door's own arity reading
# --------------------------------------------------------------------------- #


class _KwargsValidator(IValidationFunction):
    """``**kwargs`` absorbs keywords, and the context is passed positionally."""

    def validate(self, data: Any, **kwargs: Any) -> bool:  # type: ignore[override]
        return "a" in data

    def get_validation_rules(self) -> Dict[str, Any]:
        return {"requires": "a"}


async def test_a_kwargs_implementation_runs_through_the_config_door() -> None:
    """``**kwargs`` is not a second positional parameter.

    The config door's arity reading counted ``VAR_KEYWORD`` as "accepts the
    context" and then passed the context *positionally*, which no ``(data,
    **kwargs)`` signature can receive. The shared reading counts positional
    parameters only.
    """
    config = _config(transform=REGISTERED, gate=True)
    config["states"][1]["pre_validators"] = [
        {"type": "custom", "module": __name__, "name": "_KwargsValidator"}
    ]

    result = await _run_async(config, {"fn": _Marker()}, {"a": 1})

    assert result["data"].get("seen"), f"the gate refused a valid record: {result}"


# --------------------------------------------------------------------------- #
# The convention the heuristic was written for
# --------------------------------------------------------------------------- #


async def test_an_inline_lambda_still_receives_the_state_wrapper() -> None:
    """``lambda state: state.data[...]`` is the documented inline form.

    ``wrap_for_lambda`` exists for it, and the config guide teaches it, so the
    one-argument heuristic keeps serving the population it was introduced for
    --- this test is what stops the fix above from being bought at its expense.
    """
    config = _config(
        transform={"type": "inline", "code": "lambda state: {**state.data, 'kind': 'lambda'}"}
    )

    result = await _run_async(config, None, {"a": 1})

    assert result["success"], result
    assert result["data"] == {"a": 1, "kind": "lambda"}, result["data"]
