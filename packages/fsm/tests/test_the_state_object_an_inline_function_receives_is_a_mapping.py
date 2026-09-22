# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""The object a one-argument function receives answers as the record it holds.

``wrap_for_lambda`` hands a ``StateDataWrapper`` to every one-argument *plain*
callable --- a registered function, a registered gate, an inline ``lambda
state: ...`` --- and the documented way to read it is ``state.data[...]``. But
the wrapper also forwards ``state['field']``, which invites reading it as the
record itself, and there it stopped: ``__getitem__`` and ``__setitem__`` were
the whole type-level surface.

Python answers ``in`` and iteration from the *type*, so with no
``__contains__`` the ``in`` operator fell back to the old integer-indexing
protocol and asked for key ``0``. The record is keyed by strings, so the
answer was ``KeyError: 0`` --- an exception naming a key the caller never
wrote, from a line that only asked whether a field was present. Measured, the
same three shapes were missing everywhere it mattered:

===========================  ==============================================
``'a' in state``             ``KeyError: 0``
``len(state)``               ``TypeError: ... has no len()``
``for key in state``         ``KeyError: 0``
===========================  ==============================================

and three more followed from them: ``bool(state)`` was ``True`` for an empty
record, ``state == {...}`` was ``False`` for the record it wrapped, and
``repr(state)`` printed an address instead of the data.

None of it surfaces as itself. A transform reports ``State transform failed in:
<state>``; a gate is worse --- the crash inside it reads as a refusal, so a
valid record is simply turned away with the arc reported as failed.

The sibling in the same module, :class:`FSMData`, is a real
``MutableMapping`` and answers all of it. Real constructs only: real
``SimpleFSM`` / ``AsyncSimpleFSM`` over real configs.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, MutableMapping
from typing import Any, Dict

from dataknobs_fsm.api.async_simple import AsyncSimpleFSM
from dataknobs_fsm.api.simple import SimpleFSM
from dataknobs_fsm.core.data_modes import DataHandlingMode
from dataknobs_fsm.core.data_wrapper import FSMData, StateDataWrapper, wrap_for_lambda


def _config(*, transform: Dict[str, Any] | None = None, gate: bool = False) -> Dict[str, Any]:
    """One ``work`` state, optionally gated."""
    work: Dict[str, Any] = {"name": "work", "resources": []}
    if transform is not None:
        work["functions"] = {"transform": transform}
    if gate:
        work["pre_validators"] = [{"type": "registered", "name": "gate"}]

    return {
        "name": "mapping_surface_probe",
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


REGISTERED = {"type": "registered", "name": "fn"}


async def _run_async(config: Dict[str, Any], functions: Dict[str, Any] | None, data: Any) -> Any:
    fsm = AsyncSimpleFSM(config, data_mode=DataHandlingMode.COPY, custom_functions=functions)
    return await fsm.process(dict(data))


def _run_sync(config: Dict[str, Any], functions: Dict[str, Any] | None, data: Any) -> Any:
    fsm = SimpleFSM(config, data_mode=DataHandlingMode.COPY, custom_functions=functions)
    return fsm.process(dict(data))


# --------------------------------------------------------------------------- #
# Through the doors that hand a one-argument callable the wrapper
# --------------------------------------------------------------------------- #


def _asks_membership(state: Any) -> Dict[str, Any]:
    """The shape a consumer writes: a presence check on what it was handed."""
    return {"has_a": "a" in state, "has_z": "z" in state}


async def test_a_registered_transform_can_ask_whether_a_field_is_present() -> None:
    """``'a' in state`` reached the integer-index fallback and raised.

    The caller sees ``success: False`` with ``State transform failed in:
    work`` --- the state, but not the reason, and not a hint that the object
    it was handed is the thing that cannot answer.
    """
    result = await _run_async(_config(transform=REGISTERED), {"fn": _asks_membership}, {"a": 1})

    assert result["success"], result
    assert result["data"] == {"has_a": True, "has_z": False}, result["data"]


def test_a_registered_transform_can_ask_on_the_sync_facade() -> None:
    """The synchronous door prepares the same wrapper from the same base engine."""
    result = _run_sync(_config(transform=REGISTERED), {"fn": _asks_membership}, {"a": 1})

    assert result["success"], result
    assert result["data"] == {"has_a": True, "has_z": False}, result["data"]


async def test_an_inline_lambda_can_ask_whether_a_field_is_present() -> None:
    """The inline form is the population ``wrap_for_lambda`` exists to serve."""
    result = await _run_async(
        _config(transform={"type": "inline", "code": "lambda state: {'has_a': 'a' in state}"}),
        None,
        {"a": 1},
    )

    assert result["success"], result
    assert result["data"] == {"has_a": True}, result["data"]


async def test_a_registered_transform_can_count_the_record() -> None:
    """``len(state)`` raised ``TypeError`` rather than counting the fields."""
    result = await _run_async(
        _config(transform=REGISTERED),
        {"fn": lambda state: {"n": len(state)}},
        {"a": 1, "b": 2},
    )

    assert result["success"], result
    assert result["data"] == {"n": 2}, result["data"]


async def test_a_registered_transform_can_iterate_the_record() -> None:
    """Iteration took the same integer-index fallback that ``in`` took."""
    result = await _run_async(
        _config(transform=REGISTERED),
        {"fn": lambda state: {"fields": sorted(state)}},
        {"b": 2, "a": 1},
    )

    assert result["success"], result
    assert result["data"] == {"fields": ["a", "b"]}, result["data"]


async def _gated(data: Dict[str, Any]) -> bool:
    """Whether a one-argument gate that asks ``'a' in state`` admitted a record."""
    result = await _run_async(
        _config(transform=REGISTERED, gate=True),
        {"fn": lambda state: {**state.data, "seen": True}, "gate": lambda state: "a" in state},
        data,
    )
    return bool(result["data"].get("seen"))


async def test_a_one_argument_gate_admits_the_record_it_should() -> None:
    """The severe shape: the crash inside the gate reads as a refusal.

    Nothing reports an error the caller can act on --- the arc is simply
    reported as having failed, for every record, including the ones the gate
    would have admitted.
    """
    assert await _gated({"a": 1}), "the gate refused a record that satisfies it"


async def test_a_one_argument_gate_rejects_the_record_it_should() -> None:
    """The other half, so the test above cannot pass by the gate being absent."""
    assert not await _gated({"z": 1}), "the gate admitted a record that violates it"


# --------------------------------------------------------------------------- #
# The surface itself
# --------------------------------------------------------------------------- #


def test_the_state_object_is_a_mutable_mapping() -> None:
    """What the three missing methods amount to, asked once.

    ``FSMData`` in the same module already declares this; the two wrappers
    over the same record disagreed about what a record is.
    """
    wrapper = wrap_for_lambda({"a": 1})

    assert isinstance(wrapper, Mapping)
    assert isinstance(wrapper, MutableMapping)
    assert isinstance(FSMData({"a": 1}), MutableMapping), "the sibling, for comparison"


def test_an_empty_state_object_is_falsy() -> None:
    """``bool(state)`` answered ``True`` for an empty record.

    Nothing defined ``__bool__`` or ``__len__``, so every wrapper was truthy
    by default --- which makes ``if not state:`` a branch that never runs.
    """
    assert not StateDataWrapper({}), "an empty record read as truthy"
    assert StateDataWrapper({"a": 1}), "a populated record must stay truthy"


def test_a_state_object_equals_the_record_it_wraps() -> None:
    """Identity comparison answered ``False`` for the record it holds."""
    assert StateDataWrapper({"a": 1, "b": 2}) == {"a": 1, "b": 2}
    assert StateDataWrapper({"a": 1}) != {"a": 2}
    assert StateDataWrapper({"a": 1}) == FSMData({"a": 1})


def test_a_state_object_shows_its_record_when_printed() -> None:
    """A test failure printed ``<... object at 0x...>`` and no data."""
    assert repr(StateDataWrapper({"a": 1})) == "StateDataWrapper({'a': 1})"


def test_a_state_object_can_be_copied() -> None:
    """``copy`` and ``deepcopy`` both died of ``RecursionError``.

    ``__getattr__`` forwarded every miss to ``self.data``, so the lookup of a
    copy protocol method on a half-built instance --- one whose ``data`` is
    not set yet --- asked ``__getattr__`` for ``data``, which asked for
    ``data``, without end.
    """
    original = StateDataWrapper({"a": 1, "nested": {"x": 1}})

    shallow = copy.copy(original)
    assert shallow == {"a": 1, "nested": {"x": 1}}

    deep = copy.deepcopy(original)
    assert deep == {"a": 1, "nested": {"x": 1}}
    deep["nested"]["x"] = 2
    assert original["nested"]["x"] == 1, "the deep copy still shared the record"


def test_a_state_object_answers_to_dict_like_its_sibling() -> None:
    """The two wrappers over one record answer the same question the same way.

    They did not: ``ensure_dict`` had to carry a special case, added after
    calling ``to_dict()`` on this wrapper raised ``AttributeError`` --- the
    method existed on ``FSMData`` and on the raw dict that ``.data`` had
    become, but not on the thing in between.
    """
    record = {"a": 1}

    assert wrap_for_lambda(record).to_dict() is record
    assert FSMData(record).to_dict() is record


def test_the_mapping_methods_read_through_to_the_record() -> None:
    """The mixin must read the live record, not a snapshot taken at wrap time."""
    record: Dict[str, Any] = {"a": 1}
    wrapper = wrap_for_lambda(record)

    record["b"] = 2

    assert dict(wrapper) == {"a": 1, "b": 2}
    assert sorted(wrapper.keys()) == ["a", "b"]
    assert wrapper.get("b") == 2
    assert wrapper.get("zz", "dflt") == "dflt"
    assert len(wrapper) == 2


# --------------------------------------------------------------------------- #
# Regression guards: what the wrapper already did, it must keep doing
# --------------------------------------------------------------------------- #


def test_the_documented_state_data_access_still_works() -> None:
    """``state.data[...]`` is the form every example and guide teaches."""
    record = {"a": 1}
    wrapper = wrap_for_lambda(record)

    assert wrapper.data is record, "``.data`` must stay the raw record, not a copy"
    assert wrapper.data["a"] == 1


def test_writing_through_the_state_object_reaches_the_record() -> None:
    """A transform that mutates in place must still be mutating the record."""
    record: Dict[str, Any] = {"a": 1}
    wrapper = wrap_for_lambda(record)

    wrapper["b"] = 2
    del wrapper["a"]

    assert record == {"b": 2}


def test_a_state_object_built_from_fsm_data_exposes_the_raw_record() -> None:
    """The ``FSMData`` arm keeps the invariant the other two arms hold."""
    fsm_data = FSMData({"x": 9})
    wrapper = wrap_for_lambda(fsm_data)

    assert wrapper.data is fsm_data.to_dict()
    assert wrapper["x"] == 9
    assert "x" in wrapper
