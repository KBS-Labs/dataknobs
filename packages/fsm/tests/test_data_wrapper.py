"""Tests for the data-wrapper conversion helpers."""

from __future__ import annotations

import pytest

from dataknobs_fsm.core.data_wrapper import (
    FSMData,
    StateDataWrapper,
    ensure_dict,
)


def test_ensure_dict_passes_through_plain_dict():
    payload = {"a": 1, "b": 2}
    assert ensure_dict(payload) == {"a": 1, "b": 2}


def test_ensure_dict_unwraps_fsm_data():
    assert ensure_dict(FSMData({"a": 1})) == {"a": 1}


def test_ensure_dict_unwraps_state_data_wrapper():
    """``StateDataWrapper`` must convert to its underlying raw dict.

    ``StateDataWrapper.data`` always stores the raw dict (class invariant), so
    ``ensure_dict`` must return that dict directly. The previous implementation
    called ``.to_dict()`` on it, which raised ``AttributeError`` because a plain
    ``dict`` has no ``to_dict`` — a latent crash for any engine call site that
    passed a wrapper (e.g. ``context.data``) through ``ensure_dict``.
    """
    wrapper = StateDataWrapper({"a": 1, "b": 2})
    assert ensure_dict(wrapper) == {"a": 1, "b": 2}


def test_ensure_dict_unwraps_state_data_wrapper_built_from_fsm_data():
    wrapper = StateDataWrapper(FSMData({"x": 9}))
    assert ensure_dict(wrapper) == {"x": 9}


def test_ensure_dict_does_not_return_a_non_dict_from_a_wrapper():
    """The contract is in the name, and a duck-typed branch used to break it.

    Two branches returned ``_data`` on sight of the attribute, whatever it
    held. A wrapper whose ``_data`` is a list made ``ensure_dict`` hand a list
    back to callers that store it straight into ``context.data`` --- a record
    that is not a record, from a function whose name is the promise that it
    is one. Checking the attribute rather than only its presence keeps the
    promise: such an object now falls through to the conversion at the
    bottom, which either converts it or fails where it happened.
    """

    class ListBacked:
        def __init__(self) -> None:
            self._data = [("a", 1)]

    with pytest.raises(TypeError):
        ensure_dict(ListBacked())

    class NestedListBacked:
        class Inner:
            def __init__(self) -> None:
                self._data = [("a", 1)]

        def __init__(self) -> None:
            self.data = self.Inner()

    with pytest.raises(TypeError):
        ensure_dict(NestedListBacked())


def test_ensure_dict_still_unwraps_a_dict_backed_wrapper():
    """The duck-typed branch the check is guarding must keep working."""

    class DictBacked:
        def __init__(self) -> None:
            self._data = {"a": 1}

    assert ensure_dict(DictBacked()) == {"a": 1}

    class NestedDictBacked:
        class Inner:
            def __init__(self) -> None:
                self._data = {"b": 2}

        def __init__(self) -> None:
            self.data = self.Inner()

    assert ensure_dict(NestedDictBacked()) == {"b": 2}


def test_fsm_data_update_accepts_the_shapes_a_mapping_accepts():
    """``update`` is ``MutableMapping``'s now, not a hand-written copy.

    The copy took ``other`` by keyword and treated ``None`` as "nothing to
    merge"; the inherited one is positional-only and merges through
    ``__setitem__``, which is the same write path the copy used.
    """
    data = FSMData({"a": 1})

    data.update({"b": 2})
    data.update(FSMData({"c": 3}))
    data.update([("d", 4)])
    data.update(e=5)
    data.update()

    assert data.to_dict() == {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}
