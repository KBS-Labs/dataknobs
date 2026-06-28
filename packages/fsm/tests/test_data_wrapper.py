"""Tests for the data-wrapper conversion helpers."""

from __future__ import annotations

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
