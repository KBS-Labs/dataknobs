"""Regression tests pinning that ``make_metadata_inbox_hook`` preserves
byte-identical inbox semantics after the ``InboxOnlyBridge`` substrate
switch.

The read step now delegates through ``InboxOnlyBridge.read_inbox``; the
hook's observable behavior contract is unchanged:

- Multi-key support: drain N keys per turn.
- Consume-on-read: pop, not get.
- Default merge is plain ``dict.update`` — a ``None``-valued payload
  entry is WRITTEN THROUGH into ``state.data`` (NOT evicted). Eviction
  on the next-turn read is a downstream property of the consumer seeing
  the written-through ``None``, not a merge-time pop.
- Empty-dict no-op: zero-key payload is a no-op (key still popped).
- Non-mapping payload: WARNING log + skip.
- Custom ``merge_fn``: replaces the default entirely.

The factory is keyword-only (``inbox_keys=``) and returns an ``async``
hook — these tests ``await`` the hook directly with a minimal event
payload.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from dataknobs_bots.reasoning.wizard_inbox import (
    make_metadata_inbox_hook,
    write_to_inbox,
)


def _make_event(metadata: dict, state_data: dict) -> dict:
    manager = SimpleNamespace(metadata=metadata)
    state = SimpleNamespace(data=state_data)
    return {"manager": manager, "state": state, "stage": "test"}


@pytest.mark.asyncio
async def test_drains_single_key() -> None:
    hook = make_metadata_inbox_hook(inbox_keys=["sub_strategy_output"])
    event = _make_event(
        metadata={"sub_strategy_output": {"x": 1}},
        state_data={},
    )
    await hook(event)
    assert event["state"].data == {"x": 1}
    assert "sub_strategy_output" not in event["manager"].metadata


@pytest.mark.asyncio
async def test_drains_multiple_keys_in_order() -> None:
    hook = make_metadata_inbox_hook(inbox_keys=["a", "b"])
    event = _make_event(
        metadata={"a": {"x": 1}, "b": {"y": 2}},
        state_data={},
    )
    await hook(event)
    assert event["state"].data == {"x": 1, "y": 2}


@pytest.mark.asyncio
async def test_pop_semantic_removes_key() -> None:
    hook = make_metadata_inbox_hook(inbox_keys=["k"])
    event = _make_event(metadata={"k": {"x": 1}}, state_data={})
    await hook(event)
    # First turn drains; the key is consumed.
    assert "k" not in event["manager"].metadata
    await hook(event)  # Second call: no key, no-op merge.
    assert event["state"].data == {"x": 1}


@pytest.mark.asyncio
async def test_none_written_through_with_default_merge() -> None:
    """Shipped default merge is plain ``dict.update``, so a None-valued
    payload entry is WRITTEN THROUGH (not evicted at merge time).
    """
    hook = make_metadata_inbox_hook(inbox_keys=["k"])
    event = _make_event(
        metadata={"k": {"existing": None}},
        state_data={"existing": "value"},
    )
    await hook(event)
    assert event["state"].data["existing"] is None


@pytest.mark.asyncio
async def test_empty_dict_payload_is_noop() -> None:
    hook = make_metadata_inbox_hook(inbox_keys=["k"])
    event = _make_event(metadata={"k": {}}, state_data={"x": 1})
    await hook(event)
    assert event["state"].data == {"x": 1}
    # Empty payload still consumes the key (pop semantics).
    assert "k" not in event["manager"].metadata


@pytest.mark.asyncio
async def test_non_mapping_payload_warns_and_skips(caplog) -> None:
    hook = make_metadata_inbox_hook(inbox_keys=["k"])
    event = _make_event(metadata={"k": "not-a-dict"}, state_data={"x": 1})
    caplog.set_level("WARNING", logger="dataknobs_bots.reasoning.wizard_inbox")
    await hook(event)
    assert event["state"].data == {"x": 1}  # No merge.
    assert any("not a mapping" in r.message.lower() for r in caplog.records)


@pytest.mark.asyncio
async def test_custom_merge_fn_used_when_supplied() -> None:
    merge_calls: list[tuple[dict, dict]] = []

    def custom_merge(state_data: dict, payload: dict) -> None:
        merge_calls.append((state_data, payload))
        state_data.update(payload)

    hook = make_metadata_inbox_hook(inbox_keys=["k"], merge_fn=custom_merge)
    event = _make_event(metadata={"k": {"a": None}}, state_data={})
    await hook(event)
    assert event["state"].data == {"a": None}
    assert len(merge_calls) == 1


@pytest.mark.asyncio
async def test_hook_handles_missing_manager_or_state_gracefully() -> None:
    """Opaque-dict payload contract: a malformed event passes through
    without raising.
    """
    hook = make_metadata_inbox_hook(inbox_keys=["k"])
    await hook({"stage": "test"})  # No manager / state keys; no raise.


@pytest.mark.asyncio
async def test_hook_tolerates_host_without_metadata() -> None:
    """The hook's ``hasattr`` guard precedes the bridge read, so a
    manager lacking ``metadata`` is a silent no-op (not an
    ``AttributeError`` from the bridge).
    """
    hook = make_metadata_inbox_hook(inbox_keys=["k"])
    state = SimpleNamespace(data={"x": 1})
    await hook({"manager": SimpleNamespace(), "state": state, "stage": "t"})
    assert state.data == {"x": 1}


def test_write_to_inbox_overwrites_existing() -> None:
    manager = SimpleNamespace(metadata={"k": {"old": True}})
    write_to_inbox(manager, "k", {"new": True})
    assert manager.metadata["k"] == {"new": True}


def test_write_to_inbox_requires_metadata_attr() -> None:
    with pytest.raises(AttributeError, match="metadata"):
        write_to_inbox(SimpleNamespace(), "k", {})
