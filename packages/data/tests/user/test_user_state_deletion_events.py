"""Deletion / erasure delta-event tests for the per-user state coordinator.

Deletions and erasure fire a metadata-only ``user_state:section_deleted``
event — one op-discriminated event per delete-method call, only when data was
actually removed. Real constructs only (``AsyncMemoryDatabase`` /
``SyncMemoryDatabase`` for storage, ``InMemoryEventBus`` for fan-out); events
are captured by registering a sync callback on ``store._callbacks`` — the same
no-mocks pattern the write-event tests use. Every behavioral case is written
for both the async and sync variants. Time is driven by an injected clock so
prune tests are deterministic with no ``sleep``.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from dataknobs_common.events import InMemoryEventBus
from dataknobs_common.tenancy import BoundTenantContext
from dataknobs_data.backends.memory import AsyncMemoryDatabase
from dataknobs_data.user import (
    SECTION_DELETED_TOPIC,
    AsyncUserStateStore,
    UserStateStore,
)

# ``alerts`` never expires; ``activity`` prunes at 30 days; ``prefs`` is a
# document (documents are not collection-deletable / prunable).
_SECTIONS = [
    {"name": "prefs", "kind": "document"},
    {"name": "alerts", "kind": "collection"},
    {"name": "activity", "kind": "collection", "retention_days": 30},
]

_START = datetime(2026, 1, 1, tzinfo=timezone.utc)


class _Clock:
    """A deterministic, advanceable UTC clock injected as the ``now`` component."""

    def __init__(self, start: datetime) -> None:
        self.value = start

    def __call__(self) -> datetime:
        return self.value

    def advance(self, **kwargs: Any) -> None:
        self.value = self.value + timedelta(**kwargs)


def _config(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "backend": "memory",
        "namespace": "acme",
        "sections": list(_SECTIONS),
    }
    base.update(overrides)
    return base


def _capture(store: Any) -> list[dict[str, Any]]:
    """Register a sync capture callback for the delete topic (no mocks)."""
    captured: list[dict[str, Any]] = []
    store._callbacks.register(SECTION_DELETED_TOPIC, captured.append)
    return captured


# --------------------------------------------------------------------- #
# 1. delete_record fires one op="delete_record" event (async + sync).
# --------------------------------------------------------------------- #


async def test_delete_record_fires_event_async() -> None:
    store = await AsyncUserStateStore.from_config(_config())
    try:
        captured = _capture(store)
        await store.add_record("u1", "alerts", {"text": "a"})
        rows = await store.query("u1", "alerts")
        record_id = rows[0].storage_id

        assert await store.delete_record("u1", "alerts", record_id) is True

        assert len(captured) == 1
        event = captured[0]
        assert event["op"] == "delete_record"
        assert event["section"] == "alerts"
        assert event["record_id"] == record_id
        assert event["count"] == 1
    finally:
        await store.close()


def test_delete_record_fires_event_sync() -> None:
    store = UserStateStore.from_config(_config())
    try:
        captured = _capture(store)
        store.add_record("u1", "alerts", {"text": "a"})
        record_id = store.query("u1", "alerts")[0].storage_id

        assert store.delete_record("u1", "alerts", record_id) is True

        assert len(captured) == 1
        event = captured[0]
        assert event["op"] == "delete_record"
        assert event["section"] == "alerts"
        assert event["record_id"] == record_id
        assert event["count"] == 1
    finally:
        store.close()


# --------------------------------------------------------------------- #
# 2. A no-op delete (missing / out-of-scope id) fires nothing.
# --------------------------------------------------------------------- #


async def test_no_op_delete_fires_nothing_async() -> None:
    store = await AsyncUserStateStore.from_config(_config())
    try:
        captured = _capture(store)
        # A missing id returns False and fires nothing.
        assert await store.delete_record("u1", "alerts", "nope") is False
        # An out-of-scope id is rejected (u2 cannot delete u1's record) — the
        # scope guard raises before any delete, so still no event.
        await store.add_record("u1", "alerts", {"text": "a"})
        record_id = (await store.query("u1", "alerts"))[0].storage_id
        with pytest.raises(ValueError):
            await store.delete_record("u2", "alerts", record_id)
        assert captured == []
    finally:
        await store.close()


def test_no_op_delete_fires_nothing_sync() -> None:
    store = UserStateStore.from_config(_config())
    try:
        captured = _capture(store)
        assert store.delete_record("u1", "alerts", "nope") is False
        store.add_record("u1", "alerts", {"text": "a"})
        record_id = store.query("u1", "alerts")[0].storage_id
        with pytest.raises(ValueError):
            store.delete_record("u2", "alerts", record_id)
        assert captured == []
    finally:
        store.close()


# --------------------------------------------------------------------- #
# 3. prune fires one op="prune" event with the real deleted count;
#    a prune that expires nothing fires nothing.
# --------------------------------------------------------------------- #


async def test_prune_fires_event_async() -> None:
    clock = _Clock(_START)
    store = await AsyncUserStateStore.from_config(_config(), now=clock)
    try:
        captured = _capture(store)
        await store.add_record("u1", "activity", {"event": "old"})
        clock.advance(days=40)
        await store.add_record("u1", "activity", {"event": "new"})

        assert await store.prune("u1", "activity") == 1
        assert len(captured) == 1
        assert captured[0] == {
            "namespace": "acme",
            "tenant_id": None,
            "user_id": "u1",
            "section": "activity",
            "op": "prune",
            "count": 1,
        }

        # A second prune expires nothing → no event.
        assert await store.prune("u1", "activity") == 0
        assert len(captured) == 1
    finally:
        await store.close()


def test_prune_fires_event_sync() -> None:
    clock = _Clock(_START)
    store = UserStateStore.from_config(_config(), now=clock)
    try:
        captured = _capture(store)
        store.add_record("u1", "activity", {"event": "old"})
        clock.advance(days=40)
        store.add_record("u1", "activity", {"event": "new"})

        assert store.prune("u1", "activity") == 1
        assert len(captured) == 1
        assert captured[0]["op"] == "prune"
        assert captured[0]["section"] == "activity"
        assert captured[0]["count"] == 1

        assert store.prune("u1", "activity") == 0
        assert len(captured) == 1
    finally:
        store.close()


# --------------------------------------------------------------------- #
# 4. A lazy prune_on_query deletion is observable on the read path.
# --------------------------------------------------------------------- #


async def test_prune_on_query_delete_is_observable_async() -> None:
    clock = _Clock(_START)
    store = await AsyncUserStateStore.from_config(
        _config(prune_on_query=True), now=clock
    )
    try:
        captured = _capture(store)
        await store.add_record("u1", "activity", {"event": "old"})
        clock.advance(days=40)
        await store.add_record("u1", "activity", {"event": "new"})

        # The read prunes first; the deletion must surface as an event.
        rows = await store.query("u1", "activity")
        assert {r.get_value("event") for r in rows} == {"new"}
        assert len(captured) == 1
        assert captured[0]["op"] == "prune"
        assert captured[0]["count"] == 1
    finally:
        await store.close()


def test_prune_on_query_delete_is_observable_sync() -> None:
    clock = _Clock(_START)
    store = UserStateStore.from_config(_config(prune_on_query=True), now=clock)
    try:
        captured = _capture(store)
        store.add_record("u1", "activity", {"event": "old"})
        clock.advance(days=40)
        store.add_record("u1", "activity", {"event": "new"})

        rows = store.query("u1", "activity")
        assert {r.get_value("event") for r in rows} == {"new"}
        assert len(captured) == 1
        assert captured[0]["op"] == "prune"
    finally:
        store.close()


# --------------------------------------------------------------------- #
# 5. clear fires one op="clear" event, section=None, count=total erased;
#    a clear of an empty user fires nothing.
# --------------------------------------------------------------------- #


async def test_clear_fires_event_async() -> None:
    store = await AsyncUserStateStore.from_config(_config())
    try:
        captured = _capture(store)
        # Empty user → nothing to erase → no event.
        assert await store.clear("ghost") == 0
        assert captured == []

        await store.put_document("u1", "prefs", {"theme": "dark"})
        await store.add_record("u1", "alerts", {"text": "a"})
        await store.add_record("u1", "alerts", {"text": "b"})

        total = await store.clear("u1")
        assert total == 3
        assert len(captured) == 1
        assert captured[0]["op"] == "clear"
        assert captured[0]["section"] is None
        assert captured[0]["count"] == 3
        assert "record_id" not in captured[0]
    finally:
        await store.close()


def test_clear_fires_event_sync() -> None:
    store = UserStateStore.from_config(_config())
    try:
        captured = _capture(store)
        assert store.clear("ghost") == 0
        assert captured == []

        store.put_document("u1", "prefs", {"theme": "dark"})
        store.add_record("u1", "alerts", {"text": "a"})
        store.add_record("u1", "alerts", {"text": "b"})

        assert store.clear("u1") == 3
        assert len(captured) == 1
        assert captured[0]["op"] == "clear"
        assert captured[0]["section"] is None
        assert captured[0]["count"] == 3
        assert "record_id" not in captured[0]
    finally:
        store.close()


# --------------------------------------------------------------------- #
# 6. Metadata-only across all three paths + tenancy stamping.
# --------------------------------------------------------------------- #


async def test_delete_events_are_metadata_only_with_tenant() -> None:
    from dataknobs_data.user import UserStateStoreConfig

    db = AsyncMemoryDatabase()
    cfg = UserStateStoreConfig.from_dict(
        _config(
            sections=[
                {
                    "name": "profile",
                    "kind": "collection",
                    "sensitivity": "sensitive",
                }
            ]
        )
    )
    store = AsyncUserStateStore.from_components(
        cfg, db=db, tenant=BoundTenantContext("t1", "acme")
    )
    try:
        captured = _capture(store)
        await store.add_record("u1", "profile", {"ssn": "secret"})
        await store.add_record("u1", "profile", {"ssn": "other"})
        record_id = (await store.query("u1", "profile"))[0].storage_id
        # One single delete, then erase the rest — two delete events.
        await store.delete_record("u1", "profile", record_id)
        await store.clear("u1")

        assert len(captured) == 2
        for event in captured:
            # The SENSITIVE value never appears; only metadata keys are present.
            assert "ssn" not in event
            assert event["namespace"] == "acme"
            assert event["user_id"] == "u1"
            assert event["tenant_id"] == "t1"
    finally:
        await store.close()


# --------------------------------------------------------------------- #
# 7. Fan-out (async only): an injected EventBus receives delete events, and a
#    failing subscriber is isolated (the delete still succeeds).
# --------------------------------------------------------------------- #


async def test_delete_events_fan_out_and_isolate_failure() -> None:
    bus = InMemoryEventBus()
    await bus.connect()
    received: list[dict[str, Any]] = []

    async def handler(event: Any) -> None:
        received.append(event.payload)

    await bus.subscribe(SECTION_DELETED_TOPIC, handler)

    store = await AsyncUserStateStore.from_config(_config(), event_bus=bus)
    local = _capture(store)

    def failing(_: dict[str, Any]) -> None:
        raise RuntimeError("subscriber down")

    store._callbacks.register(SECTION_DELETED_TOPIC, failing)

    await store.add_record("u1", "alerts", {"text": "a"})
    record_id = (await store.query("u1", "alerts"))[0].storage_id

    # The delete must succeed despite the failing subscriber.
    assert await store.delete_record("u1", "alerts", record_id) is True

    assert local and local[0]["op"] == "delete_record"
    assert received and received[0]["op"] == "delete_record"
    assert received[0]["record_id"] == record_id
    await store.close()


# --------------------------------------------------------------------- #
# 8. Export symmetry + both variants carry _fire_deleted.
# --------------------------------------------------------------------- #


def test_topic_exported_symmetrically() -> None:
    import dataknobs_data as top
    import dataknobs_data.user as user_pkg

    assert top.SECTION_DELETED_TOPIC == "user_state:section_deleted"
    assert user_pkg.SECTION_DELETED_TOPIC == "user_state:section_deleted"
    assert "SECTION_DELETED_TOPIC" in top.__all__
    assert "SECTION_DELETED_TOPIC" in user_pkg.__all__


def test_both_variants_have_fire_deleted() -> None:
    assert hasattr(AsyncUserStateStore, "_fire_deleted")
    assert hasattr(UserStateStore, "_fire_deleted")
