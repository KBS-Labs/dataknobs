"""Persisted append-only event-log tests for the per-user state coordinator.

Reproduce-first for the audit-log layer: with ``enable_event_log`` set, every
data write (``put_document`` / ``add_record`` / ``update_record``) and every
scoped deletion (``delete_record`` / ``prune``) appends one metadata-only
record to a reserved ``events`` collection section. Whole-user erasure
(``clear``) deliberately appends **nothing** — re-materializing a record in the
just-erased user's own log would defeat the erasure. The log is read through
the dedicated ``query_events`` accessor (the reserved section is walled off from
the generic content API) and honours its own ``event_log_retention_days``
window through the ordinary section-less ``prune`` sweep.

Real constructs only (``AsyncMemoryDatabase`` / ``SyncMemoryDatabase`` for
storage, ``InMemoryEventBus`` where fan-out is exercised). Time is driven by an
injected clock so retention cases are deterministic with no ``sleep``. Every
behavioral case is written for both the async and sync variants.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from dataknobs_common.exceptions import ConfigurationError, ConsentRequiredError
from dataknobs_common.tenancy import BoundTenantContext
from dataknobs_data.backends.memory import (
    AsyncMemoryDatabase,
    SyncMemoryDatabase,
)
from dataknobs_data.records import Record
from dataknobs_data.user import (
    SECTION_DELETED_TOPIC,
    AsyncUserStateStore,
    UserStateStore,
    UserStateStoreConfig,
)
from dataknobs_data.user.store import RESERVED_EVENTS_SECTION

# ``prefs`` is a document; ``alerts`` an unwindowed collection; ``activity`` a
# 30-day windowed collection.
_SECTIONS = [
    {"name": "prefs", "kind": "document"},
    {"name": "alerts", "kind": "collection"},
    {"name": "activity", "kind": "collection", "retention_days": 30},
]

_START = datetime(2026, 1, 1, tzinfo=UTC)


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
        "enable_event_log": True,
    }
    base.update(overrides)
    return base


# --------------------------------------------------------------------- #
# 1. Data writes append exactly one metadata-only events record.
# --------------------------------------------------------------------- #


async def test_put_document_appends_metadata_only_event_async() -> None:
    store = await AsyncUserStateStore.from_config(_config())
    try:
        await store.put_document("u1", "prefs", {"theme": "dark"})

        events = await store.query_events("u1")
        assert len(events) == 1
        data = events[0].data
        assert data["op"] == "put_document"
        assert data["op_section"] == "prefs"
        assert data["op_record_id"]  # the document id
        # Metadata only — the written value never lands in the log.
        assert "theme" not in data
    finally:
        await store.close()


def test_put_document_appends_metadata_only_event_sync() -> None:
    store = UserStateStore.from_config(_config())
    try:
        store.put_document("u1", "prefs", {"theme": "dark"})

        events = store.query_events("u1")
        assert len(events) == 1
        data = events[0].data
        assert data["op"] == "put_document"
        assert data["op_section"] == "prefs"
        assert data["op_record_id"]
        assert "theme" not in data
    finally:
        store.close()


async def test_add_and_update_record_each_append_one_event_async() -> None:
    store = await AsyncUserStateStore.from_config(_config())
    try:
        rid = await store.add_record("u1", "alerts", {"text": "a"})
        await store.update_record("u1", "alerts", rid, {"text": "b"})

        events = await store.query_events("u1")
        ops = sorted(e.data["op"] for e in events)
        assert ops == ["add_record", "update_record"]
        for e in events:
            assert e.data["op_section"] == "alerts"
            assert e.data["op_record_id"] == rid
            assert "text" not in e.data
    finally:
        await store.close()


def test_add_and_update_record_each_append_one_event_sync() -> None:
    store = UserStateStore.from_config(_config())
    try:
        rid = store.add_record("u1", "alerts", {"text": "a"})
        store.update_record("u1", "alerts", rid, {"text": "b"})

        events = store.query_events("u1")
        ops = sorted(e.data["op"] for e in events)
        assert ops == ["add_record", "update_record"]
        for e in events:
            assert e.data["op_section"] == "alerts"
            assert e.data["op_record_id"] == rid
            assert "text" not in e.data
    finally:
        store.close()


async def test_failed_update_appends_no_event_async() -> None:
    store = await AsyncUserStateStore.from_config(_config())
    try:
        # No such record → update returns False → nothing logged.
        assert await store.update_record("u1", "alerts", "missing", {"text": "x"}) is False
        assert await store.query_events("u1") == []
    finally:
        await store.close()


def test_failed_update_appends_no_event_sync() -> None:
    store = UserStateStore.from_config(_config())
    try:
        assert store.update_record("u1", "alerts", "missing", {"text": "x"}) is False
        assert store.query_events("u1") == []
    finally:
        store.close()


# --------------------------------------------------------------------- #
# 2. Scoped deletions append; no-op deletions append nothing.
# --------------------------------------------------------------------- #


async def test_delete_record_appends_event_async() -> None:
    store = await AsyncUserStateStore.from_config(_config())
    try:
        rid = await store.add_record("u1", "alerts", {"text": "a"})
        assert await store.delete_record("u1", "alerts", rid) is True

        events = await store.query_events("u1")
        delete_events = [e for e in events if e.data["op"] == "delete_record"]
        assert len(delete_events) == 1
        data = delete_events[0].data
        assert data["op_section"] == "alerts"
        assert data["op_record_id"] == rid
        assert data["op_count"] == 1
    finally:
        await store.close()


def test_delete_record_appends_event_sync() -> None:
    store = UserStateStore.from_config(_config())
    try:
        rid = store.add_record("u1", "alerts", {"text": "a"})
        assert store.delete_record("u1", "alerts", rid) is True

        delete_events = [e for e in store.query_events("u1") if e.data["op"] == "delete_record"]
        assert len(delete_events) == 1
        data = delete_events[0].data
        assert data["op_section"] == "alerts"
        assert data["op_record_id"] == rid
        assert data["op_count"] == 1
    finally:
        store.close()


async def test_no_op_delete_appends_no_event_async() -> None:
    store = await AsyncUserStateStore.from_config(_config())
    try:
        # Missing id → delete returns False → nothing logged.
        assert await store.delete_record("u1", "alerts", "missing") is False
        assert await store.query_events("u1") == []
    finally:
        await store.close()


def test_no_op_delete_appends_no_event_sync() -> None:
    store = UserStateStore.from_config(_config())
    try:
        assert store.delete_record("u1", "alerts", "missing") is False
        assert store.query_events("u1") == []
    finally:
        store.close()


# --------------------------------------------------------------------- #
# 3. prune appends; section-less prune carries the per-section split.
# --------------------------------------------------------------------- #


async def test_single_section_prune_appends_event_async() -> None:
    clock = _Clock(_START)
    store = await AsyncUserStateStore.from_config(_config(), now=clock)
    try:
        await store.add_record("u1", "activity", {"event": "old"})
        clock.advance(days=40)

        assert await store.prune("u1", "activity") == 1

        prune_events = [e for e in await store.query_events("u1") if e.data["op"] == "prune"]
        assert len(prune_events) == 1
        data = prune_events[0].data
        assert data["op_section"] == "activity"
        assert data["op_count"] == 1
        assert "op_sections" not in data
    finally:
        await store.close()


def test_single_section_prune_appends_event_sync() -> None:
    clock = _Clock(_START)
    store = UserStateStore.from_config(_config(), now=clock)
    try:
        store.add_record("u1", "activity", {"event": "old"})
        clock.advance(days=40)

        assert store.prune("u1", "activity") == 1
        prune_events = [e for e in store.query_events("u1") if e.data["op"] == "prune"]
        assert len(prune_events) == 1
        data = prune_events[0].data
        assert data["op_section"] == "activity"
        assert data["op_count"] == 1
        assert "op_sections" not in data
    finally:
        store.close()


async def test_section_less_prune_appends_split_event_async() -> None:
    clock = _Clock(_START)
    store = await AsyncUserStateStore.from_config(_config(), now=clock)
    try:
        await store.add_record("u1", "activity", {"event": "old"})
        clock.advance(days=40)

        assert await store.prune("u1") == 1

        prune_events = [e for e in await store.query_events("u1") if e.data["op"] == "prune"]
        assert len(prune_events) == 1
        data = prune_events[0].data
        assert data["op_section"] is None
        assert data["op_count"] == 1
        assert data["op_sections"] == {"activity": 1}
    finally:
        await store.close()


def test_section_less_prune_appends_split_event_sync() -> None:
    clock = _Clock(_START)
    store = UserStateStore.from_config(_config(), now=clock)
    try:
        store.add_record("u1", "activity", {"event": "old"})
        clock.advance(days=40)

        assert store.prune("u1") == 1
        prune_events = [e for e in store.query_events("u1") if e.data["op"] == "prune"]
        assert len(prune_events) == 1
        data = prune_events[0].data
        assert data["op_section"] is None
        assert data["op_count"] == 1
        assert data["op_sections"] == {"activity": 1}
    finally:
        store.close()


async def test_empty_prune_appends_no_event_async() -> None:
    clock = _Clock(_START)
    store = await AsyncUserStateStore.from_config(_config(), now=clock)
    try:
        await store.add_record("u1", "activity", {"event": "fresh"})
        # Nothing has expired yet.
        assert await store.prune("u1", "activity") == 0
        prune_events = [e for e in await store.query_events("u1") if e.data["op"] == "prune"]
        assert prune_events == []
    finally:
        await store.close()


# --------------------------------------------------------------------- #
# 4. clear (erasure) appends nothing — the ephemeral event still fires.
# --------------------------------------------------------------------- #


async def test_clear_appends_no_persisted_event_but_fires_ephemeral_async() -> None:
    store = await AsyncUserStateStore.from_config(_config())
    try:
        deleted_ops: list[str] = []
        store._callbacks.register(SECTION_DELETED_TOPIC, lambda p: deleted_ops.append(p["op"]))
        await store.add_record("u1", "alerts", {"text": "a"})
        await store.put_document("u1", "prefs", {"theme": "dark"})

        removed = await store.clear("u1")
        assert removed > 0

        # Erasure leaves no per-user trace — the log itself is gone, and no
        # ``clear`` record is re-materialized into the just-erased section.
        assert await store.query_events("u1") == []
        # The ephemeral stream still records the erasure for real-time audit.
        assert "clear" in deleted_ops
    finally:
        await store.close()


def test_clear_appends_no_persisted_event_but_fires_ephemeral_sync() -> None:
    store = UserStateStore.from_config(_config())
    try:
        deleted_ops: list[str] = []
        store._callbacks.register(SECTION_DELETED_TOPIC, lambda p: deleted_ops.append(p["op"]))
        store.add_record("u1", "alerts", {"text": "a"})
        store.put_document("u1", "prefs", {"theme": "dark"})

        assert store.clear("u1") > 0
        assert store.query_events("u1") == []
        assert "clear" in deleted_ops
    finally:
        store.close()


# --------------------------------------------------------------------- #
# 5. A consent-refused write logs nothing (the gate raises before the write).
# --------------------------------------------------------------------- #


async def test_consent_refused_write_appends_no_event_async() -> None:
    store = await AsyncUserStateStore.from_config(
        _config(sections=[{"name": "notes", "kind": "collection", "consent_scope": "pii"}])
    )
    try:
        with pytest.raises(ConsentRequiredError):
            await store.add_record("u1", "notes", {"text": "secret"})
        assert await store.query_events("u1") == []
    finally:
        await store.close()


def test_consent_refused_write_appends_no_event_sync() -> None:
    store = UserStateStore.from_config(
        _config(sections=[{"name": "notes", "kind": "collection", "consent_scope": "pii"}])
    )
    try:
        with pytest.raises(ConsentRequiredError):
            store.add_record("u1", "notes", {"text": "secret"})
        assert store.query_events("u1") == []
    finally:
        store.close()


# --------------------------------------------------------------------- #
# 6. query_events is user-scoped; a second user's log is isolated.
# --------------------------------------------------------------------- #


async def test_query_events_is_user_scoped_async() -> None:
    store = await AsyncUserStateStore.from_config(_config())
    try:
        await store.add_record("u1", "alerts", {"text": "a"})
        await store.add_record("u2", "alerts", {"text": "b"})

        assert len(await store.query_events("u1")) == 1
        assert len(await store.query_events("u2")) == 1
    finally:
        await store.close()


def test_query_events_is_user_scoped_sync() -> None:
    store = UserStateStore.from_config(_config())
    try:
        store.add_record("u1", "alerts", {"text": "a"})
        store.add_record("u2", "alerts", {"text": "b"})

        assert len(store.query_events("u1")) == 1
        assert len(store.query_events("u2")) == 1
    finally:
        store.close()


# --------------------------------------------------------------------- #
# 7. Disabled event log: query_events raises; no reserved section registered.
# --------------------------------------------------------------------- #


async def test_query_events_raises_when_disabled_async() -> None:
    store = await AsyncUserStateStore.from_config(_config(enable_event_log=False))
    try:
        with pytest.raises(ConfigurationError):
            await store.query_events("u1")
    finally:
        await store.close()


def test_query_events_raises_when_disabled_sync() -> None:
    store = UserStateStore.from_config(_config(enable_event_log=False))
    try:
        with pytest.raises(ConfigurationError):
            store.query_events("u1")
    finally:
        store.close()


# --------------------------------------------------------------------- #
# 8. The event log honours its own retention window via section-less prune.
# --------------------------------------------------------------------- #


async def test_event_log_honours_retention_window_async() -> None:
    clock = _Clock(_START)
    store = await AsyncUserStateStore.from_config(_config(event_log_retention_days=30), now=clock)
    try:
        await store.add_record("u1", "alerts", {"text": "old"})  # logs 1 event
        clock.advance(days=40)

        # Section-less prune sweeps the windowed ``events`` section too; the
        # single old add_record event is past the 30-day window.
        await store.prune("u1")
        remaining_ops = sorted(e.data["op"] for e in await store.query_events("u1"))
        # The stale add_record event is gone; only the fresh prune record remains.
        assert remaining_ops == ["prune"]
    finally:
        await store.close()


def test_event_log_honours_retention_window_sync() -> None:
    clock = _Clock(_START)
    store = UserStateStore.from_config(_config(event_log_retention_days=30), now=clock)
    try:
        store.add_record("u1", "alerts", {"text": "old"})
        clock.advance(days=40)

        store.prune("u1")
        remaining_ops = sorted(e.data["op"] for e in store.query_events("u1"))
        assert remaining_ops == ["prune"]
    finally:
        store.close()


# --------------------------------------------------------------------- #
# 9. Metadata-only under a bound tenant; the log is tenant-scoped.
# --------------------------------------------------------------------- #


async def test_event_log_metadata_only_with_tenant_async() -> None:
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
    store = AsyncUserStateStore.from_components(cfg, db=db, tenant=BoundTenantContext("t1", "acme"))
    try:
        await store.add_record("u1", "profile", {"ssn": "secret"})

        events = await store.query_events("u1")
        assert len(events) == 1
        data = events[0].data
        assert "ssn" not in data
        assert data["op"] == "add_record"
        assert data["op_section"] == "profile"
        assert data["tenant_id"] == "t1"
    finally:
        await store.close()


def test_event_log_metadata_only_with_tenant_sync() -> None:
    db = SyncMemoryDatabase()
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
    store = UserStateStore.from_components(cfg, db=db, tenant=BoundTenantContext("t1", "acme"))
    try:
        store.add_record("u1", "profile", {"ssn": "secret"})

        events = store.query_events("u1")
        assert len(events) == 1
        data = events[0].data
        assert "ssn" not in data
        assert data["op"] == "add_record"
        assert data["op_section"] == "profile"
        assert data["tenant_id"] == "t1"
    finally:
        store.close()


# --------------------------------------------------------------------- #
# 10. Config guards: reserved name + non-positive retention window.
# --------------------------------------------------------------------- #


def test_reserved_events_section_name_rejected() -> None:
    with pytest.raises(ConfigurationError):
        UserStateStoreConfig.from_dict(_config(sections=[{"name": "events", "kind": "collection"}]))


def test_non_positive_event_log_retention_rejected() -> None:
    with pytest.raises(ConfigurationError):
        UserStateStoreConfig.from_dict(_config(event_log_retention_days=0))
    with pytest.raises(ConfigurationError):
        UserStateStoreConfig.from_dict(_config(event_log_retention_days=-5))


# --------------------------------------------------------------------- #
# 11. Parity: both variants carry the event-log surface.
# --------------------------------------------------------------------- #


def test_both_variants_have_event_log_surface() -> None:
    for method in ("query_events", "_append_event"):
        assert hasattr(AsyncUserStateStore, method)
        assert hasattr(UserStateStore, method)
    assert RESERVED_EVENTS_SECTION == "events"


# --------------------------------------------------------------------- #
# 12. Best-effort append: a failed audit write never fails the primary op.
# --------------------------------------------------------------------- #
#
# Real backend subclasses (not mocks) whose ``create`` fails only for the
# reserved ``events`` section — the audit append. Every other write (the
# primary content record) goes through untouched, so the primary op persists
# and only the secondary audit append hits the injected fault.


class _EventsAppendFailsAsyncDB(AsyncMemoryDatabase):
    async def create(self, record: Record) -> str:
        if record.get_value("section") == RESERVED_EVENTS_SECTION:
            raise RuntimeError("simulated audit-append backend failure")
        return await super().create(record)


class _EventsAppendFailsSyncDB(SyncMemoryDatabase):
    def create(self, record: Record) -> str:
        if record.get_value("section") == RESERVED_EVENTS_SECTION:
            raise RuntimeError("simulated audit-append backend failure")
        return super().create(record)


async def test_append_failure_does_not_fail_primary_write_async(
    caplog: pytest.LogCaptureFixture,
) -> None:
    cfg = UserStateStoreConfig.from_dict(_config())
    store = AsyncUserStateStore.from_components(cfg, db=_EventsAppendFailsAsyncDB())
    try:
        with caplog.at_level(logging.WARNING, logger="dataknobs_data.user.store"):
            # The audit append raises inside _append_event; the primary
            # add_record must still return the persisted record's id.
            rid = await store.add_record("u1", "alerts", {"text": "a"})

        assert rid  # primary write persisted despite the audit fault
        # And the primary record is durably readable.
        records = await store.query("u1", "alerts")
        assert [r.get_value("text") for r in records] == ["a"]
        # The swallowed failure is logged (metadata only), not silently dropped.
        assert any(
            "event-log append failed" in r.message
            and "add_record" in r.message
            and r.levelno == logging.WARNING
            for r in caplog.records
        )
    finally:
        await store.close()


def test_append_failure_does_not_fail_primary_write_sync(
    caplog: pytest.LogCaptureFixture,
) -> None:
    cfg = UserStateStoreConfig.from_dict(_config())
    store = UserStateStore.from_components(cfg, db=_EventsAppendFailsSyncDB())
    try:
        with caplog.at_level(logging.WARNING, logger="dataknobs_data.user.store"):
            rid = store.add_record("u1", "alerts", {"text": "a"})

        assert rid
        records = store.query("u1", "alerts")
        assert [r.get_value("text") for r in records] == ["a"]
        assert any(
            "event-log append failed" in r.message
            and "add_record" in r.message
            and r.levelno == logging.WARNING
            for r in caplog.records
        )
    finally:
        store.close()
