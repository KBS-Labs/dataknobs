"""Retention-pruning tests for the per-user state coordinator.

Reproduce-first for the lifecycle layer: a collection section carrying a
``retention_days`` window has records older than the window pruned by an
explicit ``prune()`` call (or lazily on ``query()`` when ``prune_on_query`` is
set). Time is driven by an **injected clock** so the tests are deterministic
with no ``sleep``. A ``retention_days`` on a document section is rejected at
config-load time. Real constructs only (``AsyncMemoryDatabase`` /
``SyncMemoryDatabase``).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from dataknobs_common.exceptions import ConfigurationError
from dataknobs_data.backends.memory import AsyncMemoryDatabase
from dataknobs_data.records import Record
from dataknobs_data.user import (
    AsyncUserStateStore,
    UserStateStore,
    UserStateStoreConfig,
)
from dataknobs_data.user.store import _is_expired

# ``activity`` prunes at 30 days; ``notes`` has no window (never prunes);
# ``prefs`` is a document (documents never expire).
_SECTIONS = [
    {"name": "prefs", "kind": "document"},
    {"name": "activity", "kind": "collection", "retention_days": 30},
    {"name": "notes", "kind": "collection"},
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
    }
    base.update(overrides)
    return base


# --------------------------------------------------------------------- #
# 6. prune() removes records past the retention window (injected clock).
# --------------------------------------------------------------------- #


async def test_prune_removes_expired_records_async() -> None:
    clock = _Clock(_START)
    store = await AsyncUserStateStore.from_config(_config(), now=clock)
    try:
        await store.add_record("u1", "activity", {"event": "old"})
        clock.advance(days=40)
        await store.add_record("u1", "activity", {"event": "new"})

        pruned = await store.prune("u1", "activity")
        assert pruned == 1

        remaining = await store.query("u1", "activity")
        assert {r.get_value("event") for r in remaining} == {"new"}
    finally:
        await store.close()


def test_prune_removes_expired_records_sync() -> None:
    clock = _Clock(_START)
    store = UserStateStore.from_config(_config(), now=clock)
    try:
        store.add_record("u1", "activity", {"event": "old"})
        clock.advance(days=40)
        store.add_record("u1", "activity", {"event": "new"})

        assert store.prune("u1", "activity") == 1
        remaining = store.query("u1", "activity")
        assert {r.get_value("event") for r in remaining} == {"new"}
    finally:
        store.close()


async def test_prune_all_sections_skips_unwindowed_and_documents() -> None:
    clock = _Clock(_START)
    store = await AsyncUserStateStore.from_config(_config(), now=clock)
    try:
        await store.add_record("u1", "activity", {"event": "old"})
        await store.add_record("u1", "notes", {"text": "keep"})
        await store.put_document("u1", "prefs", {"theme": "dark"})
        clock.advance(days=100)

        # section=None prunes every windowed collection section; ``notes`` has
        # no window and ``prefs`` is a document, so both are untouched.
        pruned = await store.prune("u1")
        assert pruned == 1
        assert await store.query("u1", "activity") == []
        assert len(await store.query("u1", "notes")) == 1
        assert (await store.get_document("u1", "prefs")) is not None
    finally:
        await store.close()


async def test_prune_document_section_rejected() -> None:
    clock = _Clock(_START)
    store = await AsyncUserStateStore.from_config(_config(), now=clock)
    try:
        # A document section cannot be pruned explicitly (wrong kind).
        with pytest.raises(ValueError):
            await store.prune("u1", "prefs")
    finally:
        await store.close()


# --------------------------------------------------------------------- #
# 7. prune_on_query prunes lazily on read when enabled; off by default.
# --------------------------------------------------------------------- #


async def test_prune_on_query_when_enabled() -> None:
    clock = _Clock(_START)
    store = await AsyncUserStateStore.from_config(
        _config(prune_on_query=True), now=clock
    )
    try:
        await store.add_record("u1", "activity", {"event": "old"})
        clock.advance(days=40)
        await store.add_record("u1", "activity", {"event": "new"})

        # The read prunes first, so only the fresh record surfaces...
        rows = await store.query("u1", "activity")
        assert {r.get_value("event") for r in rows} == {"new"}
        # ...and the expired record is actually deleted, not just filtered.
        rows_again = await store.query("u1", "activity")
        assert {r.get_value("event") for r in rows_again} == {"new"}
    finally:
        await store.close()


async def test_prune_on_query_off_by_default() -> None:
    clock = _Clock(_START)
    store = await AsyncUserStateStore.from_config(_config(), now=clock)
    try:
        await store.add_record("u1", "activity", {"event": "old"})
        clock.advance(days=40)
        await store.add_record("u1", "activity", {"event": "new"})

        # No prune_on_query → the expired record is still returned.
        rows = await store.query("u1", "activity")
        assert {r.get_value("event") for r in rows} == {"old", "new"}
    finally:
        await store.close()


def test_prune_on_query_config_default_false() -> None:
    cfg = UserStateStoreConfig.from_dict(_config())
    assert cfg.prune_on_query is False
    cfg_on = UserStateStoreConfig.from_dict(_config(prune_on_query=True))
    assert cfg_on.prune_on_query is True


# --------------------------------------------------------------------- #
# 8. retention_days on a document section is a load-time ConfigurationError.
# --------------------------------------------------------------------- #


def test_retention_days_on_document_section_rejected() -> None:
    with pytest.raises(ConfigurationError):
        UserStateStoreConfig.from_dict(
            {
                "namespace": "acme",
                "sections": [
                    {"name": "prefs", "kind": "document", "retention_days": 30}
                ],
            }
        )


def test_retention_days_on_collection_section_allowed() -> None:
    cfg = UserStateStoreConfig.from_dict(
        {
            "namespace": "acme",
            "sections": [
                {"name": "activity", "kind": "collection", "retention_days": 30}
            ],
        }
    )
    assert cfg.sections[0].retention_days == 30


# --------------------------------------------------------------------- #
# 16. Method parity — prune exists on both variants with matching signature.
# --------------------------------------------------------------------- #


def test_prune_method_parity() -> None:
    import inspect

    assert hasattr(AsyncUserStateStore, "prune")
    assert hasattr(UserStateStore, "prune")
    assert inspect.signature(AsyncUserStateStore.prune) == inspect.signature(
        UserStateStore.prune
    )


# --------------------------------------------------------------------- #
# A non-positive retention window is a load-time ConfigurationError.
#
# A mis-signed window (0 or negative) turns ``_is_expired`` into "everything
# is expired" — ``written < now - timedelta(days=-30)`` == ``written < now +
# 30 days`` — so the next prune deletes live data. Reject it at the boundary,
# the same place a document+retention mistake is caught.
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("bad", [0, -1, -30])
def test_non_positive_retention_days_rejected(bad: int) -> None:
    with pytest.raises(ConfigurationError):
        UserStateStoreConfig.from_dict(
            {
                "namespace": "acme",
                "sections": [
                    {
                        "name": "activity",
                        "kind": "collection",
                        "retention_days": bad,
                    }
                ],
            }
        )


# --------------------------------------------------------------------- #
# Timezone-mismatch fail-safe: an aware stamp compared to a naive ``now``
# (or vice-versa) is treated as not-expired, never raising TypeError.
# ``_is_expired`` promises it "never deletes a record it cannot confidently
# date" — a tz mismatch is exactly "cannot confidently compare".
# --------------------------------------------------------------------- #


def test_is_expired_tz_mismatch_is_not_expired() -> None:
    aware = datetime(2026, 1, 1, tzinfo=UTC)
    naive = datetime(2099, 1, 1)  # far future — WOULD expire if comparable

    aware_stamp = Record({"_written_at": aware.isoformat()})
    naive_stamp = Record({"_written_at": naive.isoformat()})

    # Aware stamp vs a naive far-future clock: uncomparable -> not expired.
    assert _is_expired(aware_stamp, 30, naive) is False
    # Naive stamp vs an aware far-future clock: symmetric, also not expired.
    assert _is_expired(naive_stamp, 30, aware) is False


async def test_prune_tz_mismatch_does_not_crash_or_delete() -> None:
    # Store A writes with the default tz-aware wall clock; store B prunes the
    # same backend with a NAIVE far-future clock. Pre-fix this raised
    # TypeError inside prune; the record must instead survive (return 0).
    db = AsyncMemoryDatabase()
    cfg = UserStateStoreConfig.from_dict(_config())
    writer = AsyncUserStateStore.from_components(cfg, db=db)  # aware default
    pruner = AsyncUserStateStore.from_components(
        cfg, db=db, now=lambda: datetime(2099, 1, 1),  # naive far future
    )
    try:
        await writer.add_record("u1", "activity", {"event": "keep"})
        assert await pruner.prune("u1", "activity") == 0
        rows = await pruner.query("u1", "activity")
        assert {r.get_value("event") for r in rows} == {"keep"}
    finally:
        await writer.close()
        await pruner.close()
