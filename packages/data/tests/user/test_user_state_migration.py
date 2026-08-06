"""Section schema-version + lazy on-read migration tests.

Reproduce-first for the migration layer: a record stamped with a
``_section_version`` behind its section's current ``version`` is upgraded in
memory on read by the registered per-section upgrader chain, without touching
the stored record by default. Opt-in ``persist_migrations`` writes the upgraded
record back with an optimistic-concurrency (compare-and-set) guard; a conflict
is swallowed and the in-memory upgrade is still returned. A record *newer* than
the running spec (a rollback) passes through unchanged with a WARNING
(read fail-open); a missing step in the upgrade chain raises at read time (a
consumer wiring bug, not a data condition).

Real constructs only (``AsyncMemoryDatabase`` / ``SyncMemoryDatabase``; a small
real ``AsyncDatabase`` / ``SyncDatabase`` subclass whose ``update`` raises to
exercise the CAS-conflict swallow — no mocks). A record at an older version is
seeded by writing it through a store whose section is declared at the older
``version`` and reading it through a second store, sharing one database, whose
section is declared at the newer ``version``. Time is an injected clock so the
retention-clock-preservation case is deterministic. Every behavioral case is
written for both the async and sync variants.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from dataknobs_common.exceptions import ConcurrencyError, ConfigurationError
from dataknobs_common.tenancy import BoundTenantContext
from dataknobs_data.backends.memory import (
    AsyncMemoryDatabase,
    SyncMemoryDatabase,
)
from dataknobs_data.records import Record
from dataknobs_data.user import (
    AsyncUserStateStore,
    UserStateStore,
    UserStateStoreConfig,
)
from dataknobs_data.user.migration import (
    SectionMigrator,
    register_section_migrator,
    resolve_chain,
    section_migrators,
)
from dataknobs_data.user.store import SECTION_WRITTEN_TOPIC

_START = datetime(2026, 1, 1, tzinfo=UTC)


class _Clock:
    """A deterministic, advanceable UTC clock injected as the ``now`` component."""

    def __init__(self, start: datetime) -> None:
        self.value = start

    def __call__(self) -> datetime:
        return self.value

    def advance(self, **kwargs: Any) -> None:
        self.value = self.value + timedelta(**kwargs)


# --- upgraders (pure; take a v_n payload, return the v_{n+1} payload) --- #


def _rename_color_to_theme(payload: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(payload)
    if "color" in out:
        out["theme"] = out.pop("color")
    return out


def _rename_theme_to_appearance(payload: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(payload)
    if "theme" in out:
        out["appearance"] = out.pop("theme")
    return out


def _leaky_upgrader(payload: Mapping[str, Any]) -> dict[str, Any]:
    """A buggy upgrader that (wrongly) emits coordinator-owned scope stamps.

    An upgrader is contracted to return the *consumer payload only*. This one
    additionally returns a literal ``tenant_id`` / ``_written_at`` /
    ``_section_version`` — the reserved scope stamps. The migration boundary
    strips reserved keys from the upgrader's *output* (symmetric with the input
    strip), so none of these can survive into the returned or persisted record;
    the coordinator's own re-stamp is the sole authority on those fields.
    """
    out = dict(payload)
    out["theme"] = out.pop("color", "light")
    out["tenant_id"] = "SPOOFED"
    out["_written_at"] = "SPOOFED"
    out["_section_version"] = 999
    return out


def _doc_cfg(version: int, **overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "backend": "memory",
        "namespace": "acme",
        "sections": [{"name": "prefs", "kind": "document", "version": version}],
    }
    base.update(overrides)
    return base


def _coll_cfg(version: int, **overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "backend": "memory",
        "namespace": "acme",
        "sections": [{"name": "notes", "kind": "collection", "version": version}],
    }
    base.update(overrides)
    return base


@pytest.fixture
def migrator():
    """Register per-section upgraders and unregister them after the test.

    The ``section_migrators`` registry is process-global (mirroring the
    stage-synthesizer / intent-classifier registries), so a test that registers
    a migrator must remove it afterward or it leaks into the next test.
    """
    registered: list[str] = []

    def _reg(section: str, from_version: int, fn: Any) -> None:
        register_section_migrator(section, from_version, fn)
        if section not in registered:
            registered.append(section)

    yield _reg
    for section in registered:
        if section_migrators.has(section):
            section_migrators.unregister(section)


# --------------------------------------------------------------------- #
# SectionMigrator / registry unit behavior (no store).
# --------------------------------------------------------------------- #


def test_section_migrator_chain_orders_steps() -> None:
    m = (
        SectionMigrator("prefs")
        .with_step(1, _rename_color_to_theme)
        .with_step(2, _rename_theme_to_appearance)
    )
    chain = m.chain(1, 3)
    assert chain == [_rename_color_to_theme, _rename_theme_to_appearance]
    # A no-op window returns an empty chain.
    assert m.chain(3, 3) == []
    assert m.chain(3, 1) == []


def test_section_migrator_is_immutable_with_step() -> None:
    base = SectionMigrator("prefs")
    extended = base.with_step(1, _rename_color_to_theme)
    # ``with_step`` returns a new migrator; the original is untouched.
    assert dict(base.upgraders) == {}
    assert dict(extended.upgraders) == {1: _rename_color_to_theme}
    assert extended.chain(1, 2) == [_rename_color_to_theme]
    assert base is not extended


def test_section_migrator_chain_gap_raises() -> None:
    m = SectionMigrator("prefs").with_step(1, _rename_color_to_theme)
    # 1->3 needs a v2->v3 step that was never registered.
    with pytest.raises(ConfigurationError):
        m.chain(1, 3)


def test_register_section_migrator_accumulates_steps(migrator: Any) -> None:
    migrator("prefs", 1, _rename_color_to_theme)
    migrator("prefs", 2, _rename_theme_to_appearance)
    chain = resolve_chain("prefs", 1, 3)
    assert chain == [_rename_color_to_theme, _rename_theme_to_appearance]


def test_resolve_chain_no_migrator_registered_raises() -> None:
    # A record needs migration but the section has no registered migrator.
    with pytest.raises(ConfigurationError):
        resolve_chain("never-registered-section", 1, 2)


def test_resolve_chain_noop_window_is_empty() -> None:
    # No migration needed -> empty chain, no registry lookup, no raise.
    assert resolve_chain("unknown", 2, 2) == []
    assert resolve_chain("unknown", 3, 1) == []


# --------------------------------------------------------------------- #
# 12. Behind-version record upgraded on read (in-memory); stored record
#     unchanged when persist_migrations is off (the default).
# --------------------------------------------------------------------- #


async def test_document_migrated_on_read_in_memory_async(migrator: Any) -> None:
    migrator("prefs", 1, _rename_color_to_theme)
    db = AsyncMemoryDatabase()
    store_v1 = AsyncUserStateStore.from_components(
        UserStateStoreConfig.from_dict(_doc_cfg(1)), db=db
    )
    store_v2 = AsyncUserStateStore.from_components(
        UserStateStoreConfig.from_dict(_doc_cfg(2)), db=db
    )
    try:
        await store_v1.put_document("u1", "prefs", {"color": "dark"})

        migrated = await store_v2.get_document("u1", "prefs")
        assert migrated is not None
        assert migrated.get_value("theme") == "dark"
        assert "color" not in migrated.data
        assert migrated.get_value("_section_version") == 2

        # The stored record is untouched (persist_migrations off by default).
        raw = await store_v1.get_document("u1", "prefs")
        assert raw is not None
        assert raw.get_value("color") == "dark"
        assert raw.get_value("_section_version") == 1
    finally:
        await store_v2.close()


def test_document_migrated_on_read_in_memory_sync(migrator: Any) -> None:
    migrator("prefs", 1, _rename_color_to_theme)
    db = SyncMemoryDatabase()
    store_v1 = UserStateStore.from_components(
        UserStateStoreConfig.from_dict(_doc_cfg(1)), db=db
    )
    store_v2 = UserStateStore.from_components(
        UserStateStoreConfig.from_dict(_doc_cfg(2)), db=db
    )
    try:
        store_v1.put_document("u1", "prefs", {"color": "dark"})

        migrated = store_v2.get_document("u1", "prefs")
        assert migrated is not None
        assert migrated.get_value("theme") == "dark"
        assert "color" not in migrated.data
        assert migrated.get_value("_section_version") == 2

        raw = store_v1.get_document("u1", "prefs")
        assert raw is not None
        assert raw.get_value("color") == "dark"
        assert raw.get_value("_section_version") == 1
    finally:
        store_v2.close()


async def test_multi_step_chain_applied_in_order_async(migrator: Any) -> None:
    migrator("prefs", 1, _rename_color_to_theme)
    migrator("prefs", 2, _rename_theme_to_appearance)
    db = AsyncMemoryDatabase()
    store_v1 = AsyncUserStateStore.from_components(
        UserStateStoreConfig.from_dict(_doc_cfg(1)), db=db
    )
    store_v3 = AsyncUserStateStore.from_components(
        UserStateStoreConfig.from_dict(_doc_cfg(3)), db=db
    )
    try:
        await store_v1.put_document("u1", "prefs", {"color": "dark"})
        migrated = await store_v3.get_document("u1", "prefs")
        assert migrated is not None
        assert migrated.get_value("appearance") == "dark"
        assert "color" not in migrated.data
        assert "theme" not in migrated.data
        assert migrated.get_value("_section_version") == 3
    finally:
        await store_v3.close()


# --------------------------------------------------------------------- #
# 13. persist_migrations writes the upgrade back (CAS); a conflict is
#     swallowed and the in-memory upgrade is still returned.
# --------------------------------------------------------------------- #


async def test_persist_migration_writes_back_async(migrator: Any) -> None:
    migrator("prefs", 1, _rename_color_to_theme)
    db = AsyncMemoryDatabase()
    store_v1 = AsyncUserStateStore.from_components(
        UserStateStoreConfig.from_dict(_doc_cfg(1)), db=db
    )
    store_v2 = AsyncUserStateStore.from_components(
        UserStateStoreConfig.from_dict(_doc_cfg(2, persist_migrations=True)),
        db=db,
    )
    try:
        await store_v1.put_document("u1", "prefs", {"color": "dark"})

        migrated = await store_v2.get_document("u1", "prefs")
        assert migrated is not None
        assert migrated.get_value("theme") == "dark"

        # The upgrade was persisted: reading the raw stored record now shows v2.
        raw = await store_v1.get_document("u1", "prefs")
        assert raw is not None
        assert raw.get_value("theme") == "dark"
        assert "color" not in raw.data
        assert raw.get_value("_section_version") == 2
    finally:
        await store_v2.close()


def test_persist_migration_writes_back_sync(migrator: Any) -> None:
    migrator("prefs", 1, _rename_color_to_theme)
    db = SyncMemoryDatabase()
    store_v1 = UserStateStore.from_components(
        UserStateStoreConfig.from_dict(_doc_cfg(1)), db=db
    )
    store_v2 = UserStateStore.from_components(
        UserStateStoreConfig.from_dict(_doc_cfg(2, persist_migrations=True)),
        db=db,
    )
    try:
        store_v1.put_document("u1", "prefs", {"color": "dark"})

        migrated = store_v2.get_document("u1", "prefs")
        assert migrated is not None
        assert migrated.get_value("theme") == "dark"

        raw = store_v1.get_document("u1", "prefs")
        assert raw is not None
        assert raw.get_value("theme") == "dark"
        assert raw.get_value("_section_version") == 2
    finally:
        store_v2.close()


class _ConflictOnUpdateAsyncDB(AsyncMemoryDatabase):
    """Real async backend whose ``update`` always signals a CAS conflict.

    A purpose-built test construct (not a mock) exercising the persist-on-read
    ``ConcurrencyError`` swallow path: seeding still goes through the real
    ``create`` / ``upsert`` code, only the migration write-back conflicts.
    """

    async def update(
        self, id: str, record: Record, *, expected_version: str | None = None
    ) -> bool:
        raise ConcurrencyError("simulated concurrent bump", context={"id": id})


class _ConflictOnUpdateSyncDB(SyncMemoryDatabase):
    """Sync mirror of :class:`_ConflictOnUpdateAsyncDB`."""

    def update(
        self, id: str, record: Record, *, expected_version: str | None = None
    ) -> bool:
        raise ConcurrencyError("simulated concurrent bump", context={"id": id})


async def test_persist_migration_cas_conflict_swallowed_async(
    migrator: Any,
) -> None:
    migrator("prefs", 1, _rename_color_to_theme)
    db = _ConflictOnUpdateAsyncDB()
    store_v1 = AsyncUserStateStore.from_components(
        UserStateStoreConfig.from_dict(_doc_cfg(1)), db=db
    )
    store_v2 = AsyncUserStateStore.from_components(
        UserStateStoreConfig.from_dict(_doc_cfg(2, persist_migrations=True)),
        db=db,
    )
    try:
        await store_v1.put_document("u1", "prefs", {"color": "dark"})

        # The CAS write conflicts; it is swallowed and the in-memory upgrade
        # is still returned (no raise reaches the caller).
        migrated = await store_v2.get_document("u1", "prefs")
        assert migrated is not None
        assert migrated.get_value("theme") == "dark"
        assert migrated.get_value("_section_version") == 2

        # The stored record was never advanced (the write was blocked).
        raw = await store_v1.get_document("u1", "prefs")
        assert raw is not None
        assert raw.get_value("color") == "dark"
        assert raw.get_value("_section_version") == 1
    finally:
        await store_v2.close()


def test_persist_migration_cas_conflict_swallowed_sync(migrator: Any) -> None:
    migrator("prefs", 1, _rename_color_to_theme)
    db = _ConflictOnUpdateSyncDB()
    store_v1 = UserStateStore.from_components(
        UserStateStoreConfig.from_dict(_doc_cfg(1)), db=db
    )
    store_v2 = UserStateStore.from_components(
        UserStateStoreConfig.from_dict(_doc_cfg(2, persist_migrations=True)),
        db=db,
    )
    try:
        store_v1.put_document("u1", "prefs", {"color": "dark"})

        migrated = store_v2.get_document("u1", "prefs")
        assert migrated is not None
        assert migrated.get_value("theme") == "dark"

        raw = store_v1.get_document("u1", "prefs")
        assert raw is not None
        assert raw.get_value("color") == "dark"
        assert raw.get_value("_section_version") == 1
    finally:
        store_v2.close()


# --------------------------------------------------------------------- #
# 14. Newer-than-spec record passes through + logs a WARNING (fail-open).
# --------------------------------------------------------------------- #


async def test_downgrade_passthrough_warns_async(
    migrator: Any, caplog: pytest.LogCaptureFixture
) -> None:
    # No migrator needed: a rollback reads a record newer than its spec.
    db = AsyncMemoryDatabase()
    store_v2 = AsyncUserStateStore.from_components(
        UserStateStoreConfig.from_dict(_doc_cfg(2)), db=db
    )
    store_v1 = AsyncUserStateStore.from_components(
        UserStateStoreConfig.from_dict(_doc_cfg(1)), db=db
    )
    try:
        await store_v2.put_document("u1", "prefs", {"theme": "dark"})

        with caplog.at_level(logging.WARNING):
            record = await store_v1.get_document("u1", "prefs")

        assert record is not None
        # Passed through unchanged: still the v2 content and stamp.
        assert record.get_value("theme") == "dark"
        assert record.get_value("_section_version") == 2
        assert any(
            "newer than" in r.message and r.levelno == logging.WARNING
            for r in caplog.records
        )
    finally:
        await store_v1.close()


def test_downgrade_passthrough_warns_sync(
    migrator: Any, caplog: pytest.LogCaptureFixture
) -> None:
    db = SyncMemoryDatabase()
    store_v2 = UserStateStore.from_components(
        UserStateStoreConfig.from_dict(_doc_cfg(2)), db=db
    )
    store_v1 = UserStateStore.from_components(
        UserStateStoreConfig.from_dict(_doc_cfg(1)), db=db
    )
    try:
        store_v2.put_document("u1", "prefs", {"theme": "dark"})

        with caplog.at_level(logging.WARNING):
            record = store_v1.get_document("u1", "prefs")

        assert record is not None
        assert record.get_value("theme") == "dark"
        assert record.get_value("_section_version") == 2
        assert any(
            "newer than" in r.message and r.levelno == logging.WARNING
            for r in caplog.records
        )
    finally:
        store_v1.close()


# --------------------------------------------------------------------- #
# 15. A migration-chain gap raises ConfigurationError at read.
# --------------------------------------------------------------------- #


async def test_chain_gap_raises_at_read_async(migrator: Any) -> None:
    # Only v1->v2 is registered; the record must reach v3 -> gap.
    migrator("prefs", 1, _rename_color_to_theme)
    db = AsyncMemoryDatabase()
    store_v1 = AsyncUserStateStore.from_components(
        UserStateStoreConfig.from_dict(_doc_cfg(1)), db=db
    )
    store_v3 = AsyncUserStateStore.from_components(
        UserStateStoreConfig.from_dict(_doc_cfg(3)), db=db
    )
    try:
        await store_v1.put_document("u1", "prefs", {"color": "dark"})
        with pytest.raises(ConfigurationError):
            await store_v3.get_document("u1", "prefs")
    finally:
        await store_v3.close()


def test_chain_gap_raises_at_read_sync(migrator: Any) -> None:
    migrator("prefs", 1, _rename_color_to_theme)
    db = SyncMemoryDatabase()
    store_v1 = UserStateStore.from_components(
        UserStateStoreConfig.from_dict(_doc_cfg(1)), db=db
    )
    store_v3 = UserStateStore.from_components(
        UserStateStoreConfig.from_dict(_doc_cfg(3)), db=db
    )
    try:
        store_v1.put_document("u1", "prefs", {"color": "dark"})
        with pytest.raises(ConfigurationError):
            store_v3.get_document("u1", "prefs")
    finally:
        store_v3.close()


async def test_no_migrator_registered_raises_at_read_async() -> None:
    # A behind-version record but no migrator at all for the section.
    db = AsyncMemoryDatabase()
    store_v1 = AsyncUserStateStore.from_components(
        UserStateStoreConfig.from_dict(_doc_cfg(1)), db=db
    )
    store_v2 = AsyncUserStateStore.from_components(
        UserStateStoreConfig.from_dict(_doc_cfg(2)), db=db
    )
    try:
        await store_v1.put_document("u1", "prefs", {"color": "dark"})
        with pytest.raises(ConfigurationError):
            await store_v2.get_document("u1", "prefs")
    finally:
        await store_v2.close()


# --------------------------------------------------------------------- #
# 16. Collection query + snapshot migrate transitively.
# --------------------------------------------------------------------- #


async def test_collection_query_migrates_records_async(migrator: Any) -> None:
    migrator("notes", 1, _rename_color_to_theme)
    db = AsyncMemoryDatabase()
    store_v1 = AsyncUserStateStore.from_components(
        UserStateStoreConfig.from_dict(_coll_cfg(1)), db=db
    )
    store_v2 = AsyncUserStateStore.from_components(
        UserStateStoreConfig.from_dict(_coll_cfg(2)), db=db
    )
    try:
        await store_v1.add_record("u1", "notes", {"color": "dark"})
        await store_v1.add_record("u1", "notes", {"color": "light"})

        records = await store_v2.query("u1", "notes")
        assert len(records) == 2
        themes = sorted(r.get_value("theme") for r in records)
        assert themes == ["dark", "light"]
        assert all(r.get_value("_section_version") == 2 for r in records)
        assert all("color" not in r.data for r in records)

        # snapshot migrates transitively (it reads through query).
        view = await store_v2.snapshot("u1")
        snap_themes = sorted(item["theme"] for item in view["notes"])
        assert snap_themes == ["dark", "light"]
    finally:
        await store_v2.close()


def test_collection_query_migrates_records_sync(migrator: Any) -> None:
    migrator("notes", 1, _rename_color_to_theme)
    db = SyncMemoryDatabase()
    store_v1 = UserStateStore.from_components(
        UserStateStoreConfig.from_dict(_coll_cfg(1)), db=db
    )
    store_v2 = UserStateStore.from_components(
        UserStateStoreConfig.from_dict(_coll_cfg(2)), db=db
    )
    try:
        store_v1.add_record("u1", "notes", {"color": "dark"})
        records = store_v2.query("u1", "notes")
        assert len(records) == 1
        assert records[0].get_value("theme") == "dark"
        assert records[0].get_value("_section_version") == 2

        view = store_v2.snapshot("u1")
        assert view["notes"][0]["theme"] == "dark"
    finally:
        store_v2.close()


# --------------------------------------------------------------------- #
# 17. Migration preserves the retention clock (_written_at) and identity.
# --------------------------------------------------------------------- #


async def test_persist_migration_preserves_written_at_async(
    migrator: Any,
) -> None:
    migrator("prefs", 1, _rename_color_to_theme)
    db = AsyncMemoryDatabase()
    old_clock = _Clock(_START)
    store_v1 = AsyncUserStateStore.from_components(
        UserStateStoreConfig.from_dict(_doc_cfg(1)), db=db, now=old_clock
    )
    new_clock = _Clock(_START)
    new_clock.advance(days=100)
    store_v2 = AsyncUserStateStore.from_components(
        UserStateStoreConfig.from_dict(_doc_cfg(2, persist_migrations=True)),
        db=db,
        now=new_clock,
    )
    try:
        await store_v1.put_document("u1", "prefs", {"color": "dark"})
        original = await store_v1.get_document("u1", "prefs")
        assert original is not None
        original_written_at = original.get_value("_written_at")

        migrated = await store_v2.get_document("u1", "prefs")
        assert migrated is not None
        # A lazy read-migration must NOT reset the retention clock.
        assert migrated.get_value("_written_at") == original_written_at

        # The persisted record keeps the original stamp too (not new_clock).
        raw = await store_v1.get_document("u1", "prefs")
        assert raw is not None
        assert raw.get_value("_written_at") == original_written_at
    finally:
        await store_v2.close()


async def test_migration_preserves_tenant_scope_async(migrator: Any) -> None:
    migrator("prefs", 1, _rename_color_to_theme)
    db = AsyncMemoryDatabase()
    tenant = BoundTenantContext("t1", "acme")
    store_v1 = AsyncUserStateStore.from_components(
        UserStateStoreConfig.from_dict(_doc_cfg(1)), db=db, tenant=tenant
    )
    store_v2 = AsyncUserStateStore.from_components(
        UserStateStoreConfig.from_dict(_doc_cfg(2)), db=db, tenant=tenant
    )
    try:
        await store_v1.put_document("u1", "prefs", {"color": "dark"})
        migrated = await store_v2.get_document("u1", "prefs")
        assert migrated is not None
        assert migrated.get_value("theme") == "dark"
        assert migrated.get_value("tenant_id") == "t1"
        assert migrated.get_value("_section_version") == 2
    finally:
        await store_v2.close()


# --------------------------------------------------------------------- #
# 18. Parity: config flag + method surface on both variants.
# --------------------------------------------------------------------- #


def test_persist_migrations_config_flag_defaults_false() -> None:
    cfg = UserStateStoreConfig.from_dict(_doc_cfg(1))
    assert cfg.persist_migrations is False
    cfg2 = UserStateStoreConfig.from_dict(_doc_cfg(1, persist_migrations=True))
    assert cfg2.persist_migrations is True


def test_migration_method_parity() -> None:
    for name in ("_migrate_read_record", "_persist_migration"):
        assert hasattr(AsyncUserStateStore, name)
        assert hasattr(UserStateStore, name)
    # The shared resolve/migrate helpers live on the common mixin.
    from dataknobs_data.user.store import _UserStateStoreCommon

    assert hasattr(_UserStateStoreCommon, "_migrate_on_read")


# --------------------------------------------------------------------- #
# 19. A buggy upgrader cannot leak a coordinator-owned scope stamp: the
#     migration boundary strips reserved keys from the upgrader OUTPUT
#     (symmetric with the input strip), so the isolation guarantee holds
#     in both directions. Reproduce-first for the output-strip fix — the
#     load-bearing assertion is the single-tenant ``tenant_id`` leak, which
#     the coordinator's re-stamp does not otherwise overwrite.
# --------------------------------------------------------------------- #


async def test_upgrader_cannot_leak_scope_stamp_async(migrator: Any) -> None:
    migrator("prefs", 1, _leaky_upgrader)
    db = AsyncMemoryDatabase()
    store_v1 = AsyncUserStateStore.from_components(
        UserStateStoreConfig.from_dict(_doc_cfg(1)), db=db
    )
    # Single-tenant store: no bound tenant, so the coordinator re-stamp never
    # adds a ``tenant_id`` — a leaked one would survive unless stripped.
    store_v2 = AsyncUserStateStore.from_components(
        UserStateStoreConfig.from_dict(_doc_cfg(2)), db=db
    )
    try:
        await store_v1.put_document("u1", "prefs", {"color": "dark"})
        original = await store_v1.get_document("u1", "prefs")
        assert original is not None

        migrated = await store_v2.get_document("u1", "prefs")
        assert migrated is not None
        # The real upgrade still applied.
        assert migrated.get_value("theme") == "dark"
        # The upgrader's spurious reserved stamps are stripped, not leaked:
        assert migrated.get_value("tenant_id") is None
        assert "tenant_id" not in migrated.data
        # The coordinator's own re-stamp is authoritative (never the spoof).
        assert migrated.get_value("_section_version") == 2
        assert (
            migrated.get_value("_written_at")
            == original.get_value("_written_at")
        )
        assert migrated.get_value("_written_at") != "SPOOFED"
    finally:
        await store_v2.close()


def test_upgrader_cannot_leak_scope_stamp_sync(migrator: Any) -> None:
    migrator("prefs", 1, _leaky_upgrader)
    db = SyncMemoryDatabase()
    store_v1 = UserStateStore.from_components(
        UserStateStoreConfig.from_dict(_doc_cfg(1)), db=db
    )
    store_v2 = UserStateStore.from_components(
        UserStateStoreConfig.from_dict(_doc_cfg(2)), db=db
    )
    try:
        store_v1.put_document("u1", "prefs", {"color": "dark"})
        original = store_v1.get_document("u1", "prefs")
        assert original is not None

        migrated = store_v2.get_document("u1", "prefs")
        assert migrated is not None
        assert migrated.get_value("theme") == "dark"
        assert migrated.get_value("tenant_id") is None
        assert "tenant_id" not in migrated.data
        assert migrated.get_value("_section_version") == 2
        assert (
            migrated.get_value("_written_at")
            == original.get_value("_written_at")
        )
        assert migrated.get_value("_written_at") != "SPOOFED"
    finally:
        store_v2.close()


# --------------------------------------------------------------------- #
# 20. A section ``version`` below 1 is rejected at config-load time (the
#     floor matches SectionMigrator.with_step's from_version >= 1), so a
#     0/negative version fails fast at construction rather than surfacing as
#     an unregisterable-step ConfigurationError at read time.
# --------------------------------------------------------------------- #


def test_section_version_below_one_rejected_at_config_load() -> None:
    with pytest.raises(ConfigurationError):
        UserStateStoreConfig.from_dict(_doc_cfg(0))
    with pytest.raises(ConfigurationError):
        UserStateStoreConfig.from_dict(_doc_cfg(-1))
    # Version 1 (the floor) is accepted.
    cfg = UserStateStoreConfig.from_dict(_doc_cfg(1))
    assert cfg.sections[0].version == 1


# --------------------------------------------------------------------- #
# 21. A persist-on-read migration is a representation upgrade, not a
#     semantic write: it emits NO delta event and appends NO audit record,
#     even when the store has the event log enabled.
# --------------------------------------------------------------------- #


async def test_persist_migration_emits_no_event_or_audit_async(
    migrator: Any,
) -> None:
    migrator("prefs", 1, _rename_color_to_theme)
    db = AsyncMemoryDatabase()
    # store_v1 writes without an event log (no audit seeded by the write).
    store_v1 = AsyncUserStateStore.from_components(
        UserStateStoreConfig.from_dict(_doc_cfg(1)), db=db
    )
    store_v2 = AsyncUserStateStore.from_components(
        UserStateStoreConfig.from_dict(
            _doc_cfg(2, persist_migrations=True, enable_event_log=True)
        ),
        db=db,
    )
    try:
        await store_v1.put_document("u1", "prefs", {"color": "dark"})
        written: list[Any] = []
        store_v2._callbacks.register(SECTION_WRITTEN_TOPIC, written.append)

        migrated = await store_v2.get_document("u1", "prefs")
        assert migrated is not None and migrated.get_value("theme") == "dark"

        # The upgrade WAS persisted (a v1 store reading it downgrades-through).
        raw = await store_v1.get_document("u1", "prefs")
        assert raw is not None and raw.get_value("_section_version") == 2

        # ...but the write-back fired no delta event and appended no audit.
        assert written == []
        assert await store_v2.query_events("u1") == []
    finally:
        await store_v2.close()


def test_persist_migration_emits_no_event_or_audit_sync(migrator: Any) -> None:
    migrator("prefs", 1, _rename_color_to_theme)
    db = SyncMemoryDatabase()
    store_v1 = UserStateStore.from_components(
        UserStateStoreConfig.from_dict(_doc_cfg(1)), db=db
    )
    store_v2 = UserStateStore.from_components(
        UserStateStoreConfig.from_dict(
            _doc_cfg(2, persist_migrations=True, enable_event_log=True)
        ),
        db=db,
    )
    try:
        store_v1.put_document("u1", "prefs", {"color": "dark"})
        written: list[Any] = []
        store_v2._callbacks.register(SECTION_WRITTEN_TOPIC, written.append)

        migrated = store_v2.get_document("u1", "prefs")
        assert migrated is not None and migrated.get_value("theme") == "dark"

        raw = store_v1.get_document("u1", "prefs")
        assert raw is not None and raw.get_value("_section_version") == 2

        assert written == []
        assert store_v2.query_events("u1") == []
    finally:
        store_v2.close()


# --------------------------------------------------------------------- #
# 22. Erasure/persist race: a clear() landing between the migration read and
#     its CAS write-back cannot resurrect the deleted record — the memory
#     backend's ``update`` never inserts (returns False for an absent id), so
#     the stale-token write-back is a no-op. The in-memory upgrade is still
#     returned to the caller.
# --------------------------------------------------------------------- #


class _EraseOnGetVersionAsyncDB(AsyncMemoryDatabase):
    """Deletes the record inside ``get_version``, simulating a concurrent
    ``clear()`` landing between the migration read and the CAS write-back."""

    async def get_version(self, id: str) -> str | None:
        token = await super().get_version(id)
        await super().delete(id)
        return token


class _EraseOnGetVersionSyncDB(SyncMemoryDatabase):
    """Sync mirror of :class:`_EraseOnGetVersionAsyncDB`."""

    def get_version(self, id: str) -> str | None:
        token = super().get_version(id)
        super().delete(id)
        return token


async def test_persist_migration_erasure_race_no_resurrection_async(
    migrator: Any,
) -> None:
    migrator("prefs", 1, _rename_color_to_theme)
    db = _EraseOnGetVersionAsyncDB()
    store_v1 = AsyncUserStateStore.from_components(
        UserStateStoreConfig.from_dict(_doc_cfg(1)), db=db
    )
    store_v2 = AsyncUserStateStore.from_components(
        UserStateStoreConfig.from_dict(_doc_cfg(2, persist_migrations=True)),
        db=db,
    )
    try:
        await store_v1.put_document("u1", "prefs", {"color": "dark"})

        # The record is erased mid-migration (inside the write-back's
        # get_version); the caller still receives the in-memory upgrade.
        migrated = await store_v2.get_document("u1", "prefs")
        assert migrated is not None and migrated.get_value("theme") == "dark"

        # The stale-token CAS write-back was a no-op: NOT resurrected.
        raw = await store_v1.get_document("u1", "prefs")
        assert raw is None
    finally:
        await store_v2.close()


def test_persist_migration_erasure_race_no_resurrection_sync(
    migrator: Any,
) -> None:
    migrator("prefs", 1, _rename_color_to_theme)
    db = _EraseOnGetVersionSyncDB()
    store_v1 = UserStateStore.from_components(
        UserStateStoreConfig.from_dict(_doc_cfg(1)), db=db
    )
    store_v2 = UserStateStore.from_components(
        UserStateStoreConfig.from_dict(_doc_cfg(2, persist_migrations=True)),
        db=db,
    )
    try:
        store_v1.put_document("u1", "prefs", {"color": "dark"})

        migrated = store_v2.get_document("u1", "prefs")
        assert migrated is not None and migrated.get_value("theme") == "dark"

        raw = store_v1.get_document("u1", "prefs")
        assert raw is None
    finally:
        store_v2.close()


# --------------------------------------------------------------------- #
# 23. Collection persist-back path: a collection record upgraded on read is
#     written back to storage (the document persist tests cover the document
#     path; this covers the ``query`` -> ``r.storage_id`` collection path).
# --------------------------------------------------------------------- #


async def test_persist_migration_collection_writes_back_async(
    migrator: Any,
) -> None:
    migrator("notes", 1, _rename_color_to_theme)
    db = AsyncMemoryDatabase()
    store_v1 = AsyncUserStateStore.from_components(
        UserStateStoreConfig.from_dict(_coll_cfg(1)), db=db
    )
    store_v2 = AsyncUserStateStore.from_components(
        UserStateStoreConfig.from_dict(_coll_cfg(2, persist_migrations=True)),
        db=db,
    )
    try:
        await store_v1.add_record("u1", "notes", {"color": "dark"})

        migrated = await store_v2.query("u1", "notes")
        assert len(migrated) == 1 and migrated[0].get_value("theme") == "dark"

        # Persisted to the stored collection record (a v1 store downgrades
        # through, surfacing the stored content unchanged).
        stored = await store_v1.query("u1", "notes")
        assert len(stored) == 1
        assert stored[0].get_value("theme") == "dark"
        assert "color" not in stored[0].data
        assert stored[0].get_value("_section_version") == 2
    finally:
        await store_v2.close()


def test_persist_migration_collection_writes_back_sync(migrator: Any) -> None:
    migrator("notes", 1, _rename_color_to_theme)
    db = SyncMemoryDatabase()
    store_v1 = UserStateStore.from_components(
        UserStateStoreConfig.from_dict(_coll_cfg(1)), db=db
    )
    store_v2 = UserStateStore.from_components(
        UserStateStoreConfig.from_dict(_coll_cfg(2, persist_migrations=True)),
        db=db,
    )
    try:
        store_v1.add_record("u1", "notes", {"color": "dark"})

        migrated = store_v2.query("u1", "notes")
        assert len(migrated) == 1 and migrated[0].get_value("theme") == "dark"

        stored = store_v1.query("u1", "notes")
        assert len(stored) == 1
        assert stored[0].get_value("theme") == "dark"
        assert "color" not in stored[0].data
        assert stored[0].get_value("_section_version") == 2
    finally:
        store_v2.close()
