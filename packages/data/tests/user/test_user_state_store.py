"""Behavioral tests for the per-user cross-session state coordinator.

Real constructs only — ``AsyncMemoryDatabase`` / ``SyncMemoryDatabase`` for
storage and ``InMemoryEventBus`` for events (no mocks). The PostgreSQL CAS /
tenant path lives in ``test_user_state_store_postgres.py``.
"""

from __future__ import annotations

from typing import Any

import pytest

from dataknobs_data.user import (
    AsyncUserStateStore,
    SectionKind,
    Sensitivity,
    UserStateSectionSpec,
    UserStateStore,
    UserStateStoreConfig,
)
from dataknobs_data.user.store import SECTION_WRITTEN_TOPIC
from dataknobs_common.events import InMemoryEventBus
from dataknobs_common.exceptions import ConcurrencyError, ConfigurationError
from dataknobs_common.tenancy import BoundTenantContext
from dataknobs_common.testing import (
    assert_structured_config_consumer,
    assert_structured_config_roundtrip,
)
from dataknobs_data.backends.memory import AsyncMemoryDatabase, SyncMemoryDatabase

# Neutral, domain-free sections used across the suite.
_SECTIONS = [
    {"name": "preferences", "kind": "document"},
    {"name": "profile", "kind": "document", "sensitivity": "sensitive"},
    {"name": "alerts", "kind": "collection"},
]


def _config(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "backend": "memory",
        "namespace": "acme",
        "sections": list(_SECTIONS),
    }
    base.update(overrides)
    return base


async def _make_async(**overrides: Any) -> AsyncUserStateStore:
    return await AsyncUserStateStore.from_config(_config(**overrides))


# --------------------------------------------------------------------- #
# 1. Document round-trip + version token advances on re-put.
# --------------------------------------------------------------------- #


async def test_document_round_trip() -> None:
    store = await _make_async()
    try:
        await store.put_document("u1", "preferences", {"theme": "dark"})
        record = await store.get_document("u1", "preferences")
        assert record is not None
        assert record.get_value("theme") == "dark"

        v1 = await store.document_version("u1", "preferences")
        await store.put_document("u1", "preferences", {"theme": "light"})
        v2 = await store.document_version("u1", "preferences")
        assert v1 is not None and v2 is not None and v1 != v2
    finally:
        await store.close()


async def test_get_missing_document_returns_none() -> None:
    store = await _make_async()
    try:
        assert await store.get_document("nobody", "preferences") is None
    finally:
        await store.close()


# --------------------------------------------------------------------- #
# 2. Collection round-trip + per-user scoping proof.
# --------------------------------------------------------------------- #


async def test_collection_scoping() -> None:
    store = await _make_async()
    try:
        await store.add_record("u1", "alerts", {"text": "a"})
        await store.add_record("u1", "alerts", {"text": "b"})
        await store.add_record("u2", "alerts", {"text": "c"})

        u1 = await store.query("u1", "alerts")
        assert sorted(r.get_value("text") for r in u1) == ["a", "b"]

        u2 = await store.query("u2", "alerts")
        assert [r.get_value("text") for r in u2] == ["c"]
    finally:
        await store.close()


async def test_query_with_caller_filter() -> None:
    from dataknobs_data import Filter, Operator, Query

    store = await _make_async()
    try:
        await store.add_record("u1", "alerts", {"text": "keep", "level": "high"})
        await store.add_record("u1", "alerts", {"text": "drop", "level": "low"})
        q = Query(filters=[Filter("level", Operator.EQ, "high")])
        rows = await store.query("u1", "alerts", q)
        assert [r.get_value("text") for r in rows] == ["keep"]
    finally:
        await store.close()


# --------------------------------------------------------------------- #
# 3. CAS (reproduce-first): the losing writer must be rejected.
# --------------------------------------------------------------------- #


async def test_cas_without_guard_shows_lost_update() -> None:
    """Reproduce-first: without ``expected_version``, the second write wins
    silently (the lost-update the CAS guard exists to prevent).
    """
    store = await _make_async()
    try:
        await store.put_document("u1", "preferences", {"theme": "dark"})
        # Two writers read the same state, both write unconditionally.
        await store.put_document("u1", "preferences", {"theme": "light"})
        await store.put_document("u1", "preferences", {"theme": "blue"})
        record = await store.get_document("u1", "preferences")
        assert record is not None and record.get_value("theme") == "blue"
    finally:
        await store.close()


async def test_cas_document_rejects_stale_writer() -> None:
    store = await _make_async()
    try:
        await store.put_document("u1", "preferences", {"theme": "dark"})
        token = await store.document_version("u1", "preferences")

        # First writer wins with the token.
        await store.put_document("u1", "preferences", {"theme": "light"}, expected_version=token)
        # Second writer holds the now-stale token → conflict.
        with pytest.raises(ConcurrencyError):
            await store.put_document("u1", "preferences", {"theme": "blue"}, expected_version=token)
    finally:
        await store.close()


async def test_cas_collection_rejects_stale_writer() -> None:
    store = await _make_async()
    try:
        record_id = await store.add_record("u1", "alerts", {"text": "v1"})
        token = await store.record_version("u1", "alerts", record_id)
        assert await store.update_record(
            "u1", "alerts", record_id, {"text": "v2"}, expected_version=token
        )
        with pytest.raises(ConcurrencyError):
            await store.update_record(
                "u1",
                "alerts",
                record_id,
                {"text": "v3"},
                expected_version=token,
            )
    finally:
        await store.close()


# --------------------------------------------------------------------- #
# 4. Cross-session persistence — survives a "restart" over the same backend.
# --------------------------------------------------------------------- #


async def test_cross_session_persistence() -> None:
    db = AsyncMemoryDatabase()
    cfg = UserStateStoreConfig.from_dict(_config())
    session_a = AsyncUserStateStore.from_components(cfg, db=db)
    await session_a.put_document("u1", "preferences", {"theme": "dark"})
    await session_a.add_record("u1", "alerts", {"text": "hi"})
    await session_a.close()  # injected db stays open

    # A fresh coordinator over the SAME backend sees prior state.
    session_b = AsyncUserStateStore.from_components(cfg, db=db)
    record = await session_b.get_document("u1", "preferences")
    assert record is not None and record.get_value("theme") == "dark"
    assert len(await session_b.query("u1", "alerts")) == 1
    await session_b.close()


# --------------------------------------------------------------------- #
# 5. Sync/async parity — one scenario table, identical outcomes.
# --------------------------------------------------------------------- #


def test_sync_round_trip() -> None:
    store = UserStateStore.from_config(_config())
    try:
        store.put_document("u1", "preferences", {"theme": "dark"})
        assert store.get_document("u1", "preferences").get_value("theme") == "dark"

        store.add_record("u1", "alerts", {"text": "a"})
        store.add_record("u2", "alerts", {"text": "b"})
        assert [r.get_value("text") for r in store.query("u1", "alerts")] == ["a"]

        token = store.document_version("u1", "preferences")
        store.put_document("u1", "preferences", {"theme": "light"}, expected_version=token)
        with pytest.raises(ConcurrencyError):
            store.put_document("u1", "preferences", {"theme": "x"}, expected_version=token)
        assert store.clear("u1") == 2
    finally:
        store.close()


@pytest.mark.parametrize("is_async", [False, True])
async def test_snapshot_parity(is_async: bool) -> None:
    """The snapshot view is identical across the sync and async variants."""
    if is_async:
        store: Any = await _make_async()
        await store.put_document("u1", "preferences", {"theme": "dark"})
        await store.put_document("u1", "profile", {"ssn": "secret"})
        await store.add_record("u1", "alerts", {"text": "a"})
        default = await store.snapshot("u1")
        with_sensitive = await store.snapshot("u1", include_sensitive=True)
        await store.close()
    else:
        store = UserStateStore.from_config(_config())
        store.put_document("u1", "preferences", {"theme": "dark"})
        store.put_document("u1", "profile", {"ssn": "secret"})
        store.add_record("u1", "alerts", {"text": "a"})
        default = store.snapshot("u1")
        with_sensitive = store.snapshot("u1", include_sensitive=True)
        store.close()

    # SENSITIVE 'profile' section omitted by default; coordinator fields stripped.
    assert default == {"preferences": {"theme": "dark"}, "alerts": [{"text": "a"}]}
    assert with_sensitive["profile"] == {"ssn": "secret"}


# --------------------------------------------------------------------- #
# 6. Owned-vs-injected teardown.
# --------------------------------------------------------------------- #


class _ClosableMemoryDB(AsyncMemoryDatabase):
    """A real async memory database that records whether it was closed.

    Not a mock — it exercises every real code path of ``AsyncMemoryDatabase``
    and only adds a ``closed`` observation so the ownership teardown contract
    can be verified behaviorally (the base ``close()`` is a no-op).
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.closed = False

    async def close(self) -> None:
        self.closed = True
        await super().close()


async def test_injected_db_left_open() -> None:
    db = _ClosableMemoryDB()
    cfg = UserStateStoreConfig.from_dict(_config())
    store = AsyncUserStateStore.from_components(cfg, db=db)
    assert store._owns_db is False
    await store.close()
    assert db.closed is False  # caller owns the injected db


async def test_owned_db_is_closed() -> None:
    from dataknobs_data.backends import async_backends

    async_backends.register("closable_memory_test", _ClosableMemoryDB)
    try:
        store = await AsyncUserStateStore.from_config(_config(backend="closable_memory_test"))
        assert store._owns_db is True
        db = store._db
        assert isinstance(db, _ClosableMemoryDB)
        await store.close()
        assert db.closed is True  # coordinator owns and closes its own db
    finally:
        async_backends.unregister("closable_memory_test")


def test_from_components_requires_db() -> None:
    cfg = UserStateStoreConfig.from_dict(_config())
    with pytest.raises(TypeError):
        AsyncUserStateStore.from_components(cfg)


# --------------------------------------------------------------------- #
# 7. Tenant scoping over a shared backend.
# --------------------------------------------------------------------- #


async def test_tenant_isolation_over_shared_backend() -> None:
    db = AsyncMemoryDatabase()
    cfg = UserStateStoreConfig.from_dict(_config())
    t1 = AsyncUserStateStore.from_components(cfg, db=db, tenant=BoundTenantContext("t1", "acme"))
    t2 = AsyncUserStateStore.from_components(cfg, db=db, tenant=BoundTenantContext("t2", "acme"))
    # Same user id, same section, different tenants → disjoint state.
    await t1.put_document("u", "preferences", {"theme": "dark"})
    await t2.put_document("u", "preferences", {"theme": "light"})
    await t1.add_record("u", "alerts", {"text": "t1"})
    await t2.add_record("u", "alerts", {"text": "t2"})

    assert (await t1.get_document("u", "preferences")).get_value("theme") == "dark"
    assert (await t2.get_document("u", "preferences")).get_value("theme") == "light"
    assert [r.get_value("text") for r in await t1.query("u", "alerts")] == ["t1"]
    assert [r.get_value("text") for r in await t2.query("u", "alerts")] == ["t2"]


async def test_admin_explicit_tenant_filter_crosses_tenants() -> None:
    from dataknobs_data import Filter, Operator, Query

    db = AsyncMemoryDatabase()
    cfg = UserStateStoreConfig.from_dict(_config())
    t1 = AsyncUserStateStore.from_components(cfg, db=db, tenant=BoundTenantContext("t1", "acme"))
    admin = AsyncUserStateStore.from_components(
        cfg, db=db, tenant=BoundTenantContext("admin", "acme")
    )
    await t1.add_record("u", "alerts", {"text": "t1"})
    # Admin explicitly targets t1 via a tenant_id filter (explicit-filter-wins).
    q = Query(filters=[Filter("tenant_id", Operator.EQ, "t1")])
    rows = await admin.query("u", "alerts", q)
    assert [r.get_value("text") for r in rows] == ["t1"]


# --------------------------------------------------------------------- #
# 8. Events — metadata-only fire + bus fan-out + subscriber-error isolation.
# --------------------------------------------------------------------- #


async def test_write_emits_metadata_only_event() -> None:
    bus = InMemoryEventBus()
    await bus.connect()
    received: list[dict[str, Any]] = []

    async def handler(event: Any) -> None:
        received.append(event.payload)

    await bus.subscribe(SECTION_WRITTEN_TOPIC, handler)

    store = await AsyncUserStateStore.from_config(_config(), event_bus=bus)
    local: list[dict[str, Any]] = []
    store._callbacks.register(SECTION_WRITTEN_TOPIC, local.append)

    def failing(_: dict[str, Any]) -> None:
        raise RuntimeError("subscriber down")

    store._callbacks.register(SECTION_WRITTEN_TOPIC, failing)

    # The write must succeed despite the failing subscriber.
    await store.put_document("u1", "profile", {"ssn": "secret"})

    assert local and local[0]["op"] == "put_document"
    assert local[0]["section"] == "profile"
    # Metadata only — the SENSITIVE payload never appears in the event.
    assert "ssn" not in local[0]
    assert received and received[0]["section"] == "profile" and "ssn" not in received[0]
    await store.close()


# --------------------------------------------------------------------- #
# 10. user_id opacity — the §1.3 filter-based-scoping guard.
# --------------------------------------------------------------------- #


async def test_user_id_opacity() -> None:
    """A slash/scheme-bearing opaque user id round-trips and never leaks."""
    opaque = "https://issuer.example/subject#42"
    other = "https://issuer.example/subject#99"
    store = await _make_async()
    try:
        await store.put_document(opaque, "preferences", {"theme": "dark"})
        await store.add_record(opaque, "alerts", {"text": "mine"})
        await store.add_record(other, "alerts", {"text": "theirs"})

        assert (await store.get_document(opaque, "preferences")).get_value("theme") == "dark"
        rows = await store.query(opaque, "alerts")
        assert [r.get_value("text") for r in rows] == ["mine"]
        # The two opaque ids derive distinct document ids (no collision).
        assert store._doc_id(opaque, "preferences") != store._doc_id(other, "preferences")
    finally:
        await store.close()


async def test_cross_scope_update_rejected() -> None:
    store = await _make_async()
    try:
        rid = await store.add_record("u1", "alerts", {"text": "u1"})
        # Another user cannot hijack the record by id.
        with pytest.raises(ValueError):
            await store.update_record("u2", "alerts", rid, {"text": "stolen"})
        with pytest.raises(ValueError):
            await store.delete_record("u2", "alerts", rid)
    finally:
        await store.close()


async def test_clear_erases_all_sections() -> None:
    store = await _make_async()
    try:
        await store.put_document("u1", "preferences", {"theme": "dark"})
        await store.put_document("u1", "profile", {"ssn": "secret"})
        await store.add_record("u1", "alerts", {"text": "a"})
        await store.add_record("u1", "alerts", {"text": "b"})
        await store.add_record("u2", "alerts", {"text": "keep"})

        assert await store.clear("u1") == 4
        assert await store.get_document("u1", "preferences") is None
        assert await store.query("u1", "alerts") == []
        # A different user is untouched.
        assert len(await store.query("u2", "alerts")) == 1
    finally:
        await store.close()


# --------------------------------------------------------------------- #
# Section-kind validation.
# --------------------------------------------------------------------- #


async def test_unknown_section_raises() -> None:
    store = await _make_async()
    try:
        with pytest.raises(ConfigurationError):
            await store.get_document("u1", "does_not_exist")
    finally:
        await store.close()


async def test_wrong_section_kind_raises() -> None:
    store = await _make_async()
    try:
        with pytest.raises(ValueError):
            await store.get_document("u1", "alerts")  # a collection section
        with pytest.raises(ValueError):
            await store.query("u1", "preferences")  # a document section
    finally:
        await store.close()


# --------------------------------------------------------------------- #
# Parity guards.
# --------------------------------------------------------------------- #


def test_config_consumer_parity() -> None:
    assert_structured_config_consumer(AsyncUserStateStore)
    assert_structured_config_consumer(UserStateStore)


def test_config_roundtrip() -> None:
    cfg = UserStateStoreConfig.from_dict(_config())
    assert_structured_config_roundtrip(cfg)
    assert_structured_config_roundtrip(
        UserStateSectionSpec(name="p", kind=SectionKind.COLLECTION, sensitivity=Sensitivity.PUBLIC)
    )


# --------------------------------------------------------------------- #
# Config-time section validation (finding #1).
# --------------------------------------------------------------------- #


def test_duplicate_section_name_rejected() -> None:
    """A duplicate section name would silently collapse in the name map — it
    must fail at config-load time, not as a confusing runtime error later.
    """
    with pytest.raises(ConfigurationError):
        UserStateStoreConfig.from_dict(
            {
                "namespace": "acme",
                "sections": [
                    {"name": "dup", "kind": "document"},
                    {"name": "dup", "kind": "collection"},
                ],
            }
        )


def test_empty_section_name_rejected() -> None:
    """A section with no name (the field default) is a config error."""
    with pytest.raises(ConfigurationError):
        UserStateStoreConfig.from_dict({"sections": [{"kind": "document"}]})


def test_empty_sections_tuple_is_allowed() -> None:
    """A store with zero declared sections is inert, not invalid."""
    cfg = UserStateStoreConfig.from_dict({"namespace": "acme", "sections": []})
    assert cfg.sections == ()


# --------------------------------------------------------------------- #
# Sync store + EventBus is rejected at construction (finding #2).
# --------------------------------------------------------------------- #


def test_sync_store_rejects_event_bus() -> None:
    """The sync store cannot serve async EventBus fan-out — it must fail fast
    at construction, not after a write under a running loop.
    """
    with pytest.raises(ConfigurationError):
        UserStateStore.from_config(_config(), event_bus=InMemoryEventBus())


def test_sync_store_rejects_injected_event_bus() -> None:
    """The rejection also covers the from_components construction path."""
    cfg = UserStateStoreConfig.from_dict(_config())

    with pytest.raises(ConfigurationError):
        UserStateStore.from_components(cfg, db=SyncMemoryDatabase(), event_bus=InMemoryEventBus())


def test_sync_store_local_callbacks_still_fire() -> None:
    """Rejecting bus fan-out does not remove in-process sync callbacks."""
    store = UserStateStore.from_config(_config())
    try:
        seen: list[dict[str, Any]] = []
        store._callbacks.register(SECTION_WRITTEN_TOPIC, seen.append)
        store.put_document("u1", "preferences", {"theme": "dark"})
        assert seen and seen[0]["op"] == "put_document"
    finally:
        store.close()


# --------------------------------------------------------------------- #
# Storage-identity payload keys are rejected (finding #4).
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("id_key", ["id", "storage_id", "_id", "record_id"])
async def test_add_record_rejects_identity_key_async(id_key: str) -> None:
    """A payload carrying a storage-identity key is rejected — closing the
    sync/async divergence where the sync backend would key a collection
    ``create`` off a payload ``id`` while the async backend mints a UUID.
    """
    store = await _make_async()
    try:
        with pytest.raises(ValueError):
            await store.add_record("u1", "alerts", {id_key: "x", "text": "y"})
    finally:
        await store.close()


@pytest.mark.parametrize("id_key", ["id", "storage_id", "_id", "record_id"])
def test_add_record_rejects_identity_key_sync(id_key: str) -> None:
    store = UserStateStore.from_config(_config())
    try:
        with pytest.raises(ValueError):
            store.add_record("u1", "alerts", {id_key: "x", "text": "y"})
    finally:
        store.close()


async def test_put_document_rejects_identity_key() -> None:
    store = await _make_async()
    try:
        with pytest.raises(ValueError):
            await store.put_document("u1", "preferences", {"id": "x"})
    finally:
        await store.close()


def test_put_document_rejects_identity_key_sync() -> None:
    store = UserStateStore.from_config(_config())
    try:
        with pytest.raises(ValueError):
            store.put_document("u1", "preferences", {"id": "x"})
    finally:
        store.close()


# --------------------------------------------------------------------- #
# record_version is scope-checked (finding #5).
# --------------------------------------------------------------------- #


async def test_record_version_scoped() -> None:
    """An out-of-scope or missing record id yields None (no existence leak);
    the owning scope still gets a usable CAS token.
    """
    store = await _make_async()
    try:
        rid = await store.add_record("u1", "alerts", {"text": "mine"})
        # Owner reads a real token.
        assert await store.record_version("u1", "alerts", rid) is not None
        # Another user probing the same id learns nothing.
        assert await store.record_version("u2", "alerts", rid) is None
        # A never-seen id is likewise None.
        assert await store.record_version("u1", "alerts", "no-such-id") is None
    finally:
        await store.close()


def test_record_version_scoped_sync() -> None:
    store = UserStateStore.from_config(_config())
    try:
        rid = store.add_record("u1", "alerts", {"text": "mine"})
        assert store.record_version("u1", "alerts", rid) is not None
        assert store.record_version("u2", "alerts", rid) is None
    finally:
        store.close()
