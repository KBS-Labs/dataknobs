"""Capability advertisement across the database backends.

Every backend enforces compare-and-set on ``update``/``upsert`` when an
``expected_version`` token (read via ``get_version``) is supplied, so all of
them advertise ``Capability.CONDITIONAL_WRITE`` through the
``CapabilityContract`` surface — a consumer queries ``supports(...)`` instead
of knowing the backend matrix out-of-band.

Three things are pinned here:

1. **Conformance** — all 14 backend classes are ``CapabilityContract`` hosts
   and advertise ``CONDITIONAL_WRITE`` at the class level (no instance / no
   service connection needed; the guarantee is uniform and declared on the two
   ABCs).
2. **Shadow guard** — no backend instance drops ``CONDITIONAL_WRITE`` from its
   ``instance_capabilities()``. ``CapabilityMixin`` does NOT auto-union across
   the MRO, so a future backend-specific ``SUPPORTED_CAPABILITIES`` that forgot
   to union the ABC set would silently drop the capability; this test fails if
   that ever happens.
3. **Truth of advertisement** — for a representative ABA-safe backend (memory)
   and a content-hash backend (sqlite/file/duckdb), the advertised capability
   is backed by real behavior: a fresh token succeeds, a stale token raises
   ``ConcurrencyError``. The bit is tied to the enforcement, not an aspiration.

No mocks: real in-process backends only. Service backends (Postgres, S3,
Elasticsearch) are covered for enforcement under ``tests/integration/`` behind
their service markers; their class-level advertisement is covered here without
a connection.
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from dataknobs_common import Capability, CapabilityContract, CapabilityMixin
from dataknobs_data import ConcurrencyError, Record
from dataknobs_data.backends.duckdb import AsyncDuckDBDatabase, SyncDuckDBDatabase
from dataknobs_data.backends.elasticsearch import SyncElasticsearchDatabase
from dataknobs_data.backends.elasticsearch_async import AsyncElasticsearchDatabase
from dataknobs_data.backends.file import AsyncFileDatabase, SyncFileDatabase
from dataknobs_data.backends.memory import AsyncMemoryDatabase, SyncMemoryDatabase
from dataknobs_data.backends.postgres import AsyncPostgresDatabase, SyncPostgresDatabase
from dataknobs_data.backends.s3 import SyncS3Database
from dataknobs_data.backends.s3_async import AsyncS3Database
from dataknobs_data.backends.sqlite import SyncSQLiteDatabase
from dataknobs_data.backends.sqlite_async import AsyncSQLiteDatabase

# All 14 concrete backends (7 sync + 7 async). Class-level advertisement needs
# no driver / no connection — the drivers are lazy-imported at construction.
_ALL_BACKENDS: list[type] = [
    SyncMemoryDatabase,
    SyncSQLiteDatabase,
    SyncPostgresDatabase,
    SyncElasticsearchDatabase,
    SyncS3Database,
    SyncDuckDBDatabase,
    SyncFileDatabase,
    AsyncMemoryDatabase,
    AsyncSQLiteDatabase,
    AsyncPostgresDatabase,
    AsyncElasticsearchDatabase,
    AsyncS3Database,
    AsyncDuckDBDatabase,
    AsyncFileDatabase,
]


@pytest.mark.parametrize("backend_cls", _ALL_BACKENDS, ids=lambda c: c.__name__)
def test_backend_class_advertises_conditional_write(backend_cls: type) -> None:
    """Every backend class advertises CONDITIONAL_WRITE without an instance."""
    assert Capability.CONDITIONAL_WRITE in backend_cls.supported_capabilities()


@pytest.mark.parametrize("backend_cls", _ALL_BACKENDS, ids=lambda c: c.__name__)
def test_backend_class_is_capability_contract_host(backend_cls: type) -> None:
    """Class-level capability-contract host guarantee.

    ``issubclass`` against ``CapabilityContract`` is unavailable — it is a
    ``runtime_checkable`` Protocol with a data member, and Python forbids
    ``issubclass`` on such Protocols. So the class-level guarantee is pinned two
    ways: the backend inherits the contract implementation
    (``CapabilityMixin``), and the three contract methods resolve on the class.
    Runtime-checkable ``isinstance`` conformance is asserted on a live instance
    in ``test_instance_advertises_and_is_contract``.
    """
    assert issubclass(backend_cls, CapabilityMixin)
    for name in ("supported_capabilities", "instance_capabilities", "supports"):
        assert callable(getattr(backend_cls, name, None))


# ---------------------------------------------------------------------------
# Shadow guard + truth-of-advertisement on real in-process instances.
# ---------------------------------------------------------------------------
@pytest.fixture(params=["memory", "file", "sqlite", "duckdb"])
def sync_db(request: pytest.FixtureRequest) -> Iterator[object]:
    """A connected in-process sync backend, one per realized family."""
    kind = request.param
    with tempfile.TemporaryDirectory() as d:
        db: object
        if kind == "memory":
            db = SyncMemoryDatabase()
        elif kind == "file":
            db = SyncFileDatabase({"path": str(Path(d) / "records.json")})
        elif kind == "sqlite":
            db = SyncSQLiteDatabase({"path": str(Path(d) / "records.db")})
            db.connect()
        else:
            db = SyncDuckDBDatabase(
                {"path": str(Path(d) / "records.duckdb"), "table": "records"}
            )
            db.connect()
        try:
            yield db
        finally:
            close = getattr(db, "close", None)
            if callable(close):
                close()


def test_instance_advertises_and_is_contract(sync_db: object) -> None:
    """A live instance is a CapabilityContract and does not shadow the ABC set.

    Guards the MRO no-auto-union caveat: if a backend ever declares its own
    ``SUPPORTED_CAPABILITIES`` without unioning the ABC's, ``CONDITIONAL_WRITE``
    would silently drop from ``instance_capabilities()`` and this fails.
    """
    assert isinstance(sync_db, CapabilityContract)
    assert sync_db.supports(Capability.CONDITIONAL_WRITE)
    assert Capability.CONDITIONAL_WRITE in sync_db.instance_capabilities()


def test_advertised_conditional_write_actually_enforces(sync_db: object) -> None:
    """The advertised capability is backed by real CAS enforcement.

    Ties the advertisement bit to behavior so the docs cannot over-promise: a
    fresh token succeeds and a stale token raises ``ConcurrencyError`` rather
    than last-writer-wins.
    """
    assert sync_db.supports(Capability.CONDITIONAL_WRITE)
    sync_db.create(Record({"v": 0}, id="k"))
    stale = sync_db.get_version("k")
    # A conditional write with the fresh token succeeds and advances it.
    assert sync_db.update("k", Record({"v": 1}, id="k"), expected_version=stale) is True
    # A second writer holding the now-stale token loses.
    with pytest.raises(ConcurrencyError):
        sync_db.update("k", Record({"v": 2}, id="k"), expected_version=stale)
    assert sync_db.read("k").get_value("v") == 1
