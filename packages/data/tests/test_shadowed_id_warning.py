"""Write-time signal when a record is stored under the reserved storage-key name.

A top-level data field named ``id`` is shadowed by the record's storage key on
every backend: a ``Filter``/``SortSpec`` on ``id`` resolves to the storage key,
never to the stored value, so the value is silently unreachable by query. The
base ``AsyncDatabase``/``SyncDatabase`` classes carry a pre-write inspection
point — installed on every write verb of every backend at class-definition time
— that emits a one-time signal (DEBUG by default, WARNING when
``DK_WARN_SHADOWED_ID=true``) so a consumer investigating an empty result can
see the cause.

These tests are the reproduce-first anchor: the uniform-coverage matrix fails
against any backend whose write body bypasses the inspection (e.g. a bulk
``upsert_batch``/``update_batch`` that never calls the per-record verb, or a
``create``/``update`` that inlines its own id logic) and passes only when the
seam covers every record-persisting verb -- single and bulk -- on every
backend. The structural guard is a second, source-text-independent check that a
new backend cannot silently miss the seam.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import threading

import pytest

from dataknobs_data import Record
from dataknobs_data.backends.duckdb import AsyncDuckDBDatabase, SyncDuckDBDatabase
from dataknobs_data.backends.file import AsyncFileDatabase, SyncFileDatabase
from dataknobs_data.backends.memory import AsyncMemoryDatabase, SyncMemoryDatabase
from dataknobs_data.backends.sqlite import SyncSQLiteDatabase
from dataknobs_data.backends.sqlite_async import AsyncSQLiteDatabase
from dataknobs_data.database import (
    AsyncDatabase,
    SyncDatabase,
    _reset_shadowed_id_warning_state,
    _WARN_SHADOWED_ID_ENV,
    _WRITE_METHODS,
)

_LOGGER_NAME = "dataknobs_data.database"
_SIGNAL_MARKER = "reserved for the record's storage key"
_WRITE_VERBS = (
    "create",
    "upsert",
    "update",
    "create_batch",
    "upsert_batch",
    "update_batch",
)


# ---- backend builders (real in-process backends, no mocks) ------------------

_SYNC_BUILDERS = {
    "memory": lambda _tmp: SyncMemoryDatabase(),
    "sqlite": lambda _tmp: SyncSQLiteDatabase({"path": ":memory:"}),
    "duckdb": lambda _tmp: SyncDuckDBDatabase({"path": ":memory:", "table": "records"}),
    "file": lambda tmp: SyncFileDatabase({"path": str(tmp / "records.json")}),
}

_ASYNC_BUILDERS = {
    "memory": lambda _tmp: AsyncMemoryDatabase(),
    "sqlite": lambda _tmp: AsyncSQLiteDatabase({"path": ":memory:"}),
    "duckdb": lambda _tmp: AsyncDuckDBDatabase({"path": ":memory:", "table": "records"}),
    "file": lambda tmp: AsyncFileDatabase({"path": str(tmp / "records.json")}),
}


def _shadowed_record(storage_id: str = "sk-shadow") -> Record:
    """A record whose data carries the reserved ``id`` field, stored elsewhere.

    ``data["id"]`` (``"shadow-value"``) differs from the storage key
    (``storage_id``), so the value is genuinely unreachable by an ``id`` query —
    the exact footgun the signal warns about.
    """
    record = Record({"id": "shadow-value", "name": "x"})
    record.storage_id = storage_id
    return record


def _plain_record(storage_id: str = "sk-plain") -> Record:
    """A record with no reserved-name field — the entity id is properly named."""
    record = Record({"entity_id": "e1", "name": "y"})
    record.storage_id = storage_id
    return record


def _signals(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if _SIGNAL_MARKER in r.getMessage()]


def _write_sync(db: SyncDatabase, verb: str, records: list[Record]) -> None:
    if verb == "create":
        db.create(records[0])
    elif verb == "upsert":
        db.upsert(records[0])
    elif verb == "update":
        # The inspection fires pre-write, so a not-yet-persisted id is fine —
        # what matters is that the update verb routes through the seam.
        db.update(records[0].storage_id, records[0])
    elif verb == "create_batch":
        db.create_batch(records)
    elif verb == "upsert_batch":
        db.upsert_batch(records)
    else:  # update_batch
        db.update_batch([(r.storage_id, r) for r in records])


async def _write_async(db: AsyncDatabase, verb: str, records: list[Record]) -> None:
    if verb == "create":
        await db.create(records[0])
    elif verb == "upsert":
        await db.upsert(records[0])
    elif verb == "update":
        # The inspection fires pre-write, so a not-yet-persisted id is fine —
        # what matters is that the update verb routes through the seam.
        await db.update(records[0].storage_id, records[0])
    elif verb == "create_batch":
        await db.create_batch(records)
    elif verb == "upsert_batch":
        await db.upsert_batch(records)
    else:  # update_batch
        await db.update_batch([(r.storage_id, r) for r in records])


@pytest.fixture(autouse=True)
def _reset_latch() -> None:
    """Isolate the one-time-per-process latch between tests."""
    _reset_shadowed_id_warning_state()
    yield
    _reset_shadowed_id_warning_state()


# ---- uniform coverage (reproduce-first anchor) ------------------------------


@pytest.mark.parametrize("backend", list(_SYNC_BUILDERS))
@pytest.mark.parametrize("verb", _WRITE_VERBS)
def test_sync_write_signals_shadowed_id(
    backend: str, verb: str, tmp_path, caplog: pytest.LogCaptureFixture
) -> None:
    """Every sync backend signals a shadowed ``id`` on every write verb."""
    caplog.set_level(logging.DEBUG, logger=_LOGGER_NAME)
    db = _SYNC_BUILDERS[backend](tmp_path)
    db.connect()
    try:
        _write_sync(db, verb, [_shadowed_record()])
        assert len(_signals(caplog)) == 1, (
            f"{backend}.{verb} did not emit the shadowed-id signal exactly once"
        )
    finally:
        db.close()


@pytest.mark.parametrize("backend", list(_ASYNC_BUILDERS))
@pytest.mark.parametrize("verb", _WRITE_VERBS)
async def test_async_write_signals_shadowed_id(
    backend: str, verb: str, tmp_path, caplog: pytest.LogCaptureFixture
) -> None:
    """Every async backend signals a shadowed ``id`` on every write verb."""
    caplog.set_level(logging.DEBUG, logger=_LOGGER_NAME)
    db = _ASYNC_BUILDERS[backend](tmp_path)
    await db.connect()
    try:
        await _write_async(db, verb, [_shadowed_record()])
        assert len(_signals(caplog)) == 1, (
            f"{backend}.{verb} did not emit the shadowed-id signal exactly once"
        )
    finally:
        await db.close()


# ---- level, promotion, fail-closed ------------------------------------------


def test_signal_is_debug_by_default(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """With no env override, the signal is DEBUG (silent under normal config)."""
    monkeypatch.delenv(_WARN_SHADOWED_ID_ENV, raising=False)
    caplog.set_level(logging.DEBUG, logger=_LOGGER_NAME)
    SyncMemoryDatabase().create(_shadowed_record())
    signals = _signals(caplog)
    assert len(signals) == 1
    assert signals[0].levelno == logging.DEBUG


def test_env_flag_promotes_to_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """``DK_WARN_SHADOWED_ID=true`` promotes the signal to WARNING."""
    monkeypatch.setenv(_WARN_SHADOWED_ID_ENV, "true")
    caplog.set_level(logging.DEBUG, logger=_LOGGER_NAME)
    SyncMemoryDatabase().create(_shadowed_record())
    signals = _signals(caplog)
    assert len(signals) == 1
    assert signals[0].levelno == logging.WARNING


def test_env_flag_is_case_insensitive_true(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Only the value ``true`` promotes, but casing does not matter."""
    monkeypatch.setenv(_WARN_SHADOWED_ID_ENV, "TRUE")
    caplog.set_level(logging.DEBUG, logger=_LOGGER_NAME)
    SyncMemoryDatabase().create(_shadowed_record())
    signals = _signals(caplog)
    assert len(signals) == 1
    assert signals[0].levelno == logging.WARNING


@pytest.mark.parametrize("value", ["", "false", "1", "yes", "TRUE ", "0"])
def test_env_flag_fails_closed(
    value: str, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Any value other than exactly ``true`` keeps the signal at DEBUG."""
    monkeypatch.setenv(_WARN_SHADOWED_ID_ENV, value)
    caplog.set_level(logging.DEBUG, logger=_LOGGER_NAME)
    SyncMemoryDatabase().create(_shadowed_record())
    signals = _signals(caplog)
    assert len(signals) == 1
    assert signals[0].levelno == logging.DEBUG


# ---- silence + dedup --------------------------------------------------------


def test_non_shadowed_write_is_silent(caplog: pytest.LogCaptureFixture) -> None:
    """A properly-named entity id (``entity_id``) emits nothing."""
    caplog.set_level(logging.DEBUG, logger=_LOGGER_NAME)
    SyncMemoryDatabase().create(_plain_record())
    assert _signals(caplog) == []


def test_one_time_per_process_across_bulk_write(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A bulk write of many shadowed records emits a single signal line."""
    caplog.set_level(logging.DEBUG, logger=_LOGGER_NAME)
    db = SyncMemoryDatabase()
    records = [_shadowed_record(f"sk-{i}") for i in range(5)]
    db.create_batch(records)
    assert len(_signals(caplog)) == 1


def test_one_time_per_process_across_repeated_writes(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Repeated single writes across a process emit a single signal line."""
    caplog.set_level(logging.DEBUG, logger=_LOGGER_NAME)
    db = SyncMemoryDatabase()
    for i in range(3):
        db.create(_shadowed_record(f"sk-{i}"))
    assert len(_signals(caplog)) == 1


def test_one_time_signal_is_thread_safe(caplog: pytest.LogCaptureFixture) -> None:
    """Concurrent writers still emit exactly one signal.

    The latch's check-then-set is lock-guarded, so "at most once per process"
    holds even when many threads race into the inspection simultaneously. A
    barrier forces maximal contention on the transition. Without the lock this
    can emit more than one line (racily); with it, exactly one.
    """
    caplog.set_level(logging.DEBUG, logger=_LOGGER_NAME)
    thread_count = 32
    barrier = threading.Barrier(thread_count)
    db = SyncMemoryDatabase()

    def worker(index: int) -> None:
        barrier.wait()  # release all threads into the check-then-set at once
        db.create(_shadowed_record(f"sk-{index}"))

    threads = [
        threading.Thread(target=worker, args=(i,)) for i in range(thread_count)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(_signals(caplog)) == 1


# ---- structural coverage guard (source-text-independent) --------------------


def _concrete_backends() -> list[type]:
    """Every importable concrete ``AsyncDatabase``/``SyncDatabase`` subclass."""
    for module in (
        "memory",
        "file",
        "sqlite",
        "sqlite_async",
        "duckdb",
        "postgres",
        "elasticsearch",
        "elasticsearch_async",
        "s3",
        "s3_async",
    ):
        try:
            importlib.import_module(f"dataknobs_data.backends.{module}")
        except Exception:  # optional-dependency backend not installed
            pass
    seen: set[type] = set()

    def walk(cls: type) -> None:
        for sub in cls.__subclasses__():
            if sub not in seen:
                seen.add(sub)
                walk(sub)

    walk(AsyncDatabase)
    walk(SyncDatabase)
    return [cls for cls in seen if not inspect.isabstract(cls)]


def test_every_backend_write_method_carries_the_inspection() -> None:
    """No backend can silently miss the write seam.

    This guard checks the resolved method objects for the ``_write_inspected``
    marker, not the source text — so unlike a grep-based single-source guard it
    cannot be evaded by a backend whose write method carries no matching
    literal. A 15th backend (including a consumer-defined one) is covered
    automatically or this fails.
    """
    backends = _concrete_backends()
    assert backends, "expected concrete backends to be discovered"
    gaps: list[str] = []
    for cls in backends:
        for name in _WRITE_METHODS:
            fn = getattr(cls, name, None)
            if fn is None:
                continue
            if not getattr(fn, "_write_inspected", False):
                gaps.append(f"{cls.__name__}.{name}")
    assert not gaps, "write methods missing the pre-write inspection: " + ", ".join(
        sorted(gaps)
    )


# ---- behavior identity (the seam changed no write behavior) -----------------


def test_seam_preserves_write_behavior() -> None:
    """create/upsert/create_batch/upsert_batch still return ids and store records."""
    db = SyncMemoryDatabase()

    created = db.create(_plain_record("c1"))
    assert created == "c1"
    assert db.read("c1") is not None

    upserted = db.upsert(_plain_record("u1"))
    assert upserted == "u1"
    assert db.read("u1") is not None

    batch_ids = db.create_batch([_plain_record("b1"), _plain_record("b2")])
    assert batch_ids == ["b1", "b2"]
    assert db.read("b1") is not None and db.read("b2") is not None

    up_ids = db.upsert_batch([_plain_record("b1"), _plain_record("ub2")])
    assert up_ids == ["b1", "ub2"]
    assert db.read("ub2") is not None
