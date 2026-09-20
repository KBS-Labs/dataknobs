"""The store connects the database it builds, on both twins.

``close()`` releases the backing database when the store owns it --
``close_if_owned`` / ``close_if_owned_sync``, the line this package draws
everywhere between a handle it opened and one it was handed. Only half of
that line was drawn: the store built its own handle from config and never
connected it, so the pairing was a close with no open.

It went unseen because every existing test names ``backend: "memory"``, and
an in-process store needs no connection -- ``connect()`` on it is a no-op,
so the missing call costs nothing and asserts nothing. Every backend that
*has* a connection failed on first use with ``RuntimeError: Database not
connected``, which is to say the store worked with exactly the one backend
nobody deploys.

A handle handed in through ``from_components`` is untouched here for the
same reason ``close()`` leaves it alone: its owner connects it, and a store
that connected someone else's handle would be reaching past the boundary the
teardown half already respects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any


from dataknobs_common.testing import requires_package

from dataknobs_data.backends.memory import AsyncMemoryDatabase, SyncMemoryDatabase
from dataknobs_data.user.store import AsyncUserStateStore, UserStateStore

if TYPE_CHECKING:
    from pathlib import Path

    from dataknobs_data.records import Record

SECTIONS = [{"name": "notes", "kind": "collection"}]


def _config(tmp_path: Path, name: str) -> dict[str, Any]:
    """A sqlite-backed store: a real backend whose connection is not optional."""
    return {
        "backend": "sqlite",
        "path": str(tmp_path / name),
        "sections": list(SECTIONS),
    }


# --------------------------------------------------------------------------
# The handle the store built
# --------------------------------------------------------------------------


@requires_package("aiosqlite")
async def test_the_async_store_can_use_the_database_it_built(tmp_path: Path) -> None:
    """Round-trips a record through a backend that has a connection to open."""
    store = await AsyncUserStateStore.from_config(_config(tmp_path, "async.db"))
    try:
        record_id = await store.add_record("user-1", "notes", {"text": "hello"})
        rows = await store.query("user-1", "notes")

        assert record_id
        assert [r.get_value("text") for r in rows] == ["hello"]
    finally:
        await store.close()


def test_the_sync_store_can_use_the_database_it_built(tmp_path: Path) -> None:
    """The twin's claim, which the shared docstring already promised."""
    store = UserStateStore.from_config(_config(tmp_path, "sync.db"))
    try:
        record_id = store.add_record("user-1", "notes", {"text": "hello"})
        rows = store.query("user-1", "notes")

        assert record_id
        assert [r.get_value("text") for r in rows] == ["hello"]
    finally:
        store.close()


# --------------------------------------------------------------------------
# The handle the store was handed
# --------------------------------------------------------------------------


async def test_an_injected_async_handle_is_not_connected_by_the_store() -> None:
    """Symmetry with ``close()``: what the store did not open, it does not open.

    Asserted with a real memory database whose ``connect`` records that it was
    called, rather than by inspecting private connection state -- the claim is
    about who makes the call.
    """
    calls: list[str] = []

    class _RecordsConnect(AsyncMemoryDatabase):
        async def connect(self) -> None:
            calls.append("connect")
            await super().connect()

    database = _RecordsConnect()
    await database.connect()
    calls.clear()

    store = AsyncUserStateStore.from_components(config={"sections": list(SECTIONS)}, db=database)
    try:
        await store.add_record("user-1", "notes", {"text": "hello"})
        assert calls == []
    finally:
        await store.close()


def test_an_injected_sync_handle_is_not_connected_by_the_store() -> None:
    """The twin's half of the same boundary."""
    calls: list[str] = []

    class _RecordsConnect(SyncMemoryDatabase):
        def connect(self) -> None:
            calls.append("connect")
            super().connect()

    database = _RecordsConnect()
    database.connect()
    calls.clear()

    store = UserStateStore.from_components(config={"sections": list(SECTIONS)}, db=database)
    try:
        store.add_record("user-1", "notes", {"text": "hello"})
        assert calls == []
    finally:
        store.close()


# --------------------------------------------------------------------------
# The handle the store was handed through the config door
# --------------------------------------------------------------------------
#
# ``from_components`` short-circuits the build entirely (``_prebuilt``), so the
# two tests above exercise a path that never reaches the question "was a
# database injected?". The config door is where that question is asked, and
# where both wrong answers are silent: a store that builds its own handle
# despite the injection answers every call correctly while writing into a
# database nobody else holds, and a store that adopts the handle but claims
# ownership of it closes a collaborator its caller is still using.


async def test_an_async_handle_injected_with_a_config_is_used_and_not_owned() -> None:
    """One handle, supplied through the config door: it is used, and left open."""
    calls: list[str] = []

    class _RecordsCalls(AsyncMemoryDatabase):
        async def create(self, record: Record) -> str:
            calls.append("create")
            return await super().create(record)

        async def close(self) -> None:
            calls.append("close")
            await super().close()

    database = _RecordsCalls()
    store = await AsyncUserStateStore.from_config(
        {"backend": "memory", "sections": list(SECTIONS)}, db=database
    )
    await store.add_record("user-1", "notes", {"text": "hello"})
    await store.close()

    # The write went through this handle, and ``close()`` left it alone.
    assert calls == ["create"]


def test_a_sync_handle_injected_with_a_config_is_used_and_not_owned() -> None:
    """The twin's half of the same door."""
    calls: list[str] = []

    class _RecordsCalls(SyncMemoryDatabase):
        def create(self, record: Record) -> str:
            calls.append("create")
            return super().create(record)

        def close(self) -> None:
            calls.append("close")
            super().close()

    database = _RecordsCalls()
    store = UserStateStore.from_config(
        {"backend": "memory", "sections": list(SECTIONS)}, db=database
    )
    store.add_record("user-1", "notes", {"text": "hello"})
    store.close()

    assert calls == ["create"]
