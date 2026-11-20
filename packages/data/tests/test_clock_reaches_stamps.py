"""Every aware timestamp the data package writes is read from the common clock.

Under :func:`dataknobs_common.clock.use_clock` with a
:class:`~dataknobs_common.testing.FixedClock`, the stamps below **equal** the
fixed instant: a memory vector store's ``_created_at`` / ``_updated_at``, and
the per-user state store's ``_written_at`` when no ``components["now"]`` was
injected. (The S3 backends are covered against LocalStack in
``integration/test_s3_clock.py``; an injected ``now`` in ``tests/user``.)
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
from dataknobs_common.clock import use_clock
from dataknobs_common.testing import FixedClock

from dataknobs_data.user import AsyncUserStateStore, UserStateStore
from dataknobs_data.vector.stores.memory import MemoryVectorStore

AT = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
LATER = AT + timedelta(hours=1)
VECTOR = np.array([[1.0, 0.0]], dtype=np.float32)
SECTIONS = {
    "backend": "memory",
    "namespace": "acme",
    "sections": [{"name": "activity", "kind": "collection", "retention_days": 30}],
}


async def _stamps(store: MemoryVectorStore, vector_id: str) -> tuple[str, str]:
    [(_, metadata)] = await store.get_vectors([vector_id], include_timestamps=True)
    assert metadata is not None
    return metadata["_created_at"], metadata["_updated_at"]


async def test_a_memory_vector_store_stamps_by_the_clock() -> None:
    store = MemoryVectorStore({"dimensions": 2})
    clock = FixedClock(AT)
    with use_clock(clock):
        [vector_id] = await store.add_vectors(VECTOR, metadata=[{"k": 1}])
        assert await _stamps(store, vector_id) == (AT.isoformat(), AT.isoformat())
        clock.set(LATER)
        await store.update_metadata([vector_id], [{"k": 2}])
    assert await _stamps(store, vector_id) == (AT.isoformat(), LATER.isoformat())


async def test_a_backward_clock_step_moves_updated_at_backward() -> None:
    """Characterises current behaviour; it does not endorse it.

    The memory store writes ``_updated_at`` as whatever the clock reads, so a
    wall clock corrected backwards between two writes leaves the later write
    with the earlier stamp. Ordering by ``_updated_at`` then misorders them.
    This pins that, deterministically, for whoever changes it.
    """
    store = MemoryVectorStore({"dimensions": 2})
    clock = FixedClock(LATER)
    with use_clock(clock):
        [vector_id] = await store.add_vectors(VECTOR, metadata=[{"k": 1}])
        _, first = await _stamps(store, vector_id)
        clock.set(AT)  # the wall clock is corrected backwards
        await store.update_metadata([vector_id], [{"k": 2}])
        _, second = await _stamps(store, vector_id)
    assert second < first


def test_the_user_state_store_reads_the_ambient_clock_by_default() -> None:
    store = UserStateStore.from_config(dict(SECTIONS))
    try:
        with use_clock(FixedClock(AT)):
            store.add_record("u1", "activity", {"event": "e"})
        [record] = store.query("u1", "activity")
        assert record.get_value("_written_at") == AT.isoformat()
    finally:
        store.close()


async def test_the_async_user_state_store_reads_the_ambient_clock_by_default() -> None:
    store = await AsyncUserStateStore.from_config(dict(SECTIONS))
    try:
        with use_clock(FixedClock(AT)):
            await store.add_record("u1", "activity", {"event": "e"})
        [record] = await store.query("u1", "activity")
        assert record.get_value("_written_at") == AT.isoformat()
    finally:
        await store.close()


def test_an_injected_clock_wins_over_the_ambient_one() -> None:
    store = UserStateStore.from_config(dict(SECTIONS), now=FixedClock(LATER))
    try:
        with use_clock(FixedClock(AT)):
            store.add_record("u1", "activity", {"event": "e"})
        [record] = store.query("u1", "activity")
        assert record.get_value("_written_at") == LATER.isoformat()
    finally:
        store.close()
