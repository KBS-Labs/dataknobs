"""Behavioral tests for the ``AsyncDatabase`` buffered-transaction capability.

Reproduce-first: written against an ``AsyncDatabase`` that had no transaction
primitive — every test below referenced ``db.transaction`` /
``db.begin_transaction`` / ``db.supports_transactions`` and failed with
``AttributeError`` before the capability landed.

Covers the two guarantees of :class:`dataknobs_data.BufferedTransaction`:
universal rollback (an exception before commit persists nothing on any backend)
and atomic commit on transactional backends (sqlite here), plus the
strict/emulate policy gate and the buffer's own lifecycle invariants.
"""

import pytest

from dataknobs_common import CapabilityNotSupportedError
from dataknobs_common.exceptions import ConfigurationError, OperationError
from dataknobs_common.testing import assert_no_blocking
from dataknobs_data import Record, async_database_factory
from dataknobs_data.backends.sqlite_async import AsyncSQLiteDatabase


async def _memory_db():
    db = async_database_factory.create(backend="memory")
    await db.connect()
    return db


async def _sqlite_db():
    db = AsyncSQLiteDatabase({"path": ":memory:"})
    await db.connect()
    return db


# ---- capability flag ----------------------------------------------------


async def test_memory_does_not_support_transactions():
    db = await _memory_db()
    try:
        assert db.supports_transactions() is False
    finally:
        await db.close()


async def test_sqlite_supports_transactions():
    db = await _sqlite_db()
    try:
        assert db.supports_transactions() is True
    finally:
        await db.close()


# ---- policy gate --------------------------------------------------------


async def test_strict_policy_raises_on_non_transactional_backend():
    db = await _memory_db()
    try:
        with pytest.raises(CapabilityNotSupportedError):
            await db.begin_transaction(policy="strict")
        # And via the context-manager form: the raise happens on entry.
        with pytest.raises(CapabilityNotSupportedError):
            async with db.transaction(policy="strict"):
                pass
    finally:
        await db.close()


async def test_unknown_policy_raises_configuration_error():
    db = await _memory_db()
    try:
        with pytest.raises(ConfigurationError):
            await db.begin_transaction(policy="bogus")
    finally:
        await db.close()


async def test_strict_policy_succeeds_on_transactional_backend():
    db = await _sqlite_db()
    try:
        tx = await db.begin_transaction(policy="strict")
        assert tx.is_atomic is True
        await tx.rollback()
    finally:
        await db.close()


# ---- universal rollback (holds on every backend) ------------------------


async def test_emulate_commit_persists_on_memory():
    db = await _memory_db()
    try:
        async with db.transaction(policy="emulate") as tx:
            await tx.create(Record({"name": "a"}))
            await tx.create(Record({"name": "b"}))
            # Buffered: nothing written until the block exits cleanly.
            assert await db.count() == 0
        assert await db.count() == 2
    finally:
        await db.close()


async def test_exception_in_block_persists_nothing_on_memory():
    db = await _memory_db()
    try:
        with pytest.raises(RuntimeError, match="boom"):
            async with db.transaction(policy="emulate") as tx:
                await tx.create(Record({"name": "a"}))
                raise RuntimeError("boom")
        assert await db.count() == 0
    finally:
        await db.close()


async def test_explicit_rollback_persists_nothing():
    db = await _memory_db()
    try:
        tx = await db.begin_transaction(policy="emulate")
        await tx.create(Record({"name": "a"}))
        await tx.rollback()
        assert await db.count() == 0
    finally:
        await db.close()


# ---- atomic commit on a transactional backend ---------------------------


async def test_sqlite_commit_persists_without_blocking():
    db = await _sqlite_db()
    try:
        # Wrap the staged-write commit in the blocking detector: the sqlite
        # path must reach the backend via aiosqlite (async transport), never a
        # blocking syscall on the loop.
        with assert_no_blocking():
            async with db.transaction() as tx:  # default policy="strict"
                await tx.create(Record({"name": "a"}))
                await tx.create(Record({"name": "b"}))
                await tx.create(Record({"name": "c"}))
        assert await db.count() == 3
    finally:
        await db.close()


async def test_sqlite_exception_rolls_back():
    db = await _sqlite_db()
    try:
        with pytest.raises(ValueError, match="nope"):
            async with db.transaction() as tx:
                await tx.create(Record({"name": "a"}))
                raise ValueError("nope")
        assert await db.count() == 0
    finally:
        await db.close()


# ---- is_atomic reflects the staged-op composition ------------------------
#
# Reproduce-first for the over-claim: ``is_atomic`` previously returned the
# backend capability unconditionally, so a mixed-operation or upsert buffer on
# sqlite reported ``True`` while its commit was, in fact, a sequence of
# independent batches that can partially persist. These pin the honest boundary.


async def test_is_atomic_reflects_op_composition_on_transactional_backend():
    db = await _sqlite_db()
    try:
        tx = await db.begin_transaction()  # strict ok on sqlite
        assert tx.is_atomic is True  # empty buffer: trivially atomic
        await tx.create(Record({"name": "a"}))
        await tx.create(Record({"name": "b"}))
        assert tx.is_atomic is True  # all creates → one coalesced create_batch
        await tx.delete("x")
        assert tx.is_atomic is False  # create + delete → two independent batches
        await tx.rollback()

        tx2 = await db.begin_transaction()
        await tx2.upsert("id", Record({"name": "u"}))
        assert tx2.is_atomic is False  # any upsert → applied row-by-row
        await tx2.rollback()
    finally:
        await db.close()


async def test_is_atomic_false_on_non_transactional_backend():
    db = await _memory_db()
    try:
        tx = await db.begin_transaction(policy="emulate")
        await tx.create(Record({"name": "a"}))
        assert tx.is_atomic is False  # memory never wraps the flush in a txn
        await tx.rollback()
    finally:
        await db.close()


class _MidFlushDeleteFailure(AsyncSQLiteDatabase):
    """Real sqlite backend subclass that injects a mid-flush ``delete_batch`` failure.

    A real ``AsyncSQLiteDatabase`` (not a mock) with only ``delete_batch``
    overridden to raise, so a commit can fail *between* batches — after an earlier
    ``create_batch`` has already committed — proving a mixed buffer is not
    all-or-nothing.
    """

    async def delete_batch(self, ids):  # type: ignore[override]
        raise RuntimeError("delete batch failed mid-flush")


async def test_mixed_buffer_partially_persists_on_midflush_failure():
    db = _MidFlushDeleteFailure({"path": ":memory:"})
    await db.connect()
    try:
        tx = await db.begin_transaction()  # strict; sqlite supports it
        await tx.create(Record({"name": "early"}))  # batch 1: create_batch
        await tx.delete("whatever")  # batch 2: delete_batch → raises
        await tx.create(Record({"name": "late"}))  # batch 3: never reached
        # A mixed create+delete buffer correctly reports non-atomic...
        assert tx.is_atomic is False
        # ...and the partial persistence it warns about is real: the first
        # create_batch commits before the delete_batch fails.
        with pytest.raises(RuntimeError, match="mid-flush"):
            await tx.commit()
        assert await db.count() == 1  # "early" persisted; "late" never flushed
    finally:
        await db.close()


# ---- mixed operations + buffer lifecycle --------------------------------


async def test_mixed_create_upsert_delete_flush():
    db = await _memory_db()
    try:
        seed_id = await db.create(Record({"name": "seed"}))
        async with db.transaction(policy="emulate") as tx:
            await tx.create(Record({"name": "fresh"}))
            await tx.upsert(seed_id, Record({"name": "seed-updated"}))
            doomed_id = await db.create(Record({"name": "doomed"}))
            await tx.delete(doomed_id)
        # seed updated, one fresh created, doomed deleted: count back to 2
        # (seed + fresh), the doomed row created mid-block then removed by the
        # staged delete on commit.
        updated = await db.read(seed_id)
        assert updated is not None
        assert updated.get_value("name") == "seed-updated"
        # Count: seed (updated) + fresh = 2 (doomed created then deleted).
        assert await db.count() == 2
        assert await db.read(doomed_id) is None
    finally:
        await db.close()


async def test_create_batch_staging_flushes_all():
    db = await _memory_db()
    try:
        async with db.transaction(policy="emulate") as tx:
            await tx.create_batch(
                [Record({"i": 1}), Record({"i": 2}), Record({"i": 3})]
            )
        assert await db.count() == 3
    finally:
        await db.close()


async def test_staging_after_commit_raises():
    db = await _memory_db()
    try:
        tx = await db.begin_transaction(policy="emulate")
        await tx.create(Record({"name": "a"}))
        await tx.commit()
        with pytest.raises(OperationError):
            await tx.create(Record({"name": "b"}))
    finally:
        await db.close()


async def test_double_commit_is_noop():
    db = await _memory_db()
    try:
        tx = await db.begin_transaction(policy="emulate")
        await tx.create(Record({"name": "a"}))
        first = await tx.commit()
        second = await tx.commit()
        assert first == {"affected_rows": 1}
        assert second == {"affected_rows": 0}
        assert await db.count() == 1
    finally:
        await db.close()
