"""Test that FileDatabase uses temporary files by default."""

import asyncio
import os
import tempfile
import threading

import pytest

from dataknobs_data.backends.file import AsyncFileDatabase, SyncFileDatabase
from dataknobs_data.records import Record


class TestFileDatabaseTempDefault:
    """Test that FileDatabase uses temp files when no path is specified."""

    def test_sync_file_database_uses_temp_file_by_default(self):
        """Test SyncFileDatabase creates temp file when no path specified."""
        # Create database without specifying path
        db = SyncFileDatabase({})

        # Should have created a temp file
        assert db.filepath is not None
        assert db._is_temp_file is True
        assert "dataknobs_sync_db_" in db.filepath
        assert db.filepath.startswith(tempfile.gettempdir())

        # Should work normally
        record = Record({"test": "value"})
        record_id = db.create(record)
        retrieved = db.read(record_id)
        assert retrieved is not None
        assert retrieved.get_value("test") == "value"

        # File should exist
        assert os.path.exists(db.filepath)

        # Close should clean up
        db.close()
        assert not os.path.exists(db.filepath)

    def test_sync_file_database_respects_explicit_path(self):
        """Test SyncFileDatabase uses explicit path when provided."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            db = SyncFileDatabase({"path": tmp_path})

            # Should use the provided path
            assert db.filepath == tmp_path
            assert db._is_temp_file is False

            # Should work normally
            record = Record({"test": "value"})
            record_id = db.create(record)
            retrieved = db.read(record_id)
            assert retrieved is not None

            # Close should NOT delete the file (not a temp file)
            db.close()
            assert os.path.exists(tmp_path)
        finally:
            # Manual cleanup
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    @pytest.mark.asyncio
    async def test_async_file_database_uses_temp_file_by_default(self):
        """Test AsyncFileDatabase creates temp file when no path specified."""
        # Create database without specifying path
        db = AsyncFileDatabase({})

        # Should have created a temp file
        assert db.filepath is not None
        assert db._is_temp_file is True
        assert "dataknobs_async_db_" in db.filepath
        assert db.filepath.startswith(tempfile.gettempdir())

        # Should work normally
        record = Record({"test": "value"})
        record_id = await db.create(record)
        retrieved = await db.read(record_id)
        assert retrieved is not None
        assert retrieved.get_value("test") == "value"

        # File should exist
        assert os.path.exists(db.filepath)

        # Close should clean up
        await db.close()
        assert not os.path.exists(db.filepath)

    @pytest.mark.asyncio
    async def test_async_file_database_respects_explicit_path(self):
        """Test AsyncFileDatabase uses explicit path when provided."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            db = AsyncFileDatabase({"path": tmp_path})

            # Should use the provided path
            assert db.filepath == tmp_path
            assert db._is_temp_file is False

            # Should work normally
            record = Record({"test": "value"})
            record_id = await db.create(record)
            retrieved = await db.read(record_id)
            assert retrieved is not None

            # Close should NOT delete the file (not a temp file)
            await db.close()
            assert os.path.exists(tmp_path)
        finally:
            # Manual cleanup
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


class TestTempCleanupWaitsForInFlightWork:
    """``close()`` must not unlink the lockfile out from under an operation.

    Removing ``<path>.lock`` is the defect ``FileLock`` was fixed to stop
    doing — a waiter handed the lock ends up holding a nameless inode
    while the next acquirer creates and locks a fresh one. Cleanup here
    is still correct, because the path belongs to exactly one instance,
    but only once it is serialized against that instance's own work: the
    cleanup ran outside the instance lock, so a ``close()`` concurrent
    with an in-flight write removed the lockfile that write was holding.
    """

    @pytest.mark.asyncio
    async def test_async_close_waits_for_the_instance_lock(self):
        """A close racing an in-flight operation waits instead of deleting."""
        db = AsyncFileDatabase({})
        await db.create(Record({"test": "value"}))

        # Holding ``_lock`` is what an in-flight operation does; every
        # public method takes it. Pre-fix ``close()`` ignored it and went
        # straight to the unlink.
        async with db._lock:
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(db.close(), timeout=0.5)
            assert os.path.exists(db.filepath), "close() removed the file mid-operation"

        # Released: the same close now completes and cleans up.
        await db.close()
        assert not os.path.exists(db.filepath)
        assert not os.path.exists(db.filepath + ".lock")

    def test_sync_close_waits_for_the_instance_lock(self):
        """The sync sibling holds its ``RLock`` for the same reason."""
        db = SyncFileDatabase({})
        db.create(Record({"test": "value"}))
        filepath = db.filepath

        holding = threading.Event()
        may_release = threading.Event()
        closed = threading.Event()

        def hold_the_instance_lock() -> None:
            with db._lock:
                holding.set()
                may_release.wait(timeout=10)

        def close_it() -> None:
            db.close()
            closed.set()

        holder = threading.Thread(target=hold_the_instance_lock)
        holder.start()
        assert holding.wait(timeout=10)

        closer = threading.Thread(target=close_it)
        closer.start()
        assert not closed.wait(timeout=0.5), "close() cleaned up while the lock was held"
        assert os.path.exists(filepath), "close() removed the file mid-operation"

        may_release.set()
        holder.join(timeout=10)
        closer.join(timeout=10)
        assert closed.is_set()
        assert not os.path.exists(filepath)
        assert not os.path.exists(filepath + ".lock")
