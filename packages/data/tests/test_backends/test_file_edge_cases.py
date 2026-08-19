"""Edge case tests for file backend implementation."""

import asyncio
import os
import tempfile
import platform
import threading
from unittest.mock import MagicMock, patch

import pytest

from dataknobs_common.testing import is_package_available
from dataknobs_data.backends.file import (
    FileLock,
    FileFormat,
    JSONFormat,
    CSVFormat,
    ParquetFormat,
    AsyncFileDatabase,
    SyncFileDatabase,
)
from dataknobs_data.records import Record


class TestFileLock:
    """Test FileLock edge cases.

    Imported from ``dataknobs_data.backends.file``, which re-exports it
    from ``dataknobs_common.locks`` — so these also pin that the move
    did not break the import shape callers already had.
    """

    def test_lock_acquire_release(self):
        """Basic acquisition and release; the lockfile survives both."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name

        try:
            lock = FileLock(filepath)
            lock.acquire()
            assert os.path.exists(filepath + ".lock")
            lock.release()
            # Deliberately NOT removed. Unlinking on release hands the
            # lock to a waiter and then lets the next acquire create a
            # fresh inode to lock instead — two holders, no error.
            assert os.path.exists(filepath + ".lock")
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)
            if os.path.exists(filepath + ".lock"):
                os.remove(filepath + ".lock")

    def test_lock_context_manager(self):
        """Test FileLock as context manager."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name

        try:
            with FileLock(filepath):
                assert os.path.exists(filepath + ".lock")
            assert os.path.exists(filepath + ".lock")
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)
            if os.path.exists(filepath + ".lock"):
                os.remove(filepath + ".lock")

    @pytest.mark.skipif(platform.system() != "Windows", reason="Windows-specific test")
    def test_windows_lock_retry(self):
        """Test Windows lock retry mechanism.

        ``msvcrt`` is substituted rather than exercised: its blocking
        variant has no non-blocking probe to script, and the branch only
        runs on Windows, where this test runs for real. The branch is
        selected by ``sys.platform``, which is already ``win32`` here —
        so nothing patches the platform check itself.

        The lockfile is left real. ``acquire`` keys its intra-process
        mutex on the lockfile's inode, so a substituted open would hand
        it a handle with no ``fileno`` to stat; and the retry loop now
        holds one handle across attempts rather than reopening per
        attempt, which is the behaviour worth pinning here.
        """
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name

        try:
            msvcrt_mock = MagicMock()
            lock_attempts = [OSError("locked"), OSError("locked"), None]
            msvcrt_mock.locking.side_effect = lock_attempts
            msvcrt_mock.LK_NBLCK = 1
            msvcrt_mock.LK_UNLCK = 2

            with patch.dict("sys.modules", {"msvcrt": msvcrt_mock}):
                with patch("time.sleep") as sleep_mock:
                    lock = FileLock(filepath)
                    assert lock.acquire() is True
                    # Should retry on OSError
                    assert sleep_mock.call_count == 2
                    lock.release()
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)
            if os.path.exists(filepath + ".lock"):
                os.remove(filepath + ".lock")

    def test_release_without_acquire_is_a_no_op(self):
        """A release that never acquired must not raise or free a mutex.

        The intra-process mutex is released in ``release()``; releasing
        one this instance never took would raise ``RuntimeError`` and,
        worse, hand the section to a thread that is legitimately inside
        it.
        """
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name

        try:
            FileLock(filepath).release()  # no acquire — must not raise

            # And the lock is still takeable afterwards.
            with FileLock(filepath):
                pass
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)
            if os.path.exists(filepath + ".lock"):
                os.remove(filepath + ".lock")

    def test_two_instances_over_one_path_do_not_both_hold(self):
        """Two ``FileLock`` objects on one path exclude each other.

        This is the shape two database instances in one process have,
        and the one POSIX record locks do not cover on their own: their
        owner is the process, so the second acquire used to be granted
        immediately.
        """
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name

        try:
            first = FileLock(filepath)
            second = FileLock(filepath)
            first.acquire()

            inside = threading.Event()

            def take_second():
                second.acquire()
                inside.set()
                second.release()

            waiter = threading.Thread(target=take_second)
            waiter.start()
            try:
                assert not inside.wait(timeout=0.3), "second holder was let in"
            finally:
                first.release()
                waiter.join(timeout=10)
            assert inside.is_set(), "second holder never got the lock"
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)
            if os.path.exists(filepath + ".lock"):
                os.remove(filepath + ".lock")


class TestFileFormats:
    """Test file format handlers edge cases."""

    def test_base_format_not_implemented(self):
        """Test that base FileFormat methods raise NotImplementedError."""
        with pytest.raises(NotImplementedError):
            FileFormat.load("test.json")

        with pytest.raises(NotImplementedError):
            FileFormat.save("test.json", {})

    def test_json_format_empty_file(self):
        """Test JSON format with empty file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            filepath = f.name
            # Create empty file
            f.write("")

        try:
            data = JSONFormat.load(filepath)
            assert data == {}
        finally:
            os.remove(filepath)

    def test_json_format_nonexistent_file(self):
        """Test JSON format with nonexistent file."""
        filepath = "/tmp/nonexistent_test_file.json"
        if os.path.exists(filepath):
            os.remove(filepath)

        data = JSONFormat.load(filepath)
        assert data == {}

    def test_json_format_corrupted_file(self):
        """Test JSON format with corrupted file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            filepath = f.name
            f.write("{invalid json")

        try:
            # JSONFormat.load now returns empty dict on JSONDecodeError
            data = JSONFormat.load(filepath)
            assert data == {}
        finally:
            os.remove(filepath)

    def test_json_format_save_with_indent(self):
        """Test JSON format save with proper indentation."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name

        try:
            data = {"id1": {"name": "test", "value": 123}}
            JSONFormat.save(filepath, data)

            # Read and verify formatting
            with open(filepath) as f:
                content = f.read()
                # Should be indented
                assert "  " in content or "    " in content

            # Verify data integrity
            loaded = JSONFormat.load(filepath)
            assert loaded == data
        finally:
            os.remove(filepath)

    def test_csv_format_empty_file(self):
        """Test CSV format with empty file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            filepath = f.name

        try:
            data = CSVFormat.load(filepath)
            assert data == {}
        finally:
            os.remove(filepath)

    def test_csv_format_with_complex_data(self):
        """Test CSV format with nested data structures."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            filepath = f.name

        try:
            # Save data with nested structures - CSVFormat expects "fields" key
            data = {
                "id1": {"fields": {"name": "test", "nested": {"key": "value"}, "list": [1, 2, 3]}},
                "id2": {"fields": {"name": "test2", "nested": {"key": "value2"}}},
            }
            CSVFormat.save(filepath, data)

            # Load and verify
            loaded = CSVFormat.load(filepath)
            assert "id1" in loaded
            assert "fields" in loaded["id1"]
            assert loaded["id1"]["fields"]["name"] == "test"
            # Complex types are now properly deserialized
            assert loaded["id1"]["fields"]["nested"]["key"] == "value"
            assert loaded["id1"]["fields"]["list"] == [1, 2, 3]
        finally:
            os.remove(filepath)

    def test_parquet_format_basic(self):
        """Test Parquet format basic operations."""
        pytest.importorskip("pyarrow")  # Skip if pyarrow not installed

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            filepath = f.name

        try:
            data = {"id1": {"name": "test", "value": 123}, "id2": {"name": "test2", "value": 456}}
            ParquetFormat.save(filepath, data)

            loaded = ParquetFormat.load(filepath)
            assert "id1" in loaded
            assert loaded["id1"]["name"] == "test"
            assert loaded["id1"]["value"] == 123
        finally:
            os.remove(filepath)

    def test_parquet_format_empty_file(self):
        """Test Parquet format with empty/nonexistent file."""
        filepath = "/tmp/nonexistent_test_file.parquet"
        if os.path.exists(filepath):
            os.remove(filepath)

        data = ParquetFormat.load(filepath)
        assert data == {}


class TestFileDatabaseEdgeCases:
    """Test FileDatabase edge cases."""

    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary database path.

        Sync on purpose — the body awaits nothing, so running it as a
        coroutine only served to put its `mkstemp`/`remove` calls on the
        test's event loop.
        """
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        yield path
        # Cleanup
        if os.path.exists(path):
            os.remove(path)
        lock_file = path + ".lock"
        if os.path.exists(lock_file):
            os.remove(lock_file)

    @pytest.mark.asyncio
    async def test_unsupported_format(self):
        """Test error handling for unsupported file format."""
        with tempfile.NamedTemporaryFile(suffix=".unknown", delete=False) as f:
            filepath = f.name

        try:
            # FileDatabase now defaults to JSON for unknown formats
            db = AsyncFileDatabase({"path": filepath})
            assert db.format == "json"  # Should default to JSON
            await db.close()
        finally:
            os.remove(filepath)

    @pytest.mark.asyncio
    async def test_gzip_json_format(self):
        """Test gzipped JSON format operations."""
        with tempfile.NamedTemporaryFile(suffix=".json.gz", delete=False) as f:
            filepath = f.name

        try:
            db = AsyncFileDatabase({"path": filepath})

            # Create records
            record = Record({"name": "test", "compressed": True})
            record_id = await db.create(record)

            # Verify file is handled properly (JSONFormat handles .gz extension)
            # The file is saved through JSONFormat which handles compression
            raw_data = JSONFormat.load(filepath)
            assert record_id in raw_data

            # Read record
            retrieved = await db.read(record_id)
            assert retrieved.get_value("compressed") is True

            await db.close()
        finally:
            os.remove(filepath)

    @pytest.mark.asyncio
    async def test_bz2_json_format(self):
        """Test bz2 compressed JSON format."""
        with tempfile.NamedTemporaryFile(suffix=".json.bz2", delete=False) as f:
            filepath = f.name

        try:
            db = AsyncFileDatabase({"path": filepath})

            record = Record({"name": "bz2_test"})
            await db.create(record)

            # Verify record was saved (FileDatabase doesn't actually support bz2)
            # The test shows that FileDatabase accepts the path
            # Note: FileDatabase doesn't have built-in bz2 support, it will use JSON format

            await db.close()
        finally:
            os.remove(filepath)

    @pytest.mark.asyncio
    async def test_xz_json_format(self):
        """Test xz/lzma compressed JSON format."""
        with tempfile.NamedTemporaryFile(suffix=".json.xz", delete=False) as f:
            filepath = f.name

        try:
            db = AsyncFileDatabase({"path": filepath})

            record = Record({"name": "xz_test"})
            await db.create(record)

            # Verify record was saved (FileDatabase doesn't actually support xz)
            # The test shows that FileDatabase accepts the path
            # Note: FileDatabase doesn't have built-in xz support, it will use JSON format

            await db.close()
        finally:
            os.remove(filepath)

    @pytest.mark.asyncio
    async def test_from_config_with_compression(self):
        """Test creating database from config with compression."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name

        try:
            config = {"path": filepath, "format": "json", "compression": "gzip"}

            db = AsyncFileDatabase.from_config(config)
            # FileDatabase appends .gz when compression is set
            assert db.filepath == filepath + ".gz"
            assert db.compression == "gzip"

            # Test operations
            record = Record({"configured": True})
            await db.create(record)

            await db.close()
        finally:
            os.remove(filepath)
            # Compressed file might be created
            if os.path.exists(filepath + ".gz"):
                os.remove(filepath + ".gz")

    @pytest.mark.asyncio
    async def test_record_without_id_field(self):
        """Test handling records without explicit ID field."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name

        try:
            db = AsyncFileDatabase({"path": filepath})

            # Create record without ID
            record = Record({"name": "no_id"})
            record_id = await db.create(record)

            # ID should be generated
            assert record_id is not None
            assert len(record_id) > 0

            # Retrieve and verify
            retrieved = await db.read(record_id)
            assert retrieved.get_value("name") == "no_id"

            await db.close()
        finally:
            os.remove(filepath)

    @pytest.mark.asyncio
    async def test_concurrent_writes(self):
        """Test concurrent write operations with file locking."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name

        try:
            db = AsyncFileDatabase({"path": filepath})

            # Create multiple records concurrently
            async def create_record(i):
                record = Record({"index": i, "data": f"record_{i}"})
                return await db.create(record)

            # Run concurrent creates
            tasks = [create_record(i) for i in range(10)]
            record_ids = await asyncio.gather(*tasks)

            # Verify all records were created
            assert len(record_ids) == 10
            assert len(set(record_ids)) == 10  # All IDs should be unique

            # Verify data integrity
            for i, record_id in enumerate(record_ids):
                retrieved = await db.read(record_id)
                assert retrieved.get_value("index") == i

            await db.close()
        finally:
            os.remove(filepath)


class TestSyncFileDatabaseEdgeCases:
    """Test SyncFileDatabase edge cases."""

    def test_thread_safety(self):
        """Test thread-safe operations."""
        import threading

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name

        try:
            db = SyncFileDatabase({"path": filepath})
            results = []

            def create_records(thread_id):
                for i in range(5):
                    record = Record({"thread": thread_id, "index": i})
                    record_id = db.create(record)
                    results.append((thread_id, record_id))

            # Create threads
            threads = []
            for i in range(3):
                t = threading.Thread(target=create_records, args=(i,))
                threads.append(t)
                t.start()

            # Wait for completion
            for t in threads:
                t.join()

            # Verify results
            assert len(results) == 15  # 3 threads * 5 records

            # Verify all records exist
            for thread_id, record_id in results:
                record = db.read(record_id)
                assert record is not None
                assert record.get_value("thread") == thread_id

            db.close()
        finally:
            os.remove(filepath)

    def test_invalid_path_permissions(self):
        """Test handling of invalid path or permission errors."""
        # Try to create database in non-writable location
        invalid_path = "/root/test_db.json"  # Typically not writable

        if not os.access("/root", os.W_OK):
            with pytest.raises((PermissionError, OSError)):
                db = SyncFileDatabase({"path": invalid_path})
                db.create(Record({"test": "data"}))

    def test_format_detection_from_extension(self):
        """Test automatic format detection from file extension."""
        test_cases = [
            (".json", "json"),
            (".csv", "csv"),
            (".json.gz", "json"),
            # Only ".gz" is stripped before the extension is read, so a
            # ".bz2" path never reaches its inner ".csv" and falls through
            # to the json default. Pinned as the behaviour that exists, not
            # as the behaviour intended -- if bz2 gains real support this
            # row is meant to fail and be updated.
            (".csv.bz2", "json"),
            (".JSON", "json"),  # Case insensitive
            (".CSV", "csv"),
        ]

        # Add parquet only if pyarrow is available
        if is_package_available("pyarrow"):
            test_cases.append((".parquet", "parquet"))

        for ext, expected_format in test_cases:
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
                filepath = f.name

            try:
                db = SyncFileDatabase({"path": filepath})
                # The detected format is the subject of this test, so assert
                # it directly rather than inferring it from a round-trip that
                # the json default would also pass.
                assert db.format == expected_format, f"{ext} detected as {db.format}"
                # And the round-trip still has to work under that format.
                record = Record({"test": "format_detection"})
                record_id = db.create(record)
                retrieved = db.read(record_id)
                assert retrieved is not None
                assert retrieved.get_value("test") == "format_detection"
                db.close()
            finally:
                if os.path.exists(filepath):
                    os.remove(filepath)
                lock_file = filepath + ".lock"
                if os.path.exists(lock_file):
                    os.remove(lock_file)
