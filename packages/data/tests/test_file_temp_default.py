"""Test that FileDatabase uses temporary files by default."""

import os
import tempfile
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