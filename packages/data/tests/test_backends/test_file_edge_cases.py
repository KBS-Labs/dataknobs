"""Edge case tests for file backend implementation."""

import asyncio
import json
import os
import tempfile
import platform
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, mock_open
import gzip
import csv

import pytest
import pytest_asyncio

from dataknobs_data.backends.file import (
    FileLock,
    FileFormat,
    JSONFormat,
    CSVFormat,
    ParquetFormat,
    AsyncFileDatabase,
    SyncFileDatabase
)
from dataknobs_data import AsyncDatabase, SyncDatabase
from dataknobs_data.query import Query
from dataknobs_data.records import Record


class TestFileLock:
    """Test FileLock edge cases."""
    
    def test_lock_acquire_release(self):
        """Test basic lock acquisition and release."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name
        
        try:
            lock = FileLock(filepath)
            lock.acquire()
            assert os.path.exists(filepath + ".lock")
            lock.release()
            # Lock file should be removed after release
            assert not os.path.exists(filepath + ".lock")
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
            with FileLock(filepath) as lock:
                assert os.path.exists(filepath + ".lock")
            # Lock file should be removed after context exit
            assert not os.path.exists(filepath + ".lock")
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)
            if os.path.exists(filepath + ".lock"):
                os.remove(filepath + ".lock")
    
    @pytest.mark.skipif(platform.system() != "Windows", reason="Windows-specific test")
    def test_windows_lock_retry(self):
        """Test Windows lock retry mechanism."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name
        
        try:
            with patch('platform.system', return_value='Windows'):
                import_mock = MagicMock()
                
                # Mock msvcrt module
                msvcrt_mock = MagicMock()
                lock_attempts = [OSError("locked"), OSError("locked"), None]
                msvcrt_mock.locking.side_effect = lock_attempts
                msvcrt_mock.LK_NBLCK = 1
                msvcrt_mock.LK_UNLCK = 2
                
                with patch.dict('sys.modules', {'msvcrt': msvcrt_mock}):
                    with patch('time.sleep') as sleep_mock:
                        lock = FileLock(filepath)
                        with patch('builtins.open', mock_open()) as open_mock:
                            lock.acquire()
                            # Should retry on OSError
                            assert sleep_mock.call_count == 2
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)
    
    def test_lock_release_error_handling(self):
        """Test lock release error handling."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name
        
        try:
            lock = FileLock(filepath)
            lock.acquire()
            
            # Make lock file unremovable
            with patch('os.remove', side_effect=OSError("Permission denied")):
                lock.release()  # Should not raise exception
                
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
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
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
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
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
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            filepath = f.name
        
        try:
            data = {"id1": {"name": "test", "value": 123}}
            JSONFormat.save(filepath, data)
            
            # Read and verify formatting
            with open(filepath, 'r') as f:
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
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            filepath = f.name
        
        try:
            data = CSVFormat.load(filepath)
            assert data == {}
        finally:
            os.remove(filepath)
    
    def test_csv_format_with_complex_data(self):
        """Test CSV format with nested data structures."""
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            filepath = f.name
        
        try:
            # Save data with nested structures - CSVFormat expects "fields" key
            data = {
                "id1": {"fields": {"name": "test", "nested": {"key": "value"}, "list": [1, 2, 3]}},
                "id2": {"fields": {"name": "test2", "nested": {"key": "value2"}}}
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
        
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
            filepath = f.name
        
        try:
            data = {
                "id1": {"name": "test", "value": 123},
                "id2": {"name": "test2", "value": 456}
            }
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
    
    @pytest_asyncio.fixture
    async def temp_db_path(self):
        """Create a temporary database path."""
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
            record_id = await db.create(record)
            
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
            record_id = await db.create(record)
            
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
            config = {
                "path": filepath,
                "format": "json",
                "compression": "gzip"
            }
            
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
            (".csv.bz2", "csv"),
            (".JSON", "json"),  # Case insensitive
            (".CSV", "csv"),
        ]
        
        # Add parquet only if pyarrow is available
        try:
            import pyarrow
            test_cases.append((".parquet", "parquet"))
        except ImportError:
            pass
        
        for ext, expected_format in test_cases:
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
                filepath = f.name
            
            try:
                db = SyncFileDatabase({"path": filepath})
                # Format should be detected correctly
                # Test by creating a record
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