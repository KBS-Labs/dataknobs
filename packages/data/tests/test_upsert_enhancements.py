"""Tests for enhanced upsert functionality that accepts just a Record."""

import uuid
import pytest
from dataknobs_data.records import Record
from dataknobs_data.backends.memory import AsyncMemoryDatabase, SyncMemoryDatabase


class TestSyncUpsertEnhancements:
    """Test synchronous upsert enhancements."""
    
    def test_upsert_with_id_and_record(self):
        """Test traditional upsert with explicit ID and record."""
        db = SyncMemoryDatabase()
        
        # Create a record
        record = Record({"name": "Test", "value": 42})
        
        # Upsert with explicit ID
        id = "test-id-1"
        result_id = db.upsert(id, record)
        
        assert result_id == id
        
        # Verify the record was stored
        retrieved = db.read(id)
        assert retrieved is not None
        assert retrieved["name"] == "Test"
        assert retrieved["value"] == 42
    
    def test_upsert_with_record_only_no_id(self):
        """Test upsert with just a record that has no ID."""
        db = SyncMemoryDatabase()
        
        # Create a record without an ID
        record = Record({"name": "Test", "value": 42})
        assert record.id is None
        
        # Upsert with just the record
        result_id = db.upsert(record)
        
        # Should generate an ID
        assert result_id is not None
        assert len(result_id) > 0
        
        # Verify the record was stored
        retrieved = db.read(result_id)
        assert retrieved is not None
        assert retrieved["name"] == "Test"
        assert retrieved["value"] == 42
    
    def test_upsert_with_record_with_id_field(self):
        """Test upsert with a record that has an 'id' field."""
        db = SyncMemoryDatabase()
        
        # Create a record with an 'id' field
        record = Record({"id": "my-custom-id", "name": "Test", "value": 42})
        assert record.id == "my-custom-id"
        
        # Upsert with just the record
        result_id = db.upsert(record)
        
        # Should use the record's ID
        assert result_id == "my-custom-id"
        
        # Verify the record was stored
        retrieved = db.read("my-custom-id")
        assert retrieved is not None
        assert retrieved["name"] == "Test"
        assert retrieved["value"] == 42
    
    def test_upsert_with_record_with_storage_id(self):
        """Test upsert with a record that has a storage_id."""
        db = SyncMemoryDatabase()
        
        # Create a record and set storage_id
        record = Record({"name": "Test", "value": 42})
        record.storage_id = "storage-id-123"
        
        # Upsert with just the record
        result_id = db.upsert(record)
        
        # Should use the storage_id
        assert result_id == "storage-id-123"
        
        # Verify the record was stored
        retrieved = db.read("storage-id-123")
        assert retrieved is not None
        assert retrieved["name"] == "Test"
        assert retrieved["value"] == 42
    
    def test_upsert_update_existing(self):
        """Test that upsert updates existing records."""
        db = SyncMemoryDatabase()
        
        # Create initial record
        record1 = Record({"id": "update-test", "name": "Initial", "value": 1})
        db.upsert(record1)
        
        # Update with new values
        record2 = Record({"id": "update-test", "name": "Updated", "value": 2})
        db.upsert(record2)
        
        # Verify the update
        retrieved = db.read("update-test")
        assert retrieved is not None
        assert retrieved["name"] == "Updated"
        assert retrieved["value"] == 2
    
    def test_upsert_error_on_missing_record(self):
        """Test that upsert raises error when ID provided but no record."""
        db = SyncMemoryDatabase()
        
        with pytest.raises(ValueError, match="Record required when ID is provided"):
            db.upsert("some-id", None)
    
    def test_backwards_compatibility(self):
        """Test that old-style upsert calls still work."""
        db = SyncMemoryDatabase()
        
        # Old style: upsert(id, record)
        record = Record({"name": "Backwards", "value": 99})
        result_id = db.upsert("old-style-id", record)
        
        assert result_id == "old-style-id"
        
        # Verify
        retrieved = db.read("old-style-id")
        assert retrieved is not None
        assert retrieved["name"] == "Backwards"


class TestAsyncUpsertEnhancements:
    """Test asynchronous upsert enhancements."""
    
    @pytest.mark.asyncio
    async def test_async_upsert_with_id_and_record(self):
        """Test traditional async upsert with explicit ID and record."""
        db = AsyncMemoryDatabase()
        
        # Create a record
        record = Record({"name": "AsyncTest", "value": 42})
        
        # Upsert with explicit ID
        id = "async-test-id-1"
        result_id = await db.upsert(id, record)
        
        assert result_id == id
        
        # Verify the record was stored
        retrieved = await db.read(id)
        assert retrieved is not None
        assert retrieved["name"] == "AsyncTest"
        assert retrieved["value"] == 42
    
    @pytest.mark.asyncio
    async def test_async_upsert_with_record_only_no_id(self):
        """Test async upsert with just a record that has no ID."""
        db = AsyncMemoryDatabase()
        
        # Create a record without an ID
        record = Record({"name": "AsyncTest", "value": 42})
        assert record.id is None
        
        # Upsert with just the record
        result_id = await db.upsert(record)
        
        # Should generate an ID
        assert result_id is not None
        assert len(result_id) > 0
        
        # Verify the record was stored
        retrieved = await db.read(result_id)
        assert retrieved is not None
        assert retrieved["name"] == "AsyncTest"
        assert retrieved["value"] == 42
    
    @pytest.mark.asyncio
    async def test_async_upsert_with_record_with_id_field(self):
        """Test async upsert with a record that has an 'id' field."""
        db = AsyncMemoryDatabase()
        
        # Create a record with an 'id' field
        record = Record({"id": "async-custom-id", "name": "AsyncTest", "value": 42})
        assert record.id == "async-custom-id"
        
        # Upsert with just the record
        result_id = await db.upsert(record)
        
        # Should use the record's ID
        assert result_id == "async-custom-id"
        
        # Verify the record was stored
        retrieved = await db.read("async-custom-id")
        assert retrieved is not None
        assert retrieved["name"] == "AsyncTest"
        assert retrieved["value"] == 42
    
    @pytest.mark.asyncio
    async def test_async_upsert_with_record_with_storage_id(self):
        """Test async upsert with a record that has a storage_id."""
        db = AsyncMemoryDatabase()
        
        # Create a record and set storage_id
        record = Record({"name": "AsyncTest", "value": 42})
        record.storage_id = "async-storage-id-123"
        
        # Upsert with just the record
        result_id = await db.upsert(record)
        
        # Should use the storage_id
        assert result_id == "async-storage-id-123"
        
        # Verify the record was stored
        retrieved = await db.read("async-storage-id-123")
        assert retrieved is not None
        assert retrieved["name"] == "AsyncTest"
        assert retrieved["value"] == 42
    
    @pytest.mark.asyncio
    async def test_async_upsert_update_existing(self):
        """Test that async upsert updates existing records."""
        db = AsyncMemoryDatabase()
        
        # Create initial record
        record1 = Record({"id": "async-update-test", "name": "Initial", "value": 1})
        await db.upsert(record1)
        
        # Update with new values
        record2 = Record({"id": "async-update-test", "name": "Updated", "value": 2})
        await db.upsert(record2)
        
        # Verify the update
        retrieved = await db.read("async-update-test")
        assert retrieved is not None
        assert retrieved["name"] == "Updated"
        assert retrieved["value"] == 2
    
    @pytest.mark.asyncio
    async def test_async_upsert_error_on_missing_record(self):
        """Test that async upsert raises error when ID provided but no record."""
        db = AsyncMemoryDatabase()
        
        with pytest.raises(ValueError, match="Record required when ID is provided"):
            await db.upsert("some-id", None)
    
    @pytest.mark.asyncio
    async def test_async_backwards_compatibility(self):
        """Test that old-style async upsert calls still work."""
        db = AsyncMemoryDatabase()
        
        # Old style: upsert(id, record)
        record = Record({"name": "AsyncBackwards", "value": 99})
        result_id = await db.upsert("async-old-style-id", record)
        
        assert result_id == "async-old-style-id"
        
        # Verify
        retrieved = await db.read("async-old-style-id")
        assert retrieved is not None
        assert retrieved["name"] == "AsyncBackwards"


class TestUpsertWithDifferentBackends:
    """Test that upsert works consistently across different backends."""
    
    def test_memory_backend_sync(self):
        """Test sync memory backend."""
        from dataknobs_data.backends.memory import SyncMemoryDatabase
        
        db = SyncMemoryDatabase()
        record = Record({"test": "memory"})
        
        # New style
        id1 = db.upsert(record)
        assert id1 is not None
        
        # Old style
        id2 = db.upsert("mem-id", Record({"test": "memory2"}))
        assert id2 == "mem-id"
    
    @pytest.mark.asyncio
    async def test_memory_backend_async(self):
        """Test async memory backend."""
        from dataknobs_data.backends.memory import AsyncMemoryDatabase
        
        db = AsyncMemoryDatabase()
        record = Record({"test": "async-memory"})
        
        # New style
        id1 = await db.upsert(record)
        assert id1 is not None
        
        # Old style
        id2 = await db.upsert("async-mem-id", Record({"test": "async-memory2"}))
        assert id2 == "async-mem-id"


class TestRecordIDManagement:
    """Test how Record IDs are managed during upsert."""
    
    def test_record_id_priority(self):
        """Test the priority of ID sources: storage_id > id field > generated."""
        db = SyncMemoryDatabase()
        
        # Case 1: Record with both storage_id and id field - storage_id wins
        record1 = Record({"id": "field-id", "data": "test"})
        record1.storage_id = "storage-id"
        result1 = db.upsert(record1)
        assert result1 == "storage-id"
        
        # Case 2: Record with only id field
        record2 = Record({"id": "field-id-2", "data": "test"})
        result2 = db.upsert(record2)
        assert result2 == "field-id-2"
        
        # Case 3: Record with neither - generates UUID
        record3 = Record({"data": "test"})
        result3 = db.upsert(record3)
        assert result3 is not None
        # Check it's a valid UUID format
        try:
            uuid.UUID(result3)
            uuid_valid = True
        except ValueError:
            uuid_valid = False
        assert uuid_valid
    
    def test_storage_id_assignment(self):
        """Test that storage_id is assigned when a new ID is generated."""
        db = SyncMemoryDatabase()
        
        # Create record without ID
        record = Record({"data": "test"})
        assert record.storage_id is None
        
        # Upsert should assign storage_id
        result_id = db.upsert(record)
        
        # After upsert, record should have storage_id set
        assert record.storage_id == result_id