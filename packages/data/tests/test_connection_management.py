"""Test proper connection management in database backends."""

import pytest
from dataknobs_data.backends.memory import AsyncMemoryDatabase, SyncMemoryDatabase
from dataknobs_data.records import Record


class TestConnectionManagement:
    """Test that database backends properly implement connect/close."""
    
    def test_sync_memory_connect_close(self):
        """Test sync memory database connect/close (should be no-ops)."""
        db = SyncMemoryDatabase()
        
        # Should be able to connect
        db.connect()
        
        # Should be able to use after connect
        record = Record({"test": "data"})
        id = db.create(record)
        assert id is not None
        
        # Should be able to close
        db.close()
        
        # Memory database should still work after close (no real connection)
        retrieved = db.read(id)
        assert retrieved is not None
    
    @pytest.mark.asyncio
    async def test_async_memory_connect_close(self):
        """Test async memory database connect/close."""
        db = AsyncMemoryDatabase()
        
        # Should be able to connect
        await db.connect()
        
        # Should be able to use after connect
        record = Record({"test": "data"})
        id = await db.create(record)
        assert id is not None
        
        # Should be able to close
        await db.close()
        
        # Memory database should still work after close (no real connection)
        retrieved = await db.read(id)
        assert retrieved is not None
    
    def test_sync_context_manager(self):
        """Test sync database as context manager calls connect/close."""
        with SyncMemoryDatabase() as db:
            # Should automatically connect
            record = Record({"test": "data"})
            id = db.create(record)
            assert id is not None
        # Should automatically close on exit
    
    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        """Test async database as context manager calls connect/close."""
        async with AsyncMemoryDatabase() as db:
            # Should automatically connect
            record = Record({"test": "data"})
            id = await db.create(record)
            assert id is not None
        # Should automatically close on exit
    
    def test_operations_without_connect(self):
        """Test that operations work without explicit connect for memory database."""
        # Memory database doesn't require real connection
        db = SyncMemoryDatabase()
        
        # Should work even without explicit connect
        record = Record({"test": "data"})
        id = db.create(record)
        assert id is not None
        
        retrieved = db.read(id)
        assert retrieved is not None
        assert retrieved.get_value("test") == "data"