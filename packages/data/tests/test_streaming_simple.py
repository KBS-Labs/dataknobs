"""Simplified test for v2 modules streaming integration."""

import asyncio
import pytest

from dataknobs_data.backends.memory import AsyncMemoryDatabase, SyncMemoryDatabase
from dataknobs_data.records import Record
from dataknobs_data.query import Query

from dataknobs_data.validation import Schema, Required, Range
from dataknobs_data.migration import (
    Migration,
    AddField,
    RenameField,
    Transformer,
)


class TestAsyncV2Streaming:
    """Test v2 modules with async streaming."""
    
    @pytest.mark.asyncio
    async def test_basic_streaming_with_migration(self):
        """Test basic streaming with migration operations."""
        # Setup database with test data
        db = AsyncMemoryDatabase()
        await db.connect()
        
        # Create test records
        for i in range(1, 11):
            record = Record({"id": i, "name": f"item_{i}", "value": i * 10})
            await db.create(record)
        
        # Create migration
        migration = Migration("1.0", "2.0", "Test migration")
        migration.add(RenameField("id", "item_id"))
        migration.add(AddField("version", "2.0"))
        
        # Stream and migrate
        migrated_records = []
        async for record in db.stream_read(Query()):
            migrated = migration.apply(record)
            migrated_records.append(migrated)
        
        # Verify
        assert len(migrated_records) == 10
        first = migrated_records[0]
        assert "item_id" in first.fields
        assert "version" in first.fields
        assert first.get_value("version") == "2.0"
        
        await db.close()
    
    @pytest.mark.asyncio
    async def test_streaming_with_validation(self):
        """Test streaming with validation."""
        db = AsyncMemoryDatabase()
        await db.connect()
        
        # Create mixed valid/invalid records
        for i in range(1, 11):
            record = Record({
                "id": i,
                "price": i * 10 if i != 5 else -10,  # One invalid price
                "quantity": i
            })
            await db.create(record)
        
        # Create validation schema
        schema = Schema("test")
        schema.field("id", "INTEGER", required=True)
        schema.field("price", "FLOAT", constraints=[Range(min=0)])
        schema.field("quantity", "INTEGER", required=True)
        
        # Stream and validate
        valid_count = 0
        invalid_count = 0
        
        async for record in db.stream_read(Query()):
            price = record.get_value("price")
            if record.get_value("id") == 5:
                print(f"Record 5 price: {price}")
            result = schema.validate(record)
            if result.valid:
                valid_count += 1
            else:
                invalid_count += 1
                print(f"Invalid record {record.get_value('id')}: {result.errors}")
        
        # Range constraint validation now properly catches negative price
        assert valid_count == 9
        assert invalid_count == 1
        assert valid_count + invalid_count == 10  # Verify all were processed
        
        await db.close()
    
    @pytest.mark.asyncio
    async def test_streaming_with_transformer(self):
        """Test streaming with transformer."""
        db = AsyncMemoryDatabase()
        await db.connect()
        
        # Create test records
        for i in range(1, 6):
            record = Record({
                "old_id": i,
                "old_name": f"item_{i}",
                "temp": "remove_me",
                "keep": f"data_{i}"
            })
            await db.create(record)
        
        # Create transformer
        transformer = Transformer()
        transformer.rename("old_id", "id")
        transformer.rename("old_name", "name")
        transformer.exclude("temp")
        transformer.add("processed", True)
        
        # Stream and transform
        transformed_records = []
        async for record in db.stream_read(Query()):
            transformed = transformer.transform(record)
            transformed_records.append(transformed)
        
        # Verify
        assert len(transformed_records) == 5
        first = transformed_records[0]
        assert "id" in first.fields
        assert "name" in first.fields
        assert "temp" not in first.fields
        assert first.get_value("processed") is True
        assert "keep" in first.fields
        
        await db.close()


class TestSyncV2Streaming:
    """Test v2 modules with sync streaming."""
    
    def test_sync_streaming_with_migration(self):
        """Test sync streaming with migration."""
        db = SyncMemoryDatabase()
        db.connect()
        
        # Create test records
        for i in range(1, 21):
            record = Record({
                "user_id": i,
                "username": f"user_{i}",
                "score": i * 5
            })
            db.create(record)
        
        # Create migration
        migration = Migration("1.0", "2.0")
        migration.add(RenameField("user_id", "id"))
        migration.add(AddField("migrated", True))
        
        # Stream and process
        migrated_count = 0
        for record in db.stream_read(Query()):
            migrated = migration.apply(record)
            assert "id" in migrated.fields
            assert "migrated" in migrated.fields
            migrated_count += 1
        
        assert migrated_count == 20
        
        db.close()
    
    def test_sync_streaming_performance(self):
        """Test that streaming is memory efficient."""
        db = SyncMemoryDatabase()
        db.connect()
        
        # Create a larger dataset
        record_count = 100
        for i in range(record_count):
            record = Record({
                "id": i,
                "data": "x" * 100,  # Some bulk data
                "index": i
            })
            db.create(record)
        
        # Stream and count (simulating processing without storing all)
        processed_count = 0
        for record in db.stream_read(Query()):
            # Process record (in real scenario, might write to another system)
            assert record.get_value("id") == processed_count
            processed_count += 1
        
        assert processed_count == record_count
        
        db.close()


class TestStreamingIntegration:
    """Test integration between streaming and v2 modules."""
    
    @pytest.mark.asyncio
    async def test_migration_chain(self):
        """Test chaining multiple migrations in streaming."""
        db = AsyncMemoryDatabase()
        await db.connect()
        
        # Create test data
        for i in range(1, 11):
            record = Record({
                "v1_id": i,
                "v1_data": f"data_{i}",
                "deprecated": "old_value"
            })
            await db.create(record)
        
        # Create migration chain
        migration1 = Migration("1.0", "1.5")
        migration1.add(RenameField("v1_id", "v15_id"))
        migration1.add(RenameField("v1_data", "v15_data"))
        
        migration2 = Migration("1.5", "2.0")
        migration2.add(RenameField("v15_id", "id"))
        migration2.add(RenameField("v15_data", "data"))
        migration2.add(RemoveField("deprecated"))
        migration2.add(AddField("version", "2.0"))
        
        # Stream and apply migrations
        final_records = []
        async for record in db.stream_read(Query()):
            # Apply migration chain
            migrated = migration1.apply(record)
            migrated = migration2.apply(migrated)
            final_records.append(migrated)
        
        # Verify final state
        assert len(final_records) == 10
        first = final_records[0]
        assert "id" in first.fields
        assert "data" in first.fields
        assert "version" in first.fields
        assert "deprecated" not in first.fields
        
        await db.close()


# Import RemoveField for the test
from dataknobs_data.migration import RemoveField