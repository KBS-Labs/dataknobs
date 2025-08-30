"""
Cross-backend integration tests for streaming and migration.

Tests real data movement between different backend types.
"""

import pytest
import asyncio
import tempfile
import os
from typing import List

from dataknobs_data.records import Record
from dataknobs_data.fields import FieldType
from dataknobs_data.backends.memory import SyncMemoryDatabase, AsyncMemoryDatabase
from dataknobs_data.backends.file import SyncFileDatabase, AsyncFileDatabase
from dataknobs_data.query import Query
from dataknobs_data.streaming import StreamConfig, StreamResult
from dataknobs_data.migration import (
    Migrator,
    Transformer,
    Migration,
    AddField,
    RenameField,
    MigrationProgress,
)


class TestSyncCrossBackendStreaming:
    """Test streaming between different synchronous backends."""
    
    def setup_method(self):
        """Set up test databases."""
        self.memory_db = SyncMemoryDatabase()
        
        # Create a temporary file for the file database
        self.temp_file = tempfile.NamedTemporaryFile(
            mode='w', 
            suffix='.json', 
            delete=False
        )
        self.temp_file.close()
        self.file_db = SyncFileDatabase({"path": self.temp_file.name})
        
        # Populate memory database with test data
        self.test_records = []
        for i in range(100):
            record = Record(
                {
                    "name": f"Item {i}",
                    "value": i * 10,
                    "category": "A" if i % 2 == 0 else "B",
                    "tags": [f"tag{i}", f"tag{i+1}"],
                },
                metadata={"index": i, "processed": False},
                id=f"rec_{i}",
            )
            self.memory_db.create(record)
            self.test_records.append(record)
    
    def teardown_method(self):
        """Clean up test files."""
        try:
            os.unlink(self.temp_file.name)
        except (FileNotFoundError, AttributeError):
            pass
    
    def test_memory_to_file_streaming(self):
        """Test streaming from memory to file database."""
        config = StreamConfig(batch_size=20)
        
        # Stream all records from memory to file
        source_stream = self.memory_db.stream_read(config=config)
        result = self.file_db.stream_write(source_stream, config)
        
        assert result.total_processed == 100
        assert result.successful == 100
        assert result.failed == 0
        assert result.success_rate == 100.0
        
        # Verify data in file database
        assert self.file_db.count() == 100
        file_records = self.file_db.search(Query())
        assert len(file_records) == 100
    
    def test_filtered_streaming(self):
        """Test streaming with query filters."""
        query = Query().filter("category", "=", "A")
        config = StreamConfig(batch_size=10)
        
        # Stream only category A records
        source_stream = self.memory_db.stream_read(query, config)
        result = self.file_db.stream_write(source_stream, config)
        
        assert result.successful == 50  # Half are category A
        
        # Verify filtered data
        file_records = self.file_db.search(Query())
        assert all(r.get_value("category") == "A" for r in file_records)
    
    def test_transformed_streaming(self):
        """Test streaming with transformation."""
        config = StreamConfig(batch_size=15)
        
        # Define transformation
        def transform_record(record):
            new_record = record.copy()
            new_record.set_field("value", record.get_value("value") * 2)
            new_record.set_field("transformed", True)
            # Filter out category B
            if record.get_value("category") == "B":
                return None
            return new_record
        
        # Stream with transformation
        source_stream = self.memory_db.stream_transform(
            transform=transform_record,
            config=config
        )
        result = self.file_db.stream_write(source_stream, config)
        
        assert result.successful == 50  # Only category A records
        
        # Verify transformation
        file_records = self.file_db.search(Query())

        assert all(r.get_value("transformed") is True for r in file_records)
        assert all(r.get_value("category") == "A" for r in file_records)
        
        # Check values were doubled
        for record in file_records:
            original_index = int(record.metadata["index"])
            expected_value = original_index * 10 * 2
            assert record.get_value("value") == expected_value
    
    def test_bidirectional_streaming(self):
        """Test streaming in both directions."""
        config = StreamConfig(batch_size=25)
        
        # First: Memory -> File
        stream1 = self.memory_db.stream_read(config=config)
        result1 = self.file_db.stream_write(stream1, config)
        assert result1.successful == 100
        
        # Clear memory database
        for record in self.memory_db.search(Query()):
            self.memory_db.delete(record.id)
        assert self.memory_db.count() == 0
        
        # Second: File -> Memory
        stream2 = self.file_db.stream_read(config=config)
        result2 = self.memory_db.stream_write(stream2, config)
        assert result2.successful == 100
        
        # Verify round-trip
        assert self.memory_db.count() == 100
        memory_records = self.memory_db.search(Query())
        assert len(memory_records) == 100


class TestSyncCrossBackendMigration:
    """Test migration between different synchronous backends."""
    
    def setup_method(self):
        """Set up test environment."""
        self.source_db = SyncMemoryDatabase()
        
        # Create a temporary file for the target database
        self.temp_file = tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.json',
            delete=False
        )
        self.temp_file.close()
        self.target_db = SyncFileDatabase({"path": self.temp_file.name})
        
        # Populate source with v1 schema data
        for i in range(50):
            record = Record(
                {
                    "user_id": f"user_{i}",
                    "username": f"john_doe_{i}",
                    "email": f"john{i}@example.com",
                    "age": 20 + i,
                    "created": "2023-01-01"
                }
            )
            record.metadata["schema_version"] = "v1"
            self.source_db.create(record)
    
    def teardown_method(self):
        """Clean up test files."""
        try:
            os.unlink(self.temp_file.name)
        except (FileNotFoundError, AttributeError):
            pass
    
    def test_simple_migration(self):
        """Test basic migration between backends."""
        migrator = Migrator()
        progress = migrator.migrate(
            source=self.source_db,
            target=self.target_db,
            batch_size=10
        )
        
        assert progress.succeeded == 50
        assert progress.failed == 0
        
        # Verify data in target
        assert self.target_db.count() == 50
        target_records = self.target_db.search(Query())
        assert len(target_records) == 50
    
    def test_migration_with_schema_change(self):
        """Test migration with schema transformation."""
        # Create migration for v1 -> v2
        migration = Migration("v1", "v2", "Upgrade user schema")
        migration.add(RenameField("user_id", "id"))
        migration.add(RenameField("username", "display_name"))
        migration.add(AddField("account_type", "standard"))
        migration.add(AddField("is_active", True))
        migration.add(AddField("updated", "2024-01-01"))
        
        migrator = Migrator()
        progress = migrator.migrate(
            source=self.source_db,
            target=self.target_db,
            transform=migration,
            batch_size=10
        )
        
        assert progress.succeeded == 50
        
        # Verify schema changes
        target_records = self.target_db.search(Query())
        for record in target_records:
            # Old fields renamed
            assert "id" in record.fields
            assert "user_id" not in record.fields
            assert "display_name" in record.fields
            assert "username" not in record.fields
            
            # New fields added
            assert record.get_value("account_type") == "standard"
            assert record.get_value("is_active") is True
            assert record.get_value("updated") == "2024-01-01"
            
            # Metadata updated
            assert record.metadata.get("version") == "v2"
    
    def test_migration_with_complex_transform(self):
        """Test migration with complex transformation logic."""
        transformer = Transformer()
        
        # Complex transformations
        transformer.map("age", "age_group", lambda age: 
            "young" if age < 30 else "middle" if age < 50 else "senior"
        )
        transformer.map("email", "email_domain", lambda email: 
            email.split("@")[1] if "@" in email else "unknown"
        )
        transformer.add("migration_timestamp", "2024-01-01T00:00:00")
        transformer.exclude("created")  # Remove old timestamp
        
        migrator = Migrator()
        progress = migrator.migrate(
            source=self.source_db,
            target=self.target_db,
            transform=transformer,
            batch_size=15
        )
        
        assert progress.succeeded == 50
        
        # Verify transformations
        target_records = self.target_db.search(Query())
        for record in target_records:
            # Extract the number from user_id to get original age
            user_num = int(record.get_value("user_id").split("_")[1])
            age = 20 + user_num
            expected_group = "young" if age < 30 else "middle" if age < 50 else "senior"
            assert record.get_value("age_group") == expected_group
            assert record.get_value("email_domain") == "example.com"
            assert "created" not in record.fields
            assert record.get_value("migration_timestamp") == "2024-01-01T00:00:00"
    
    def test_migration_validation(self):
        """Test migration validation between backends."""
        # First migrate
        migrator = Migrator()
        progress = migrator.migrate(
            source=self.source_db,
            target=self.target_db
        )
        
        assert progress.succeeded == 50
        
        # Validate migration
        is_valid, issues = migrator.validate_migration(
            self.source_db,
            self.target_db
        )
        
        assert is_valid
        assert len(issues) == 0
        
        # Add extra record to source
        extra = Record({"user_id": "extra", "username": "extra_user"})
        self.source_db.create(extra)
        
        # Validation should fail
        is_valid, issues = migrator.validate_migration(
            self.source_db,
            self.target_db
        )
        
        assert not is_valid
        assert any("count mismatch" in issue.lower() for issue in issues)


@pytest.mark.asyncio
class TestAsyncCrossBackendOperations:
    """Test cross-backend operations with async databases."""
    
    async def create_test_environment(self):
        """Create a unique test environment for each async test."""
        self.async_memory = AsyncMemoryDatabase()
        
        # Create a unique temporary file for this test
        self.temp_file = tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.json',
            delete=False
        )
        self.temp_file.close()
        self.async_file = AsyncFileDatabase({"path": self.temp_file.name})
        
        # Populate async memory database
        for i in range(75):
            record = Record(
                {
                    "value": i * 5,
                    "status": "pending" if i % 3 == 0 else "active"
                },
                id=f"async_{i}",
            )
            await self.async_memory.create(record)
    
    async def cleanup_test_environment(self):
        """Clean up test files."""
        try:
            os.unlink(self.temp_file.name)
        except (FileNotFoundError, AttributeError):
            pass
    
    async def test_async_memory_to_file_streaming(self):
        """Test async streaming from memory to file."""
        await self.create_test_environment()
        
        config = StreamConfig(batch_size=15)
        
        # Stream from async memory to async file
        source_stream = self.async_memory.stream_read(config=config)
        result = await self.async_file.stream_write(source_stream, config)
        
        assert result.total_processed == 75
        assert result.successful == 75
        assert result.failed == 0
        
        # Verify data
        assert await self.async_file.count() == 75
        
        await self.cleanup_test_environment()
    
    async def test_async_filtered_migration(self):
        """Test async migration with filtering."""
        await self.create_test_environment()
        
        # Only migrate "pending" status records
        query = Query().filter("status", "=", "pending")
        
        migrator = Migrator()
        progress = await migrator.migrate_async(
            source=self.async_memory,
            target=self.async_file,
            query=query
        )
        
        # 25 records should have pending status (indices 0,3,6,9...72)
        assert progress.succeeded == 25
        
        # Verify filtered data
        target_records = await self.async_file.search(Query())
        assert all(r.get_value("status") == "pending" for r in target_records)
        
        await self.cleanup_test_environment()
    
    async def test_async_transform_migration(self):
        """Test async migration with transformation."""
        await self.create_test_environment()
        
        transformer = Transformer()
        transformer.map("value", "doubled_value", lambda x: x * 2)
        transformer.map("status", "is_active", lambda s: s == "active")
        transformer.add("migrated_at", "2024-01-01")
        
        migrator = Migrator()
        progress = await migrator.migrate_async(
            source=self.async_memory,
            target=self.async_file,
            transform=transformer
        )
        
        assert progress.succeeded == 75
        
        # Verify transformations
        target_records = await self.async_file.search(Query())

        for record in target_records:
            # Find original value from user id
            user_id = record.get_user_id()
            if user_id and "_" in str(user_id):
                id_parts = str(user_id).split("_")
                if len(id_parts) >= 2:
                    id_num = int(id_parts[1])
                else:
                    # Skip this record if we can't parse the ID
                    continue
            elif record.id and "_" in str(record.id):
                # Fall back to storage id if user id not available
                id_parts = str(record.id).split("_")
                if len(id_parts) >= 2:
                    id_num = int(id_parts[1])
                else:
                    # Skip this record if we can't parse the ID
                    continue
            else:
                # Skip this record if neither ID has the expected format
                continue
                
            original_value = id_num * 5
            assert record.get_value("doubled_value") == original_value * 2
            assert record.get_value("migrated_at") == "2024-01-01"
        
        await self.cleanup_test_environment()


class TestErrorHandlingAcrossBackends:
    """Test error handling in cross-backend operations."""
    
    def test_streaming_with_write_errors(self):
        """Test handling of write errors during streaming."""
        source = SyncMemoryDatabase()
        
        # Create a temporary file for the target
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            target = SyncFileDatabase({"path": temp_path})
            
            # Add test records
            for i in range(20):
                record = Record({"value": i}, id=f"rec_{i}")
                source.create(record)
            
            # Mock target to fail on specific records
            # stream_write uses create_batch, which uses create so we need to mock both
            original_create = target.create
            def failing_create(record):
                if record.get_value("value") in [5, 10, 15]:
                    raise ValueError(f"Cannot write value {record.get_value('value')}")
                return original_create(record)
            target.create = failing_create

            original_create_batch = target.create_batch
            def failing_create_batch(records):
                results = []
                for record in records:
                    failing_create(record)
                    results.append(record.id)
                return results
            target.create_batch = failing_create_batch
            
            # Stream with error handler
            def handle_error(error, record):
                # Continue on errors
                return True
            
            config = StreamConfig(batch_size=5, on_error=handle_error)
            
            # Stream should continue despite errors
            source_stream = source.stream_read(config=config)
            result = target.stream_write(source_stream, config)
            
            # Should have 3 failures
            assert result.failed >= 3
            assert result.successful <= 17
            assert len(result.errors) >= 3
        finally:
            # Clean up
            try:
                os.unlink(temp_path)
            except (FileNotFoundError, OSError):
                pass
    
    def test_migration_with_mixed_errors(self):
        """Test migration with errors in both transformation and writing."""
        source = SyncMemoryDatabase()
        target = SyncMemoryDatabase()
        
        # Add test records
        for i in range(15):
            record = Record(
                {
                    "value": i,
                    "type": "normal"
                    if i % 3 != 0 else "special"
                },
                id=i,
            )
            source.create(record)
        
        # Transformer that fails on certain values
        class FailingTransformer(Transformer):
            def transform(self, record):
                if record.get_value("value") == 7:
                    raise ValueError("Cannot transform 7")
                # Normal transformation
                result = super().transform(record)
                if result:
                    result.set_field("processed", True)
                return result
        
        # Mock target to fail on "special" type
        original_create = target.create
        def failing_create(record):
            if record.get_value("type") == "special":
                raise ValueError("Cannot store special type")
            return original_create(record)
        
        target.create = failing_create
        
        # Error handler that continues
        def error_handler(error, record):
            return True
        
        migrator = Migrator()
        progress = migrator.migrate(
            source=source,
            target=target,
            transform=FailingTransformer(),
            on_error=error_handler
        )
        
        # Should have failures from both transformer and target
        # Value 7 fails in transformer
        # Values 0, 3, 6, 9, 12 are "special" and fail in target
        assert progress.failed >= 5  # At least the special ones
        assert progress.succeeded <= 10
        assert len(progress.errors) >= 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
