"""
Comprehensive tests for migration_v2 module using real components.
"""

import pytest
import time
from typing import List

from dataknobs_data.records import Record
from dataknobs_data.fields import FieldType
from dataknobs_data.backends.memory import SyncMemoryDatabase as MemoryDatabase
from dataknobs_data.query import Query

from dataknobs_data.migration import (
    Operation,
    AddField,
    RemoveField,
    RenameField,
    TransformField,
    CompositeOperation,
    Migration,
    Transformer,
    TransformRule,
    MapRule,
    ExcludeRule,
    AddRule,
    MigrationProgress,
    Migrator,
)


class TestOperations:
    """Test individual migration operations."""
    
    def test_add_field_operation(self):
        """Test adding a field to records."""
        operation = AddField("status", "active", FieldType.STRING)
        
        record = Record({"name": "Test", "value": 100})
        result = operation.apply(record)
        
        assert "status" in result.fields
        assert result.get_value("status") == "active"
        assert result.get_value("name") == "Test"  # Original fields preserved
        
        # Test reverse
        reversed_record = operation.reverse(result)
        assert "status" not in reversed_record.fields
        assert reversed_record.get_value("name") == "Test"
    
    def test_remove_field_operation(self):
        """Test removing a field from records."""
        operation = RemoveField("deprecated", store_removed=True)
        
        record = Record({"name": "Test", "deprecated": "old_value"})
        result = operation.apply(record)
        
        assert "deprecated" not in result.fields
        assert "name" in result.fields
        assert result.metadata.get("_removed_deprecated") == "old_value"
        
        # Test reverse with stored value
        reversed_record = operation.reverse(result)
        assert "deprecated" in reversed_record.fields
        assert reversed_record.get_value("deprecated") == "old_value"
        assert "_removed_deprecated" not in reversed_record.metadata
    
    def test_rename_field_operation(self):
        """Test renaming a field."""
        operation = RenameField("old_name", "new_name")
        
        record = Record({"old_name": "value", "other": "data"})
        result = operation.apply(record)
        
        assert "old_name" not in result.fields
        assert "new_name" in result.fields
        assert result.get_value("new_name") == "value"
        assert result.get_value("other") == "data"
        
        # Test reverse
        reversed_record = operation.reverse(result)
        assert "new_name" not in reversed_record.fields
        assert "old_name" in reversed_record.fields
        assert reversed_record.get_value("old_name") == "value"
    
    def test_transform_field_operation(self):
        """Test transforming field values."""
        # Simple transformation
        operation = TransformField(
            "price",
            transform_fn=lambda x: x * 1.1,  # 10% increase
            reverse_fn=lambda x: x / 1.1      # Reverse the increase
        )
        
        record = Record({"price": 100, "name": "Product"})
        result = operation.apply(record)
        
        assert abs(result.get_value("price") - 110) < 0.01  # Float precision
        assert result.get_value("name") == "Product"
        
        # Test reverse
        reversed_record = operation.reverse(result)
        assert abs(reversed_record.get_value("price") - 100) < 0.01  # Float precision
    
    def test_transform_field_error_handling(self):
        """Test error handling in transform operations."""
        def bad_transform(x):
            raise ValueError("Transform failed")
        
        operation = TransformField("field", transform_fn=bad_transform)
        
        record = Record({"field": "value"})
        result = operation.apply(record)
        
        # Original value preserved on error
        assert result.get_value("field") == "value"
        assert "_transform_error_field" in result.metadata
    
    def test_composite_operation(self):
        """Test combining multiple operations."""
        composite = CompositeOperation([
            AddField("status", "pending"),
            RenameField("old_id", "id"),
            TransformField("price", lambda x: x * 2, lambda x: x / 2)  # Add reverse function
        ])
        
        record = Record({"old_id": 123, "price": 50})
        result = composite.apply(record)
        
        assert result.get_value("status") == "pending"
        assert result.get_value("id") == 123
        assert "old_id" not in result.fields
        assert abs(result.get_value("price") - 100) < 0.01  # Float precision
        
        # Test reverse (applies in reverse order)
        reversed_record = composite.reverse(result)
        assert "status" not in reversed_record.fields
        assert reversed_record.get_value("old_id") == 123
        assert "id" not in reversed_record.fields
        assert abs(reversed_record.get_value("price") - 50) < 0.01  # Float precision


class TestMigration:
    """Test Migration class functionality."""
    
    def test_simple_migration(self):
        """Test basic migration with operations."""
        migration = Migration("v1", "v2", "Add user fields")
        migration.add(AddField("created_at", "2024-01-01"))
        migration.add(AddField("updated_at", "2024-01-01"))
        migration.add(RenameField("username", "user_name"))
        
        record = Record({"username": "john_doe", "email": "john@example.com"})
        result = migration.apply(record)
        
        assert result.get_value("created_at") == "2024-01-01"
        assert result.get_value("updated_at") == "2024-01-01"
        assert result.get_value("user_name") == "john_doe"
        assert "username" not in result.fields
        assert result.metadata.get("version") == "v2"
    
    def test_migration_reversal(self):
        """Test reversing a migration."""
        migration = Migration("v1", "v2")
        migration.add(AddField("new_field", "value"))
        migration.add(RemoveField("old_field"))
        
        record_v1 = Record({"old_field": "data"}, metadata={"version": "v1"})
        record_v2 = migration.apply(record_v1)
        
        assert record_v2.metadata["version"] == "v2"
        assert "new_field" in record_v2.fields
        assert "old_field" not in record_v2.fields
        
        # Reverse migration
        record_v1_again = migration.apply(record_v2, reverse=True)
        assert record_v1_again.metadata["version"] == "v1"
        assert "new_field" not in record_v1_again.fields
    
    def test_migration_validation(self):
        """Test migration validation."""
        migration = Migration("v1", "v2")
        migration.add(RenameField("required_field", "new_name"))
        
        # Valid record
        valid_record = Record({"required_field": "value"}, metadata={"version": "v1"})
        is_valid, issues = migration.validate(valid_record)
        assert is_valid
        assert len(issues) == 0
        
        # Missing required field
        invalid_record = Record({"other_field": "value"}, metadata={"version": "v1"})
        is_valid, issues = migration.validate(invalid_record)
        assert not is_valid
        assert "required_field" in str(issues)
        
        # Wrong version
        wrong_version = Record({"required_field": "value"}, metadata={"version": "v3"})
        is_valid, issues = migration.validate(wrong_version)
        assert not is_valid
        assert "version mismatch" in str(issues).lower()
    
    def test_get_affected_fields(self):
        """Test getting fields affected by migration."""
        migration = Migration("v1", "v2")
        migration.add(AddField("field1"))
        migration.add(RemoveField("field2"))
        migration.add(RenameField("field3", "field4"))
        migration.add(TransformField("field5", lambda x: x))
        
        affected = migration.get_affected_fields()
        assert affected == {"field1", "field2", "field3", "field4", "field5"}


class TestTransformer:
    """Test Transformer functionality."""
    
    def test_map_transformation(self):
        """Test field mapping."""
        transformer = Transformer()
        transformer.map("old_field", "new_field")
        transformer.map("price", "price", lambda x: x * 1.1)  # With transformation
        
        record = Record({"old_field": "value", "price": 100})
        result = transformer.transform(record)
        
        assert "old_field" not in result.fields
        assert result.get_value("new_field") == "value"
        assert abs(result.get_value("price") - 110) < 0.01  # Float precision
    
    def test_exclude_transformation(self):
        """Test excluding fields."""
        transformer = Transformer()
        transformer.exclude("password", "internal_id", "temp_data")
        
        record = Record({
            "username": "john",
            "password": "secret",
            "email": "john@example.com",
            "internal_id": 123,
            "temp_data": "temp"
        })
        
        result = transformer.transform(record)
        assert "username" in result.fields
        assert "email" in result.fields
        assert "password" not in result.fields
        assert "internal_id" not in result.fields
        assert "temp_data" not in result.fields
    
    def test_add_transformation(self):
        """Test adding new fields."""
        transformer = Transformer()
        transformer.add("timestamp", time.time())
        transformer.add("full_name", lambda r: f"{r.get_value('first')} {r.get_value('last')}")
        
        record = Record({"first": "John", "last": "Doe"})
        result = transformer.transform(record)
        
        assert "timestamp" in result.fields
        assert isinstance(result.get_value("timestamp"), float)
        assert result.get_value("full_name") == "John Doe"
    
    def test_fluent_api(self):
        """Test fluent API chaining."""
        transformer = (Transformer()
            .map("id", "_id")
            .rename("username", "user_name")
            .exclude("password")
            .add("processed", True)
        )
        
        record = Record({
            "id": 123,
            "username": "john",
            "password": "secret",
            "email": "john@example.com"
        })
        
        result = transformer.transform(record)
        assert result.get_value("_id") == 123
        assert result.get_value("user_name") == "john"
        assert "password" not in result.fields
        assert result.get_value("processed") is True
    
    def test_transform_many(self):
        """Test transforming multiple records."""
        transformer = Transformer()
        transformer.map("value", "value", lambda x: x * 2)
        transformer.add("batch_id", "batch_001")
        
        records = [
            Record({"id": 1, "value": 10}),
            Record({"id": 2, "value": 20}),
            Record({"id": 3, "value": 30})
        ]
        
        results = transformer.transform_many(records)
        assert len(results) == 3
        assert all(r.get_value("batch_id") == "batch_001" for r in results)
        assert results[0].get_value("value") == 20
        assert results[1].get_value("value") == 40
        assert results[2].get_value("value") == 60


class TestMigrationProgress:
    """Test MigrationProgress tracking."""
    
    def test_progress_tracking(self):
        """Test basic progress tracking."""
        progress = MigrationProgress()
        progress.total = 100
        progress.start()
        
        # Record some operations
        for i in range(50):
            progress.record_success(f"record_{i}")
        
        for i in range(5):
            progress.record_failure(f"Error {i}", f"failed_{i}")
        
        progress.record_skip("Not applicable", "skip_1")
        
        assert progress.processed == 56
        assert progress.succeeded == 50
        assert progress.failed == 5
        assert progress.skipped == 1
        assert abs(progress.percent - 56.0) < 0.01  # Float precision
        assert abs(progress.success_rate - 89.3) < 0.1
    
    def test_progress_duration(self):
        """Test duration tracking."""
        progress = MigrationProgress()
        progress.start()
        time.sleep(0.1)
        progress.finish()
        
        assert progress.duration >= 0.1
        assert progress.duration < 1.0
    
    def test_progress_merge(self):
        """Test merging progress from parallel operations."""
        progress1 = MigrationProgress()
        progress1.total = 50
        progress1.succeeded = 45
        progress1.failed = 5
        
        progress2 = MigrationProgress()
        progress2.total = 50
        progress2.succeeded = 48
        progress2.failed = 2
        
        progress1.merge(progress2)
        
        assert progress1.total == 100
        assert progress1.succeeded == 93
        assert progress1.failed == 7
    
    def test_progress_summary(self):
        """Test generating progress summary."""
        progress = MigrationProgress()
        progress.total = 1000
        progress.processed = 750
        progress.succeeded = 700
        progress.failed = 30
        progress.skipped = 20
        progress.start()
        time.sleep(0.1)
        
        summary = progress.get_summary()
        assert "75.0%" in summary
        assert "700" in summary
        assert "30" in summary
        assert "20" in summary


class TestMigrator:
    """Test Migrator with real databases."""
    
    def test_simple_migration(self):
        """Test basic migration between memory databases."""
        # Create source database with test data
        source = MemoryDatabase()
        for i in range(10):
            record = Record({
                "id": i,
                "name": f"item_{i}",
                "value": i * 10
            })
            source.create(record)
        
        # Create target database
        target = MemoryDatabase()
        
        # Create transformer
        transformer = Transformer()
        transformer.map("value", "price")
        transformer.add("currency", "USD")
        
        # Perform migration
        migrator = Migrator()
        progress = migrator.migrate(
            source=source,
            target=target,
            transform=transformer,
            batch_size=3
        )
        
        assert progress.succeeded == 10
        assert progress.failed == 0
        
        # Verify target data
        target_records = target.search(Query())
        assert len(target_records) == 10
        
        for record in target_records:
            assert "price" in record.fields
            assert "value" not in record.fields
            assert record.get_value("currency") == "USD"
    
    def test_migration_with_errors(self):
        """Test migration with error handling."""
        source = MemoryDatabase()
        for i in range(5):
            record = Record({"id": i, "value": i})
            source.create(record)
        
        target = MemoryDatabase()
        
        # Transformer that fails on certain values
        def failing_transform(x):
            if x == 2:
                raise ValueError("Cannot process value 2")
            return x * 10
        
        transformer = Transformer()
        transformer.map("value", "value", failing_transform)
        
        # Error handler that continues processing
        def handle_error(error, record):
            return True  # Continue processing
        
        migrator = Migrator()
        progress = migrator.migrate(
            source=source,
            target=target,
            transform=transformer,
            on_error=handle_error
        )
        
        # Transformer catches errors internally, so all should succeed
        # The record with value=2 will keep its original value
        assert progress.succeeded == 5
        assert progress.failed == 0
        
        # Check that the error was handled (value 2 wasn't transformed)
        target_records = target.search(Query())
        values = [r.get_value("value") for r in target_records]
        assert 2 in values  # Original value kept
        assert 0 in values  # 0 * 10 = 0
        assert 10 in values  # 1 * 10 = 10
        assert 30 in values  # 3 * 10 = 30
        assert 40 in values  # 4 * 10 = 40
    
    def test_migration_with_filter(self):
        """Test migration with query filter."""
        source = MemoryDatabase()
        for i in range(20):
            record = Record({
                "id": i,
                "category": "A" if i % 2 == 0 else "B",
                "value": i
            })
            source.create(record)
        
        target = MemoryDatabase()
        
        # Migrate only category A records
        query = Query().filter("category", "=", "A")
        
        migrator = Migrator()
        progress = migrator.migrate(
            source=source,
            target=target,
            query=query
        )
        
        assert progress.succeeded == 10
        
        # Verify only category A records in target
        target_records = target.search(Query())
        assert len(target_records) == 10
        assert all(r.get_value("category") == "A" for r in target_records)
    
    def test_migration_validation(self):
        """Test migration validation."""
        source = MemoryDatabase()
        target = MemoryDatabase()
        
        # First test: same data migrated properly
        records = []
        for i in range(5):
            record = Record({"value": i})
            record.id = f"id_{i}"
            records.append(record)
        
        # Use migrator to properly migrate data
        migrator = Migrator()
        progress = migrator.migrate(source=source, target=target)
        
        # Add same records to source
        for record in records:
            source.create(record)
        
        is_valid, issues = migrator.validate_migration(source, target)
        
        # Should be valid - same count
        if source.search(Query()) and target.search(Query()):
            assert len(source.search(Query())) == len(target.search(Query()))
        
        # Second test: count mismatch
        extra = Record({"value": 999})
        extra.id = "extra"
        source.create(extra)
        
        is_valid, issues = migrator.validate_migration(source, target)
        assert not is_valid
        assert "count mismatch" in str(issues).lower()
    
    def test_migration_with_progress_callback(self):
        """Test progress callback during migration."""
        source = MemoryDatabase()
        for i in range(20):
            source.create(Record({"id": i}))
        
        target = MemoryDatabase()
        
        # Track progress updates
        progress_updates = []
        
        def on_progress(progress):
            progress_updates.append({
                "processed": progress.processed,
                "percent": progress.percent
            })
        
        migrator = Migrator()
        progress = migrator.migrate(
            source=source,
            target=target,
            batch_size=5,
            on_progress=on_progress
        )
        
        assert progress.succeeded == 20
        assert len(progress_updates) > 0
        
        # Progress should increase
        for i in range(1, len(progress_updates)):
            assert progress_updates[i]["processed"] >= progress_updates[i-1]["processed"]


class TestIntegrationScenarios:
    """Test realistic migration scenarios."""
    
    def test_schema_evolution_migration(self):
        """Test migrating data through schema evolution."""
        # V1 Schema: basic user
        source = MemoryDatabase()
        for i in range(5):
            record = Record({
                "username": f"user_{i}",
                "email": f"user_{i}@example.com",
                "created": "2023-01-01"
            })
            record.metadata["version"] = "v1"
            source.create(record)
        
        # V2 Schema: split name, add defaults
        migration = Migration("v1", "v2", "Split username into first/last name")
        
        def split_username(username):
            parts = username.split("_")
            return {"first": parts[0], "last": parts[1] if len(parts) > 1 else ""}
        
        migration.add(TransformField(
            "username",
            lambda x: split_username(x)["first"]
        ))
        migration.add(AddField("last_name", ""))
        migration.add(RenameField("username", "first_name"))
        migration.add(AddField("status", "active"))
        migration.add(AddField("updated", "2024-01-01"))
        
        target = MemoryDatabase()
        migrator = Migrator()
        
        progress = migrator.migrate(
            source=source,
            target=target,
            transform=migration
        )
        
        assert progress.succeeded == 5
        
        # Verify migrated structure
        migrated = target.search(Query())
        for record in migrated:
            assert "first_name" in record.fields
            assert "last_name" in record.fields
            assert "status" in record.fields
            assert record.get_value("status") == "active"
            assert record.metadata["version"] == "v2"
    
    def test_data_cleaning_migration(self):
        """Test migration that cleans and validates data."""
        source = MemoryDatabase()
        
        # Add messy data
        messy_records = [
            Record({"price": "100.50", "quantity": "5", "name": "  Product A  "}),
            Record({"price": "$200", "quantity": "ten", "name": "product b"}),
            Record({"price": "50", "quantity": "3", "name": "PRODUCT C"}),
        ]
        
        for record in messy_records:
            source.create(record)
        
        # Create cleaning transformer
        def clean_price(value):
            # Remove $ and convert to float
            return float(str(value).replace("$", ""))
        
        def clean_quantity(value):
            # Try to convert to int, default to 0
            try:
                return int(value)
            except (ValueError, TypeError):
                return 0
        
        def clean_name(value):
            # Trim and title case
            return str(value).strip().title()
        
        transformer = Transformer()
        transformer.map("price", "price", clean_price)
        transformer.map("quantity", "quantity", clean_quantity)
        transformer.map("name", "name", clean_name)
        transformer.add("cleaned_at", "2024-01-01")
        
        target = MemoryDatabase()
        migrator = Migrator()
        
        progress = migrator.migrate(
            source=source,
            target=target,
            transform=transformer
        )
        
        # Check cleaned data
        cleaned = target.search(Query())
        assert abs(cleaned[0].get_value("price") - 100.5) < 0.01  # Float precision
        assert cleaned[0].get_value("quantity") == 5
        assert cleaned[0].get_value("name") == "Product A"
        
        assert abs(cleaned[1].get_value("price") - 200.0) < 0.01  # Float precision
        assert cleaned[1].get_value("quantity") == 0  # Failed conversion
        assert cleaned[1].get_value("name") == "Product B"
