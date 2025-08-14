"""Tests for migration utilities."""

import pytest
import asyncio
from datetime import datetime
from typing import Any, Dict, List

from dataknobs_data.records import Record
from dataknobs_data.fields import Field, FieldType
from dataknobs_data.backends.memory import MemoryDatabase, SyncMemoryDatabase
from dataknobs_data.query import Query
from dataknobs_data.migration import (
    DataMigrator,
    MigrationResult,
    MigrationProgress,
    SchemaEvolution,
    SchemaVersion,
    Migration,
    MigrationType,
    DataTransformer,
    FieldMapping,
    ValueTransformer,
    TransformationPipeline,
)
from dataknobs_data.migration.schema_evolution import SchemaField


class TestDataMigrator:
    """Test DataMigrator class."""
    
    def test_sync_migration_basic(self):
        """Test basic synchronous migration."""
        # Create source and target databases
        source = SyncMemoryDatabase()
        target = SyncMemoryDatabase()
        
        # Add test records to source
        records = []
        for i in range(10):
            record = Record()
            record.fields["id"] = Field(name="id", value=f"item_{i}")
            record.fields["value"] = Field(name="value", value=i * 10)
            source.create(record)
            records.append(record)
        
        # Perform migration
        migrator = DataMigrator(source, target)
        result = migrator.migrate_sync()
        
        # Verify migration result
        assert result.success
        assert result.progress.total_records == 10
        assert result.progress.successful_records == 10
        assert result.progress.failed_records == 0
        
        # Verify target contains all records
        target_records = target.search(Query())
        assert len(target_records) == 10
    
    def test_sync_migration_with_transformation(self):
        """Test migration with record transformation."""
        source = SyncMemoryDatabase()
        target = SyncMemoryDatabase()
        
        # Add test records
        for i in range(5):
            record = Record()
            record.fields["value"] = Field(name="value", value=i)
            source.create(record)
        
        # Define transformation function
        def double_value(record: Record) -> Record:
            new_record = record.copy()
            if "value" in new_record.fields:
                new_record.fields["value"].value *= 2
            return new_record
        
        # Migrate with transformation
        migrator = DataMigrator(source, target)
        result = migrator.migrate_sync(transform=double_value)
        
        assert result.success
        assert result.progress.successful_records == 5
        
        # Verify transformed values
        target_records = target.search(Query())
        values = [r.fields["value"].value for r in target_records]
        assert sorted(values) == [0, 2, 4, 6, 8]
    
    def test_sync_migration_with_filter(self):
        """Test migration with record filtering."""
        source = SyncMemoryDatabase()
        target = SyncMemoryDatabase()
        
        # Add test records
        for i in range(10):
            record = Record()
            record.fields["value"] = Field(name="value", value=i)
            source.create(record)
        
        # Define filter transformation
        def filter_even(record: Record) -> Record:
            if record.fields["value"].value % 2 == 0:
                return record
            return None  # Skip odd values
        
        # Migrate with filter
        migrator = DataMigrator(source, target)
        result = migrator.migrate_sync(transform=filter_even)
        
        assert result.success
        
        # Verify only even values migrated
        target_records = target.search(Query())
        values = [r.fields["value"].value for r in target_records]
        assert sorted(values) == [0, 2, 4, 6, 8]
    
    def test_sync_migration_with_progress_callback(self):
        """Test migration with progress tracking."""
        source = SyncMemoryDatabase()
        target = SyncMemoryDatabase()
        
        # Add test records
        for i in range(100):
            record = Record()
            record.fields["id"] = Field(name="id", value=i)
            source.create(record)
        
        # Track progress
        progress_updates = []
        
        def track_progress(progress: MigrationProgress):
            progress_updates.append({
                'processed': progress.processed_records,
                'percentage': progress.progress_percentage
            })
        
        # Migrate with progress tracking
        migrator = DataMigrator(source, target)
        result = migrator.migrate_sync(
            batch_size=20,
            progress_callback=track_progress
        )
        
        assert result.success
        assert len(progress_updates) > 0
        assert progress_updates[-1]['processed'] == 100
        assert progress_updates[-1]['percentage'] == 100.0
    
    @pytest.mark.asyncio
    async def test_async_migration_basic(self):
        """Test basic asynchronous migration."""
        # Create async databases
        source = MemoryDatabase()
        target = MemoryDatabase()
        
        # Add test records
        for i in range(10):
            record = Record()
            record.fields["id"] = Field(name="id", value=f"async_{i}")
            record.fields["value"] = Field(name="value", value=i * 100)
            await source.create(record)
        
        # Perform async migration
        migrator = DataMigrator(source, target)
        result = await migrator.migrate_async()
        
        assert result.success
        assert result.progress.total_records == 10
        assert result.progress.successful_records == 10
        
        # Verify target contains all records
        target_records = await target.search(Query())
        assert len(target_records) == 10


class TestSchemaEvolution:
    """Test SchemaEvolution class."""
    
    def test_schema_version_creation(self):
        """Test creating schema versions."""
        version = SchemaVersion(
            version="1.0.0",
            description="Initial schema",
            fields={
                "name": SchemaField(name="name", type=FieldType.STRING, required=True),
                "age": SchemaField(name="age", type=FieldType.INTEGER, default=0),
            }
        )
        
        assert version.version == "1.0.0"
        assert "name" in version.fields
        assert "age" in version.fields
        
        # Test serialization
        data = version.to_dict()
        assert data["version"] == "1.0.0"
        assert "fields" in data
        
        # Test deserialization
        version2 = SchemaVersion.from_dict(data)
        assert version2.version == version.version
        assert len(version2.fields) == len(version.fields)
    
    def test_migration_add_field(self):
        """Test migration that adds a field."""
        migration = Migration(
            from_version="1.0.0",
            to_version="1.1.0",
            migration_type=MigrationType.ADD_FIELD,
            operations=[{
                'type': MigrationType.ADD_FIELD.value,
                'field_name': 'email',
                'field_type': 'str',
                'default_value': ''
            }]
        )
        
        # Create test record
        record = Record()
        record.fields["name"] = Field(name="name", value="Alice")
        
        # Apply forward migration
        migrated = migration.apply_forward(record)
        assert "email" in migrated.fields
        assert migrated.fields["email"].value == ''
        
        # Apply backward migration
        reverted = migration.apply_backward(migrated)
        assert "email" not in reverted.fields
    
    def test_migration_rename_field(self):
        """Test migration that renames a field."""
        migration = Migration(
            from_version="1.0.0",
            to_version="1.1.0",
            migration_type=MigrationType.RENAME_FIELD,
            operations=[{
                'type': MigrationType.RENAME_FIELD.value,
                'old_name': 'username',
                'new_name': 'user_name'
            }]
        )
        
        # Create test record
        record = Record()
        record.fields["username"] = Field(name="username", value="bob123")
        
        # Apply forward migration
        migrated = migration.apply_forward(record)
        assert "user_name" in migrated.fields
        assert "username" not in migrated.fields
        assert migrated.fields["user_name"].value == "bob123"
        
        # Apply backward migration
        reverted = migration.apply_backward(migrated)
        assert "username" in reverted.fields
        assert "user_name" not in reverted.fields
    
    def test_schema_evolution_workflow(self):
        """Test complete schema evolution workflow."""
        evolution = SchemaEvolution()
        
        # Add versions
        v1 = SchemaVersion(
            version="1.0.0",
            fields={
                "id": SchemaField(name="id", type=FieldType.STRING, required=True),
                "name": SchemaField(name="name", type=FieldType.STRING, required=True),
            }
        )
        evolution.add_version(v1)
        
        v2 = SchemaVersion(
            version="2.0.0",
            fields={
                "id": SchemaField(name="id", type=FieldType.STRING, required=True),
                "full_name": SchemaField(name="full_name", type=FieldType.STRING, required=True),
                "email": SchemaField(name="email", type=FieldType.STRING, default=""),
            }
        )
        evolution.add_version(v2)
        
        # Add migration
        migration = Migration(
            from_version="1.0.0",
            to_version="2.0.0",
            migration_type=MigrationType.CUSTOM,
            operations=[
                {
                    'type': MigrationType.RENAME_FIELD.value,
                    'old_name': 'name',
                    'new_name': 'full_name'
                },
                {
                    'type': MigrationType.ADD_FIELD.value,
                    'field_name': 'email',
                    'field_type': 'str',
                    'default_value': ''
                }
            ]
        )
        evolution.add_migration(migration)
        
        # Test migration path
        path = evolution.get_migration_path("1.0.0", "2.0.0")
        assert len(path) == 1
        assert path[0].from_version == "1.0.0"
        
        # Migrate a record
        record = Record()
        record.fields["id"] = Field(name="id", value="123")
        record.fields["name"] = Field(name="name", value="John Doe")
        
        migrated = evolution.migrate_record(record, "1.0.0", "2.0.0")
        assert "full_name" in migrated.fields
        assert migrated.fields["full_name"].value == "John Doe"
        assert "email" in migrated.fields
        assert migrated.metadata["schema_version"] == "2.0.0"
    
    def test_auto_detect_changes(self):
        """Test automatic change detection between schemas."""
        evolution = SchemaEvolution()
        
        v1 = SchemaVersion(
            version="1.0.0",
            fields={
                "id": SchemaField(name="id", type=FieldType.STRING),
                "name": SchemaField(name="name", type=FieldType.STRING),
                "age": SchemaField(name="age", type=FieldType.INTEGER),
            }
        )
        
        v2 = SchemaVersion(
            version="2.0.0",
            fields={
                "id": SchemaField(name="id", type=FieldType.STRING),
                "name": SchemaField(name="name", type=FieldType.STRING),
                "age": SchemaField(name="age", type=FieldType.FLOAT),  # Type change
                "email": SchemaField(name="email", type=FieldType.STRING),  # Added field
            }
        )
        
        migration = evolution.auto_detect_changes(v1, v2)
        
        assert migration.from_version == "1.0.0"
        assert migration.to_version == "2.0.0"
        
        # Check detected operations
        op_types = [op['type'] for op in migration.operations]
        assert MigrationType.ADD_FIELD.value in op_types
        assert MigrationType.CHANGE_TYPE.value in op_types


class TestDataTransformer:
    """Test DataTransformer class."""
    
    def test_field_mapping(self):
        """Test basic field mapping."""
        transformer = DataTransformer()
        transformer.add_field_mapping("old_name", "new_name")
        
        record = Record()
        record.fields["old_name"] = Field(name="old_name", value="test_value")
        
        transformed = transformer.transform(record)
        assert transformed is not None
        assert "new_name" in transformed.fields
        assert transformed.fields["new_name"].value == "test_value"
        assert "old_name" not in transformed.fields
    
    def test_field_transformation(self):
        """Test field value transformation."""
        transformer = DataTransformer()
        transformer.add_field_mapping(
            "price",
            "price_cents",
            transformer=lambda x: int(round(x * 100))  # Use round to handle float precision
        )
        
        record = Record()
        record.fields["price"] = Field(name="price", value=19.99)
        
        transformed = transformer.transform(record)
        assert "price_cents" in transformed.fields
        assert transformed.fields["price_cents"].value == 1999
    
    def test_exclude_fields(self):
        """Test field exclusion."""
        transformer = DataTransformer()
        transformer.exclude_fields("password", "secret")
        
        record = Record()
        record.fields["username"] = Field(name="username", value="alice")
        record.fields["password"] = Field(name="password", value="secret123")
        record.fields["secret"] = Field(name="secret", value="hidden")
        
        transformed = transformer.transform(record)
        assert "username" in transformed.fields
        assert "password" not in transformed.fields
        assert "secret" not in transformed.fields
    
    def test_record_filter(self):
        """Test record filtering."""
        transformer = DataTransformer()
        transformer.add_record_filter(
            lambda r: r.fields["age"].value >= 18
        )
        
        # Adult record - should pass
        adult = Record()
        adult.fields["name"] = Field(name="name", value="Alice")
        adult.fields["age"] = Field(name="age", value=25)
        
        transformed = transformer.transform(adult)
        assert transformed is not None
        
        # Minor record - should be filtered
        minor = Record()
        minor.fields["name"] = Field(name="name", value="Bob")
        minor.fields["age"] = Field(name="age", value=16)
        
        transformed = transformer.transform(minor)
        assert transformed is None
    
    def test_transformation_pipeline(self):
        """Test transformation pipeline."""
        # First transformer: rename field
        t1 = DataTransformer()
        t1.add_field_mapping("username", "user_name")
        
        # Second transformer: add prefix
        def add_prefix(record: Record) -> Record:
            for field_name in list(record.fields.keys()):
                if field_name == "user_name":
                    record.fields[field_name].value = f"USER_{record.fields[field_name].value}"
            return record
        
        # Create pipeline
        pipeline = TransformationPipeline(t1, add_prefix)
        
        record = Record()
        record.fields["username"] = Field(name="username", value="alice")
        
        transformed = pipeline.transform(record)
        assert "user_name" in transformed.fields
        assert transformed.fields["user_name"].value == "USER_alice"


class TestValueTransformer:
    """Test ValueTransformer utility functions."""
    
    def test_to_string(self):
        """Test string conversion."""
        assert ValueTransformer.to_string(123) == "123"
        assert ValueTransformer.to_string(True) == "True"
        assert ValueTransformer.to_string(None) == ""
        # Dict is converted to str, not JSON
        assert ValueTransformer.to_string({"key": "value"}) == "{'key': 'value'}"
    
    def test_to_int(self):
        """Test integer conversion."""
        assert ValueTransformer.to_int("123") == 123
        assert ValueTransformer.to_int(123.45) == 123
        assert ValueTransformer.to_int("123.45") == 123
        assert ValueTransformer.to_int(True) == 1
        assert ValueTransformer.to_int(False) == 0
        assert ValueTransformer.to_int(None) is None
    
    def test_to_float(self):
        """Test float conversion."""
        assert ValueTransformer.to_float("123.45") == 123.45
        assert ValueTransformer.to_float(123) == 123.0
        assert ValueTransformer.to_float(True) == 1.0
        assert ValueTransformer.to_float(None) is None
    
    def test_to_bool(self):
        """Test boolean conversion."""
        assert ValueTransformer.to_bool("true") is True
        assert ValueTransformer.to_bool("yes") is True
        assert ValueTransformer.to_bool("1") is True
        assert ValueTransformer.to_bool("false") is False
        assert ValueTransformer.to_bool("no") is False
        assert ValueTransformer.to_bool("0") is False
        assert ValueTransformer.to_bool(1) is True
        assert ValueTransformer.to_bool(0) is False
    
    def test_normalize_string(self):
        """Test string normalization."""
        assert ValueTransformer.normalize_string("  HELLO  ") == "hello"
        assert ValueTransformer.normalize_string("Test") == "test"
    
    def test_truncate(self):
        """Test truncation transformer."""
        truncate5 = ValueTransformer.truncate(5)
        assert truncate5("Hello World") == "Hello"
        assert truncate5("Hi") == "Hi"
    
    def test_map_values(self):
        """Test value mapping."""
        status_map = ValueTransformer.map_values({
            0: "inactive",
            1: "active",
            2: "pending"
        }, default="unknown")
        
        assert status_map(0) == "inactive"
        assert status_map(1) == "active"
        assert status_map(99) == "unknown"
    
    def test_chain(self):
        """Test transformer chaining."""
        chained = ValueTransformer.chain(
            ValueTransformer.to_string,
            ValueTransformer.normalize_string,
            ValueTransformer.truncate(3)
        )
        
        assert chained("  HELLO  ") == "hel"
        assert chained(123) == "123"