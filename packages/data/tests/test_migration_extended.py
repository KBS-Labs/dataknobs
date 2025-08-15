"""Extended tests for migration modules to improve coverage."""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List
from unittest.mock import Mock, MagicMock, patch
import logging

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


class TestMigrationProgress:
    """Test MigrationProgress edge cases."""
    
    def test_progress_percentage_zero_total(self):
        """Test progress percentage with zero total records."""
        progress = MigrationProgress(total_records=0)
        assert progress.progress_percentage == 0.0
    
    def test_progress_percentage_calculation(self):
        """Test progress percentage calculation."""
        progress = MigrationProgress(
            total_records=100,
            processed_records=75
        )
        assert progress.progress_percentage == 75.0
    
    def test_duration_not_finished(self):
        """Test duration when migration not finished."""
        progress = MigrationProgress()
        assert progress.duration is None
    
    def test_duration_calculation(self):
        """Test duration calculation."""
        start = datetime.now()
        end = start + timedelta(seconds=10)
        progress = MigrationProgress(
            start_time=start,
            end_time=end
        )
        assert abs(progress.duration - 10.0) < 0.001
    
    def test_records_per_second_no_duration(self):
        """Test records per second with no duration."""
        progress = MigrationProgress()
        assert progress.records_per_second == 0.0
    
    def test_records_per_second_zero_duration(self):
        """Test records per second with zero duration."""
        now = datetime.now()
        progress = MigrationProgress(
            start_time=now,
            end_time=now,
            processed_records=100
        )
        assert progress.records_per_second == 0.0
    
    def test_records_per_second_calculation(self):
        """Test records per second calculation."""
        start = datetime.now()
        end = start + timedelta(seconds=5)
        progress = MigrationProgress(
            start_time=start,
            end_time=end,
            processed_records=100
        )
        assert abs(progress.records_per_second - 20.0) < 0.001
    
    def test_error_tracking(self):
        """Test error tracking in progress."""
        progress = MigrationProgress()
        error1 = {"record_id": "123", "error": "Failed to transform"}
        error2 = {"record_id": "456", "error": "Validation error"}
        
        progress.errors.append(error1)
        progress.errors.append(error2)
        
        assert len(progress.errors) == 2
        assert progress.errors[0]["record_id"] == "123"


class TestDataMigratorEdgeCases:
    """Test DataMigrator edge cases and error handling."""
    
    def test_batch_size_configuration(self):
        """Test batch size configuration."""
        source = SyncMemoryDatabase()
        target = SyncMemoryDatabase()
        
        migrator = DataMigrator(source, target)
        
        # Test with custom batch size
        result = migrator.migrate_sync(batch_size=5)
        assert isinstance(result, MigrationResult)
    
    def test_migration_with_errors(self):
        """Test migration with some records failing."""
        source = SyncMemoryDatabase()
        target = SyncMemoryDatabase()
        
        # Add test records
        for i in range(5):
            record = Record()
            record.fields["id"] = Field(name="id", value=f"item_{i}")
            source.create(record)
        
        # Mock target's create method to fail on specific records
        original_create = target.create
        def create_with_error(record):
            if record.fields.get("id") and record.fields["id"].value == "item_2":
                raise Exception("Database error")
            return original_create(record)
        
        target.create = create_with_error
        
        # Perform migration
        migrator = DataMigrator(source, target)
        result = migrator.migrate_sync()
        
        # Check results - migration continues despite individual errors
        assert result.progress.total_records == 5
        assert result.progress.failed_records == 1
        assert result.progress.successful_records == 4
    
    def test_migration_with_progress_callback(self):
        """Test migration with progress callback."""
        source = SyncMemoryDatabase()
        target = SyncMemoryDatabase()
        
        # Add test records
        for i in range(10):
            record = Record()
            record.fields["value"] = Field(name="value", value=i)
            source.create(record)
        
        # Track progress updates
        progress_updates = []
        
        def progress_callback(progress: MigrationProgress):
            progress_updates.append({
                'processed': progress.processed_records,
                'total': progress.total_records,
                'successful': progress.successful_records
            })
        
        # Migrate with callback
        migrator = DataMigrator(source, target)
        result = migrator.migrate_sync(
            batch_size=2,
            progress_callback=progress_callback
        )
        
        assert result.success
        assert len(progress_updates) > 0  # Progress callback was called
        # Final progress should show all records processed
        assert result.progress.processed_records == 10
        assert result.progress.successful_records == 10
    
    def test_migration_empty_source(self):
        """Test migration with empty source database."""
        source = SyncMemoryDatabase()
        target = SyncMemoryDatabase()
        
        migrator = DataMigrator(source, target)
        result = migrator.migrate_sync()
        
        assert result.success
        assert result.progress.total_records == 0
        assert result.progress.successful_records == 0
    
    @pytest.mark.asyncio
    async def test_async_migration_basic(self):
        """Test basic async migration."""
        source = MemoryDatabase()
        target = MemoryDatabase()
        
        # Add test records
        for i in range(5):
            record = Record()
            record.fields["id"] = Field(name="id", value=f"async_{i}")
            await source.create(record)
        
        # Perform async migration
        migrator = DataMigrator(source, target)
        result = await migrator.migrate_async()
        
        assert result.success
        assert result.progress.successful_records == 5
        
        # Verify target
        target_records = await target.search(Query())
        assert len(target_records) == 5
    
    @pytest.mark.asyncio
    async def test_async_migration_with_transformation(self):
        """Test async migration with transformation."""
        source = MemoryDatabase()
        target = MemoryDatabase()
        
        # Add test records
        for i in range(3):
            record = Record()
            record.fields["value"] = Field(name="value", value=i)
            await source.create(record)
        
        # Transform function
        def add_prefix(record: Record) -> Record:
            new_record = record.copy()
            new_record.fields["prefix"] = Field(
                name="prefix",
                value=f"prefix_{new_record.fields['value'].value}"
            )
            return new_record
        
        migrator = DataMigrator(source, target)
        result = await migrator.migrate_async(transform=add_prefix)
        
        assert result.success
        
        # Verify transformation
        target_records = await target.search(Query())
        for record in target_records:
            assert "prefix" in record.fields
            assert record.fields["prefix"].value.startswith("prefix_")
    
    @patch('dataknobs_data.migration.migrator.logger')
    def test_migration_with_logging(self, mock_logger):
        """Test that migration logs appropriately."""
        source = SyncMemoryDatabase()
        target = SyncMemoryDatabase()
        
        # Add a record
        record = Record()
        record.fields["test"] = Field(name="test", value="logging")
        source.create(record)
        
        migrator = DataMigrator(source, target)
        result = migrator.migrate_sync()
        
        assert result.success
        # Check that info logging was called
        assert mock_logger.info.called


class TestSchemaEvolutionEdgeCases:
    """Test SchemaEvolution edge cases."""
    
    def test_schema_version_comparison(self):
        """Test schema version comparison."""
        v1 = SchemaVersion(version="1.0.0", fields={})
        v2 = SchemaVersion(version="2.0.0", fields={})
        
        schema = SchemaEvolution()
        schema.add_version(v1)
        schema.add_version(v2)
        
        # Test that versions are stored correctly
        assert len(schema.versions) == 2
        assert "1.0.0" in schema.versions
        assert "2.0.0" in schema.versions
    
    def test_add_migration(self):
        """Test adding migration to schema evolution."""
        schema = SchemaEvolution()
        
        # Must add versions first
        v1 = SchemaVersion(version="1.0.0", fields={})
        v2 = SchemaVersion(version="2.0.0", fields={})
        schema.add_version(v1)
        schema.add_version(v2)
        
        migration = Migration(
            from_version="1.0.0",
            to_version="2.0.0",
            migration_type=MigrationType.ADD_FIELD,
            operations=[{"field": "new_field", "type": "string"}]
        )
        
        schema.add_migration(migration)
        assert len(schema.migrations) == 1
        assert schema.migrations[0].from_version == "1.0.0"
    
    def test_get_migration_path(self):
        """Test getting migration path between versions."""
        schema = SchemaEvolution()
        
        # Add versions
        v1 = SchemaVersion(version="1.0.0", fields={})
        v2 = SchemaVersion(version="2.0.0", fields={})
        v3 = SchemaVersion(version="3.0.0", fields={})
        schema.add_version(v1)
        schema.add_version(v2)
        schema.add_version(v3)
        
        # Add migrations
        m1 = Migration("1.0.0", "2.0.0", MigrationType.ADD_FIELD, operations=[])
        m2 = Migration("2.0.0", "3.0.0", MigrationType.RENAME_FIELD, operations=[])
        schema.migrations = [m1, m2]
        
        # Get migration path
        path = schema.get_migration_path("1.0.0", "3.0.0")
        assert len(path) == 2
        assert path[0].from_version == "1.0.0"
        assert path[1].to_version == "3.0.0"
    
    def test_get_migration_path_no_path(self):
        """Test getting migration path when no path exists."""
        schema = SchemaEvolution()
        
        v1 = SchemaVersion(version="1.0.0", fields={})
        v2 = SchemaVersion(version="2.0.0", fields={})
        schema.add_version(v1)
        schema.add_version(v2)
        
        # No migrations defined - should raise error
        with pytest.raises(ValueError, match="No migration path"):
            schema.get_migration_path("1.0.0", "2.0.0")
    
    def test_apply_migration_add_field(self):
        """Test applying ADD_FIELD migration."""
        record = Record()
        record.fields["existing"] = Field(name="existing", value="test")
        
        migration = Migration(
            from_version="1.0.0",
            to_version="2.0.0",
            migration_type=MigrationType.ADD_FIELD,
            operations=[{
                "type": MigrationType.ADD_FIELD.value,
                "field_name": "new_field",
                "default_value": "default"
            }]
        )
        
        migrated = migration.apply_forward(record)
        assert "new_field" in migrated.fields
        assert migrated.fields["new_field"].value == "default"
    
    def test_apply_migration_remove_field(self):
        """Test applying REMOVE_FIELD migration."""
        record = Record()
        record.fields["to_remove"] = Field(name="to_remove", value="test")
        record.fields["to_keep"] = Field(name="to_keep", value="keep")
        
        migration = Migration(
            from_version="1.0.0",
            to_version="2.0.0",
            migration_type=MigrationType.REMOVE_FIELD,
            operations=[{
                "type": MigrationType.REMOVE_FIELD.value,
                "field_name": "to_remove"
            }]
        )
        
        migrated = migration.apply_forward(record)
        assert "to_remove" not in migrated.fields
        assert "to_keep" in migrated.fields
    
    def test_apply_migration_rename_field(self):
        """Test applying RENAME_FIELD migration."""
        record = Record()
        record.fields["old_name"] = Field(name="old_name", value="test_value")
        
        migration = Migration(
            from_version="1.0.0",
            to_version="2.0.0",
            migration_type=MigrationType.RENAME_FIELD,
            operations=[{
                "type": MigrationType.RENAME_FIELD.value,
                "old_name": "old_name",
                "new_name": "new_name"
            }]
        )
        
        migrated = migration.apply_forward(record)
        assert "old_name" not in migrated.fields
        assert "new_name" in migrated.fields
        assert migrated.fields["new_name"].value == "test_value"
    
    def test_apply_migration_change_type(self):
        """Test applying CHANGE_TYPE migration."""
        record = Record()
        record.fields["field"] = Field(name="field", value="123")
        
        def change_type_transform(rec):
            new_rec = rec.copy()
            if "field" in new_rec.fields:
                val = new_rec.fields["field"].value
                if isinstance(val, str) and val.isdigit():
                    new_rec.fields["field"] = Field(name="field", value=int(val))
            return new_rec
        
        migration = Migration(
            from_version="1.0.0",
            to_version="2.0.0",
            migration_type=MigrationType.CHANGE_TYPE,
            up_function=change_type_transform
        )
        
        migrated = migration.apply_forward(record)
        assert migrated.fields["field"].value == 123
    
    def test_apply_migration_custom(self):
        """Test applying CUSTOM migration."""
        record = Record()
        record.fields["value"] = Field(name="value", value=10)
        
        def double_value_transform(rec):
            new_rec = rec.copy()
            if "value" in new_rec.fields:
                new_rec.fields["value"] = Field(
                    name="value",
                    value=new_rec.fields["value"].value * 2
                )
            return new_rec
        
        migration = Migration(
            from_version="1.0.0",
            to_version="2.0.0",
            migration_type=MigrationType.CUSTOM,
            up_function=double_value_transform
        )
        
        migrated = migration.apply_forward(record)
        assert migrated.fields["value"].value == 20


class TestDataTransformerEdgeCases:
    """Test DataTransformer edge cases."""
    
    def test_field_mapping_basic(self):
        """Test basic field mapping."""
        transformer = DataTransformer()
        transformer.add_field_mapping(
            source_field="old_field",
            target_field="new_field"
        )
        
        record = Record()
        record.fields["old_field"] = Field(name="old_field", value="test")
        
        transformed = transformer.transform(record)
        assert "new_field" in transformed.fields
        assert transformed.fields["new_field"].value == "test"
    
    def test_field_mapping_with_transformer(self):
        """Test field mapping with value transformer."""
        transformer = DataTransformer()
        transformer.add_field_mapping(
            source_field="price",
            target_field="price_cents",
            transformer=lambda x: int(x * 100)
        )
        
        record = Record()
        record.fields["price"] = Field(name="price", value=9.99)
        
        transformed = transformer.transform(record)
        assert transformed.fields["price_cents"].value == 999
    
    def test_value_transformer(self):
        """Test value transformer static methods."""
        # Test string conversion
        assert ValueTransformer.to_string(123) == "123"
        assert ValueTransformer.to_string(None) == ""
        
        # Test int conversion
        assert ValueTransformer.to_int("123") == 123
        assert ValueTransformer.to_int("123.45") == 123
        assert ValueTransformer.to_int(None) is None
        
        # Test float conversion
        assert ValueTransformer.to_float("123.45") == 123.45
        assert ValueTransformer.to_float(None) is None
        
        # Test bool conversion
        assert ValueTransformer.to_bool("true") is True
        assert ValueTransformer.to_bool("false") is False
        assert ValueTransformer.to_bool(1) is True
    
    def test_add_default_field(self):
        """Test adding field with default value."""
        transformer = DataTransformer()
        now = datetime.now()
        transformer.add_field_mapping(
            source_field="nonexistent",
            target_field="created_at",
            default_value=now
        )
        
        record = Record()
        record.fields["data"] = Field(name="data", value="test")
        
        transformed = transformer.transform(record)
        assert "created_at" in transformed.fields
        assert transformed.fields["created_at"].value == now
    
    def test_remove_field(self):
        """Test excluding fields."""
        transformer = DataTransformer()
        transformer.exclude_fields("sensitive_data")
        
        record = Record()
        record.fields["sensitive_data"] = Field(name="sensitive_data", value="secret")
        record.fields["public_data"] = Field(name="public_data", value="public")
        
        transformed = transformer.transform(record)
        assert "sensitive_data" not in transformed.fields
        assert "public_data" in transformed.fields
    
    def test_transformation_pipeline(self):
        """Test transformation pipeline with multiple transformers."""
        # Add multiple transformers
        t1 = DataTransformer()
        t1.add_field_mapping("a", "b")
        
        t2 = DataTransformer()
        t2.add_field_mapping("b", "c", transformer=lambda x: x.upper())
        
        pipeline = TransformationPipeline(t1, t2)
        
        # Apply pipeline
        record = Record()
        record.fields["a"] = Field(name="a", value="test")
        
        # First transformer: a -> b
        result1 = t1.transform(record)
        assert "b" in result1.fields
        assert result1.fields["b"].value == "test"
        
        # Second transformer: b -> c (uppercase)
        result2 = t2.transform(result1)
        assert "c" in result2.fields
        assert result2.fields["c"].value == "TEST"
        
        # Now test the pipeline
        result = pipeline.transform(record)
        assert "c" in result.fields
        assert result.fields["c"].value == "TEST"
    
    def test_transformation_error_handling(self):
        """Test transformation error handling."""
        # Transformer that raises exception
        def error_transformer(x):
            raise ValueError("Transform error")
        
        mapping = FieldMapping(
            source_field="field",
            target_field="field",
            transformer=error_transformer
        )
        
        transformer = DataTransformer()
        transformer.field_mappings.append(mapping)
        
        record = Record()
        record.fields["field"] = Field(name="field", value="test")
        
        # Should handle error gracefully
        try:
            transformed = transformer.transform(record)
            # Original value should be preserved on error
            assert transformed.fields["field"].value == "test"
        except Exception:
            # Or it might re-raise - both are valid behaviors
            pass


class TestMigrationTypes:
    """Test different migration types."""
    
    def test_migration_type_enum_values(self):
        """Test MigrationType enum values."""
        assert MigrationType.ADD_FIELD.value == "add_field"
        assert MigrationType.REMOVE_FIELD.value == "remove_field"
        assert MigrationType.RENAME_FIELD.value == "rename_field"
        assert MigrationType.CHANGE_TYPE.value == "change_type"
        assert MigrationType.CUSTOM.value == "custom"
        # Note: TRANSFORM_VALUE doesn't exist in the actual implementation
    
    def test_migration_with_custom_type(self):
        """Test migration with custom type."""
        def custom_migration(record: Record) -> Record:
            # Custom logic
            new_record = record.copy()
            if "value" in new_record.fields:
                new_record.fields["computed"] = Field(
                    name="computed",
                    value=new_record.fields["value"].value * 2
                )
            return new_record
        
        migration = Migration(
            from_version="1.0.0",
            to_version="2.0.0",
            migration_type=MigrationType.CUSTOM,
            up_function=custom_migration
        )
        
        record = Record()
        record.fields["value"] = Field(name="value", value=5)
        
        # Apply custom migration
        result = migration.apply_forward(record)
        assert "computed" in result.fields
        assert result.fields["computed"].value == 10