"""Test factory integration for validation and migration modules."""

import pytest

from dataknobs_data.validation import (
    SchemaFactory,
    CoercerFactory,
    Schema,
    Coercer,
)
from dataknobs_data.migration import (
    MigrationFactory,
    TransformerFactory,
    MigratorFactory,
    Migration,
    Transformer,
    Migrator,
)
from dataknobs_data.records import Record


class TestValidationFactories:
    """Test validation v2 factory classes."""
    
    def test_schema_factory_basic(self):
        """Test creating a basic schema from factory."""
        factory = SchemaFactory()
        
        config = {
            "name": "test_schema",
            "strict": True,
            "description": "Test schema",
            "fields": [
                {
                    "name": "username",
                    "type": "STRING",
                    "required": True,
                    "constraints": [
                        {"type": "length", "min": 3, "max": 20}
                    ]
                },
                {
                    "name": "age",
                    "type": "INTEGER",
                    "constraints": [
                        {"type": "range", "min": 0, "max": 150}
                    ]
                }
            ]
        }
        
        schema = factory.create(**config)
        
        assert isinstance(schema, Schema)
        assert schema.name == "test_schema"
        assert schema.strict is True
        assert len(schema.fields) == 2
        
        # Test validation with the created schema
        valid_record = Record({"username": "john_doe", "age": 30})
        result = schema.validate(valid_record)
        assert result.valid
        
        invalid_record = Record({"username": "ab", "age": 200})
        result = schema.validate(invalid_record)
        assert not result.valid
    
    def test_coercer_factory(self):
        """Test creating a coercer from factory."""
        factory = CoercerFactory()
        coercer = factory.create()
        
        assert isinstance(coercer, Coercer)
        
        # Test coercion
        result = coercer.coerce("123", int)
        assert result.valid
        assert result.value == 123


class TestMigrationFactories:
    """Test migration v2 factory classes."""
    
    def test_migration_factory_basic(self):
        """Test creating a migration from factory."""
        factory = MigrationFactory()
        
        config = {
            "from_version": "1.0",
            "to_version": "2.0",
            "description": "Add metadata fields",
            "operations": [
                {
                    "type": "add_field",
                    "field_name": "created_at",
                    "default_value": "2024-01-01"
                },
                {
                    "type": "rename_field",
                    "old_name": "id",
                    "new_name": "record_id"
                },
                {
                    "type": "remove_field",
                    "field_name": "temp_field"
                }
            ]
        }
        
        migration = factory.create(**config)
        
        assert isinstance(migration, Migration)
        assert migration.from_version == "1.0"
        assert migration.to_version == "2.0"
        assert len(migration.operations) == 3
        
        # Test applying the migration
        record = Record({"id": 123, "temp_field": "temp", "data": "value"})
        migrated = migration.apply(record)
        
        assert "created_at" in migrated.fields
        assert "record_id" in migrated.fields
        assert "temp_field" not in migrated.fields
        assert migrated.get_value("record_id") == 123
    
    def test_transformer_factory(self):
        """Test creating a transformer from factory."""
        factory = TransformerFactory()
        
        config = {
            "rules": [
                {
                    "type": "rename",
                    "old_name": "old_field",
                    "new_name": "new_field"
                },
                {
                    "type": "exclude",
                    "fields": ["temp1", "temp2"]
                },
                {
                    "type": "add",
                    "field_name": "processed",
                    "value": True
                }
            ]
        }
        
        transformer = factory.create(**config)
        
        assert isinstance(transformer, Transformer)
        assert len(transformer.rules) == 3
        
        # Test transformation
        record = Record({
            "old_field": "value",
            "temp1": "remove1",
            "temp2": "remove2",
            "keep": "this"
        })
        
        transformed = transformer.transform(record)
        assert "new_field" in transformed.fields
        assert "old_field" not in transformed.fields
        assert "temp1" not in transformed.fields
        assert "temp2" not in transformed.fields
        assert transformed.get_value("processed") is True
        assert transformed.get_value("keep") == "this"
    
    def test_migrator_factory(self):
        """Test creating a migrator from factory."""
        factory = MigratorFactory()
        migrator = factory.create()
        
        assert isinstance(migrator, Migrator)


class TestFactoryIntegration:
    """Test integration between factories."""
    
    def test_schema_validation_in_migration(self):
        """Test using schema validation during migration."""
        # Create schema
        schema_factory = SchemaFactory()
        schema = schema_factory.create(
            name="target_schema",
            fields=[
                {
                    "name": "id",
                    "type": "INTEGER",
                    "required": True
                },
                {
                    "name": "name",
                    "type": "STRING",
                    "required": True,
                    "constraints": [
                        {"type": "length", "min": 1}
                    ]
                }
            ]
        )
        
        # Create transformer
        transformer_factory = TransformerFactory()
        transformer = transformer_factory.create(
            rules=[
                {
                    "type": "add",
                    "field_name": "name",
                    "value": "default_name"
                }
            ]
        )
        
        # Transform and validate
        record = Record({"id": 1})
        transformed = transformer.transform(record)
        
        result = schema.validate(transformed)
        assert result.valid
        assert result.value.get_value("name") == "default_name"