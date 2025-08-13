"""Tests for Record class."""

import pytest
from collections import OrderedDict

from dataknobs_data import Record, Field, FieldType


class TestRecord:
    """Test Record class."""
    
    def test_record_creation_from_dict(self):
        """Test creating a record from a dictionary of values."""
        data = {
            "name": "John Doe",
            "age": 30,
            "email": "john@example.com"
        }
        record = Record(data)
        
        assert record.get_value("name") == "John Doe"
        assert record.get_value("age") == 30
        assert record.get_value("email") == "john@example.com"
        assert len(record) == 3
    
    def test_record_creation_from_fields(self):
        """Test creating a record from Field objects."""
        fields = OrderedDict([
            ("name", Field(name="name", value="Jane Doe")),
            ("age", Field(name="age", value=25)),
        ])
        record = Record(fields)
        
        assert record.get_value("name") == "Jane Doe"
        assert record.get_value("age") == 25
        assert isinstance(record.get_field("name"), Field)
    
    def test_record_with_metadata(self):
        """Test record with metadata."""
        metadata = {"created_at": "2024-01-01", "version": 1}
        record = Record({"name": "Test"}, metadata=metadata)
        
        assert record.metadata == metadata
    
    def test_get_and_set_field(self):
        """Test getting and setting fields."""
        record = Record()
        
        # Set field
        record.set_field("name", "Alice", FieldType.STRING)
        assert record.get_value("name") == "Alice"
        
        # Get field
        field = record.get_field("name")
        assert field.name == "name"
        assert field.value == "Alice"
        assert field.type == FieldType.STRING
        
        # Get non-existent field
        assert record.get_field("missing") is None
        assert record.get_value("missing", "default") == "default"
    
    def test_remove_field(self):
        """Test removing fields."""
        record = Record({"a": 1, "b": 2, "c": 3})
        
        assert record.remove_field("b") is True
        assert "b" not in record
        assert len(record) == 2
        
        assert record.remove_field("missing") is False
    
    def test_has_field(self):
        """Test checking field existence."""
        record = Record({"name": "Test"})
        
        assert record.has_field("name") is True
        assert record.has_field("missing") is False
        assert "name" in record
        assert "missing" not in record
    
    def test_field_names_and_count(self):
        """Test getting field names and count."""
        record = Record({"a": 1, "b": 2, "c": 3})
        
        assert record.field_names() == ["a", "b", "c"]
        assert record.field_count() == 3
        assert len(record) == 3
    
    def test_dict_like_access(self):
        """Test dictionary-like access methods."""
        record = Record({"name": "Test", "value": 42})
        
        # Get by key
        assert record["name"].value == "Test"
        assert record["value"].value == 42
        
        # Get by index
        assert record[0].value == "Test"
        assert record[1].value == 42
        
        # Set by key
        record["new_field"] = "new_value"
        assert record.get_value("new_field") == "new_value"
        
        record["name"] = Field(name="name", value="Updated")
        assert record.get_value("name") == "Updated"
        
        # Delete by key
        del record["value"]
        assert "value" not in record
        
        # Invalid operations
        with pytest.raises(KeyError):
            _ = record["missing"]
        
        with pytest.raises(IndexError):
            _ = record[10]
        
        with pytest.raises(KeyError):
            del record["missing"]
    
    def test_iteration(self):
        """Test iterating over record."""
        record = Record({"a": 1, "b": 2, "c": 3})
        
        field_names = list(record)
        assert field_names == ["a", "b", "c"]
        
        for name in record:
            assert name in ["a", "b", "c"]
    
    def test_validation(self):
        """Test record validation."""
        # Valid record
        record = Record({
            "name": "Test",
            "age": 30,
            "active": True
        })
        assert record.validate() is True
        
        # Record with mismatched types
        record = Record()
        record.fields["bad"] = Field(name="bad", value="text", type=FieldType.INTEGER)
        assert record.validate() is False
    
    def test_to_dict(self):
        """Test converting record to dictionary."""
        record = Record(
            {"name": "Test", "value": 42},
            metadata={"version": 1}
        )
        
        # Full representation
        full_dict = record.to_dict(include_metadata=True, flatten=False)
        assert "fields" in full_dict
        assert "metadata" in full_dict
        assert full_dict["metadata"] == {"version": 1}
        
        # Without metadata
        no_meta_dict = record.to_dict(include_metadata=False, flatten=False)
        assert "fields" in no_meta_dict
        assert "metadata" not in no_meta_dict
        
        # Flattened
        flat_dict = record.to_dict(flatten=True)
        assert flat_dict == {"name": "Test", "value": 42}
    
    def test_from_dict(self):
        """Test creating record from dictionary representation."""
        # From full representation
        data = {
            "fields": {
                "name": {"name": "name", "value": "Test", "type": "string", "metadata": {}},
                "age": {"name": "age", "value": 30, "type": "integer", "metadata": {}}
            },
            "metadata": {"version": 1}
        }
        record = Record.from_dict(data)
        assert record.get_value("name") == "Test"
        assert record.get_value("age") == 30
        assert record.metadata == {"version": 1}
        
        # From simple dict
        simple_data = {"name": "Simple", "count": 5}
        record = Record.from_dict(simple_data)
        assert record.get_value("name") == "Simple"
        assert record.get_value("count") == 5
    
    def test_copy(self):
        """Test copying a record."""
        original = Record(
            {"name": "Original", "data": {"nested": "value"}},
            metadata={"id": 1}
        )
        
        # Deep copy
        deep_copy = original.copy(deep=True)
        deep_copy.set_field("name", "Copy")
        deep_copy.get_field("data").value["nested"] = "changed"
        deep_copy.metadata["id"] = 2
        
        assert original.get_value("name") == "Original"
        assert original.get_value("data") == {"nested": "value"}
        assert original.metadata["id"] == 1
        
        # Shallow copy
        shallow_copy = original.copy(deep=False)
        shallow_copy.metadata["new"] = "value"
        assert "new" not in original.metadata
    
    def test_project(self):
        """Test field projection."""
        record = Record({
            "name": "Test",
            "age": 30,
            "email": "test@example.com",
            "phone": "123-456-7890"
        })
        
        projected = record.project(["name", "email"])
        assert len(projected) == 2
        assert projected.get_value("name") == "Test"
        assert projected.get_value("email") == "test@example.com"
        assert "age" not in projected
        assert "phone" not in projected
    
    def test_merge(self):
        """Test merging records."""
        record1 = Record({"a": 1, "b": 2}, metadata={"v": 1})
        record2 = Record({"b": 3, "c": 4}, metadata={"v": 2})
        
        # Merge with overwrite
        merged = record1.merge(record2, overwrite=True)
        assert merged.get_value("a") == 1
        assert merged.get_value("b") == 3  # Overwritten
        assert merged.get_value("c") == 4
        assert merged.metadata["v"] == 2
        
        # Merge without overwrite
        merged = record1.merge(record2, overwrite=False)
        assert merged.get_value("a") == 1
        assert merged.get_value("b") == 2  # Not overwritten
        assert merged.get_value("c") == 4
        assert merged.metadata["v"] == 1