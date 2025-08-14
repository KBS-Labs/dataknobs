"""Tests for schema validation utilities."""

import pytest
import re
from datetime import datetime
from typing import Any

from dataknobs_data.records import Record
from dataknobs_data.fields import Field, FieldType
from dataknobs_data.validation import (
    Schema,
    FieldDefinition,
    ValidationResult,
    ValidationError,
    SchemaValidator,
    RequiredConstraint,
    UniqueConstraint,
    MinValueConstraint,
    MaxValueConstraint,
    MinLengthConstraint,
    MaxLengthConstraint,
    PatternConstraint,
    EnumConstraint,
    CustomConstraint,
    TypeCoercer,
    CoercionError,
)


class TestFieldDefinition:
    """Test FieldDefinition class."""
    
    def test_basic_field_definition(self):
        """Test creating a basic field definition."""
        field_def = FieldDefinition(
            name="username",
            type=str,
            required=True,
            description="User's username"
        )
        
        assert field_def.name == "username"
        assert field_def.type == str
        assert field_def.required is True
        
        # Valid value
        result = field_def.validate("alice")
        assert result.is_valid
        
        # Invalid value (None for required field)
        result = field_def.validate(None)
        assert not result.is_valid
        assert len(result.errors) == 1
        assert result.errors[0].error_type == "required"
    
    def test_field_with_default(self):
        """Test field with default value."""
        field_def = FieldDefinition(
            name="status",
            type=str,
            default="pending"
        )
        
        # None is valid for non-required field
        result = field_def.validate(None)
        assert result.is_valid
        
        assert field_def.default == "pending"
    
    def test_field_type_validation(self):
        """Test type validation."""
        field_def = FieldDefinition(
            name="age",
            type=int,
            required=True
        )
        
        # Valid integer
        result = field_def.validate(25)
        assert result.is_valid
        
        # Invalid type
        result = field_def.validate("twenty-five")
        assert not result.is_valid
        assert result.errors[0].error_type == "type"
    
    def test_field_with_constraints(self):
        """Test field with constraints."""
        field_def = FieldDefinition(
            name="age",
            type=int,
            constraints=[
                MinValueConstraint(0),
                MaxValueConstraint(120)
            ]
        )
        
        # Valid value
        result = field_def.validate(25)
        assert result.is_valid
        
        # Too small
        result = field_def.validate(-5)
        assert not result.is_valid
        
        # Too large
        result = field_def.validate(150)
        assert not result.is_valid
    
    def test_custom_validator(self):
        """Test custom validation function."""
        def is_even(value: int) -> bool:
            return value % 2 == 0
        
        field_def = FieldDefinition(
            name="even_number",
            type=int,
            custom_validator=is_even
        )
        
        result = field_def.validate(4)
        assert result.is_valid
        
        result = field_def.validate(5)
        assert not result.is_valid
        assert result.errors[0].error_type == "custom"


class TestSchema:
    """Test Schema class."""
    
    def test_schema_creation(self):
        """Test creating a schema."""
        schema = Schema(
            name="UserSchema",
            version="1.0.0",
            fields={
                "username": FieldDefinition(
                    name="username",
                    type=str,
                    required=True
                ),
                "age": FieldDefinition(
                    name="age",
                    type=int,
                    required=False
                )
            }
        )
        
        assert schema.name == "UserSchema"
        assert schema.version == "1.0.0"
        assert len(schema.fields) == 2
    
    def test_schema_validation(self):
        """Test validating records against schema."""
        schema = Schema(
            fields={
                "name": FieldDefinition(
                    name="name",
                    type=str,
                    required=True
                ),
                "email": FieldDefinition(
                    name="email",
                    type=str,
                    required=True,
                    constraints=[
                        PatternConstraint(r'^[\w\.-]+@[\w\.-]+\.\w+$')
                    ]
                )
            }
        )
        
        # Valid record
        record = Record()
        record.fields["name"] = Field(name="name", value="Alice")
        record.fields["email"] = Field(name="email", value="alice@example.com")
        
        result = schema.validate(record)
        assert result.is_valid
        
        # Invalid email
        record.fields["email"] = Field(name="email", value="invalid-email")
        result = schema.validate(record)
        assert not result.is_valid
        
        # Missing required field
        record2 = Record()
        record2.fields["name"] = Field(name="name", value="Bob")
        result = schema.validate(record2)
        assert not result.is_valid
    
    def test_strict_mode(self):
        """Test strict mode rejecting extra fields."""
        schema = Schema(
            fields={
                "name": FieldDefinition(name="name", type=str)
            },
            strict=True
        )
        
        record = Record()
        record.fields["name"] = Field(name="name", value="Alice")
        record.fields["extra"] = Field(name="extra", value="not allowed")
        
        result = schema.validate(record)
        assert not result.is_valid
        assert any(e.error_type == "extra_field" for e in result.errors)
    
    def test_schema_coercion(self):
        """Test coercing data to match schema."""
        schema = Schema(
            fields={
                "age": FieldDefinition(
                    name="age",
                    type=int,
                    default=0
                ),
                "active": FieldDefinition(
                    name="active",
                    type=bool,
                    default=True
                )
            }
        )
        
        # Coerce from dict
        data = {
            "age": "25",  # String that should be coerced to int
            "active": "yes"  # String that should be coerced to bool
        }
        
        record = schema.coerce(data)
        assert record.fields["age"].value == 25
        assert record.fields["active"].value is True
        
        # Check metadata
        assert record.metadata["schema_version"] == "1.0.0"
    
    def test_schema_serialization(self):
        """Test schema serialization."""
        schema = Schema(
            name="TestSchema",
            version="2.0.0",
            fields={
                "id": FieldDefinition(
                    name="id",
                    type=str,
                    required=True
                )
            }
        )
        
        # To dict
        data = schema.to_dict()
        assert data["name"] == "TestSchema"
        assert data["version"] == "2.0.0"
        assert "fields" in data
        
        # From dict
        schema2 = Schema.from_dict(data)
        assert schema2.name == schema.name
        assert schema2.version == schema.version
        assert len(schema2.fields) == len(schema.fields)


class TestConstraints:
    """Test constraint classes."""
    
    def test_required_constraint(self):
        """Test RequiredConstraint."""
        constraint = RequiredConstraint()
        
        assert constraint.validate("value") is True
        assert constraint.validate(0) is True
        assert constraint.validate(None) is False
        
        # Test with allow_empty
        constraint2 = RequiredConstraint(allow_empty=False)
        assert constraint2.validate("") is False
        assert constraint2.validate([]) is False
        assert constraint2.validate({}) is False
    
    def test_unique_constraint(self):
        """Test UniqueConstraint."""
        constraint = UniqueConstraint()
        
        assert constraint.validate("first") is True
        assert constraint.validate("second") is True
        assert constraint.validate("first") is False  # Duplicate
        
        # Reset and test again
        constraint.reset()
        assert constraint.validate("first") is True
    
    def test_min_max_value_constraints(self):
        """Test Min/MaxValueConstraint."""
        min_constraint = MinValueConstraint(10)
        max_constraint = MaxValueConstraint(100)
        
        assert min_constraint.validate(10) is True
        assert min_constraint.validate(9) is False
        assert min_constraint.validate(50) is True
        
        assert max_constraint.validate(100) is True
        assert max_constraint.validate(101) is False
        assert max_constraint.validate(50) is True
        
        # Test exclusive bounds
        min_exclusive = MinValueConstraint(10, inclusive=False)
        assert min_exclusive.validate(10) is False
        assert min_exclusive.validate(11) is True
    
    def test_length_constraints(self):
        """Test Min/MaxLengthConstraint."""
        min_length = MinLengthConstraint(3)
        max_length = MaxLengthConstraint(10)
        
        assert min_length.validate("abc") is True
        assert min_length.validate("ab") is False
        assert min_length.validate("abcdef") is True
        
        assert max_length.validate("short") is True
        assert max_length.validate("this is too long") is False
        
        # Test with lists
        assert min_length.validate([1, 2, 3]) is True
        assert min_length.validate([1, 2]) is False
    
    def test_pattern_constraint(self):
        """Test PatternConstraint."""
        email_pattern = PatternConstraint(r'^[\w\.-]+@[\w\.-]+\.\w+$')
        
        assert email_pattern.validate("user@example.com") is True
        assert email_pattern.validate("invalid.email") is False
        
        # Test with flags
        case_insensitive = PatternConstraint(r'^hello', re.IGNORECASE)
        assert case_insensitive.validate("Hello world") is True
        assert case_insensitive.validate("HELLO there") is True
        assert case_insensitive.validate("goodbye") is False
    
    def test_enum_constraint(self):
        """Test EnumConstraint."""
        colors = EnumConstraint(["red", "green", "blue"])
        
        assert colors.validate("red") is True
        assert colors.validate("green") is True
        assert colors.validate("yellow") is False
        assert colors.validate("RED") is False  # Case sensitive
    
    def test_custom_constraint(self):
        """Test CustomConstraint."""
        def is_positive(value):
            return value > 0
        
        constraint = CustomConstraint(
            validator=is_positive,
            error_message="Value must be positive"
        )
        
        assert constraint.validate(5) is True
        assert constraint.validate(-3) is False
        assert constraint.get_error_message(-3) == "Value must be positive"
    
    def test_constraint_serialization(self):
        """Test constraint to/from dict."""
        constraint = MinValueConstraint(10)
        data = constraint.to_dict()
        
        assert data["type"] == "MinValueConstraint"
        assert data["min_value"] == 10
        
        # Note: from_dict would need proper implementation
        # This is a simplified test


class TestTypeCoercer:
    """Test TypeCoercer class."""
    
    def test_string_coercion(self):
        """Test coercing to string."""
        coercer = TypeCoercer()
        
        assert coercer.coerce(123, str) == "123"
        assert coercer.coerce(True, str) == "True"
        assert coercer.coerce([1, 2], str) == "[1, 2]"
        assert coercer.coerce(None, str) is None
    
    def test_int_coercion(self):
        """Test coercing to integer."""
        coercer = TypeCoercer()
        
        assert coercer.coerce("123", int) == 123
        assert coercer.coerce(123.45, int) == 123
        assert coercer.coerce("123.45", int) == 123
        assert coercer.coerce(True, int) == 1
        assert coercer.coerce("true", int) == 1
        assert coercer.coerce("false", int) == 0
        
        with pytest.raises(CoercionError):
            coercer.coerce("not a number", int)
    
    def test_float_coercion(self):
        """Test coercing to float."""
        coercer = TypeCoercer()
        
        assert coercer.coerce("123.45", float) == 123.45
        assert coercer.coerce(123, float) == 123.0
        assert coercer.coerce("inf", float) == float('inf')
        assert coercer.coerce("-inf", float) == float('-inf')
    
    def test_bool_coercion(self):
        """Test coercing to boolean."""
        coercer = TypeCoercer()
        
        assert coercer.coerce("true", bool) is True
        assert coercer.coerce("yes", bool) is True
        assert coercer.coerce("1", bool) is True
        assert coercer.coerce("false", bool) is False
        assert coercer.coerce("no", bool) is False
        assert coercer.coerce("0", bool) is False
        assert coercer.coerce("", bool) is False
        assert coercer.coerce(1, bool) is True
        assert coercer.coerce(0, bool) is False
    
    def test_list_coercion(self):
        """Test coercing to list."""
        coercer = TypeCoercer()
        
        assert coercer.coerce("[1, 2, 3]", list) == [1, 2, 3]
        assert coercer.coerce("a,b,c", list) == ["a", "b", "c"]
        assert coercer.coerce((1, 2, 3), list) == [1, 2, 3]
        assert coercer.coerce({1, 2, 3}, list) == [1, 2, 3]
        assert coercer.coerce("single", list) == ["single"]
    
    def test_dict_coercion(self):
        """Test coercing to dictionary."""
        coercer = TypeCoercer()
        
        assert coercer.coerce('{"key": "value"}', dict) == {"key": "value"}
        assert coercer.coerce("key1=val1,key2=val2", dict) == {"key1": "val1", "key2": "val2"}
        assert coercer.coerce([("a", 1), ("b", 2)], dict) == {"a": 1, "b": 2}
        assert coercer.coerce("", dict) == {}
    
    def test_datetime_coercion(self):
        """Test coercing to datetime."""
        coercer = TypeCoercer()
        
        # ISO format
        dt = coercer.coerce("2023-01-15T10:30:00", datetime)
        assert dt.year == 2023
        assert dt.month == 1
        assert dt.day == 15
        
        # Date only
        dt = coercer.coerce("2023-01-15", datetime)
        assert dt.year == 2023
        
        # Timestamp
        dt = coercer.coerce(1673779800, datetime)
        assert isinstance(dt, datetime)
        
        with pytest.raises(CoercionError):
            coercer.coerce("not a date", datetime)
    
    def test_field_type_coercion(self):
        """Test coercing with FieldType enum."""
        coercer = TypeCoercer()
        
        assert coercer.coerce("123", FieldType.INTEGER) == 123
        assert coercer.coerce("true", FieldType.BOOLEAN) is True
        assert coercer.coerce(123, FieldType.STRING) == "123"
    
    def test_string_type_name_coercion(self):
        """Test coercing with string type names."""
        coercer = TypeCoercer()
        
        assert coercer.coerce("123", "int") == 123
        assert coercer.coerce("123", "integer") == 123
        assert coercer.coerce("true", "bool") is True
        assert coercer.coerce("true", "boolean") is True
        assert coercer.coerce(123, "string") == "123"


class TestSchemaValidator:
    """Test SchemaValidator class."""
    
    def test_batch_validation(self):
        """Test validating multiple records."""
        schema = Schema(
            fields={
                "id": FieldDefinition(name="id", type=int, required=True),
                "name": FieldDefinition(name="name", type=str, required=True)
            }
        )
        
        validator = SchemaValidator(schema)
        
        # Create test records
        records = []
        for i in range(5):
            record = Record()
            record.fields["id"] = Field(name="id", value=i)
            record.fields["name"] = Field(name="name", value=f"Item {i}")
            records.append(record)
        
        # Add an invalid record
        invalid = Record()
        invalid.fields["id"] = Field(name="id", value="not_an_int")
        records.append(invalid)
        
        results = validator.validate_batch(records)
        
        assert len(results) == 6
        assert sum(1 for r in results if r.is_valid) == 5
        assert sum(1 for r in results if not r.is_valid) == 1
    
    def test_validation_caching(self):
        """Test validation result caching."""
        schema = Schema(
            fields={
                "value": FieldDefinition(name="value", type=int)
            }
        )
        
        validator = SchemaValidator(schema)
        
        record = Record()
        record.fields["value"] = Field(name="value", value=42)
        
        # First validation
        result1 = validator.validate_with_cache(record)
        
        # Second validation (should use cache)
        result2 = validator.validate_with_cache(record)
        
        assert result1.is_valid == result2.is_valid
        
        # Clear cache
        validator.clear_cache()
        
        # Validation after cache clear
        result3 = validator.validate_with_cache(record)
        assert result3.is_valid == result1.is_valid