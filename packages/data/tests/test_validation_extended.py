"""Extended tests for validation modules to improve coverage."""

import pytest
from datetime import datetime, date
from decimal import Decimal
from typing import Any, Dict, List
from unittest.mock import Mock, patch
import json

from dataknobs_data.records import Record
from dataknobs_data.fields import Field, FieldType
from dataknobs_data.validation import (
    ValidationResult,
    ValidationError,
    SchemaValidator,
    Schema,
    FieldDefinition,
    TypeCoercer,
    CoercionError,
    RequiredConstraint,
    UniqueConstraint,
    MinValueConstraint,
    MaxValueConstraint,
    MinLengthConstraint,
    MaxLengthConstraint,
    PatternConstraint,
    EnumConstraint,
    CustomConstraint,
    Constraint
)


class TestConstraintClasses:
    """Test individual constraint classes."""
    
    def test_required_constraint(self):
        """Test RequiredConstraint."""
        constraint = RequiredConstraint()
        
        # None value should fail
        assert not constraint.validate(None)
        
        # Empty string should fail (when allow_empty=False, the default)
        assert not constraint.validate("")
        
        # Empty list should fail
        assert not constraint.validate([])
        
        # Valid values should pass
        assert constraint.validate("value")
        assert constraint.validate(0)
        assert constraint.validate(False)
        
        # Test with allow_empty=True
        constraint_allow_empty = RequiredConstraint(allow_empty=True)
        assert not constraint_allow_empty.validate(None)  # None still fails
        assert constraint_allow_empty.validate("")  # Empty string is allowed
        assert constraint_allow_empty.validate([])  # Empty list is allowed
    
    def test_unique_constraint(self):
        """Test UniqueConstraint."""
        constraint = UniqueConstraint(scope="global")
        
        # First value should pass
        assert constraint.validate("apple")
        
        # Duplicate value should fail
        assert not constraint.validate("apple")
        
        # New value should pass
        assert constraint.validate("banana")
        
        # Case sensitive check
        assert constraint.validate("Apple")  # Different case is considered unique
    
    def test_min_value_constraint(self):
        """Test MinValueConstraint."""
        constraint = MinValueConstraint(min_value=10)
        
        # Value below minimum should fail
        assert not constraint.validate(5)
        
        # Exact minimum should pass
        assert constraint.validate(10)
        
        # Value above minimum should pass
        assert constraint.validate(15)
        
        # Test with floats
        constraint = MinValueConstraint(min_value=10.5)
        assert not constraint.validate(10.4)
        assert constraint.validate(10.6)
    
    def test_max_value_constraint(self):
        """Test MaxValueConstraint."""
        constraint = MaxValueConstraint(max_value=100)
        
        # Value above maximum should fail
        assert not constraint.validate(101)
        
        # Exact maximum should pass
        assert constraint.validate(100)
        
        # Value below maximum should pass
        assert constraint.validate(50)
    
    def test_min_length_constraint(self):
        """Test MinLengthConstraint."""
        constraint = MinLengthConstraint(min_length=5)
        
        # String too short should fail
        assert not constraint.validate("abc")
        
        # Exact minimum length should pass
        assert constraint.validate("12345")
        
        # Longer string should pass
        assert constraint.validate("hello world")
        
        # Test with list
        assert not constraint.validate([1, 2])
        assert constraint.validate([1, 2, 3, 4, 5])
    
    def test_max_length_constraint(self):
        """Test MaxLengthConstraint."""
        constraint = MaxLengthConstraint(max_length=10)
        
        # String too long should fail
        assert not constraint.validate("this is too long")
        
        # Exact maximum length should pass
        assert constraint.validate("1234567890")
        
        # Shorter string should pass
        assert constraint.validate("short")
    
    def test_pattern_constraint(self):
        """Test PatternConstraint."""
        # Email pattern
        constraint = PatternConstraint(pattern=r'^[\w\.-]+@[\w\.-]+\.\w+$')
        
        # Valid email should pass
        assert constraint.validate("user@example.com")
        
        # Invalid email should fail
        assert not constraint.validate("invalid-email")
        
        # Phone pattern
        constraint = PatternConstraint(pattern=r'^\d{3}-\d{3}-\d{4}$')
        assert constraint.validate("123-456-7890")
        assert not constraint.validate("1234567890")
    
    def test_enum_constraint(self):
        """Test EnumConstraint."""
        constraint = EnumConstraint(allowed_values=["red", "green", "blue"])
        
        # Valid value should pass
        assert constraint.validate("red")
        
        # Invalid value should fail
        assert not constraint.validate("yellow")
        
        # Test with numbers
        constraint = EnumConstraint(allowed_values=[1, 2, 3])
        assert constraint.validate(2)
        assert not constraint.validate(4)
    
    def test_custom_constraint(self):
        """Test CustomConstraint."""
        # Even number validator
        def is_even(value):
            return isinstance(value, int) and value % 2 == 0
        
        constraint = CustomConstraint(validator=is_even, error_message="Value must be even")
        
        # Even number should pass
        assert constraint.validate(4)
        
        # Odd number should fail
        assert not constraint.validate(3)
        
        # Non-integer should fail
        assert not constraint.validate("4")
    
    def test_constraint_from_dict(self):
        """Test creating constraints from dictionary."""
        # Required constraint
        config = {"type": "RequiredConstraint"}
        constraint = Constraint.from_dict(config)
        assert isinstance(constraint, RequiredConstraint)
        
        # Min value constraint
        config = {"type": "MinValueConstraint", "min_value": 10}
        constraint = Constraint.from_dict(config)
        assert isinstance(constraint, MinValueConstraint)
        
        # Pattern constraint
        config = {"type": "PatternConstraint", "pattern": r"^\d+$"}
        constraint = Constraint.from_dict(config)
        assert isinstance(constraint, PatternConstraint)
        
        # Enum constraint
        config = {"type": "EnumConstraint", "allowed_values": ["a", "b", "c"]}
        constraint = Constraint.from_dict(config)
        assert isinstance(constraint, EnumConstraint)
    


class TestTypeCoercer:
    """Test TypeCoercer class."""
    
    def test_coerce_to_string(self):
        """Test coercion to string."""
        coercer = TypeCoercer()
        
        # Various types to string
        result = coercer.coerce(123, str)
        assert result == "123"
        
        result = coercer.coerce(True, str)
        assert result == "True"
        
        result = coercer.coerce(None, str)
        assert result == ""
        
        result = coercer.coerce([1, 2, 3], str)
        assert result == "[1, 2, 3]"
        
        result = coercer.coerce({"key": "value"}, str)
        assert result == "{'key': 'value'}"
    
    def test_coerce_to_integer(self):
        """Test coercion to integer."""
        coercer = TypeCoercer()
        
        # Valid conversions
        result = coercer.coerce("123", int)
        assert result == 123
        
        result = coercer.coerce(123.5, int)
        assert result == 123
        
        result = coercer.coerce(True, int)
        assert result == 1
        
        result = coercer.coerce(False, int)
        assert result == 0
        
        # Invalid conversions should raise exceptions
        with pytest.raises(CoercionError):
            coercer.coerce("abc", int)
        
        with pytest.raises(CoercionError):
            coercer.coerce(None, int)
        
        with pytest.raises(CoercionError):
            coercer.coerce([1, 2], int)
    
    def test_coerce_to_float(self):
        """Test coercion to float."""
        coercer = TypeCoercer()
        
        # Valid conversions
        result = coercer.coerce("123.45", float)
        assert result == 123.45
        
        result = coercer.coerce(123, float)
        assert result == 123.0
        
        result = coercer.coerce(True, float)
        assert result == 1.0
        
        # Invalid conversions
        with pytest.raises(CoercionError):
            coercer.coerce("abc", float)
        
        with pytest.raises(CoercionError):
            coercer.coerce(None, float)
    
    def test_coerce_to_boolean(self):
        """Test coercion to boolean."""
        coercer = TypeCoercer()
        
        # String values
        result = coercer.coerce("true", bool)
        assert result is True
        
        result = coercer.coerce("false", bool)
        assert result is False
        
        result = coercer.coerce("yes", bool)
        assert result is True
        
        result = coercer.coerce("no", bool)
        assert result is False
        
        result = coercer.coerce("1", bool)
        assert result is True
        
        result = coercer.coerce("0", bool)
        assert result is False
        
        # Numeric values
        result = coercer.coerce(1, bool)
        assert result is True
        
        result = coercer.coerce(0, bool)
        assert result is False
        
        # Empty/None values
        result = coercer.coerce("", bool)
        assert result is False
        
        result = coercer.coerce(None, bool)
        assert result is False
    
    def test_coerce_to_datetime(self):
        """Test coercion to datetime."""
        coercer = TypeCoercer()
        
        # Valid date strings
        result = coercer.coerce("2023-01-15", datetime)
        assert result.year == 2023
        assert result.month == 1
        assert result.day == 15
        
        result = coercer.coerce("2023-01-15 14:30:00", datetime)
        assert result.hour == 14
        assert result.minute == 30
        
        # ISO format
        result = coercer.coerce("2023-01-15T14:30:00", datetime)
        assert result is not None
        
        # Invalid conversions
        with pytest.raises(CoercionError):
            coercer.coerce("invalid-date", datetime)
        
        with pytest.raises(CoercionError):
            coercer.coerce(None, datetime)
        
        with pytest.raises(CoercionError):
            coercer.coerce(123, datetime)
    
    def test_coerce_to_list(self):
        """Test coercion to list."""
        coercer = TypeCoercer()
        
        # Comma-separated string
        result = coercer.coerce("1,2,3", list)
        assert result == ["1", "2", "3"]
        
        # Tuple
        result = coercer.coerce((1, 2, 3), list)
        assert result == [1, 2, 3]
        
        # Set
        result = coercer.coerce({1, 2, 3}, list)
        assert set(result) == {1, 2, 3}
        
        # Single value
        result = coercer.coerce("single", list)
        assert result == ["single"]
        
        # Already a list
        result = coercer.coerce([1, 2, 3], list)
        assert result == [1, 2, 3]
        
        # None
        result = coercer.coerce(None, list)
        assert result == []
    
    def test_coerce_to_dict(self):
        """Test coercion to dict."""
        coercer = TypeCoercer()
        
        # Valid JSON string
        result = coercer.coerce('{"key": "value"}', dict)
        assert result == {"key": "value"}
        
        # Already a dict
        result = coercer.coerce({"key": "value"}, dict)
        assert result == {"key": "value"}
        
        # Invalid conversions
        with pytest.raises(CoercionError):
            coercer.coerce("not-json", dict)
        
        with pytest.raises(CoercionError):
            coercer.coerce(None, dict)
        
        with pytest.raises(CoercionError):
            coercer.coerce(123, dict)
    
    def test_auto_coerce(self):
        """Test automatic type coercion."""
        coercer = TypeCoercer()
        
        # String to int
        result = coercer.coerce("123", FieldType.INTEGER)
        assert result == 123
        
        # String to float
        result = coercer.coerce("123.45", FieldType.FLOAT)
        assert result == 123.45
        
        # String to bool
        result = coercer.coerce("true", FieldType.BOOLEAN)
        assert result is True
        
        # String to datetime
        result = coercer.coerce("2023-01-15", FieldType.DATETIME)
        assert result.year == 2023
        
        # String to dict (JSON field type)
        result = coercer.coerce('{"key": "value"}', FieldType.JSON)
        assert result == {"key": "value"}
        
        # Unknown type - returns original
        result = coercer.coerce("test", "unknown_type")
        assert result == "test"


class TestSchemaAndValidation:
    """Test Schema and SchemaValidator classes."""
    
    def test_field_definition_creation(self):
        """Test FieldDefinition creation."""
        field = FieldDefinition(
            name="email",
            field_type=FieldType.STRING,
            required=True,
            description="User email address",
            constraints=[
                PatternConstraint(pattern=r'^[\w\.-]+@[\w\.-]+\.\w+$')
            ]
        )
        
        assert field.name == "email"
        assert field.field_type == FieldType.STRING
        assert field.required is True
        assert len(field.constraints) == 1
    
    def test_schema_creation(self):
        """Test Schema creation and methods."""
        schema = Schema(
            name="UserSchema",
            version="1.0.0",
            description="User data schema"
        )
        
        # Add field definitions
        schema.add_field(FieldDefinition(
            name="id",
            field_type=FieldType.STRING,
            required=True
        ))
        
        schema.add_field(FieldDefinition(
            name="age",
            field_type=FieldType.INTEGER,
            required=False,
            default_value=0,
            constraints=[
                MinValueConstraint(min_value=0),
                MaxValueConstraint(max_value=120)
            ]
        ))
        
        assert len(schema.fields) == 2
        assert "id" in schema.fields
        assert "age" in schema.fields
        assert schema.fields["id"].required
        assert not schema.fields["age"].required
    
    def test_schema_validator_basic(self):
        """Test SchemaValidator with basic validation."""
        # Create schema
        schema = Schema(name="TestSchema")
        schema.add_field(FieldDefinition(
            name="required_field",
            field_type=FieldType.STRING,
            required=True
        ))
        schema.add_field(FieldDefinition(
            name="optional_field",
            field_type=FieldType.STRING,
            required=False
        ))
        
        validator = SchemaValidator(schema)
        
        # Valid record
        record = Record()
        record.fields["required_field"] = Field(name="required_field", value="test")
        record.fields["optional_field"] = Field(name="optional_field", value="optional")
        
        result = validator.validate(record)
        assert result.is_valid
        
        # Missing required field
        record = Record()
        record.fields["optional_field"] = Field(name="optional_field", value="optional")
        
        result = validator.validate(record)
        assert not result.is_valid
        assert len(result.errors) > 0
    
    def test_schema_validator_with_constraints(self):
        """Test SchemaValidator with various constraints."""
        # Create schema with constraints
        schema = Schema(name="ConstraintSchema")
        
        schema.add_field(FieldDefinition(
            name="email",
            field_type=FieldType.STRING,
            required=True,
            constraints=[
                PatternConstraint(pattern=r'^[\w\.-]+@[\w\.-]+\.\w+$')
            ]
        ))
        
        schema.add_field(FieldDefinition(
            name="age",
            field_type=FieldType.INTEGER,
            required=True,
            constraints=[
                MinValueConstraint(min_value=18),
                MaxValueConstraint(max_value=100)
            ]
        ))
        
        schema.add_field(FieldDefinition(
            name="status",
            field_type=FieldType.STRING,
            required=True,
            constraints=[
                EnumConstraint(allowed_values=["active", "inactive", "pending"])
            ]
        ))
        
        validator = SchemaValidator(schema)
        
        # Valid record
        record = Record()
        record.fields["email"] = Field(name="email", value="user@example.com", type=FieldType.STRING)
        record.fields["age"] = Field(name="age", value=25, type=FieldType.INTEGER)
        record.fields["status"] = Field(name="status", value="active", type=FieldType.STRING)
        
        result = validator.validate(record)
        assert result.is_valid
        
        # Invalid email
        record.fields["email"] = Field(name="email", value="invalid-email", type=FieldType.STRING)
        result = validator.validate(record)
        assert not result.is_valid
        
        # Invalid age (too young)
        record.fields["email"] = Field(name="email", value="user@example.com", type=FieldType.STRING)
        record.fields["age"] = Field(name="age", value=15, type=FieldType.INTEGER)
        result = validator.validate(record)
        assert not result.is_valid
        
        # Invalid status
        record.fields["age"] = Field(name="age", value=25, type=FieldType.INTEGER)
        record.fields["status"] = Field(name="status", value="unknown", type=FieldType.STRING)
        result = validator.validate(record)
        assert not result.is_valid
    
    def test_schema_validator_with_coercion(self):
        """Test SchemaValidator with type coercion enabled."""
        schema = Schema(name="CoercionSchema")
        
        schema.add_field(FieldDefinition(
            name="count",
            field_type=FieldType.INTEGER,
            required=True
        ))
        
        schema.add_field(FieldDefinition(
            name="price",
            field_type=FieldType.FLOAT,
            required=True
        ))
        
        schema.add_field(FieldDefinition(
            name="active",
            field_type=FieldType.BOOLEAN,
            required=True
        ))
        
        validator = SchemaValidator(schema, enable_coercion=True)
        
        # Record with string values that need coercion
        record = Record()
        record.fields["count"] = Field(name="count", value="42", type=FieldType.STRING)
        record.fields["price"] = Field(name="price", value="19.99", type=FieldType.STRING)
        record.fields["active"] = Field(name="active", value="true", type=FieldType.STRING)
        
        result = validator.validate(record)
        assert result.is_valid
        
        # Check that values were coerced
        assert record.fields["count"].value == 42
        assert record.fields["price"].value == 19.99
        assert record.fields["active"].value is True
    
    def test_schema_to_dict_from_dict(self):
        """Test Schema serialization and deserialization."""
        schema = Schema(
            name="SerializableSchema",
            version="1.0.0",
            description="Test serialization"
        )
        
        schema.add_field(FieldDefinition(
            name="field1",
            field_type=FieldType.STRING,
            required=True,
            constraints=[
                MinLengthConstraint(min_length=5),
                MaxLengthConstraint(max_length=50)
            ]
        ))
        
        # Convert to dict
        schema_dict = schema.to_dict()
        assert schema_dict["name"] == "SerializableSchema"
        assert schema_dict["version"] == "1.0.0"
        assert "field1" in schema_dict["fields"]
        
        # Convert back from dict
        restored_schema = Schema.from_dict(schema_dict)
        assert restored_schema.name == schema.name
        assert restored_schema.version == schema.version
        assert "field1" in restored_schema.fields
    
    def test_validation_result(self):
        """Test ValidationResult class."""
        # Valid result
        result = ValidationResult(is_valid=True)
        assert result.is_valid
        assert len(result.errors) == 0
        assert len(result.warnings) == 0
        
        # Invalid result with errors
        result = ValidationResult(is_valid=False)
        result.add_error("field1", "Invalid value")
        result.add_error("field2", "Required field missing")
        
        assert not result.is_valid
        assert len(result.errors) == 2
        assert "field1" in result.errors
        assert "field2" in result.errors
        
        # Result with warnings
        result = ValidationResult(is_valid=True)
        result.add_warning("field1", "Value might be outdated")
        
        assert result.is_valid
        assert len(result.warnings) == 1
    
    def test_validation_error(self):
        """Test ValidationError class."""
        error = ValidationError(
            field="email",
            message="Invalid email format",
            constraint_type="pattern"
        )
        
        assert error.field == "email"
        assert error.message == "Invalid email format"
        assert error.constraint_type == "pattern"
    
    def test_custom_validation_with_schema(self):
        """Test custom validation rules in schema."""
        def validate_even_number(value):
            """Custom validator for even numbers."""
            return isinstance(value, int) and value % 2 == 0
        
        schema = Schema(name="CustomValidationSchema")
        schema.add_field(FieldDefinition(
            name="even_number",
            field_type=FieldType.INTEGER,
            required=True,
            constraints=[
                CustomConstraint(
                    validator=validate_even_number,
                    error_message="Value must be an even number"
                )
            ]
        ))
        
        validator = SchemaValidator(schema)
        
        # Valid even number
        record = Record()
        record.fields["even_number"] = Field(name="even_number", value=4, type=FieldType.INTEGER)
        result = validator.validate(record)
        assert result.is_valid
        
        # Invalid odd number
        record.fields["even_number"] = Field(name="even_number", value=3, type=FieldType.INTEGER)
        result = validator.validate(record)
        assert not result.is_valid
    
    def test_field_definition_with_metadata(self):
        """Test FieldDefinition with metadata."""
        field = FieldDefinition(
            name="temperature",
            field_type=FieldType.FLOAT,
            required=True,
            description="Temperature reading",
            metadata={
                "unit": "celsius",
                "sensor_id": "TEMP001",
                "precision": 0.1
            }
        )
        
        assert field.metadata["unit"] == "celsius"
        assert field.metadata["sensor_id"] == "TEMP001"
        assert field.metadata["precision"] == 0.1