"""
Comprehensive tests for validation_v2 module using real components.
"""

import pytest
from datetime import datetime

from dataknobs_data.records import Record
from dataknobs_data.fields import FieldType

from dataknobs_data.validation import (
    ValidationResult,
    ValidationContext,
    Constraint,
    All,
    AnyOf,
    Required,
    Range,
    Length,
    Pattern,
    Enum,
    Unique,
    Custom,
    Schema,
    Field,
    Coercer,
)


class TestValidationResult:
    """Test ValidationResult functionality."""
    
    def test_success_result(self):
        """Test creating a successful result."""
        result = ValidationResult.success(42)
        assert result.valid is True
        assert result.value == 42
        assert result.errors == []
        assert result.warnings == []
        assert bool(result) is True
    
    def test_failure_result(self):
        """Test creating a failed result."""
        result = ValidationResult.failure("invalid", ["Error 1", "Error 2"])
        assert result.valid is False
        assert result.value == "invalid"
        assert result.errors == ["Error 1", "Error 2"]
        assert result.warnings == []
        assert bool(result) is False
    
    def test_merge_results(self):
        """Test merging validation results."""
        result1 = ValidationResult.success(10, warnings=["Warning 1"])
        result2 = ValidationResult.failure(20, ["Error 1"])
        
        merged = result1.merge(result2)
        assert merged.valid is False
        assert merged.value == 10  # Keeps first value when second fails
        assert merged.errors == ["Error 1"]
        assert merged.warnings == ["Warning 1"]
    
    def test_add_error(self):
        """Test adding errors fluently."""
        result = ValidationResult.success(5)
        result.add_error("Something went wrong")
        
        assert result.valid is False
        assert result.errors == ["Something went wrong"]
    
    def test_add_warning(self):
        """Test adding warnings fluently."""
        result = ValidationResult.success(5)
        result.add_warning("Be careful")
        
        assert result.valid is True
        assert result.warnings == ["Be careful"]


class TestValidationContext:
    """Test ValidationContext for stateful validation."""
    
    def test_seen_values_tracking(self):
        """Test tracking seen values."""
        context = ValidationContext()
        
        assert not context.has_seen("field1", "value1")
        
        context.mark_seen("field1", "value1")
        assert context.has_seen("field1", "value1")
        assert not context.has_seen("field1", "value2")
        assert not context.has_seen("field2", "value1")
    
    def test_clear_seen_values(self):
        """Test clearing seen values."""
        context = ValidationContext()
        context.mark_seen("field1", "value1")
        context.mark_seen("field2", "value2")
        
        context.clear("field1")
        assert not context.has_seen("field1", "value1")
        assert context.has_seen("field2", "value2")
        
        context.clear()
        assert not context.has_seen("field2", "value2")
    
    def test_metadata_storage(self):
        """Test metadata storage and retrieval."""
        context = ValidationContext()
        
        context.set_metadata("key1", "value1")
        assert context.get_metadata("key1") == "value1"
        assert context.get_metadata("key2", "default") == "default"


class TestConstraints:
    """Test constraint implementations with real data."""
    
    def test_required_constraint(self):
        """Test Required constraint."""
        constraint = Required(allow_empty=False)
        
        # Test with None
        result = constraint.check(None)
        assert not result.valid
        assert "required" in result.errors[0].lower()
        
        # Test with empty string
        result = constraint.check("")
        assert not result.valid
        assert "empty" in result.errors[0].lower()
        
        # Test with value
        result = constraint.check("value")
        assert result.valid
        
        # Test allow_empty
        constraint = Required(allow_empty=True)
        result = constraint.check("")
        assert result.valid
    
    def test_range_constraint(self):
        """Test Range constraint."""
        import math
        
        constraint = Range(min=0, max=100)
        
        # Test in range
        assert constraint.check(50).valid
        assert constraint.check(0).valid
        assert constraint.check(100).valid
        
        # Test out of range
        assert not constraint.check(-1).valid
        assert not constraint.check(101).valid
        
        # Test non-numeric
        result = constraint.check("not a number")
        assert not result.valid
        assert "must be a number" in result.errors[0]
        
        # Test None (should be valid)
        assert constraint.check(None).valid
        
        # Test special float values
        # Infinity should fail (out of range)
        assert not constraint.check(math.inf).valid
        assert not constraint.check(-math.inf).valid
        
        # NaN should fail with specific error
        result = constraint.check(math.nan)
        assert not result.valid
        assert "NaN" in result.errors[0]
    
    def test_length_constraint(self):
        """Test Length constraint."""
        constraint = Length(min=2, max=5)
        
        # Test strings
        assert constraint.check("ab").valid
        assert constraint.check("abcde").valid
        assert not constraint.check("a").valid
        assert not constraint.check("abcdef").valid
        
        # Test lists
        assert constraint.check([1, 2]).valid
        assert not constraint.check([1]).valid
        
        # Test None (should be valid)
        assert constraint.check(None).valid
    
    def test_pattern_constraint(self):
        """Test Pattern constraint."""
        constraint = Pattern(r"^[A-Z][a-z]+$")
        
        assert constraint.check("Hello").valid
        assert not constraint.check("hello").valid
        assert not constraint.check("HELLO").valid
        assert not constraint.check("123").valid
        
        # Test non-string
        result = constraint.check(123)
        assert not result.valid
        assert "must be a string" in result.errors[0]
    
    def test_enum_constraint(self):
        """Test Enum constraint."""
        constraint = Enum(["red", "green", "blue"])
        
        assert constraint.check("red").valid
        assert constraint.check("green").valid
        assert not constraint.check("yellow").valid
        assert not constraint.check("RED").valid  # Case sensitive
        
        # Test error message
        result = constraint.check("yellow")
        assert "not in allowed values" in result.errors[0]
    
    def test_unique_constraint(self):
        """Test Unique constraint with context."""
        constraint = Unique("username")
        context = ValidationContext()
        
        # First occurrence should be valid
        assert constraint.check("user1", context).valid
        
        # Duplicate should fail
        result = constraint.check("user1", context)
        assert not result.valid
        assert "Duplicate" in result.errors[0]
        
        # Different value should be valid
        assert constraint.check("user2", context).valid
        
        # Without context, should succeed with warning
        result = constraint.check("user3", None)
        assert result.valid
        assert len(result.warnings) > 0
    
    def test_custom_constraint(self):
        """Test Custom constraint."""
        # Boolean validator
        def is_even(value):
            return value % 2 == 0
        
        constraint = Custom(is_even, "Value must be even")
        assert constraint.check(4).valid
        assert not constraint.check(3).valid
        
        # ValidationResult validator
        def complex_check(value):
            if value < 0:
                return ValidationResult.failure(value, ["Must be positive"])
            elif value > 100:
                return ValidationResult.success(value, warnings=["Value is large"])
            else:
                return ValidationResult.success(value)
        
        constraint = Custom(complex_check)
        assert constraint.check(50).valid
        assert not constraint.check(-5).valid
        
        result = constraint.check(150)
        assert result.valid
        assert len(result.warnings) > 0
    
    def test_constraint_composition(self):
        """Test combining constraints with AND/OR operators."""
        # AND composition
        constraint = Required() & Range(min=0, max=100)
        assert not constraint.check(None).valid
        assert not constraint.check(-1).valid
        assert constraint.check(50).valid
        
        # OR composition
        constraint = Pattern(r"^\d+$") | Pattern(r"^[a-z]+$")
        assert constraint.check("123").valid
        assert constraint.check("abc").valid
        assert not constraint.check("ABC").valid
        assert not constraint.check("123abc").valid
        
        # NOT operator
        constraint = ~Enum(["admin", "root"])
        assert constraint.check("user").valid
        assert not constraint.check("admin").valid


class TestSchema:
    """Test Schema validation with real Records."""
    
    def test_simple_schema(self):
        """Test basic schema validation."""
        schema = Schema("UserSchema", strict=False)
        schema.field("username", FieldType.STRING, required=True)
        schema.field("age", FieldType.INTEGER, constraints=[Range(min=0, max=150)])
        schema.field("email", FieldType.STRING, constraints=[Pattern(r"^.+@.+\..+$")])
        
        # Valid record
        record = Record({
            "username": "john_doe",
            "age": 30,
            "email": "john@example.com"
        })
        result = schema.validate(record)
        assert result.valid
        
        # Missing required field
        record = Record({
            "age": 30,
            "email": "john@example.com"
        })
        result = schema.validate(record)
        assert not result.valid
        assert "username" in str(result.errors)
        
        # Invalid constraint
        record = Record({
            "username": "john_doe",
            "age": 200,
            "email": "john@example.com"
        })
        result = schema.validate(record)
        assert not result.valid
        assert "age" in str(result.errors)
    
    def test_fluent_schema_api(self):
        """Test fluent API for schema definition."""
        schema = (Schema("Product")
            .field("id", "INTEGER", required=True)
            .field("name", "STRING", required=True, constraints=[Length(min=1, max=100)])
            .field("price", "FLOAT", constraints=[Range(min=0)])
            .field("in_stock", "BOOLEAN", default=True)
            .with_description("Product catalog schema")
        )
        
        assert len(schema.fields) == 4
        assert schema.description == "Product catalog schema"
        assert schema.fields["in_stock"].default is True
    
    def test_strict_mode(self):
        """Test strict mode rejecting unknown fields."""
        schema = Schema("StrictSchema", strict=True)
        schema.field("allowed_field", FieldType.STRING)
        
        record = Record({
            "allowed_field": "value",
            "unknown_field": "should fail"
        })
        
        result = schema.validate(record)
        assert not result.valid
        assert "Unknown fields" in str(result.errors)
    
    def test_validate_many(self):
        """Test validating multiple records with shared context."""
        schema = Schema("UniqueSchema")
        schema.field("id", FieldType.INTEGER, constraints=[Unique("id")])
        
        records = [
            Record({"id": 1}),
            Record({"id": 2}),
            Record({"id": 1}),  # Duplicate
        ]
        
        results = schema.validate_many(records)
        assert results[0].valid
        assert results[1].valid
        assert not results[2].valid  # Duplicate ID
    
    def test_schema_serialization(self):
        """Test converting schema to/from dict."""
        schema = Schema("TestSchema")
        schema.field("field1", FieldType.STRING, required=True)
        schema.field("field2", FieldType.INTEGER)
        
        # To dict
        data = schema.to_dict()
        assert data["name"] == "TestSchema"
        assert "field1" in data["fields"]
        assert data["fields"]["field1"]["required"] is True
        
        # From dict
        schema2 = Schema.from_dict(data)
        assert schema2.name == "TestSchema"
        assert len(schema2.fields) == 2


class TestCoercer:
    """Test type coercion with predictable behavior."""
    
    def test_string_coercion(self):
        """Test coercing to string."""
        coercer = Coercer()
        
        # Various types to string
        assert coercer.coerce(123, str).value == "123"
        assert coercer.coerce(True, str).value == "True"
        assert coercer.coerce(3.14, str).value == "3.14"
        
        # None should fail
        result = coercer.coerce(None, str)
        assert not result.valid
        assert "Cannot coerce None" in result.errors[0]
    
    def test_integer_coercion(self):
        """Test coercing to integer."""
        coercer = Coercer()
        
        # String to int
        assert coercer.coerce("123", int).value == 123
        assert coercer.coerce("0xFF", int).value == 255  # Hex
        assert coercer.coerce("0b1010", int).value == 10  # Binary
        
        # Float to int (lossless only)
        assert coercer.coerce(5.0, int).value == 5
        result = coercer.coerce(5.5, int)
        assert not result.valid
        assert "losslessly" in result.errors[0]
        
        # Boolean to int
        assert coercer.coerce(True, int).value == 1
        assert coercer.coerce(False, int).value == 0
    
    def test_float_coercion(self):
        """Test coercing to float."""
        coercer = Coercer()
        
        assert coercer.coerce("3.14", float).value == 3.14
        assert coercer.coerce(42, float).value == 42.0
        assert coercer.coerce(True, float).value == 1.0
    
    def test_boolean_coercion(self):
        """Test coercing to boolean."""
        coercer = Coercer()
        
        # String to bool
        assert coercer.coerce("true", bool).value is True
        assert coercer.coerce("false", bool).value is False
        assert coercer.coerce("yes", bool).value is True
        assert coercer.coerce("no", bool).value is False
        assert coercer.coerce("1", bool).value is True
        assert coercer.coerce("0", bool).value is False
        
        # Invalid string
        result = coercer.coerce("maybe", bool)
        assert not result.valid
        assert "not a valid boolean" in result.errors[0]
        
        # Numbers to bool
        assert coercer.coerce(1, bool).value is True
        assert coercer.coerce(0, bool).value is False
        assert coercer.coerce(3.14, bool).value is True
    
    def test_datetime_coercion(self):
        """Test coercing to datetime."""
        coercer = Coercer()
        
        # String to datetime
        result = coercer.coerce("2024-01-15 10:30:00", datetime)
        assert result.valid
        assert result.value.year == 2024
        assert result.value.month == 1
        assert result.value.day == 15
        
        # Unix timestamp to datetime
        result = coercer.coerce(1705315800, datetime)
        assert result.valid
        assert isinstance(result.value, datetime)
        
        # Invalid format
        result = coercer.coerce("not a date", datetime)
        assert not result.valid
    
    def test_field_type_coercion(self):
        """Test coercing using FieldType enum."""
        coercer = Coercer()
        
        assert coercer.coerce("123", FieldType.INTEGER).value == 123
        assert coercer.coerce(123, FieldType.STRING).value == "123"
        assert coercer.coerce("true", FieldType.BOOLEAN).value is True
    
    def test_coerce_many(self):
        """Test coercing multiple values."""
        coercer = Coercer()
        
        values = {
            "age": "25",
            "height": "5.9",
            "active": "true"
        }
        
        types = {
            "age": int,
            "height": float,
            "active": bool
        }
        
        results = coercer.coerce_many(values, types)
        
        assert results["age"].value == 25
        assert results["height"].value == 5.9
        assert results["active"].value is True


class TestIntegration:
    """Integration tests using multiple components together."""
    
    def test_schema_with_coercion(self):
        """Test schema validation with type coercion."""
        schema = Schema("CoercionSchema")
        schema.field("age", FieldType.INTEGER, required=True)
        schema.field("score", FieldType.FLOAT)
        schema.field("active", FieldType.BOOLEAN)
        
        # Record with wrong types
        record = Record({
            "age": "25",
            "score": "98.5",
            "active": "yes"
        })
        
        # Without coercion should fail
        result = schema.validate(record, coerce=False)
        assert not result.valid
        
        # With coercion should succeed
        result = schema.validate(record, coerce=True)
        assert result.valid
        validated_record = result.value
        assert validated_record.get_value("age") == 25
        assert validated_record.get_value("score") == 98.5
        assert validated_record.get_value("active") is True
    
    def test_complex_validation_scenario(self):
        """Test a realistic validation scenario."""
        # Define user registration schema
        schema = Schema("UserRegistration", strict=True)
        
        # Username: required, unique, alphanumeric, 3-20 chars
        schema.field(
            "username",
            FieldType.STRING,
            required=True,
            constraints=[
                Length(min=3, max=20),
                Pattern(r"^[a-zA-Z0-9_]+$"),
                Unique("username")
            ]
        )
        
        # Email: required, valid format
        schema.field(
            "email",
            FieldType.STRING,
            required=True,
            constraints=[
                Pattern(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
            ]
        )
        
        # Age: optional, 13-120
        schema.field(
            "age",
            FieldType.INTEGER,
            constraints=[Range(min=13, max=120)]
        )
        
        # Terms accepted: required, must be true
        schema.field(
            "terms_accepted",
            FieldType.BOOLEAN,
            required=True,
            constraints=[
                Custom(lambda x: x is True, "Terms must be accepted")
            ]
        )
        
        # Valid registrations
        users = [
            Record({
                "username": "john_doe",
                "email": "john@example.com",
                "age": 25,
                "terms_accepted": True
            }),
            Record({
                "username": "jane_smith",
                "email": "jane@example.org",
                "terms_accepted": True
            })
        ]
        
        results = schema.validate_many(users)
        assert all(r.valid for r in results)
        
        # Invalid registrations - combine with valid users to test uniqueness
        all_users = users + [
            Record({  # Duplicate username
                "username": "john_doe",
                "email": "another@example.com",
                "terms_accepted": True
            }),
            Record({  # Invalid email
                "username": "bob",
                "email": "not-an-email",
                "terms_accepted": True
            }),
            Record({  # Terms not accepted
                "username": "alice",
                "email": "alice@example.com",
                "terms_accepted": False
            })
        ]
        
        all_results = schema.validate_many(all_users)
        assert all_results[0].valid  # First john_doe is valid
        assert all_results[1].valid  # jane_smith is valid
        assert not all_results[2].valid  # Duplicate john_doe
        assert not all_results[3].valid  # Invalid email
        assert not all_results[4].valid  # Terms not accepted
