"""Tests for Field and FieldType classes."""

from datetime import datetime

import pytest

from dataknobs_data import Field, FieldType
from dataknobs_data.fields import field_type_backends, register_field_class


class TestFieldType:
    """Test FieldType enumeration."""

    def test_field_type_values(self):
        """Test that all field types have correct values."""
        assert FieldType.STRING.value == "string"
        assert FieldType.INTEGER.value == "integer"
        assert FieldType.FLOAT.value == "float"
        assert FieldType.BOOLEAN.value == "boolean"
        assert FieldType.DATETIME.value == "datetime"
        assert FieldType.JSON.value == "json"
        assert FieldType.BINARY.value == "binary"
        assert FieldType.TEXT.value == "text"


class TestField:
    """Test Field class."""

    def test_field_creation_with_type(self):
        """Test creating a field with explicit type."""
        field = Field(name="age", value=25, type=FieldType.INTEGER)
        assert field.name == "age"
        assert field.value == 25
        assert field.type == FieldType.INTEGER
        assert field.metadata == {}

    def test_field_creation_with_metadata(self):
        """Test creating a field with metadata."""
        metadata = {"unit": "years", "required": True}
        field = Field(name="age", value=25, metadata=metadata)
        assert field.metadata == metadata

    def test_field_type_auto_detection(self):
        """Test automatic type detection."""
        # Test various types
        test_cases = [
            (None, FieldType.STRING),
            (True, FieldType.BOOLEAN),
            (False, FieldType.BOOLEAN),
            (42, FieldType.INTEGER),
            (3.14, FieldType.FLOAT),
            ("hello", FieldType.STRING),
            ("x" * 1001, FieldType.TEXT),
            (datetime.now(), FieldType.DATETIME),
            ({"key": "value"}, FieldType.JSON),
            ([1, 2, 3], FieldType.JSON),
            (b"binary data", FieldType.BINARY),
        ]

        for value, expected_type in test_cases:
            field = Field(name="test", value=value)
            assert field.type == expected_type, f"Failed for value {value}"

    def test_field_validation(self):
        """Test field validation."""
        # Valid fields
        assert Field(name="str", value="text", type=FieldType.STRING).validate()
        assert Field(name="int", value=42, type=FieldType.INTEGER).validate()
        assert Field(name="float", value=3.14, type=FieldType.FLOAT).validate()
        assert Field(name="bool", value=True, type=FieldType.BOOLEAN).validate()
        assert Field(name="dt", value=datetime.now(), type=FieldType.DATETIME).validate()
        assert Field(name="json", value={"key": "value"}, type=FieldType.JSON).validate()
        assert Field(name="bin", value=b"data", type=FieldType.BINARY).validate()
        assert Field(name="text", value="long text", type=FieldType.TEXT).validate()

        # None is valid for any type
        assert Field(name="null", value=None, type=FieldType.INTEGER).validate()

        # Invalid fields
        assert not Field(name="bad_int", value="text", type=FieldType.INTEGER).validate()
        assert not Field(name="bad_bool", value=1, type=FieldType.BOOLEAN).validate()
        assert not Field(name="bad_dt", value="2024-01-01", type=FieldType.DATETIME).validate()

    def test_field_conversion(self):
        """Test field type conversion."""
        # Integer to string
        field = Field(name="num", value=42, type=FieldType.INTEGER)
        converted = field.convert_to(FieldType.STRING)
        assert converted.value == "42"
        assert converted.type == FieldType.STRING

        # String to integer
        field = Field(name="str", value="123", type=FieldType.STRING)
        converted = field.convert_to(FieldType.INTEGER)
        assert converted.value == 123
        assert converted.type == FieldType.INTEGER

        # Boolean to string
        field = Field(name="bool", value=True, type=FieldType.BOOLEAN)
        converted = field.convert_to(FieldType.STRING)
        assert converted.value == "true"

        # String to boolean
        field = Field(name="str", value="true", type=FieldType.STRING)
        converted = field.convert_to(FieldType.BOOLEAN)
        assert converted.value is True

        field = Field(name="str", value="false", type=FieldType.STRING)
        converted = field.convert_to(FieldType.BOOLEAN)
        assert converted.value is False

        # Invalid conversion should raise error
        field = Field(name="str", value="not_a_number", type=FieldType.STRING)
        with pytest.raises(ValueError):
            field.convert_to(FieldType.INTEGER)

    def test_field_to_dict(self):
        """Test converting field to dictionary."""
        metadata = {"unit": "kg"}
        field = Field(name="weight", value=70.5, type=FieldType.FLOAT, metadata=metadata)

        dict_repr = field.to_dict()
        assert dict_repr == {"name": "weight", "value": 70.5, "type": "float", "metadata": metadata}

    def test_field_from_dict(self):
        """Test creating field from dictionary."""
        data = {"name": "height", "value": 180, "type": "integer", "metadata": {"unit": "cm"}}

        field = Field.from_dict(data)
        assert field.name == "height"
        assert field.value == 180
        assert field.type == FieldType.INTEGER
        assert field.metadata == {"unit": "cm"}

        # Test without type (auto-detection)
        data_no_type = {"name": "age", "value": 25, "metadata": {}}
        field = Field.from_dict(data_no_type)
        assert field.type == FieldType.INTEGER


class TestFieldTypeRegistry:
    """The dispatch ``Field.from_dict`` performs is registered, not hardcoded."""

    def test_builtin_vector_types_are_registered(self):
        """dataknobs_data registers VectorField for both vector types on import."""
        from dataknobs_data import VectorField

        assert field_type_backends.get("vector") is VectorField
        assert field_type_backends.get("sparse_vector") is VectorField

    def test_unregistered_type_builds_the_class_asked_for(self):
        """A type with no registered class is built as ``cls``, not refused."""
        field = Field.from_dict({"name": "n", "value": "v", "type": "string"})

        assert type(field) is Field

    def test_consumer_subclass_is_reached_by_registering_it(self):
        """A consumer's own Field subclass is dispatched to without a code change.

        The extension point exists because the dispatch used to be a
        hardcoded pair of enum members, which no consumer could extend.
        """

        class TaggedField(Field):
            @classmethod
            def from_dict(cls, data):
                built = cls(
                    name=data["name"],
                    value=data["value"],
                    type=FieldType(data["type"]) if data.get("type") else None,
                    metadata=data.get("metadata", {}),
                )
                built.tag = "tagged"
                return built

        previous = field_type_backends.get_optional(FieldType.BINARY.value)
        register_field_class(FieldType.BINARY, TaggedField)
        try:
            field = Field.from_dict({"name": "blob", "value": b"x", "type": "binary"})

            assert isinstance(field, TaggedField)
            assert field.tag == "tagged"
        finally:
            if previous is None:
                field_type_backends.unregister(FieldType.BINARY.value)
            else:
                register_field_class(FieldType.BINARY, previous)

    def test_subclass_from_dict_does_not_recurse_into_itself(self):
        """Calling ``from_dict`` on the registered class builds it directly."""
        from dataknobs_data import VectorField

        field = VectorField.from_dict(
            {"name": "e", "value": [0.1, 0.2], "type": "vector", "metadata": {}}
        )

        assert isinstance(field, VectorField)
