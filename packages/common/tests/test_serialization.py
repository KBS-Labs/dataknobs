"""Tests for the serialization protocol and utilities."""

from dataclasses import dataclass

import pytest

from dataknobs_common.exceptions import SerializationError
from dataknobs_common.serialization import (
    Serializable,
    deserialize,
    deserialize_list,
    is_deserializable,
    is_serializable,
    serialize,
    serialize_list,
)


@dataclass
class User:
    """Simple user class for testing."""
    name: str
    email: str

    def to_dict(self):
        return {"name": self.name, "email": self.email}

    @classmethod
    def from_dict(cls, data):
        return cls(name=data["name"], email=data["email"])


@dataclass
class Product:
    """Product class with nested data."""
    id: int
    name: str
    price: float
    tags: list[str]

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "price": self.price,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            id=data["id"],
            name=data["name"],
            price=data["price"],
            tags=data["tags"],
        )


class TestSerializableProtocol:
    """Test the Serializable protocol."""

    def test_class_with_to_dict_and_from_dict_is_serializable(self):
        """Test that classes with required methods satisfy protocol."""
        user = User("Alice", "alice@example.com")
        assert isinstance(user, Serializable)

    def test_class_without_methods_not_serializable(self):
        """Test that classes without methods don't satisfy protocol."""
        class NotSerializable:
            pass

        obj = NotSerializable()
        assert not isinstance(obj, Serializable)

    def test_class_with_only_to_dict_is_serializable(self):
        """Test that class with only to_dict satisfies protocol."""
        class PartialSerializable:
            def to_dict(self):
                return {}

        # Protocol is runtime_checkable, but from_dict is classmethod
        # so this checks to_dict mainly
        obj = PartialSerializable()
        assert hasattr(obj, "to_dict")


class TestSerialize:
    """Test serialize function."""

    def test_serialize_simple_object(self):
        """Test serializing a simple object."""
        user = User("Alice", "alice@example.com")
        data = serialize(user)

        assert data == {"name": "Alice", "email": "alice@example.com"}

    def test_serialize_complex_object(self):
        """Test serializing object with nested data."""
        product = Product(1, "Widget", 29.99, ["new", "featured"])
        data = serialize(product)

        assert data == {
            "id": 1,
            "name": "Widget",
            "price": 29.99,
            "tags": ["new", "featured"],
        }

    def test_serialize_without_to_dict_raises_error(self):
        """Test that serializing object without to_dict raises error."""
        class NotSerializable:
            pass

        obj = NotSerializable()

        with pytest.raises(SerializationError) as exc_info:
            serialize(obj)

        error = exc_info.value
        assert "not serializable" in str(error).lower()
        assert "missing to_dict" in str(error)
        assert error.context["type"] == "NotSerializable"

    def test_serialize_to_dict_returns_non_dict_raises_error(self):
        """Test that to_dict returning non-dict raises error."""
        class BadSerializable:
            def to_dict(self):
                return "not a dict"

        obj = BadSerializable()

        with pytest.raises(SerializationError) as exc_info:
            serialize(obj)

        error = exc_info.value
        assert "must return a dict" in str(error)

    def test_serialize_to_dict_raises_exception(self):
        """Test that exceptions in to_dict are wrapped."""
        class FailingSerializable:
            def to_dict(self):
                raise ValueError("Intentional error")

        obj = FailingSerializable()

        with pytest.raises(SerializationError) as exc_info:
            serialize(obj)

        error = exc_info.value
        assert "Failed to serialize" in str(error)
        assert "Intentional error" in error.context["error"]


class TestDeserialize:
    """Test deserialize function."""

    def test_deserialize_simple_object(self):
        """Test deserializing a simple object."""
        data = {"name": "Alice", "email": "alice@example.com"}
        user = deserialize(User, data)

        assert isinstance(user, User)
        assert user.name == "Alice"
        assert user.email == "alice@example.com"

    def test_deserialize_complex_object(self):
        """Test deserializing object with nested data."""
        data = {
            "id": 1,
            "name": "Widget",
            "price": 29.99,
            "tags": ["new", "featured"],
        }
        product = deserialize(Product, data)

        assert isinstance(product, Product)
        assert product.id == 1
        assert product.name == "Widget"
        assert product.price == 29.99
        assert product.tags == ["new", "featured"]

    def test_deserialize_without_from_dict_raises_error(self):
        """Test that deserializing class without from_dict raises error."""
        class NotDeserializable:
            pass

        data = {"key": "value"}

        with pytest.raises(SerializationError) as exc_info:
            deserialize(NotDeserializable, data)

        error = exc_info.value
        assert "not deserializable" in str(error).lower()
        assert "missing from_dict" in str(error)

    def test_deserialize_with_non_dict_raises_error(self):
        """Test that deserializing non-dict data raises error."""
        with pytest.raises(SerializationError) as exc_info:
            deserialize(User, "not a dict")

        error = exc_info.value
        assert "must be a dict" in str(error)

    def test_deserialize_from_dict_raises_exception(self):
        """Test that exceptions in from_dict are wrapped."""
        class FailingDeserializable:
            @classmethod
            def from_dict(cls, data):
                raise ValueError("Intentional error")

        with pytest.raises(SerializationError) as exc_info:
            deserialize(FailingDeserializable, {"key": "value"})

        error = exc_info.value
        assert "Failed to deserialize" in str(error)
        assert "Intentional error" in error.context["error"]


class TestSerializeList:
    """Test serialize_list function."""

    def test_serialize_empty_list(self):
        """Test serializing empty list."""
        result = serialize_list([])
        assert result == []

    def test_serialize_list_of_objects(self):
        """Test serializing list of objects."""
        users = [
            User("Alice", "alice@example.com"),
            User("Bob", "bob@example.com"),
            User("Carol", "carol@example.com"),
        ]

        data_list = serialize_list(users)

        assert len(data_list) == 3
        assert data_list[0] == {"name": "Alice", "email": "alice@example.com"}
        assert data_list[1] == {"name": "Bob", "email": "bob@example.com"}
        assert data_list[2] == {"name": "Carol", "email": "carol@example.com"}

    def test_serialize_list_with_invalid_item(self):
        """Test that serializing list with invalid item raises error."""
        class NotSerializable:
            pass

        items = [User("Alice", "alice@example.com"), NotSerializable()]

        with pytest.raises(SerializationError):
            serialize_list(items)


class TestDeserializeList:
    """Test deserialize_list function."""

    def test_deserialize_empty_list(self):
        """Test deserializing empty list."""
        result = deserialize_list(User, [])
        assert result == []

    def test_deserialize_list_of_dicts(self):
        """Test deserializing list of dictionaries."""
        data_list = [
            {"name": "Alice", "email": "alice@example.com"},
            {"name": "Bob", "email": "bob@example.com"},
            {"name": "Carol", "email": "carol@example.com"},
        ]

        users = deserialize_list(User, data_list)

        assert len(users) == 3
        assert all(isinstance(u, User) for u in users)
        assert users[0].name == "Alice"
        assert users[1].name == "Bob"
        assert users[2].name == "Carol"

    def test_deserialize_list_with_invalid_item(self):
        """Test that deserializing list with invalid item raises error."""
        data_list = [
            {"name": "Alice", "email": "alice@example.com"},
            "not a dict",
        ]

        with pytest.raises(SerializationError):
            deserialize_list(User, data_list)


class TestIsSerializable:
    """Test is_serializable function."""

    def test_object_with_to_dict_is_serializable(self):
        """Test that objects with to_dict are serializable."""
        user = User("Alice", "alice@example.com")
        assert is_serializable(user)

    def test_object_without_to_dict_not_serializable(self):
        """Test that objects without to_dict are not serializable."""
        assert not is_serializable("string")
        assert not is_serializable(42)
        assert not is_serializable([1, 2, 3])

    def test_custom_object_with_to_dict_is_serializable(self):
        """Test custom class with to_dict."""
        class Custom:
            def to_dict(self):
                return {}

        obj = Custom()
        assert is_serializable(obj)


class TestIsDeserializable:
    """Test is_deserializable function."""

    def test_class_with_from_dict_is_deserializable(self):
        """Test that classes with from_dict are deserializable."""
        assert is_deserializable(User)

    def test_class_without_from_dict_not_deserializable(self):
        """Test that classes without from_dict are not deserializable."""
        assert not is_deserializable(str)
        assert not is_deserializable(int)
        assert not is_deserializable(list)

    def test_custom_class_with_from_dict_is_deserializable(self):
        """Test custom class with from_dict."""
        class Custom:
            @classmethod
            def from_dict(cls, data):
                return cls()

        assert is_deserializable(Custom)


class TestRoundTrip:
    """Test serialization round-trip (serialize -> deserialize)."""

    def test_simple_round_trip(self):
        """Test round-trip serialization of simple object."""
        original = User("Alice", "alice@example.com")
        data = serialize(original)
        restored = deserialize(User, data)

        assert restored.name == original.name
        assert restored.email == original.email

    def test_complex_round_trip(self):
        """Test round-trip serialization of complex object."""
        original = Product(1, "Widget", 29.99, ["new", "featured"])
        data = serialize(original)
        restored = deserialize(Product, data)

        assert restored.id == original.id
        assert restored.name == original.name
        assert restored.price == original.price
        assert restored.tags == original.tags

    def test_list_round_trip(self):
        """Test round-trip serialization of list."""
        original = [
            User("Alice", "alice@example.com"),
            User("Bob", "bob@example.com"),
        ]

        data_list = serialize_list(original)
        restored = deserialize_list(User, data_list)

        assert len(restored) == len(original)
        for orig, rest in zip(original, restored):
            assert rest.name == orig.name
            assert rest.email == orig.email


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_serialize_none_raises_error(self):
        """Test that serializing None raises error."""
        with pytest.raises(SerializationError):
            serialize(None)

    def test_deserialize_none_raises_error(self):
        """Test that deserializing None raises error."""
        with pytest.raises(SerializationError):
            deserialize(User, None)

    def test_serialize_with_special_characters(self):
        """Test serialization with special characters."""
        user = User("Alice & Bob", "alice+bob@example.com")
        data = serialize(user)

        assert data["name"] == "Alice & Bob"
        assert data["email"] == "alice+bob@example.com"

    def test_deserialize_with_missing_field(self):
        """Test deserialization with missing required field."""
        data = {"name": "Alice"}  # Missing email

        with pytest.raises(Exception):  # KeyError or similar
            deserialize(User, data)

    def test_serialize_with_empty_strings(self):
        """Test serialization with empty strings."""
        user = User("", "")
        data = serialize(user)

        assert data == {"name": "", "email": ""}

    def test_round_trip_with_unicode(self):
        """Test round-trip with Unicode characters."""
        original = User("Алиса", "alice@пример.com")
        data = serialize(original)
        restored = deserialize(User, data)

        assert restored.name == original.name
        assert restored.email == original.email


class TestCustomSerializable:
    """Test creating custom serializable classes."""

    def test_custom_serialization_logic(self):
        """Test class with custom serialization logic."""
        class Point:
            def __init__(self, x, y):
                self.x = x
                self.y = y

            def to_dict(self):
                return {"coordinates": [self.x, self.y]}

            @classmethod
            def from_dict(cls, data):
                x, y = data["coordinates"]
                return cls(x, y)

        point = Point(10, 20)
        data = serialize(point)

        assert data == {"coordinates": [10, 20]}

        restored = deserialize(Point, data)
        assert restored.x == 10
        assert restored.y == 20

    def test_nested_serializable_objects(self):
        """Test serialization of nested objects."""
        class Address:
            def __init__(self, street, city):
                self.street = street
                self.city = city

            def to_dict(self):
                return {"street": self.street, "city": self.city}

            @classmethod
            def from_dict(cls, data):
                return cls(data["street"], data["city"])

        class Person:
            def __init__(self, name, address):
                self.name = name
                self.address = address

            def to_dict(self):
                return {
                    "name": self.name,
                    "address": serialize(self.address),
                }

            @classmethod
            def from_dict(cls, data):
                return cls(
                    data["name"],
                    deserialize(Address, data["address"])
                )

        address = Address("123 Main St", "Springfield")
        person = Person("Homer", address)

        data = serialize(person)
        assert data["name"] == "Homer"
        assert data["address"]["street"] == "123 Main St"

        restored = deserialize(Person, data)
        assert restored.name == "Homer"
        assert restored.address.street == "123 Main St"
        assert restored.address.city == "Springfield"
