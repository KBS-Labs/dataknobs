"""Serialization protocols and utilities for dataknobs packages.

This module provides standard interfaces for objects that can be serialized
to and from dictionaries. This enables consistent serialization patterns
across all dataknobs packages.

The serialization framework supports:
- Type-safe protocols for serializable objects
- Utility functions for serialization/deserialization
- Runtime type checking with isinstance()
- Integration with dataclasses and custom classes

Example:
    ```python
    from dataknobs_common.serialization import Serializable
    from dataclasses import dataclass

    @dataclass
    class User:
        name: str
        email: str

        def to_dict(self) -> dict:
            return {"name": self.name, "email": self.email}

        @classmethod
        def from_dict(cls, data: dict) -> "User":
            return cls(name=data["name"], email=data["email"])

    # Type checking works
    user = User("Alice", "alice@example.com")
    assert isinstance(user, Serializable)  # True

    # Use utilities
    from dataknobs_common.serialization import serialize, deserialize

    data = serialize(user)
    restored = deserialize(User, data)
    ```
"""

from typing import Any, Dict, Protocol, Type, TypeVar, runtime_checkable

from dataknobs_common.exceptions import SerializationError

T = TypeVar("T")


@runtime_checkable
class Serializable(Protocol):
    """Protocol for objects that can be serialized to/from dict.

    Implement this protocol by providing to_dict() and from_dict() methods.
    The @runtime_checkable decorator allows isinstance() checks at runtime.

    Methods:
        to_dict: Convert object to dictionary representation
        from_dict: Create object from dictionary representation

    Example:
        ```python
        class MyClass:
            def __init__(self, value: str):
                self.value = value

            def to_dict(self) -> dict:
                return {"value": self.value}

            @classmethod
            def from_dict(cls, data: dict) -> "MyClass":
                return cls(data["value"])

        obj = MyClass("test")
        isinstance(obj, Serializable)
        # True
        ```
    """

    def to_dict(self) -> Dict[str, Any]:
        """Convert object to dictionary representation.

        Returns:
            Dictionary with serialized data

        Raises:
            SerializationError: If serialization fails
        """
        ...

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """Create object from dictionary representation.

        Args:
            data: Dictionary with serialized data

        Returns:
            Deserialized object instance

        Raises:
            SerializationError: If deserialization fails
        """
        ...


def serialize(obj: Any) -> Dict[str, Any]:
    """Serialize an object to dictionary.

    Convenience function that calls to_dict() with error handling.

    Args:
        obj: Object to serialize (must have to_dict method)

    Returns:
        Serialized dictionary

    Raises:
        SerializationError: If object doesn't support serialization or serialization fails

    Example:
        ```python
        class Point:
            def __init__(self, x: int, y: int):
                self.x, self.y = x, y
            def to_dict(self):
                return {"x": self.x, "y": self.y}

        point = Point(10, 20)
        data = serialize(point)
        # {'x': 10, 'y': 20}
        ```
    """
    if not hasattr(obj, "to_dict"):
        raise SerializationError(
            f"Object of type {type(obj).__name__} is not serializable (missing to_dict method)",
            context={"type": type(obj).__name__, "object": str(obj)},
        )

    try:
        result = obj.to_dict()
        if not isinstance(result, dict):
            raise SerializationError(
                f"to_dict() must return a dict, got {type(result).__name__}",
                context={"type": type(obj).__name__, "result_type": type(result).__name__},
            )
        return result
    except Exception as e:
        if isinstance(e, SerializationError):
            raise
        raise SerializationError(
            f"Failed to serialize {type(obj).__name__}: {e}",
            context={"type": type(obj).__name__, "error": str(e)},
        ) from e


def deserialize(cls: Type[T], data: Dict[str, Any]) -> T:
    """Deserialize dictionary into an object.

    Convenience function that calls from_dict() with error handling.

    Args:
        cls: Class to deserialize into (must have from_dict classmethod)
        data: Dictionary with serialized data

    Returns:
        Deserialized object instance

    Raises:
        SerializationError: If class doesn't support deserialization or deserialization fails

    Example:
        ```python
        class Point:
            def __init__(self, x: int, y: int):
                self.x, self.y = x, y
            @classmethod
            def from_dict(cls, data: dict):
                return cls(data["x"], data["y"])

        data = {"x": 10, "y": 20}
        point = deserialize(Point, data)
        # point.x, point.y
        # (10, 20)
        ```
    """
    if not hasattr(cls, "from_dict"):
        raise SerializationError(
            f"Class {cls.__name__} is not deserializable (missing from_dict classmethod)",
            context={"class": cls.__name__},
        )

    if not isinstance(data, dict):
        raise SerializationError(
            f"Data must be a dict, got {type(data).__name__}",
            context={"class": cls.__name__, "data_type": type(data).__name__},
        )

    try:
        return cls.from_dict(data)
    except Exception as e:
        if isinstance(e, SerializationError):
            raise
        raise SerializationError(
            f"Failed to deserialize {cls.__name__}: {e}",
            context={"class": cls.__name__, "error": str(e), "data": data},
        ) from e


def serialize_list(items: list[Any]) -> list[Dict[str, Any]]:
    """Serialize a list of objects to list of dictionaries.

    Args:
        items: List of serializable objects

    Returns:
        List of serialized dictionaries

    Raises:
        SerializationError: If any item cannot be serialized

    Example:
        ```python
        items = [Point(1, 2), Point(3, 4)]
        data_list = serialize_list(items)
        len(data_list)
        # 2
        ```
    """
    return [serialize(item) for item in items]


def deserialize_list(cls: Type[T], data_list: list[Dict[str, Any]]) -> list[T]:
    """Deserialize a list of dictionaries into objects.

    Args:
        cls: Class to deserialize into
        data_list: List of serialized dictionaries

    Returns:
        List of deserialized objects

    Raises:
        SerializationError: If any item cannot be deserialized

    Example:
        ```python
        data_list = [{"x": 1, "y": 2}, {"x": 3, "y": 4}]
        points = deserialize_list(Point, data_list)
        len(points)
        # 2
        ```
    """
    return [deserialize(cls, data) for data in data_list]


def is_serializable(obj: Any) -> bool:
    """Check if an object is serializable.

    Args:
        obj: Object to check

    Returns:
        True if object has to_dict method

    Example:
        ```python
        class Point:
            def to_dict(self): return {}

        is_serializable(Point())
        # True
        is_serializable("string")
        # False
        ```
    """
    return isinstance(obj, Serializable) or hasattr(obj, "to_dict")


def is_deserializable(cls: Type) -> bool:
    """Check if a class is deserializable.

    Args:
        cls: Class to check

    Returns:
        True if class has from_dict classmethod

    Example:
        ```python
        class Point:
            @classmethod
            def from_dict(cls, data): return cls()

        is_deserializable(Point)
        # True
        is_deserializable(str)
        # False
        ```
    """
    return hasattr(cls, "from_dict")


__all__ = [
    "Serializable",
    "serialize",
    "deserialize",
    "serialize_list",
    "deserialize_list",
    "is_serializable",
    "is_deserializable",
]
