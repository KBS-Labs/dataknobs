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

import dataclasses
import logging
from typing import Any, Dict, Protocol, Type, TypeVar, runtime_checkable

from dataknobs_common.exceptions import SerializationError

logger = logging.getLogger(__name__)

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


def sanitize_for_json(
    value: Any,
    on_drop: str = "silent",
    _path: str = "",
    _dropped: list[str] | None = None,
) -> Any:
    """Recursively ensure a value is JSON-serializable.

    Converts known structured types (dataclasses, Serializable objects) to
    dicts and drops anything that cannot be represented in JSON. Designed
    for wizard state data that may contain a mix of serializable primitives,
    dataclass instances, Serializable objects, and live runtime objects.

    Conversion rules (in order):
    - ``None``, ``bool``, ``int``, ``float``, ``str`` → pass through
    - ``dict`` → recurse on values; drop entries whose values are not convertible
    - ``list`` / ``tuple`` → recurse on elements; filter out non-convertible items
    - dataclass instance → field-by-field recursive sanitization
    - Object with ``to_dict()`` → call ``to_dict()``
    - Anything else → dropped (behaviour controlled by *on_drop*)

    Args:
        value: Any Python value to sanitize.
        on_drop: What to do when a non-serializable value is encountered.
            ``"silent"`` (default) — log at DEBUG (backward-compatible).
            ``"warn"`` — log at WARNING with the key path.
            ``"error"`` — collect all dropped paths and raise
            :class:`SerializationError` after traversal.
        _path: Internal — dot-delimited key path for diagnostics.
        _dropped: Internal — accumulator for ``on_drop="error"`` mode.

    Returns:
        A JSON-safe version of *value*.

    Raises:
        SerializationError: When *on_drop* is ``"error"`` and at least one
            non-serializable value is encountered.
    """
    # For "error" mode the top-level caller owns the accumulator.
    is_top_level = _dropped is None and on_drop == "error"
    if is_top_level:
        _dropped = []

    result = _sanitize_recursive(value, on_drop, _path, _dropped)

    if is_top_level and _dropped:
        raise SerializationError(
            f"Non-serializable values at: {', '.join(_dropped)}",
            context={"dropped_paths": _dropped},
        )

    return result


def _sanitize_recursive(
    value: Any,
    on_drop: str,
    _path: str,
    _dropped: list[str] | None,
) -> Any:
    """Inner recursive traversal for :func:`sanitize_for_json`."""
    # JSON primitives
    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    # Dicts: recurse on values
    if isinstance(value, dict):
        result = {}
        for k, v in value.items():
            child_path = f"{_path}.{k}" if _path else str(k)
            sanitized = _sanitize_recursive(v, on_drop, child_path, _dropped)
            if sanitized is not _SENTINEL:
                result[k] = sanitized
        return result

    # Lists/tuples: recurse on elements, filter non-convertible
    if isinstance(value, (list, tuple)):
        items = []
        for i, item in enumerate(value):
            child_path = f"{_path}[{i}]"
            sanitized = _sanitize_recursive(item, on_drop, child_path, _dropped)
            if sanitized is not _SENTINEL:
                items.append(sanitized)
        return items

    # Dataclass instances → field-by-field recursive sanitization.
    # We intentionally avoid ``dataclasses.asdict()`` here because it uses
    # ``copy.deepcopy`` on every field value, which crashes on non-picklable
    # objects (e.g., ``asyncio.Task`` stored in wizard data snapshots).
    # Field-by-field recursion is both safer and more thorough — each field
    # value passes through the full sanitize logic, so nested non-serializable
    # values are dropped cleanly rather than causing a crash.
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        result = {}
        for f in dataclasses.fields(value):
            child_path = f"{_path}.{f.name}" if _path else f.name
            sanitized = _sanitize_recursive(
                getattr(value, f.name), on_drop, child_path, _dropped
            )
            if sanitized is not _SENTINEL:
                result[f.name] = sanitized
        return result

    # Serializable protocol (has to_dict)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            dict_result = value.to_dict()
            if isinstance(dict_result, dict):
                return _sanitize_recursive(dict_result, on_drop, _path, _dropped)
        except Exception:
            logger.debug(
                "to_dict() failed for %s, dropping", type(value).__name__
            )

    # Not convertible — handle according to on_drop mode
    path_label = _path or "<root>"
    type_name = type(value).__name__

    if on_drop == "warn":
        logger.warning(
            "Dropping non-serializable '%s' (type=%s)", path_label, type_name
        )
    elif on_drop == "error" and _dropped is not None:
        _dropped.append(f"{path_label} (type={type_name})")
    else:
        logger.debug(
            "Dropping non-serializable value of type %s", type_name
        )

    return _SENTINEL


def validate_json_safe(value: Any, _path: str = "") -> list[str]:
    """Return paths of non-serializable values without modifying the input.

    Uses the same traversal logic as :func:`sanitize_for_json` but is
    read-only — the input is never modified.

    Args:
        value: Any Python value to check.
        _path: Internal — dot-delimited key path for diagnostics.

    Returns:
        List of key paths to non-serializable values.  An empty list
        means *value* is fully JSON-safe.
    """
    # JSON primitives
    if value is None or isinstance(value, (bool, int, float, str)):
        return []

    paths: list[str] = []

    if isinstance(value, dict):
        for k, v in value.items():
            child_path = f"{_path}.{k}" if _path else str(k)
            paths.extend(validate_json_safe(v, child_path))
        return paths

    if isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            paths.extend(validate_json_safe(item, f"{_path}[{i}]"))
        return paths

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        for f in dataclasses.fields(value):
            child_path = f"{_path}.{f.name}" if _path else f.name
            paths.extend(validate_json_safe(getattr(value, f.name), child_path))
        return paths

    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            dict_result = value.to_dict()
            if isinstance(dict_result, dict):
                return validate_json_safe(dict_result, _path)
        except Exception:
            pass

    # Not convertible
    path_label = _path or "<root>"
    return [f"{path_label} (type={type(value).__name__})"]


# Internal sentinel to distinguish "value was dropped" from "value is None"
_SENTINEL = object()


__all__ = [
    "Serializable",
    "sanitize_for_json",
    "validate_json_safe",
    "serialize",
    "deserialize",
    "serialize_list",
    "deserialize_list",
    "is_serializable",
    "is_deserializable",
]
