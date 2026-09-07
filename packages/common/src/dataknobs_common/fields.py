"""Field type definitions and metadata for structured data records.

This module defines field types, validation, and metadata structures used by
Record objects to represent typed data fields with constraints and transformations.

It holds no vector type. ``VectorField`` needs ``numpy`` at runtime and
``dataknobs-common`` declares no dependencies, so it lives in
``dataknobs_data.fields`` and reaches ``Field.from_dict`` by registering itself
in :data:`field_type_backends`.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Self

from dataknobs_common.registry import Registry

if TYPE_CHECKING:
    from collections.abc import Callable
else:
    from typing import Callable


class FieldType(Enum):
    """Enumeration of supported field types.

    Defines the data types that can be stored in Record fields. Field types enable
    type validation, schema enforcement, and backend-specific optimizations.

    Attributes:
        STRING: Short text (< 1000 chars)
        TEXT: Long text content
        INTEGER: Whole numbers
        FLOAT: Decimal numbers
        BOOLEAN: True/False values
        DATETIME: Date and time values
        JSON: Structured JSON data (dicts, lists)
        BINARY: Binary data (bytes)
        VECTOR: Dense vector embeddings for similarity search
        SPARSE_VECTOR: Sparse vector representations

    Example:
        ```python
        from dataknobs_common import Field, FieldType

        # Create typed fields
        name_field = Field(name="name", value="Alice", type=FieldType.STRING)
        age_field = Field(name="age", value=30, type=FieldType.INTEGER)
        tags_field = Field(name="tags", value=["python", "data"], type=FieldType.JSON)

        # Auto-detection (type is inferred from value)
        auto_field = Field(name="score", value=95.5)  # Auto-detected as FLOAT
        ```
    """

    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    JSON = "json"
    BINARY = "binary"
    TEXT = "text"
    VECTOR = "vector"
    SPARSE_VECTOR = "sparse_vector"


@dataclass
class Field:
    """Represents a single field in a record.

    A Field encapsulates a named value along with its type and optional metadata.
    Field types are automatically detected if not explicitly provided.

    Attributes:
        name: The field name
        value: The field value (can be any Python type)
        type: The field type (auto-detected if None)
        metadata: Optional metadata dictionary

    Example:
        ```python
        from dataknobs_common import Field, FieldType

        # Auto-detected type
        name = Field(name="name", value="Alice")
        print(name.type)  # FieldType.STRING

        # Explicit type
        score = Field(name="score", value=95.5, type=FieldType.FLOAT)

        # With metadata
        vector = Field(
            name="embedding",
            value=[0.1, 0.2, 0.3],
            type=FieldType.VECTOR,
            metadata={"dimensions": 3, "model": "text-embedding-3-small"}
        )

        # Validation
        is_valid = name.validate()  # True

        # Type conversion
        str_score = score.convert_to(FieldType.STRING)
        print(str_score.value)  # "95.5"
        ```
    """

    name: str
    value: Any
    type: FieldType | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Auto-detect type if not provided."""
        if self.type is None:
            self.type = self._detect_type(self.value)

    def _detect_type(self, value: Any) -> FieldType:
        """Detect the field type from the value.

        Args:
            value: The value to analyze

        Returns:
            The detected FieldType

        Example:
            ```python
            field = Field(name="data", value=[1, 2, 3])
            detected_type = field._detect_type([1, 2, 3])
            print(detected_type)  # FieldType.JSON
            ```
        """
        if value is None:
            return FieldType.STRING
        elif isinstance(value, bool):
            return FieldType.BOOLEAN
        elif isinstance(value, int):
            return FieldType.INTEGER
        elif isinstance(value, float):
            return FieldType.FLOAT
        elif isinstance(value, datetime):
            return FieldType.DATETIME
        elif isinstance(value, (dict, list)):
            return FieldType.JSON
        elif isinstance(value, bytes):
            return FieldType.BINARY
        elif isinstance(value, str):
            if len(value) > 1000:
                return FieldType.TEXT
            return FieldType.STRING
        else:
            return FieldType.JSON

    def copy(self) -> Self:
        """Create a deep copy of the field, keeping its class.

        Copied through the instance rather than reconstructed as a ``Field``,
        so a subclass keeps both its type and whatever attributes it added
        without having to override this. The reconstructing form named its own
        class in its return statement, which silently downgraded a
        ``VectorField`` to a ``Field`` and dropped the four attributes the
        subclass carries; an override per subclass would only move that
        obligation onto every future one.
        """
        return copy.deepcopy(self)

    def validate(self) -> bool:
        """Validate that the value matches the field type.

        Returns:
            True if the value is valid for the field type, False otherwise

        Example:
            ```python
            # Valid field
            age = Field(name="age", value=30, type=FieldType.INTEGER)
            print(age.validate())  # True

            # Invalid field (wrong type for value)
            bad_field = Field(name="count", value="not a number", type=FieldType.INTEGER)
            print(bad_field.validate())  # False
            ```
        """
        if self.value is None:
            return True

        type_validators = {
            FieldType.STRING: lambda v: isinstance(v, str),
            FieldType.INTEGER: lambda v: isinstance(v, int) and not isinstance(v, bool),
            FieldType.FLOAT: lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
            FieldType.BOOLEAN: lambda v: isinstance(v, bool),
            FieldType.DATETIME: lambda v: isinstance(v, datetime),
            FieldType.JSON: lambda v: isinstance(v, (dict, list)),
            FieldType.BINARY: lambda v: isinstance(v, bytes),
            FieldType.TEXT: lambda v: isinstance(v, str),
        }

        if self.type is None:
            return True
        validator = type_validators.get(self.type)
        if validator:
            return validator(self.value)
        return True

    def convert_to(self, target_type: FieldType) -> Field:
        """Convert the field to a different type.

        Args:
            target_type: The target FieldType to convert to

        Returns:
            A new Field with the converted value and type

        Raises:
            ValueError: If conversion is not possible or fails

        Example:
            ```python
            # Integer to string
            age = Field(name="age", value=30, type=FieldType.INTEGER)
            age_str = age.convert_to(FieldType.STRING)
            print(age_str.value)  # "30"

            # String to integer
            count = Field(name="count", value="42", type=FieldType.STRING)
            count_int = count.convert_to(FieldType.INTEGER)
            print(count_int.value)  # 42
            ```
        """
        if self.type == target_type:
            return self

        converters: dict[tuple[FieldType, FieldType], Callable[[Any], Any]] = {
            (FieldType.INTEGER, FieldType.STRING): str,
            (FieldType.INTEGER, FieldType.FLOAT): float,
            (FieldType.FLOAT, FieldType.STRING): str,
            (FieldType.FLOAT, FieldType.INTEGER): int,
            (FieldType.BOOLEAN, FieldType.STRING): lambda v: "true" if v else "false",
            (FieldType.BOOLEAN, FieldType.INTEGER): int,
            (FieldType.STRING, FieldType.INTEGER): int,
            (FieldType.STRING, FieldType.FLOAT): float,
            (FieldType.STRING, FieldType.BOOLEAN): lambda v: v.lower() in ("true", "1", "yes"),
            (FieldType.STRING, FieldType.TEXT): lambda v: v,
            (FieldType.TEXT, FieldType.STRING): lambda v: v,
        }

        if self.type is None:
            raise ValueError(f"Cannot convert {self.name} from None to {target_type}")

        converter_key = (self.type, target_type)
        if converter_key in converters:
            try:
                converter = converters[converter_key]
                new_value = converter(self.value)
                return Field(
                    name=self.name, value=new_value, type=target_type, metadata=self.metadata.copy()
                )
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"Cannot convert {self.name} from {self.type} to {target_type}: {e}"
                ) from e
        else:
            raise ValueError(f"No converter available from {self.type} to {target_type}")

    def to_dict(self) -> dict[str, Any]:
        """Convert the field to a dictionary representation."""
        return {
            "name": self.name,
            "value": self.value,
            "type": self.type.value if self.type else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Field:
        """Create a field from a dictionary representation.

        The class built for a given ``type`` comes from
        :data:`field_type_backends`, so a subclass is reached by having
        registered itself rather than by being named here. A type with no
        registered class builds ``cls``.
        """
        field_type = None
        if data.get("type"):
            field_type = FieldType(data["type"])

        registered = (
            field_type_backends.get_optional(field_type.value) if field_type is not None else None
        )
        if registered is not None and registered is not cls:
            return registered.from_dict(data)

        return cls(
            name=data["name"],
            value=data["value"],
            type=field_type,
            metadata=data.get("metadata", {}),
        )


field_type_backends: Registry[type[Field]] = Registry(
    name="field_type_backends",
)
"""Registry of :class:`Field` subclasses, keyed by :attr:`FieldType.value`.

:meth:`Field.from_dict` consults it to decide which class to build, so a
subclass is reached by being registered rather than by the base class naming
it. It follows the module-level backend-registry pattern
:data:`~dataknobs_common.locks.lock_backends` and
:data:`~dataknobs_common.resolver.resolver_backends` use, on the plain
:class:`~dataknobs_common.registry.Registry` rather than
:class:`~dataknobs_common.registry.PluginRegistry`: what is stored is a class
to call ``from_dict`` on, so there is no config to route and no lazy
construction to buy.

A type with no entry builds the class ``from_dict`` was called on, which is
the honest answer where the subclass is not importable: ``dataknobs_common``
cannot construct a ``VectorField``, because that class needs ``numpy`` and
``dataknobs-common`` declares no dependencies.
"""


def register_field_class(field_type: FieldType, field_class: type[Field]) -> None:
    """Register the class :meth:`Field.from_dict` builds for one field type.

    Re-registering a type overwrites the prior class, so a consumer can
    substitute their own subclass for a built-in one. Registration happens at
    import of the module defining the subclass, so the substituting import
    must come after the one it replaces.

    Args:
        field_type: The field type this class handles
        field_class: The ``Field`` subclass to build for it
    """
    field_type_backends.register(field_type.value, field_class, allow_overwrite=True)
