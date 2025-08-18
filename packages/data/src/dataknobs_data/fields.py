import copy
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class FieldType(Enum):
    """Enumeration of supported field types."""

    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    JSON = "json"
    BINARY = "binary"
    TEXT = "text"


@dataclass
class Field:
    """Represents a single field in a record."""

    name: str
    value: Any
    type: FieldType | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Auto-detect type if not provided."""
        if self.type is None:
            self.type = self._detect_type(self.value)

    def _detect_type(self, value: Any) -> FieldType:
        """Detect the field type from the value."""
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

    def copy(self) -> "Field":
        """Create a deep copy of the field."""
        return Field(
            name=self.name,
            value=copy.deepcopy(self.value),
            type=self.type,
            metadata=copy.deepcopy(self.metadata)
        )

    def validate(self) -> bool:
        """Validate that the value matches the field type."""
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

        validator = type_validators.get(self.type)
        if validator:
            return validator(self.value)
        return True

    def convert_to(self, target_type: FieldType) -> "Field":
        """Convert the field to a different type."""
        if self.type == target_type:
            return self

        converters = {
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

        converter_key = (self.type, target_type)
        if converter_key in converters:
            try:
                new_value = converters[converter_key](self.value)
                return Field(
                    name=self.name, value=new_value, type=target_type, metadata=self.metadata.copy()
                )
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"Cannot convert {self.name} from {self.type} to {target_type}: {e}"
                )
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
    def from_dict(cls, data: dict[str, Any]) -> "Field":
        """Create a field from a dictionary representation."""
        field_type = None
        if data.get("type"):
            field_type = FieldType(data["type"])

        return cls(
            name=data["name"],
            value=data["value"],
            type=field_type,
            metadata=data.get("metadata", {}),
        )
