"""Type coercion with predictable, consistent behavior.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from dataknobs_data.fields import FieldType

from .result import ValidationResult


class Coercer:
    """Type coercion with predictable results.
    
    Always returns ValidationResult, never raises exceptions.
    Provides clear error messages when coercion fails.
    """

    def coerce(
        self,
        value: Any,
        target_type: type | FieldType
    ) -> ValidationResult:
        """Coerce a value to the target type.
        
        Args:
            value: Value to coerce
            target_type: Target type (Python type or FieldType enum)
            
        Returns:
            ValidationResult with coerced value or error
        """
        # Handle None values
        if value is None:
            return ValidationResult.failure(
                None,
                [f"Cannot coerce None to {self._type_name(target_type)}"]
            )

        # Convert FieldType to Python type
        if isinstance(target_type, FieldType):
            target_type = self._field_type_to_python(target_type)

        # If already correct type, return as-is
        if isinstance(value, target_type):
            return ValidationResult.success(value)

        # Attempt coercion
        try:
            coerced = self._coerce_value(value, target_type)
            return ValidationResult.success(coerced)
        except Exception as e:
            return ValidationResult.failure(
                value,
                [f"Cannot coerce {type(value).__name__} to {self._type_name(target_type)}: {e!s}"]
            )

    def _field_type_to_python(self, field_type: FieldType) -> type:
        """Convert FieldType enum to Python type."""
        type_map: dict[FieldType, type] = {
            FieldType.STRING: str,
            FieldType.INTEGER: int,
            FieldType.FLOAT: float,
            FieldType.BOOLEAN: bool,
            FieldType.DATETIME: datetime,
            FieldType.JSON: dict,  # Using dict as primary type for JSON
            FieldType.BINARY: bytes,
        }
        return type_map.get(field_type, object)

    def _type_name(self, target_type: type | FieldType | tuple[type, ...]) -> str:
        """Get readable name for type."""
        if isinstance(target_type, FieldType):
            return target_type.name
        elif isinstance(target_type, tuple):
            # Union type represented as tuple
            return f"Union[{', '.join(t.__name__ if hasattr(t, '__name__') else str(t) for t in target_type)}]"
        elif isinstance(target_type, type):
            return target_type.__name__
        # Fallback for unknown types (for runtime safety)
        return str(target_type)  # type: ignore[unreachable]

    def _coerce_value(self, value: Any, target_type: type) -> Any:
        """Perform the actual coercion.
        
        Args:
            value: Value to coerce
            target_type: Target Python type
            
        Returns:
            Coerced value
            
        Raises:
            Exception: If coercion fails
        """
        # Handle union types (like dict|list for JSON)
        if isinstance(target_type, tuple):
            for t in target_type:
                try:
                    return self._coerce_value(value, t)
                except (ValueError, TypeError):
                    continue
            raise ValueError(f"Could not coerce to any of {target_type}")

        # String coercion
        if target_type == str:
            return str(value)

        # Integer coercion
        elif target_type == int:
            if isinstance(value, str):
                # Remove whitespace and handle common formats
                value = value.strip()
                if value.lower() in ('true', 'false'):
                    return 1 if value.lower() == 'true' else 0
                # Handle hex, octal, binary
                if value.startswith('0x') or value.startswith('0X'):
                    return int(value, 16)
                elif value.startswith('0o') or value.startswith('0O'):
                    return int(value, 8)
                elif value.startswith('0b') or value.startswith('0B'):
                    return int(value, 2)
                return int(value)
            elif isinstance(value, float):
                # Check for data loss
                if value != int(value):
                    raise ValueError(f"Float {value} cannot be losslessly converted to int")
                return int(value)
            elif isinstance(value, bool):
                return 1 if value else 0
            else:
                return int(value)

        # Float coercion
        elif target_type == float:
            if isinstance(value, str):
                value = value.strip()
                if value.lower() in ('true', 'false'):
                    return 1.0 if value.lower() == 'true' else 0.0
                return float(value)
            elif isinstance(value, bool):
                return 1.0 if value else 0.0
            else:
                return float(value)

        # Boolean coercion
        elif target_type == bool:
            if isinstance(value, str):
                value = value.strip().lower()
                if value in ('true', '1', 'yes', 'y', 'on'):
                    return True
                elif value in ('false', '0', 'no', 'n', 'off'):
                    return False
                else:
                    raise ValueError(f"String '{value}' is not a valid boolean")
            elif isinstance(value, (int, float)):
                return bool(value)
            else:
                return bool(value)

        # DateTime coercion
        elif target_type == datetime:
            if isinstance(value, str):
                # Try common datetime formats
                formats = [
                    '%Y-%m-%d %H:%M:%S',
                    '%Y-%m-%dT%H:%M:%S',
                    '%Y-%m-%dT%H:%M:%SZ',
                    '%Y-%m-%dT%H:%M:%S.%f',
                    '%Y-%m-%dT%H:%M:%S.%fZ',
                    '%Y/%m/%d %H:%M:%S',
                    '%Y/%m/%d',
                    '%Y-%m-%d',
                    '%d/%m/%Y',
                    '%d-%m-%Y',
                    '%m/%d/%Y',
                    '%m-%d-%Y',
                ]

                for fmt in formats:
                    try:
                        return datetime.strptime(value, fmt)
                    except ValueError:
                        continue

                # Try parsing as ISO format
                try:
                    return datetime.fromisoformat(value.replace('Z', '+00:00'))
                except (ValueError, AttributeError):
                    pass

                raise ValueError(f"Could not parse datetime from '{value}'")
            elif isinstance(value, (int, float)):
                # Assume Unix timestamp
                return datetime.fromtimestamp(value)
            else:
                raise ValueError(f"Cannot coerce {type(value).__name__} to datetime")

        # Dict coercion (for JSON type)
        elif target_type == dict:
            if isinstance(value, str):
                return json.loads(value)
            elif hasattr(value, '__dict__'):
                return vars(value)
            elif isinstance(value, (list, tuple)):
                # Try to convert list of pairs to dict
                if all(isinstance(item, (list, tuple)) and len(item) == 2 for item in value):
                    return dict(value)
                raise ValueError("Cannot convert list to dict")
            else:
                return dict(value)

        # List coercion (for JSON type)
        elif target_type == list:
            if isinstance(value, str):
                # Try parsing as JSON
                try:
                    result = json.loads(value)
                    if not isinstance(result, list):
                        return [result]
                    return result
                except (json.JSONDecodeError, TypeError):
                    # Split comma-separated values
                    if ',' in value:
                        return [v.strip() for v in value.split(',')]
                    return [value]
            elif isinstance(value, dict):
                # Convert dict to list of key-value pairs
                return list(value.items())
            elif hasattr(value, '__iter__') and not isinstance(value, (str, bytes)):
                return list(value)
            else:
                return [value]

        # Bytes coercion
        elif target_type == bytes:
            if isinstance(value, str):
                return value.encode('utf-8')
            elif isinstance(value, (list, tuple)):
                # Assume list of integers
                return bytes(value)
            elif isinstance(value, int):
                return bytes([value])
            else:
                return bytes(value)

        # Unknown type - attempt direct conversion
        else:
            return target_type(value)

    def coerce_many(
        self,
        values: dict[str, Any],
        types: dict[str, type | FieldType]
    ) -> dict[str, ValidationResult]:
        """Coerce multiple values.
        
        Args:
            values: Dictionary of field names to values
            types: Dictionary of field names to target types
            
        Returns:
            Dictionary of field names to ValidationResults
        """
        results = {}
        for field_name, value in values.items():
            if field_name in types:
                results[field_name] = self.coerce(value, types[field_name])
            else:
                # No type specified, pass through
                results[field_name] = ValidationResult.success(value)
        return results
