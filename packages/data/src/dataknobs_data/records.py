"""Structured data records with typed fields and metadata.

This module defines the Record class for representing structured data with
typed fields, validation, and conversion utilities for database operations.
"""

from __future__ import annotations

import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from .fields import Field, FieldType

if TYPE_CHECKING:
    from collections.abc import Iterator


@dataclass
class Record:
    """Represents a structured data record with fields and metadata.

    The record ID can be accessed via the `id` property, which:
    - Returns the storage_id if set (database-assigned ID)
    - Falls back to user-defined 'id' field if present
    - Returns None if no ID is available

    This separation allows records to have both:
    - A user-defined 'id' field as part of their data
    - A system-assigned storage_id for database operations

    Example:
        ```python
        from dataknobs_data import Record, Field, FieldType

        # Create record from dict
        record = Record({"name": "Alice", "age": 30, "email": "alice@example.com"})

        # Access field values
        print(record.get_value("name"))  # "Alice"
        print(record["age"])  # 30
        print(record.name)  # "Alice" (attribute access)

        # Set field values
        record.set_value("age", 31)
        record["city"] = "New York"

        # Work with metadata
        record.metadata["source"] = "user_input"

        # Convert to dict
        data = record.to_dict()  # {"name": "Alice", "age": 31, "email": "...", "city": "..."}
        ```
    """

    fields: OrderedDict[str, Field] = field(default_factory=OrderedDict)
    metadata: dict[str, Any] = field(default_factory=dict)
    _id: str | None = field(default=None, repr=False)  # Deprecated, use storage_id
    _storage_id: str | None = field(default=None, repr=False)

    def __init__(
        self,
        data: dict[str, Any] | OrderedDict[str, Field] | None = None,
        metadata: dict[str, Any] | None = None,
        id: str | None = None,
        storage_id: str | None = None,
    ):
        """Initialize a record from various data formats.

        Args:
            data: Can be a dict of field names to values, or an OrderedDict of Field objects
            metadata: Optional metadata for the record
            id: Optional unique identifier for the record (deprecated, use storage_id)
            storage_id: Optional storage system identifier for the record

        Example:
            ```python
            # From simple dict
            record = Record({"name": "Alice", "age": 30})

            # With metadata
            record = Record(
                data={"name": "Bob"},
                metadata={"source": "api", "timestamp": "2024-01-01"}
            )

            # With storage_id
            record = Record(
                data={"name": "Charlie"},
                storage_id="550e8400-e29b-41d4-a716-446655440000"
            )
            ```
        """
        self.metadata = metadata or {}
        self.fields = OrderedDict()
        self._id = id  # Deprecated
        self._storage_id = storage_id or id  # Use storage_id if provided, fall back to id

        # Process data first to populate fields
        if data:
            if isinstance(data, OrderedDict) and all(
                isinstance(v, Field) for v in data.values()
            ):
                self.fields = data
            else:
                for key, value in data.items():
                    if isinstance(value, Field):
                        # Ensure the field has the correct name
                        if value.name is None or value.name == "embedding":
                            value.name = key
                        self.fields[key] = value
                    else:
                        self.fields[key] = Field(name=key, value=value)

        # Now check for ID from various sources if not explicitly provided
        if self._id is None:
            # Check metadata
            if "id" in self.metadata:
                self._id = str(self.metadata["id"])
            # Check fields for id
            elif "id" in self.fields:
                value = self.get_value("id")
                if value is not None:
                    self._id = str(value)
                    # Sync to metadata
                    self.metadata["id"] = self._id
            # Check fields for record_id
            elif "record_id" in self.fields:
                value = self.get_value("record_id")
                if value is not None:
                    self._id = str(value)
                    # Sync to metadata
                    self.metadata["id"] = self._id

    @property
    def storage_id(self) -> str | None:
        """Get the storage system ID (database-assigned ID)."""
        return self._storage_id

    @storage_id.setter
    def storage_id(self, value: str | None) -> None:
        """Set the storage system ID."""
        self._storage_id = value
        # Also update _id for backwards compatibility
        self._id = value

    @property
    def id(self) -> str | None:
        """Get the record ID.

        Priority order:
        1. Storage ID (database-assigned) if set
        2. User-defined 'id' field value
        3. Metadata 'id' (for backwards compatibility)
        4. record_id field (common in DataFrames)

        Returns the first ID found, or None if no ID is present.
        """
        # 1. Prefer storage ID (database-assigned)
        if self._storage_id is not None:
            return self._storage_id

        # 2. Fall back to legacy _id if set
        if self._id is not None:
            return self._id

        # 3. Check for 'id' field in user data
        if "id" in self.fields:
            value = self.get_value("id")
            if value is not None:
                return str(value)

        # 4. Check metadata (backwards compatibility)
        if "id" in self.metadata:
            return str(self.metadata["id"])

        # 5. Check for 'record_id' field (common in DataFrames)
        if "record_id" in self.fields:
            value = self.get_value("record_id")
            if value is not None:
                return str(value)

        return None

    @id.setter
    def id(self, value: str | None) -> None:
        """Set the record ID.

        This sets the storage_id, which is the database-assigned ID.
        It does NOT modify user data fields.
        """
        self._storage_id = value
        self._id = value  # Backwards compatibility

        # Update metadata for backward compatibility
        if value is not None:
            self.metadata["id"] = value
        elif "id" in self.metadata:
            del self.metadata["id"]

    def generate_id(self) -> str:
        """Generate and set a new UUID for this record.

        Returns:
            The generated UUID string
        """
        new_id = str(uuid.uuid4())
        self.id = new_id
        return new_id

    def get_user_id(self) -> str | None:
        """Get the user-defined ID field value (not the storage ID).
        
        This explicitly returns the value of the 'id' field in the record's data,
        ignoring any storage_id that may be set.
        
        Returns:
            The value of the 'id' field if present, None otherwise
        """
        if "id" in self.fields:
            value = self.get_value("id")
            if value is not None:
                return str(value)
        return None

    def has_storage_id(self) -> bool:
        """Check if this record has a storage system ID assigned.
        
        Returns:
            True if storage_id is set, False otherwise
        """
        return self._storage_id is not None

    def get_field(self, name: str) -> Field | None:
        """Get a field by name."""
        return self.fields.get(name)

    def get_value(self, name: str, default: Any = None) -> Any:
        """Get a field's value by name, supporting dot-notation for nested paths.

        Args:
            name: Field name or dot-notation path (e.g., "metadata.type")
            default: Default value if field not found

        Returns:
            The field value or default

        Example:
            ```python
            record = Record({
                "name": "Alice",
                "config": {"timeout": 30, "retries": 3}
            })

            # Simple field access
            name = record.get_value("name")  # "Alice"

            # Nested path access
            timeout = record.get_value("config.timeout")  # 30

            # With default
            missing = record.get_value("missing_field", "default")  # "default"
            ```
        """
        # Check if this is a nested path
        if "." in name:
            return self.get_nested_value(name, default)

        # Simple field lookup
        field = self.get_field(name)
        return field.value if field else default

    def get_nested_value(self, path: str, default: Any = None) -> Any:
        """Get a value from a nested path using dot notation.

        Supports paths like:
        - "metadata.type" - access metadata field (if exists) or metadata dict attribute
        - "fields.temperature" - access field values
        - "metadata.config.timeout" - nested dict access

        Args:
            path: Dot-notation path to the value
            default: Default value if path not found

        Returns:
            The value at the path or default
        """
        parts = path.split(".", 1)
        if len(parts) == 1:
            # No more nesting, get the value
            return self.get_value(parts[0], default)

        root, remaining = parts

        # Handle special root paths
        if root == "metadata":
            # Check if "metadata" is a field first, before falling back to attribute
            if root in self.fields:
                # It's a field, navigate through its value
                field_value = self.get_value(root, None)
                if isinstance(field_value, dict):
                    return self._traverse_dict(field_value, remaining, default)
                return default
            elif self.metadata:
                # Fall back to record's metadata attribute
                return self._traverse_dict(self.metadata, remaining, default)
            else:
                return default
        elif root == "fields":
            # Get field value by name
            if "." in remaining:
                # Nested path within field value (if it's a dict)
                field_name, field_path = remaining.split(".", 1)
                field_value = self.get_value(field_name, None)
                if isinstance(field_value, dict):
                    return self._traverse_dict(field_value, field_path, default)
                return default
            else:
                # Simple field access
                return self.get_value(remaining, default)
        else:
            # Check if it's a field containing a dict
            field_value = self.get_value(root, None)
            if isinstance(field_value, dict):
                return self._traverse_dict(field_value, remaining, default)
            return default

    def _traverse_dict(self, data: dict, path: str, default: Any = None) -> Any:
        """Traverse a dictionary using dot notation.

        Args:
            data: Dictionary to traverse
            path: Dot-notation path
            default: Default value if path not found

        Returns:
            Value at path or default
        """
        parts = path.split(".")
        current = data

        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default

        return current

    def set_field(
        self,
        name: str,
        value: Any,
        field_type: FieldType | None = None,
        field_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Set or update a field."""
        self.fields[name] = Field(
            name=name, value=value, type=field_type, metadata=field_metadata or {}
        )

    def set_value(self, name: str, value: Any) -> None:
        """Set a field's value by name.
        
        Convenience method that creates the field if it doesn't exist.
        """
        if name in self.fields:
            self.fields[name].value = value
        else:
            self.set_field(name, value)

    @property
    def data(self) -> dict[str, Any]:
        """Get all field values as a dictionary.
        
        Provides a simple dict-like view of the record's data.
        """
        return {name: field.value for name, field in self.fields.items()}

    def remove_field(self, name: str) -> bool:
        """Remove a field by name. Returns True if field was removed."""
        if name in self.fields:
            del self.fields[name]
            return True
        return False

    def has_field(self, name: str) -> bool:
        """Check if a field exists."""
        return name in self.fields

    def field_names(self) -> list[str]:
        """Get list of field names."""
        return list(self.fields.keys())

    def field_count(self) -> int:
        """Get the number of fields."""
        return len(self.fields)

    def __getitem__(self, key: str | int) -> Any:
        """Get field value by name or field by index.

        For string keys, returns the field value directly (dict-like access).
        For integer keys, returns the Field object at that index for backward compatibility.
        """
        if isinstance(key, str):
            if key not in self.fields:
                raise KeyError(f"Field '{key}' not found")
            return self.fields[key].value
        elif isinstance(key, int):
            field_list = list(self.fields.values())
            if key < 0 or key >= len(field_list):
                raise IndexError(f"Field index {key} out of range")
            return field_list[key]
        else:
            raise TypeError(f"Key must be str or int, got {type(key)}")

    def __setitem__(self, key: str, value: Field | Any) -> None:
        """Set field by name.

        Can accept either a Field object or a raw value.
        When given a raw value, creates a new Field automatically.
        """
        if isinstance(value, Field):
            self.fields[key] = value
        else:
            self.set_field(key, value)

    def __delitem__(self, key: str) -> None:
        """Delete field by name."""
        if key not in self.fields:
            raise KeyError(f"Field '{key}' not found")
        del self.fields[key]

    def __contains__(self, key: str) -> bool:
        """Check if field exists."""
        return key in self.fields

    def __iter__(self) -> Iterator[str]:
        """Iterate over field names."""
        return iter(self.fields)

    def __len__(self) -> int:
        """Get number of fields."""
        return len(self.fields)

    def validate(self) -> bool:
        """Validate all fields in the record."""
        return all(field.validate() for field in self.fields.values())

    def get_field_object(self, key: str) -> Field:
        """Get the Field object by name.

        Use this method when you need access to the Field object itself,
        not just its value.

        Args:
            key: Field name

        Returns:
            The Field object

        Raises:
            KeyError: If field not found
        """
        if key not in self.fields:
            raise KeyError(f"Field '{key}' not found")
        return self.fields[key]

    def __getattr__(self, name: str) -> Any:
        """Get field value by attribute access.

        Provides convenient attribute-style access to field values.
        Falls back to normal attribute access for non-field attributes.

        Args:
            name: Attribute/field name

        Returns:
            Field value if field exists, otherwise raises AttributeError
        """
        # Avoid infinite recursion for special attributes
        if name.startswith("_") or name in ("fields", "metadata", "id"):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

        # Check if it's a field
        if hasattr(self, "fields") and name in self.fields:
            return self.fields[name].value

        raise AttributeError(f"'{type(self).__name__}' object has no field '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        """Set field value by attribute access.

        Allows setting field values using attribute syntax.
        Special attributes (fields, metadata, _id, _storage_id) are handled normally.
        Properties (id, storage_id) are also handled specially.

        Args:
            name: Attribute/field name
            value: Value to set
        """
        # Handle special attributes and private attributes normally
        if name in ("fields", "metadata", "_id", "_storage_id") or name.startswith("_"):
            super().__setattr__(name, value)
        # Handle properties that have setters
        elif name in ("id", "storage_id"):
            # Use the property setter
            object.__setattr__(self, name, value)
        elif hasattr(self, "fields") and name in self.fields:
            # Update existing field value
            self.fields[name].value = value
        else:
            # For new fields during normal operation, create them
            # But during __init__, we need to use normal attribute setting
            if hasattr(self, "fields"):
                self.set_field(name, value)
            else:
                super().__setattr__(name, value)

    def to_dict(
        self,
        include_metadata: bool = False,
        flatten: bool = True,
        include_field_objects: bool = True,
    ) -> dict[str, Any]:
        """Convert record to dictionary.

        Args:
            include_metadata: Whether to include metadata in the output
            flatten: If True (default), return just field values; if False, return structured format
            include_field_objects: If True and not flattened, return full Field objects

        Returns:
            Dictionary representation of the record
        """
        if flatten:
            # Simple dict with just values (default behavior for ergonomics)
            result = {}
            for name, field in self.fields.items():
                # Handle VectorField specially to ensure JSON serialization
                if hasattr(field, 'to_list') and callable(field.to_list):
                    # VectorField has a to_list() method for serialization
                    result[name] = field.to_list()
                else:
                    result[name] = field.value
            if self.id:
                result["_id"] = self.id
            if include_metadata and self.metadata:
                result["_metadata"] = self.metadata
        else:
            # Structured format for serialization
            if include_field_objects:
                result = {
                    "fields": {
                        name: field.to_dict() for name, field in self.fields.items()
                    }
                }
            else:
                result = {
                    "fields": {name: field.value for name, field in self.fields.items()}
                }
            if self.id:
                result["id"] = self.id
            if include_metadata:
                result["metadata"] = self.metadata
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Record:
        """Create a record from a dictionary representation.

        Args:
            data: Dictionary containing record data

        Returns:
            A new Record instance

        Example:
            ```python
            # From simple dict
            data = {"name": "Alice", "age": 30}
            record = Record.from_dict(data)

            # From structured format
            data = {
                "fields": {
                    "name": {"value": "Alice", "type": "string"},
                    "age": {"value": 30, "type": "integer"}
                },
                "metadata": {"source": "api"}
            }
            record = Record.from_dict(data)
            ```
        """
        if "fields" in data:
            fields = OrderedDict()
            for name, field_data in data["fields"].items():
                if isinstance(field_data, dict) and "value" in field_data:
                    # Add name to field_data for Field.from_dict
                    field_data_with_name = {"name": name, **field_data}
                    fields[name] = Field.from_dict(field_data_with_name)
                else:
                    fields[name] = Field(name=name, value=field_data)
            metadata = data.get("metadata", {})
            record_id = data.get("id") or data.get("_id")
            return cls(data=fields, metadata=metadata, id=record_id)
        else:
            # Check for _id in flattened format
            record_id = data.pop("_id", None) if "_id" in data else None
            return cls(data=data, id=record_id)

    def copy(self, deep: bool = True) -> Record:
        """Create a copy of the record.

        Args:
            deep: If True, create deep copies of fields and metadata
        """
        if deep:
            import copy

            new_fields = OrderedDict()
            for name, field in self.fields.items():
                # Preserve the actual field type (Field or VectorField)
                if hasattr(field, '__class__'):
                    # Use the actual class of the field
                    field_class = field.__class__
                    if field_class.__name__ == 'VectorField':
                        # Import VectorField if needed
                        from dataknobs_data.fields import VectorField
                        new_fields[name] = VectorField(
                            name=field.name,
                            value=copy.deepcopy(field.value),
                            dimensions=getattr(field, 'dimensions', None),
                            source_field=getattr(field, 'source_field', None),
                            model_name=getattr(field, 'model_name', None),
                            model_version=getattr(field, 'model_version', None),
                            metadata=copy.deepcopy(field.metadata),
                        )
                    else:
                        new_fields[name] = Field(
                            name=field.name,
                            value=copy.deepcopy(field.value),
                            type=field.type,
                            metadata=copy.deepcopy(field.metadata),
                        )
                else:
                    # Fallback to regular Field
                    new_fields[name] = Field(
                        name=field.name,
                        value=copy.deepcopy(field.value),
                        type=field.type,
                        metadata=copy.deepcopy(field.metadata),
                    )
            new_metadata = copy.deepcopy(self.metadata)
        else:
            new_fields = OrderedDict(self.fields)  # type: ignore[arg-type]
            new_metadata = self.metadata.copy()

        return Record(data=new_fields, metadata=new_metadata, id=self.id)

    def project(self, field_names: list[str]) -> Record:
        """Create a new record with only specified fields.

        Args:
            field_names: List of field names to include in the projection

        Returns:
            A new Record containing only the specified fields

        Example:
            ```python
            record = Record({"name": "Alice", "age": 30, "email": "alice@example.com"})

            # Project to specific fields
            subset = record.project(["name", "age"])
            print(subset.field_names())  # ["name", "age"]
            ```
        """
        projected_fields = OrderedDict()
        for name in field_names:
            if name in self.fields:
                projected_fields[name] = self.fields[name]
        return Record(data=projected_fields, metadata=self.metadata.copy(), id=self.id)

    def merge(self, other: Record, overwrite: bool = True) -> Record:
        """Merge another record into this one.

        Args:
            other: The record to merge
            overwrite: If True, overwrite existing fields; if False, keep existing

        Returns:
            A new merged record
        """
        merged_fields = OrderedDict(self.fields)
        for name, field_obj in other.fields.items():
            if overwrite or name not in merged_fields:
                merged_fields[name] = field_obj

        merged_metadata = self.metadata.copy()
        if overwrite:
            merged_metadata.update(other.metadata)

        # Use the ID from this record, or from other if this doesn't have one
        merged_id = self.id if self.id else other.id

        return Record(data=merged_fields, metadata=merged_metadata, id=merged_id)
