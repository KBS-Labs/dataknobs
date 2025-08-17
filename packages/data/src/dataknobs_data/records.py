from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import uuid

from .fields import Field, FieldType


@dataclass
class Record:
    """Represents a structured data record with fields and metadata.
    
    The record ID can be accessed via the `id` property, which:
    - Returns the explicitly set ID if available
    - Falls back to metadata['id'] if present
    - Returns None if no ID is set
    """

    fields: OrderedDict[str, Field] = field(default_factory=OrderedDict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    _id: Optional[str] = field(default=None, repr=False)

    def __init__(
        self,
        data: Union[Dict[str, Any], OrderedDict[str, Field]] | None = None,
        metadata: Dict[str, Any] | None = None,
        id: Optional[str] = None,
    ):
        """Initialize a record from various data formats.

        Args:
            data: Can be a dict of field names to values, or an OrderedDict of Field objects
            metadata: Optional metadata for the record
            id: Optional unique identifier for the record
        """
        self.metadata = metadata or {}
        self.fields = OrderedDict()
        self._id = id
        
        # Process data first to populate fields
        if data:
            if isinstance(data, OrderedDict) and all(isinstance(v, Field) for v in data.values()):
                self.fields = data
            else:
                for key, value in data.items():
                    if isinstance(value, Field):
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
    def id(self) -> Optional[str]:
        """Get the record ID.
        
        Checks for ID in the following priority order:
        1. Explicitly set ID (_id)
        2. ID in metadata
        3. ID field in record fields
        4. record_id field in record fields (common in DataFrames)
        
        Returns the first ID found, or None if no ID is present.
        """
        # 1. Check explicitly set ID
        if self._id is not None:
            return self._id
        
        # 2. Check metadata
        if "id" in self.metadata:
            return str(self.metadata["id"])
        
        # 3. Check for 'id' field
        if "id" in self.fields:
            value = self.get_value("id")
            if value is not None:
                return str(value)
        
        # 4. Check for 'record_id' field (common in DataFrames)
        if "record_id" in self.fields:
            value = self.get_value("record_id")
            if value is not None:
                return str(value)
        
        return None
    
    @id.setter
    def id(self, value: Optional[str]) -> None:
        """Set the record ID.
        
        Updates the ID in all locations for consistency:
        - Internal _id attribute
        - Metadata (for backward compatibility)
        - ID field if it exists
        """
        self._id = value
        
        # Update metadata for backward compatibility
        if value is not None:
            self.metadata["id"] = value
        elif "id" in self.metadata:
            del self.metadata["id"]
        
        # Update ID field if it exists (don't create it if it doesn't)
        if "id" in self.fields and value is not None:
            self.fields["id"].value = value
        
        # Update record_id field if it exists (common in DataFrames)
        if "record_id" in self.fields and value is not None:
            self.fields["record_id"].value = value
    
    def generate_id(self) -> str:
        """Generate and set a new UUID for this record.
        
        Returns:
            The generated UUID string
        """
        new_id = str(uuid.uuid4())
        self.id = new_id
        return new_id

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
        - "metadata.type" - access metadata dict
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
            # Navigate through metadata dict
            if not self.metadata:
                return default
            return self._traverse_dict(self.metadata, remaining, default)
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
        field_metadata: Dict[str, Any] | None = None,
    ) -> None:
        """Set or update a field."""
        self.fields[name] = Field(
            name=name, value=value, type=field_type, metadata=field_metadata or {}
        )

    def remove_field(self, name: str) -> bool:
        """Remove a field by name. Returns True if field was removed."""
        if name in self.fields:
            del self.fields[name]
            return True
        return False

    def has_field(self, name: str) -> bool:
        """Check if a field exists."""
        return name in self.fields

    def field_names(self) -> List[str]:
        """Get list of field names."""
        return list(self.fields.keys())

    def field_count(self) -> int:
        """Get the number of fields."""
        return len(self.fields)

    def __getitem__(self, key: Union[str, int]) -> Field:
        """Get field by name or index."""
        if isinstance(key, str):
            if key not in self.fields:
                raise KeyError(f"Field '{key}' not found")
            return self.fields[key]
        elif isinstance(key, int):
            field_list = list(self.fields.values())
            if key < 0 or key >= len(field_list):
                raise IndexError(f"Field index {key} out of range")
            return field_list[key]
        else:
            raise TypeError(f"Key must be str or int, got {type(key)}")

    def __setitem__(self, key: str, value: Union[Field, Any]) -> None:
        """Set field by name."""
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

    def to_dict(self, include_metadata: bool = True, flatten: bool = False) -> Dict[str, Any]:
        """Convert record to dictionary.

        Args:
            include_metadata: Whether to include metadata in the output
            flatten: If True, return just field values; if False, return full field dictionaries
        """
        if flatten:
            result = {name: field.value for name, field in self.fields.items()}
            if self.id:
                result["_id"] = self.id
        else:
            result = {"fields": {name: field.to_dict() for name, field in self.fields.items()}}
            if self.id:
                result["id"] = self.id
            if include_metadata:
                result["metadata"] = self.metadata
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Record":
        """Create a record from a dictionary representation."""
        if "fields" in data:
            fields = OrderedDict()
            for name, field_data in data["fields"].items():
                if isinstance(field_data, dict) and "value" in field_data:
                    fields[name] = Field.from_dict(field_data)
                else:
                    fields[name] = Field(name=name, value=field_data)
            metadata = data.get("metadata", {})
            record_id = data.get("id") or data.get("_id")
            return cls(data=fields, metadata=metadata, id=record_id)
        else:
            # Check for _id in flattened format
            record_id = data.pop("_id", None) if "_id" in data else None
            return cls(data=data, id=record_id)

    def copy(self, deep: bool = True) -> "Record":
        """Create a copy of the record.

        Args:
            deep: If True, create deep copies of fields and metadata
        """
        if deep:
            import copy

            new_fields = OrderedDict()
            for name, field in self.fields.items():
                new_fields[name] = Field(
                    name=field.name,
                    value=copy.deepcopy(field.value),
                    type=field.type,
                    metadata=copy.deepcopy(field.metadata),
                )
            new_metadata = copy.deepcopy(self.metadata)
        else:
            new_fields = OrderedDict(self.fields)
            new_metadata = self.metadata.copy()

        return Record(data=new_fields, metadata=new_metadata, id=self.id)

    def project(self, field_names: List[str]) -> "Record":
        """Create a new record with only specified fields."""
        projected_fields = OrderedDict()
        for name in field_names:
            if name in self.fields:
                projected_fields[name] = self.fields[name]
        return Record(data=projected_fields, metadata=self.metadata.copy(), id=self.id)

    def merge(self, other: "Record", overwrite: bool = True) -> "Record":
        """Merge another record into this one.

        Args:
            other: The record to merge
            overwrite: If True, overwrite existing fields; if False, keep existing

        Returns:
            A new merged record
        """
        merged_fields = OrderedDict(self.fields)
        for name, field in other.fields.items():
            if overwrite or name not in merged_fields:
                merged_fields[name] = field

        merged_metadata = self.metadata.copy()
        if overwrite:
            merged_metadata.update(other.metadata)
        
        # Use the ID from this record, or from other if this doesn't have one
        merged_id = self.id if self.id else other.id

        return Record(data=merged_fields, metadata=merged_metadata, id=merged_id)
