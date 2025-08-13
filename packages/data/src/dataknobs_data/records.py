from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Union

from .fields import Field, FieldType


@dataclass
class Record:
    """Represents a structured data record with fields and metadata."""
    
    fields: OrderedDict[str, Field] = field(default_factory=OrderedDict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __init__(self, data: Optional[Union[Dict[str, Any], OrderedDict[str, Field]]] = None, 
                 metadata: Optional[Dict[str, Any]] = None):
        """Initialize a record from various data formats.
        
        Args:
            data: Can be a dict of field names to values, or an OrderedDict of Field objects
            metadata: Optional metadata for the record
        """
        self.metadata = metadata or {}
        self.fields = OrderedDict()
        
        if data:
            if isinstance(data, OrderedDict) and all(isinstance(v, Field) for v in data.values()):
                self.fields = data
            else:
                for key, value in data.items():
                    if isinstance(value, Field):
                        self.fields[key] = value
                    else:
                        self.fields[key] = Field(name=key, value=value)
    
    def get_field(self, name: str) -> Optional[Field]:
        """Get a field by name."""
        return self.fields.get(name)
    
    def get_value(self, name: str, default: Any = None) -> Any:
        """Get a field's value by name."""
        field = self.get_field(name)
        return field.value if field else default
    
    def set_field(self, name: str, value: Any, field_type: Optional[FieldType] = None,
                  field_metadata: Optional[Dict[str, Any]] = None) -> None:
        """Set or update a field."""
        self.fields[name] = Field(
            name=name,
            value=value,
            type=field_type,
            metadata=field_metadata or {}
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
        else:
            result = {
                "fields": {name: field.to_dict() for name, field in self.fields.items()}
            }
            if include_metadata:
                result["metadata"] = self.metadata
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Record':
        """Create a record from a dictionary representation."""
        if "fields" in data:
            fields = OrderedDict()
            for name, field_data in data["fields"].items():
                if isinstance(field_data, dict) and "value" in field_data:
                    fields[name] = Field.from_dict(field_data)
                else:
                    fields[name] = Field(name=name, value=field_data)
            metadata = data.get("metadata", {})
            return cls(data=fields, metadata=metadata)
        else:
            return cls(data=data)
    
    def copy(self, deep: bool = True) -> 'Record':
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
                    metadata=copy.deepcopy(field.metadata)
                )
            new_metadata = copy.deepcopy(self.metadata)
        else:
            new_fields = OrderedDict(self.fields)
            new_metadata = self.metadata.copy()
        
        return Record(data=new_fields, metadata=new_metadata)
    
    def project(self, field_names: List[str]) -> 'Record':
        """Create a new record with only specified fields."""
        projected_fields = OrderedDict()
        for name in field_names:
            if name in self.fields:
                projected_fields[name] = self.fields[name]
        return Record(data=projected_fields, metadata=self.metadata.copy())
    
    def merge(self, other: 'Record', overwrite: bool = True) -> 'Record':
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
        
        return Record(data=merged_fields, metadata=merged_metadata)