"""Database schema definitions for field structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .fields import FieldType


@dataclass
class FieldSchema:
    """Schema definition for a field without actual data."""

    name: str
    type: FieldType
    metadata: dict[str, Any] = field(default_factory=dict)
    required: bool = False
    default: Any = None

    def is_vector_field(self) -> bool:
        """Check if this is a vector field."""
        return self.type in (FieldType.VECTOR, FieldType.SPARSE_VECTOR)

    def get_dimensions(self) -> int | None:
        """Get vector dimensions if this is a vector field."""
        if self.is_vector_field():
            return self.metadata.get("dimensions")
        return None

    def get_source_field(self) -> str | None:
        """Get source field if this is a derived vector field."""
        if self.is_vector_field():
            return self.metadata.get("source_field")
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "type": self.type.value,
            "metadata": self.metadata,
            "required": self.required,
            "default": self.default,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FieldSchema:
        """Create from dictionary representation."""
        return cls(
            name=data["name"],
            type=FieldType(data["type"]),
            metadata=data.get("metadata", {}),
            required=data.get("required", False),
            default=data.get("default"),
        )


@dataclass
class DatabaseSchema:
    """Schema definition for a database."""

    fields: dict[str, FieldSchema] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(cls, **field_definitions) -> DatabaseSchema:
        """Create a schema from keyword arguments.
        
        Examples:
            schema = DatabaseSchema.create(
                content=FieldType.TEXT,
                embedding=(FieldType.VECTOR, {"dimensions": 384, "source_field": "content"}),
                title=FieldType.TEXT,
                score=(FieldType.FLOAT, {"required": True})
            )
        """
        schema = cls()
        for name, definition in field_definitions.items():
            if isinstance(definition, FieldType):
                # Simple field type
                schema.add_field(FieldSchema(name=name, type=definition))
            elif isinstance(definition, tuple):
                # Field type with metadata/options
                field_type, options = definition
                field_metadata = options.get("metadata", {})
                if "dimensions" in options:
                    field_metadata["dimensions"] = options["dimensions"]
                if "source_field" in options:
                    field_metadata["source_field"] = options["source_field"]

                schema.add_field(FieldSchema(
                    name=name,
                    type=field_type,
                    metadata=field_metadata,
                    required=options.get("required", False),
                    default=options.get("default")
                ))
            else:
                raise ValueError(f"Invalid field definition for {name}: {definition}")
        return schema

    def add_field(self, field_schema: FieldSchema) -> DatabaseSchema:
        """Add a field to the schema.
        
        Returns self for chaining.
        """
        self.fields[field_schema.name] = field_schema
        return self

    def add_text_field(self, name: str, required: bool = False) -> DatabaseSchema:
        """Add a text field to the schema."""
        return self.add_field(FieldSchema(name=name, type=FieldType.TEXT, required=required))

    def add_vector_field(
        self,
        name: str,
        dimensions: int,
        source_field: str | None = None,
        required: bool = False
    ) -> DatabaseSchema:
        """Add a vector field to the schema."""
        return self.add_field(FieldSchema(
            name=name,
            type=FieldType.VECTOR,
            metadata={"dimensions": dimensions, "source_field": source_field},
            required=required
        ))

    def remove_field(self, name: str) -> bool:
        """Remove a field from the schema."""
        if name in self.fields:
            del self.fields[name]
            return True
        return False

    def get_vector_fields(self) -> dict[str, FieldSchema]:
        """Get all vector fields in the schema."""
        return {
            name: field
            for name, field in self.fields.items()
            if field.is_vector_field()
        }

    def get_source_fields(self) -> dict[str, list[str]]:
        """Get mapping of source fields to their dependent vector fields."""
        source_map = {}
        for name, field_obj in self.fields.items():
            if field_obj.is_vector_field():
                source = field_obj.get_source_field()
                if source:
                    if source not in source_map:
                        source_map[source] = []
                    source_map[source].append(name)
        return source_map

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "fields": {name: f.to_dict() for name, f in self.fields.items()},
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatabaseSchema:
        """Create from dictionary representation.
        
        Supports multiple formats:
        1. Full format with FieldSchema dicts
        2. Simple format with just field types
        3. Mixed format
        
        Examples:
            # Simple format
            {"fields": {"content": "text", "score": "float"}}
            
            # Full format
            {"fields": {"content": {"type": "text", "required": true}}}
            
            # Vector fields
            {"fields": {"embedding": {"type": "vector", "dimensions": 384}}}
        """
        schema = cls(metadata=data.get("metadata", {}))

        for name, field_data in data.get("fields", {}).items():
            if isinstance(field_data, str):
                # Simple string type
                schema.fields[name] = FieldSchema(
                    name=name,
                    type=FieldType(field_data)
                )
            elif isinstance(field_data, dict):
                if "type" in field_data:
                    # Full field schema dict
                    field_type = FieldType(field_data["type"])
                    metadata = {}

                    # Handle vector-specific fields
                    if "dimensions" in field_data:
                        metadata["dimensions"] = field_data["dimensions"]
                    if "source_field" in field_data:
                        metadata["source_field"] = field_data["source_field"]

                    # Merge with explicit metadata
                    if "metadata" in field_data:
                        metadata.update(field_data["metadata"])

                    schema.fields[name] = FieldSchema(
                        name=name,
                        type=field_type,
                        metadata=metadata,
                        required=field_data.get("required", False),
                        default=field_data.get("default")
                    )
                else:
                    # Try to parse as FieldSchema dict
                    schema.fields[name] = FieldSchema.from_dict(field_data)

        return schema
