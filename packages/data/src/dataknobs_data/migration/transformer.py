"""Data transformation with fluent API.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from dataknobs_data.records import Record

if TYPE_CHECKING:
    from collections.abc import Callable
    from dataknobs_data.fields import FieldType


class TransformRule(ABC):
    """Base class for transformation rules."""

    @abstractmethod
    def apply(self, record: Record) -> Record:
        """Apply this transformation rule to a record.
        
        Args:
            record: Record to transform
            
        Returns:
            Transformed record
        """
        pass


@dataclass
class MapRule(TransformRule):
    """Map a field to another field, optionally transforming the value."""

    source: str
    target: str
    transform: Callable[[Any], Any] | None = None

    def apply(self, record: Record) -> Record:
        """Apply field mapping."""
        result = Record(
            data=dict(record.fields),
            metadata=record.metadata.copy(),
            id=record.id
        )

        if self.source in record.fields:
            value = record.fields[self.source].value

            # Apply transformation if provided
            if self.transform:
                try:
                    value = self.transform(value)
                except Exception as e:
                    # Store error in metadata and keep original value
                    result.metadata[f"_transform_error_{self.source}"] = str(e)
                    value = record.fields[self.source].value

            # If target is different from source, remove source field
            if self.target != self.source:
                del result.fields[self.source]

            # Set target field
            result.set_field(self.target, value)

        return result


@dataclass
class ExcludeRule(TransformRule):
    """Exclude specified fields from the record."""

    fields: list[str]

    def apply(self, record: Record) -> Record:
        """Remove excluded fields."""
        result = Record(
            data={},
            metadata=record.metadata.copy(),
            id=record.id
        )

        # Copy all fields except excluded ones
        for field_name, field in record.fields.items():
            if field_name not in self.fields:
                result.fields[field_name] = field

        return result


@dataclass
class AddRule(TransformRule):
    """Add a new field with a computed or default value."""

    field_name: str
    value: Any | Callable[[Record], Any]
    field_type: FieldType | None = None

    def apply(self, record: Record) -> Record:
        """Add new field."""
        result = Record(
            data=dict(record.fields),
            metadata=record.metadata.copy(),
            id=record.id
        )

        # Compute value if it's a callable
        if callable(self.value):
            try:
                computed_value = self.value(record)
            except Exception as e:
                # Store error and use None as value
                result.metadata[f"_compute_error_{self.field_name}"] = str(e)
                computed_value = None
        else:
            computed_value = self.value

        result.set_field(self.field_name, computed_value, field_type=self.field_type)
        return result


class Transformer:
    """Stateless record transformer with fluent API.
    
    Provides a clean, chainable interface for defining record transformations
    that can be applied during migrations or data processing.
    """

    def __init__(self):
        """Initialize transformer with empty rule set."""
        self.rules: list[TransformRule] = []

    def map(
        self,
        source: str,
        target: str | None = None,
        transform: Callable[[Any], Any] | None = None
    ) -> Transformer:
        """Map a field, optionally transforming its value (fluent API).
        
        Args:
            source: Source field name
            target: Target field name (defaults to source)
            transform: Optional transformation function
            
        Returns:
            Self for chaining
        """
        self.rules.append(MapRule(
            source=source,
            target=target or source,
            transform=transform
        ))
        return self

    def rename(self, old_name: str, new_name: str) -> Transformer:
        """Rename a field (fluent API).
        
        Args:
            old_name: Current field name
            new_name: New field name
            
        Returns:
            Self for chaining
        """
        return self.map(old_name, new_name)

    def exclude(self, *fields: str) -> Transformer:
        """Exclude fields from the record (fluent API).
        
        Args:
            *fields: Field names to exclude
            
        Returns:
            Self for chaining
        """
        self.rules.append(ExcludeRule(list(fields)))
        return self

    def add(
        self,
        field_name: str,
        value: Any | Callable[[Record], Any],
        field_type: FieldType | None = None
    ) -> Transformer:
        """Add a new field (fluent API).
        
        Args:
            field_name: Name of field to add
            value: Static value or function to compute value
            field_type: Optional field type
            
        Returns:
            Self for chaining
        """
        self.rules.append(AddRule(
            field_name=field_name,
            value=value,
            field_type=field_type
        ))
        return self

    def add_rule(self, rule: TransformRule) -> Transformer:
        """Add a custom transformation rule (fluent API).
        
        Args:
            rule: Custom transformation rule
            
        Returns:
            Self for chaining
        """
        self.rules.append(rule)
        return self

    def transform(self, record: Record) -> Record | None:
        """Apply all transformation rules to a record.
        
        Args:
            record: Record to transform
            
        Returns:
            Transformed record, or None to filter out the record
        """
        result = record
        for rule in self.rules:
            result = rule.apply(result)

        return result

    def transform_many(self, records: list[Record]) -> list[Record]:
        """Transform multiple records.
        
        Args:
            records: List of records to transform
            
        Returns:
            List of transformed records (filtered records are excluded)
        """
        results = []
        for record in records:
            transformed = self.transform(record)
            if transformed is not None:
                results.append(transformed)
        return results

    def clear(self) -> Transformer:
        """Clear all transformation rules (fluent API).
        
        Returns:
            Self for chaining
        """
        self.rules.clear()
        return self

    def __len__(self) -> int:
        """Get number of transformation rules."""
        return len(self.rules)

    def __repr__(self) -> str:
        """String representation."""
        return f"Transformer(rules={len(self.rules)})"
