"""Reversible operations for data migration.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from dataknobs_data.records import Record

if TYPE_CHECKING:
    from collections.abc import Callable
    from dataknobs_data.fields import FieldType


@dataclass
class Operation(ABC):
    """Base class for reversible migration operations.
    
    Each operation can be applied forward or reversed for rollback support.
    """

    @abstractmethod
    def apply(self, record: Record) -> Record:
        """Apply this operation to a record.
        
        Args:
            record: Record to transform
            
        Returns:
            Transformed record
        """
        pass

    @abstractmethod
    def reverse(self, record: Record) -> Record:
        """Reverse this operation on a record.
        
        Args:
            record: Record to reverse transform
            
        Returns:
            Record with operation reversed
        """
        pass

    def __repr__(self) -> str:
        """String representation of operation."""
        return f"{self.__class__.__name__}()"


@dataclass
class AddField(Operation):
    """Add a new field to records."""

    field_name: str
    default_value: Any = None
    field_type: FieldType | None = None

    def apply(self, record: Record) -> Record:
        """Add field with default value."""
        result = Record(
            data=dict(record.fields),
            metadata=record.metadata.copy(),
            id=record.id
        )

        # Only add if field doesn't exist
        if self.field_name not in result.fields:
            result.set_field(
                self.field_name,
                self.default_value,
                field_type=self.field_type
            )

        return result

    def reverse(self, record: Record) -> Record:
        """Remove the added field."""
        result = Record(
            data=dict(record.fields),
            metadata=record.metadata.copy(),
            id=record.id
        )

        if self.field_name in result.fields:
            del result.fields[self.field_name]

        return result

    def __repr__(self) -> str:
        return f"AddField(field_name='{self.field_name}', default_value={self.default_value})"


@dataclass
class RemoveField(Operation):
    """Remove a field from records."""

    field_name: str
    store_removed: bool = False  # If True, store removed value in metadata

    def apply(self, record: Record) -> Record:
        """Remove the specified field."""
        result = Record(
            data=dict(record.fields),
            metadata=record.metadata.copy(),
            id=record.id
        )

        if self.field_name in result.fields:
            if self.store_removed:
                # Store removed value in metadata for potential recovery
                result.metadata[f"_removed_{self.field_name}"] = result.fields[self.field_name].value
            del result.fields[self.field_name]

        return result

    def reverse(self, record: Record) -> Record:
        """Restore the removed field if possible."""
        result = Record(
            data=dict(record.fields),
            metadata=record.metadata.copy(),
            id=record.id
        )

        # Try to restore from metadata if available
        metadata_key = f"_removed_{self.field_name}"
        if self.store_removed and metadata_key in result.metadata:
            result.set_field(self.field_name, result.metadata[metadata_key])
            del result.metadata[metadata_key]

        return result

    def __repr__(self) -> str:
        return f"RemoveField(field_name='{self.field_name}')"


@dataclass
class RenameField(Operation):
    """Rename a field."""

    old_name: str
    new_name: str

    def apply(self, record: Record) -> Record:
        """Rename field from old_name to new_name."""
        result = Record(
            data={},
            metadata=record.metadata.copy(),
            id=record.id
        )

        # Copy fields with renaming
        for field_name, field in record.fields.items():
            if field_name == self.old_name:
                result.fields[self.new_name] = field
                # Update field's internal name
                result.fields[self.new_name].name = self.new_name
            else:
                result.fields[field_name] = field

        return result

    def reverse(self, record: Record) -> Record:
        """Rename field from new_name back to old_name."""
        result = Record(
            data={},
            metadata=record.metadata.copy(),
            id=record.id
        )

        # Copy fields with reverse renaming
        for field_name, field in record.fields.items():
            if field_name == self.new_name:
                result.fields[self.old_name] = field
                # Update field's internal name
                result.fields[self.old_name].name = self.old_name
            else:
                result.fields[field_name] = field

        return result

    def __repr__(self) -> str:
        return f"RenameField(old_name='{self.old_name}', new_name='{self.new_name}')"


@dataclass
class TransformField(Operation):
    """Transform a field's value using a function."""

    field_name: str
    transform_fn: Callable[[Any], Any]
    reverse_fn: Callable[[Any], Any] | None = None

    def apply(self, record: Record) -> Record:
        """Apply transformation to field value."""
        result = Record(
            data=dict(record.fields),
            metadata=record.metadata.copy(),
            id=record.id
        )

        if self.field_name in result.fields:
            old_value = result.fields[self.field_name].value
            try:
                new_value = self.transform_fn(old_value)
                result.set_field(
                    self.field_name,
                    new_value,
                    field_type=result.fields[self.field_name].type,
                    field_metadata=result.fields[self.field_name].metadata
                )
            except Exception as e:
                # If transformation fails, keep original value
                # Could optionally store error in metadata
                result.metadata[f"_transform_error_{self.field_name}"] = str(e)

        return result

    def reverse(self, record: Record) -> Record:
        """Reverse the transformation if reverse function provided."""
        if self.reverse_fn is None:
            # Can't reverse without reverse function
            return record

        result = Record(
            data=dict(record.fields),
            metadata=record.metadata.copy(),
            id=record.id
        )

        if self.field_name in result.fields:
            old_value = result.fields[self.field_name].value
            try:
                new_value = self.reverse_fn(old_value)
                result.set_field(
                    self.field_name,
                    new_value,
                    field_type=result.fields[self.field_name].type,
                    field_metadata=result.fields[self.field_name].metadata
                )
            except Exception as e:
                # If reverse fails, keep original value
                result.metadata[f"_reverse_error_{self.field_name}"] = str(e)

        # Clean up any transform error metadata
        error_key = f"_transform_error_{self.field_name}"
        if error_key in result.metadata:
            del result.metadata[error_key]

        return result

    def __repr__(self) -> str:
        return f"TransformField(field_name='{self.field_name}')"


@dataclass
class CompositeOperation(Operation):
    """Combine multiple operations into one."""

    operations: list[Operation]

    def apply(self, record: Record) -> Record:
        """Apply all operations in sequence."""
        result = record
        for operation in self.operations:
            result = operation.apply(result)
        return result

    def reverse(self, record: Record) -> Record:
        """Reverse all operations in reverse order."""
        result = record
        for operation in reversed(self.operations):
            result = operation.reverse(result)
        return result

    def add(self, operation: Operation) -> CompositeOperation:
        """Add an operation (fluent API)."""
        self.operations.append(operation)
        return self

    def __repr__(self) -> str:
        return f"CompositeOperation(operations={len(self.operations)})"
