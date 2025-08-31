"""Migration definition with reversible operations.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dataknobs_data.records import Record
    from .operations import Operation
    

class Migration:
    """Migration between data versions with reversible operations.
    
    Provides a clean API for defining and applying migrations with
    support for rollback via operation reversal.
    """

    def __init__(self, from_version: str, to_version: str, description: str | None = None):
        """Initialize migration.
        
        Args:
            from_version: Source version identifier
            to_version: Target version identifier
            description: Optional migration description
        """
        self.from_version = from_version
        self.to_version = to_version
        self.description = description
        self.operations: list[Operation] = []

    def add(self, operation: Operation) -> Migration:
        """Add an operation to the migration (fluent API).
        
        Args:
            operation: Operation to add
            
        Returns:
            Self for chaining
        """
        self.operations.append(operation)
        return self

    def add_many(self, operations: list[Operation]) -> Migration:
        """Add multiple operations (fluent API).
        
        Args:
            operations: List of operations to add
            
        Returns:
            Self for chaining
        """
        self.operations.extend(operations)
        return self

    def apply(self, record: Record, reverse: bool = False) -> Record:
        """Apply migration to a record.
        
        Args:
            record: Record to migrate
            reverse: If True, apply operations in reverse
            
        Returns:
            Migrated record
        """
        result = record

        if reverse:
            # Apply operations in reverse order with reverse method
            for operation in reversed(self.operations):
                result = operation.reverse(result)
        else:
            # Apply operations in forward order
            for operation in self.operations:
                result = operation.apply(result)

        # Update version metadata
        if reverse:
            result.metadata["version"] = self.from_version
        else:
            result.metadata["version"] = self.to_version

        return result

    def apply_many(self, records: list[Record], reverse: bool = False) -> list[Record]:
        """Apply migration to multiple records.
        
        Args:
            records: List of records to migrate
            reverse: If True, apply operations in reverse
            
        Returns:
            List of migrated records
        """
        return [self.apply(record, reverse) for record in records]

    def can_reverse(self) -> bool:
        """Check if this migration can be reversed.
        
        All operations must support reversal for the migration to be reversible.
        
        Returns:
            True if migration can be reversed
        """
        # All our operations support reversal by design
        return True

    def get_affected_fields(self) -> set[str]:
        """Get set of field names affected by this migration.
        
        Returns:
            Set of field names that will be modified
        """
        affected = set()

        for operation in self.operations:
            # Extract field names based on operation type
            if hasattr(operation, 'field_name'):
                affected.add(operation.field_name)
            elif hasattr(operation, 'old_name'):
                affected.add(operation.old_name)
                if hasattr(operation, 'new_name'):
                    affected.add(operation.new_name)
            elif hasattr(operation, 'operations'):
                # Composite operation - recursively get affected fields
                for sub_op in operation.operations:
                    if hasattr(sub_op, 'field_name'):
                        affected.add(sub_op.field_name)
                    elif hasattr(sub_op, 'old_name'):
                        affected.add(sub_op.old_name)
                        if hasattr(sub_op, 'new_name'):
                            affected.add(sub_op.new_name)

        return affected

    def validate(self, record: Record) -> tuple[bool, list[str]]:
        """Validate if a record can be migrated.
        
        Args:
            record: Record to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []

        # Check current version
        current_version = record.metadata.get("version")
        if current_version and current_version != self.from_version:
            issues.append(
                f"Record version mismatch: expected {self.from_version}, got {current_version}"
            )

        # Check for required fields for operations
        for operation in self.operations:
            if hasattr(operation, 'old_name'):
                # RenameField operation
                if operation.old_name not in record.fields:
                    issues.append(f"Field '{operation.old_name}' not found for rename operation")
            elif hasattr(operation, 'field_name') and operation.__class__.__name__ == 'TransformField':
                # TransformField operation
                if operation.field_name not in record.fields:
                    issues.append(f"Field '{operation.field_name}' not found for transform operation")

        return len(issues) == 0, issues

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"Migration(from='{self.from_version}', to='{self.to_version}', "
            f"operations={len(self.operations)})"
        )

    def __str__(self) -> str:
        """Human-readable string."""
        desc = f"Migration from {self.from_version} to {self.to_version}"
        if self.description:
            desc += f": {self.description}"
        desc += f"\n  Operations ({len(self.operations)}):"
        for i, op in enumerate(self.operations, 1):
            desc += f"\n    {i}. {op}"
        return desc
