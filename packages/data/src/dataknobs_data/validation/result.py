"""Validation result types with consistent, predictable behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ValidationResult:
    """Unified result object for all validation operations.
    
    This class provides a consistent return type for all validation operations,
    making the API predictable and easy to use.
    """

    valid: bool
    value: Any  # The (possibly coerced) value
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        """Allow 'if result:' usage to check validity."""
        return self.valid

    def merge(self, other: ValidationResult) -> ValidationResult:
        """Combine results for composite validation.
        
        Args:
            other: Another ValidationResult to merge with this one
            
        Returns:
            New ValidationResult with combined state
        """
        return ValidationResult(
            valid=self.valid and other.valid,
            value=other.value if other.valid else self.value,
            errors=self.errors + other.errors,
            warnings=self.warnings + other.warnings
        )

    def add_error(self, error: str) -> ValidationResult:
        """Add an error and mark as invalid (fluent API).
        
        Args:
            error: Error message to add
            
        Returns:
            Self for chaining
        """
        self.errors.append(error)
        self.valid = False
        return self

    def add_warning(self, warning: str) -> ValidationResult:
        """Add a warning without affecting validity (fluent API).
        
        Args:
            warning: Warning message to add
            
        Returns:
            Self for chaining
        """
        self.warnings.append(warning)
        return self

    @classmethod
    def success(cls, value: Any, warnings: list[str] | None = None) -> ValidationResult:
        """Create a successful validation result.
        
        Args:
            value: The validated value
            warnings: Optional list of warnings
            
        Returns:
            Successful ValidationResult
        """
        return cls(
            valid=True,
            value=value,
            errors=[],
            warnings=warnings or []
        )

    @classmethod
    def failure(cls, value: Any, errors: list[str], warnings: list[str] | None = None) -> ValidationResult:
        """Create a failed validation result.
        
        Args:
            value: The value that failed validation
            errors: List of error messages
            warnings: Optional list of warnings
            
        Returns:
            Failed ValidationResult
        """
        return cls(
            valid=False,
            value=value,
            errors=errors,
            warnings=warnings or []
        )


@dataclass
class ValidationContext:
    """Context for stateful validation operations.
    
    Used by constraints like Unique that need to track state across
    multiple validations.
    """

    seen_values: dict[str, set[Any]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def has_seen(self, field: str, value: Any) -> bool:
        """Check if a value has been seen for a field.
        
        Args:
            field: Field name
            value: Value to check
            
        Returns:
            True if value has been seen for this field
        """
        return field in self.seen_values and value in self.seen_values[field]

    def mark_seen(self, field: str, value: Any) -> None:
        """Mark a value as seen for a field.
        
        Args:
            field: Field name
            value: Value to mark as seen
        """
        if field not in self.seen_values:
            self.seen_values[field] = set()
        self.seen_values[field].add(value)

    def clear(self, field: str | None = None) -> None:
        """Clear seen values.
        
        Args:
            field: Optional field to clear. If None, clears all fields.
        """
        if field:
            self.seen_values.pop(field, None)
        else:
            self.seen_values.clear()

    def set_metadata(self, key: str, value: Any) -> None:
        """Store metadata in the context.
        
        Args:
            key: Metadata key
            value: Metadata value
        """
        self.metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Retrieve metadata from the context.
        
        Args:
            key: Metadata key
            default: Default value if key not found
            
        Returns:
            Metadata value or default
        """
        return self.metadata.get(key, default)
