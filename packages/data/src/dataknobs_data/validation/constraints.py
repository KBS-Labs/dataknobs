"""Constraint implementations with consistent, composable API.
"""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from numbers import Number
from re import Pattern as RegexPattern
from typing import Any as AnyType, TYPE_CHECKING

from .result import ValidationContext, ValidationResult

if TYPE_CHECKING:
    from collections.abc import Callable


class Constraint(ABC):
    """Base class for all constraints with composable operators."""

    @abstractmethod
    def check(self, value: AnyType, context: ValidationContext | None = None) -> ValidationResult:
        """Validate a value against this constraint.
        
        Args:
            value: Value to validate
            context: Optional validation context for stateful constraints
            
        Returns:
            ValidationResult with validation outcome
        """
        pass

    def __and__(self, other: Constraint) -> All:
        """Combine with AND: both constraints must pass."""
        if isinstance(self, All):
            return All(self.constraints + [other])
        elif isinstance(other, All):
            return All([self] + other.constraints)
        return All([self, other])

    def __or__(self, other: Constraint) -> AnyOf:
        """Combine with OR: at least one constraint must pass."""
        if isinstance(self, AnyOf):
            return AnyOf(self.constraints + [other])
        elif isinstance(other, AnyOf):
            return AnyOf([self] + other.constraints)
        return AnyOf([self, other])

    def __invert__(self) -> Not:
        """Negate this constraint."""
        return Not(self)


class All(Constraint):
    """All constraints must pass (AND logic)."""

    def __init__(self, constraints: list[Constraint]):
        """Initialize with list of constraints.
        
        Args:
            constraints: List of constraints that must all pass
        """
        self.constraints = constraints

    def check(self, value: AnyType, context: ValidationContext | None = None) -> ValidationResult:
        """Check all constraints."""
        result = ValidationResult.success(value)

        for constraint in self.constraints:
            check_result = constraint.check(value, context)
            if not check_result.valid:
                result = result.merge(check_result)
                # Continue checking to collect all errors

        return result


class AnyOf(Constraint):
    """At least one constraint must pass (OR logic)."""

    def __init__(self, constraints: list[Constraint]):
        """Initialize with list of constraints.
        
        Args:
            constraints: List of constraints where at least one must pass
        """
        self.constraints = constraints

    def check(self, value: AnyType, context: ValidationContext | None = None) -> ValidationResult:
        """Check if any constraint passes."""
        all_errors = []

        for constraint in self.constraints:
            check_result = constraint.check(value, context)
            if check_result.valid:
                return check_result
            all_errors.extend(check_result.errors)

        return ValidationResult.failure(
            value,
            [f"None of the constraints passed: {', '.join(all_errors)}"]
        )


class Not(Constraint):
    """Negates a constraint."""

    def __init__(self, constraint: Constraint):
        """Initialize with constraint to negate.
        
        Args:
            constraint: Constraint to negate
        """
        self.constraint = constraint

    def check(self, value: AnyType, context: ValidationContext | None = None) -> ValidationResult:
        """Check if constraint fails (negation)."""
        result = self.constraint.check(value, context)
        if result.valid:
            return ValidationResult.failure(
                value,
                ["Value should not satisfy constraint but it does"]
            )
        return ValidationResult.success(value)


class Required(Constraint):
    """Field must be present and non-null."""

    def __init__(self, allow_empty: bool = False):
        """Initialize required constraint.
        
        Args:
            allow_empty: If True, empty strings/collections are allowed
        """
        self.allow_empty = allow_empty

    def check(self, value: AnyType, context: ValidationContext | None = None) -> ValidationResult:
        """Check if value is present and non-null."""
        if value is None:
            return ValidationResult.failure(value, ["Value is required"])

        if not self.allow_empty:
            # Check for empty strings and collections
            if isinstance(value, (str, list, dict, set, tuple)) and len(value) == 0:
                return ValidationResult.failure(value, ["Value cannot be empty"])

        return ValidationResult.success(value)


class Range(Constraint):
    """Numeric value must be in specified range."""

    def __init__(
        self,
        min: Number | None = None,
        max: Number | None = None,
        min_exclusive: bool = False,
        max_exclusive: bool = False,
    ):
        """Initialize range constraint.

        Args:
            min: Minimum value (inclusive by default)
            max: Maximum value (inclusive by default)
            min_exclusive: If True, minimum is exclusive (value must be > min)
            max_exclusive: If True, maximum is exclusive (value must be < max)
        """
        if min is not None and max is not None and float(min) > float(max):  # type: ignore[arg-type]
            raise ValueError(f"min ({min}) cannot be greater than max ({max})")
        self.min = min
        self.max = max
        self.min_exclusive = min_exclusive
        self.max_exclusive = max_exclusive

    def check(self, value: AnyType, context: ValidationContext | None = None) -> ValidationResult:
        """Check if value is in range."""
        if value is None:
            return ValidationResult.success(value)  # None is considered valid (use Required to enforce)

        if not isinstance(value, Number):
            return ValidationResult.failure(
                value,
                [f"Value must be a number, got {type(value).__name__}"]
            )

        # Check for NaN (Not a Number) - NaN is not valid for range comparisons
        if isinstance(value, float) and math.isnan(value):
            return ValidationResult.failure(
                value,
                ["Value is NaN (Not a Number), which is not valid for range comparisons"]
            )

        errors = []
        if self.min is not None:
            if self.min_exclusive:
                if float(value) <= float(self.min):  # type: ignore[arg-type]
                    errors.append(f"Value {value} must be greater than {self.min}")
            else:
                if float(value) < float(self.min):  # type: ignore[arg-type]
                    errors.append(f"Value {value} is less than minimum {self.min}")

        if self.max is not None:
            if self.max_exclusive:
                if float(value) >= float(self.max):  # type: ignore[arg-type]
                    errors.append(f"Value {value} must be less than {self.max}")
            else:
                if float(value) > float(self.max):  # type: ignore[arg-type]
                    errors.append(f"Value {value} is greater than maximum {self.max}")

        if errors:
            return ValidationResult.failure(value, errors)
        return ValidationResult.success(value)


class Length(Constraint):
    """String/collection length must be in specified range."""

    def __init__(self, min: int | None = None, max: int | None = None):
        """Initialize length constraint.
        
        Args:
            min: Minimum length (inclusive)
            max: Maximum length (inclusive)
        """
        if min is not None and min < 0:
            raise ValueError(f"min length cannot be negative: {min}")
        if max is not None and max < 0:
            raise ValueError(f"max length cannot be negative: {max}")
        if min is not None and max is not None and min > max:
            raise ValueError(f"min length ({min}) cannot be greater than max ({max})")
        self.min = min
        self.max = max

    def check(self, value: AnyType, context: ValidationContext | None = None) -> ValidationResult:
        """Check if value length is in range."""
        if value is None:
            return ValidationResult.success(value)

        if not hasattr(value, '__len__'):
            return ValidationResult.failure(
                value,
                [f"Value does not have a length: {type(value).__name__}"]
            )

        length = len(value)
        errors = []

        if self.min is not None and length < self.min:
            errors.append(f"Length {length} is less than minimum {self.min}")
        if self.max is not None and length > self.max:
            errors.append(f"Length {length} is greater than maximum {self.max}")

        if errors:
            return ValidationResult.failure(value, errors)
        return ValidationResult.success(value)


class Pattern(Constraint):
    """String value must match regex pattern."""

    def __init__(self, pattern: str | RegexPattern):
        """Initialize pattern constraint.
        
        Args:
            pattern: Regex pattern (string or compiled pattern)
        """
        if isinstance(pattern, str):
            self.regex = re.compile(pattern)
        else:
            self.regex = pattern
        self.pattern_str = pattern if isinstance(pattern, str) else pattern.pattern

    def check(self, value: AnyType, context: ValidationContext | None = None) -> ValidationResult:
        """Check if value matches pattern."""
        if value is None:
            return ValidationResult.success(value)

        if not isinstance(value, str):
            return ValidationResult.failure(
                value,
                [f"Value must be a string for pattern matching, got {type(value).__name__}"]
            )

        if not self.regex.match(value):
            return ValidationResult.failure(
                value,
                [f"Value '{value}' does not match pattern '{self.pattern_str}'"]
            )

        return ValidationResult.success(value)


class Enum(Constraint):
    """Value must be in allowed set."""

    def __init__(self, values: list[AnyType], case_sensitive: bool = True):
        """Initialize enum constraint.

        Args:
            values: List of allowed values
            case_sensitive: If False, string comparisons ignore case
        """
        if not values:
            raise ValueError("Enum constraint requires at least one allowed value")
        self.values = values
        self.case_sensitive = case_sensitive
        self.allowed_str = ', '.join(repr(v) for v in values)

        if case_sensitive:
            self.allowed = set(values)
        else:
            # For case-insensitive, store lowercase versions for comparison
            self.allowed_lower = {
                v.lower() if isinstance(v, str) else v for v in values
            }

    def check(self, value: AnyType, context: ValidationContext | None = None) -> ValidationResult:
        """Check if value is in allowed set."""
        if value is None:
            return ValidationResult.success(value)

        if self.case_sensitive:
            if value not in self.allowed:
                return ValidationResult.failure(
                    value,
                    [f"Value '{value}' is not in allowed values: {self.allowed_str}"]
                )
        else:
            # Case-insensitive comparison
            check_value = value.lower() if isinstance(value, str) else value
            if check_value not in self.allowed_lower:
                return ValidationResult.failure(
                    value,
                    [f"Value '{value}' is not in allowed values: {self.allowed_str}"]
                )

        return ValidationResult.success(value)


class Unique(Constraint):
    """Value must be unique (uses context for tracking)."""

    def __init__(self, field_name: str | None = None):
        """Initialize unique constraint.
        
        Args:
            field_name: Optional field name for context tracking
        """
        self.field_name = field_name or "default"

    def check(self, value: AnyType, context: ValidationContext | None = None) -> ValidationResult:
        """Check if value is unique using context."""
        if value is None:
            return ValidationResult.success(value)

        if context is None:
            # Without context, we can't track uniqueness
            return ValidationResult.success(
                value,
                warnings=["Unique constraint requires context for tracking"]
            )

        if context.has_seen(self.field_name, value):
            return ValidationResult.failure(
                value,
                [f"Duplicate value '{value}' for field '{self.field_name}'"]
            )

        context.mark_seen(self.field_name, value)
        return ValidationResult.success(value)


class Custom(Constraint):
    """Custom constraint using a callable."""

    def __init__(
        self,
        validator: Callable[[AnyType], bool | ValidationResult],
        error_message: str = "Custom validation failed"
    ):
        """Initialize custom constraint.
        
        Args:
            validator: Callable that returns bool or ValidationResult
            error_message: Error message if validation fails
        """
        self.validator = validator
        self.error_message = error_message

    def check(self, value: AnyType, context: ValidationContext | None = None) -> ValidationResult:
        """Check using custom validator."""
        try:
            result = self.validator(value)

            if isinstance(result, ValidationResult):
                return result
            elif isinstance(result, bool):
                if result:
                    return ValidationResult.success(value)
                else:
                    return ValidationResult.failure(value, [self.error_message])
            else:
                return ValidationResult.failure(  # type: ignore[unreachable]
                    value,
                    [f"Custom validator returned unexpected type: {type(result).__name__}"]
                )
        except Exception as e:
            return ValidationResult.failure(
                value,
                [f"Custom validation error: {e!s}"]
            )
