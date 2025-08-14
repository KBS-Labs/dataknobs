"""Field constraints for schema validation."""

import re
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Set, Union


class Constraint(ABC):
    """Base class for field constraints."""
    
    def __init__(self, name: str = ""):
        """Initialize constraint.
        
        Args:
            name: Constraint name for error messages
        """
        self.name = name or self.__class__.__name__
    
    @abstractmethod
    def validate(self, value: Any) -> bool:
        """Validate a value against this constraint.
        
        Args:
            value: Value to validate
            
        Returns:
            True if valid, False otherwise
        """
        pass
    
    @abstractmethod
    def get_error_message(self, value: Any) -> str:
        """Get error message for failed validation.
        
        Args:
            value: The invalid value
            
        Returns:
            Error message string
        """
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert constraint to dictionary."""
        return {
            'type': self.__class__.__name__,
            'name': self.name
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Constraint":
        """Create constraint from dictionary."""
        constraint_type = data.get('type', 'Constraint')
        
        # Map type names to classes
        type_map = {
            'RequiredConstraint': RequiredConstraint,
            'UniqueConstraint': UniqueConstraint,
            'MinValueConstraint': MinValueConstraint,
            'MaxValueConstraint': MaxValueConstraint,
            'MinLengthConstraint': MinLengthConstraint,
            'MaxLengthConstraint': MaxLengthConstraint,
            'PatternConstraint': PatternConstraint,
            'EnumConstraint': EnumConstraint,
            'CustomConstraint': CustomConstraint,
        }
        
        constraint_class = type_map.get(constraint_type, cls)
        return constraint_class(**data)


class RequiredConstraint(Constraint):
    """Constraint that requires a non-null value."""
    
    def __init__(self, allow_empty: bool = False, **kwargs):
        """Initialize required constraint.
        
        Args:
            allow_empty: Whether to allow empty strings/collections
        """
        super().__init__(**kwargs)
        self.allow_empty = allow_empty
    
    def validate(self, value: Any) -> bool:
        """Check if value is not None."""
        if value is None:
            return False
        
        if not self.allow_empty:
            if isinstance(value, (str, list, dict)) and len(value) == 0:
                return False
        
        return True
    
    def get_error_message(self, value: Any) -> str:
        """Get error message."""
        if value is None:
            return "Value is required"
        return "Value cannot be empty"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = super().to_dict()
        data['allow_empty'] = self.allow_empty
        return data


class UniqueConstraint(Constraint):
    """Constraint that requires unique values."""
    
    def __init__(self, scope: str = "global", **kwargs):
        """Initialize unique constraint.
        
        Args:
            scope: Uniqueness scope ('global', 'collection', etc.)
        """
        super().__init__(**kwargs)
        self.scope = scope
        self.seen_values: Set[Any] = set()
    
    def validate(self, value: Any) -> bool:
        """Check if value is unique."""
        if value in self.seen_values:
            return False
        self.seen_values.add(value)
        return True
    
    def get_error_message(self, value: Any) -> str:
        """Get error message."""
        return f"Value must be unique (duplicate: {value})"
    
    def reset(self) -> None:
        """Reset seen values."""
        self.seen_values.clear()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = super().to_dict()
        data['scope'] = self.scope
        return data


class MinValueConstraint(Constraint):
    """Constraint for minimum numeric value."""
    
    def __init__(self, min_value: Union[int, float], inclusive: bool = True, **kwargs):
        """Initialize minimum value constraint.
        
        Args:
            min_value: Minimum allowed value
            inclusive: Whether the minimum is inclusive
        """
        super().__init__(**kwargs)
        self.min_value = min_value
        self.inclusive = inclusive
    
    def validate(self, value: Any) -> bool:
        """Check if value meets minimum."""
        try:
            numeric_value = float(value)
            if self.inclusive:
                return numeric_value >= self.min_value
            return numeric_value > self.min_value
        except (TypeError, ValueError):
            return False
    
    def get_error_message(self, value: Any) -> str:
        """Get error message."""
        op = ">=" if self.inclusive else ">"
        return f"Value must be {op} {self.min_value} (got {value})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = super().to_dict()
        data['min_value'] = self.min_value
        data['inclusive'] = self.inclusive
        return data


class MaxValueConstraint(Constraint):
    """Constraint for maximum numeric value."""
    
    def __init__(self, max_value: Union[int, float], inclusive: bool = True, **kwargs):
        """Initialize maximum value constraint.
        
        Args:
            max_value: Maximum allowed value
            inclusive: Whether the maximum is inclusive
        """
        super().__init__(**kwargs)
        self.max_value = max_value
        self.inclusive = inclusive
    
    def validate(self, value: Any) -> bool:
        """Check if value meets maximum."""
        try:
            numeric_value = float(value)
            if self.inclusive:
                return numeric_value <= self.max_value
            return numeric_value < self.max_value
        except (TypeError, ValueError):
            return False
    
    def get_error_message(self, value: Any) -> str:
        """Get error message."""
        op = "<=" if self.inclusive else "<"
        return f"Value must be {op} {self.max_value} (got {value})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = super().to_dict()
        data['max_value'] = self.max_value
        data['inclusive'] = self.inclusive
        return data


class MinLengthConstraint(Constraint):
    """Constraint for minimum string/collection length."""
    
    def __init__(self, min_length: int, **kwargs):
        """Initialize minimum length constraint.
        
        Args:
            min_length: Minimum required length
        """
        super().__init__(**kwargs)
        self.min_length = min_length
    
    def validate(self, value: Any) -> bool:
        """Check if value meets minimum length."""
        try:
            return len(value) >= self.min_length
        except TypeError:
            return False
    
    def get_error_message(self, value: Any) -> str:
        """Get error message."""
        try:
            actual_length = len(value)
            return f"Length must be at least {self.min_length} (got {actual_length})"
        except TypeError:
            return f"Value must have a length (got {type(value).__name__})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = super().to_dict()
        data['min_length'] = self.min_length
        return data


class MaxLengthConstraint(Constraint):
    """Constraint for maximum string/collection length."""
    
    def __init__(self, max_length: int, **kwargs):
        """Initialize maximum length constraint.
        
        Args:
            max_length: Maximum allowed length
        """
        super().__init__(**kwargs)
        self.max_length = max_length
    
    def validate(self, value: Any) -> bool:
        """Check if value meets maximum length."""
        try:
            return len(value) <= self.max_length
        except TypeError:
            return False
    
    def get_error_message(self, value: Any) -> str:
        """Get error message."""
        try:
            actual_length = len(value)
            return f"Length must be at most {self.max_length} (got {actual_length})"
        except TypeError:
            return f"Value must have a length (got {type(value).__name__})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = super().to_dict()
        data['max_length'] = self.max_length
        return data


class PatternConstraint(Constraint):
    """Constraint for regex pattern matching."""
    
    def __init__(self, pattern: str, flags: int = 0, **kwargs):
        """Initialize pattern constraint.
        
        Args:
            pattern: Regular expression pattern
            flags: Regex flags (e.g., re.IGNORECASE)
        """
        super().__init__(**kwargs)
        self.pattern = pattern
        self.flags = flags
        self._regex = re.compile(pattern, flags)
    
    def validate(self, value: Any) -> bool:
        """Check if value matches pattern."""
        try:
            string_value = str(value)
            return self._regex.match(string_value) is not None
        except Exception:
            return False
    
    def get_error_message(self, value: Any) -> str:
        """Get error message."""
        return f"Value must match pattern: {self.pattern} (got {value})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = super().to_dict()
        data['pattern'] = self.pattern
        data['flags'] = self.flags
        return data


class EnumConstraint(Constraint):
    """Constraint for enumerated values."""
    
    def __init__(self, allowed_values: List[Any], **kwargs):
        """Initialize enum constraint.
        
        Args:
            allowed_values: List of allowed values
        """
        super().__init__(**kwargs)
        self.allowed_values = set(allowed_values)
    
    def validate(self, value: Any) -> bool:
        """Check if value is in allowed set."""
        return value in self.allowed_values
    
    def get_error_message(self, value: Any) -> str:
        """Get error message."""
        allowed_str = ", ".join(str(v) for v in sorted(self.allowed_values))
        return f"Value must be one of: {allowed_str} (got {value})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = super().to_dict()
        data['allowed_values'] = list(self.allowed_values)
        return data


class CustomConstraint(Constraint):
    """Constraint with custom validation function."""
    
    def __init__(
        self,
        validator: Optional[Callable[[Any], bool]] = None,
        error_message: str = "Custom validation failed",
        **kwargs
    ):
        """Initialize custom constraint.
        
        Args:
            validator: Custom validation function
            error_message: Error message for failures
        """
        super().__init__(**kwargs)
        self.validator = validator or (lambda x: True)
        self.error_message = error_message
    
    def validate(self, value: Any) -> bool:
        """Apply custom validation."""
        try:
            return self.validator(value)
        except Exception:
            return False
    
    def get_error_message(self, value: Any) -> str:
        """Get error message."""
        return self.error_message.format(value=value) if "{value}" in self.error_message else self.error_message
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = super().to_dict()
        data['error_message'] = self.error_message
        # Note: Cannot serialize arbitrary functions
        return data