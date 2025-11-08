"""Built-in validator functions for FSM.

This module provides commonly used validation functions that can be
referenced in FSM configurations.
"""

import re
from typing import Any, Dict, List, Union

from pydantic import BaseModel, ValidationError

from dataknobs_fsm.functions.base import IValidationFunction, ValidationError as FSMValidationError


class RequiredFieldsValidator(IValidationFunction):
    """Validate that required fields are present in data."""

    def __init__(self, fields: List[str], allow_none: bool = False):
        """Initialize the validator.
        
        Args:
            fields: List of required field names.
            allow_none: Whether to allow None values for required fields.
        """
        self.fields = fields
        self.allow_none = allow_none

    def validate(self, data: Dict[str, Any]) -> bool:
        """Validate that all required fields are present.
        
        Args:
            data: Data to validate.
            
        Returns:
            True if valid, False otherwise.
            
        Raises:
            FSMValidationError: If validation fails with details.
        """
        if not isinstance(data, dict):
            raise FSMValidationError(f"Expected dict, got {type(data).__name__}")
        
        missing_fields = []
        none_fields = []
        
        for field in self.fields:
            if field not in data:
                missing_fields.append(field)
            elif not self.allow_none and data[field] is None:
                none_fields.append(field)
        
        if missing_fields:
            raise FSMValidationError(
                f"Missing required fields: {', '.join(missing_fields)}"
            )
        
        if none_fields:
            raise FSMValidationError(
                f"Fields cannot be None: {', '.join(none_fields)}"
            )

        return True

    def get_validation_rules(self) -> Dict[str, Any]:
        """Get the validation rules."""
        return {
            "required_fields": self.fields,
            "allow_none": self.allow_none
        }


class SchemaValidator(IValidationFunction):
    """Validate data against a Pydantic schema."""

    def __init__(self, schema: Union[type[BaseModel], Dict[str, Any]]):
        """Initialize the validator.
        
        Args:
            schema: Pydantic model class or schema dictionary.
        """
        if isinstance(schema, dict):
            # Create a dynamic Pydantic model from dictionary
            from pydantic import create_model
            self.schema = create_model("DynamicSchema", **schema)
        else:
            self.schema = schema

    def validate(self, data: Dict[str, Any]) -> bool:
        """Validate data against the schema.
        
        Args:
            data: Data to validate.
            
        Returns:
            True if valid, False otherwise.
            
        Raises:
            FSMValidationError: If validation fails with details.
        """
        try:
            self.schema(**data)
            return True
        except ValidationError as e:
            errors = []
            for error in e.errors():
                field_path = ".".join(str(loc) for loc in error["loc"])
                errors.append(f"{field_path}: {error['msg']}")
            
            raise FSMValidationError(
                f"Schema validation failed: {'; '.join(errors)}"
            ) from e
    
    def get_validation_rules(self) -> Dict[str, Any]:
        """Get the validation rules."""
        if hasattr(self.schema, 'model_json_schema'):
            return self.schema.model_json_schema()
        elif hasattr(self.schema, '__annotations__'):
            return dict(self.schema.__annotations__)
        else:
            return {"schema": str(self.schema)}


class RangeValidator(IValidationFunction):
    """Validate that numeric values are within specified ranges."""

    def __init__(
        self,
        field_ranges: Dict[str, Dict[str, Union[int, float]]],
    ):
        """Initialize the validator.
        
        Args:
            field_ranges: Dictionary mapping field names to range specifications.
                         Each range can have 'min', 'max', or both.
        """
        self.field_ranges = field_ranges

    def validate(self, data: Dict[str, Any]) -> bool:
        """Validate that values are within specified ranges.
        
        Args:
            data: Data to validate.
            
        Returns:
            True if valid, False otherwise.
            
        Raises:
            FSMValidationError: If validation fails with details.
        """
        errors = []
        
        for field, range_spec in self.field_ranges.items():
            if field not in data:
                continue
            
            value = data[field]
            if not isinstance(value, (int, float)):
                errors.append(f"{field}: Expected numeric value, got {type(value).__name__}")
                continue
            
            if "min" in range_spec and value < range_spec["min"]:
                errors.append(f"{field}: Value {value} is below minimum {range_spec['min']}")
            
            if "max" in range_spec and value > range_spec["max"]:
                errors.append(f"{field}: Value {value} is above maximum {range_spec['max']}")
        
        if errors:
            raise FSMValidationError("; ".join(errors))

        return True

    def get_validation_rules(self) -> Dict[str, Any]:
        """Get the validation rules."""
        return {
            "type": "range",
            "field_ranges": self.field_ranges
        }


class PatternValidator(IValidationFunction):
    """Validate that string values match specified patterns."""

    def __init__(
        self,
        field_patterns: Dict[str, str],
        flags: int = 0,
    ):
        """Initialize the validator.
        
        Args:
            field_patterns: Dictionary mapping field names to regex patterns.
            flags: Regex flags to apply (e.g., re.IGNORECASE).
        """
        self.field_patterns = {}
        for field, pattern in field_patterns.items():
            self.field_patterns[field] = re.compile(pattern, flags)

    def validate(self, data: Dict[str, Any]) -> bool:
        """Validate that values match specified patterns.
        
        Args:
            data: Data to validate.
            
        Returns:
            True if valid, False otherwise.
            
        Raises:
            FSMValidationError: If validation fails with details.
        """
        errors = []
        
        for field, pattern in self.field_patterns.items():
            if field not in data:
                continue
            
            value = data[field]
            if not isinstance(value, str):
                errors.append(f"{field}: Expected string value, got {type(value).__name__}")
                continue
            
            if not pattern.match(value):
                errors.append(f"{field}: Value '{value}' does not match pattern")

        if errors:
            raise FSMValidationError("; ".join(errors))

        return True

    def get_validation_rules(self) -> Dict[str, Any]:
        """Get the validation rules."""
        return {
            "type": "pattern",
            "field_patterns": {field: pattern.pattern for field, pattern in self.field_patterns.items()}
        }


class TypeValidator(IValidationFunction):
    """Validate that fields have expected types."""

    def __init__(
        self,
        field_types: Dict[str, Union[type, List[type]]],
        strict: bool = False,
    ):
        """Initialize the validator.
        
        Args:
            field_types: Dictionary mapping field names to expected types.
            strict: If True, reject extra fields not in field_types.
        """
        self.field_types = field_types
        self.strict = strict

    def validate(self, data: Dict[str, Any]) -> bool:
        """Validate that fields have expected types.
        
        Args:
            data: Data to validate.
            
        Returns:
            True if valid, False otherwise.
            
        Raises:
            FSMValidationError: If validation fails with details.
        """
        errors = []
        
        # Check field types
        for field, expected_type in self.field_types.items():
            if field not in data:
                continue
            
            value = data[field]
            if isinstance(expected_type, list):
                # Multiple allowed types
                if not any(isinstance(value, t) for t in expected_type):
                    type_names = ", ".join(t.__name__ for t in expected_type)
                    errors.append(
                        f"{field}: Expected one of [{type_names}], "
                        f"got {type(value).__name__}"
                    )
            else:
                # Single expected type
                if not isinstance(value, expected_type):
                    errors.append(
                        f"{field}: Expected {expected_type.__name__}, "
                        f"got {type(value).__name__}"
                    )
        
        # Check for extra fields if strict mode
        if self.strict:
            extra_fields = set(data.keys()) - set(self.field_types.keys())
            if extra_fields:
                errors.append(f"Unexpected fields: {', '.join(extra_fields)}")

        if errors:
            raise FSMValidationError("; ".join(errors))

        return True

    def get_validation_rules(self) -> Dict[str, Any]:
        """Get the validation rules."""
        field_type_names = {}
        for field, ftype in self.field_types.items():
            if isinstance(ftype, list):
                field_type_names[field] = [t.__name__ for t in ftype]
            else:
                field_type_names[field] = ftype.__name__
        return {
            "type": "type_check",
            "field_types": field_type_names,
            "strict": self.strict
        }


class LengthValidator(IValidationFunction):
    """Validate that collections have expected lengths."""

    def __init__(
        self,
        field_lengths: Dict[str, Dict[str, int]],
    ):
        """Initialize the validator.
        
        Args:
            field_lengths: Dictionary mapping field names to length specifications.
                          Each spec can have 'min', 'max', or 'exact'.
        """
        self.field_lengths = field_lengths

    def validate(self, data: Dict[str, Any]) -> bool:
        """Validate that collections have expected lengths.
        
        Args:
            data: Data to validate.
            
        Returns:
            True if valid, False otherwise.
            
        Raises:
            FSMValidationError: If validation fails with details.
        """
        errors = []
        
        for field, length_spec in self.field_lengths.items():
            if field not in data:
                continue
            
            value = data[field]
            if not hasattr(value, "__len__"):
                errors.append(f"{field}: Value does not have a length")
                continue
            
            length = len(value)
            
            if "exact" in length_spec and length != length_spec["exact"]:
                errors.append(
                    f"{field}: Length {length} does not match expected {length_spec['exact']}"
                )
            
            if "min" in length_spec and length < length_spec["min"]:
                errors.append(f"{field}: Length {length} is below minimum {length_spec['min']}")
            
            if "max" in length_spec and length > length_spec["max"]:
                errors.append(f"{field}: Length {length} is above maximum {length_spec['max']}")

        if errors:
            raise FSMValidationError("; ".join(errors))

        return True

    def get_validation_rules(self) -> Dict[str, Any]:
        """Get the validation rules."""
        return {
            "type": "length",
            "field_lengths": self.field_lengths
        }


class UniqueValidator(IValidationFunction):
    """Validate that values in collections are unique."""

    def __init__(
        self,
        fields: List[str],
        key: str | None = None,
    ):
        """Initialize the validator.
        
        Args:
            fields: List of field names to check for uniqueness.
            key: Optional key to extract from collection items for uniqueness check.
        """
        self.fields = fields
        self.key = key

    def validate(self, data: Dict[str, Any]) -> bool:
        """Validate that values are unique.
        
        Args:
            data: Data to validate.
            
        Returns:
            True if valid, False otherwise.
            
        Raises:
            FSMValidationError: If validation fails with details.
        """
        errors = []
        
        for field in self.fields:
            if field not in data:
                continue
            
            value = data[field]
            if not isinstance(value, (list, tuple, set)):
                errors.append(f"{field}: Expected collection, got {type(value).__name__}")
                continue
            
            if self.key:
                # Extract values using key
                try:
                    values = [item[self.key] if isinstance(item, dict) else getattr(item, self.key)
                             for item in value]
                except (KeyError, AttributeError) as e:
                    errors.append(f"{field}: Cannot extract key '{self.key}': {e}")
                    continue
            else:
                values = list(value)
            
            # Check for duplicates
            seen = set()
            duplicates = set()
            for v in values:
                if v in seen:
                    duplicates.add(str(v))
                seen.add(v)
            
            if duplicates:
                errors.append(f"{field}: Duplicate values found: {', '.join(duplicates)}")

        if errors:
            raise FSMValidationError("; ".join(errors))

        return True

    def get_validation_rules(self) -> Dict[str, Any]:
        """Get the validation rules."""
        return {
            "type": "unique",
            "fields": self.fields,
            "key": self.key
        }


class DependencyValidator(IValidationFunction):
    """Validate field dependencies (if field A exists, field B must also exist)."""

    def __init__(
        self,
        dependencies: Dict[str, Union[str, List[str]]],
    ):
        """Initialize the validator.
        
        Args:
            dependencies: Dictionary mapping field names to their dependencies.
        """
        self.dependencies = dependencies

    def validate(self, data: Dict[str, Any]) -> bool:
        """Validate field dependencies.
        
        Args:
            data: Data to validate.
            
        Returns:
            True if valid, False otherwise.
            
        Raises:
            FSMValidationError: If validation fails with details.
        """
        errors = []
        
        for field, deps in self.dependencies.items():
            if field not in data:
                continue
            
            deps_list = deps if isinstance(deps, list) else [deps]
            
            missing_deps = [dep for dep in deps_list if dep not in data]
            
            if missing_deps:
                errors.append(
                    f"Field '{field}' requires: {', '.join(missing_deps)}"
                )

        if errors:
            raise FSMValidationError("; ".join(errors))

        return True

    def get_validation_rules(self) -> Dict[str, Any]:
        """Get the validation rules."""
        return {
            "type": "dependency",
            "dependencies": self.dependencies
        }


class CompositeValidator(IValidationFunction):
    """Compose multiple validators into a single validator."""

    def __init__(
        self,
        validators: List[IValidationFunction],
        stop_on_first_error: bool = False,
    ):
        """Initialize the composite validator.
        
        Args:
            validators: List of validators to apply.
            stop_on_first_error: If True, stop at first validation error.
        """
        self.validators = validators
        self.stop_on_first_error = stop_on_first_error

    def validate(self, data: Dict[str, Any]) -> bool:
        """Apply all validators to the data.
        
        Args:
            data: Data to validate.
            
        Returns:
            True if all validators pass.
            
        Raises:
            FSMValidationError: If any validation fails.
        """
        errors = []
        
        for validator in self.validators:
            try:
                validator.validate(data)
            except FSMValidationError as e:
                if self.stop_on_first_error:
                    raise
                errors.append(str(e))
        
        if errors:
            raise FSMValidationError("; ".join(errors))
        
        return True


# Convenience functions for creating validators
def required_fields(*fields: str, allow_none: bool = False) -> RequiredFieldsValidator:
    """Create a RequiredFieldsValidator."""
    return RequiredFieldsValidator(list(fields), allow_none)


def schema(model: Union[type[BaseModel], Dict[str, Any]]) -> SchemaValidator:
    """Create a SchemaValidator."""
    return SchemaValidator(model)


def range_check(**field_ranges: Dict[str, Union[int, float]]) -> RangeValidator:
    """Create a RangeValidator."""
    return RangeValidator(field_ranges)


def pattern(**field_patterns: str) -> PatternValidator:
    """Create a PatternValidator."""
    return PatternValidator(field_patterns)


def type_check(**field_types: Union[type, List[type]]) -> TypeValidator:
    """Create a TypeValidator."""
    return TypeValidator(field_types)


def length(**field_lengths: Dict[str, int]) -> LengthValidator:
    """Create a LengthValidator."""
    return LengthValidator(field_lengths)


def unique(*fields: str, key: str | None = None) -> UniqueValidator:
    """Create a UniqueValidator."""
    return UniqueValidator(list(fields), key)


def depends_on(**dependencies: Union[str, List[str]]) -> DependencyValidator:
    """Create a DependencyValidator."""
    return DependencyValidator(dependencies)
