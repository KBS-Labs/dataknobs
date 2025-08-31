"""Schema definition with fluent API for record validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any

from dataknobs_data.fields import FieldType
from dataknobs_data.records import Record

from .coercer import Coercer
from .constraints import Constraint, Required
from .result import ValidationContext, ValidationResult


@dataclass
class Field:
    """Field definition for schema validation.
    
    Note: This is different from dataknobs_data.fields.Field - this defines
    the expected structure and validation rules for a field in a schema.
    """

    name: str
    field_type: FieldType
    required: bool = False
    default: Any = None
    constraints: list[Constraint] = dataclass_field(default_factory=list)
    description: str | None = None

    def add_constraint(self, constraint: Constraint) -> Field:
        """Add a constraint to this field (fluent API).
        
        Args:
            constraint: Constraint to add
            
        Returns:
            Self for chaining
        """
        self.constraints.append(constraint)
        return self

    def validate(
        self,
        value: Any,
        context: ValidationContext | None = None,
        coerce: bool = False
    ) -> ValidationResult:
        """Validate a value against this field definition.
        
        Args:
            value: Value to validate
            context: Optional validation context
            coerce: If True, attempt type coercion
            
        Returns:
            ValidationResult with outcome
        """
        # Handle None values
        if value is None:
            if self.required:
                return ValidationResult.failure(value, [f"Field '{self.name}' is required"])
            elif self.default is not None:
                value = self.default
            else:
                return ValidationResult.success(None)

        # Type coercion if requested
        if coerce and not self._is_correct_type(value):
            coercer = Coercer()
            coerce_result = coercer.coerce(value, self.field_type)
            if not coerce_result.valid:
                return coerce_result
            value = coerce_result.value

        # Type validation
        if not self._is_correct_type(value):
            return ValidationResult.failure(
                value,
                [f"Field '{self.name}' expects type {self.field_type.name}, got {type(value).__name__}"]
            )

        # Apply constraints
        result = ValidationResult.success(value)
        for constraint in self.constraints:
            check_result = constraint.check(value, context)
            if not check_result.valid:
                # Add field name to error messages for clarity
                check_result.errors = [
                    f"Field '{self.name}': {error}" for error in check_result.errors
                ]
                result = result.merge(check_result)

        return result

    def _is_correct_type(self, value: Any) -> bool:
        """Check if value matches expected field type."""
        if value is None:
            return True  # None is handled separately

        type_map: dict[FieldType, type | tuple[type, ...]] = {
            FieldType.STRING: str,
            FieldType.INTEGER: int,
            FieldType.FLOAT: (int, float),  # Accept both
            FieldType.BOOLEAN: bool,
            FieldType.DATETIME: str,  # Will be validated more strictly later
            FieldType.JSON: (dict, list),
            FieldType.BINARY: bytes,
        }

        expected_type = type_map.get(self.field_type)
        if expected_type:
            return isinstance(value, expected_type)
        return True  # Unknown types are considered valid


class Schema:
    """Schema definition with fluent API for validation.
    
    Provides a clean, chainable interface for defining record schemas
    and validating records against them.
    """

    def __init__(self, name: str, strict: bool = False):
        """Initialize schema.
        
        Args:
            name: Schema name for identification
            strict: If True, reject records with unknown fields
        """
        self.name = name
        self.strict = strict
        self.fields: dict[str, Field] = {}
        self.description: str | None = None

    def field(
        self,
        name: str,
        field_type: FieldType | str,
        required: bool = False,
        default: Any = None,
        constraints: list[Constraint] | None = None,
        description: str | None = None
    ) -> Schema:
        """Add a field definition (fluent API).
        
        Args:
            name: Field name
            field_type: Field type (FieldType enum or string)
            required: Whether field is required
            default: Default value if field is missing
            constraints: List of constraints to apply
            description: Field description
            
        Returns:
            Self for chaining
        """
        # Convert string to FieldType if needed
        if isinstance(field_type, str):
            try:
                field_type = FieldType[field_type.upper()]
            except KeyError as e:
                raise ValueError(f"Invalid field type: {field_type}") from e

        # Add Required constraint if field is required
        field_constraints = constraints or []
        if required and not any(isinstance(c, Required) for c in field_constraints):
            field_constraints.insert(0, Required())

        self.fields[name] = Field(
            name=name,
            field_type=field_type,
            required=required,
            default=default,
            constraints=field_constraints,
            description=description
        )
        return self

    def with_description(self, description: str) -> Schema:
        """Set schema description (fluent API).
        
        Args:
            description: Schema description
            
        Returns:
            Self for chaining
        """
        self.description = description
        return self

    def validate(
        self,
        record: Record | dict[str, Any],
        coerce: bool = False,
        context: ValidationContext | None = None
    ) -> ValidationResult:
        """Validate a record against this schema.
        
        Args:
            record: Record or dict to validate
            coerce: If True, attempt type coercion
            context: Optional validation context
            
        Returns:
            ValidationResult with validation outcome
        """
        if context is None:
            context = ValidationContext()

        # Convert dict to Record if needed
        if isinstance(record, dict):
            record = Record(data=record)

        errors = []
        warnings = []
        validated_fields = {}

        # Validate defined fields
        for field_name, field_def in self.fields.items():
            field_value = record.get_value(field_name)

            # Validate field
            result = field_def.validate(field_value, context, coerce)

            if not result.valid:
                errors.extend(result.errors)
            else:
                validated_fields[field_name] = result.value

            warnings.extend(result.warnings)

        # Check for unknown fields if strict mode
        if self.strict:
            unknown_fields = set(record.fields.keys()) - set(self.fields.keys())
            if unknown_fields:
                errors.append(f"Unknown fields in strict mode: {', '.join(unknown_fields)}")

        # Create validated record with coerced values
        if errors:
            return ValidationResult.failure(record, errors, warnings)
        else:
            # Create new record with validated/coerced values
            validated_record = Record(
                data=validated_fields,
                metadata=record.metadata,
                id=record.id
            )
            return ValidationResult.success(validated_record, warnings)

    def validate_many(
        self,
        records: list[Record | dict[str, Any]],
        coerce: bool = False,
        stop_on_error: bool = False
    ) -> list[ValidationResult]:
        """Validate multiple records.
        
        Args:
            records: List of records to validate
            coerce: If True, attempt type coercion
            stop_on_error: If True, stop validation on first error
            
        Returns:
            List of ValidationResults
        """
        context = ValidationContext()  # Shared context for uniqueness checks
        results = []

        for record in records:
            result = self.validate(record, coerce, context)
            results.append(result)

            if not result.valid and stop_on_error:
                break

        return results

    def to_dict(self) -> dict[str, Any]:
        """Convert schema to dictionary representation.
        
        Returns:
            Dictionary representation of schema
        """
        return {
            "name": self.name,
            "strict": self.strict,
            "description": self.description,
            "fields": {
                name: {
                    "type": field_def.field_type.name,
                    "required": field_def.required,
                    "default": field_def.default,
                    "description": field_def.description,
                    "constraints": len(field_def.constraints)
                }
                for name, field_def in self.fields.items()
            }
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Schema:
        """Create schema from dictionary representation.
        
        Args:
            data: Dictionary with schema definition
            
        Returns:
            Schema instance
        """
        schema = cls(
            name=data.get("name", "unnamed"),
            strict=data.get("strict", False)
        )
        schema.description = data.get("description")

        # Add fields
        fields = data.get("fields", {})
        for field_name, field_data in fields.items():
            schema.field(
                name=field_name,
                field_type=field_data.get("type", "STRING"),
                required=field_data.get("required", False),
                default=field_data.get("default"),
                description=field_data.get("description")
            )

        return schema
