"""Schema definition and validation."""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union

from dataknobs_data.fields import Field, FieldType
from dataknobs_data.records import Record
from .constraints import Constraint
from .type_coercion import TypeCoercer

logger = logging.getLogger(__name__)


@dataclass
class ValidationError:
    """Represents a validation error."""
    field_name: str
    error_type: str
    message: str
    value: Any = None
    
    def __str__(self) -> str:
        if self.value is not None:
            return f"{self.field_name}: {self.message} (value: {self.value})"
        return f"{self.field_name}: {self.message}"


@dataclass
class ValidationResult:
    """Result of schema validation."""
    is_valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def add_error(self, field_name: str, error_type: str, message: str, value: Any = None) -> None:
        """Add a validation error."""
        self.errors.append(ValidationError(field_name, error_type, message, value))
        self.is_valid = False
    
    def add_warning(self, message: str) -> None:
        """Add a validation warning."""
        self.warnings.append(message)
    
    def merge(self, other: "ValidationResult") -> None:
        """Merge another validation result into this one."""
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.is_valid = self.is_valid and other.is_valid
    
    def __str__(self) -> str:
        if self.is_valid:
            return "Validation successful"
        error_messages = [str(error) for error in self.errors]
        return f"Validation failed with {len(self.errors)} error(s):\n" + "\n".join(error_messages)


@dataclass
class FieldDefinition:
    """Definition of a field in a schema."""
    name: str
    type: Union[Type, FieldType, str]
    required: bool = False
    default: Any = None
    constraints: List[Constraint] = field(default_factory=list)
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    custom_validator: Optional[Callable[[Any], bool]] = None
    
    def validate(self, value: Any, coerce: bool = False) -> ValidationResult:
        """Validate a value against this field definition."""
        result = ValidationResult(is_valid=True)
        
        # Check required
        if self.required and (value is None or value == ""):
            result.add_error(self.name, "required", "Field is required", value)
            return result
        
        # Skip further validation if value is None and not required
        if value is None and not self.required:
            return result
        
        # Type coercion if requested
        if coerce:
            try:
                coercer = TypeCoercer()
                value = coercer.coerce(value, self.type)
            except Exception as e:
                result.add_error(self.name, "type", f"Type coercion failed: {e}", value)
                return result
        
        # Type validation
        if not self._validate_type(value):
            expected_type = self.type.__name__ if hasattr(self.type, '__name__') else str(self.type)
            actual_type = type(value).__name__
            result.add_error(
                self.name,
                "type",
                f"Expected type {expected_type}, got {actual_type}",
                value
            )
            return result
        
        # Apply constraints
        for constraint in self.constraints:
            if not constraint.validate(value):
                result.add_error(
                    self.name,
                    constraint.name,
                    constraint.get_error_message(value),
                    value
                )
        
        # Custom validation
        if self.custom_validator:
            try:
                if not self.custom_validator(value):
                    result.add_error(self.name, "custom", "Custom validation failed", value)
            except Exception as e:
                result.add_error(self.name, "custom", f"Custom validation error: {e}", value)
        
        return result
    
    def _validate_type(self, value: Any) -> bool:
        """Check if value matches the expected type."""
        if isinstance(self.type, type):
            return isinstance(value, self.type)
        elif isinstance(self.type, FieldType):
            # Map FieldType to Python types
            type_map = {
                FieldType.STRING: str,
                FieldType.INTEGER: int,
                FieldType.FLOAT: (int, float),
                FieldType.BOOLEAN: bool,
                FieldType.DATETIME: (datetime, str),  # Allow string for datetime
                FieldType.JSON: (list, dict),  # JSON can be list or dict
                FieldType.TEXT: str,
                FieldType.BINARY: bytes,
            }
            expected = type_map.get(self.type, object)
            return isinstance(value, expected)
        elif isinstance(self.type, str):
            # String type name
            type_map = {
                'str': str,
                'int': int,
                'float': (int, float),
                'bool': bool,
                'list': list,
                'dict': dict,
                'datetime': (datetime, str),
            }
            expected = type_map.get(self.type.lower(), object)
            return isinstance(value, expected)
        
        return True  # Unknown type, allow anything


class Schema:
    """Define and validate record schemas."""
    
    def __init__(
        self,
        fields: Optional[Dict[str, FieldDefinition]] = None,
        name: str = "",
        version: str = "1.0.0",
        strict: bool = False
    ):
        """Initialize schema.
        
        Args:
            fields: Field definitions
            name: Schema name
            version: Schema version
            strict: If True, reject records with extra fields
        """
        self.fields = fields or {}
        self.name = name
        self.version = version
        self.strict = strict
        self._field_index: Dict[str, FieldDefinition] = {}
        self._build_index()
    
    def _build_index(self) -> None:
        """Build field index for faster lookups."""
        self._field_index = {field.name: field for field in self.fields.values()}
    
    def add_field(self, field_def: FieldDefinition) -> None:
        """Add a field definition to the schema."""
        self.fields[field_def.name] = field_def
        self._field_index[field_def.name] = field_def
    
    def remove_field(self, field_name: str) -> None:
        """Remove a field from the schema."""
        if field_name in self.fields:
            del self.fields[field_name]
            del self._field_index[field_name]
    
    def validate(self, record: Record, coerce: bool = False) -> ValidationResult:
        """Validate a record against this schema.
        
        Args:
            record: Record to validate
            coerce: Whether to attempt type coercion
            
        Returns:
            ValidationResult with validation status and errors
        """
        result = ValidationResult(is_valid=True)
        
        # Check for extra fields in strict mode
        if self.strict:
            record_fields = set(record.fields.keys())
            schema_fields = set(self._field_index.keys())
            extra_fields = record_fields - schema_fields
            
            for field_name in extra_fields:
                result.add_error(field_name, "extra_field", "Field not defined in schema")
        
        # Validate each field in schema
        for field_name, field_def in self._field_index.items():
            if field_name in record.fields:
                field_value = record.fields[field_name].value
            else:
                field_value = None
            
            field_result = field_def.validate(field_value, coerce=coerce)
            result.merge(field_result)
        
        return result
    
    def coerce(self, data: Union[Dict[str, Any], Record]) -> Record:
        """Coerce raw data to match schema types.
        
        Args:
            data: Raw data dictionary or existing record
            
        Returns:
            Record with coerced field values
        """
        if isinstance(data, Record):
            record = data.copy()
        else:
            record = Record()
            for key, value in data.items():
                record.fields[key] = Field(name=key, value=value)
        
        coercer = TypeCoercer()
        
        for field_name, field_def in self._field_index.items():
            if field_name in record.fields:
                try:
                    current_value = record.fields[field_name].value
                    coerced_value = coercer.coerce(current_value, field_def.type)
                    record.fields[field_name].value = coerced_value
                    record.fields[field_name].type = field_def.type
                except Exception as e:
                    logger.warning(f"Failed to coerce {field_name}: {e}")
                    # Use default if coercion fails
                    if field_def.default is not None:
                        record.fields[field_name].value = field_def.default
            elif field_def.default is not None:
                # Add field with default value
                record.fields[field_name] = Field(
                    name=field_name,
                    value=field_def.default,
                    type=field_def.type,
                    metadata=field_def.metadata.copy()
                )
            elif field_def.required:
                # Required field is missing
                raise ValueError(f"Required field '{field_name}' is missing")
        
        # Add schema version to metadata
        if not record.metadata:
            record.metadata = {}
        record.metadata['schema_name'] = self.name
        record.metadata['schema_version'] = self.version
        
        return record
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert schema to dictionary."""
        return {
            'name': self.name,
            'version': self.version,
            'strict': self.strict,
            'fields': {
                name: {
                    'type': str(field_def.type),
                    'required': field_def.required,
                    'default': field_def.default,
                    'description': field_def.description,
                    'metadata': field_def.metadata,
                    'constraints': [c.to_dict() for c in field_def.constraints]
                }
                for name, field_def in self.fields.items()
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Schema":
        """Create schema from dictionary."""
        from .constraints import Constraint
        
        fields = {}
        for field_name, field_data in data.get('fields', {}).items():
            field_def = FieldDefinition(
                name=field_name,
                type=field_data.get('type', 'str'),
                required=field_data.get('required', False),
                default=field_data.get('default'),
                description=field_data.get('description', ''),
                metadata=field_data.get('metadata', {}),
                constraints=[
                    Constraint.from_dict(c)
                    for c in field_data.get('constraints', [])
                ]
            )
            fields[field_name] = field_def
        
        return cls(
            fields=fields,
            name=data.get('name', ''),
            version=data.get('version', '1.0.0'),
            strict=data.get('strict', False)
        )


class SchemaValidator:
    """Batch schema validation with caching."""
    
    def __init__(self, schema: Schema):
        """Initialize validator with a schema."""
        self.schema = schema
        self._validation_cache: Dict[int, ValidationResult] = {}
    
    def validate_batch(
        self,
        records: List[Record],
        coerce: bool = False,
        parallel: bool = False
    ) -> List[ValidationResult]:
        """Validate multiple records.
        
        Args:
            records: Records to validate
            coerce: Whether to attempt type coercion
            parallel: Whether to validate in parallel
            
        Returns:
            List of validation results
        """
        if parallel:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(self.schema.validate, record, coerce)
                    for record in records
                ]
                results = [future.result() for future in futures]
        else:
            results = [self.schema.validate(record, coerce) for record in records]
        
        return results
    
    def validate_with_cache(self, record: Record, coerce: bool = False) -> ValidationResult:
        """Validate with caching based on record hash."""
        record_hash = hash(str(record.to_dict()))
        
        if record_hash in self._validation_cache:
            return self._validation_cache[record_hash]
        
        result = self.schema.validate(record, coerce)
        self._validation_cache[record_hash] = result
        
        return result
    
    def clear_cache(self) -> None:
        """Clear validation cache."""
        self._validation_cache.clear()