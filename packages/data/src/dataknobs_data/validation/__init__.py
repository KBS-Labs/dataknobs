"""
Validation Module v2 - Clean, predictable validation API.

This module provides a complete rewrite of the validation system with:
- Consistent API across all components
- Predictable return types (always ValidationResult)
- Composable constraints with AND/OR operators
- Fluent schema definition API
- Clear separation of concerns
"""

from .result import ValidationResult, ValidationContext
from .constraints import (
    Constraint,
    All,
    Any,
    Required,
    Range,
    Length,
    Pattern,
    Enum,
    Unique,
    Custom,
)
from .schema import Schema, Field
from .coercer import Coercer
from .factory import schema_factory, coercer_factory, SchemaFactory, CoercerFactory

__all__ = [
    # Result types
    "ValidationResult",
    "ValidationContext",
    # Constraints
    "Constraint",
    "All",
    "Any",
    "Required",
    "Range",
    "Length",
    "Pattern",
    "Enum",
    "Unique",
    "Custom",
    # Schema
    "Schema",
    "Field",
    # Coercion
    "Coercer",
    # Factories
    "schema_factory",
    "coercer_factory",
    "SchemaFactory",
    "CoercerFactory",
]