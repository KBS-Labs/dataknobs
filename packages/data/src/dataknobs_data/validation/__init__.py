"""Validation Module v2 - Clean, predictable validation API.

This module provides a complete rewrite of the validation system with:
- Consistent API across all components
- Predictable return types (always ValidationResult)
- Composable constraints with AND/OR operators
- Fluent schema definition API
- Clear separation of concerns
"""

from .coercer import Coercer
from .constraints import (
    All,
    AnyOf,
    Constraint,
    Custom,
    Enum,
    Length,
    Pattern,
    Range,
    Required,
    Unique,
)
from .factory import CoercerFactory, SchemaFactory, coercer_factory, schema_factory
from .result import ValidationContext, ValidationResult
from .schema import Field, Schema

__all__ = [
    # Result types
    "ValidationResult",
    "ValidationContext",
    # Constraints
    "Constraint",
    "All",
    "AnyOf",
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
