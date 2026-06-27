"""Built-in function library for FSM operations.

This package provides commonly used functions that can be referenced
in FSM configurations:

- **database**: Database query and manipulation functions
- **transformers**: Data transformation and mapping functions
- **validators**: Data validation functions
- **streaming**: Streaming data processing functions

These functions implement standard FSM interfaces and can be used
directly in FSM state definitions.
"""

from .validators import (
    CompositeValidator,
    DependencyValidator,
    LengthValidator,
    PatternValidator,
    RangeValidator,
    RequiredFieldsValidator,
    SchemaValidator,
    TypeValidator,
    UniqueValidator,
    build_gate_arcs,
    build_record_validator,
)

__all__ = [
    # Record-gate substrate (the consumer extension surface for rolling your
    # own validation gate — accepts a friendly dict schema, an
    # IValidationFunction, or a callable predicate; build_gate_arcs wires the
    # two-arc gate shape both the file-processing and ETL patterns use).
    "build_record_validator",
    "build_gate_arcs",
    # Library validators.
    "RequiredFieldsValidator",
    "SchemaValidator",
    "RangeValidator",
    "PatternValidator",
    "TypeValidator",
    "LengthValidator",
    "UniqueValidator",
    "DependencyValidator",
    "CompositeValidator",
]
