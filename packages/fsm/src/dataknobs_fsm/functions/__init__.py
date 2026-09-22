# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""FSM functions module."""

from dataknobs_fsm.functions.base import (
    BaseFunction,
    CompositeFunction,
    ConfigurationError,
    ExecutionResult,
    FSMError,
    Function,
    FunctionContext,
    FunctionError,
    FunctionRegistry,
    FunctionType,
    IEndStateTestFunction,
    IResource,
    IStateTestFunction,
    ITransformFunction,
    IValidationFunction,
    RegisteredFunction,
    ResourceConfig,
    ResourceError,
    ResourceStatus,
    StateTransitionError,
    TransformError,
    TransformOutcome,
    ValidationError,
    ValidationOutcome,
)

__all__ = [
    # Enums
    "FunctionType",
    "ResourceStatus",
    # Result classes
    "ExecutionResult",
    "FunctionContext",
    # Core Classes
    "Function",
    "FunctionRegistry",
    # Interfaces
    "IValidationFunction",
    "ITransformFunction",
    "IStateTestFunction",
    "IEndStateTestFunction",
    "IResource",
    # What the interfaces hand back, and what may be registered by name
    "TransformOutcome",
    "ValidationOutcome",
    "RegisteredFunction",
    # Config classes
    "ResourceConfig",
    # Exceptions
    "FSMError",
    "ValidationError",
    "TransformError",
    "StateTransitionError",
    "FunctionError",
    "ResourceError",
    "ConfigurationError",
    # Base classes
    "BaseFunction",
    "CompositeFunction",
]
