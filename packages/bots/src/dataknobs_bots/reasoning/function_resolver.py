"""Function resolution utilities for loading callables from module paths.

This module provides utilities for resolving function references
specified as strings (e.g., "module.path:function_name") to actual
callable objects. Used by WizardConfigLoader and WizardHooks.
"""

import logging
from typing import Any, Callable

from dataknobs_common.imports import resolve_callable

logger = logging.getLogger(__name__)


def resolve_function(func_ref: str) -> Callable[..., Any]:
    """Resolve a function reference string to a callable.

    Supports two formats:
    - "module.path:function_name" (preferred, explicit)
    - "module.path.function_name" (accepted, last segment is function)

    Args:
        func_ref: Function reference string

    Returns:
        The resolved callable

    Raises:
        DottedPathError: If the reference is malformed, the module cannot be
            imported, the function is not found, or it is not callable. These
            were three separate stdlib exception types — ``ValueError``,
            ``ImportError``, ``AttributeError`` — before the dotted-path
            resolvers were consolidated.

    Warning:
        Resolving a reference **imports and executes** the target module. See
        :mod:`dataknobs_common.imports` for the trust boundary that implies.

    Example:
        ```python
        # Colon format (preferred)
        func = resolve_function("myapp.utils:validate_email")

        # Dot format (also accepted)
        func = resolve_function("myapp.utils.validate_email")
        ```
    """
    return resolve_callable(func_ref)


def resolve_functions(
    func_refs: dict[str, str | Callable[..., Any]],
) -> dict[str, Callable[..., Any]]:
    """Resolve a dict of function references to callables.

    Values that are already callables are passed through unchanged.
    String values are resolved using resolve_function().

    Args:
        func_refs: Dict mapping names to either:
            - String references ("module:function" or "module.function")
            - Callable objects (passed through)

    Returns:
        Dict mapping names to resolved callables

    Raises:
        DottedPathError: If a string reference cannot be resolved to a
            callable.
        ValueError: If a value is neither a string nor a callable.

    Example:
        ```python
        refs = {
            "validate": "myapp.validators:validate_data",
            "transform": some_callable,  # Already a callable
        }
        resolved = resolve_functions(refs)
        # resolved["validate"] is now the actual function
        # resolved["transform"] is unchanged
        ```
    """
    resolved: dict[str, Callable[..., Any]] = {}

    for name, ref in func_refs.items():
        if callable(ref):
            # Already a callable, use as-is
            resolved[name] = ref
        elif isinstance(ref, str):
            # String reference, resolve it (raises on failure)
            resolved[name] = resolve_function(ref)
        else:
            raise ValueError(
                f"Invalid function reference type for '{name}': "
                f"expected string or callable, got {type(ref).__name__}"
            )

    return resolved
