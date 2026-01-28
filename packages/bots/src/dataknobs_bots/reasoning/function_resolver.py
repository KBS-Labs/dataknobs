"""Function resolution utilities for loading callables from module paths.

This module provides utilities for resolving function references
specified as strings (e.g., "module.path:function_name") to actual
callable objects. Used by WizardConfigLoader and WizardHooks.
"""

import importlib
import logging
from typing import Any, Callable

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
        ValueError: If reference format is invalid or empty
        ImportError: If module cannot be imported
        AttributeError: If function not found in module

    Example:
        ```python
        # Colon format (preferred)
        func = resolve_function("myapp.utils:validate_email")

        # Dot format (also accepted)
        func = resolve_function("myapp.utils.validate_email")
        ```
    """
    if not func_ref or not func_ref.strip():
        raise ValueError(
            "Empty function reference. Expected format: 'module.path:function_name' "
            "or 'module.path.function_name'"
        )

    func_ref = func_ref.strip()

    # Determine module and function name
    if ":" in func_ref:
        # Explicit format: module.path:function_name
        parts = func_ref.split(":", 1)
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError(
                f"Invalid function reference: '{func_ref}'. "
                f"Expected format: 'module.path:function_name'. "
                f"Example: 'myapp.transforms:my_function'"
            )
        module_path, func_name = parts
    else:
        # Dot format: treat last segment as function name
        if "." not in func_ref:
            raise ValueError(
                f"Invalid function reference: '{func_ref}'. "
                f"Expected format: 'module.path:function_name' "
                f"or 'module.path.function_name'. "
                f"Reference must contain at least one '.' or ':'. "
                f"Example: 'myapp.transforms:my_function'"
            )
        parts = func_ref.rsplit(".", 1)
        module_path, func_name = parts

    # Validate module path and function name
    if not module_path:
        raise ValueError(
            f"Invalid function reference: '{func_ref}'. "
            f"Module path is empty. "
            f"Example: 'myapp.transforms:my_function'"
        )

    if not func_name:
        raise ValueError(
            f"Invalid function reference: '{func_ref}'. "
            f"Function name is empty. "
            f"Example: 'myapp.transforms:my_function'"
        )

    # Import module
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(
            f"Cannot import module '{module_path}' from reference '{func_ref}': {e}. "
            f"Ensure the module is installed and the path is correct."
        ) from e

    # Get function from module
    if not hasattr(module, func_name):
        # List available functions for helpful error
        available = [
            name for name in dir(module)
            if callable(getattr(module, name, None)) and not name.startswith("_")
        ]
        available_str = ", ".join(available[:10])
        if len(available) > 10:
            available_str += f", ... ({len(available) - 10} more)"

        raise AttributeError(
            f"Function '{func_name}' not found in module '{module_path}'. "
            f"Available functions: {available_str or '(none)'}"
        )

    func = getattr(module, func_name)
    if not callable(func):
        raise ValueError(
            f"'{func_name}' in module '{module_path}' is not callable "
            f"(got {type(func).__name__})"
        )

    return func


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
        ValueError: If a string reference has invalid format
        ImportError: If a referenced module cannot be imported
        AttributeError: If a referenced function is not found

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
