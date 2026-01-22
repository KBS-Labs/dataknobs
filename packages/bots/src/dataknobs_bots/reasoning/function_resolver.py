"""Function resolution utilities for loading callables from module paths.

This module provides utilities for resolving function references
specified as strings (e.g., "module.path:function_name") to actual
callable objects. Used by WizardConfigLoader and WizardHooks.
"""

import importlib
import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)


def resolve_function(func_ref: str) -> Callable[..., Any] | None:
    """Resolve a function reference string to a callable.

    Supports the format "module.path:function_name" where:
    - module.path is a dotted Python module path
    - function_name is the name of a function/class in that module

    Args:
        func_ref: Function reference string in "module:function" format

    Returns:
        The resolved callable, or None if resolution failed

    Example:
        ```python
        func = resolve_function("myapp.utils:validate_email")
        if func:
            result = func("test@example.com")
        ```
    """
    if not func_ref:
        return None

    # Parse module:function format
    if ":" not in func_ref:
        logger.warning(
            "Invalid function reference format: %s (expected module:func)",
            func_ref,
        )
        return None

    module_path, func_name = func_ref.rsplit(":", 1)

    try:
        module = importlib.import_module(module_path)
        return getattr(module, func_name)
    except ImportError as e:
        logger.warning("Failed to import module %s: %s", module_path, e)
        return None
    except AttributeError as e:
        logger.warning(
            "Function %s not found in module %s: %s",
            func_name,
            module_path,
            e,
        )
        return None


def resolve_functions(
    func_refs: dict[str, str | Callable[..., Any]],
) -> dict[str, Callable[..., Any]]:
    """Resolve a dict of function references to callables.

    Values that are already callables are passed through unchanged.
    String values are resolved using resolve_function().
    Failed resolutions are logged and omitted from the result.

    Args:
        func_refs: Dict mapping names to either:
            - String references ("module:function")
            - Callable objects (passed through)

    Returns:
        Dict mapping names to resolved callables

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
            # String reference, resolve it
            func = resolve_function(ref)
            if func:
                resolved[name] = func
            else:
                logger.warning(
                    "Skipping unresolvable function reference: %s = %s",
                    name,
                    ref,
                )
        else:
            logger.warning(
                "Invalid function reference type for %s: %s",
                name,
                type(ref).__name__,
            )

    return resolved
