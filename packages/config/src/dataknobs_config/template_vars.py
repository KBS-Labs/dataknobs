"""Template variable substitution for nested configuration structures.

This module provides utilities for recursively substituting {{var}} placeholders
in nested dicts and lists with values from a variables map.

Example:
    ```python
    from dataknobs_config import substitute_template_vars

    variables = {"name": "Math Tutor", "count": 10}
    data = {
        "title": "{{name}}",
        "max": "{{count}}",
        "desc": "Has {{count}} items"
    }
    result = substitute_template_vars(data, variables)
    # {'title': 'Math Tutor', 'max': 10, 'desc': 'Has 10 items'}
    ```
"""

import re
from typing import Any


# Pattern to match {{variable}} - captures the variable name
_VAR_PATTERN = re.compile(r"\{\{\s*(\w+)\s*\}\}")


def substitute_template_vars(
    data: Any,
    variables: dict[str, Any],
    *,
    preserve_missing: bool = True,
    type_cast: bool = True,
) -> Any:
    """Recursively substitute {{var}} placeholders in configuration data.

    Walks through nested dicts and lists, replacing {{var}} placeholders
    with values from the variables map. Supports type preservation for
    entire-value placeholders.

    Args:
        data: Configuration data (dict, list, string, or primitive)
        variables: Dictionary of variable names to values
        preserve_missing: If True, leave {{var}} intact when var not in variables.
                         If False, replace missing vars with empty string.
        type_cast: If True, preserve Python types for entire-value placeholders.
                  "{{count}}" with count=10 becomes int 10, not string "10".
                  If False, always return strings for substituted values.

    Returns:
        Data with template variables substituted

    Example:
        >>> variables = {"name": "Math Tutor", "count": 10, "items": ["a", "b"]}
        >>> data = {"title": "{{name}}", "max": "{{count}}", "desc": "Has {{count}} items"}
        >>> substitute_template_vars(data, variables)
        {'title': 'Math Tutor', 'max': 10, 'desc': 'Has 10 items'}

        # Type preservation for entire-value placeholders
        >>> substitute_template_vars({"list": "{{items}}"}, variables)
        {'list': ['a', 'b']}

        # Missing variables preserved by default
        >>> substitute_template_vars("Hello {{name}}, {{unknown}}", {"name": "World"})
        'Hello World, {{unknown}}'
    """
    if isinstance(data, dict):
        return {
            k: substitute_template_vars(
                v, variables, preserve_missing=preserve_missing, type_cast=type_cast
            )
            for k, v in data.items()
        }
    elif isinstance(data, list):
        return [
            substitute_template_vars(
                item, variables, preserve_missing=preserve_missing, type_cast=type_cast
            )
            for item in data
        ]
    elif isinstance(data, str):
        return _substitute_string(data, variables, preserve_missing, type_cast)
    else:
        return data


def _substitute_string(
    value: str,
    variables: dict[str, Any],
    preserve_missing: bool,
    type_cast: bool,
) -> Any:
    """Substitute template variables in a string.

    If the entire string is a single {{var}} placeholder and type_cast is True,
    return the variable's native Python type. Otherwise, return a string with
    all placeholders substituted.

    Args:
        value: String potentially containing {{var}} placeholders
        variables: Dictionary of variable names to values
        preserve_missing: Whether to preserve missing variable placeholders
        type_cast: Whether to preserve types for entire-value placeholders

    Returns:
        Substituted value (may be non-string if type_cast and entire-value)
    """
    # Check for entire-value placeholder: "{{var}}" with optional whitespace
    entire_match = re.fullmatch(r"\s*\{\{\s*(\w+)\s*\}\}\s*", value)
    if entire_match:
        var_name = entire_match.group(1)
        if var_name in variables:
            var_value = variables[var_name]
            # Return native type if type_cast is True
            if type_cast:
                return var_value
            else:
                return str(var_value) if var_value is not None else ""
        elif preserve_missing:
            return value
        else:
            return ""

    # Multiple placeholders or mixed content: always returns string
    def replacer(match: re.Match[str]) -> str:
        var_name = match.group(1)
        if var_name in variables:
            var_value = variables[var_name]
            return str(var_value) if var_value is not None else ""
        elif preserve_missing:
            return match.group(0)  # Keep {{var}} as-is
        else:
            return ""

    return _VAR_PATTERN.sub(replacer, value)
