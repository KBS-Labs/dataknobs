"""Environment variable substitution for configuration values.

DEPRECATED: Use :func:`substitute_env_vars` from
:mod:`dataknobs_config.inheritance` (or the top-level
``from dataknobs_config import substitute_env_vars``) instead. This class
is retained as a thin compatibility shim and will be removed in a future
release. See :mod:`dataknobs_config.inheritance` for the canonical helper
and its options (``type_coerce``, ``expand_user_paths``, ``substitute_keys``).
"""

import warnings
from typing import Any

from .inheritance import (
    VAR_PATTERN as _VAR_PATTERN,
)
from .inheritance import (
    RequiredEnvVarError,
    substitute_env_vars,
)


class VariableSubstitution:
    """Deprecated env-var substitution helper.

    Use :func:`substitute_env_vars(data, type_coerce=True, expand_user_paths=False, substitute_keys=False) <substitute_env_vars>`
    from :mod:`dataknobs_config.inheritance` instead. This class is a thin
    shim that preserves the historical semantics of ``VariableSubstitution``:

    - ``type_coerce=True`` — whole-value placeholders coerce to int/float/bool.
    - ``expand_user_paths=False`` — does not expand ``~`` in values.
    - ``substitute_keys=False`` — only values are substituted, not dict keys.

    Will be removed in a future release.
    """

    # Retained for backward compatibility — out-of-tree consumers may
    # introspect this attribute.
    VAR_PATTERN = _VAR_PATTERN

    def __init__(self) -> None:
        warnings.warn(
            "VariableSubstitution is deprecated; use "
            "dataknobs_config.substitute_env_vars(data, type_coerce=True, "
            "expand_user_paths=False, substitute_keys=False) instead. "
            "This class will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )

    def substitute(self, value: Any) -> Any:
        """Recursively substitute environment variables in a value.

        Args:
            value: Value to process (string, dict, list, or other).

        Returns:
            Value with environment variables substituted; whole-value
            ``${VAR}`` placeholders may be coerced to int/float/bool.

        Raises:
            ValueError: If a required environment variable is unset. For
                the bare ``${VAR}`` form, the message format matches the
                historical ``VariableSubstitution`` wording
                (``"Environment variable 'VAR' not found"``). For the
                bash-style ``${VAR:?error_msg}`` form, the canonical
                helper's wording is preserved verbatim
                (``"Required environment variable not set: <error_msg>"``,
                using the variable name as the message when ``error_msg``
                is empty), so user-supplied messages are never reframed
                as variable names.
        """
        try:
            return substitute_env_vars(
                value,
                type_coerce=True,
                expand_user_paths=False,
                substitute_keys=False,
            )
        except RequiredEnvVarError as exc:
            if exc.bash_form:
                raise
            raise ValueError(
                f"Environment variable '{exc.var_name}' not found"
            ) from exc

    def has_variables(self, value: Any) -> bool:
        """Check if a value contains environment variable references.

        Args:
            value: Value to check.

        Returns:
            True if value contains ``${...}`` patterns.
        """
        if isinstance(value, str):
            return bool(self.VAR_PATTERN.search(value))
        if isinstance(value, dict):
            return any(self.has_variables(v) for v in value.values())
        if isinstance(value, list):
            return any(self.has_variables(item) for item in value)
        return False
