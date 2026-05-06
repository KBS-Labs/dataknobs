"""Environment variable substitution for configuration values.

DEPRECATED: Use :func:`substitute_env_vars` from
:mod:`dataknobs_config.inheritance` (or the top-level
``from dataknobs_config import substitute_env_vars``) instead. This class
is retained as a thin compatibility shim and will be removed in a future
release. See :mod:`dataknobs_config.inheritance` for the canonical helper
and its options (``type_coerce``, ``expand_user_paths``, ``substitute_keys``).
"""

import re
import warnings
from typing import Any

from .inheritance import VAR_PATTERN as _VAR_PATTERN
from .inheritance import substitute_env_vars

# Sentinel prefix from the canonical helper's required-var error message.
# We translate it back to the historical VariableSubstitution wording so
# out-of-tree consumers that ``pytest.raises(match=...)`` on the old
# string keep passing through the deprecation period — but only when the
# suffix is a bare identifier (i.e., the bare ``${VAR}`` form). For the
# bash-style ``${VAR:?msg}`` form the suffix is a free-form message that
# would produce garbled wording if rewritten as a variable name, so we
# pass those errors through unchanged.
_REQUIRED_ERROR_PREFIX = "Required environment variable not set: "
_VAR_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class VariableSubstitution:
    """Deprecated env-var substitution helper.

    Use :func:`substitute_env_vars(data, type_coerce=True) <substitute_env_vars>`
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
            "dataknobs_config.substitute_env_vars(data, type_coerce=True) "
            "instead. This class will be removed in a future release.",
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
                (``"Required environment variable not set: <error_msg>"``)
                because rewriting a free-form error message as a variable
                name would produce garbled output.
        """
        try:
            return substitute_env_vars(
                value,
                type_coerce=True,
                expand_user_paths=False,
                substitute_keys=False,
            )
        except ValueError as exc:
            msg = str(exc)
            if msg.startswith(_REQUIRED_ERROR_PREFIX):
                suffix = msg[len(_REQUIRED_ERROR_PREFIX):]
                if _VAR_NAME_RE.fullmatch(suffix):
                    raise ValueError(
                        f"Environment variable '{suffix}' not found"
                    ) from exc
            raise

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
