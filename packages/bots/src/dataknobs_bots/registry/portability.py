"""Portability validation utilities for bot configurations.

This module provides utilities to validate that bot configurations
are portable across environments. Portable configs use $resource
references instead of hardcoded values like local paths or localhost URLs.
"""

from __future__ import annotations

import re
from typing import Any


class PortabilityError(Exception):
    r"""Raised when a config contains non-portable values.

    Non-portable values include:
    - Local file paths (/Users/..., /home/..., C:\Users\...)
    - Localhost URLs (localhost:port, 127.0.0.1, 0.0.0.0)

    Portable configs should use $resource references that are
    resolved at runtime based on the environment.

    Example:
        ```python
        # This will raise PortabilityError
        validate_portability({
            "storage": {"path": "/Users/dev/data"}  # Local path!
        })

        # This is OK
        validate_portability({
            "storage": {"$resource": "default", "type": "databases"}
        })
        ```
    """

    pass


# Patterns that indicate resolved local values (not portable)
# Note: Windows paths may appear with single or double backslashes depending
# on whether we're matching against repr() output or actual string values
SUSPICIOUS_PATTERNS: list[tuple[str, str]] = [
    (r"/Users/\w+", "macOS home directory"),
    (r"/home/\w+", "Linux home directory"),
    (r"C:\\+Users\\+\w+", "Windows home directory"),  # Matches C:\Users or C:\\Users
    (r"localhost:\d+", "localhost with port"),
    (r"127\.0\.0\.1", "localhost IP"),
    (r"0\.0\.0\.0", "all interfaces IP"),
]

# Patterns that are OK (environment variable placeholders)
SAFE_PATTERNS: list[str] = [
    r"\$\{[^}]+\}",  # ${VAR} or ${VAR:default}
    r"\$[A-Z_][A-Z0-9_]*",  # $VAR
]


def validate_portability(
    config: dict[str, Any],
    raise_on_error: bool = True,
) -> list[str]:
    """Validate that a config is portable (no resolved local values).

    Checks for patterns that indicate resolved environment values
    that would break portability across environments.

    Args:
        config: Configuration dictionary to validate
        raise_on_error: If True, raise PortabilityError; otherwise return issues

    Returns:
        List of portability issues found (empty if portable)

    Raises:
        PortabilityError: If non-portable and raise_on_error=True

    Example:
        ```python
        # This will raise PortabilityError
        validate_portability({
            "llm": {"api_key": "sk-..."},  # OK - not a path
            "storage": {"path": "/Users/dev/data"},  # NOT OK - local path
        })

        # Check without raising
        issues = validate_portability(config, raise_on_error=False)
        if issues:
            print(f"Found {len(issues)} portability issues")

        # This is OK - uses $resource references
        validate_portability({
            "llm": {"$resource": "default", "type": "llm_providers"},
            "storage": {"$resource": "db", "type": "databases"},
        })

        # Environment variables are OK
        validate_portability({
            "storage": {"path": "${DATA_PATH}"},  # OK - env var placeholder
        })
        ```
    """
    config_str = str(config)
    issues: list[str] = []

    for pattern, description in SUSPICIOUS_PATTERNS:
        matches = re.findall(pattern, config_str)
        for match in matches:
            # Check if this match is inside a safe pattern (env var)
            is_safe = _is_in_safe_pattern(match, config_str)

            if not is_safe:
                issues.append(f"Found {description}: '{match}'")

    if issues and raise_on_error:
        raise PortabilityError(
            "Config appears to contain resolved local values that would break "
            "portability. Store portable config with $resource references instead.\n"
            "Issues found:\n" + "\n".join(f"  - {issue}" for issue in issues)
        )

    return issues


def _is_in_safe_pattern(match: str, config_str: str) -> bool:
    """Check if a suspicious match is inside a safe pattern (env var).

    Args:
        match: The suspicious string that was matched
        config_str: The full config string

    Returns:
        True if the match appears inside an env var pattern
    """
    for safe_pattern in SAFE_PATTERNS:
        # Check if the suspicious pattern appears inside a safe pattern
        # e.g., "${HOME}/data" contains "/home" but it's inside ${...}
        combined_pattern = f"{safe_pattern}[^'\"]*{re.escape(match)}"
        if re.search(combined_pattern, config_str):
            return True
    return False


def has_resource_references(config: dict[str, Any]) -> bool:
    """Check if config contains $resource references.

    $resource references indicate a portable config that needs
    environment resolution before use.

    Args:
        config: Configuration dictionary

    Returns:
        True if config contains $resource references

    Example:
        ```python
        # Portable config with $resource refs
        config = {
            "bot": {
                "llm": {"$resource": "default", "type": "llm_providers"},
            }
        }
        assert has_resource_references(config) is True

        # Resolved config (no $resource refs)
        config = {
            "bot": {
                "llm": {"provider": "openai", "model": "gpt-4"},
            }
        }
        assert has_resource_references(config) is False
        ```
    """
    return "$resource" in str(config)


def is_portable(config: dict[str, Any]) -> bool:
    """Check if config appears to be portable.

    A config is considered portable if it either:
    - Contains $resource references (for late binding), or
    - Contains no suspicious local values

    Args:
        config: Configuration dictionary

    Returns:
        True if config appears to be portable

    Example:
        ```python
        # Portable: uses $resource
        assert is_portable({"llm": {"$resource": "default"}}) is True

        # Portable: no local paths
        assert is_portable({"llm": {"provider": "openai"}}) is True

        # Not portable: contains local path
        assert is_portable({"path": "/Users/dev/data"}) is False
        ```
    """
    # If it has $resource refs, it's portable (will be resolved later)
    if has_resource_references(config):
        return True

    # Otherwise, check for suspicious patterns
    issues = validate_portability(config, raise_on_error=False)
    return len(issues) == 0
