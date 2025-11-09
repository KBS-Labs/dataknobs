"""Custom exceptions for the config package.

This module defines exception types for the config package,
built on the common exception framework from dataknobs_common.
"""

from dataknobs_common import (
    ConfigurationError as BaseConfigurationError,
    NotFoundError,
    ValidationError as BaseValidationError,
)

# Create ConfigError as alias to ConfigurationError for backward compatibility
ConfigError = BaseConfigurationError


class ConfigNotFoundError(NotFoundError):
    """Raised when a requested configuration is not found."""

    pass


class InvalidReferenceError(BaseValidationError):
    """Raised when a configuration reference is invalid."""

    pass


# Use common ValidationError directly (no config-specific behavior needed)
ValidationError = BaseValidationError


class FileNotFoundError(NotFoundError):
    """Raised when a referenced configuration file is not found."""

    pass


class CircularReferenceError(BaseValidationError):
    """Raised when circular references are detected."""

    pass
