"""Custom exceptions for the config package."""


class ConfigError(Exception):
    """Base exception for configuration errors."""

    pass


class ConfigNotFoundError(ConfigError):
    """Raised when a requested configuration is not found."""

    pass


class InvalidReferenceError(ConfigError):
    """Raised when a configuration reference is invalid."""

    pass


class ValidationError(ConfigError):
    """Raised when configuration validation fails."""

    pass


class FileNotFoundError(ConfigError):
    """Raised when a referenced configuration file is not found."""

    pass


class CircularReferenceError(ConfigError):
    """Raised when circular references are detected."""

    pass
