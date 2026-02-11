"""Configuration validation for DynaBot configs.

Provides a universal validation result type and a pluggable validation engine
for checking DynaBot configurations against schema, portability, and
completeness requirements.

Example:
    ```python
    from dataknobs_bots.config.validation import ConfigValidator, ValidationResult

    validator = ConfigValidator()
    result = validator.validate({"llm": {"provider": "ollama"}})
    if not result.valid:
        for error in result.errors:
            print(f"Error: {error}")
    ```
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from dataknobs_bots.registry.portability import PortabilityError, validate_portability

if TYPE_CHECKING:
    from .schema import DynaBotConfigSchema

logger = logging.getLogger(__name__)


class ValidatorFn(Protocol):
    """Protocol for validation functions."""

    def __call__(self, config: dict[str, Any]) -> ValidationResult: ...


@dataclass
class ValidationResult:
    """Result of validating a configuration.

    Attributes:
        valid: Whether the configuration passed validation.
        errors: List of error messages (validation failures).
        warnings: List of warning messages (non-blocking issues).
    """

    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def merge(self, other: ValidationResult) -> ValidationResult:
        """Merge another validation result into this one.

        The merged result is valid only if both results are valid.

        Args:
            other: Another validation result to merge.

        Returns:
            A new ValidationResult with combined errors and warnings.
        """
        return ValidationResult(
            valid=self.valid and other.valid,
            errors=self.errors + other.errors,
            warnings=self.warnings + other.warnings,
        )

    @classmethod
    def ok(cls) -> ValidationResult:
        """Create a successful validation result."""
        return cls(valid=True)

    @classmethod
    def error(cls, message: str) -> ValidationResult:
        """Create a failed validation result with a single error.

        Args:
            message: The error message.
        """
        return cls(valid=False, errors=[message])

    @classmethod
    def warning(cls, message: str) -> ValidationResult:
        """Create a successful validation result with a warning.

        Args:
            message: The warning message.
        """
        return cls(valid=True, warnings=[message])

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "valid": self.valid,
            "errors": self.errors,
            "warnings": self.warnings,
        }


class ConfigValidator:
    """Pluggable validation engine for DynaBot configurations.

    Runs a pipeline of validators against a config dict and collects
    all errors and warnings into a single ValidationResult.

    Example:
        ```python
        validator = ConfigValidator()

        # Add custom validator
        def check_api_key(config):
            if "api_key" in str(config):
                return ValidationResult.warning("Config contains an API key")
            return ValidationResult.ok()

        validator.register_validator("api_key_check", check_api_key)
        result = validator.validate(my_config)
        ```
    """

    def __init__(self, schema: DynaBotConfigSchema | None = None) -> None:
        """Initialize the validator.

        Args:
            schema: Optional config schema for schema-based validation.
        """
        self._schema = schema
        self._validators: dict[str, ValidatorFn] = {}

    def register_validator(self, name: str, validator: ValidatorFn) -> None:
        """Register a named validation function.

        Args:
            name: Unique name for this validator.
            validator: Function that takes a config dict and returns ValidationResult.
        """
        self._validators[name] = validator
        logger.debug("Registered validator: %s", name)

    def validate(self, config: dict[str, Any]) -> ValidationResult:
        """Run all validators against a configuration.

        Runs completeness check, schema validation (if schema provided),
        and all registered custom validators.

        Args:
            config: Configuration dictionary to validate.

        Returns:
            Merged ValidationResult from all validators.
        """
        result = self.validate_completeness(config)

        if self._schema is not None:
            result = result.merge(self._schema.validate(config))

        for name, validator in self._validators.items():
            try:
                result = result.merge(validator(config))
            except Exception:
                logger.exception("Validator '%s' raised an exception", name)
                result = result.merge(
                    ValidationResult.error(f"Validator '{name}' failed with an error")
                )

        return result

    def validate_completeness(self, config: dict[str, Any]) -> ValidationResult:
        """Check that a config has the minimum required fields.

        A valid DynaBot config must have at minimum an LLM configuration
        and conversation storage configuration.

        Args:
            config: Configuration dictionary to check.

        Returns:
            ValidationResult with errors for missing required fields.
        """
        result = ValidationResult.ok()

        # Check for LLM config (flat or portable format)
        bot = config.get("bot", config)
        has_llm = "llm" in bot
        if not has_llm:
            result = result.merge(
                ValidationResult.error(
                    "Missing required 'llm' configuration. "
                    "Set llm.provider and llm.model, or use a $resource reference."
                )
            )

        # Check for conversation storage
        has_storage = "conversation_storage" in bot
        if not has_storage:
            result = result.merge(
                ValidationResult.error(
                    "Missing required 'conversation_storage' configuration. "
                    "Set conversation_storage.backend, or use a $resource reference."
                )
            )

        return result

    def validate_portability(self, config: dict[str, Any]) -> ValidationResult:
        """Check that a config is portable across environments.

        Wraps the portability checker from registry.portability to return
        a ValidationResult instead of raising exceptions.

        Args:
            config: Configuration dictionary to check.

        Returns:
            ValidationResult with portability issues as warnings.
        """
        try:
            issues = validate_portability(config, raise_on_error=False)
        except PortabilityError as e:
            return ValidationResult.error(str(e))

        if issues:
            return ValidationResult(
                valid=True,
                warnings=[f"Portability: {issue}" for issue in issues],
            )
        return ValidationResult.ok()

    def validate_component(
        self, component: str, config: dict[str, Any]
    ) -> ValidationResult:
        """Validate a specific component section of the config.

        Args:
            component: Component name (e.g., 'llm', 'memory').
            config: The component's configuration dictionary.

        Returns:
            ValidationResult for that component.
        """
        if self._schema is None:
            return ValidationResult.ok()

        schema = self._schema.get_component_schema(component)
        if schema is None:
            return ValidationResult.warning(
                f"No schema registered for component '{component}'"
            )

        return _validate_against_schema(component, config, schema)


def _validate_against_schema(
    component: str,
    config: dict[str, Any],
    schema: dict[str, Any],
) -> ValidationResult:
    """Validate a config dict against a JSON Schema-like definition.

    Performs basic structural validation: required fields, type checking
    for enum fields, and nested property validation.

    Args:
        component: Component name for error messages.
        config: The configuration to validate.
        schema: JSON Schema-like dictionary.

    Returns:
        ValidationResult with any schema violations.
    """
    result = ValidationResult.ok()
    properties = schema.get("properties", {})
    required = schema.get("required", [])

    for req_field in required:
        if req_field not in config:
            result = result.merge(
                ValidationResult.error(
                    f"Component '{component}' is missing required field '{req_field}'"
                )
            )

    for key, value in config.items():
        if key.startswith("$"):
            continue
        if key in properties:
            prop_schema = properties[key]
            enum_values = prop_schema.get("enum")
            if enum_values is not None and value not in enum_values:
                result = result.merge(
                    ValidationResult.error(
                        f"Component '{component}': field '{key}' has invalid value "
                        f"'{value}'. Valid options: {enum_values}"
                    )
                )

    return result
