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
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from dataknobs_bots.registry.portability import PortabilityError, validate_portability
from dataknobs_config import RESOURCE_MARKER_KEYS

if TYPE_CHECKING:
    from .schema import DynaBotConfigSchema

logger = logging.getLogger(__name__)


class ValidatorFn(Protocol):
    """Protocol for validation functions."""

    def __call__(self, config: dict[str, Any]) -> ValidationResult: ...


def _unique(messages: list[str]) -> list[str]:
    """Return *messages* with duplicates removed, first occurrence kept."""
    return list(dict.fromkeys(messages))


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
        Messages are concatenated in order and kept as they are: a
        validator reporting the same string twice is reporting two
        findings, and deciding they are one is a judgement this
        primitive is in no position to make. Use :meth:`merge_unique`
        where the repetition comes from the composition rather than
        from the config.

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

    def merge_unique(self, other: ValidationResult) -> ValidationResult:
        """Merge *other*, reporting an identical message once.

        For composing validators that cover overlapping ground. Two
        validators both running ``validate_completeness`` over one
        config each find the same missing key, and the second copy of
        that message tells the reader nothing -- the repetition is an
        artefact of running two validators, not a second defect.

        Distinct messages are never collapsed and order is preserved,
        so this is only safe where a repeated string genuinely means one
        finding. That is a property of the composition, which is why the
        caller chooses it rather than :meth:`merge` deciding for every
        caller in the package.

        Args:
            other: Another validation result to merge.

        Returns:
            A new ValidationResult with combined messages, de-duplicated.
        """
        return ValidationResult(
            valid=self.valid and other.valid,
            errors=_unique(self.errors + other.errors),
            warnings=_unique(self.warnings + other.warnings),
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
                    "Set conversation_storage.backend, "
                    "conversation_storage.storage_class, "
                    "or use a $resource reference."
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

    def validate_component(self, component: str, config: dict[str, Any]) -> ValidationResult:
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
            return ValidationResult.warning(f"No schema registered for component '{component}'")

        return _validate_against_schema(component, config, schema)


#: Schema fields whose valid values come from a live registry rather than a
#: fixed list, keyed by the name a property's ``enum_registry`` declares.
#:
#: A field belongs here when its value set is *open* — extensible by consumers
#: at runtime — so transcribing it into an ``enum`` would both drift from the
#: registry and reject anything registered after the literal was written. A
#: genuinely closed set (``memory.type``, ``reasoning.strategy``) keeps a plain
#: ``enum``.
#:
#: Lives in this module rather than beside the schema definitions because
#: ``schema`` already imports from ``validation``; putting it the other way
#: round would make the two mutually dependent.
_ENUM_REGISTRIES: dict[str, Callable[[], list[str]]] = {}


def _llm_provider_families() -> list[str]:
    """Registered LLM provider family keys.

    Imported lazily: config validation is reachable without ever building a
    provider, and a top-level import would pull the whole LLM provider package
    in behind it.
    """
    from dataknobs_llm import LLMProviderFactory

    return LLMProviderFactory.list_providers()


_ENUM_REGISTRIES["llm_providers"] = _llm_provider_families


def resolve_enum_options(prop_schema: dict[str, Any]) -> list[str] | None:
    """Valid values for a property, or ``None`` when it is unconstrained.

    Resolves ``enum_registry`` against the live registry first, then a literal
    ``enum``. Shared by the validator and by the schema's query/documentation
    surface so the two cannot disagree about what a field accepts — the failure
    guarded against being a validator that rejects a value its own generated
    documentation offers.

    An unknown registry name yields ``None`` (unconstrained) rather than an
    error: a consumer extension may name a registry this build does not have,
    and declining to constrain the field beats rejecting every value for it.

    That leniency is logged at WARNING, not DEBUG, because the outcome is a
    validator quietly switching itself off for that field — indistinguishable,
    from the outside, from a field that passed. A misspelled ``enum_registry``
    would otherwise disable checking with no signal at any normal log level.
    """
    registry_name = prop_schema.get("enum_registry")
    if registry_name is not None:
        resolver = _ENUM_REGISTRIES.get(registry_name)
        if resolver is None:
            logger.warning(
                "No enum registry named %r; leaving the field unconstrained. Known registries: %s",
                registry_name,
                sorted(_ENUM_REGISTRIES),
            )
            return None
        return resolver()
    enum_values = prop_schema.get("enum")
    return list(enum_values) if enum_values is not None else None


def _matches_option(value: Any, options: list[str]) -> bool:
    """Whether *value* is one of *options*, case-insensitively for strings.

    Case folding is not leniency here — it is agreement with the runtime.
    Provider and backend keys resolve through registries built with
    ``canonicalize_keys=True``, so ``provider: OpenAI`` constructs an
    ``OpenAIProvider`` without complaint. A validator that rejects it is
    contradicting the code it validates, and points the author at a working
    line.

    Non-string values (a numeric or boolean enum) compare exactly.
    """
    if isinstance(value, str):
        folded = {opt.lower() for opt in options if isinstance(opt, str)}
        return value.lower() in folded
    return value in options


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

    # A section may be a `$resource` reference rather than a literal config,
    # whose keys are markers and not schema fields. The skip is for those --
    # but a reference's marker vocabulary is closed, so skipping every
    # `$`-prefixed key let a misspelling through the one check that runs
    # before resolution. `$requred: true` reads as *not required*, and
    # deferring it means it first appears in whichever deployment lacks the
    # resource. The set is imported rather than transcribed so this cannot
    # drift from the resolver that enforces it.
    is_reference = "$resource" in config
    for key, value in config.items():
        if key.startswith("$"):
            if is_reference and key not in RESOURCE_MARKER_KEYS:
                markers = ", ".join(
                    sorted(marker for marker in RESOURCE_MARKER_KEYS if marker.startswith("$"))
                )
                result = result.merge(
                    ValidationResult.error(
                        f"Component '{component}': unknown marker key '{key}' in a "
                        f"$resource reference. A $-prefixed key must be one of: "
                        f"{markers}. Anything else is treated as an inline default "
                        f"and passed to a factory as a keyword argument."
                    )
                )
            continue
        if key in properties:
            prop_schema = properties[key]
            # Resolved through the schema module's helper so the validator and
            # the documentation/query surface cannot disagree about what a
            # field accepts, and so a registry-backed field is checked against
            # the live registry rather than a snapshot.
            enum_values = resolve_enum_options(prop_schema)
            if enum_values is not None and not _matches_option(value, enum_values):
                result = result.merge(
                    ValidationResult.error(
                        f"Component '{component}': field '{key}' has invalid value "
                        f"'{value}'. Valid options: {enum_values}"
                    )
                )

    return result
