"""Configuration template system for DynaBot.

Provides template definition, registration, and application for building
DynaBot configurations from reusable templates with variable substitution.

Example:
    ```python
    from pathlib import Path
    from dataknobs_bots.config.templates import ConfigTemplateRegistry

    registry = ConfigTemplateRegistry()
    registry.load_from_directory(Path("configs/templates"))

    templates = registry.list_templates(tags=["educational"])
    config = registry.apply_template("tutor", {"domain_id": "bio-tutor"})
    ```
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dataknobs_config.template_vars import substitute_template_vars

from .validation import ValidationResult

logger = logging.getLogger(__name__)


@dataclass
class TemplateVariable:
    """Definition of a template variable.

    Attributes:
        name: Variable name used in {{name}} placeholders.
        description: Human-readable description.
        type: Variable type (string, integer, boolean, enum, array).
        required: Whether the variable must be provided.
        default: Default value if not provided.
        choices: Valid values for enum-type variables.
        validation: JSON Schema constraints for the value.
    """

    name: str
    description: str = ""
    type: str = "string"
    required: bool = False
    default: Any = None
    choices: list[Any] | None = None
    validation: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        result: dict[str, Any] = {
            "name": self.name,
            "type": self.type,
            "required": self.required,
        }
        if self.description:
            result["description"] = self.description
        if self.default is not None:
            result["default"] = self.default
        if self.choices is not None:
            result["choices"] = self.choices
        if self.validation is not None:
            result["validation"] = self.validation
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TemplateVariable:
        """Create a TemplateVariable from a dictionary.

        Args:
            data: Dictionary with variable fields.

        Returns:
            A new TemplateVariable instance.
        """
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            type=data.get("type", "string"),
            required=data.get("required", False),
            default=data.get("default"),
            choices=data.get("choices"),
            validation=data.get("validation"),
        )


@dataclass
class ConfigTemplate:
    """A reusable DynaBot configuration template.

    Templates define a configuration structure with variable placeholders
    (``{{var}}``) that are substituted when the template is applied.

    Attributes:
        name: Template identifier (underscores internally).
        description: Human-readable description.
        version: Semantic version string.
        tags: Tags for filtering and categorization.
        variables: List of template variables.
        structure: The config structure with {{var}} placeholders.
    """

    name: str
    description: str = ""
    version: str = "1.0.0"
    tags: list[str] = field(default_factory=list)
    variables: list[TemplateVariable] = field(default_factory=list)
    structure: dict[str, Any] = field(default_factory=dict)

    def get_required_variables(self) -> list[TemplateVariable]:
        """Get variables that must be provided."""
        return [v for v in self.variables if v.required]

    def get_optional_variables(self) -> list[TemplateVariable]:
        """Get variables that have defaults or are not required."""
        return [v for v in self.variables if not v.required]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "tags": self.tags,
            "variables": [v.to_dict() for v in self.variables],
            "structure": self.structure,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConfigTemplate:
        """Create a ConfigTemplate from a dictionary.

        Args:
            data: Dictionary with template fields.

        Returns:
            A new ConfigTemplate instance.
        """
        variables = [
            TemplateVariable.from_dict(v) for v in data.get("variables", [])
        ]
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            version=data.get("version", "1.0.0"),
            tags=data.get("tags", []),
            variables=variables,
            structure=data.get("structure", {}),
        )

    @classmethod
    def from_yaml_file(cls, path: Path) -> ConfigTemplate:
        """Load a ConfigTemplate from a YAML file.

        Args:
            path: Path to the YAML file.

        Returns:
            A new ConfigTemplate instance.

        Raises:
            FileNotFoundError: If the file does not exist.
            yaml.YAMLError: If the file is not valid YAML.
        """
        with open(path) as f:
            data = yaml.safe_load(f)
        if data is None:
            data = {}
        template = cls.from_dict(data)
        if not template.name:
            template.name = path.stem.replace("-", "_")
        return template


class ConfigTemplateRegistry:
    """Registry for managing and applying configuration templates.

    Supports registration, tag-based filtering, variable validation,
    and template application with variable substitution.
    """

    def __init__(self) -> None:
        self._templates: dict[str, ConfigTemplate] = {}

    def register(self, template: ConfigTemplate) -> None:
        """Register a template.

        Args:
            template: The template to register.
        """
        self._templates[template.name] = template
        logger.debug("Registered template: %s", template.name)

    def get(self, name: str) -> ConfigTemplate | None:
        """Get a template by name.

        Args:
            name: Template name.

        Returns:
            The template, or None if not found.
        """
        return self._templates.get(name)

    def list_templates(
        self, tags: list[str] | None = None
    ) -> list[ConfigTemplate]:
        """List templates, optionally filtered by tags.

        Args:
            tags: If provided, only return templates that have all specified tags.

        Returns:
            List of matching templates.
        """
        templates = list(self._templates.values())
        if tags:
            tag_set = set(tags)
            templates = [t for t in templates if tag_set.issubset(set(t.tags))]
        return templates

    def load_from_file(self, path: Path) -> ConfigTemplate:
        """Load and register a template from a YAML file.

        Args:
            path: Path to the YAML file.

        Returns:
            The loaded template.
        """
        template = ConfigTemplate.from_yaml_file(path)
        self.register(template)
        return template

    def load_from_directory(self, directory: Path) -> int:
        """Load and register all templates from a directory.

        Scans for ``*.yaml`` and ``*.yml`` files, skipping files named
        ``README`` or ``base``.

        Args:
            directory: Directory to scan.

        Returns:
            Number of templates loaded.
        """
        count = 0
        for ext in ("*.yaml", "*.yml"):
            for path in sorted(directory.glob(ext)):
                if path.stem.lower() in ("readme", "base"):
                    continue
                try:
                    self.load_from_file(path)
                    count += 1
                except Exception:
                    logger.exception("Failed to load template from %s", path)
        return count

    def apply_template(
        self,
        name: str,
        variables: dict[str, Any],
    ) -> dict[str, Any]:
        """Apply a template with variable substitution.

        Deep-copies the template structure and substitutes all ``{{var}}``
        placeholders with values from the variables dict.

        Args:
            name: Template name.
            variables: Variable values to substitute.

        Returns:
            The resolved configuration dict.

        Raises:
            KeyError: If the template is not found.
        """
        template = self._templates.get(name)
        if template is None:
            raise KeyError(f"Template not found: {name}")

        # Build full variable map: user values + defaults
        var_map = _build_variable_map(template, variables)

        structure = copy.deepcopy(template.structure)
        result: dict[str, Any] = substitute_template_vars(
            structure, var_map, preserve_missing=True
        )
        return result

    def validate_variables(
        self,
        name: str,
        variables: dict[str, Any],
    ) -> ValidationResult:
        """Validate variables against a template's requirements.

        Checks that required variables are present and that values
        match any defined choices constraints.

        Args:
            name: Template name.
            variables: Variable values to validate.

        Returns:
            ValidationResult with any issues found.
        """
        template = self._templates.get(name)
        if template is None:
            return ValidationResult.error(f"Template not found: {name}")

        result = ValidationResult.ok()

        for var in template.variables:
            if var.required and var.name not in variables:
                if var.default is None:
                    result = result.merge(
                        ValidationResult.error(
                            f"Missing required variable: {var.name}"
                        )
                    )
            if var.choices is not None and var.name in variables:
                value = variables[var.name]
                if value not in var.choices:
                    result = result.merge(
                        ValidationResult.error(
                            f"Variable '{var.name}' has invalid value '{value}'. "
                            f"Valid choices: {var.choices}"
                        )
                    )

        return result


def _build_variable_map(
    template: ConfigTemplate,
    variables: dict[str, Any],
) -> dict[str, Any]:
    """Build a complete variable map from user values and template defaults.

    Args:
        template: The template being applied.
        variables: User-provided variable values.

    Returns:
        Dict mapping variable names to resolved values.
    """
    var_map: dict[str, Any] = {}

    for var in template.variables:
        if var.name in variables:
            var_map[var.name] = variables[var.name]
        elif var.default is not None:
            var_map[var.name] = var.default

    # Include any extra variables not in the template definition
    for key, value in variables.items():
        if key not in var_map:
            var_map[key] = value

    return var_map
