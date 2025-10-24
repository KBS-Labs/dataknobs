"""Template renderer with validation support.

This module wraps the CONDITIONAL template strategy from dataknobs_llm.llm.utils
and adds validation capabilities for missing parameters.
"""

import re
import logging
from typing import Any, Dict, List, Optional, Set

from dataknobs_llm.llm.utils import render_conditional_template
from ..base.types import (
    ValidationLevel,
    ValidationConfig,
    PromptTemplate,
    RenderResult
)

logger = logging.getLogger(__name__)


class TemplateRenderer:
    """Template renderer with configurable validation.

    This class provides:
    - Rendering using CONDITIONAL template strategy
    - Validation of required parameters (ERROR/WARN/IGNORE)
    - Tracking of used and missing parameters
    - Detailed render results with metadata
    """

    def __init__(
        self,
        default_validation: ValidationLevel = ValidationLevel.WARN
    ):
        """Initialize the template renderer.

        Args:
            default_validation: Default validation level for templates
                               without explicit validation configuration
        """
        self._default_validation = default_validation

    def render(
        self,
        template: str,
        params: Dict[str, Any],
        validation: Optional[ValidationConfig] = None,
        template_metadata: Optional[Dict[str, Any]] = None
    ) -> RenderResult:
        """Render a template with parameters and validation.

        Args:
            template: Template string with {{variables}} and ((conditionals))
            params: Parameters to substitute in the template
            validation: Optional validation configuration (overrides template default)
            template_metadata: Optional metadata about the template

        Returns:
            RenderResult with rendered content and validation information

        Raises:
            ValueError: If validation level is ERROR and required params are missing
        """
        # Use provided validation or create default
        if validation is None:
            validation = ValidationConfig(level=self._default_validation)

        # Determine effective validation level (inherit from renderer if not set)
        effective_level = validation.level if validation.level is not None else self._default_validation

        # Extract all variables from the template
        template_vars = self._extract_variables(template)

        # Determine which parameters were used and which are missing
        params_used = {k: v for k, v in params.items() if k in template_vars}
        params_missing = []

        # Check for missing required parameters
        for var in validation.required_params:
            if var not in params or params[var] is None:
                params_missing.append(var)

        # Handle validation
        validation_warnings = []
        if params_missing:
            missing_str = ", ".join(params_missing)
            if effective_level == ValidationLevel.ERROR:
                raise ValueError(
                    f"Missing required parameters: {missing_str}"
                )
            elif effective_level == ValidationLevel.WARN:
                warning_msg = f"Missing required parameters: {missing_str}"
                validation_warnings.append(warning_msg)
                logger.warning(warning_msg)
            # IGNORE level: do nothing

        # Render the template
        content = render_conditional_template(template, params)

        # Build result
        return RenderResult(
            content=content,
            params_used=params_used,
            params_missing=params_missing,
            validation_warnings=validation_warnings,
            metadata={
                "validation_level": effective_level.value,
                "template_vars": list(template_vars),
                **(template_metadata or {})
            }
        )

    def render_prompt_template(
        self,
        prompt_template: PromptTemplate,
        params: Dict[str, Any],
        validation_override: Optional[ValidationLevel] = None
    ) -> RenderResult:
        """Render a PromptTemplate structure with validation.

        Args:
            prompt_template: PromptTemplate dictionary with template, defaults, validation
            params: Parameters to substitute (merged with template defaults)
            validation_override: Optional runtime validation level override

        Returns:
            RenderResult with rendered content and validation information
        """
        # Merge defaults with provided params (params take priority)
        merged_params = {
            **prompt_template.get("defaults", {}),
            **params
        }

        # Get validation config from template
        template_validation = prompt_template.get("validation")

        # Apply validation override if provided
        if validation_override is not None:
            if template_validation:
                # Create new config with overridden level
                validation = ValidationConfig(
                    level=validation_override,
                    required_params=list(template_validation.required_params),
                    optional_params=list(template_validation.optional_params)
                )
            else:
                # Create new config with just the override level
                validation = ValidationConfig(level=validation_override)
        else:
            validation = template_validation

        # Render with merged params and validation
        return self.render(
            template=prompt_template["template"],
            params=merged_params,
            validation=validation,
            template_metadata=prompt_template.get("metadata")
        )

    def batch_render(
        self,
        templates: List[str],
        params: Dict[str, Any],
        validation: Optional[ValidationConfig] = None
    ) -> List[RenderResult]:
        """Render multiple templates with the same parameters.

        Args:
            templates: List of template strings
            params: Parameters to substitute in all templates
            validation: Optional validation configuration for all templates

        Returns:
            List of RenderResult objects, one per template
        """
        return [
            self.render(template, params, validation)
            for template in templates
        ]

    @staticmethod
    def _extract_variables(template: str) -> Set[str]:
        """Extract all variable names from a template.

        Args:
            template: Template string with {{variable}} syntax

        Returns:
            Set of variable names found in the template
        """
        # Pattern to match {{variable}} with optional whitespace
        var_pattern = r'\{\{(\s*)(\w+)(\s*)\}\}'
        matches = re.finditer(var_pattern, template)

        # Extract just the variable names (group 2 from regex)
        variables = {match.group(2) for match in matches}

        return variables

    @staticmethod
    def validate_template_syntax(template: str) -> List[str]:
        """Validate template syntax and return any issues.

        Args:
            template: Template string to validate

        Returns:
            List of error messages (empty if valid)
        """
        errors = []

        # Check for unmatched braces
        # Match single braces that are not part of {{ or }}
        brace_pattern = r'(?<!\{)\{(?!\{)|(?<!\})\}(?!\})'
        unmatched = re.findall(brace_pattern, template)
        if unmatched:
            errors.append(
                f"Found {len(unmatched)} unmatched brace(s). "
                "Use {{ }} for variables, not { }."
            )

        # Check for unmatched conditional sections
        open_count = template.count('((')
        close_count = template.count('))')
        if open_count != close_count:
            errors.append(
                f"Unmatched conditional sections: {open_count} '((' vs "
                f"{close_count} '))'."
            )

        # Check for malformed variable patterns
        # Variables should be \w+ (word characters only)
        malformed_pattern = r'\{\{[^}]*[^}\w\s][^}]*\}\}'
        malformed = re.findall(malformed_pattern, template)
        if malformed:
            errors.append(
                f"Found {len(malformed)} malformed variable(s). "
                "Variables should contain only letters, numbers, and underscores."
            )

        return errors


# Convenience functions for one-off rendering

def render_template(
    template: str,
    params: Dict[str, Any],
    validation_level: ValidationLevel = ValidationLevel.WARN
) -> str:
    """Convenience function to render a template with parameters.

    Args:
        template: Template string with {{variables}} and ((conditionals))
        params: Parameters to substitute
        validation_level: Validation level to use (default: WARN)

    Returns:
        Rendered template string

    Example:
        >>> result = render_template(
        ...     "Hello {{name}}((, you are {{age}} years old))",
        ...     {"name": "Alice", "age": 30}
        ... )
        >>> print(result)
        "Hello Alice, you are 30 years old"
    """
    renderer = TemplateRenderer(default_validation=validation_level)
    result = renderer.render(template, params)
    return result.content


def render_template_strict(
    template: str,
    params: Dict[str, Any],
    required_params: List[str]
) -> str:
    """Render a template with strict validation (ERROR level).

    Args:
        template: Template string
        params: Parameters to substitute
        required_params: List of required parameter names

    Returns:
        Rendered template string

    Raises:
        ValueError: If any required parameters are missing
    """
    renderer = TemplateRenderer()
    validation = ValidationConfig(
        level=ValidationLevel.ERROR,
        required_params=required_params
    )
    result = renderer.render(template, params, validation)
    return result.content
