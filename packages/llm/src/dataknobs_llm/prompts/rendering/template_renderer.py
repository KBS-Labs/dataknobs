"""Template renderer with validation support and Jinja2 integration.

This module provides template rendering with:
- Custom (( )) conditional syntax (backward compatible)
- Jinja2 integration for filters, advanced conditionals, includes
- Two rendering modes: "mixed" (both syntaxes) and "jinja2" (pure Jinja2)
- Validation capabilities for missing parameters
"""

import re
import logging
from typing import Any, Callable, Dict, List, Set, Tuple
from dataclasses import dataclass

from jinja2 import Environment, TemplateSyntaxError as Jinja2SyntaxError, Undefined

from dataknobs_llm.template_utils import render_conditional_template
from ..base.types import (
    ValidationLevel,
    ValidationConfig,
    PromptTemplateDict,
    RenderResult,
    TemplateMode
)

logger = logging.getLogger(__name__)


class PreserveUndefined(Undefined):
    """Jinja2 Undefined handler that preserves placeholders for undefined variables.

    This maintains backward compatibility with the old template behavior where
    undefined variables are left as {{variable}} instead of being rendered as
    empty strings.
    """
    def __str__(self) -> str:
        """Return the original placeholder for undefined variables."""
        return f"{{{{{self._undefined_name}}}}}"

    def __repr__(self) -> str:
        """Return the original placeholder for undefined variables."""
        return f"{{{{{self._undefined_name}}}}}"


@dataclass
class TemplateSyntaxError:
    """Represents a template syntax error with location information."""
    message: str
    line: int
    column: int
    snippet: str
    error_type: str  # 'unmatched_brace', 'unmatched_conditional', 'malformed_variable'

    def __str__(self) -> str:
        """Format error message with location."""
        return (
            f"{self.error_type} at line {self.line}, column {self.column}: {self.message}\n"
            f"  {self.snippet}"
        )


class TemplateRenderer:
    """Template renderer with configurable validation and Jinja2 support.

    This class provides:
    - Two rendering modes: "mixed" (default) and "jinja2"
    - Custom (( )) conditional syntax (backward compatible)
    - Jinja2 filters, conditionals, includes, loops, macros
    - Validation of required parameters (ERROR/WARN/IGNORE)
    - Tracking of used and missing parameters
    - Detailed render results with metadata
    """

    def __init__(
        self,
        default_validation: ValidationLevel = ValidationLevel.WARN,
        default_mode: TemplateMode = TemplateMode.MIXED
    ):
        """Initialize the template renderer with Jinja2.

        Args:
            default_validation: Default validation level for templates
                               without explicit validation configuration
            default_mode: Default template mode (mixed or jinja2)
        """
        self._default_validation = default_validation
        self._default_mode = default_mode

        # Initialize Jinja2 environment
        self._jinja_env = Environment(
            # Keep same delimiters as our custom syntax
            variable_start_string='{{',
            variable_end_string='}}',
            block_start_string='{%',
            block_end_string='%}',
            comment_start_string='{#',
            comment_end_string='#}',
            # Prompt generation, not HTML - no autoescaping
            autoescape=False,
            # Better whitespace handling
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True,
            # Preserve undefined variables (backward compatibility)
            undefined=PreserveUndefined,
        )

        # Register custom filters (domain-specific)
        self._register_custom_filters()

    def render(
        self,
        template: str,
        params: Dict[str, Any],
        validation: ValidationConfig | None = None,
        template_metadata: Dict[str, Any] | None = None,
        mode: TemplateMode | None = None
    ) -> RenderResult:
        """Render a template with parameters and validation.

        Args:
            template: Template string with {{variables}} and ((conditionals))
            params: Parameters to substitute in the template
            validation: Optional validation configuration (overrides template default)
            template_metadata: Optional metadata about the template
            mode: Template mode (mixed or jinja2), defaults to renderer default

        Returns:
            RenderResult with rendered content and validation information

        Raises:
            ValueError: If validation fails or syntax errors occur
        """
        # Determine mode
        effective_mode = mode if mode is not None else self._default_mode

        # Use provided validation or create default
        if validation is None:
            validation = ValidationConfig(level=self._default_validation)

        # Determine effective validation level (inherit from renderer if not set)
        effective_level = validation.level if validation.level is not None else self._default_validation

        try:
            # Step 1: Pre-process (( )) if in mixed mode
            if effective_mode == TemplateMode.MIXED:
                # Validate: no Jinja2 syntax inside (( ))
                self._validate_no_jinja_in_conditionals(template)

                # Pre-process conditionals
                intermediate = render_conditional_template(template, params)
            else:
                # Pure Jinja2 mode - use template as-is
                intermediate = template

            # Step 2: Render with Jinja2
            jinja_template = self._jinja_env.from_string(intermediate)
            content = jinja_template.render(**params)

        except Jinja2SyntaxError as e:
            raise ValueError(
                f"Template syntax error at line {e.lineno}: {e.message}\n"
                f"Template: {e.source or 'N/A'}"
            ) from e
        except Exception as e:
            raise ValueError(f"Template rendering error: {e}") from e

        # Step 3: Validation
        template_vars = self._extract_variables(template)
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

        # Step 4: Build result
        return RenderResult(
            content=content,
            params_used=params_used,
            params_missing=params_missing,
            validation_warnings=validation_warnings,
            metadata={
                "validation_level": effective_level.value,
                "template_vars": list(template_vars),
                "template_mode": effective_mode.value,
                **(template_metadata or {})
            }
        )

    def render_prompt_template(
        self,
        prompt_template: PromptTemplateDict,
        params: Dict[str, Any],
        validation_override: ValidationLevel | None = None,
        mode_override: TemplateMode | None = None
    ) -> RenderResult:
        """Render a PromptTemplateDict structure with validation.

        Args:
            prompt_template: PromptTemplateDict dictionary with template, defaults, validation
            params: Parameters to substitute (merged with template defaults)
            validation_override: Optional runtime validation level override
            mode_override: Optional template mode override

        Returns:
            RenderResult with rendered content and validation information
        """
        # Merge defaults with provided params (params take priority)
        merged_params = {
            **prompt_template.get("defaults", {}),
            **params
        }

        # Get template mode from template or use override
        template_mode_str = prompt_template.get("template_mode", "mixed")
        template_mode = TemplateMode.from_string(template_mode_str)
        effective_mode = mode_override if mode_override is not None else template_mode

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

        # Render with merged params, validation, and effective mode
        return self.render(
            template=prompt_template["template"],
            params=merged_params,
            validation=validation,
            template_metadata=prompt_template.get("metadata"),
            mode=effective_mode
        )

    def batch_render(
        self,
        templates: List[str],
        params: Dict[str, Any],
        validation: ValidationConfig | None = None
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
        """Extract all variable names from a template (including Jinja2 syntax).

        Args:
            template: Template string with {{variable}} or {{variable|filter}} syntax

        Returns:
            Set of variable names found in the template
        """
        # Pattern to match {{var}} or {{var|filter}} - extract just the variable name
        var_pattern = r'\{\{\s*(\w+)(?:\s*\|[^}]*)?\s*\}\}'
        matches = re.finditer(var_pattern, template)

        # Extract variable names (group 1 from regex)
        variables = {match.group(1) for match in matches}

        return variables

    def _validate_no_jinja_in_conditionals(self, template: str):
        """Validate that no Jinja2 syntax appears inside (( )) blocks.

        Args:
            template: Template string to validate

        Raises:
            ValueError: If Jinja2 syntax found inside (( ))
        """
        # Find all (( ... )) blocks
        pattern = r'\(\(((?:[^()]|\((?!\()|(?<!\))\))*)\)\)'

        for match in re.finditer(pattern, template):
            block_content = match.group(1)

            # Check for {% %} blocks
            if '{%' in block_content:
                raise ValueError(
                    f"Jinja2 block syntax ('{{% %}}') not allowed inside "
                    f"conditional blocks '(( ))'.\n"
                    f"Found in: ((... {block_content[:50]} ...))\n"
                    f"Hint: Move {{% %}} blocks outside (( )) or use "
                    f"template_mode='jinja2' for pure Jinja2 templates."
                )

            # Check for filters (| after {{)
            if re.search(r'\{\{\s*\w+\s*\|', block_content):
                raise ValueError(
                    f"Jinja2 filters (|filter) not allowed inside "
                    f"conditional blocks '(( ))'.\n"
                    f"Found in: ((... {block_content[:50]} ...))\n"
                    f"Hint: Apply filters outside (( )) blocks or use "
                    f"template_mode='jinja2' for pure Jinja2 templates."
                )

    def _register_custom_filters(self):
        """Register custom domain-specific filters."""

        # Example: Token counting filter
        def count_tokens(text: str, model: str = "gpt-4") -> int:
            """Count approximate tokens in text."""
            # Simple approximation: ~4 chars per token
            return len(text) // 4

        self._jinja_env.filters['count_tokens'] = count_tokens

        # Example: Prompt formatting
        def format_code(code: str, language: str = "python") -> str:
            """Format code in markdown code block."""
            return f"```{language}\n{code}\n```"

        self._jinja_env.filters['format_code'] = format_code

        # Users can add more via add_custom_filter()

    def add_custom_filter(
        self,
        name: str,
        filter_func: Callable[..., Any]
    ):
        """Register a custom filter with Jinja2.

        Args:
            name: Filter name (used in templates as |name)
            filter_func: Filter function (first arg is value to filter)

        Example:
            >>> renderer.add_custom_filter(
            ...     'double',
            ...     lambda x: x * 2
            ... )
            >>> result = renderer.render("{{count|double}}", {"count": 5})
            >>> result.content
            "10"
        """
        self._jinja_env.filters[name] = filter_func

    @staticmethod
    def _get_line_col(template: str, position: int) -> Tuple[int, int]:
        """Get line and column number for a position in the template.

        Args:
            template: Template string
            position: Character position in template

        Returns:
            Tuple of (line_number, column_number) (1-indexed)
        """
        lines = template[:position].split('\n')
        line = len(lines)
        column = len(lines[-1]) + 1
        return line, column

    @staticmethod
    def _get_snippet(template: str, position: int, context: int = 20) -> str:
        """Get a snippet of text around a position.

        Args:
            template: Template string
            position: Character position
            context: Number of characters to show before/after

        Returns:
            Snippet with error position marked
        """
        start = max(0, position - context)
        end = min(len(template), position + context)
        snippet = template[start:end]

        # Replace newlines for better display
        snippet = snippet.replace('\n', '\\n')

        # Mark the error position
        error_pos = min(position - start, len(snippet))
        if error_pos < len(snippet):
            snippet = snippet[:error_pos] + '⮜HERE⮞' + snippet[error_pos:]

        return snippet

    @staticmethod
    def validate_template_syntax_detailed(template: str) -> List[TemplateSyntaxError]:
        """Validate template syntax and return detailed errors with locations.

        Args:
            template: Template string to validate

        Returns:
            List of TemplateSyntaxError objects (empty if valid)
        """
        errors = []

        # Check for unmatched braces
        brace_pattern = r'(?<!\{)\{(?!\{)|(?<!\})\}(?!\})'
        for match in re.finditer(brace_pattern, template):
            position = match.start()
            line, col = TemplateRenderer._get_line_col(template, position)
            snippet = TemplateRenderer._get_snippet(template, position)

            errors.append(TemplateSyntaxError(
                message="Unmatched brace. Use {{ }} for variables, not { }.",
                line=line,
                column=col,
                snippet=snippet,
                error_type="unmatched_brace"
            ))

        # Check for unmatched conditional sections
        open_positions = [m.start() for m in re.finditer(r'\(\(', template)]
        close_positions = [m.start() for m in re.finditer(r'\)\)', template)]

        # Simple stack-based matching
        stack = []
        all_positions = sorted(
            [(pos, 'open') for pos in open_positions] +
            [(pos, 'close') for pos in close_positions]
        )

        for position, bracket_type in all_positions:
            if bracket_type == 'open':
                stack.append(position)
            else:  # close
                if not stack:
                    # Closing without opening
                    line, col = TemplateRenderer._get_line_col(template, position)
                    snippet = TemplateRenderer._get_snippet(template, position)
                    errors.append(TemplateSyntaxError(
                        message="Closing ')) without matching opening '(('.",
                        line=line,
                        column=col,
                        snippet=snippet,
                        error_type="unmatched_conditional"
                    ))
                else:
                    stack.pop()

        # Remaining unclosed openings
        for position in stack:
            line, col = TemplateRenderer._get_line_col(template, position)
            snippet = TemplateRenderer._get_snippet(template, position)
            errors.append(TemplateSyntaxError(
                message="Opening '((' without matching closing '))'.",
                line=line,
                column=col,
                snippet=snippet,
                error_type="unmatched_conditional"
            ))

        # Check for malformed variable patterns
        # Look for {{ }} that don't contain valid variable names
        var_pattern = r'\{\{[^}]*\}\}'
        for match in re.finditer(var_pattern, template):
            var_content = match.group(0)[2:-2].strip()  # Remove {{ }}

            # Valid variable: only word characters (letters, digits, underscores)
            if var_content and not re.match(r'^\w+$', var_content):
                position = match.start()
                line, col = TemplateRenderer._get_line_col(template, position)
                snippet = TemplateRenderer._get_snippet(template, position, context=30)

                errors.append(TemplateSyntaxError(
                    message=(
                        f"Malformed variable '{{{{' {var_content} '}}}}'. "
                        "Variables should contain only letters, numbers, and underscores."
                    ),
                    line=line,
                    column=col,
                    snippet=snippet,
                    error_type="malformed_variable"
                ))
            elif not var_content:
                # Empty variable {{}}
                position = match.start()
                line, col = TemplateRenderer._get_line_col(template, position)
                snippet = TemplateRenderer._get_snippet(template, position)

                errors.append(TemplateSyntaxError(
                    message="Empty variable {{}}. Variables must have a name.",
                    line=line,
                    column=col,
                    snippet=snippet,
                    error_type="malformed_variable"
                ))

        return errors

    @staticmethod
    def validate_template_syntax(template: str) -> List[str]:
        """Validate template syntax and return error messages.

        This is a convenience wrapper around validate_template_syntax_detailed()
        that returns simple string messages instead of detailed error objects.

        Args:
            template: Template string to validate

        Returns:
            List of error messages (empty if valid)
        """
        detailed_errors = TemplateRenderer.validate_template_syntax_detailed(template)
        return [str(error) for error in detailed_errors]


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
