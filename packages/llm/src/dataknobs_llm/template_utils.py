"""Template rendering utilities for dataknobs-llm.

This module provides shared template rendering functions used by both the LLM
and prompt library components. By keeping these utilities separate, we avoid
circular dependencies between the llm and prompts packages.
"""

import re
from enum import Enum
from typing import Any, Dict


class TemplateStrategy(Enum):
    """Template rendering strategies."""
    SIMPLE = "simple"  # Python str.format() with {variable} syntax
    CONDITIONAL = "conditional"  # Advanced with {{variable}} and ((conditional)) syntax


def render_conditional_template(template: str, params: Dict[str, Any]) -> str:
    """Render a template with variable substitution and conditional sections.

    Variable substitution:
    - {{variable}} syntax for placeholders
    - Variables in params dict are replaced with their values
    - Variables not in params are left unchanged ({{variable}} remains as-is)
    - Whitespace handling: {{ var }} -> " value " when substituted, " {{var}} " when not

    Conditional sections:
    - ((optional content)) syntax for conditional blocks
    - Section is removed if all {{variables}} inside are empty/None/missing
    - Section is rendered (without parentheses) if any variable has a value
    - Variables inside conditionals are replaced with empty strings if missing
    - Nested conditionals are processed recursively

    Example:
        template = "Hello {{name}}((, you have {{count}} messages))"
        params = {"name": "Alice", "count": 5}
        result = "Hello Alice, you have 5 messages"

        params = {"name": "Bob"}  # no count
        result = "Hello Bob"  # conditional section removed

    Args:
        template: The template string
        params: Dictionary of parameters to substitute

    Returns:
        The rendered template
    """
    def replace_variable(text: str, params: Dict[str, Any], in_conditional: bool = False) -> str:
        """Replace variables in text with proper whitespace handling."""
        # Pattern to match variables with optional whitespace
        var_pattern = r'\{\{(\s*)(\w+)(\s*)\}\}'

        def replace_var(match):
            """Replace a single variable with whitespace handling."""
            prefix_ws = match.group(1)
            var_name = match.group(2)
            suffix_ws = match.group(3)

            if var_name not in params:
                if in_conditional:
                    # In conditional sections, missing variables become empty
                    return ""
                else:
                    # Outside conditionals, preserve the pattern but move whitespace outside
                    if prefix_ws or suffix_ws:
                        return f"{prefix_ws}{{{{{var_name}}}}}{suffix_ws}"
                    else:
                        return match.group(0)

            value = params[var_name]
            if value is None:
                if in_conditional:
                    return ""
                else:
                    # Move whitespace outside for None values
                    if prefix_ws or suffix_ws:
                        return f"{prefix_ws}{{{{{var_name}}}}}{suffix_ws}"
                    else:
                        return ""
            else:
                # Preserve whitespace when substituting
                return f"{prefix_ws}{value!s}{suffix_ws}"

        return re.sub(var_pattern, replace_var, text)

    def find_all_variables(text: str) -> set:
        """Find all variables in text, including nested conditionals."""
        var_pattern = r'\{\{(\s*)(\w+)(\s*)\}\}'
        variables = set()
        for match in re.finditer(var_pattern, text):
            variables.add(match.group(2))
        return variables

    def process_conditionals(text: str, params: Dict[str, Any]) -> str:
        """Process conditional sections recursively."""
        result = text
        changed = True

        while changed:
            changed = False
            # Find the first (( ... )) section
            start_pos = 0
            while True:
                start = result.find('((', start_pos)
                if start == -1:
                    break

                # Find matching )) - must track ALL parens for correct nesting
                depth = 1
                paren_depth = 0  # Track single parentheses
                end = start + 2
                while end < len(result) and depth > 0:
                    if result[end:end+2] == '((':
                        depth += 1
                        end += 2
                    elif result[end:end+2] == '))':
                        # Only count as )) if we're not inside single parens
                        if paren_depth == 0:
                            depth -= 1
                            end += 2
                        else:
                            # This is ) followed by another )
                            paren_depth -= 1
                            end += 1
                    elif result[end] == '(':
                        paren_depth += 1
                        end += 1
                    elif result[end] == ')':
                        paren_depth -= 1
                        end += 1
                    else:
                        end += 1

                if depth == 0:
                    # Found a complete section
                    content = result[start+2:end-2]

                    # Find ALL variables in this section (including nested)
                    all_vars = find_all_variables(content)

                    if all_vars:
                        # Check if all variables are empty/missing
                        has_value = False
                        for var_name in all_vars:
                            if var_name in params:
                                value = params[var_name]
                                if value is not None:
                                    if isinstance(value, str):
                                        # For strings, check if non-empty after stripping
                                        if value.strip():
                                            has_value = True
                                            break
                                    else:
                                        # For non-strings, any truthy value counts
                                        if value:
                                            has_value = True
                                            break

                        if not has_value:
                            # Remove the entire section - all variables are empty/missing
                            result = result[:start] + result[end:]
                        else:
                            # At least one variable has a value, process nested conditionals
                            processed_content = process_conditionals(content, params)
                            # Then substitute variables in the processed content
                            rendered = replace_variable(processed_content, params, in_conditional=True)
                            result = result[:start] + rendered + result[end:]
                    else:
                        # No variables in this section, keep the content as-is
                        # But still process any nested conditionals
                        processed_content = process_conditionals(content, params)
                        result = result[:start] + processed_content + result[end:]

                    changed = True
                    break
                else:
                    # Unmatched parentheses, leave as-is and move on
                    start_pos = start + 1

        return result

    # First process all conditional sections
    result = process_conditionals(template, params)

    # Then handle remaining variables outside of conditional sections
    result = replace_variable(result, params, in_conditional=False)

    return result
