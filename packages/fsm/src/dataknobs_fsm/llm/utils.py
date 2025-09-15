"""Utility functions for LLM operations.

This module provides utility functions for working with LLMs.
"""

import re
import json
from typing import Any, Dict, List, Union
from dataclasses import dataclass, field

from .base import LLMMessage, LLMResponse


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


@dataclass
class PromptTemplate:
    """Template for generating prompts."""
    template: str
    variables: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Extract variables from template."""
        if not self.variables:
            # Extract {variable} patterns
            self.variables = re.findall(r'\{(\w+)\}', self.template)
            
    def format(self, **kwargs) -> str:
        """Format template with variables.
        
        Args:
            **kwargs: Variable values
            
        Returns:
            Formatted prompt
        """
        # Check all required variables are provided
        missing = set(self.variables) - set(kwargs.keys())
        if missing:
            raise ValueError(f"Missing variables: {missing}")
            
        return self.template.format(**kwargs)
        
    def partial(self, **kwargs) -> 'PromptTemplate':
        """Create partial template with some variables filled.
        
        Args:
            **kwargs: Variable values to fill
            
        Returns:
            New template with partial values
        """
        new_template = self.template
        new_variables = self.variables.copy()
        
        for key, value in kwargs.items():
            if key in new_variables:
                new_template = new_template.replace(f'{{{key}}}', str(value))
                new_variables.remove(key)
                
        return PromptTemplate(new_template, new_variables)


class MessageBuilder:
    """Builder for constructing message sequences."""
    
    def __init__(self):
        self.messages = []
        
    def system(self, content: str) -> 'MessageBuilder':
        """Add system message.
        
        Args:
            content: Message content
            
        Returns:
            Self for chaining
        """
        self.messages.append(LLMMessage(role='system', content=content))
        return self
        
    def user(self, content: str) -> 'MessageBuilder':
        """Add user message.
        
        Args:
            content: Message content
            
        Returns:
            Self for chaining
        """
        self.messages.append(LLMMessage(role='user', content=content))
        return self
        
    def assistant(self, content: str) -> 'MessageBuilder':
        """Add assistant message.
        
        Args:
            content: Message content
            
        Returns:
            Self for chaining
        """
        self.messages.append(LLMMessage(role='assistant', content=content))
        return self
        
    def function(
        self,
        name: str,
        content: str,
        function_call: Dict[str, Any] | None = None
    ) -> 'MessageBuilder':
        """Add function message.
        
        Args:
            name: Function name
            content: Function result
            function_call: Function call details
            
        Returns:
            Self for chaining
        """
        self.messages.append(LLMMessage(
            role='function',
            name=name,
            content=content,
            function_call=function_call
        ))
        return self
        
    def from_template(
        self,
        role: str,
        template: PromptTemplate,
        **kwargs
    ) -> 'MessageBuilder':
        """Add message from template.
        
        Args:
            role: Message role
            template: Prompt template
            **kwargs: Template variables
            
        Returns:
            Self for chaining
        """
        content = template.format(**kwargs)
        self.messages.append(LLMMessage(role=role, content=content))
        return self
        
    def build(self) -> List[LLMMessage]:
        """Build message list.
        
        Returns:
            List of messages
        """
        return self.messages.copy()
        
    def clear(self) -> 'MessageBuilder':
        """Clear all messages.
        
        Returns:
            Self for chaining
        """
        self.messages.clear()
        return self


class ResponseParser:
    """Parser for LLM responses."""
    
    @staticmethod
    def extract_json(response: Union[str, LLMResponse]) -> Dict[str, Any] | None:
        """Extract JSON from response.
        
        Args:
            response: LLM response
            
        Returns:
            Extracted JSON or None
        """
        text = response.content if isinstance(response, LLMResponse) else response
        
        # Try to find JSON in the text
        json_patterns = [
            r'\{[^}]*\}',  # Simple object
            r'\[[^\]]*\]',  # Array
            r'```json\s*(.*?)\s*```',  # Markdown code block
            r'```\s*(.*?)\s*```',  # Generic code block
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue
                    
        # Try parsing the entire text as JSON
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None
            
    @staticmethod
    def extract_code(
        response: Union[str, LLMResponse],
        language: str | None = None
    ) -> List[str]:
        """Extract code blocks from response.
        
        Args:
            response: LLM response
            language: Optional language filter
            
        Returns:
            List of code blocks
        """
        text = response.content if isinstance(response, LLMResponse) else response
        
        if language:
            # Language-specific code blocks
            pattern = rf'```{language}\s*(.*?)\s*```'
        else:
            # All code blocks
            pattern = r'```(?:\w+)?\s*(.*?)\s*```'
            
        matches = re.findall(pattern, text, re.DOTALL)
        return [m.strip() for m in matches]
        
    @staticmethod
    def extract_list(
        response: Union[str, LLMResponse],
        numbered: bool = False
    ) -> List[str]:
        """Extract list items from response.
        
        Args:
            response: LLM response
            numbered: Whether to look for numbered lists
            
        Returns:
            List of items
        """
        text = response.content if isinstance(response, LLMResponse) else response
        
        if numbered:
            # Numbered list (1. item, 2. item, etc.)
            pattern = r'^\d+\.\s+(.+)$'
        else:
            # Bullet points (-, *, •)
            pattern = r'^[-*•]\s+(.+)$'
            
        matches = re.findall(pattern, text, re.MULTILINE)
        return [m.strip() for m in matches]
        
    @staticmethod
    def extract_sections(
        response: Union[str, LLMResponse]
    ) -> Dict[str, str]:
        """Extract sections from response.
        
        Args:
            response: LLM response
            
        Returns:
            Dictionary of section name to content
        """
        text = response.content if isinstance(response, LLMResponse) else response
        
        # Split by headers (# Header, ## Header, etc.)
        sections = {}
        current_section = 'main'
        current_content = []
        
        for line in text.split('\n'):
            header_match = re.match(r'^#+\s+(.+)$', line)
            if header_match:
                # Save previous section
                if current_content:
                    sections[current_section] = '\n'.join(current_content).strip()
                # Start new section
                current_section = header_match.group(1).strip()
                current_content = []
            else:
                current_content.append(line)
                
        # Save last section
        if current_content:
            sections[current_section] = '\n'.join(current_content).strip()
            
        return sections


class TokenCounter:
    """Estimate token counts for different models."""
    
    # Approximate tokens per character for different models
    TOKENS_PER_CHAR = {
        'gpt-4': 0.25,
        'gpt-3.5': 0.25,
        'claude': 0.25,
        'llama': 0.3,
        'default': 0.25
    }
    
    @classmethod
    def estimate_tokens(
        cls,
        text: str,
        model: str = 'default'
    ) -> int:
        """Estimate token count for text.
        
        Args:
            text: Input text
            model: Model name
            
        Returns:
            Estimated token count
        """
        # Find matching model pattern
        ratio = cls.TOKENS_PER_CHAR['default']
        for pattern, r in cls.TOKENS_PER_CHAR.items():
            if pattern in model.lower():
                ratio = r
                break
                
        # Estimate based on character count
        return int(len(text) * ratio)
        
    @classmethod
    def estimate_messages_tokens(
        cls,
        messages: List[LLMMessage],
        model: str = 'default'
    ) -> int:
        """Estimate token count for messages.
        
        Args:
            messages: List of messages
            model: Model name
            
        Returns:
            Estimated token count
        """
        total = 0
        for msg in messages:
            # Add role tokens (approximately 4 tokens)
            total += 4
            # Add content tokens
            total += cls.estimate_tokens(msg.content, model)
            # Add name tokens if present
            if msg.name:
                total += cls.estimate_tokens(msg.name, model)
                
        return total
        
    @classmethod
    def fits_in_context(
        cls,
        text: str,
        model: str,
        max_tokens: int
    ) -> bool:
        """Check if text fits in context window.
        
        Args:
            text: Input text
            model: Model name
            max_tokens: Maximum token limit
            
        Returns:
            True if fits
        """
        estimated = cls.estimate_tokens(text, model)
        return estimated <= max_tokens


class CostCalculator:
    """Calculate costs for LLM usage."""
    
    # Cost per 1K tokens (in USD)
    PRICING = {
        'gpt-4': {'input': 0.03, 'output': 0.06},
        'gpt-4-32k': {'input': 0.06, 'output': 0.12},
        'gpt-3.5-turbo': {'input': 0.0015, 'output': 0.002},
        'claude-3-opus': {'input': 0.015, 'output': 0.075},
        'claude-3-sonnet': {'input': 0.003, 'output': 0.015},
        'claude-3-haiku': {'input': 0.00025, 'output': 0.00125},
    }
    
    @classmethod
    def calculate_cost(
        cls,
        response: LLMResponse,
        model: str | None = None
    ) -> float | None:
        """Calculate cost for LLM response.
        
        Args:
            response: LLM response with usage info
            model: Model name (if not in response)
            
        Returns:
            Cost in USD or None if cannot calculate
        """
        if not response.usage:
            return None
            
        model = model or response.model
        
        # Find matching pricing
        pricing = None
        for pattern, prices in cls.PRICING.items():
            if pattern in model.lower():
                pricing = prices
                break
                
        if not pricing:
            return None
            
        # Calculate cost
        input_cost = (response.usage.get('prompt_tokens', 0) / 1000) * pricing['input']
        output_cost = (response.usage.get('completion_tokens', 0) / 1000) * pricing['output']
        
        return input_cost + output_cost
        
    @classmethod
    def estimate_cost(
        cls,
        text: str,
        model: str,
        expected_output_tokens: int = 100
    ) -> float | None:
        """Estimate cost for text completion.
        
        Args:
            text: Input text
            model: Model name
            expected_output_tokens: Expected output length
            
        Returns:
            Estimated cost in USD
        """
        # Find matching pricing
        pricing = None
        for pattern, prices in cls.PRICING.items():
            if pattern in model.lower():
                pricing = prices
                break
                
        if not pricing:
            return None
            
        # Estimate tokens
        input_tokens = TokenCounter.estimate_tokens(text, model)
        
        # Calculate cost
        input_cost = (input_tokens / 1000) * pricing['input']
        output_cost = (expected_output_tokens / 1000) * pricing['output']
        
        return input_cost + output_cost


def chain_prompts(
    *templates: PromptTemplate
) -> PromptTemplate:
    """Chain multiple prompt templates.
    
    Args:
        *templates: Templates to chain
        
    Returns:
        Combined template
    """
    combined_template = '\n\n'.join(t.template for t in templates)
    combined_variables = []
    seen = set()
    
    for t in templates:
        for var in t.variables:
            if var not in seen:
                combined_variables.append(var)
                seen.add(var)
                
    return PromptTemplate(combined_template, combined_variables)


def create_few_shot_prompt(
    instruction: str,
    examples: List[Dict[str, str]],
    query_key: str = 'input',
    response_key: str = 'output'
) -> PromptTemplate:
    """Create few-shot learning prompt.
    
    Args:
        instruction: Task instruction
        examples: List of example input/output pairs
        query_key: Key for input in examples
        response_key: Key for output in examples
        
    Returns:
        Few-shot prompt template
    """
    template_parts = [instruction, '']
    
    # Add examples
    for i, example in enumerate(examples, 1):
        template_parts.append(f"Example {i}:")
        template_parts.append(f"Input: {example[query_key]}")
        template_parts.append(f"Output: {example[response_key]}")
        template_parts.append('')
        
    # Add query placeholder
    template_parts.append("Now, process this input:")
    template_parts.append("Input: {query}")
    template_parts.append("Output:")
    
    return PromptTemplate('\n'.join(template_parts), ['query'])
