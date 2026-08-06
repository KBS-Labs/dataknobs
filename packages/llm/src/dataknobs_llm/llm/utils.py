"""Utility functions for LLM operations.

This module provides utility functions for working with LLMs.
Template rendering utilities have been moved to dataknobs_llm.template_utils
to avoid circular dependencies.
"""

import re
import json
from typing import Any, Dict, List, Union
from dataclasses import dataclass, field

from .base import LLMMessage, LLMResponse
from .model_profile import ModelPricing
from ..template_utils import TemplateStrategy, render_conditional_template


@dataclass
class MessageTemplate:
    """Template for generating message content with multiple rendering strategies.

    Supports two template strategies:
    1. SIMPLE (default): Uses Python str.format() with {variable} syntax.
       - All variables must be provided
       - Clean and straightforward
       - Example: "Hello {name}!"

    2. CONDITIONAL: Advanced conditional rendering with {{variable}} and ((conditional)) syntax.
       - Variables can be optional
       - Conditional sections with (( ... ))
       - Whitespace-aware substitution
       - Example: "Hello {{name}}((, you have {{count}} messages))"
    """
    template: str
    variables: List[str] = field(default_factory=list)
    strategy: TemplateStrategy = TemplateStrategy.SIMPLE

    def __post_init__(self):
        """Extract variables from template based on strategy."""
        if not self.variables:
            if self.strategy == TemplateStrategy.SIMPLE:
                # Extract {variable} patterns (single braces)
                self.variables = re.findall(r'\{(\w+)\}', self.template)
            elif self.strategy == TemplateStrategy.CONDITIONAL:
                # Extract {{variable}} patterns (double braces)
                # Extract just the variable names (group 2 from the regex)
                self.variables = [match.group(2) for match in re.finditer(r'\{\{(\s*)(\w+)(\s*)\}\}', self.template)]
                # Remove duplicates while preserving order
                seen = set()
                unique_vars = []
                for var in self.variables:
                    if var not in seen:
                        seen.add(var)
                        unique_vars.append(var)
                self.variables = unique_vars

    def format(self, **kwargs: Any) -> str:
        """Format template with variables using the selected strategy.

        Args:
            **kwargs: Variable values

        Returns:
            Formatted prompt

        Raises:
            ValueError: If using SIMPLE strategy and required variables are missing
        """
        if self.strategy == TemplateStrategy.SIMPLE:
            # Simple strategy: all variables must be provided
            missing = set(self.variables) - set(kwargs.keys())
            if missing:
                raise ValueError(f"Missing variables: {missing}")
            return self.template.format(**kwargs)

        elif self.strategy == TemplateStrategy.CONDITIONAL:
            # Conditional strategy: use render_conditional_template
            return render_conditional_template(self.template, kwargs)

        else:
            raise ValueError(f"Unknown template strategy: {self.strategy}")

    def partial(self, **kwargs: Any) -> 'MessageTemplate':
        """Create partial template with some variables filled.

        Args:
            **kwargs: Variable values to fill

        Returns:
            New template with partial values
        """
        if self.strategy == TemplateStrategy.SIMPLE:
            # Simple strategy: replace {variable} patterns
            new_template = self.template
            new_variables = self.variables.copy()

            for key, value in kwargs.items():
                if key in new_variables:
                    new_template = new_template.replace(f'{{{key}}}', str(value))
                    new_variables.remove(key)

            return MessageTemplate(new_template, new_variables, self.strategy)

        elif self.strategy == TemplateStrategy.CONDITIONAL:
            # For conditional templates, render with provided variables
            # and keep the template structure for remaining variables
            new_template = self.template
            new_variables = self.variables.copy()

            # Replace only the provided variables with single-brace format
            # so they become literals in the new template
            for key, value in kwargs.items():
                if key in new_variables:
                    # Replace {{var}} with the value, but keep it as a literal
                    # We do this by using a placeholder that won't match the patterns
                    pattern = r'\{\{\s*' + key + r'\s*\}\}'
                    new_template = re.sub(
                        pattern,
                        str(value),
                        new_template
                    )
                    new_variables.remove(key)

            return MessageTemplate(new_template, new_variables, self.strategy)

        else:
            raise ValueError(f"Unknown template strategy: {self.strategy}")

    @classmethod
    def from_conditional(cls, template: str, variables: List[str] | None = None) -> 'MessageTemplate':
        """Create a MessageTemplate using the CONDITIONAL strategy.

        Convenience method for creating templates with advanced conditional rendering.

        Args:
            template: Template string with {{variable}} and ((conditional)) syntax
            variables: Optional explicit list of variables

        Returns:
            MessageTemplate configured with CONDITIONAL strategy

        Example:
            ```python
            template = MessageTemplate.from_conditional(
                "Hello {{name}}((, you have {{count}} messages))"
            )
            template.format(name="Alice", count=5)
            # "Hello Alice, you have 5 messages"
            template.format(name="Bob")
            # "Hello Bob"
            ```
        """
        return cls(template=template, variables=variables or [], strategy=TemplateStrategy.CONDITIONAL)


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
        template: MessageTemplate,
        **kwargs: Any
    ) -> 'MessageBuilder':
        """Add message from template.

        Args:
            role: Message role
            template: Message template
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
        text: str | None,
        model: str = 'default'
    ) -> int:
        """Estimate token count for text.

        Args:
            text: Input text. ``None`` / empty → 0 (an assistant message that
                carried only tool calls has no text content).
            model: Model name

        Returns:
            Estimated token count
        """
        if not text:
            return 0

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
            # Add tool-call payload tokens.  An assistant ``tool_use`` turn
            # carries its arguments here while ``content`` is ``None``; in a
            # long tool loop these call payloads (and the paired observations)
            # dominate the history, so counting only ``content`` systematically
            # under-reports the very turns a history budget exists to bound.
            if msg.tool_calls:
                serialized = json.dumps(
                    [tc.to_dict() for tc in msg.tool_calls],
                    default=str,
                    sort_keys=True,
                )
                total += cls.estimate_tokens(serialized, model)

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
    """Calculate costs for LLM usage from a :class:`ModelPricing`.

    The provider-agnostic **arithmetic** home for cost computation, paired with a
    provider's **facts** accessors (:meth:`~dataknobs_llm.llm.base.LLMProvider.get_pricing`
    / :meth:`~dataknobs_llm.llm.base.LLMProvider.estimate_cost`). Pricing is unified
    on :class:`~dataknobs_llm.llm.model_profile.ModelPricing` (per-million-token):
    pass an explicit ``pricing=`` — typically resolved through a provider's model
    profile — for the accurate, config-overridable value, or rely on the built-in
    :data:`PRICING` fallback for a provider-less, offline estimate.
    """

    #: Provider-less fallback pricing, keyed by model-id substring, as
    #: :class:`~dataknobs_llm.llm.model_profile.ModelPricing` (USD per 1M tokens).
    #: Migrated from the historical per-1K literal table onto the unified pricing
    #: shape; superseded per-provider by profile-sourced pricing (pass ``pricing=``
    #: / use ``provider.estimate_cost``). Kept small and back-compatible so the
    #: provider-less :meth:`calculate_cost` / :meth:`estimate_cost` calls still work.
    PRICING: dict[str, ModelPricing] = {
        'gpt-4': ModelPricing(input_per_mtok=30.0, output_per_mtok=60.0),
        'gpt-4-32k': ModelPricing(input_per_mtok=60.0, output_per_mtok=120.0),
        'gpt-3.5-turbo': ModelPricing(input_per_mtok=1.5, output_per_mtok=2.0),
        'claude-3-opus': ModelPricing(input_per_mtok=15.0, output_per_mtok=75.0),
        'claude-3-sonnet': ModelPricing(input_per_mtok=3.0, output_per_mtok=15.0),
        'claude-3-haiku': ModelPricing(input_per_mtok=0.25, output_per_mtok=1.25),
    }

    @classmethod
    def _fallback_pricing(cls, model: str | None) -> ModelPricing | None:
        """Resolve *model* to a :data:`PRICING` entry by substring, or ``None``."""
        if not model:
            return None
        model_lower = model.lower()
        for pattern, pricing in cls.PRICING.items():
            if pattern in model_lower:
                return pricing
        return None

    @staticmethod
    def cost_from_tokens(
        pricing: ModelPricing, input_tokens: int, output_tokens: int
    ) -> float:
        """Compute USD cost from a :class:`ModelPricing` (per-Mtok) + token counts.

        The public entry point for callers that hold **token counts** rather
        than an :class:`~dataknobs_llm.llm.base.LLMResponse` — usage totals
        accumulated across a multi-call turn, a stored usage record, an
        estimate. :meth:`calculate_cost` is the response-shaped equivalent and
        delegates here.

        A ``None`` rate contributes nothing (an embedding model prices input only).

        Args:
            pricing: Per-million-token rates, typically resolved through
                :meth:`~dataknobs_llm.llm.base.LLMProvider.get_pricing`.
            input_tokens: Prompt token count.
            output_tokens: Completion token count.

        Returns:
            Cost in USD.
        """
        cost = 0.0
        if pricing.input_per_mtok is not None:
            cost += (input_tokens / 1_000_000) * pricing.input_per_mtok
        if pricing.output_per_mtok is not None:
            cost += (output_tokens / 1_000_000) * pricing.output_per_mtok
        return cost

    @classmethod
    def calculate_cost(
        cls,
        response: LLMResponse,
        model: str | None = None,
        *,
        pricing: ModelPricing | None = None,
    ) -> float | None:
        """Calculate the USD cost of an LLM response.

        Args:
            response: LLM response with token ``usage``.
            model: Model name (defaults to ``response.model``); used only to look
                up the provider-less :data:`PRICING` fallback when ``pricing`` is
                not supplied.
            pricing: An explicit :class:`ModelPricing` — typically resolved
                through a provider's model profile
                (:meth:`~dataknobs_llm.llm.base.LLMProvider.get_pricing`). When
                given it is authoritative; otherwise the fallback table is used.

        Returns:
            Cost in USD, or ``None`` when the response carries no usage or no
            pricing can be resolved.
        """
        if not response.usage:
            return None
        if pricing is None:
            pricing = cls._fallback_pricing(model or response.model)
        if pricing is None:
            return None
        return cls.cost_from_tokens(
            pricing,
            response.usage.get('prompt_tokens', 0),
            response.usage.get('completion_tokens', 0),
        )

    @classmethod
    def estimate_cost(
        cls,
        text: str,
        model: str,
        expected_output_tokens: int = 100
    ) -> float | None:
        """Estimate cost for a text completion from the fallback pricing table.

        Args:
            text: Input text
            model: Model name
            expected_output_tokens: Expected output length

        Returns:
            Estimated cost in USD, or ``None`` when the model has no fallback
            pricing.
        """
        pricing = cls._fallback_pricing(model)
        if pricing is None:
            return None
        input_tokens = TokenCounter.estimate_tokens(text, model)
        return cls.cost_from_tokens(pricing, input_tokens, expected_output_tokens)


def chain_prompts(
    *templates: MessageTemplate
) -> MessageTemplate:
    """Chain multiple message templates.

    All templates must use the same strategy. The combined template
    will use the strategy of the first template.

    Args:
        *templates: Templates to chain

    Returns:
        Combined template

    Raises:
        ValueError: If templates use different strategies
    """
    if not templates:
        return MessageTemplate("", [])

    # Check that all templates use the same strategy
    first_strategy = templates[0].strategy
    if not all(t.strategy == first_strategy for t in templates):
        raise ValueError(
            "Cannot chain templates with different strategies. "
            "All templates must use the same TemplateStrategy."
        )

    combined_template = '\n\n'.join(t.template for t in templates)
    combined_variables = []
    seen = set()

    for t in templates:
        for var in t.variables:
            if var not in seen:
                combined_variables.append(var)
                seen.add(var)

    return MessageTemplate(combined_template, combined_variables, first_strategy)


def create_few_shot_prompt(
    instruction: str,
    examples: List[Dict[str, str]],
    query_key: str = 'input',
    response_key: str = 'output'
) -> MessageTemplate:
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

    return MessageTemplate('\n'.join(template_parts), ['query'])
