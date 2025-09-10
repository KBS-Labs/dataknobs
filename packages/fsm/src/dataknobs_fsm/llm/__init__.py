"""LLM abstraction layer for FSM patterns.

This module provides a unified abstraction layer for working with various LLM providers
including OpenAI, Anthropic, Ollama, and others.
"""

from .base import (
    LLMProvider,
    LLMConfig,
    LLMMessage,
    LLMResponse,
    LLMStreamResponse,
    CompletionMode,
    ModelCapability,
)
from .providers import (
    OpenAIProvider,
    AnthropicProvider,
    OllamaProvider,
    HuggingFaceProvider,
    create_llm_provider,
)
from .utils import (
    PromptTemplate,
    MessageBuilder,
    ResponseParser,
    TokenCounter,
    CostCalculator,
)

__all__ = [
    # Base classes
    'LLMProvider',
    'LLMConfig',
    'LLMMessage',
    'LLMResponse',
    'LLMStreamResponse',
    'CompletionMode',
    'ModelCapability',
    # Providers
    'OpenAIProvider',
    'AnthropicProvider',
    'OllamaProvider',
    'HuggingFaceProvider',
    'create_llm_provider',
    # Utils
    'PromptTemplate',
    'MessageBuilder',
    'ResponseParser',
    'TokenCounter',
    'CostCalculator',
]
