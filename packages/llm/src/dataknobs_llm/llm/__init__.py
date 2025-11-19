"""LLM abstraction layer for FSM patterns.

This module provides a unified abstraction layer for working with various LLM providers
including OpenAI, Anthropic, Ollama, and others.
"""

from .base import (
    LLMProvider,
    AsyncLLMProvider,
    SyncLLMProvider,
    LLMConfig,
    LLMMessage,
    LLMResponse,
    LLMStreamResponse,
    CompletionMode,
    ModelCapability,
    ToolCall,
    normalize_llm_config,
)
from .providers import (
    OpenAIProvider,
    AnthropicProvider,
    OllamaProvider,
    HuggingFaceProvider,
    EchoProvider,
    LLMProviderFactory,
    create_llm_provider,
)
from .utils import (
    TemplateStrategy,
    MessageTemplate,
    MessageBuilder,
    ResponseParser,
    TokenCounter,
    CostCalculator,
    render_conditional_template,
    chain_prompts,
    create_few_shot_prompt,
)

__all__ = [
    # Base classes
    'LLMProvider',
    'AsyncLLMProvider',
    'SyncLLMProvider',
    'LLMConfig',
    'LLMMessage',
    'LLMResponse',
    'LLMStreamResponse',
    'CompletionMode',
    'ModelCapability',
    'ToolCall',
    'normalize_llm_config',
    # Providers
    'OpenAIProvider',
    'AnthropicProvider',
    'OllamaProvider',
    'HuggingFaceProvider',
    'EchoProvider',
    'LLMProviderFactory',
    'create_llm_provider',
    # Utils
    'TemplateStrategy',
    'MessageTemplate',
    'MessageBuilder',
    'ResponseParser',
    'TokenCounter',
    'CostCalculator',
    'render_conditional_template',
    'chain_prompts',
    'create_few_shot_prompt',
]
