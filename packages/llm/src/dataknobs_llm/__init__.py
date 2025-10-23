"""LLM utilities for DataKnobs."""

from dataknobs_llm.llm import (
    LLMProvider,
    LLMConfig,
    LLMMessage,
    LLMResponse,
    LLMStreamResponse,
    CompletionMode,
    ModelCapability,
    normalize_llm_config,
    OpenAIProvider,
    AnthropicProvider,
    OllamaProvider,
    HuggingFaceProvider,
    EchoProvider,
    LLMProviderFactory,
    create_llm_provider,
    TemplateStrategy,
    PromptTemplate,
    MessageBuilder,
    ResponseParser,
    TokenCounter,
    CostCalculator,
    render_conditional_template,
)

__version__ = "0.1.0"

__all__ = [
    # Base classes
    "LLMProvider",
    "LLMConfig",
    "LLMMessage",
    "LLMResponse",
    "LLMStreamResponse",
    "CompletionMode",
    "ModelCapability",
    "normalize_llm_config",
    # Providers
    "OpenAIProvider",
    "AnthropicProvider",
    "OllamaProvider",
    "HuggingFaceProvider",
    "EchoProvider",
    "LLMProviderFactory",
    "create_llm_provider",
    # Utils
    "TemplateStrategy",
    "PromptTemplate",
    "MessageBuilder",
    "ResponseParser",
    "TokenCounter",
    "CostCalculator",
    "render_conditional_template",
]
