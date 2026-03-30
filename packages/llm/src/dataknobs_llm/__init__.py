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
    CachingEmbedProvider,
    EmbeddingCache,
    MemoryEmbeddingCache,
    LLMProviderFactory,
    create_llm_provider,
    create_caching_provider,
    TemplateStrategy,
    MessageTemplate,
    MessageBuilder,
    ResponseParser,
    TokenCounter,
    CostCalculator,
    render_conditional_template,
)

from dataknobs_llm.tools import (
    Tool,
    ToolRegistry,
)

# Execution utilities
from dataknobs_llm.execution.parallel import (
    DeterministicTask,
    LLMTask,
    ParallelLLMExecutor,
    TaskResult,
)

# Exceptions
from dataknobs_llm.exceptions import ResponseQueueExhaustedError, ToolsNotSupportedError

# Testing utilities (for test code)
from dataknobs_llm.testing import (
    CallTracker,
    CapturedCall,
    CapturingProvider,
    ErrorResponse,
    ResponseSequenceBuilder,
    extraction_response,
    llm_message_from_dict,
    llm_message_to_dict,
    llm_response_from_dict,
    llm_response_to_dict,
    multi_tool_response,
    text_response,
    tool_call_from_dict,
    tool_call_response,
    tool_call_to_dict,
)

__version__ = "0.5.4"

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
    "CachingEmbedProvider",
    "EmbeddingCache",
    "MemoryEmbeddingCache",
    "LLMProviderFactory",
    "create_llm_provider",
    "create_caching_provider",
    # Utils
    "TemplateStrategy",
    "MessageTemplate",
    "MessageBuilder",
    "ResponseParser",
    "TokenCounter",
    "CostCalculator",
    "render_conditional_template",
    # Tools
    "Tool",
    "ToolRegistry",
    # Execution
    "ParallelLLMExecutor",
    "LLMTask",
    "DeterministicTask",
    "TaskResult",
    # Exceptions
    "ResponseQueueExhaustedError",
    "ToolsNotSupportedError",
    # Testing utilities
    "CallTracker",
    "CapturedCall",
    "CapturingProvider",
    "ErrorResponse",
    "ResponseSequenceBuilder",
    "extraction_response",
    "llm_message_from_dict",
    "llm_message_to_dict",
    "llm_response_from_dict",
    "llm_response_to_dict",
    "multi_tool_response",
    "text_response",
    "tool_call_from_dict",
    "tool_call_response",
    "tool_call_to_dict",
]
