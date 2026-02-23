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
from dataknobs_llm.exceptions import ToolsNotSupportedError

# Testing utilities (for test code)
from dataknobs_llm.testing import (
    ErrorResponse,
    text_response,
    tool_call_response,
    multi_tool_response,
    extraction_response,
    ResponseSequenceBuilder,
)

__version__ = "0.3.6"

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
    "ToolsNotSupportedError",
    # Testing utilities
    "ErrorResponse",
    "text_response",
    "tool_call_response",
    "multi_tool_response",
    "extraction_response",
    "ResponseSequenceBuilder",
]
