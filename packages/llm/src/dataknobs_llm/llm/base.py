"""Base LLM abstraction components.

This module provides the base abstractions for unified LLM operations across
different providers (OpenAI, Anthropic, Ollama, etc.). It defines standard
interfaces for completions, streaming, embeddings, and function calling.

The architecture follows a provider pattern where all LLM providers implement
common interfaces (AsyncLLMProvider or SyncLLMProvider) and use standardized
data structures (LLMMessage, LLMResponse, LLMConfig).

Key Components:
    - LLMProvider: Base provider interface with initialization and lifecycle
    - AsyncLLMProvider: Async provider with complete(), stream_complete(), embed()
    - SyncLLMProvider: Synchronous version for non-async applications
    - LLMMessage: Standard message format for conversations
    - LLMResponse: Standard response with content, usage, and cost tracking
    - LLMConfig: Comprehensive configuration with 20+ parameters
    - LLMAdapter: Format adapters for provider-specific APIs
    - LLMMiddleware: Request/response processing pipeline

Example:
    ```python
    from dataknobs_llm import create_llm_provider
    from dataknobs_llm.llm.base import LLMConfig, LLMMessage

    # Create provider with config
    config = LLMConfig(
        provider="openai",
        model="gpt-4",
        temperature=0.7,
        max_tokens=500
    )

    # Async usage
    async with create_llm_provider(config) as llm:
        # Simple completion
        response = await llm.complete("What is Python?")
        print(response.content)

        # Streaming
        async for chunk in llm.stream_complete("Tell me a story"):
            print(chunk.delta, end="", flush=True)

        # Multi-turn conversation
        messages = [
            LLMMessage(role="system", content="You are helpful"),
            LLMMessage(role="user", content="Hello!"),
        ]
        response = await llm.complete(messages)
    ```

See Also:
    - dataknobs_llm.llm.providers: Provider implementations
    - dataknobs_llm.conversations: Multi-turn conversation management
    - dataknobs_llm.prompts: Prompt rendering and RAG integration
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any, Dict, List, Union, AsyncIterator, Iterator,
    Callable, Protocol
)
from datetime import datetime

# Import prompt builder types - clean one-way dependency (llm depends on prompts)
from dataknobs_llm.prompts import AsyncPromptBuilder, PromptBuilder
from dataknobs_config.config import Config


class CompletionMode(Enum):
    """LLM completion modes.

    Defines the operation mode for LLM requests. Different modes use
    different APIs and formatting requirements.

    Attributes:
        CHAT: Chat completion with conversational message history
        TEXT: Raw text completion (legacy models)
        INSTRUCT: Instruction-following mode
        EMBEDDING: Generate vector embeddings for semantic search
        FUNCTION: Function/tool calling mode

    Example:
        ```python
        from dataknobs_llm.llm.base import LLMConfig, CompletionMode

        # Chat mode (default for modern models)
        config = LLMConfig(
            provider="openai",
            model="gpt-4",
            mode=CompletionMode.CHAT
        )

        # Embedding mode for vector search
        embedding_config = LLMConfig(
            provider="openai",
            model="text-embedding-ada-002",
            mode=CompletionMode.EMBEDDING
        )
        ```
    """
    CHAT = "chat"  # Chat completion with message history
    TEXT = "text"  # Text completion
    INSTRUCT = "instruct"  # Instruction following
    EMBEDDING = "embedding"  # Generate embeddings
    FUNCTION = "function"  # Function calling


class ModelCapability(Enum):
    """Model capabilities.

    Enumerates the capabilities that different LLM models support.
    Providers use this to advertise what features are available for
    a specific model.

    Attributes:
        TEXT_GENERATION: Basic text generation
        CHAT: Multi-turn conversational interactions
        EMBEDDINGS: Vector embedding generation
        FUNCTION_CALLING: Tool/function calling support
        VISION: Image understanding capabilities
        CODE: Code generation and analysis
        JSON_MODE: Structured JSON output
        STREAMING: Incremental response streaming

    Example:
        ```python
        from dataknobs_llm import create_llm_provider
        from dataknobs_llm.llm.base import ModelCapability

        # Check model capabilities
        llm = create_llm_provider("openai", model="gpt-4")
        capabilities = llm.get_capabilities()

        if ModelCapability.STREAMING in capabilities:
            # Use streaming
            async for chunk in llm.stream_complete("Hello"):
                print(chunk.delta, end="")

        if ModelCapability.FUNCTION_CALLING in capabilities:
            # Use function calling
            response = await llm.function_call(messages, functions)
        ```
    """
    TEXT_GENERATION = "text_generation"
    CHAT = "chat"
    EMBEDDINGS = "embeddings"
    FUNCTION_CALLING = "function_calling"
    VISION = "vision"
    CODE = "code"
    JSON_MODE = "json_mode"
    STREAMING = "streaming"


@dataclass
class ToolCall:
    """Represents a tool call from the LLM.

    Used when the LLM wants to invoke a tool/function during reasoning.

    Attributes:
        name: Name of the tool to call
        parameters: Arguments to pass to the tool
        id: Optional unique identifier for the tool call
    """
    name: str
    parameters: Dict[str, Any]
    id: str | None = None


@dataclass
class LLMMessage:
    """Represents a message in LLM conversation.

    Standard message format used across all providers. Messages are the
    fundamental unit of LLM interactions, containing role-based content
    for multi-turn conversations.

    Attributes:
        role: Message role - 'system', 'user', 'assistant', or 'function'
        content: Message content text
        name: Optional name for function messages or multi-user scenarios
        function_call: Function call data for tool-using models
        metadata: Additional metadata (timestamps, IDs, etc.)

    Example:
        ```python
        from dataknobs_llm.llm.base import LLMMessage

        # System message
        system_msg = LLMMessage(
            role="system",
            content="You are a helpful coding assistant."
        )

        # User message
        user_msg = LLMMessage(
            role="user",
            content="How do I reverse a list in Python?"
        )

        # Assistant message
        assistant_msg = LLMMessage(
            role="assistant",
            content="Use the reverse() method or [::-1] slicing."
        )

        # Function result message
        function_msg = LLMMessage(
            role="function",
            name="search_docs",
            content='{"result": "Found 3 examples"}'
        )

        # Build conversation
        messages = [system_msg, user_msg, assistant_msg]
        ```
    """
    role: str  # 'system', 'user', 'assistant', 'function'
    content: str
    name: str | None = None  # For function messages
    function_call: Dict[str, Any] | None = None  # For function calling
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    """Response from LLM.

    Standard response format returned by all LLM providers. Contains the
    generated content along with metadata about token usage, cost, and
    completion status.

    Attributes:
        content: Generated text content
        model: Model identifier that generated the response
        finish_reason: Why generation stopped - 'stop', 'length', 'function_call'
        usage: Token usage stats (prompt_tokens, completion_tokens, total_tokens)
        function_call: Function call data if model requested tool use
        metadata: Provider-specific metadata
        created_at: Response timestamp
        cost_usd: Estimated cost in USD for this request
        cumulative_cost_usd: Running total cost for conversation

    Example:
        ```python
        from dataknobs_llm import create_llm_provider

        llm = create_llm_provider("openai", model="gpt-4")
        response = await llm.complete("What is Python?")

        # Access response data
        print(response.content)
        # => "Python is a high-level programming language..."

        # Check token usage
        print(f"Tokens used: {response.usage['total_tokens']}")
        # => Tokens used: 87

        # Monitor costs
        if response.cost_usd:
            print(f"Cost: ${response.cost_usd:.4f}")
            print(f"Total: ${response.cumulative_cost_usd:.4f}")

        # Check completion status
        if response.finish_reason == "length":
            print("Response truncated due to max_tokens limit")
        ```

    See Also:
        LLMMessage: Request message format
        LLMStreamResponse: Streaming response format
    """
    content: str
    model: str
    finish_reason: str | None = None  # 'stop', 'length', 'function_call', 'tool_calls'
    usage: Dict[str, int] | None = None  # tokens used
    function_call: Dict[str, Any] | None = None  # Legacy single function call
    tool_calls: list["ToolCall"] | None = None  # List of tool calls (preferred)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    # Cost tracking (optional enhancement for DynaBot)
    cost_usd: float | None = None  # Estimated cost in USD
    cumulative_cost_usd: float | None = None  # Running total for conversation


@dataclass
class LLMStreamResponse:
    r"""Streaming response from LLM.

    Represents a single chunk in a streaming LLM response. Streaming
    allows displaying generated text incrementally as it's produced,
    providing better user experience for long responses.

    Attributes:
        delta: Incremental content for this chunk (not cumulative)
        is_final: True if this is the last chunk in the stream
        finish_reason: Why generation stopped (only set on final chunk)
        usage: Token usage stats (only set on final chunk)
        metadata: Additional chunk metadata

    Example:
        ```python
        from dataknobs_llm import create_llm_provider

        llm = create_llm_provider("openai", model="gpt-4")

        # Stream and display in real-time
        async for chunk in llm.stream_complete("Write a poem"):
            print(chunk.delta, end="", flush=True)

            if chunk.is_final:
                print(f"\n\nFinished: {chunk.finish_reason}")
                print(f"Tokens: {chunk.usage['total_tokens']}")

        # Accumulate full response
        full_text = ""
        chunks_received = 0

        async for chunk in llm.stream_complete("Explain Python"):
            full_text += chunk.delta
            chunks_received += 1

            # Optional: show progress
            if chunks_received % 10 == 0:
                print(f"Received {chunks_received} chunks...")

        print(f"\nComplete response ({len(full_text)} chars)")
        print(full_text)
        ```

    See Also:
        LLMResponse: Non-streaming response format
        AsyncLLMProvider.stream_complete: Streaming method
    """
    delta: str  # Incremental content
    is_final: bool = False
    finish_reason: str | None = None
    usage: Dict[str, int] | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMConfig:
    """Configuration for LLM operations.

    Comprehensive configuration for LLM providers with 20+ parameters
    controlling generation, rate limiting, function calling, and more.
    Works seamlessly with both direct instantiation and dataknobs Config objects.

    This class supports:
    - All major LLM providers (OpenAI, Anthropic, Ollama, HuggingFace)
    - Generation parameters (temperature, max_tokens, top_p, etc.)
    - Function/tool calling configuration
    - Streaming with callbacks
    - Rate limiting and retry logic
    - Provider-specific options via options dict

    Example:
        ```python
        from dataknobs_llm.llm.base import LLMConfig, CompletionMode

        # Basic configuration
        config = LLMConfig(
            provider="openai",
            model="gpt-4",
            api_key="sk-...",
            temperature=0.7,
            max_tokens=500
        )

        # Creative writing config
        creative_config = LLMConfig(
            provider="anthropic",
            model="claude-3-sonnet",
            temperature=1.2,
            top_p=0.95,
            max_tokens=2000
        )

        # Deterministic config for testing
        test_config = LLMConfig(
            provider="openai",
            model="gpt-4",
            temperature=0.0,
            seed=42,  # Reproducible outputs
            max_tokens=100
        )

        # Function calling config
        function_config = LLMConfig(
            provider="openai",
            model="gpt-4",
            functions=[{
                "name": "search_docs",
                "description": "Search documentation",
                "parameters": {"type": "object", "properties": {...}}
            }],
            function_call="auto"
        )

        # Streaming with callback
        def on_chunk(chunk):
            print(chunk.delta, end="")

        streaming_config = LLMConfig(
            provider="openai",
            model="gpt-4",
            stream=True,
            stream_callback=on_chunk
        )

        # From dictionary (Config compatibility)
        config_dict = {
            "provider": "ollama",
            "model": "llama2",
            "type": "llm",  # Config metadata (ignored)
            "temperature": 0.8
        }
        config = LLMConfig.from_dict(config_dict)

        # Clone with overrides
        new_config = config.clone(temperature=1.0, max_tokens=1000)
        ```

    See Also:
        normalize_llm_config: Convert various formats to LLMConfig
        CompletionMode: Available completion modes
    """
    provider: str  # 'openai', 'anthropic', 'ollama', etc.
    model: str  # Model name/identifier
    api_key: str | None = None
    api_base: str | None = None  # Custom API endpoint

    # Generation parameters
    temperature: float = 0.7
    max_tokens: int | None = None
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop_sequences: List[str] | None = None

    # Mode settings
    mode: CompletionMode = CompletionMode.CHAT
    system_prompt: str | None = None
    response_format: str | None = None  # 'text' or 'json'

    # Function calling
    functions: List[Dict[str, Any]] | None = None
    function_call: Union[str, Dict[str, str]] | None = None  # 'auto', 'none', or specific function

    # Streaming
    stream: bool = False
    stream_callback: Callable[[LLMStreamResponse], None] | None = None

    # Rate limiting
    rate_limit: int | None = None  # Requests per minute
    retry_count: int = 3
    retry_delay: float = 1.0
    timeout: float = 60.0

    # Advanced settings
    seed: int | None = None  # For reproducibility
    logit_bias: Dict[str, float] | None = None
    user_id: str | None = None

    # Provider-specific options
    options: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "LLMConfig":
        """Create LLMConfig from a dictionary.

        This method handles dictionaries from dataknobs Config objects,
        which may include 'type', 'name', and 'factory' attributes.
        These attributes are ignored during LLMConfig construction.

        Args:
            config_dict: Configuration dictionary

        Returns:
            LLMConfig instance
        """
        # Filter out Config-specific attributes
        config_data = {
            k: v for k, v in config_dict.items()
            if k not in ('type', 'name', 'factory')
        }

        # Handle mode conversion if it's a string
        if 'mode' in config_data and isinstance(config_data['mode'], str):
            config_data['mode'] = CompletionMode(config_data['mode'])

        # Get dataclass fields to filter unknown attributes
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in config_data.items() if k in valid_fields}

        return cls(**filtered_data)

    def to_dict(self, include_config_attrs: bool = False) -> Dict[str, Any]:
        """Convert LLMConfig to a dictionary.

        Args:
            include_config_attrs: If True, includes 'type' attribute for Config compatibility

        Returns:
            Configuration dictionary
        """
        result = {}

        for field_info in self.__dataclass_fields__.values():
            value = getattr(self, field_info.name)

            # Handle enum conversion
            if isinstance(value, Enum):
                result[field_info.name] = value.value
            # Skip None values for optional fields
            elif value is not None:
                result[field_info.name] = value
            # Include default factories even if empty for certain fields
            elif field_info.name == 'options':
                result[field_info.name] = {}

        # Optionally add Config-compatible type attribute
        if include_config_attrs:
            result['type'] = 'llm'

        return result

    def clone(self, **overrides: Any) -> "LLMConfig":
        """Create a copy of this config with optional overrides.

        This method is useful for creating runtime configuration variations
        without mutating the original config. All dataclass fields can be
        overridden via keyword arguments.

        Args:
            **overrides: Field values to override in the cloned config

        Returns:
            New LLMConfig instance with overrides applied

        Example:
            >>> base_config = LLMConfig(provider="openai", model="gpt-4", temperature=0.7)
            >>> creative_config = base_config.clone(temperature=1.2, max_tokens=500)
        """
        from dataclasses import replace
        return replace(self, **overrides)


def normalize_llm_config(config: Union["LLMConfig", Config, Dict[str, Any]]) -> "LLMConfig":
    """Normalize various config formats to LLMConfig.

    This helper function accepts LLMConfig instances, dataknobs Config objects,
    or plain dictionaries and returns a standardized LLMConfig instance.

    Args:
        config: Configuration as LLMConfig, Config object, or dictionary

    Returns:
        LLMConfig instance

    Raises:
        TypeError: If config type is not supported
    """
    # Already an LLMConfig instance
    if isinstance(config, LLMConfig):
        return config

    # Dictionary (possibly from Config.get())
    if isinstance(config, dict):
        return LLMConfig.from_dict(config)

    # dataknobs Config object - try to get the config dict
    # We check for the get method to identify Config objects
    if hasattr(config, 'get') and hasattr(config, 'get_types'):
        # It's a Config object, extract the llm configuration
        # Try to get first llm config, or fall back to first available type
        try:
            config_dict = config.get('llm', 0)
        except Exception as e:
            # If no 'llm' type, try to get first available config of any type
            types = config.get_types()
            if types:
                config_dict = config.get(types[0], 0)
            else:
                raise ValueError("Config object has no configurations") from e

        return LLMConfig.from_dict(config_dict)

    raise TypeError(
        f"Unsupported config type: {type(config).__name__}. "
        f"Expected LLMConfig, Config, or dict."
    )


class LLMProvider(ABC):
    """Base LLM provider interface."""

    def __init__(
        self,
        config: Union[LLMConfig, Config, Dict[str, Any]],
        prompt_builder: Union[PromptBuilder, AsyncPromptBuilder] | None = None
    ):
        """Initialize provider with configuration.

        Args:
            config: Configuration as LLMConfig, dataknobs Config object, or dict
            prompt_builder: Optional prompt builder for integrated prompting
        """
        self.config = normalize_llm_config(config)
        self.prompt_builder = prompt_builder
        self._client = None
        self._is_initialized = False

    def _validate_prompt_builder(self, expected_type: type) -> None:
        """Validate that prompt builder is configured and of correct type.

        Args:
            expected_type: Expected builder type (PromptBuilder or AsyncPromptBuilder)

        Raises:
            ValueError: If prompt_builder not configured
            TypeError: If prompt_builder is wrong type
        """
        if not self.prompt_builder:
            raise ValueError(
                "No prompt_builder configured. Pass prompt_builder to __init__() "
                "or use complete() directly with pre-rendered messages."
            )

        if not isinstance(self.prompt_builder, expected_type):
            raise TypeError(
                f"{self.__class__.__name__} requires {expected_type.__name__}, "
                f"got {type(self.prompt_builder).__name__}"
            )

    def _validate_render_params(
        self,
        prompt_type: str
    ) -> None:
        """Validate render parameters.

        Args:
            prompt_type: Type of prompt to render

        Raises:
            ValueError: If prompt_type is invalid
        """
        if prompt_type not in ("system", "user", "both"):
            raise ValueError(
                f"Invalid prompt_type: {prompt_type}. "
                f"Must be 'system', 'user', or 'both'"
            )

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the LLM client."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the LLM client."""
        pass

    @abstractmethod
    def validate_model(self) -> bool:
        """Validate that the model is available."""
        pass

    @abstractmethod
    def get_capabilities(self) -> List[ModelCapability]:
        """Get model capabilities."""
        pass

    @property
    def is_initialized(self) -> bool:
        """Check if provider is initialized."""
        return self._is_initialized

    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


class AsyncLLMProvider(LLMProvider):
    """Async LLM provider interface."""

    @abstractmethod
    async def complete(
        self,
        messages: Union[str, List[LLMMessage]],
        **kwargs
    ) -> LLMResponse:
        """Generate completion asynchronously.

        Primary method for getting LLM responses. Accepts either a simple
        string prompt or a list of LLMMessage objects for multi-turn
        conversations. This is the recommended async method for most use cases.

        Args:
            messages: Either a single string prompt or a list of LLMMessage
                objects for multi-turn conversations.
            **kwargs: Additional provider-specific parameters. Common options:
                - temperature (float): Sampling temperature (0.0-2.0)
                - max_tokens (int): Maximum tokens to generate
                - top_p (float): Nucleus sampling parameter (0.0-1.0)
                - stop (List[str]): Stop sequences
                - presence_penalty (float): Presence penalty (-2.0 to 2.0)
                - frequency_penalty (float): Frequency penalty (-2.0 to 2.0)

        Returns:
            LLMResponse containing generated content, usage stats, and metadata

        Raises:
            ValueError: If messages format is invalid
            ConnectionError: If API connection fails
            TimeoutError: If request exceeds timeout

        Example:
            ```python
            from dataknobs_llm import create_llm_provider
            from dataknobs_llm.llm.base import LLMMessage

            llm = create_llm_provider("openai", model="gpt-4")

            # Simple string prompt
            response = await llm.complete("What is Python?")
            print(response.content)
            # => "Python is a high-level programming language..."

            # With parameters
            response = await llm.complete(
                "Write a haiku about coding",
                temperature=0.9,
                max_tokens=100
            )

            # Multi-turn conversation
            messages = [
                LLMMessage(role="system", content="You are a helpful tutor"),
                LLMMessage(role="user", content="Explain recursion"),
                LLMMessage(role="assistant", content="Recursion is when..."),
                LLMMessage(role="user", content="Can you give an example?")
            ]
            response = await llm.complete(messages)

            # Check token usage
            print(f"Tokens: {response.usage['total_tokens']}")
            print(f"Cost: ${response.cost_usd:.4f}")
            ```

        See Also:
            stream_complete: Streaming version
            render_and_complete: Complete with prompt rendering
        """
        pass

    async def render_and_complete(
        self,
        prompt_name: str,
        params: Dict[str, Any] | None = None,
        prompt_type: str = "user",
        index: int = 0,
        include_rag: bool = True,
        **llm_kwargs
    ) -> LLMResponse:
        """Render prompt from library and execute LLM completion.

        This is a convenience method for one-off interactions that combines
        prompt rendering with LLM execution. For multi-turn conversations,
        use ConversationManager instead.

        Args:
            prompt_name: Name of prompt in library
            params: Parameters for template rendering
            prompt_type: Type of prompt ("system", "user", or "both")
            index: Prompt variant index (for user prompts)
            include_rag: Whether to execute RAG searches
            **llm_kwargs: Additional arguments passed to complete()

        Returns:
            LLM response

        Raises:
            ValueError: If prompt_builder not configured or invalid prompt_type
            TypeError: If prompt_builder is not AsyncPromptBuilder

        Example:
            >>> llm = OpenAIProvider(config, prompt_builder=builder)
            >>> result = await llm.render_and_complete(
            ...     "analyze_code",
            ...     params={"code": code, "language": "python"}
            ... )
        """
        # Validate
        from dataknobs_llm.prompts import AsyncPromptBuilder
        self._validate_prompt_builder(AsyncPromptBuilder)
        self._validate_render_params(prompt_type)

        # Render messages
        messages = await self._render_messages(
            prompt_name, params, prompt_type, index, include_rag
        )

        # Execute LLM
        return await self.complete(messages, **llm_kwargs)

    async def render_and_stream(
        self,
        prompt_name: str,
        params: Dict[str, Any] | None = None,
        prompt_type: str = "user",
        index: int = 0,
        include_rag: bool = True,
        **llm_kwargs
    ) -> AsyncIterator[LLMStreamResponse]:
        """Render prompt and stream LLM response.

        Same as render_and_complete() but returns streaming response.

        Args:
            prompt_name: Name of prompt in library
            params: Parameters for template rendering
            prompt_type: Type of prompt ("system", "user", or "both")
            index: Prompt variant index
            include_rag: Whether to execute RAG searches
            **llm_kwargs: Additional arguments passed to stream_complete()

        Yields:
            Streaming response chunks

        Raises:
            ValueError: If prompt_builder not configured or invalid prompt_type
            TypeError: If prompt_builder is not AsyncPromptBuilder

        Example:
            >>> async for chunk in llm.render_and_stream("analyze_code", params={"code": code}):
            ...     print(chunk.delta, end="")
        """
        # Validate
        from dataknobs_llm.prompts import AsyncPromptBuilder
        self._validate_prompt_builder(AsyncPromptBuilder)
        self._validate_render_params(prompt_type)

        # Render messages
        messages = await self._render_messages(
            prompt_name, params, prompt_type, index, include_rag
        )

        # Stream LLM response
        async for chunk in self.stream_complete(messages, **llm_kwargs):
            yield chunk

    async def _render_messages(
        self,
        prompt_name: str,
        params: Dict[str, Any] | None,
        prompt_type: str,
        index: int,
        include_rag: bool
    ) -> List[LLMMessage]:
        """Render messages from prompt library (async version).

        Args:
            prompt_name: Name of prompt in library
            params: Parameters for template rendering
            prompt_type: Type of prompt ("system", "user", or "both")
            index: Prompt variant index
            include_rag: Whether to execute RAG searches

        Returns:
            List of rendered LLM messages
        """
        from dataknobs_llm.prompts import AsyncPromptBuilder
        builder: AsyncPromptBuilder = self.prompt_builder  # type: ignore

        messages: List[LLMMessage] = []
        params = params or {}

        if prompt_type in ("system", "both"):
            result = await builder.render_system_prompt(
                prompt_name, params=params, include_rag=include_rag
            )
            messages.append(LLMMessage(role="system", content=result.content))

        if prompt_type in ("user", "both"):
            result = await builder.render_user_prompt(
                prompt_name, index=index, params=params, include_rag=include_rag
            )
            messages.append(LLMMessage(role="user", content=result.content))

        return messages
        
    @abstractmethod
    async def stream_complete(
        self,
        messages: Union[str, List[LLMMessage]],
        **kwargs
    ) -> AsyncIterator[LLMStreamResponse]:
        r"""Generate streaming completion asynchronously.

        Streams response chunks as they are generated, enabling real-time
        display of LLM output. Each chunk contains incremental content
        (delta), and the final chunk includes usage statistics.

        Args:
            messages: Either a single string prompt or list of LLMMessage objects
            **kwargs: Provider-specific parameters (same as complete())

        Yields:
            LLMStreamResponse chunks containing incremental content. The final
            chunk has is_final=True and includes finish_reason and usage stats.

        Raises:
            ValueError: If messages format is invalid
            ConnectionError: If API connection fails
            TimeoutError: If request exceeds timeout

        Example:
            ```python
            from dataknobs_llm import create_llm_provider

            llm = create_llm_provider("openai", model="gpt-4")

            # Stream and display in real-time
            async for chunk in llm.stream_complete("Tell me a story"):
                print(chunk.delta, end="", flush=True)

                if chunk.is_final:
                    print(f"\n\nFinished: {chunk.finish_reason}")
                    print(f"Total tokens: {chunk.usage['total_tokens']}")

            # Accumulate full response
            full_text = ""
            chunk_count = 0

            async for chunk in llm.stream_complete("Explain quantum computing"):
                full_text += chunk.delta
                chunk_count += 1

            print(f"Received {chunk_count} chunks")
            print(f"Total length: {len(full_text)} characters")

            # Stream with progress callback
            async def stream_with_progress(prompt: str):
                chunks = []
                async for chunk in llm.stream_complete(prompt):
                    chunks.append(chunk)
                    # Update progress UI
                    if len(chunks) % 5 == 0:
                        print(f"Processing... ({len(chunks)} chunks)")
                return "".join(c.delta for c in chunks)

            result = await stream_with_progress("Write a tutorial")
            ```

        See Also:
            complete: Non-streaming version
            render_and_stream: Stream with prompt rendering
            LLMStreamResponse: Chunk data structure
        """
        pass
        
    @abstractmethod
    async def embed(
        self,
        texts: Union[str, List[str]],
        **kwargs
    ) -> Union[List[float], List[List[float]]]:
        """Generate embeddings asynchronously.

        Converts text into dense vector representations for semantic search,
        clustering, and similarity comparison. Returns high-dimensional
        vectors (typically 768-1536 dimensions depending on model).

        Args:
            texts: Single text string or list of texts to embed
            **kwargs: Provider-specific parameters:
                - model (str): Embedding model override
                - dimensions (int): Target dimensions (if supported)

        Returns:
            Single embedding vector (List[float]) if input is a string,
            or list of vectors (List[List[float]]) if input is a list

        Raises:
            ValueError: If texts is empty or invalid
            ConnectionError: If API connection fails

        Example:
            ```python
            from dataknobs_llm import create_llm_provider
            import numpy as np

            # Create embedding provider
            llm = create_llm_provider(
                "openai",
                model="text-embedding-ada-002"
            )

            # Single text embedding
            embedding = await llm.embed("What is machine learning?")
            print(f"Dimensions: {len(embedding)}")
            # => Dimensions: 1536

            # Batch embedding
            texts = [
                "Python is a programming language",
                "JavaScript is used for web development",
                "Machine learning uses statistical methods"
            ]
            embeddings = await llm.embed(texts)
            print(f"Generated {len(embeddings)} embeddings")

            # Compute similarity
            def cosine_similarity(v1, v2):
                return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

            query_emb = await llm.embed("Tell me about ML")
            similarities = [
                cosine_similarity(query_emb, emb)
                for emb in embeddings
            ]
            most_similar_idx = np.argmax(similarities)
            print(f"Most similar: {texts[most_similar_idx]}")
            # => Most similar: Machine learning uses statistical methods

            # Store in vector database
            from dataknobs_data import database_factory
            db = database_factory.create("vector_db")
            for text, emb in zip(texts, embeddings):
                db.create({"text": text, "embedding": emb})
            ```

        See Also:
            complete: Text generation method
        """
        pass
        
    @abstractmethod
    async def function_call(
        self,
        messages: List[LLMMessage],
        functions: List[Dict[str, Any]],
        **kwargs
    ) -> LLMResponse:
        """Execute function calling asynchronously.

        Enables the LLM to call external functions/tools. The model decides
        which function to call based on the conversation context, and returns
        the function name and arguments in a structured format.

        Args:
            messages: Conversation messages leading up to the function call
            functions: List of function definitions in JSON Schema format.
                Each function dict must have:
                - name (str): Function name
                - description (str): What the function does
                - parameters (dict): JSON Schema for parameters
            **kwargs: Provider-specific parameters:
                - function_call (str|dict): 'auto', 'none', or specific function
                - temperature (float): Sampling temperature
                - max_tokens (int): Maximum response tokens

        Returns:
            LLMResponse with function_call field populated containing:
            - name (str): Function to call
            - arguments (str): JSON string of arguments

        Raises:
            ValueError: If functions format is invalid
            ConnectionError: If API connection fails

        Example:
            ```python
            from dataknobs_llm import create_llm_provider
            from dataknobs_llm.llm.base import LLMMessage
            import json

            llm = create_llm_provider("openai", model="gpt-4")

            # Define available functions
            functions = [
                {
                    "name": "search_docs",
                    "description": "Search documentation for information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Max results"
                            }
                        },
                        "required": ["query"]
                    }
                },
                {
                    "name": "execute_code",
                    "description": "Execute Python code",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {"type": "string"}
                        },
                        "required": ["code"]
                    }
                }
            ]

            # Ask question that requires function
            messages = [
                LLMMessage(
                    role="user",
                    content="Search for information about async/await in Python"
                )
            ]

            # Model decides to call function
            response = await llm.function_call(messages, functions)

            if response.function_call:
                func_name = response.function_call["name"]
                func_args = json.loads(response.function_call["arguments"])

                print(f"Function: {func_name}")
                print(f"Arguments: {func_args}")
                # => Function: search_docs
                # => Arguments: {'query': 'async/await Python', 'limit': 5}

                # Execute function
                results = search_docs(**func_args)

                # Add function result to conversation
                messages.append(LLMMessage(
                    role="function",
                    name=func_name,
                    content=json.dumps(results)
                ))

                # Get final response
                final = await llm.complete(messages)
                print(final.content)
            ```

        See Also:
            complete: Standard completion without functions
            dataknobs_llm.tools: Tool abstraction framework
        """
        pass
        
    async def initialize(self) -> None:
        """Initialize the async LLM client."""
        self._is_initialized = True
        
    async def close(self) -> None:
        """Close the async LLM client."""
        self._is_initialized = False
        
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


class SyncLLMProvider(LLMProvider):
    """Synchronous LLM provider interface."""

    @abstractmethod
    def complete(
        self,
        messages: Union[str, List[LLMMessage]],
        **kwargs
    ) -> LLMResponse:
        """Generate completion synchronously.

        Args:
            messages: Input messages or prompt
            **kwargs: Additional parameters

        Returns:
            LLM response
        """
        pass

    def render_and_complete(
        self,
        prompt_name: str,
        params: Dict[str, Any] | None = None,
        prompt_type: str = "user",
        index: int = 0,
        include_rag: bool = True,
        **llm_kwargs
    ) -> LLMResponse:
        """Render prompt from library and execute LLM completion.

        This is a convenience method for one-off interactions that combines
        prompt rendering with LLM execution. For multi-turn conversations,
        use ConversationManager instead.

        Args:
            prompt_name: Name of prompt in library
            params: Parameters for template rendering
            prompt_type: Type of prompt ("system", "user", or "both")
            index: Prompt variant index (for user prompts)
            include_rag: Whether to execute RAG searches
            **llm_kwargs: Additional arguments passed to complete()

        Returns:
            LLM response

        Raises:
            ValueError: If prompt_builder not configured or invalid prompt_type
            TypeError: If prompt_builder is not PromptBuilder

        Example:
            >>> llm = SyncOpenAIProvider(config, prompt_builder=builder)
            >>> result = llm.render_and_complete(
            ...     "analyze_code",
            ...     params={"code": code, "language": "python"}
            ... )
        """
        # Validate
        from dataknobs_llm.prompts import PromptBuilder
        self._validate_prompt_builder(PromptBuilder)
        self._validate_render_params(prompt_type)

        # Render messages
        messages = self._render_messages(
            prompt_name, params, prompt_type, index, include_rag
        )

        # Execute LLM
        return self.complete(messages, **llm_kwargs)

    def render_and_stream(
        self,
        prompt_name: str,
        params: Dict[str, Any] | None = None,
        prompt_type: str = "user",
        index: int = 0,
        include_rag: bool = True,
        **llm_kwargs
    ) -> Iterator[LLMStreamResponse]:
        """Render prompt and stream LLM response.

        Same as render_and_complete() but returns streaming response.

        Args:
            prompt_name: Name of prompt in library
            params: Parameters for template rendering
            prompt_type: Type of prompt ("system", "user", or "both")
            index: Prompt variant index
            include_rag: Whether to execute RAG searches
            **llm_kwargs: Additional arguments passed to stream_complete()

        Yields:
            Streaming response chunks

        Raises:
            ValueError: If prompt_builder not configured or invalid prompt_type
            TypeError: If prompt_builder is not PromptBuilder

        Example:
            >>> for chunk in llm.render_and_stream("analyze_code", params={"code": code}):
            ...     print(chunk.delta, end="")
        """
        # Validate
        from dataknobs_llm.prompts import PromptBuilder
        self._validate_prompt_builder(PromptBuilder)
        self._validate_render_params(prompt_type)

        # Render messages
        messages = self._render_messages(
            prompt_name, params, prompt_type, index, include_rag
        )

        # Stream LLM response
        for chunk in self.stream_complete(messages, **llm_kwargs):
            yield chunk

    def _render_messages(
        self,
        prompt_name: str,
        params: Dict[str, Any] | None,
        prompt_type: str,
        index: int,
        include_rag: bool
    ) -> List[LLMMessage]:
        """Render messages from prompt library (sync version).

        Args:
            prompt_name: Name of prompt in library
            params: Parameters for template rendering
            prompt_type: Type of prompt ("system", "user", or "both")
            index: Prompt variant index
            include_rag: Whether to execute RAG searches

        Returns:
            List of rendered LLM messages
        """
        from dataknobs_llm.prompts import PromptBuilder
        builder: PromptBuilder = self.prompt_builder  # type: ignore

        messages: List[LLMMessage] = []
        params = params or {}

        if prompt_type in ("system", "both"):
            result = builder.render_system_prompt(
                prompt_name, params=params, include_rag=include_rag
            )
            messages.append(LLMMessage(role="system", content=result.content))

        if prompt_type in ("user", "both"):
            result = builder.render_user_prompt(
                prompt_name, index=index, params=params, include_rag=include_rag
            )
            messages.append(LLMMessage(role="user", content=result.content))

        return messages

    @abstractmethod
    def stream_complete(
        self,
        messages: Union[str, List[LLMMessage]],
        **kwargs
    ) -> Iterator[LLMStreamResponse]:
        """Generate streaming completion synchronously.

        Args:
            messages: Input messages or prompt
            **kwargs: Additional parameters

        Yields:
            Streaming response chunks
        """
        pass
        
    @abstractmethod
    def embed(
        self,
        texts: Union[str, List[str]],
        **kwargs
    ) -> Union[List[float], List[List[float]]]:
        """Generate embeddings synchronously.
        
        Args:
            texts: Input text(s)
            **kwargs: Additional parameters
            
        Returns:
            Embedding vector(s)
        """
        pass
        
    @abstractmethod
    def function_call(
        self,
        messages: List[LLMMessage],
        functions: List[Dict[str, Any]],
        **kwargs
    ) -> LLMResponse:
        """Execute function calling synchronously.
        
        Args:
            messages: Conversation messages
            functions: Available functions
            **kwargs: Additional parameters
            
        Returns:
            Response with function call
        """
        pass
        
    def initialize(self) -> None:
        """Initialize the sync LLM client."""
        self._is_initialized = True
        
    def close(self) -> None:
        """Close the sync LLM client."""
        self._is_initialized = False


class LLMAdapter(ABC):
    """Base adapter for converting between different LLM formats.

    Adapters translate between the standard dataknobs LLM format
    (LLMMessage, LLMResponse, LLMConfig) and provider-specific formats
    (OpenAI, Anthropic, etc.). Each provider implementation should
    have a corresponding adapter.

    This enables provider-agnostic code that works across different
    LLM APIs without modification.

    Example:
        ```python
        from dataknobs_llm.llm.base import LLMAdapter, LLMMessage, LLMResponse
        from typing import Any, List, Dict

        class MyProviderAdapter(LLMAdapter):
            \"\"\"Adapter for custom LLM provider.\"\"\"

            def adapt_messages(
                self,
                messages: List[LLMMessage]
            ) -> List[Dict[str, str]]:
                \"\"\"Convert to provider format.\"\"\"
                return [
                    {"role": msg.role, "content": msg.content}
                    for msg in messages
                ]

            def adapt_response(
                self,
                response: Any
            ) -> LLMResponse:
                \"\"\"Convert from provider format.\"\"\"
                return LLMResponse(
                    content=response["text"],
                    model=response["model_id"],
                    usage={
                        "total_tokens": response["tokens_used"]
                    }
                )

            def adapt_config(
                self,
                config: LLMConfig
            ) -> Dict[str, Any]:
                \"\"\"Convert config to provider format.\"\"\"
                return {
                    "model_name": config.model,
                    "temp": config.temperature,
                    "max_length": config.max_tokens
                }

        # Use adapter in provider
        adapter = MyProviderAdapter()
        provider_messages = adapter.adapt_messages(messages)
        ```

    See Also:
        LLMProvider: Base provider interface
        dataknobs_llm.llm.providers.OpenAIAdapter: Example implementation
    """

    @abstractmethod
    def adapt_messages(
        self,
        messages: List[LLMMessage]
    ) -> Any:
        """Adapt messages to provider format.

        Args:
            messages: Standard LLMMessage list

        Returns:
            Provider-specific message format
        """
        pass

    @abstractmethod
    def adapt_response(
        self,
        response: Any
    ) -> LLMResponse:
        """Adapt provider response to standard format.

        Args:
            response: Provider-specific response object

        Returns:
            Standard LLMResponse
        """
        pass

    @abstractmethod
    def adapt_config(
        self,
        config: LLMConfig
    ) -> Dict[str, Any]:
        """Adapt configuration to provider format.

        Args:
            config: Standard LLMConfig

        Returns:
            Provider-specific config dict
        """
        pass


class LLMMiddleware(Protocol):
    """Protocol for LLM middleware.

    Middleware provides hooks to transform requests before they're sent
    to the LLM and responses before they're returned to the caller.
    Useful for logging, caching, content filtering, rate limiting, etc.

    Middleware can accept configuration as LLMConfig, dataknobs Config, or dict.

    Example:
        ```python
        from dataknobs_llm.llm.base import (
            LLMMiddleware, LLMMessage, LLMResponse, LLMConfig
        )
        from typing import List, Union, Dict, Any
        import logging

        class LoggingMiddleware:
            \"\"\"Logs all LLM requests and responses.\"\"\"

            def __init__(self):
                self.logger = logging.getLogger(__name__)

            async def process_request(
                self,
                messages: List[LLMMessage],
                config: Union[LLMConfig, Config, Dict[str, Any]]
            ) -> List[LLMMessage]:
                \"\"\"Log request before sending.\"\"\"
                self.logger.info(f"Request: {len(messages)} messages")
                for msg in messages:
                    self.logger.debug(f"  {msg.role}: {msg.content[:50]}...")
                return messages

            async def process_response(
                self,
                response: LLMResponse,
                config: Union[LLMConfig, Config, Dict[str, Any]]
            ) -> LLMResponse:
                \"\"\"Log response after receiving.\"\"\"
                self.logger.info(f"Response: {len(response.content)} chars")
                self.logger.info(f"Tokens: {response.usage['total_tokens']}")
                if response.cost_usd:
                    self.logger.info(f"Cost: ${response.cost_usd:.4f}")
                return response


        class ContentFilterMiddleware:
            \"\"\"Filters sensitive content.\"\"\"

            def __init__(self, blocked_words: List[str]):
                self.blocked_words = blocked_words

            async def process_request(
                self,
                messages: List[LLMMessage],
                config: Union[LLMConfig, Config, Dict[str, Any]]
            ) -> List[LLMMessage]:
                \"\"\"Filter input messages.\"\"\"
                filtered = []
                for msg in messages:
                    content = msg.content
                    for word in self.blocked_words:
                        content = content.replace(word, "***")
                    filtered.append(LLMMessage(
                        role=msg.role,
                        content=content,
                        name=msg.name,
                        function_call=msg.function_call,
                        metadata=msg.metadata
                    ))
                return filtered

            async def process_response(
                self,
                response: LLMResponse,
                config: Union[LLMConfig, Config, Dict[str, Any]]
            ) -> LLMResponse:
                \"\"\"Filter output.\"\"\"
                content = response.content
                for word in self.blocked_words:
                    content = content.replace(word, "***")

                from dataclasses import replace
                return replace(response, content=content)


        # Use with ConversationManager
        from dataknobs_llm.conversations import ConversationManager

        manager = await ConversationManager.create(
            llm=llm,
            prompt_builder=builder,
            middleware=[
                LoggingMiddleware(),
                ContentFilterMiddleware(["password", "secret"])
            ]
        )
        ```

    See Also:
        ConversationManager: Uses middleware for request/response processing
    """

    async def process_request(
        self,
        messages: List[LLMMessage],
        config: Union[LLMConfig, Config, Dict[str, Any]]
    ) -> List[LLMMessage]:
        """Process request before sending to LLM.

        Transform, log, validate, or filter messages before they are
        sent to the LLM provider.

        Args:
            messages: Input messages to be sent to LLM
            config: Configuration (LLMConfig, Config, or dict)

        Returns:
            Processed messages (can be modified, added to, or filtered)

        Raises:
            ValueError: If messages are invalid
        """
        ...

    async def process_response(
        self,
        response: LLMResponse,
        config: Union[LLMConfig, Config, Dict[str, Any]]
    ) -> LLMResponse:
        """Process response from LLM.

        Transform, log, validate, or filter the LLM response before
        returning to the caller.

        Args:
            response: LLM response to process
            config: Configuration (LLMConfig, Config, or dict)

        Returns:
            Processed response (can be modified)

        Raises:
            ValueError: If response is invalid
        """
        ...
