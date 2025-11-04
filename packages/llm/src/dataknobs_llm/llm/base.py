"""Base LLM abstraction components.

This module provides the base abstractions for unified LLM operations.
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
    """LLM completion modes."""
    CHAT = "chat"  # Chat completion with message history
    TEXT = "text"  # Text completion
    INSTRUCT = "instruct"  # Instruction following
    EMBEDDING = "embedding"  # Generate embeddings
    FUNCTION = "function"  # Function calling


class ModelCapability(Enum):
    """Model capabilities."""
    TEXT_GENERATION = "text_generation"
    CHAT = "chat"
    EMBEDDINGS = "embeddings"
    FUNCTION_CALLING = "function_calling"
    VISION = "vision"
    CODE = "code"
    JSON_MODE = "json_mode"
    STREAMING = "streaming"


@dataclass
class LLMMessage:
    """Represents a message in LLM conversation."""
    role: str  # 'system', 'user', 'assistant', 'function'
    content: str
    name: str | None = None  # For function messages
    function_call: Dict[str, Any] | None = None  # For function calling
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    """Response from LLM."""
    content: str
    model: str
    finish_reason: str | None = None  # 'stop', 'length', 'function_call'
    usage: Dict[str, int] | None = None  # tokens used
    function_call: Dict[str, Any] | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    # Cost tracking (optional enhancement for DynaBot)
    cost_usd: float | None = None  # Estimated cost in USD
    cumulative_cost_usd: float | None = None  # Running total for conversation


@dataclass
class LLMStreamResponse:
    """Streaming response from LLM."""
    delta: str  # Incremental content
    is_final: bool = False
    finish_reason: str | None = None
    usage: Dict[str, int] | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMConfig:
    """Configuration for LLM operations.

    This configuration class works with both direct instantiation and
    dataknobs Config objects. It can be created from dictionaries that
    include optional 'type' and 'factory' attributes used by Config.
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

    def clone(self, **overrides) -> "LLMConfig":
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

        Args:
            messages: Input messages or prompt
            **kwargs: Additional parameters

        Returns:
            LLM response
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
        """Generate streaming completion asynchronously.
        
        Args:
            messages: Input messages or prompt
            **kwargs: Additional parameters
            
        Yields:
            Streaming response chunks
        """
        pass
        
    @abstractmethod
    async def embed(
        self,
        texts: Union[str, List[str]],
        **kwargs
    ) -> Union[List[float], List[List[float]]]:
        """Generate embeddings asynchronously.
        
        Args:
            texts: Input text(s)
            **kwargs: Additional parameters
            
        Returns:
            Embedding vector(s)
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
        
        Args:
            messages: Conversation messages
            functions: Available functions
            **kwargs: Additional parameters
            
        Returns:
            Response with function call
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
    """Base adapter for converting between different LLM formats."""
    
    @abstractmethod
    def adapt_messages(
        self,
        messages: List[LLMMessage]
    ) -> Any:
        """Adapt messages to provider format."""
        pass
        
    @abstractmethod
    def adapt_response(
        self,
        response: Any
    ) -> LLMResponse:
        """Adapt provider response to standard format."""
        pass
        
    @abstractmethod
    def adapt_config(
        self,
        config: LLMConfig
    ) -> Dict[str, Any]:
        """Adapt configuration to provider format."""
        pass


class LLMMiddleware(Protocol):
    """Protocol for LLM middleware.

    Middleware can accept configuration as LLMConfig, dataknobs Config, or dict.
    """

    async def process_request(
        self,
        messages: List[LLMMessage],
        config: Union[LLMConfig, Config, Dict[str, Any]]
    ) -> List[LLMMessage]:
        """Process request before sending to LLM.

        Args:
            messages: Input messages
            config: Configuration (LLMConfig, Config, or dict)

        Returns:
            Processed messages
        """
        ...

    async def process_response(
        self,
        response: LLMResponse,
        config: Union[LLMConfig, Config, Dict[str, Any]]
    ) -> LLMResponse:
        """Process response from LLM.

        Args:
            response: LLM response
            config: Configuration (LLMConfig, Config, or dict)

        Returns:
            Processed response
        """
        ...
