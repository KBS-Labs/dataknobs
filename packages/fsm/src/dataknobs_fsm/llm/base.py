"""Base LLM abstraction components.

This module provides the base abstractions for unified LLM operations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any, Dict, List, Optional, Union, AsyncIterator, Iterator,
    Callable, TypeVar, Protocol
)
import asyncio
from datetime import datetime


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
    name: Optional[str] = None  # For function messages
    function_call: Optional[Dict[str, Any]] = None  # For function calling
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    """Response from LLM."""
    content: str
    model: str
    finish_reason: Optional[str] = None  # 'stop', 'length', 'function_call'
    usage: Optional[Dict[str, int]] = None  # tokens used
    function_call: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class LLMStreamResponse:
    """Streaming response from LLM."""
    delta: str  # Incremental content
    is_final: bool = False
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, int]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMConfig:
    """Configuration for LLM operations."""
    provider: str  # 'openai', 'anthropic', 'ollama', etc.
    model: str  # Model name/identifier
    api_key: Optional[str] = None
    api_base: Optional[str] = None  # Custom API endpoint
    
    # Generation parameters
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop_sequences: Optional[List[str]] = None
    
    # Mode settings
    mode: CompletionMode = CompletionMode.CHAT
    system_prompt: Optional[str] = None
    response_format: Optional[str] = None  # 'text' or 'json'
    
    # Function calling
    functions: Optional[List[Dict[str, Any]]] = None
    function_call: Optional[Union[str, Dict[str, str]]] = None  # 'auto', 'none', or specific function
    
    # Streaming
    stream: bool = False
    stream_callback: Optional[Callable[[LLMStreamResponse], None]] = None
    
    # Rate limiting
    rate_limit: Optional[int] = None  # Requests per minute
    retry_count: int = 3
    retry_delay: float = 1.0
    timeout: float = 60.0
    
    # Advanced settings
    seed: Optional[int] = None  # For reproducibility
    logit_bias: Optional[Dict[str, float]] = None
    user_id: Optional[str] = None
    
    # Provider-specific options
    options: Dict[str, Any] = field(default_factory=dict)


class LLMProvider(ABC):
    """Base LLM provider interface."""
    
    def __init__(self, config: LLMConfig):
        """Initialize provider with configuration."""
        self.config = config
        self._client = None
        self._is_initialized = False
        
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
    """Protocol for LLM middleware."""
    
    async def process_request(
        self,
        messages: List[LLMMessage],
        config: LLMConfig
    ) -> List[LLMMessage]:
        """Process request before sending to LLM."""
        ...
        
    async def process_response(
        self,
        response: LLMResponse,
        config: LLMConfig
    ) -> LLMResponse:
        """Process response from LLM."""
        ...