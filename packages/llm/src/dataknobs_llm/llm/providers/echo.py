"""Echo provider for testing and debugging."""

import hashlib
import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Dict, List, Union, AsyncIterator

from ..base import (
    LLMConfig, LLMMessage, LLMResponse, LLMStreamResponse,
    AsyncLLMProvider, ModelCapability,
    normalize_llm_config
)
from dataknobs_llm.prompts import AsyncPromptBuilder

if TYPE_CHECKING:
    from dataknobs_config.config import Config

# Type alias for response functions
ResponseFunction = Callable[[List[LLMMessage]], Union[str, LLMResponse]]


class EchoProvider(AsyncLLMProvider):
    """Echo provider for testing and debugging.

    This provider echoes back input messages and generates deterministic
    mock embeddings. Perfect for testing without real LLM API calls.

    Features:
    - Echoes back user messages with configurable prefix
    - Generates deterministic embeddings based on content hash
    - Supports streaming (character-by-character echo)
    - Mocks function calling with deterministic responses
    - Zero external dependencies
    - Instant responses

    Scripted Response Features (for testing):
    - Response queue: Provide ordered list of responses
    - Response function: Dynamic response based on input
    - Pattern matching: Map input patterns to responses
    - Call tracking: Record all calls for assertions

    Example:
        ```python
        # Queue mode - responses consumed in order
        provider = EchoProvider(config)
        provider.set_responses(["First response", "Second response"])

        # Function mode - dynamic responses
        provider.set_response_function(
            lambda msgs: f"Got {len(msgs)} messages"
        )

        # Pattern mode - match input to responses
        provider.add_pattern_response(r"hello", "Hi there!")
        provider.add_pattern_response(r"bye", "Goodbye!")

        # Check what was called
        assert provider.call_count == 2
        assert "hello" in provider.get_call(0).messages[0].content
        ```
    """

    def __init__(
        self,
        config: Union[LLMConfig, "Config", Dict[str, Any]],
        prompt_builder: AsyncPromptBuilder | None = None,
        responses: List[Union[str, LLMResponse]] | None = None,
        response_fn: ResponseFunction | None = None,
    ):
        """Initialize EchoProvider.

        Args:
            config: LLM configuration
            prompt_builder: Optional prompt builder
            responses: Optional list of responses to return in order
            response_fn: Optional function to generate responses dynamically
        """
        # Normalize config first
        llm_config = normalize_llm_config(config)
        super().__init__(llm_config, prompt_builder=prompt_builder)

        # Echo-specific configuration from options
        self.echo_prefix = llm_config.options.get('echo_prefix', 'Echo: ')
        self.embedding_dim = llm_config.options.get('embedding_dim', 768)
        self.mock_tokens = llm_config.options.get('mock_tokens', True)
        self.stream_delay = llm_config.options.get('stream_delay', 0.0)  # seconds per char

        # Scripted response state
        self._response_queue: List[Union[str, LLMResponse]] = list(responses or [])
        self._response_fn: ResponseFunction | None = response_fn
        self._pattern_responses: List[tuple[re.Pattern[str], Union[str, LLMResponse]]] = []
        self._call_history: List[Dict[str, Any]] = []
        self._cycle_responses: bool = llm_config.options.get('cycle_responses', False)

    # =========================================================================
    # Scripted Response API
    # =========================================================================

    def set_responses(
        self,
        responses: List[Union[str, LLMResponse]],
        cycle: bool = False,
    ) -> "EchoProvider":
        """Set queue of responses to return in order.

        Args:
            responses: List of response strings or LLMResponse objects
            cycle: If True, cycle through responses instead of exhausting

        Returns:
            Self for chaining
        """
        self._response_queue = list(responses)
        self._cycle_responses = cycle
        return self

    def add_response(self, response: Union[str, LLMResponse]) -> "EchoProvider":
        """Add a single response to the queue.

        Args:
            response: Response string or LLMResponse object

        Returns:
            Self for chaining
        """
        self._response_queue.append(response)
        return self

    def set_response_function(
        self,
        fn: ResponseFunction,
    ) -> "EchoProvider":
        """Set function to generate dynamic responses.

        The function receives the message list and returns either
        a string (converted to LLMResponse) or an LLMResponse object.

        Args:
            fn: Function(messages) -> str | LLMResponse

        Returns:
            Self for chaining
        """
        self._response_fn = fn
        return self

    def add_pattern_response(
        self,
        pattern: str,
        response: Union[str, LLMResponse],
        flags: int = re.IGNORECASE,
    ) -> "EchoProvider":
        """Add pattern-matched response.

        When user message matches pattern, return the specified response.

        Args:
            pattern: Regex pattern to match against user messages
            response: Response to return when pattern matches
            flags: Regex flags (default: case-insensitive)

        Returns:
            Self for chaining
        """
        compiled = re.compile(pattern, flags)
        self._pattern_responses.append((compiled, response))
        return self

    def clear_responses(self) -> "EchoProvider":
        """Clear all scripted responses and reset to echo mode.

        Returns:
            Self for chaining
        """
        self._response_queue.clear()
        self._response_fn = None
        self._pattern_responses.clear()
        return self

    def clear_history(self) -> "EchoProvider":
        """Clear call history.

        Returns:
            Self for chaining
        """
        self._call_history.clear()
        return self

    def reset(self) -> "EchoProvider":
        """Reset all state (responses and history).

        Returns:
            Self for chaining
        """
        self.clear_responses()
        self.clear_history()
        return self

    # =========================================================================
    # Call History API
    # =========================================================================

    @property
    def call_count(self) -> int:
        """Get number of complete() calls made."""
        return len(self._call_history)

    @property
    def calls(self) -> List[Dict[str, Any]]:
        """Get all recorded calls."""
        return self._call_history.copy()

    def get_call(self, index: int) -> Dict[str, Any]:
        """Get a specific call by index.

        Args:
            index: Call index (supports negative indexing)

        Returns:
            Call record with 'messages', 'response', 'kwargs'
        """
        return self._call_history[index]

    def get_last_call(self) -> Dict[str, Any] | None:
        """Get the most recent call, or None if no calls made."""
        return self._call_history[-1] if self._call_history else None

    def get_last_user_message(self) -> str | None:
        """Get the last user message from the most recent call."""
        if not self._call_history:
            return None
        messages = self._call_history[-1].get("messages", [])
        for msg in reversed(messages):
            if isinstance(msg, LLMMessage) and msg.role == "user":
                return msg.content
            elif isinstance(msg, dict) and msg.get("role") == "user":
                return msg.get("content", "")
        return None

    # =========================================================================
    # Response Resolution
    # =========================================================================

    def _resolve_response(
        self,
        messages: List[LLMMessage],
        runtime_config: LLMConfig,
    ) -> LLMResponse | None:
        """Resolve scripted response if available.

        Priority:
        1. Response function (if set)
        2. Pattern matching (first match wins)
        3. Response queue (if not empty)
        4. None (fall back to echo behavior)

        Args:
            messages: Input messages
            runtime_config: Runtime configuration

        Returns:
            LLMResponse if scripted response available, None otherwise
        """
        # Try response function first
        if self._response_fn:
            result = self._response_fn(messages)
            return self._to_response(result, messages, runtime_config)

        # Try pattern matching
        user_content = self._get_user_content(messages)
        for pattern, response in self._pattern_responses:
            if pattern.search(user_content):
                return self._to_response(response, messages, runtime_config)

        # Try response queue
        if self._response_queue:
            if self._cycle_responses:
                # Cycle: rotate queue
                response = self._response_queue[0]
                self._response_queue = self._response_queue[1:] + [response]
            else:
                # Consume: pop from front
                response = self._response_queue.pop(0)
            return self._to_response(response, messages, runtime_config)

        # No scripted response
        return None

    def _to_response(
        self,
        value: Union[str, LLMResponse],
        messages: List[LLMMessage],
        runtime_config: LLMConfig,
    ) -> LLMResponse:
        """Convert value to LLMResponse.

        Args:
            value: String or LLMResponse
            messages: Original messages (for token counting)
            runtime_config: Runtime configuration

        Returns:
            LLMResponse object
        """
        if isinstance(value, LLMResponse):
            return value

        # Convert string to LLMResponse
        content = str(value)
        prompt_tokens = sum(self._count_tokens(msg.content) for msg in messages)
        completion_tokens = self._count_tokens(content)

        return LLMResponse(
            content=content,
            model=runtime_config.model or 'echo-model',
            finish_reason='stop',
            usage={
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'total_tokens': prompt_tokens + completion_tokens
            } if self.mock_tokens else None
        )

    def _get_user_content(self, messages: List[LLMMessage]) -> str:
        """Extract user message content for pattern matching.

        Args:
            messages: Message list

        Returns:
            Concatenated user message content
        """
        user_messages = [msg.content for msg in messages if msg.role == 'user']
        return ' '.join(user_messages)

    def _generate_embedding(self, text: str) -> List[float]:
        """Generate deterministic embedding vector from text.

        Uses SHA-256 hash to create a deterministic vector that:
        - Is always the same for the same input
        - Distributes values across [-1, 1] range
        - Has configurable dimensionality

        Args:
            text: Input text

        Returns:
            Embedding vector of size self.embedding_dim
        """
        # Create hash of the text
        hash_obj = hashlib.sha256(text.encode('utf-8'))
        hash_bytes = hash_obj.digest()

        # Generate embedding by repeatedly hashing
        embedding = []
        current_hash = hash_bytes

        while len(embedding) < self.embedding_dim:
            # Convert hash bytes to floats in [-1, 1]
            for byte in current_hash:
                if len(embedding) >= self.embedding_dim:
                    break
                # Normalize byte (0-255) to [-1, 1]
                embedding.append((byte / 127.5) - 1.0)

            # Rehash for next batch of values
            current_hash = hashlib.sha256(current_hash).digest()

        return embedding[:self.embedding_dim]

    def _count_tokens(self, text: str) -> int:
        """Mock token counting (simple character-based estimate).

        Args:
            text: Input text

        Returns:
            Estimated token count
        """
        # Rough approximation: 1 token ~= 4 characters
        return max(1, len(text) // 4)

    async def initialize(self) -> None:
        """Initialize echo provider (no-op)."""
        self._is_initialized = True

    async def close(self) -> None:
        """Close echo provider (no-op)."""
        self._is_initialized = False

    async def validate_model(self) -> bool:
        """Validate model (always true for echo)."""
        return True

    def get_capabilities(self) -> List[ModelCapability]:
        """Get echo provider capabilities."""
        return [
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CHAT,
            ModelCapability.EMBEDDINGS,
            ModelCapability.FUNCTION_CALLING,
            ModelCapability.STREAMING,
            ModelCapability.JSON_MODE
        ]

    async def complete(
        self,
        messages: Union[str, List[LLMMessage]],
        config_overrides: Dict[str, Any] | None = None,
        **kwargs: Any
    ) -> LLMResponse:
        """Echo back the input messages or return scripted response.

        Response priority:
        1. Response function (if set via set_response_function)
        2. Pattern match (if added via add_pattern_response)
        3. Response queue (if set via set_responses or add_response)
        4. Default echo behavior

        Args:
            messages: Input messages or prompt
            config_overrides: Optional dict to override config fields (model,
                temperature, max_tokens, top_p, stop_sequences, seed)
            **kwargs: Additional parameters (ignored)

        Returns:
            LLMResponse (scripted or echo)
        """
        if not self._is_initialized:
            await self.initialize()

        # Get runtime config (with overrides applied if provided)
        runtime_config = self._get_runtime_config(config_overrides)

        # Convert to message list
        if isinstance(messages, str):
            messages = [LLMMessage(role='user', content=messages)]

        # Try scripted response first
        response = self._resolve_response(messages, runtime_config)

        if response is None:
            # Fall back to default echo behavior
            user_messages = [msg for msg in messages if msg.role == 'user']
            if user_messages:
                content = self.echo_prefix + user_messages[-1].content
            else:
                content = self.echo_prefix + "(no user message)"

            # Add system prompt if configured and in echo
            if runtime_config.system_prompt and runtime_config.options.get('echo_system', False):
                content = f"[System: {runtime_config.system_prompt}]\n{content}"

            # Mock token usage
            prompt_tokens = sum(self._count_tokens(msg.content) for msg in messages)
            completion_tokens = self._count_tokens(content)

            response = LLMResponse(
                content=content,
                model=runtime_config.model or 'echo-model',
                finish_reason='stop',
                usage={
                    'prompt_tokens': prompt_tokens,
                    'completion_tokens': completion_tokens,
                    'total_tokens': prompt_tokens + completion_tokens
                } if self.mock_tokens else None
            )

        # Record the call
        self._call_history.append({
            'messages': messages,
            'response': response,
            'config_overrides': config_overrides,
            'kwargs': kwargs,
        })

        return response

    async def stream_complete(
        self,
        messages: Union[str, List[LLMMessage]],
        config_overrides: Dict[str, Any] | None = None,
        **kwargs: Any
    ) -> AsyncIterator[LLMStreamResponse]:
        """Stream echo response character by character.

        Args:
            messages: Input messages or prompt
            config_overrides: Optional dict to override config fields (model,
                temperature, max_tokens, top_p, stop_sequences, seed)
            **kwargs: Additional parameters (ignored)

        Yields:
            Streaming response chunks
        """
        if not self._is_initialized:
            await self.initialize()

        # Get full response
        response = await self.complete(messages, config_overrides=config_overrides, **kwargs)

        # Stream character by character
        for i, char in enumerate(response.content):
            is_final = (i == len(response.content) - 1)

            yield LLMStreamResponse(
                delta=char,
                is_final=is_final,
                finish_reason='stop' if is_final else None,
                usage=response.usage if is_final else None
            )

            # Optional delay for realistic streaming
            if self.stream_delay > 0:
                import asyncio
                await asyncio.sleep(self.stream_delay)

    async def embed(
        self,
        texts: Union[str, List[str]],
        **kwargs: Any
    ) -> Union[List[float], List[List[float]]]:
        """Generate deterministic mock embeddings.

        Args:
            texts: Input text(s)
            **kwargs: Additional parameters (ignored)

        Returns:
            Embedding vector(s)
        """
        if not self._is_initialized:
            await self.initialize()

        if isinstance(texts, str):
            return self._generate_embedding(texts)
        else:
            return [self._generate_embedding(text) for text in texts]

    async def function_call(
        self,
        messages: List[LLMMessage],
        functions: List[Dict[str, Any]],
        **kwargs: Any
    ) -> LLMResponse:
        """Mock function calling with deterministic response.

        Args:
            messages: Conversation messages
            functions: Available functions
            **kwargs: Additional parameters (ignored)

        Returns:
            Response with mock function call
        """
        if not self._is_initialized:
            await self.initialize()

        # Get last user message
        user_messages = [msg for msg in messages if msg.role == 'user']
        user_content = user_messages[-1].content if user_messages else ""

        # Mock function call: use first function with mock arguments
        if functions:
            first_func = functions[0]
            func_name = first_func.get('name', 'unknown_function')

            # Generate mock arguments based on parameters schema
            params = first_func.get('parameters', {})
            properties = params.get('properties', {})

            mock_args = {}
            for param_name, param_schema in properties.items():
                param_type = param_schema.get('type', 'string')

                # Generate mock value based on type
                if param_type == 'string':
                    mock_args[param_name] = f"mock_{param_name}_from_echo"
                elif param_type == 'number' or param_type == 'integer':
                    # Use hash to generate deterministic number
                    hash_val = int(hashlib.md5(user_content.encode()).hexdigest()[:8], 16)
                    mock_args[param_name] = hash_val % 100
                elif param_type == 'boolean':
                    # Deterministic boolean based on hash
                    hash_val = int(hashlib.md5(user_content.encode()).hexdigest()[:2], 16)
                    mock_args[param_name] = hash_val % 2 == 0
                elif param_type == 'array':
                    mock_args[param_name] = ["mock_item_1", "mock_item_2"]
                elif param_type == 'object':
                    mock_args[param_name] = {"mock_key": "mock_value"}
                else:
                    mock_args[param_name] = None

            # Build response with function call
            content = f"{self.echo_prefix}Calling function '{func_name}'"

            prompt_tokens = sum(self._count_tokens(msg.content) for msg in messages)
            completion_tokens = self._count_tokens(content)

            return LLMResponse(
                content=content,
                model=self.config.model or 'echo-model',
                finish_reason='function_call',
                usage={
                    'prompt_tokens': prompt_tokens,
                    'completion_tokens': completion_tokens,
                    'total_tokens': prompt_tokens + completion_tokens
                } if self.mock_tokens else None,
                function_call={
                    'name': func_name,
                    'arguments': mock_args
                }
            )
        else:
            # No functions provided, just echo
            return await self.complete(messages, **kwargs)
