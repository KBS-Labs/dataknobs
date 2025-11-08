"""Echo provider for testing and debugging."""

import hashlib
from typing import TYPE_CHECKING, Any, Dict, List, Union, AsyncIterator

from ..base import (
    LLMConfig, LLMMessage, LLMResponse, LLMStreamResponse,
    AsyncLLMProvider, ModelCapability,
    normalize_llm_config
)
from dataknobs_llm.prompts import AsyncPromptBuilder

if TYPE_CHECKING:
    from dataknobs_config.config import Config


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
    """

    def __init__(
        self,
        config: Union[LLMConfig, "Config", Dict[str, Any]],
        prompt_builder: AsyncPromptBuilder | None = None
    ):
        # Normalize config first
        llm_config = normalize_llm_config(config)
        super().__init__(llm_config, prompt_builder=prompt_builder)

        # Echo-specific configuration from options
        self.echo_prefix = llm_config.options.get('echo_prefix', 'Echo: ')
        self.embedding_dim = llm_config.options.get('embedding_dim', 768)
        self.mock_tokens = llm_config.options.get('mock_tokens', True)
        self.stream_delay = llm_config.options.get('stream_delay', 0.0)  # seconds per char

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
        **kwargs: Any
    ) -> LLMResponse:
        """Echo back the input messages.

        Args:
            messages: Input messages or prompt
            **kwargs: Additional parameters (ignored)

        Returns:
            Echo response
        """
        if not self._is_initialized:
            await self.initialize()

        # Convert to message list
        if isinstance(messages, str):
            messages = [LLMMessage(role='user', content=messages)]

        # Build echo response from last user message
        user_messages = [msg for msg in messages if msg.role == 'user']
        if user_messages:
            content = self.echo_prefix + user_messages[-1].content
        else:
            content = self.echo_prefix + "(no user message)"

        # Add system prompt if configured and in echo
        if self.config.system_prompt and self.config.options.get('echo_system', False):
            content = f"[System: {self.config.system_prompt}]\n{content}"

        # Mock token usage
        prompt_tokens = sum(self._count_tokens(msg.content) for msg in messages)
        completion_tokens = self._count_tokens(content)

        return LLMResponse(
            content=content,
            model=self.config.model or 'echo-model',
            finish_reason='stop',
            usage={
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'total_tokens': prompt_tokens + completion_tokens
            } if self.mock_tokens else None
        )

    async def stream_complete(
        self,
        messages: Union[str, List[LLMMessage]],
        **kwargs: Any
    ) -> AsyncIterator[LLMStreamResponse]:
        """Stream echo response character by character.

        Args:
            messages: Input messages or prompt
            **kwargs: Additional parameters (ignored)

        Yields:
            Streaming response chunks
        """
        if not self._is_initialized:
            await self.initialize()

        # Get full response
        response = await self.complete(messages, **kwargs)

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
