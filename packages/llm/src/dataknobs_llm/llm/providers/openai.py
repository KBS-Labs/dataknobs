"""OpenAI LLM provider implementation.

This module provides OpenAI API integration for dataknobs-llm, supporting:
- GPT-4, GPT-3.5-turbo, and other OpenAI chat models
- Text embeddings (ada-002, etc.)
- Function calling / tool use
- Streaming responses
- JSON mode for structured outputs
- Vision models (GPT-4V)

The OpenAIProvider uses the official OpenAI Python SDK and supports all
standard OpenAI API parameters.

Example:
    ```python
    from dataknobs_llm.llm.providers import OpenAIProvider
    from dataknobs_llm.llm.base import LLMConfig

    # Create provider
    config = LLMConfig(
        provider="openai",
        model="gpt-4",
        api_key="sk-...",  # or set OPENAI_API_KEY env var
        temperature=0.7,
        max_tokens=500
    )

    async with OpenAIProvider(config) as llm:
        # Simple completion
        response = await llm.complete("What is Python?")
        print(response.content)

        # Streaming
        async for chunk in llm.stream_complete("Tell a story"):
            print(chunk.delta, end="", flush=True)

        # Embeddings
        embedding = await llm.embed("sample text")
        print(f"Dimensions: {len(embedding)}")
    ```

See Also:
    - OpenAI API Documentation: https://platform.openai.com/docs
    - openai Python package: https://github.com/openai/openai-python
"""

import os
from typing import TYPE_CHECKING, Any, Dict, List, Union, AsyncIterator

from ..base import (
    LLMConfig, LLMMessage, LLMResponse, LLMStreamResponse,
    AsyncLLMProvider, ModelCapability,
    LLMAdapter, normalize_llm_config
)
from dataknobs_llm.prompts import AsyncPromptBuilder

if TYPE_CHECKING:
    from dataknobs_config.config import Config


class OpenAIAdapter(LLMAdapter):
    """Adapter for OpenAI API format."""

    def adapt_messages(self, messages: List[LLMMessage]) -> List[Dict[str, Any]]:
        """Convert messages to OpenAI format."""
        adapted = []
        for msg in messages:
            message = {
                'role': msg.role,
                'content': msg.content
            }
            if msg.name:
                message['name'] = msg.name
            if msg.function_call:
                message['function_call'] = msg.function_call
            adapted.append(message)
        return adapted

    def adapt_response(self, response: Any) -> LLMResponse:
        """Convert OpenAI response to standard format."""
        choice = response.choices[0]
        message = choice.message

        return LLMResponse(
            content=message.content or '',
            model=response.model,
            finish_reason=choice.finish_reason,
            usage={
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens
            } if response.usage else None,
            function_call=message.function_call if hasattr(message, 'function_call') else None
        )

    def adapt_config(self, config: LLMConfig) -> Dict[str, Any]:
        """Convert config to OpenAI parameters."""
        params = {
            'model': config.model,
            'temperature': config.temperature,
            'top_p': config.top_p,
            'frequency_penalty': config.frequency_penalty,
            'presence_penalty': config.presence_penalty,
        }

        if config.max_tokens:
            params['max_tokens'] = config.max_tokens
        if config.stop_sequences:
            params['stop'] = config.stop_sequences
        if config.seed:
            params['seed'] = config.seed
        if config.logit_bias:
            params['logit_bias'] = config.logit_bias
        if config.user_id:
            params['user'] = config.user_id
        if config.response_format == 'json':
            params['response_format'] = {'type': 'json_object'}
        if config.functions:
            params['functions'] = config.functions
        if config.function_call:
            params['function_call'] = config.function_call

        return params


class OpenAIProvider(AsyncLLMProvider):
    """OpenAI LLM provider with full API support.

    Provides async access to OpenAI's chat, completion, embedding, and
    function calling APIs. Supports all GPT models including GPT-4, GPT-3.5,
    and specialized models (vision, embeddings).

    Features:
        - Full GPT-4 and GPT-3.5-turbo support
        - Streaming responses for real-time output
        - Function calling for tool use
        - JSON mode for structured outputs
        - Embeddings for semantic search
        - Custom API endpoints (e.g., Azure OpenAI)
        - Automatic retry with rate limiting
        - Cost tracking

    Example:
        ```python
        from dataknobs_llm.llm.providers import OpenAIProvider
        from dataknobs_llm.llm.base import LLMConfig, LLMMessage

        # Basic usage
        config = LLMConfig(
            provider="openai",
            model="gpt-4",
            api_key="sk-...",
            temperature=0.7
        )

        async with OpenAIProvider(config) as llm:
            # Simple question
            response = await llm.complete("Explain async/await")
            print(response.content)

            # Multi-turn conversation
            messages = [
                LLMMessage(role="system", content="You are a coding tutor"),
                LLMMessage(role="user", content="How do I use asyncio?")
            ]
            response = await llm.complete(messages)

        # JSON mode for structured output
        json_config = LLMConfig(
            provider="openai",
            model="gpt-4",
            response_format="json",
            system_prompt="Return JSON only"
        )

        llm = OpenAIProvider(json_config)
        await llm.initialize()
        response = await llm.complete(
            "List 3 Python libraries as JSON: {name, description}"
        )
        import json
        data = json.loads(response.content)

        # With Azure OpenAI
        azure_config = LLMConfig(
            provider="openai",
            model="gpt-4",
            api_base="https://your-resource.openai.azure.com/",
            api_key="azure-key"
        )

        # Function calling
        functions = [{
            "name": "search",
            "description": "Search for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                }
            }
        }]

        response = await llm.function_call(messages, functions)
        if response.function_call:
            print(f"Call: {response.function_call['name']}")
        ```

    Args:
        config: LLMConfig, dataknobs Config, or dict with provider settings
        prompt_builder: Optional AsyncPromptBuilder for prompt rendering

    Attributes:
        adapter (OpenAIAdapter): Format adapter for OpenAI API
        _client: OpenAI AsyncOpenAI client instance

    See Also:
        LLMConfig: Configuration options
        AsyncLLMProvider: Base provider interface
        OpenAIAdapter: Format conversion
    """

    def __init__(
        self,
        config: Union[LLMConfig, "Config", Dict[str, Any]],
        prompt_builder: AsyncPromptBuilder | None = None
    ):
        # Normalize config first
        llm_config = normalize_llm_config(config)
        super().__init__(llm_config, prompt_builder=prompt_builder)
        self.adapter = OpenAIAdapter()

    async def initialize(self) -> None:
        """Initialize OpenAI client."""
        try:
            import openai

            api_key = self.config.api_key or os.environ.get('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OpenAI API key not provided")

            self._client = openai.AsyncOpenAI(
                api_key=api_key,
                base_url=self.config.api_base,
                timeout=self.config.timeout
            )
            self._is_initialized = True
        except ImportError as e:
            raise ImportError("openai package not installed. Install with: pip install openai") from e

    async def close(self) -> None:
        """Close OpenAI client."""
        if self._client:
            await self._client.close()  # type: ignore[unreachable]
        self._is_initialized = False

    async def validate_model(self) -> bool:
        """Validate model availability."""
        try:
            # List available models
            models = await self._client.models.list()
            model_ids = [m.id for m in models.data]
            return self.config.model in model_ids
        except Exception:
            return False

    def get_capabilities(self) -> List[ModelCapability]:
        """Get OpenAI model capabilities."""
        capabilities = [
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CHAT,
            ModelCapability.STREAMING
        ]

        if 'gpt-4' in self.config.model or 'gpt-3.5' in self.config.model:
            capabilities.extend([
                ModelCapability.FUNCTION_CALLING,
                ModelCapability.JSON_MODE
            ])

        if 'vision' in self.config.model:
            capabilities.append(ModelCapability.VISION)

        if 'embedding' in self.config.model:
            capabilities.append(ModelCapability.EMBEDDINGS)

        return capabilities

    async def complete(
        self,
        messages: Union[str, List[LLMMessage]],
        **kwargs
    ) -> LLMResponse:
        """Generate completion."""
        if not self._is_initialized:
            await self.initialize()

        # Convert string to message list
        if isinstance(messages, str):
            messages = [LLMMessage(role='user', content=messages)]

        # Add system prompt if configured
        if self.config.system_prompt and messages[0].role != 'system':
            messages.insert(0, LLMMessage(role='system', content=self.config.system_prompt))

        # Adapt messages and config
        adapted_messages = self.adapter.adapt_messages(messages)
        params = self.adapter.adapt_config(self.config)
        params.update(kwargs)

        # Make API call
        response = await self._client.chat.completions.create(
            messages=adapted_messages,
            **params
        )

        return self.adapter.adapt_response(response)

    async def stream_complete(
        self,
        messages: Union[str, List[LLMMessage]],
        **kwargs
    ) -> AsyncIterator[LLMStreamResponse]:
        """Generate streaming completion."""
        if not self._is_initialized:
            await self.initialize()

        # Convert string to message list
        if isinstance(messages, str):
            messages = [LLMMessage(role='user', content=messages)]

        # Add system prompt if configured
        if self.config.system_prompt and messages[0].role != 'system':
            messages.insert(0, LLMMessage(role='system', content=self.config.system_prompt))

        # Adapt messages and config
        adapted_messages = self.adapter.adapt_messages(messages)
        params = self.adapter.adapt_config(self.config)
        params['stream'] = True
        params.update(kwargs)

        # Stream API call
        stream = await self._client.chat.completions.create(
            messages=adapted_messages,
            **params
        )

        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield LLMStreamResponse(
                    delta=chunk.choices[0].delta.content,
                    is_final=chunk.choices[0].finish_reason is not None,
                    finish_reason=chunk.choices[0].finish_reason
                )

    async def embed(
        self,
        texts: Union[str, List[str]],
        **kwargs
    ) -> Union[List[float], List[List[float]]]:
        """Generate embeddings."""
        if not self._is_initialized:
            await self.initialize()

        if isinstance(texts, str):
            texts = [texts]
            single = True
        else:
            single = False

        response = await self._client.embeddings.create(
            input=texts,
            model=self.config.model or 'text-embedding-ada-002'
        )

        embeddings = [e.embedding for e in response.data]
        return embeddings[0] if single else embeddings

    async def function_call(
        self,
        messages: List[LLMMessage],
        functions: List[Dict[str, Any]],
        **kwargs
    ) -> LLMResponse:
        """Execute function calling."""
        if not self._is_initialized:
            await self.initialize()

        # Add system prompt if configured
        if self.config.system_prompt and messages[0].role != 'system':
            messages.insert(0, LLMMessage(role='system', content=self.config.system_prompt))

        # Adapt messages and config
        adapted_messages = self.adapter.adapt_messages(messages)
        params = self.adapter.adapt_config(self.config)
        params['functions'] = functions
        params['function_call'] = kwargs.get('function_call', 'auto')
        params.update(kwargs)

        # Make API call
        response = await self._client.chat.completions.create(
            messages=adapted_messages,
            **params
        )

        return self.adapter.adapt_response(response)
