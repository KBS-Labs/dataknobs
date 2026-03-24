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

import json
import logging
import os
import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Union, AsyncIterator

from ..base import (
    LLMConfig, LLMMessage, LLMResponse, LLMStreamResponse,
    AsyncLLMProvider, ModelCapability, ToolCall,
    LLMAdapter, normalize_llm_config
)
from dataknobs_llm.prompts import AsyncPromptBuilder

if TYPE_CHECKING:
    from dataknobs_config.config import Config

logger = logging.getLogger(__name__)


class OpenAIAdapter(LLMAdapter):
    """Adapter for OpenAI API format."""

    def adapt_messages(
        self,
        messages: List[LLMMessage],
        system_prompt: str | None = None,
    ) -> List[Dict[str, Any]]:
        """Convert messages to OpenAI format.

        Handles assistant messages with ``tool_calls`` and tool result
        messages (``role="tool"``) so that multi-turn tool calling
        conversations retain full structured history.

        ``system_prompt`` is accepted for interface compatibility but
        ignored — OpenAI passes system content as a normal message.
        """
        adapted = []
        for msg in messages:
            message: Dict[str, Any] = {
                'role': msg.role,
                'content': msg.content,
            }
            if msg.name:
                message['name'] = msg.name
            if msg.function_call:
                message['function_call'] = msg.function_call
            # Include tool_call_id on tool result messages so OpenAI can
            # pair results with the specific tool invocation.
            if msg.role == 'tool':
                if msg.tool_call_id:
                    message['tool_call_id'] = msg.tool_call_id
                elif msg.name:
                    # Fallback for backward compat with messages stored
                    # before tool_call_id was available.
                    logger.warning(
                        "Tool result message for '%s' has no tool_call_id; "
                        "falling back to name. OpenAI may reject this.",
                        msg.name,
                    )
                    message['tool_call_id'] = msg.name
            # Include tool_calls on assistant messages so the model
            # retains structured memory of what it called.
            if msg.tool_calls and msg.role == 'assistant':
                message['tool_calls'] = [
                    {
                        'id': tc.id or '',
                        'type': 'function',
                        'function': {
                            'name': tc.name,
                            'arguments': (
                                tc.parameters
                                if isinstance(tc.parameters, str)
                                else json.dumps(tc.parameters)
                            ),
                        },
                    }
                    for tc in msg.tool_calls
                ]
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
        gen = config.generation_params()
        params: Dict[str, Any] = {
            'model': config.model,
        }
        # Map canonical names to OpenAI names (most are 1:1)
        for key in ('temperature', 'top_p', 'frequency_penalty',
                    'presence_penalty', 'max_tokens', 'seed'):
            if key in gen:
                params[key] = gen[key]
        # OpenAI uses 'stop' instead of 'stop_sequences'
        if 'stop_sequences' in gen:
            params['stop'] = gen['stop_sequences']
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

    def adapt_tools(self, tools: list[Any]) -> list[Dict[str, Any]]:
        """Convert Tool objects to OpenAI tools format.

        Args:
            tools: List of Tool objects with ``name``, ``description``,
                and ``schema`` attributes.

        Returns:
            List of OpenAI tool definitions.
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.schema if hasattr(tool, "schema") else {},
                },
            }
            for tool in tools
        ]


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

    async def _close_client(self) -> None:
        """Close the OpenAI client."""
        if self._client:
            await self._client.close()  # type: ignore[unreachable]

    async def validate_model(self) -> bool:
        """Validate model availability."""
        try:
            # List available models
            models = await self._client.models.list()
            model_ids = [m.id for m in models.data]
            return self.config.model in model_ids
        except Exception:
            return False

    def _detect_capabilities(self) -> List[ModelCapability]:
        """Auto-detect OpenAI model capabilities."""
        model = self.config.model.lower()
        capabilities = [
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CHAT,
            ModelCapability.STREAMING,
        ]

        # GPT and O-series models support function calling and JSON mode
        tool_capable = [
            'gpt-4', 'gpt-3.5', 'gpt-4o', 'gpt-4-turbo', 'o1', 'o3', 'o4',
        ]
        if any(m in model for m in tool_capable):
            capabilities.extend([
                ModelCapability.FUNCTION_CALLING,
                ModelCapability.JSON_MODE,
            ])

        # Vision models
        if 'vision' in model or 'gpt-4o' in model:
            if ModelCapability.VISION not in capabilities:
                capabilities.append(ModelCapability.VISION)

        # Embedding models
        if 'embedding' in model or model.startswith('text-embedding-'):
            capabilities.append(ModelCapability.EMBEDDINGS)

        return capabilities

    async def complete(
        self,
        messages: Union[str, List[LLMMessage]],
        config_overrides: Dict[str, Any] | None = None,
        tools: list[Any] | None = None,
        **kwargs: Any
    ) -> LLMResponse:
        """Generate completion.

        Args:
            messages: Input messages or prompt
            config_overrides: Optional dict to override config fields (model,
                temperature, max_tokens, top_p, stop_sequences, seed)
            tools: Optional list of Tool objects for function calling
            **kwargs: Additional provider-specific parameters
        """
        if not self._is_initialized:
            await self.initialize()

        # Get runtime config (with overrides applied if provided)
        runtime_config = self._get_runtime_config(config_overrides)

        # Convert string to message list
        if isinstance(messages, str):
            messages = [LLMMessage(role='user', content=messages)]

        # Add system prompt if configured
        if runtime_config.system_prompt and messages[0].role != 'system':
            messages.insert(0, LLMMessage(role='system', content=runtime_config.system_prompt))

        # Adapt messages and config
        adapted_messages = self.adapter.adapt_messages(messages)
        params = self.adapter.adapt_config(runtime_config)
        params.update(kwargs)

        # Handle tools if provided
        if tools:
            params["tools"] = self.adapter.adapt_tools(tools)

        # Make API call
        response = await self._client.chat.completions.create(
            messages=adapted_messages,
            **params
        )

        return self._analyze_response(self.adapter.adapt_response(response))

    async def stream_complete(
        self,
        messages: Union[str, List[LLMMessage]],
        config_overrides: Dict[str, Any] | None = None,
        tools: list[Any] | None = None,
        **kwargs: Any
    ) -> AsyncIterator[LLMStreamResponse]:
        """Generate streaming completion.

        Args:
            messages: Input messages or prompt
            config_overrides: Optional dict to override config fields (model,
                temperature, max_tokens, top_p, stop_sequences, seed)
            tools: Optional list of Tool objects for function calling.
            **kwargs: Additional provider-specific parameters
        """
        if not self._is_initialized:
            await self.initialize()

        # Get runtime config (with overrides applied if provided)
        runtime_config = self._get_runtime_config(config_overrides)

        # Convert string to message list
        if isinstance(messages, str):
            messages = [LLMMessage(role='user', content=messages)]

        # Add system prompt if configured
        if runtime_config.system_prompt and messages[0].role != 'system':
            messages.insert(0, LLMMessage(role='system', content=runtime_config.system_prompt))

        # Adapt messages and config
        adapted_messages = self.adapter.adapt_messages(messages)
        params = self.adapter.adapt_config(runtime_config)
        params['stream'] = True
        params.update(kwargs)

        # Handle tools if provided
        if tools:
            params["tools"] = self.adapter.adapt_tools(tools)

        # Stream API call
        stream = await self._client.chat.completions.create(
            messages=adapted_messages,
            **params
        )

        # Accumulate tool call deltas across chunks. OpenAI sends them
        # incrementally via delta.tool_calls[i].index.
        tool_call_accumulators: dict[int, dict[str, Any]] = {}

        async for chunk in stream:
            choice = chunk.choices[0] if chunk.choices else None
            if not choice:
                continue

            delta = choice.delta
            finish_reason = choice.finish_reason

            # Accumulate tool call deltas
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tool_call_accumulators:
                        tool_call_accumulators[idx] = {
                            "id": "",
                            "name": "",
                            "arguments": "",
                        }
                    acc = tool_call_accumulators[idx]
                    if tc_delta.id:
                        acc["id"] += tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            acc["name"] += tc_delta.function.name
                        if tc_delta.function.arguments:
                            acc["arguments"] += tc_delta.function.arguments

            # Yield content chunks
            content = delta.content or ""
            if content or finish_reason is not None:
                # Build tool_calls on final chunk
                accumulated_tool_calls = None
                if finish_reason is not None and tool_call_accumulators:
                    accumulated_tool_calls = [
                        ToolCall(
                            name=acc["name"],
                            parameters=json.loads(acc["arguments"])
                            if acc["arguments"]
                            else {},
                            id=acc["id"] or None,
                        )
                        for _, acc in sorted(tool_call_accumulators.items())
                    ]

                yield LLMStreamResponse(
                    delta=content,
                    is_final=finish_reason is not None,
                    finish_reason=finish_reason,
                    tool_calls=accumulated_tool_calls,
                    model=runtime_config.model if finish_reason is not None else None,
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
        warnings.warn("function_call() is deprecated, use complete(tools=...) instead", DeprecationWarning, stacklevel=2)
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

        return self._analyze_response(self.adapter.adapt_response(response))
