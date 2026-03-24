"""Ollama local LLM provider implementation.

This module provides Ollama integration for dataknobs-llm, enabling local LLM
deployment and usage without cloud APIs. Perfect for privacy-sensitive applications,
offline usage, and cost reduction.

Supports:
- All Ollama models (Llama, Mistral, CodeLlama, Phi, etc.)
- Chat with message history
- Streaming responses
- Embeddings for semantic search
- Tool/function calling (Ollama 0.1.17+)
- Vision models with image inputs
- Custom model parameters (temperature, top_p, seed, etc.)
- Docker environment auto-detection
- Multi-modal capabilities

The OllamaProvider automatically detects Docker environments and adjusts
connection URLs accordingly.

Example:
    ```python
    from dataknobs_llm.llm.providers import OllamaProvider
    from dataknobs_llm.llm.base import LLMConfig

    # Basic usage (assumes Ollama running on localhost:11434)
    config = LLMConfig(
        provider="ollama",
        model="llama2",
        temperature=0.7
    )

    async with OllamaProvider(config) as llm:
        # Simple completion
        response = await llm.complete("Explain Python generators")
        print(response.content)

        # Streaming
        async for chunk in llm.stream_complete("Write a poem"):
            print(chunk.delta, end="", flush=True)

    # Custom Ollama URL (remote or Docker)
    remote_config = LLMConfig(
        provider="ollama",
        model="codellama",
        api_base="http://my-ollama-server:11434"
    )

    # Generate embeddings
    embed_config = LLMConfig(
        provider="ollama",
        model="nomic-embed-text"
    )

    llm = OllamaProvider(embed_config)
    await llm.initialize()
    embeddings = await llm.embed([
        "Python is great",
        "JavaScript is versatile"
    ])

    # Vision model with images
    vision_messages = [
        LLMMessage(
            role="user",
            content="What's in this image?",
            metadata={"images": ["base64encodedimage..."]}
        )
    ]

    vision_config = LLMConfig(provider="ollama", model="llava")
    llm = OllamaProvider(vision_config)
    await llm.initialize()
    response = await llm.complete(vision_messages)
    ```

Installation:
    1. Install Ollama from https://ollama.ai
    2. Pull a model: `ollama pull llama2`
    3. Start server: `ollama serve` (usually auto-starts)
    4. Use with dataknobs-llm (no API key needed!)

See Also:
    - Ollama: https://ollama.ai
    - Ollama Models: https://ollama.ai/library
    - Ollama GitHub: https://github.com/ollama/ollama
"""

import json
import logging
import os
import re
import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Union, AsyncIterator

from ..base import (
    LLMAdapter, LLMConfig, LLMMessage, LLMResponse, LLMStreamResponse,
    AsyncLLMProvider, ModelCapability, ToolCall,
    normalize_llm_config
)
from dataknobs_llm.prompts import AsyncPromptBuilder

if TYPE_CHECKING:
    from dataknobs_config.config import Config

logger = logging.getLogger(__name__)


def _find_matching_models(configured_model: str, available_models: list[str]) -> list[str]:
    """Find available models that match the configured model name.

    Matches the exact model name or the base name with any tag suffix.
    For example, ``"llama2"`` matches ``"llama2:latest"`` but NOT
    ``"llama2-uncensored:latest"``.

    Args:
        configured_model: The model name from configuration (e.g., ``"llama2"``).
        available_models: List of model names from the Ollama API.

    Returns:
        List of matching model names (may be empty).
    """
    if configured_model in available_models:
        return [configured_model]
    base_model = configured_model.split(":", maxsplit=1)[0]
    return [
        m for m in available_models
        if m == base_model or m.startswith(base_model + ":")
    ]


# Regex for <think>...</think> blocks emitted by reasoning models.
# DOTALL so '.' matches newlines inside the tag.
_THINK_TAG_RE = re.compile(r"^<think>(.*?)</think>\s*(.*)", re.DOTALL)


class OllamaAdapter(LLMAdapter):
    """Adapter for Ollama API format.

    Converts between dataknobs standard types and Ollama's HTTP API format.
    Handles assistant tool_calls, tool result messages, and vision images.
    """

    def adapt_messages(
        self,
        messages: List[LLMMessage],
        system_prompt: str | None = None,
    ) -> List[Dict[str, Any]]:
        """Convert LLMMessages to Ollama chat format.

        Handles assistant messages with tool_calls, tool result messages,
        and vision messages with images from metadata.

        ``system_prompt`` is accepted for interface compatibility but
        ignored — Ollama passes system content as a normal message.

        Args:
            messages: Standard LLMMessage list.

        Returns:
            List of message dicts in Ollama format.
        """
        ollama_messages = []
        for msg in messages:
            message: Dict[str, Any] = {
                "role": msg.role,
                "content": msg.content or "",
            }

            # Include tool_calls on assistant messages so the model
            # retains structured memory of what it called.
            if msg.tool_calls and msg.role == "assistant":
                message["tool_calls"] = [
                    {
                        "function": {
                            "name": tc.name,
                            "arguments": tc.parameters,
                        },
                    }
                    for tc in msg.tool_calls
                ]

            # Ollama supports images in messages for vision models
            if msg.metadata.get("images"):
                message["images"] = msg.metadata["images"]

            ollama_messages.append(message)
        return ollama_messages

    def adapt_response(self, data: Any) -> LLMResponse:
        """Parse Ollama JSON response into LLMResponse.

        Args:
            data: Parsed JSON dict from Ollama ``/api/chat`` response.

        Returns:
            Standard ``LLMResponse`` with content, tool_calls, and usage.
        """
        message = data.get("message", {})
        content = message.get("content", "")
        raw_tool_calls = message.get("tool_calls", [])

        tool_calls = None
        if raw_tool_calls:
            tool_calls = [
                ToolCall(
                    name=tc.get("function", {}).get("name", ""),
                    parameters=tc.get("function", {}).get("arguments", {}),
                    id=tc.get("id"),
                )
                for tc in raw_tool_calls
            ]

        usage = None
        if "eval_count" in data:
            usage = {
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
                "total_tokens": (
                    data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
                ),
            }

        return LLMResponse(
            content=content,
            model=data.get("model", ""),
            finish_reason=(
                "tool_calls" if tool_calls
                else ("stop" if data.get("done") else "length")
            ),
            usage=usage,
            tool_calls=tool_calls,
            metadata={
                "eval_duration": data.get("eval_duration"),
                "total_duration": data.get("total_duration"),
                "model_info": data.get("model", ""),
            },
        )

    def adapt_config(self, config: LLMConfig) -> Dict[str, Any]:
        """Build Ollama options dict from config.

        Args:
            config: Standard LLMConfig.

        Returns:
            Dictionary of Ollama options.
        """
        gen = config.generation_params()
        options: Dict[str, Any] = {}

        if "temperature" in gen:
            options["temperature"] = float(gen["temperature"])
        if "top_p" in gen:
            options["top_p"] = float(gen["top_p"])
        if "seed" in gen:
            options["seed"] = int(gen["seed"])
        if "max_tokens" in gen:
            options["num_predict"] = int(gen["max_tokens"])
        if "stop_sequences" in gen:
            options["stop"] = list(gen["stop_sequences"])

        return options

    def adapt_tools(self, tools: list[Any]) -> list[Dict[str, Any]]:
        """Convert Tool objects to Ollama tools format.

        Ollama uses an OpenAI-compatible format with ``type: "function"``
        wrapping.

        Args:
            tools: List of Tool objects with ``name``, ``description``,
                and ``schema`` attributes.

        Returns:
            List of Ollama tool definitions.
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


class OllamaProvider(AsyncLLMProvider):
    """Ollama local LLM provider for privacy-first, offline LLM usage.

    Provides async access to locally-hosted Ollama models, enabling
    on-premise LLM deployment without cloud APIs. Perfect for sensitive
    data, air-gapped environments, and cost optimization.

    Features:
        - All Ollama models (Llama 2/3, Mistral, Phi, CodeLlama, etc.)
        - No API key required - fully local
        - Chat with message history
        - Streaming responses for real-time output
        - Embeddings for RAG and semantic search
        - Tool/function calling (Ollama 0.1.17+)
        - Vision models (LLaVA, bakllava)
        - Docker environment auto-detection
        - Custom model parameters (temperature, top_p, seed)
        - Zero-cost inference

    Example:
        ```python
        from dataknobs_llm.llm.providers import OllamaProvider
        from dataknobs_llm.llm.base import LLMConfig, LLMMessage

        # Basic local usage
        config = LLMConfig(
            provider="ollama",
            model="llama2",  # or llama3, mistral, phi, etc.
            temperature=0.7
        )

        async with OllamaProvider(config) as llm:
            # Simple completion
            response = await llm.complete("Explain decorators in Python")
            print(response.content)

            # Multi-turn conversation
            messages = [
                LLMMessage(role="system", content="You are a helpful assistant"),
                LLMMessage(role="user", content="What is recursion?"),
                LLMMessage(role="assistant", content="Recursion is..."),
                LLMMessage(role="user", content="Show me an example")
            ]
            response = await llm.complete(messages)

        # Code generation with CodeLlama
        code_config = LLMConfig(
            provider="ollama",
            model="codellama",
            temperature=0.2,  # Lower for more deterministic code
            max_tokens=500
        )

        llm = OllamaProvider(code_config)
        await llm.initialize()
        response = await llm.complete(
            "Write a Python function to merge two sorted lists"
        )
        print(response.content)

        # Remote Ollama server
        remote_config = LLMConfig(
            provider="ollama",
            model="llama2",
            api_base="http://192.168.1.100:11434"  # Remote server
        )

        # Docker usage (auto-detects)
        # In Docker, automatically uses host.docker.internal
        docker_config = LLMConfig(
            provider="ollama",
            model="mistral"
        )

        # Vision model with image input
        from dataknobs_llm.llm.base import LLMMessage
        import base64

        with open("image.jpg", "rb") as f:
            image_data = base64.b64encode(f.read()).decode()

        vision_config = LLMConfig(
            provider="ollama",
            model="llava"  # or bakllava
        )

        llm = OllamaProvider(vision_config)
        await llm.initialize()

        messages = [
            LLMMessage(
                role="user",
                content="What objects are in this image?",
                metadata={"images": [image_data]}
            )
        ]

        response = await llm.complete(messages)
        print(response.content)

        # Embeddings for RAG
        embed_config = LLMConfig(
            provider="ollama",
            model="nomic-embed-text"  # or mxbai-embed-large
        )

        llm = OllamaProvider(embed_config)
        await llm.initialize()

        # Single embedding
        embedding = await llm.embed("Sample text")
        print(f"Dimensions: {len(embedding)}")

        # Batch embeddings
        texts = [
            "Python programming",
            "Machine learning basics",
            "Web development with Flask"
        ]
        embeddings = await llm.embed(texts)
        print(f"Generated {len(embeddings)} embeddings")

        # Tool use (Ollama 0.1.17+)
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"}
                        },
                        "required": ["location"]
                    }
                }
            }
        ]

        response = await llm.function_call(messages, tools)
        ```

    Args:
        config: LLMConfig, dataknobs Config, or dict with provider settings
        prompt_builder: Optional AsyncPromptBuilder for prompt rendering

    Attributes:
        base_url (str): Ollama API base URL (auto-detects Docker environment)
        _client: HTTP client for Ollama API

    See Also:
        LLMConfig: Configuration options
        AsyncLLMProvider: Base provider interface
        Ollama Documentation: https://ollama.ai
    """

    def __init__(
        self,
        config: Union[LLMConfig, "Config", Dict[str, Any]],
        prompt_builder: AsyncPromptBuilder | None = None
    ):
        # Normalize config first
        llm_config = normalize_llm_config(config)
        super().__init__(llm_config, prompt_builder=prompt_builder)

        self.adapter = OllamaAdapter()

        # Check for Docker environment and adjust URL accordingly
        default_url = 'http://localhost:11434'
        if os.path.exists('/.dockerenv'):
            # Running in Docker, use host.docker.internal
            default_url = 'http://host.docker.internal:11434'

        # Allow environment variable override
        self.base_url = llm_config.api_base or os.environ.get('OLLAMA_BASE_URL', default_url)

    def _build_options(self, config: LLMConfig | None = None) -> Dict[str, Any]:
        """Build options dict for Ollama API calls.

        Delegates to the adapter. Accepts ``None`` to use ``self.config``.
        """
        return self.adapter.adapt_config(config or self.config)

    def _analyze_response(self, response: LLMResponse) -> LLMResponse:
        """Parse ``<think>`` tags and run base-class thinking-only detection.

        Reasoning models (DeepSeek-R1, Qwen3) wrap their chain-of-thought in
        ``<think>...</think>`` tags.  This method extracts the thinking text
        into ``metadata["thinking"]`` and leaves only the visible answer in
        ``content``.  After extraction, the base-class heuristic
        (empty content + high token usage) fires if the model produced *only*
        thinking and no visible answer.
        """
        if response.content:
            match = _THINK_TAG_RE.match(response.content)
            if match:
                thinking_text = match.group(1).strip()
                visible_text = match.group(2).strip()
                if thinking_text:
                    response.metadata["thinking"] = thinking_text
                response = LLMResponse(
                    content=visible_text,
                    model=response.model,
                    finish_reason=response.finish_reason,
                    usage=response.usage,
                    function_call=response.function_call,
                    tool_calls=response.tool_calls,
                    metadata=response.metadata,
                    created_at=response.created_at,
                    cost_usd=response.cost_usd,
                    cumulative_cost_usd=response.cumulative_cost_usd,
                )
        return super()._analyze_response(response)

    def _messages_to_ollama(self, messages: List[LLMMessage]) -> List[Dict[str, Any]]:
        """Convert LLMMessage list to Ollama chat format.

        Delegates to the adapter.
        """
        return self.adapter.adapt_messages(messages)

    async def initialize(self) -> None:
        """Initialize Ollama client."""
        try:
            import aiohttp
            connector = aiohttp.TCPConnector(force_close=True)
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=aiohttp.ClientTimeout(total=self.config.timeout or 30.0),
            )

            # Test connection and verify model availability
            try:
                async with self._session.get(f"{self.base_url}/api/tags") as response:
                    if response.status == 200:
                        data = await response.json()
                        models = [m['name'] for m in data.get('models', [])]
                        if models:
                            # Check if configured model is available
                            matching = _find_matching_models(self.config.model, models)
                            if matching and matching[0] != self.config.model:
                                self.config.model = matching[0]
                                logger.info("Ollama: Using model %s", self.config.model)
                            elif not matching:
                                logger.warning(
                                    "Ollama: Model %s not found. Available: %s",
                                    self.config.model, models,
                                )
                        else:
                            logger.warning("Ollama: No models found. Please pull a model first.")
                    else:
                        logger.warning("Ollama: API returned status %s", response.status)
            except Exception as e:
                logger.warning("Ollama: Could not connect to %s: %s", self.base_url, e)

            self._is_initialized = True
        except ImportError as e:
            raise ImportError("aiohttp package not installed. Install with: pip install aiohttp") from e

    async def _close_client(self) -> None:
        """Close the aiohttp session."""
        if hasattr(self, '_session') and self._session:
            await self._session.close()

    async def validate_model(self) -> bool:
        """Validate model availability."""
        if not self._is_initialized or not hasattr(self, '_session'):
            return False

        try:
            async with self._session.get(f"{self.base_url}/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    models = [m['name'] for m in data.get('models', [])]
                    return bool(_find_matching_models(self.config.model, models))
        except Exception:
            return False
        return False

    def _detect_capabilities(self) -> List[ModelCapability]:
        """Auto-detect Ollama model capabilities."""
        detected = [
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CHAT,
            ModelCapability.STREAMING,
            ModelCapability.EMBEDDINGS,  # Ollama supports embed() for all models
        ]

        model = self.config.model.lower()

        # Models that support function calling
        tool_capable_models = [
            'llama3', 'mistral', 'mixtral', 'qwen',
            'command-r', 'phi3', 'phi4', 'nemotron',
            'firefunction', 'hermes',
        ]
        if any(m in model for m in tool_capable_models):
            detected.append(ModelCapability.FUNCTION_CALLING)

        # JSON mode: tool-capable models + gemma, deepseek support structured output
        json_capable_models = [
            *tool_capable_models,
            'gemma', 'deepseek',
        ]
        if any(m in model for m in json_capable_models):
            detected.append(ModelCapability.JSON_MODE)

        if 'llava' in model:
            detected.append(ModelCapability.VISION)

        if 'codellama' in model or 'codegemma' in model:
            detected.append(ModelCapability.CODE)

        return detected

    async def complete(
        self,
        messages: Union[str, List[LLMMessage]],
        config_overrides: Dict[str, Any] | None = None,
        tools: list[Any] | None = None,
        **kwargs: Any
    ) -> LLMResponse:
        """Generate completion using Ollama chat endpoint.

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

        # Convert to message list
        if isinstance(messages, str):
            messages = [LLMMessage(role='user', content=messages)]

        # Add system prompt if configured
        if runtime_config.system_prompt and (not messages or messages[0].role != 'system'):
            messages = [LLMMessage(role='system', content=runtime_config.system_prompt)] + list(messages)

        # Convert to Ollama format
        ollama_messages = self._messages_to_ollama(messages)

        # Build payload for chat endpoint
        payload = {
            'model': runtime_config.model,
            'messages': ollama_messages,
            'stream': False,
            'options': self._build_options(runtime_config)
        }

        # Add format if JSON mode requested
        if runtime_config.response_format == 'json':
            payload['format'] = 'json'

        # Handle tools if provided
        if tools:
            payload['tools'] = self.adapter.adapt_tools(tools)

        # Forward 'think' parameter for reasoning models (e.g. qwen3, deepseek-r1).
        # When True, the model emits <think>...</think> blocks before the answer.
        think = runtime_config.options.get('think')
        if think is not None:
            payload['think'] = bool(think)

        async with self._session.post(f"{self.base_url}/api/chat", json=payload) as response:
            if response.status != 200:
                error_text = await response.text()

                # Handle tools not supported — raise explicit error
                if response.status == 400 and "does not support tools" in error_text:
                    from ...exceptions import ToolsNotSupportedError
                    model_name = runtime_config.model
                    raise ToolsNotSupportedError(
                        model=model_name,
                        suggestion=(
                            "For tool support, use: llama3.1:8b, qwen3:8b, "
                            "mistral:7b, or command-r:latest"
                        ),
                    )
                else:
                    logger.error("Ollama API error (status %s): %s", response.status, error_text)
                    logger.error("Request payload: %s", json.dumps(payload, indent=2))
                    response.raise_for_status()
            else:
                data = await response.json()

        parsed = self.adapter.adapt_response(data)
        # Override model with runtime config model (adapter uses response model)
        parsed.model = runtime_config.model
        return self._analyze_response(parsed)

    async def stream_complete(
        self,
        messages: Union[str, List[LLMMessage]],
        config_overrides: Dict[str, Any] | None = None,
        tools: list[Any] | None = None,
        **kwargs: Any
    ) -> AsyncIterator[LLMStreamResponse]:
        """Generate streaming completion using Ollama chat endpoint.

        Uses the ``/api/chat`` endpoint with ``stream: true`` so that the
        model's native chat template is applied and tool calls are supported,
        matching the behaviour of :meth:`complete`.

        Args:
            messages: Input messages or prompt
            config_overrides: Optional dict to override config fields (model,
                temperature, max_tokens, top_p, stop_sequences, seed)
            tools: Optional list of Tool objects for function calling.
            **kwargs: Additional provider-specific parameters.
        """
        if not self._is_initialized:
            await self.initialize()

        # Get runtime config (with overrides applied if provided)
        runtime_config = self._get_runtime_config(config_overrides)

        # Convert to message list
        if isinstance(messages, str):
            messages = [LLMMessage(role='user', content=messages)]

        # Add system prompt if configured
        if runtime_config.system_prompt and (not messages or messages[0].role != 'system'):
            messages = [LLMMessage(role='system', content=runtime_config.system_prompt)] + list(messages)

        # Convert to Ollama format
        ollama_messages = self._messages_to_ollama(messages)

        # Build payload for chat endpoint (mirrors complete())
        payload: Dict[str, Any] = {
            'model': runtime_config.model,
            'messages': ollama_messages,
            'stream': True,
            'options': self._build_options(runtime_config)
        }

        # Add format if JSON mode requested
        if runtime_config.response_format == 'json':
            payload['format'] = 'json'

        # Handle tools if provided
        if tools:
            payload['tools'] = self.adapter.adapt_tools(tools)

        # Forward 'think' parameter for reasoning models (mirrors complete())
        think = runtime_config.options.get('think')
        if think is not None:
            payload['think'] = bool(think)

        async with self._session.post(f"{self.base_url}/api/chat", json=payload) as response:
            response.raise_for_status()

            async for line in response.content:
                if line:
                    data = json.loads(line.decode('utf-8'))
                    msg = data.get('message', {})
                    done = data.get('done', False)

                    if done:
                        # Use adapter for final chunk parsing
                        parsed = self.adapter.adapt_response(data)
                        yield LLMStreamResponse(
                            delta=msg.get('content', ''),
                            is_final=True,
                            finish_reason=parsed.finish_reason,
                            usage=parsed.usage,
                            tool_calls=parsed.tool_calls,
                            model=runtime_config.model,
                        )
                    else:
                        yield LLMStreamResponse(
                            delta=msg.get('content', ''),
                            is_final=False,
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

        embeddings = []
        for text in texts:
            payload = {
                'model': self.config.model,
                'prompt': text
            }

            async with self._session.post(f"{self.base_url}/api/embeddings", json=payload) as response:
                response.raise_for_status()
                data = await response.json()
                embeddings.append(data['embedding'])

        return embeddings[0] if single else embeddings

    async def function_call(
        self,
        messages: List[LLMMessage],
        functions: List[Dict[str, Any]],
        **kwargs
    ) -> LLMResponse:
        """Execute function calling with native Ollama tools support.

        For Ollama 0.1.17+, uses native tools API.
        Falls back to prompt-based approach for older versions.
        """
        warnings.warn("function_call() is deprecated, use complete(tools=...) instead", DeprecationWarning, stacklevel=2)
        if not self._is_initialized:
            await self.initialize()

        # Add system prompt if configured
        if self.config.system_prompt and (not messages or messages[0].role != 'system'):
            messages = [LLMMessage(role='system', content=self.config.system_prompt)] + list(messages)

        # Convert to Ollama format
        ollama_messages = self._messages_to_ollama(messages)

        # function_call() receives raw dicts, not Tool objects — convert
        # directly to Ollama format.
        ollama_tools = [
            {
                "type": "function",
                "function": {
                    "name": func.get("name", ""),
                    "description": func.get("description", ""),
                    "parameters": func.get("parameters", {}),
                },
            }
            for func in functions
        ]

        # Build payload with tools
        payload = {
            'model': self.config.model,
            'messages': ollama_messages,
            'tools': ollama_tools,
            'stream': False,
            'options': self._build_options()
        }

        try:
            async with self._session.post(f"{self.base_url}/api/chat", json=payload) as response:
                response.raise_for_status()
                data = await response.json()

            # Extract response and tool calls
            message = data.get('message', {})
            content = message.get('content', '')
            tool_calls = message.get('tool_calls', [])

            # Build response
            llm_response = LLMResponse(
                content=content,
                model=self.config.model,
                finish_reason='tool_calls' if tool_calls else 'stop',
                usage={
                    'prompt_tokens': data.get('prompt_eval_count', 0),
                    'completion_tokens': data.get('eval_count', 0),
                    'total_tokens': data.get('prompt_eval_count', 0) + data.get('eval_count', 0)
                } if 'eval_count' in data else None
            )

            # Add tool call information if present
            if tool_calls:
                # Use first tool call (Ollama can return multiple)
                tool_call = tool_calls[0]
                llm_response.function_call = {
                    'name': tool_call.get('function', {}).get('name', ''),
                    'arguments': tool_call.get('function', {}).get('arguments', {})
                }

            return llm_response

        except Exception as e:
            # Fallback to prompt-based approach if native tools not supported
            import logging
            logging.warning(f"Ollama native tools failed, falling back to prompt-based: {e}")

            function_descriptions = json.dumps(functions, indent=2)

            system_prompt = f"""You have access to these functions:
{function_descriptions}

To call a function, respond with JSON:
{{"function": "name", "arguments": {{...}}}}"""

            messages_with_system = [
                LLMMessage(role='system', content=system_prompt)
            ] + list(messages)

            llm_response = await self.complete(messages_with_system, **kwargs)

            # Try to parse function call
            try:
                func_data = json.loads(llm_response.content)
                if 'function' in func_data:
                    llm_response.function_call = {
                        'name': func_data['function'],
                        'arguments': func_data.get('arguments', {})
                    }
            except json.JSONDecodeError:
                pass

            return llm_response

    def _build_prompt(self, messages: List[LLMMessage]) -> str:
        """Build prompt from messages."""
        prompt = ""
        for msg in messages:
            if msg.role == 'system':
                prompt += f"System: {msg.content}\n\n"
            elif msg.role == 'user':
                prompt += f"User: {msg.content}\n\n"
            elif msg.role == 'assistant':
                prompt += f"Assistant: {msg.content}\n\n"
        return prompt
