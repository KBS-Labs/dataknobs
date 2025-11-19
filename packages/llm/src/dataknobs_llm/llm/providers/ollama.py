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

import os
import json
from typing import TYPE_CHECKING, Any, Dict, List, Union, AsyncIterator

from ..base import (
    LLMConfig, LLMMessage, LLMResponse, LLMStreamResponse,
    AsyncLLMProvider, ModelCapability,
    normalize_llm_config
)
from dataknobs_llm.prompts import AsyncPromptBuilder

if TYPE_CHECKING:
    from dataknobs_config.config import Config


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

        # Check for Docker environment and adjust URL accordingly
        default_url = 'http://localhost:11434'
        if os.path.exists('/.dockerenv'):
            # Running in Docker, use host.docker.internal
            default_url = 'http://host.docker.internal:11434'

        # Allow environment variable override
        self.base_url = llm_config.api_base or os.environ.get('OLLAMA_BASE_URL', default_url)

    def _build_options(self) -> Dict[str, Any]:
        """Build options dict for Ollama API calls.

        Returns:
            Dictionary of options for the API request.
        """
        options: Dict[str, Any] = {}

        # Only add temperature if it's not the default to avoid issues
        if self.config.temperature != 1.0:
            options['temperature'] = float(self.config.temperature)

        # Only add top_p if explicitly set and different from default
        if self.config.top_p != 1.0:
            options['top_p'] = float(self.config.top_p)

        if self.config.seed is not None:
            options['seed'] = int(self.config.seed)

        if self.config.max_tokens:
            # Ensure it's an integer
            options['num_predict'] = int(self.config.max_tokens)

        if self.config.stop_sequences:
            options['stop'] = list(self.config.stop_sequences)

        return options

    def _messages_to_ollama(self, messages: List[LLMMessage]) -> List[Dict[str, Any]]:
        """Convert LLMMessage list to Ollama chat format.

        Args:
            messages: List of LLM messages

        Returns:
            List of message dicts in Ollama format
        """
        ollama_messages = []
        for msg in messages:
            message = {
                'role': msg.role,
                'content': msg.content
            }
            # Ollama supports images in messages for vision models
            if msg.metadata.get('images'):
                message['images'] = msg.metadata['images']
            ollama_messages.append(message)
        return ollama_messages

    def _adapt_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Adapt tools to Ollama format.

        Ollama uses a similar format to OpenAI for tools.

        Args:
            tools: List of tool definitions

        Returns:
            List of tools in Ollama format
        """
        # Ollama format is similar to OpenAI
        ollama_tools = []
        for tool in tools:
            ollama_tools.append({
                'type': 'function',
                'function': {
                    'name': tool.get('name'),
                    'description': tool.get('description', ''),
                    'parameters': tool.get('parameters', {})
                }
            })
        return ollama_tools

    async def initialize(self) -> None:
        """Initialize Ollama client."""
        try:
            import aiohttp
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config.timeout or 30.0)
            )

            # Test connection and verify model availability
            try:
                async with self._session.get(f"{self.base_url}/api/tags") as response:
                    if response.status == 200:
                        data = await response.json()
                        models = [m['name'] for m in data.get('models', [])]
                        if models:
                            # Check if configured model is available
                            if self.config.model not in models:
                                # Try without tag (e.g., 'llama2' instead of 'llama2:latest')
                                base_model = self.config.model.split(':')[0]
                                matching_models = [m for m in models if m.startswith(base_model)]
                                if matching_models:
                                    # Use first matching model
                                    self.config.model = matching_models[0]
                                    import logging
                                    logging.info(f"Ollama: Using model {self.config.model}")
                                else:
                                    import logging
                                    logging.warning(f"Ollama: Model {self.config.model} not found. Available: {models}")
                        else:
                            import logging
                            logging.warning("Ollama: No models found. Please pull a model first.")
                    else:
                        import logging
                        logging.warning(f"Ollama: API returned status {response.status}")
            except Exception as e:
                import logging
                logging.warning(f"Ollama: Could not connect to {self.base_url}: {e}")

            self._is_initialized = True
        except ImportError as e:
            raise ImportError("aiohttp package not installed. Install with: pip install aiohttp") from e

    async def close(self) -> None:
        """Close Ollama client."""
        if hasattr(self, '_session') and self._session:
            await self._session.close()
        self._is_initialized = False

    async def validate_model(self) -> bool:
        """Validate model availability."""
        if not self._is_initialized or not hasattr(self, '_session'):
            return False

        try:
            async with self._session.get(f"{self.base_url}/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    models = [m['name'] for m in data.get('models', [])]
                    # Check exact match or base model match
                    if self.config.model in models:
                        return True
                    base_model = self.config.model.split(':')[0]
                    return any(m.startswith(base_model) for m in models)
        except Exception:
            return False
        return False

    def get_capabilities(self) -> List[ModelCapability]:
        """Get Ollama model capabilities."""
        # Capabilities depend on the specific model
        capabilities = [
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CHAT,
            ModelCapability.STREAMING
        ]

        # Most recent Ollama models support function calling
        if any(model in self.config.model.lower() for model in ['llama3', 'mistral', 'mixtral', 'qwen']):
            capabilities.append(ModelCapability.FUNCTION_CALLING)

        if 'llava' in self.config.model.lower():
            capabilities.append(ModelCapability.VISION)

        if 'codellama' in self.config.model.lower() or 'codegemma' in self.config.model.lower():
            capabilities.append(ModelCapability.CODE)

        return capabilities

    async def complete(
        self,
        messages: Union[str, List[LLMMessage]],
        **kwargs
    ) -> LLMResponse:
        """Generate completion using Ollama chat endpoint."""
        if not self._is_initialized:
            await self.initialize()

        # Convert to message list
        if isinstance(messages, str):
            messages = [LLMMessage(role='user', content=messages)]

        # Add system prompt if configured
        if self.config.system_prompt and (not messages or messages[0].role != 'system'):
            messages = [LLMMessage(role='system', content=self.config.system_prompt)] + list(messages)

        # Convert to Ollama format
        ollama_messages = self._messages_to_ollama(messages)

        # Build payload for chat endpoint
        payload = {
            'model': self.config.model,
            'messages': ollama_messages,
            'stream': False,
            'options': self._build_options()
        }

        # Add format if JSON mode requested
        if self.config.response_format == 'json':
            payload['format'] = 'json'

        # Handle tools if provided
        tools = kwargs.get('tools')
        if tools:
            # Convert Tool objects to dict format for _adapt_tools
            tool_dicts = []
            for tool in tools:
                tool_dicts.append({
                    'name': tool.name,
                    'description': tool.description,
                    'parameters': tool.schema if hasattr(tool, 'schema') else {}
                })
            ollama_tools = self._adapt_tools(tool_dicts)
            payload['tools'] = ollama_tools

        async with self._session.post(f"{self.base_url}/api/chat", json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                import logging
                logger = logging.getLogger(__name__)

                # Handle tools not supported - retry without tools
                if response.status == 400 and "does not support tools" in error_text:
                    model_name = self.config.model
                    logger.warning(
                        f"Model '{model_name}' does not support tools. "
                        f"Continuing without tool support. "
                        f"For tool support, use: llama3.1:8b, llama3.2:3b, mistral:7b, or qwen2.5:7b"
                    )
                    # Retry without tools
                    payload.pop('tools', None)
                    async with self._session.post(f"{self.base_url}/api/chat", json=payload) as retry_response:
                        if retry_response.status != 200:
                            retry_error = await retry_response.text()
                            logger.error(f"Ollama API error on retry (status {retry_response.status}): {retry_error}")
                            retry_response.raise_for_status()
                        data = await retry_response.json()
                else:
                    logger.error(f"Ollama API error (status {response.status}): {error_text}")
                    logger.error(f"Request payload: {json.dumps(payload, indent=2)}")
                    response.raise_for_status()
            else:
                data = await response.json()

        # Extract response and tool calls
        message = data.get('message', {})
        content = message.get('content', '')
        raw_tool_calls = message.get('tool_calls', [])

        # Convert tool calls to ToolCall objects
        from ..base import ToolCall
        tool_calls = None
        if raw_tool_calls:
            tool_calls = []
            for tc in raw_tool_calls:
                func = tc.get('function', {})
                tool_calls.append(ToolCall(
                    name=func.get('name', ''),
                    parameters=func.get('arguments', {}),
                    id=tc.get('id')
                ))

        return LLMResponse(
            content=content,
            model=self.config.model,
            finish_reason='tool_calls' if tool_calls else ('stop' if data.get('done') else 'length'),
            usage={
                'prompt_tokens': data.get('prompt_eval_count', 0),
                'completion_tokens': data.get('eval_count', 0),
                'total_tokens': data.get('prompt_eval_count', 0) + data.get('eval_count', 0)
            } if 'eval_count' in data else None,
            tool_calls=tool_calls,
            metadata={
                'eval_duration': data.get('eval_duration'),
                'total_duration': data.get('total_duration'),
                'model_info': data.get('model', '')
            }
        )

    async def stream_complete(
        self,
        messages: Union[str, List[LLMMessage]],
        **kwargs
    ) -> AsyncIterator[LLMStreamResponse]:
        """Generate streaming completion."""
        if not self._is_initialized:
            await self.initialize()

        # Convert to Ollama format
        if isinstance(messages, str):
            prompt = messages
        else:
            prompt = self._build_prompt(messages)

        # Stream API call
        payload = {
            'model': self.config.model,
            'prompt': prompt,
            'stream': True,
            'options': self._build_options()
        }

        async with self._session.post(f"{self.base_url}/api/generate", json=payload) as response:
            response.raise_for_status()

            async for line in response.content:
                if line:
                    data = json.loads(line.decode('utf-8'))
                    yield LLMStreamResponse(
                        delta=data.get('response', ''),
                        is_final=data.get('done', False),
                        finish_reason='stop' if data.get('done') else None
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
        if not self._is_initialized:
            await self.initialize()

        # Add system prompt if configured
        if self.config.system_prompt and (not messages or messages[0].role != 'system'):
            messages = [LLMMessage(role='system', content=self.config.system_prompt)] + list(messages)

        # Convert to Ollama format
        ollama_messages = self._messages_to_ollama(messages)

        # Adapt tools to Ollama format
        ollama_tools = self._adapt_tools(functions)

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
