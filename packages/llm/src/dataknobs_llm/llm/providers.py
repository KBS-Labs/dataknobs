"""LLM provider implementations.

This module provides implementations for various LLM providers.
Supports both direct instantiation and dataknobs Config-based factory pattern.
"""

import os
import json
import hashlib
from typing import TYPE_CHECKING, Any, Dict, List, Union, AsyncIterator, Type

from .base import (
    LLMConfig, LLMMessage, LLMResponse, LLMStreamResponse,
    AsyncLLMProvider, SyncLLMProvider, ModelCapability,
    LLMAdapter, normalize_llm_config
)

# Import prompt builder types - clean one-way dependency (llm depends on prompts)
from dataknobs_llm.prompts import AsyncPromptBuilder
if TYPE_CHECKING:
    from dataknobs_config.config import Config


class SyncProviderAdapter:
    """Sync adapter for async LLM providers."""
    
    def __init__(self, async_provider: AsyncLLMProvider):
        """Initialize with async provider.
        
        Args:
            async_provider: The async provider to wrap.
        """
        self.async_provider = async_provider
        
    def initialize(self) -> None:
        """Initialize the provider synchronously."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.async_provider.initialize())
    
    def close(self) -> None:
        """Close the provider synchronously."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.async_provider.close())
    
    def complete(
        self,
        messages: Union[str, List[LLMMessage]],
        **kwargs
    ) -> LLMResponse:
        """Generate completion synchronously."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.async_provider.complete(messages, **kwargs))
    
    def stream(
        self,
        messages: Union[str, List[LLMMessage]],
        **kwargs
    ):
        """Stream completion synchronously."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        async def _stream():
            async for chunk in self.async_provider.stream_complete(messages, **kwargs):
                yield chunk

        # Convert async generator to sync generator
        async_gen = _stream()
        try:
            while True:
                try:
                    yield loop.run_until_complete(async_gen.__anext__())
                except StopAsyncIteration:
                    break
        finally:
            loop.run_until_complete(async_gen.aclose())
    
    def embed(
        self,
        texts: Union[str, List[str]],
        **kwargs
    ) -> Union[List[float], List[List[float]]]:
        """Generate embeddings synchronously."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.async_provider.embed(texts, **kwargs))
    
    def function_call(
        self,
        messages: List[LLMMessage],
        functions: List[Dict[str, Any]],
        **kwargs
    ) -> LLMResponse:
        """Make function call synchronously."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.async_provider.function_call(messages, functions, **kwargs))
    
    def validate_model(self) -> bool:
        """Validate model synchronously."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.async_provider.validate_model())  # type: ignore
    
    def get_capabilities(self) -> List[ModelCapability]:
        """Get capabilities synchronously."""
        return self.async_provider.get_capabilities()
    
    @property
    def is_initialized(self) -> bool:
        """Check if provider is initialized."""
        return self.async_provider.is_initialized


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
    """OpenAI LLM provider."""

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


class AnthropicProvider(AsyncLLMProvider):
    """Anthropic Claude LLM provider.

    Supports latest Anthropic features including:
    - Native tools API (Claude 3+)
    - Vision capabilities (Claude 3+)
    - Streaming responses
    """

    def __init__(
        self,
        config: Union[LLMConfig, "Config", Dict[str, Any]],
        prompt_builder: AsyncPromptBuilder | None = None
    ):
        # Normalize config first
        llm_config = normalize_llm_config(config)
        super().__init__(llm_config, prompt_builder=prompt_builder)
        
    async def initialize(self) -> None:
        """Initialize Anthropic client."""
        try:
            import anthropic
            
            api_key = self.config.api_key or os.environ.get('ANTHROPIC_API_KEY')
            if not api_key:
                raise ValueError("Anthropic API key not provided")
                
            self._client = anthropic.AsyncAnthropic(
                api_key=api_key,
                base_url=self.config.api_base,
                timeout=self.config.timeout
            )
            self._is_initialized = True
        except ImportError as e:
            raise ImportError("anthropic package not installed. Install with: pip install anthropic") from e
            
    async def close(self) -> None:
        """Close Anthropic client."""
        if self._client:
            await self._client.close()  # type: ignore[unreachable]
        self._is_initialized = False
        
    async def validate_model(self) -> bool:
        """Validate model availability."""
        valid_models = [
            'claude-3-opus', 'claude-3-sonnet', 'claude-3-haiku',
            'claude-2.1', 'claude-2.0', 'claude-instant-1.2'
        ]
        return any(m in self.config.model for m in valid_models)
        
    def get_capabilities(self) -> List[ModelCapability]:
        """Get Anthropic model capabilities."""
        capabilities = [
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CHAT,
            ModelCapability.STREAMING,
            ModelCapability.CODE
        ]

        # Claude 3+ models support vision and tools
        if 'claude-3' in self.config.model or 'claude-sonnet' in self.config.model or 'claude-opus' in self.config.model:
            capabilities.extend([
                ModelCapability.VISION,
                ModelCapability.FUNCTION_CALLING
            ])

        return capabilities
        
    async def complete(
        self,
        messages: Union[str, List[LLMMessage]],
        **kwargs
    ) -> LLMResponse:
        """Generate completion."""
        if not self._is_initialized:
            await self.initialize()
            
        # Convert to Anthropic format
        if isinstance(messages, str):
            prompt = messages
        else:
            # Build prompt from messages
            prompt = ""
            for msg in messages:
                if msg.role == 'system':
                    prompt = msg.content + "\n\n" + prompt
                elif msg.role == 'user':
                    prompt += f"\n\nHuman: {msg.content}"
                elif msg.role == 'assistant':
                    prompt += f"\n\nAssistant: {msg.content}"
            prompt += "\n\nAssistant:"
            
        # Make API call
        response = await self._client.messages.create(
            model=self.config.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.config.max_tokens or 1024,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            stop_sequences=self.config.stop_sequences
        )
        
        return LLMResponse(
            content=response.content[0].text,
            model=response.model,
            finish_reason=response.stop_reason,
            usage={
                'prompt_tokens': response.usage.input_tokens,
                'completion_tokens': response.usage.output_tokens,
                'total_tokens': response.usage.input_tokens + response.usage.output_tokens
            } if hasattr(response, 'usage') else None
        )
        
    async def stream_complete(
        self,
        messages: Union[str, List[LLMMessage]],
        **kwargs
    ) -> AsyncIterator[LLMStreamResponse]:
        """Generate streaming completion."""
        if not self._is_initialized:
            await self.initialize()
            
        # Convert to Anthropic format
        if isinstance(messages, str):
            prompt = messages
        else:
            prompt = self._build_prompt(messages)
            
        # Stream API call
        async with self._client.messages.stream(
            model=self.config.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.config.max_tokens or 1024,
            temperature=self.config.temperature
        ) as stream:
            async for chunk in stream:
                if chunk.type == 'content_block_delta':
                    yield LLMStreamResponse(
                        delta=chunk.delta.text,
                        is_final=False
                    )
            
            # Final message
            message = await stream.get_final_message()
            yield LLMStreamResponse(
                delta='',
                is_final=True,
                finish_reason=message.stop_reason
            )
            
    async def embed(
        self,
        texts: Union[str, List[str]],
        **kwargs
    ) -> Union[List[float], List[List[float]]]:
        """Anthropic doesn't provide embeddings."""
        raise NotImplementedError("Anthropic doesn't provide embedding models")
        
    async def function_call(
        self,
        messages: List[LLMMessage],
        functions: List[Dict[str, Any]],
        **kwargs
    ) -> LLMResponse:
        """Execute function calling with native Anthropic tools API (Claude 3+)."""
        if not self._is_initialized:
            await self.initialize()

        # Convert to Anthropic message format
        anthropic_messages = []
        system_content = self.config.system_prompt or ''

        for msg in messages:
            if msg.role == 'system':
                # Anthropic uses system parameter, not system messages
                system_content = msg.content if not system_content else f"{system_content}\n\n{msg.content}"
            else:
                anthropic_messages.append({
                    'role': msg.role,
                    'content': msg.content
                })

        # Convert functions to Anthropic tools format
        tools = []
        for func in functions:
            tool = {
                'name': func.get('name', ''),
                'description': func.get('description', ''),
                'input_schema': func.get('parameters', {
                    'type': 'object',
                    'properties': {},
                    'required': []
                })
            }
            tools.append(tool)

        # Make API call with tools
        try:
            response = await self._client.messages.create(
                model=self.config.model,
                messages=anthropic_messages,
                system=system_content if system_content else None,
                tools=tools,
                max_tokens=self.config.max_tokens or 1024,
                temperature=self.config.temperature,
                top_p=self.config.top_p
            )

            # Extract response content and tool use
            content = ''
            tool_use = None

            for block in response.content:
                if block.type == 'text':
                    content += block.text
                elif block.type == 'tool_use':
                    tool_use = {
                        'name': block.name,
                        'arguments': block.input
                    }

            llm_response = LLMResponse(
                content=content,
                model=response.model,
                finish_reason=response.stop_reason,
                usage={
                    'prompt_tokens': response.usage.input_tokens,
                    'completion_tokens': response.usage.output_tokens,
                    'total_tokens': response.usage.input_tokens + response.usage.output_tokens
                },
                function_call=tool_use
            )

            return llm_response

        except Exception as e:
            # Fallback to prompt-based approach for older models
            import logging
            logging.warning(f"Anthropic native tools failed, falling back to prompt-based: {e}")

            function_descriptions = "\n".join([
                f"- {f['name']}: {f['description']}"
                for f in functions
            ])

            system_prompt = f"""You have access to the following functions:
{function_descriptions}

When you need to call a function, respond with:
FUNCTION_CALL: {{
    "name": "function_name",
    "arguments": {{...}}
}}"""

            messages_with_system = [
                LLMMessage(role='system', content=system_prompt)
            ] + list(messages)

            response = await self.complete(messages_with_system, **kwargs)

            # Parse function call from response
            if 'FUNCTION_CALL:' in response.content:
                try:
                    func_json = response.content.split('FUNCTION_CALL:')[1].strip()
                    function_call = json.loads(func_json)
                    response.function_call = function_call
                except (json.JSONDecodeError, IndexError):
                    pass

            return response
        
    def _build_prompt(self, messages: List[LLMMessage]) -> str:
        """Build Anthropic-style prompt from messages."""
        prompt = ""
        for msg in messages:
            if msg.role == 'system':
                prompt = msg.content + "\n\n" + prompt
            elif msg.role == 'user':
                prompt += f"\n\nHuman: {msg.content}"
            elif msg.role == 'assistant':
                prompt += f"\n\nAssistant: {msg.content}"
        prompt += "\n\nAssistant:"
        return prompt


class OllamaProvider(AsyncLLMProvider):
    """Ollama local LLM provider.

    Supports latest Ollama features including:
    - Native tools/function calling (Ollama 0.1.17+)
    - Chat endpoint with message history
    - Streaming responses
    - Embeddings
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
            except aiohttp.ClientError as e:
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

        async with self._session.post(f"{self.base_url}/api/chat", json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                import logging
                logging.error(f"Ollama API error (status {response.status}): {error_text}")
                logging.error(f"Request payload: {json.dumps(payload, indent=2)}")
                response.raise_for_status()
            data = await response.json()

        # Extract response
        content = data.get('message', {}).get('content', '')

        return LLMResponse(
            content=content,
            model=self.config.model,
            finish_reason='stop' if data.get('done') else 'length',
            usage={
                'prompt_tokens': data.get('prompt_eval_count', 0),
                'completion_tokens': data.get('eval_count', 0),
                'total_tokens': data.get('prompt_eval_count', 0) + data.get('eval_count', 0)
            } if 'eval_count' in data else None,
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


class HuggingFaceProvider(AsyncLLMProvider):
    """HuggingFace Inference API provider."""

    def __init__(
        self,
        config: Union[LLMConfig, "Config", Dict[str, Any]],
        prompt_builder: AsyncPromptBuilder | None = None
    ):
        # Normalize config first
        llm_config = normalize_llm_config(config)
        super().__init__(llm_config, prompt_builder=prompt_builder)
        self.base_url = llm_config.api_base or 'https://api-inference.huggingface.co/models'
        
    async def initialize(self) -> None:
        """Initialize HuggingFace client."""
        try:
            import aiohttp
            
            api_key = self.config.api_key or os.environ.get('HUGGINGFACE_API_KEY')
            if not api_key:
                raise ValueError("HuggingFace API key not provided")
                
            self._session = aiohttp.ClientSession(
                headers={'Authorization': f'Bearer {api_key}'},
                timeout=aiohttp.ClientTimeout(total=self.config.timeout)
            )
            self._is_initialized = True
        except ImportError as e:
            raise ImportError("aiohttp package not installed. Install with: pip install aiohttp") from e
            
    async def close(self) -> None:
        """Close HuggingFace client."""
        if hasattr(self, '_session') and self._session:
            await self._session.close()
        self._is_initialized = False
        
    async def validate_model(self) -> bool:
        """Validate model availability."""
        try:
            url = f"{self.base_url}/{self.config.model}"
            async with self._session.get(url) as response:
                return response.status == 200
        except Exception:
            return False
            
    def get_capabilities(self) -> List[ModelCapability]:
        """Get HuggingFace model capabilities."""
        # Basic capabilities for text generation models
        return [
            ModelCapability.TEXT_GENERATION,
            ModelCapability.EMBEDDINGS if 'embedding' in self.config.model else None  # type: ignore
        ]
        
    async def complete(
        self,
        messages: Union[str, List[LLMMessage]],
        **kwargs
    ) -> LLMResponse:
        """Generate completion."""
        if not self._is_initialized:
            await self.initialize()
            
        # Convert to prompt
        if isinstance(messages, str):
            prompt = messages
        else:
            prompt = self._build_prompt(messages)
            
        # Make API call
        url = f"{self.base_url}/{self.config.model}"
        payload = {
            'inputs': prompt,
            'parameters': {
                'temperature': self.config.temperature,
                'top_p': self.config.top_p,
                'max_new_tokens': self.config.max_tokens or 100,
                'return_full_text': False
            }
        }
        
        async with self._session.post(url, json=payload) as response:
            response.raise_for_status()
            data = await response.json()
            
        # Parse response
        if isinstance(data, list) and len(data) > 0:
            text = data[0].get('generated_text', '')
        else:
            text = str(data)
            
        return LLMResponse(
            content=text,
            model=self.config.model,
            finish_reason='stop'
        )
        
    async def stream_complete(
        self,
        messages: Union[str, List[LLMMessage]],
        **kwargs
    ) -> AsyncIterator[LLMStreamResponse]:
        """HuggingFace Inference API doesn't support streaming."""
        # Simulate streaming by yielding complete response
        response = await self.complete(messages, **kwargs)
        yield LLMStreamResponse(
            delta=response.content,
            is_final=True,
            finish_reason=response.finish_reason
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
            
        url = f"{self.base_url}/{self.config.model}"
        payload = {'inputs': texts}
        
        async with self._session.post(url, json=payload) as response:
            response.raise_for_status()
            embeddings = await response.json()
            
        return embeddings[0] if single else embeddings
        
    async def function_call(
        self,
        messages: List[LLMMessage],
        functions: List[Dict[str, Any]],
        **kwargs
    ) -> LLMResponse:
        """HuggingFace doesn't have native function calling."""
        raise NotImplementedError("Function calling not supported for HuggingFace models")
        
    def _build_prompt(self, messages: List[LLMMessage]) -> str:
        """Build prompt from messages."""
        prompt = ""
        for msg in messages:
            if msg.role == 'system':
                prompt += f"{msg.content}\n\n"
            elif msg.role == 'user':
                prompt += f"User: {msg.content}\n"
            elif msg.role == 'assistant':
                prompt += f"Assistant: {msg.content}\n"
        return prompt


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
        **kwargs
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
        **kwargs
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
        **kwargs
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
        **kwargs
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


class LLMProviderFactory:
    """Factory for creating LLM providers from configuration.

    This factory class integrates with the dataknobs Config system,
    allowing providers to be instantiated via Config.get_factory().

    Example:
        >>> from dataknobs_config import Config
        >>> config = Config({
        ...     "llm": [{
        ...         "name": "gpt4",
        ...         "provider": "openai",
        ...         "model": "gpt-4",
        ...         "factory": "dataknobs_llm.LLMProviderFactory"
        ...     }]
        ... })
        >>> factory = config.get_factory("llm", "gpt4")
        >>> provider = factory.create(config.get("llm", "gpt4"))
    """

    # Registry of provider classes
    _providers: Dict[str, Type[AsyncLLMProvider] | None] = {
        'openai': None,  # Populated lazily
        'anthropic': None,
        'ollama': None,
        'huggingface': None,
        'echo': None,
    }

    def __init__(self, is_async: bool = True):
        """Initialize the factory.

        Args:
            is_async: Whether to create async providers (default: True)
        """
        self.is_async = is_async

        # Lazily populate provider registry
        if LLMProviderFactory._providers['openai'] is None:
            LLMProviderFactory._providers.update({
                'openai': OpenAIProvider,
                'anthropic': AnthropicProvider,
                'ollama': OllamaProvider,
                'huggingface': HuggingFaceProvider,
                'echo': EchoProvider,
            })

    def create(
        self,
        config: Union[LLMConfig, "Config", Dict[str, Any]],
        **kwargs: Any
    ) -> Union[AsyncLLMProvider, SyncLLMProvider]:
        """Create an LLM provider from configuration.

        Args:
            config: Configuration (LLMConfig, Config object, or dict)
            **kwargs: Additional arguments passed to provider constructor

        Returns:
            LLM provider instance

        Raises:
            ValueError: If provider type is unknown
        """
        # Normalize config to LLMConfig
        llm_config = normalize_llm_config(config)

        # Get provider class
        provider_class = self._providers.get(llm_config.provider.lower())
        if not provider_class:
            raise ValueError(
                f"Unknown provider: {llm_config.provider}. "
                f"Available providers: {list(self._providers.keys())}"
            )

        # Create provider instance
        if self.is_async:
            return provider_class(llm_config)
        else:
            # Wrap in sync adapter
            async_provider = provider_class(llm_config)
            return SyncProviderAdapter(async_provider)  # type: ignore

    @classmethod
    def register_provider(
        cls,
        name: str,
        provider_class: Type[AsyncLLMProvider]
    ) -> None:
        """Register a custom provider class.

        Allows extending the factory with custom provider implementations.

        Args:
            name: Provider name (e.g., 'custom')
            provider_class: Provider class (must inherit from AsyncLLMProvider)

        Example:
            >>> class CustomProvider(AsyncLLMProvider):
            ...     pass
            >>> LLMProviderFactory.register_provider('custom', CustomProvider)
        """
        cls._providers[name.lower()] = provider_class

    def __call__(
        self,
        config: Union[LLMConfig, "Config", Dict[str, Any]],
        **kwargs: Any
    ) -> Union[AsyncLLMProvider, SyncLLMProvider]:
        """Allow factory to be called directly.

        Makes the factory callable for convenience.

        Args:
            config: Configuration
            **kwargs: Additional arguments

        Returns:
            LLM provider instance
        """
        return self.create(config, **kwargs)


def create_llm_provider(
    config: Union[LLMConfig, "Config", Dict[str, Any]],
    is_async: bool = True
) -> Union[AsyncLLMProvider, SyncLLMProvider]:
    """Create appropriate LLM provider based on configuration.

    Convenience function that uses LLMProviderFactory internally.
    Now supports LLMConfig, Config objects, and dictionaries.

    Args:
        config: LLM configuration (LLMConfig, Config, or dict)
        is_async: Whether to create async provider

    Returns:
        LLM provider instance

    Example:
        >>> # Direct usage with dict
        >>> provider = create_llm_provider({
        ...     "provider": "openai",
        ...     "model": "gpt-4",
        ...     "api_key": "..."
        ... })

        >>> # With Config object
        >>> from dataknobs_config import Config
        >>> config = Config({"llm": [{"provider": "openai", "model": "gpt-4"}]})
        >>> provider = create_llm_provider(config)
    """
    factory = LLMProviderFactory(is_async=is_async)
    return factory.create(config)
