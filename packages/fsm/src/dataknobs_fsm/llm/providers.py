"""LLM provider implementations.

This module provides implementations for various LLM providers.
"""

import os
import json
from typing import Any, Dict, List, Union, AsyncIterator

from .base import (
    LLMConfig, LLMMessage, LLMResponse, LLMStreamResponse,
    AsyncLLMProvider, SyncLLMProvider, ModelCapability,
    LLMAdapter
)


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
            async for chunk in self.async_provider.stream(messages, **kwargs):
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
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
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
    """Anthropic Claude LLM provider."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        
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
        return [
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CHAT,
            ModelCapability.STREAMING,
            ModelCapability.CODE,
            ModelCapability.VISION if 'claude-3' in self.config.model else None  # type: ignore
        ]
        
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
        """Anthropic doesn't have native function calling."""
        # Implement function calling through prompting
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
        ] + messages
        
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
    """Ollama local LLM provider."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        # Check for Docker environment and adjust URL accordingly
        default_url = 'http://localhost:11434'
        if os.path.exists('/.dockerenv'):
            # Running in Docker, use host.docker.internal
            default_url = 'http://host.docker.internal:11434'

        # Allow environment variable override
        self.base_url = config.api_base or os.environ.get('OLLAMA_BASE_URL', default_url)

    def _build_options(self) -> Dict[str, Any]:
        """Build options dict for Ollama API calls.

        Returns:
            Dictionary of options for the API request.
        """
        options: Dict[str, Any] = {
            'temperature': self.config.temperature,
            'top_p': self.config.top_p
        }

        if self.config.seed is not None:
            options['seed'] = self.config.seed

        if self.config.max_tokens:
            options['num_predict'] = self.config.max_tokens  # type: ignore

        if self.config.stop_sequences:
            options['stop'] = self.config.stop_sequences  # type: ignore

        return options
        
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
            ModelCapability.STREAMING
        ]
        
        if 'llava' in self.config.model.lower():
            capabilities.append(ModelCapability.VISION)
            
        if 'codellama' in self.config.model.lower():
            capabilities.append(ModelCapability.CODE)
            
        return capabilities
        
    async def complete(
        self,
        messages: Union[str, List[LLMMessage]],
        **kwargs
    ) -> LLMResponse:
        """Generate completion."""
        if not self._is_initialized:
            await self.initialize()
            
        # Convert to Ollama format
        if isinstance(messages, str):
            prompt = messages
        else:
            prompt = self._build_prompt(messages)
            
        # Make API call
        payload = {
            'model': self.config.model,
            'prompt': prompt,
            'stream': False,
            'options': self._build_options()
        }

        async with self._session.post(f"{self.base_url}/api/generate", json=payload) as response:
            response.raise_for_status()
            data = await response.json()
            
        return LLMResponse(
            content=data['response'],
            model=self.config.model,
            finish_reason='stop' if data.get('done') else 'length',
            usage={
                'prompt_tokens': data.get('prompt_eval_count', 0),
                'completion_tokens': data.get('eval_count', 0),
                'total_tokens': data.get('prompt_eval_count', 0) + data.get('eval_count', 0)
            } if 'eval_count' in data else None,
            metadata={
                'eval_duration': data.get('eval_duration'),
                'total_duration': data.get('total_duration')
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
        """Ollama doesn't have native function calling."""
        # Similar to Anthropic, implement through prompting
        function_descriptions = json.dumps(functions, indent=2)
        
        system_prompt = f"""You have access to these functions:
{function_descriptions}

To call a function, respond with JSON:
{{"function": "name", "arguments": {{...}}}}"""
        
        messages_with_system = [
            LLMMessage(role='system', content=system_prompt)
        ] + messages
        
        response = await self.complete(messages_with_system, **kwargs)
        
        # Try to parse function call
        try:
            func_data = json.loads(response.content)
            if 'function' in func_data:
                response.function_call = {
                    'name': func_data['function'],
                    'arguments': func_data.get('arguments', {})
                }
        except json.JSONDecodeError:
            pass
            
        return response
        
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
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.base_url = config.api_base or 'https://api-inference.huggingface.co/models'
        
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


def create_llm_provider(
    config: LLMConfig,
    is_async: bool = True
) -> Union[AsyncLLMProvider, SyncLLMProvider]:
    """Create appropriate LLM provider based on configuration.
    
    Args:
        config: LLM configuration
        is_async: Whether to create async provider
        
    Returns:
        LLM provider instance
    """
    provider_map = {
        'openai': OpenAIProvider,
        'anthropic': AnthropicProvider,
        'ollama': OllamaProvider,
        'huggingface': HuggingFaceProvider,
    }
    
    provider_class = provider_map.get(config.provider.lower())
    if not provider_class:
        raise ValueError(f"Unknown provider: {config.provider}")
        
    if not is_async:
        # Wrap async provider in sync adapter
        async_provider = provider_class(config)  # type: ignore
        return SyncProviderAdapter(async_provider)  # type: ignore
        
    return provider_class(config)  # type: ignore
