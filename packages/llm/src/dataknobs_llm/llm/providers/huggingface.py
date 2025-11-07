"""HuggingFace Inference API provider implementation."""

import os
from typing import TYPE_CHECKING, Any, Dict, List, Union, AsyncIterator

from ..base import (
    LLMConfig, LLMMessage, LLMResponse, LLMStreamResponse,
    AsyncLLMProvider, ModelCapability,
    normalize_llm_config
)
from dataknobs_llm.prompts import AsyncPromptBuilder

if TYPE_CHECKING:
    from dataknobs_config.config import Config


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
