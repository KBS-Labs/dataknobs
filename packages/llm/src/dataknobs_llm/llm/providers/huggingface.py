"""HuggingFace Inference API provider implementation."""

import asyncio
import os
import warnings
from typing import TYPE_CHECKING, Any, Dict, List, NoReturn, Union, AsyncIterator

from ..base import (
    LLMConfig, LLMMessage, LLMResponse, LLMStreamResponse,
    AsyncLLMProvider, ModelCapability,
    normalize_llm_config
)
from dataknobs_llm.prompts import AsyncPromptBuilder

if TYPE_CHECKING:
    from dataknobs_config.config import Config

# Seconds to sleep after aiohttp ClientSession.close() so that SSL transport
# callbacks can drain before event loop shutdown.  See dk-29 for full context.
_AIOHTTP_DRAIN_SECS = 0.25


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

    async def _close_client(self) -> None:
        """Close the aiohttp session."""
        if hasattr(self, '_session') and self._session:
            await self._session.close()
            await asyncio.sleep(_AIOHTTP_DRAIN_SECS)

    async def validate_model(self) -> bool:
        """Validate model availability."""
        try:
            url = f"{self.base_url}/{self.config.model}"
            async with self._session.get(url) as response:
                return response.status == 200
        except Exception:
            return False

    def _detect_capabilities(self) -> List[ModelCapability]:
        """Auto-detect HuggingFace model capabilities."""
        model = self.config.model.lower()
        capabilities = [ModelCapability.TEXT_GENERATION]

        if 'embedding' in model:
            capabilities.append(ModelCapability.EMBEDDINGS)

        # Chat-capable models
        if any(m in model for m in ['chat', 'instruct', 'conversational']):
            capabilities.append(ModelCapability.CHAT)

        return capabilities

    def _translate_api_error(self, exc: Exception) -> Exception | None:
        """Translate a raw aiohttp transport error into a dataknobs exception.

        Lets consumers catch by a dataknobs exception type instead of coupling
        to ``aiohttp``. The HuggingFace Inference API is spoken over ``aiohttp``
        (no vendor SDK), so the gate is aiohttp's error hierarchy plus
        ``asyncio.TimeoutError``. Extracts the status (from a
        ``ClientResponseError`` raised by ``raise_for_status()``; ``None`` for a
        connection error or timeout) and defers the status→type policy to
        :meth:`~dataknobs_llm.llm.base.LLMProvider._dataknobs_error_for_status`
        (429 → ``RateLimitError``, 400 → ``ValidationError``, everything else →
        ``OperationError``).

        Returns ``None`` for a non-transport exception so the caller re-raises
        it unchanged. The original error is preserved on ``__cause__`` — callers
        raise ``... from exc``.
        """
        import aiohttp
        if isinstance(exc, aiohttp.ClientResponseError):
            retry_after = self._retry_after_from_headers(
                getattr(exc, "headers", None)
            )
            return self._dataknobs_error_for_status(
                exc.status,
                f"HuggingFace API error: {exc}",
                retry_after=retry_after,
            )
        if isinstance(exc, (aiohttp.ClientError, asyncio.TimeoutError)):
            return self._dataknobs_error_for_status(
                None, f"HuggingFace API error: {exc}"
            )
        return None

    def _raise_translated(self, exc: Exception) -> NoReturn:
        """Raise the dataknobs translation of *exc*, else re-raise it unchanged.

        The S4 choke point for the ``aiohttp`` call sites (completion /
        embeddings). A non-transport error is re-raised as-is; a transport
        error is raised as its dataknobs type ``from`` the original.
        """
        translated = self._translate_api_error(exc)
        if translated is None:
            raise exc
        raise translated from exc

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
            tools: Optional list of Tool objects (not supported — raises
                ToolsNotSupportedError if provided)
            **kwargs: Additional provider-specific parameters
        """
        if tools:
            from ...exceptions import ToolsNotSupportedError
            raise ToolsNotSupportedError(
                model=self.config.model,
                suggestion="HuggingFace Inference API does not support tool calling.",
            )

        if not self._is_initialized:
            await self.initialize()

        # Get runtime config (with overrides applied if provided)
        runtime_config = self._get_runtime_config(config_overrides)

        # Convert to prompt
        if isinstance(messages, str):
            prompt = messages
        else:
            prompt = self._build_prompt(messages)

        # Make API call
        url = f"{self.base_url}/{runtime_config.model}"
        gen = runtime_config.generation_params()
        parameters: Dict[str, Any] = {
            'max_new_tokens': gen.get('max_tokens', 100),
            'return_full_text': False,
        }
        if 'temperature' in gen:
            parameters['temperature'] = gen['temperature']
        if 'top_p' in gen:
            parameters['top_p'] = gen['top_p']
        payload = {
            'inputs': prompt,
            'parameters': parameters,
        }

        try:
            async with self._session.post(url, json=payload) as response:
                response.raise_for_status()
                data = await response.json()
        except Exception as exc:
            self._raise_translated(exc)

        # Parse response
        if isinstance(data, list) and len(data) > 0:
            text = data[0].get('generated_text', '')
        else:
            text = str(data)

        # The HF text-generation inference path returns no stop-reason signal
        # (finish_reason is hardcoded 'stop'), so truncation cannot be
        # detected here — LLMResponse.truncated stays False. If a caller needs
        # it, request `details=True` and parse `details.finish_reason`.
        return self._analyze_response(LLMResponse(
            content=text,
            model=runtime_config.model,
            finish_reason='stop'
        ))

    async def stream_complete(
        self,
        messages: Union[str, List[LLMMessage]],
        config_overrides: Dict[str, Any] | None = None,
        tools: list[Any] | None = None,
        **kwargs: Any
    ) -> AsyncIterator[LLMStreamResponse]:
        """HuggingFace Inference API doesn't support streaming.

        Args:
            messages: Input messages or prompt
            config_overrides: Optional dict to override config fields (model,
                temperature, max_tokens, top_p, stop_sequences, seed)
            tools: Optional list of Tool objects (not supported — raises
                ToolsNotSupportedError if provided)
            **kwargs: Additional provider-specific parameters
        """
        # Simulate streaming by yielding complete response (tools forwarded to
        # complete(), which raises ToolsNotSupportedError if tools are passed)
        response = await self.complete(
            messages, config_overrides=config_overrides, tools=tools, **kwargs
        )
        yield LLMStreamResponse(
            delta=response.content,
            is_final=True,
            finish_reason=response.finish_reason,
            model=response.model,
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

        try:
            async with self._session.post(url, json=payload) as response:
                response.raise_for_status()
                embeddings = await response.json()
        except Exception as exc:
            self._raise_translated(exc)

        return embeddings[0] if single else embeddings

    async def function_call(
        self,
        messages: List[LLMMessage],
        functions: List[Dict[str, Any]],
        **kwargs
    ) -> LLMResponse:
        """HuggingFace doesn't have native function calling."""
        warnings.warn("function_call() is deprecated, use complete(tools=...) instead", DeprecationWarning, stacklevel=2)
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
