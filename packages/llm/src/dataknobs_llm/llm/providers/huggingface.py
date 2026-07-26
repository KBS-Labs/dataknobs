"""HuggingFace Inference API provider implementation."""

import asyncio
import os
import warnings
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Dict, List, Union, AsyncIterator

from ..base import (
    LLMConfig, LLMMessage, LLMResponse, LLMStreamResponse,
    AsyncLLMProvider, ModelCapability,
    normalize_llm_config
)
from ..model_profile import (
    CallableModelMetadataSource,
    ConfigOverrideSource,
    LayeredModelProfileResolver,
    ModelProfile,
)
from ..profile_detection import ProfileDetectionMixin
from ._aiohttp_shared import raise_for_status_with_body
from dataknobs_llm.prompts import AsyncPromptBuilder

if TYPE_CHECKING:
    from dataknobs_config.config import Config

# Seconds to sleep after aiohttp ClientSession.close() so that SSL transport
# callbacks can drain before event loop shutdown.  See dk-29 for full context.
_AIOHTTP_DRAIN_SECS = 0.25

#: Default output-token budget when neither the caller's ``max_tokens`` nor a
#: consumer override supplies one — the historical inline literal, now a named
#: constant routed through the request-shaping choke point.
_HF_DEFAULT_MAX_NEW_TOKENS = 100

#: HuggingFace repo-name substrings marking a chat / instruction-tuned model
#: (unchanged from the pre-binding inline test).
_HF_CHAT_MARKERS: tuple[str, ...] = ("chat", "instruct", "conversational")

#: HuggingFace repo-name substrings marking an embedding model. The pre-binding
#: bare ``'embedding'`` test missed the dominant ``sentence-transformers`` /
#: feature-extraction embedding repos; these markers are the correctness
#: widening (an embedding-only repo resolves EMBEDDINGS **without** CHAT).
_HF_EMBED_MARKERS: tuple[str, ...] = (
    "embedding",
    "sentence-transformers/",
    "feature-extraction",
    "all-minilm",
    "bge-",
    "gte-",
)


def hf_match_key(model_lower: str, keys: Iterable[str]) -> str | None:
    """Exact case-insensitive repo-id match for a per-repo override map.

    The HuggingFace matcher injected into :class:`ConfigOverrideSource` (its
    ``match=`` seam). HuggingFace repo ids are exact strings that share prefixes
    (``meta-llama/Llama-3.1-8B`` is a substring of
    ``meta-llama/Llama-3.1-8B-Instruct``), so the substrate default
    :func:`~..model_profile.match_family_key` would resolve a request for the
    base repo to the ``-Instruct`` override. This matches by exact id only (no
    substring, no family aliasing): a request either names a declared repo or it
    does not. Both arguments arrive already lowercased from the source.
    """
    for key in keys:
        if key == model_lower:
            return key
    return None


def _hf_heuristic(model: str) -> ModelProfile:
    """Repo-name capability heuristic — the **complete** capability set.

    The corrected form of the pre-binding inline ``_detect_capabilities``.
    Because a present ``capabilities`` facet whole-set-overrides lower layers
    under the substrate merge (first-non-``None``-per-facet, not a union), this
    must emit the full intended set. Mapping:

    - ``TEXT_GENERATION`` — always (every HF text model; preserves the historical
      unconditional base capability).
    - ``CHAT`` — repo name contains ``chat`` / ``instruct`` / ``conversational``.
    - ``EMBEDDINGS`` — repo name matches an embedding-family marker
      (:data:`_HF_EMBED_MARKERS`), widening the pre-binding bare ``'embedding'``
      test to the dominant ``sentence-transformers`` / feature-extraction repos.
      An embedding-only repo (no chat marker) resolves EMBEDDINGS disjoint from
      CHAT, matching its real surface.

    Deliberately asserts **no** ``STREAMING`` (HF's ``stream_complete`` is a
    simulated single yield, not real token streaming) and **no**
    ``FUNCTION_CALLING`` (the Inference API rejects tools). ``VISION`` /
    ``JSON_MODE`` / ``CODE`` are not heuristically asserted — a consumer running
    a known VLM / code / JSON-grammar repo declares them via
    ``model_profile_overrides.capabilities``. Contributes only the
    ``capabilities`` facet; every other facet is left ``None`` for a consumer
    override to supply.
    """
    model_lower = model.lower()
    caps: set[ModelCapability] = {ModelCapability.TEXT_GENERATION}
    if any(marker in model_lower for marker in _HF_EMBED_MARKERS):
        caps.add(ModelCapability.EMBEDDINGS)
    if any(marker in model_lower for marker in _HF_CHAT_MARKERS):
        caps.add(ModelCapability.CHAT)
    return ModelProfile(capabilities=frozenset(caps))


#: The stateless heuristic HuggingFace model-metadata source — a module
#: singleton (a pure name-substring rule, no per-instance state). The
#: config-override source is prepended per config in
#: :meth:`HuggingFaceProvider._profile_resolver`. There is deliberately **no**
#: live-API layer (HF has no walker-shaped offered-set — the authoritative live
#: signal is a per-model Hub lookup, a distinct source shape deferred to its own
#: design pass) and **no** bundled-resource / pricing / output-ceiling layer
#: (HF's model space is unbounded and community-driven; per-repo facts come from
#: the consumer override).
_HF_HEURISTIC_SOURCE = CallableModelMetadataSource("hf_heuristic", _hf_heuristic)


class HuggingFaceProvider(ProfileDetectionMixin, AsyncLLMProvider):
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
        """Validate model availability.

        Honors a config-override ``available`` pin first (a private-gateway /
        TGI-endpoint consumer that knows its model is live and wants to skip the
        HTTP probe): when ``model_profile_overrides`` pins ``available``, that
        value is returned without an HTTP call. Otherwise the authoritative
        ``GET {base_url}/{model}`` status check runs — HuggingFace has **no**
        source populating ``available`` (unlike Ollama's live source), so the
        live probe stays the default. Preserves the pre-binding behavior exactly
        when no override is set.
        """
        pinned = self._resolve_profile(self.config).available
        if pinned is not None:
            return pinned
        try:
            url = f"{self.base_url}/{self.config.model}"
            async with self._session.get(url) as response:
                return response.status == 200
        except Exception:
            return False

    def _profile_resolver(
        self, config: LLMConfig
    ) -> LayeredModelProfileResolver:
        """Compose the HuggingFace model-profile resolver for *config*.

        Two sources only — the leanest of the provider bindings. Precedence
        (highest first): config override (per-repo keys matched **exactly** via
        :func:`hf_match_key`, so a per-repo map does not resolve a base repo to a
        prefix-sharing variant's override) → the repo-name capability heuristic
        (:data:`_HF_HEURISTIC_SOURCE`). There is deliberately **no** live-API
        layer (HF has no walker-shaped offered-set; the per-model Hub lookup is a
        distinct source shape deferred to its own design pass) and **no**
        bundled-resource / pricing / output-ceiling layer. Every non-capability
        facet (context_window, rejected_params, param_remaps, pricing, available)
        is ``None`` from the heuristic and lit up only by a consumer's
        ``LLMConfig.model_profile_overrides``, which wins per facet.
        """
        return LayeredModelProfileResolver(
            [
                ConfigOverrideSource(
                    getattr(config, "model_profile_overrides", None),
                    match=hf_match_key,
                ),
                _HF_HEURISTIC_SOURCE,
            ]
        )

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

    def _build_hf_parameters(self, runtime_config: LLMConfig) -> Dict[str, Any]:
        """Build the Inference-API ``parameters`` dict with request shaping.

        The HuggingFace request-shaping choke point, mirroring the other
        providers' ``_build_api_kwargs`` / ``_build_shaped_options``: shapes the
        runtime config through the shared
        :meth:`~..base.LLMProvider._apply_request_constraints` (drops
        family-rejected sampling params, clamps ``max_tokens`` to a ceiling — in
        canonical config space, independent of HF's flat ``parameters`` wire
        shape), builds the ``parameters`` dict, then applies any wire-level
        :meth:`~..base.LLMProvider._apply_param_remaps`. HF's auto-detected rules
        are empty by default (no ceiling, no rejected params, no remaps), so this
        is a **byte-identical no-op** in normal use — the ``max_new_tokens``
        default still lands at :data:`_HF_DEFAULT_MAX_NEW_TOKENS` when the caller
        set none, and at the caller's ``max_tokens`` when set. Wired for the
        consumer-``constraints``-override path, symmetric with the other three
        providers. ``complete`` is the only caller; ``stream_complete`` routes
        through it, and ``embed`` posts no ``parameters`` (nothing to shape).
        """
        constraints = self.get_constraints(runtime_config)
        shaped = self._apply_request_constraints(runtime_config, constraints)
        gen = shaped.generation_params()
        parameters: Dict[str, Any] = {
            'max_new_tokens': gen.get('max_tokens', _HF_DEFAULT_MAX_NEW_TOKENS),
            'return_full_text': False,
        }
        if 'temperature' in gen:
            parameters['temperature'] = gen['temperature']
        if 'top_p' in gen:
            parameters['top_p'] = gen['top_p']
        return self._apply_param_remaps(parameters, constraints.param_remaps)

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
        payload = {
            'inputs': prompt,
            'parameters': self._build_hf_parameters(runtime_config),
        }

        try:
            async with self._session.post(url, json=payload) as response:
                await raise_for_status_with_body(response)
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
                await raise_for_status_with_body(response)
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
