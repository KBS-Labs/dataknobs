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
    LLMConfig,
    LLMMessage,
    LLMResponse,
    LLMStreamResponse,
    AsyncLLMProvider,
    ModelCapability,
    ToolCall,
    LLMAdapter,
    normalize_llm_config,
)
from ..model_profile import (
    BundledResourceSource,
    CallableModelMetadataSource,
    ConfigOverrideSource,
    LayeredModelProfileResolver,
    ModelProfile,
)
from ..profile_detection import ProfileDetectionMixin
from dataknobs_llm.prompts import AsyncPromptBuilder

if TYPE_CHECKING:
    from dataknobs_config.config import Config

logger = logging.getLogger(__name__)


#: OpenAI model-id substrings that carry tool-calling + JSON mode. Matched as
#: lowercased substrings. Every modern GPT chat model supports tools + JSON mode,
#: so the bare ``gpt`` family is included (embedding models are branched out
#: before this check); the reasoning ``o``-series is listed explicitly. Consumed
#: only by the last-resort heuristic (:func:`_openai_heuristic`) — a model listed
#: in the bundled resource resolves its capabilities from there.
_TOOL_CAPABLE_FAMILIES: tuple[str, ...] = (
    "gpt",
    "o1",
    "o3",
    "o4",
)

#: OpenAI model-id substrings that carry vision (multimodal input).
_VISION_CAPABLE_FAMILIES: tuple[str, ...] = (
    "gpt-4o",
    "gpt-4.1",
    "gpt-5",
    "o1",
    "o3",
    "o4",
    "vision",
)


def _openai_heuristic(model: str) -> ModelProfile:
    """Last-resort capability source: family-substring rules for unlisted models.

    The corrected, demoted form of the old inline ``_detect_capabilities`` logic.
    It contributes only the ``capabilities`` facet (ceilings / pricing /
    ``rejected_params`` / ``param_remaps`` come from the bundled resource for known
    models). Lowest precedence in the resolver, so a model present in
    ``openai_models.yaml`` resolves its capabilities from that resource; this only
    classifies an *unlisted* (e.g. brand-new) family by name.
    """
    model_lower = model.lower()
    capabilities = {
        ModelCapability.TEXT_GENERATION,
        ModelCapability.CHAT,
        ModelCapability.STREAMING,
    }
    # Embedding models are a disjoint family (no chat/tool capabilities).
    if "embedding" in model_lower or model_lower.startswith("text-embedding-"):
        capabilities.add(ModelCapability.EMBEDDINGS)
        # The 3-series takes a `dimensions` parameter; ada-002 rejects it.
        # Named here as well as in the bundled resource so a 3-series model
        # the table does not yet list still resolves the right answer.
        if model_lower.startswith("text-embedding-3-"):
            capabilities.add(ModelCapability.EMBEDDING_DIMENSIONS)
        return ModelProfile(capabilities=frozenset(capabilities))
    capabilities.add(ModelCapability.CODE)
    if any(m in model_lower for m in _TOOL_CAPABLE_FAMILIES):
        capabilities |= {
            ModelCapability.FUNCTION_CALLING,
            ModelCapability.JSON_MODE,
        }
    if any(m in model_lower for m in _VISION_CAPABLE_FAMILIES):
        capabilities.add(ModelCapability.VISION)
    return ModelProfile(capabilities=frozenset(capabilities))


#: The stateless lower-precedence OpenAI model-metadata sources — module
#: singletons. OpenAI serves no ceilings / capabilities / pricing on its live
#: Models API, so (unlike Anthropic) there is **no** ``LiveApiSource``: the bundled
#: resource is the primary declarative source, the heuristic a last resort, and a
#: consumer's ``LLMConfig.model_profile_overrides`` (prepended per config in
#: :meth:`OpenAIProvider._profile_resolver`) wins over both.
_OPENAI_RESOURCE_SOURCE = BundledResourceSource.from_resource(
    "dataknobs_llm.llm.providers", "data/openai_models.yaml"
)
_OPENAI_HEURISTIC_SOURCE = CallableModelMetadataSource("heuristic", _openai_heuristic)


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

        Args:
            messages: Standard LLMMessage list.
            system_prompt: Accepted for interface compatibility but
                ignored — OpenAI passes system content as a normal message.

        Returns:
            List of message dicts in OpenAI format.
        """
        adapted = []
        for msg in messages:
            message: Dict[str, Any] = {
                "role": msg.role,
                "content": msg.content,
            }
            if msg.name:
                message["name"] = msg.name
            if msg.function_call:
                message["function_call"] = msg.function_call
            # Include tool_call_id on tool result messages so OpenAI can
            # pair results with the specific tool invocation.
            if msg.role == "tool":
                if msg.tool_call_id:
                    message["tool_call_id"] = msg.tool_call_id
                elif msg.name:
                    # Fallback for backward compat with messages stored
                    # before tool_call_id was available.
                    logger.warning(
                        "Tool result message for '%s' has no tool_call_id; "
                        "falling back to name. OpenAI may reject this.",
                        msg.name,
                    )
                    message["tool_call_id"] = msg.name
                else:
                    logger.warning(
                        "Tool result message has no tool_call_id or name; "
                        "using 'unknown'. OpenAI will likely reject this.",
                    )
                    message["tool_call_id"] = "unknown"
            # Include tool_calls on assistant messages so the model
            # retains structured memory of what it called.
            if msg.tool_calls and msg.role == "assistant":
                message["tool_calls"] = [
                    {
                        "id": tc.id or "",
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": (
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
            content=message.content or "",
            model=response.model,
            finish_reason=choice.finish_reason,
            # OpenAI signals a token-budget cut-off with finish_reason
            # 'length' — the same silent-truncation hazard as Anthropic's
            # 'max_tokens' (see LLMResponse.truncated).
            truncated=choice.finish_reason == "length",
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
            if response.usage
            else None,
            function_call=message.function_call if hasattr(message, "function_call") else None,
        )

    def adapt_config(self, config: LLMConfig) -> Dict[str, Any]:
        """Convert config to OpenAI parameters."""
        gen = config.generation_params()
        params: Dict[str, Any] = {
            "model": config.model,
        }
        # Map canonical names to OpenAI names (most are 1:1)
        for key in (
            "temperature",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "max_tokens",
            "seed",
        ):
            if key in gen:
                params[key] = gen[key]
        # OpenAI uses 'stop' instead of 'stop_sequences'
        if "stop_sequences" in gen:
            params["stop"] = gen["stop_sequences"]
        if config.logit_bias:
            params["logit_bias"] = config.logit_bias
        if config.user_id:
            params["user"] = config.user_id
        if config.response_format == "json":
            params["response_format"] = {"type": "json_object"}
        if config.functions:
            params["functions"] = config.functions
        if config.function_call:
            params["function_call"] = config.function_call

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


class OpenAIProvider(ProfileDetectionMixin, AsyncLLMProvider):
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
        prompt_builder: AsyncPromptBuilder | None = None,
    ):
        # Normalize config first
        llm_config = normalize_llm_config(config)
        super().__init__(llm_config, prompt_builder=prompt_builder)
        self.adapter = OpenAIAdapter()

    # Same finding as ``AsyncLLMProvider.initialize`` one level up, and the
    # same answer: ``LLMProvider`` declares the pair sync, so the whole async
    # subtree contradicts its own base. Resolving it moves the pair down into
    # ``SyncLLMProvider`` --- a public-ABC contract change needing consumer
    # verification, argued and deferred where the base declares it. Suppressed
    # here for that decision, not because this override is wrong.
    async def initialize(self) -> None:  # type: ignore[override]
        """Initialize OpenAI client."""
        try:
            import openai

            api_key = self.config.api_key or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key not provided")

            self._client = openai.AsyncOpenAI(
                api_key=api_key, base_url=self.config.api_base, timeout=self.config.timeout
            )
            self._is_initialized = True
        except ImportError as e:
            raise ImportError(
                "openai package not installed. Install with: pip install openai"
            ) from e

    async def _close_client(self) -> None:
        """Close the OpenAI client."""
        if self._client:
            await self._client.close()  # type: ignore[unreachable]

    async def _probe_model_available(self) -> bool:
        """Authoritative liveness probe — membership in the Models API list.

        OpenAI's Models API is the live availability signal; no OpenAI profile
        source sets the ``available`` facet (the heuristic emits only capabilities;
        the bundled resource carries no ``available``). The inherited
        :meth:`~..profile_detection.ProfileDetectionMixin.validate_model` honors a
        ``model_profile_overrides.available`` pin before reaching here (aligning
        OpenAI with HuggingFace / Ollama / Bedrock); with no pin this preserves the
        pre-binding behavior exactly (always list, check membership).
        """
        try:
            models = await self._client.models.list()
            model_ids = [m.id for m in models.data]
            return self.config.model in model_ids
        except Exception:
            return False

    def _profile_resolver(self, config: LLMConfig) -> LayeredModelProfileResolver:
        """Compose the OpenAI model-profile resolver for *config*.

        Precedence (highest first): config override → bundled resource →
        heuristic. There is no live source — OpenAI's Models API serves only
        model ids, so ceilings / capabilities / pricing / request-shape rules are
        maintained-fallback (the resource), a corrected family heuristic backs
        unlisted models, and ``LLMConfig.model_profile_overrides`` wins per facet.
        """
        return LayeredModelProfileResolver(
            [
                ConfigOverrideSource(getattr(config, "model_profile_overrides", None)),
                _OPENAI_RESOURCE_SOURCE,
                _OPENAI_HEURISTIC_SOURCE,
            ]
        )

    def _build_api_kwargs(
        self, config: LLMConfig, extra: Dict[str, Any] | None = None
    ) -> Dict[str, Any]:
        """Build OpenAI API params with the family's request-shape rules applied.

        Single choke point shared by ``complete`` / ``stream_complete`` /
        ``function_call``. Delegates the request-shaping front-half to the base
        :meth:`~..base.LLMProvider._shape_request_params` (resolves constraints
        once, folds shaped per-call kwargs into the config, drops family-rejected
        sampling params, clamps ``max_tokens`` to the ceiling — all in canonical
        config space), then adapts the shaped config to OpenAI wire params, merges
        the wire-only kwarg remainder, and applies any wire-level
        :meth:`~..base.LLMProvider._apply_param_remaps` (e.g. the reasoning-family
        ``max_tokens`` → ``max_completion_tokens`` rename).

        The vendor-specific tail: ``wire_extra`` is merged **before** the remap,
        so a wire-only ``response_format`` dict like ``{"type": "json_object"}``
        (richer than the narrow ``str`` config field) or a genuine wire-only param
        (``user``) rides through untouched, while the fold-before-shape in the base
        closes the double-key footgun — a raw ``max_tokens`` kwarg appended after
        the remap would collide with the already-renamed ``max_completion_tokens``
        (an OpenAI 400), and a raw ``temperature`` would bypass the
        reasoning-family drop. An unknown model resolves an all-permissive profile,
        so this is a pass-through for it (the historical behavior).
        """
        shaped_config, wire_extra, constraints = self._shape_request_params(config, extra)
        wire = self.adapter.adapt_config(shaped_config)
        wire.update(wire_extra)
        return self._apply_param_remaps(wire, constraints.param_remaps)

    def _translate_api_error(self, exc: Exception) -> Exception | None:
        """Translate a raw OpenAI SDK error into a dataknobs exception.

        Lets consumers catch by a dataknobs exception type instead of coupling
        to the ``openai`` SDK's classes. Does the SDK-specific gate (is this an
        ``openai.APIError``?) and extraction (status, ``retry-after``), then
        defers the status→type policy to
        :meth:`~dataknobs_llm.llm.base.LLMProvider._dataknobs_error_for_status`:

        - 429 → :class:`~dataknobs_common.exceptions.RateLimitError`
          (with ``retry_after`` when the header is present),
        - a context-window-overflow 400 (identified by the ``code`` or message)
          → :class:`~dataknobs_llm.exceptions.ContextLengthExceededError`,
        - any other 400 → :class:`~dataknobs_common.exceptions.ValidationError`,
        - 401/403 and any other OpenAI API error (other status, connection,
          timeout — which carry no ``status_code``) →
          :class:`~dataknobs_common.exceptions.OperationError`.

        Returns ``None`` for a non-OpenAI exception so the caller re-raises it
        unchanged (a bug in our own code is never masked as an API error). The
        original SDK error is preserved on ``__cause__`` — callers raise
        ``... from exc``.
        """
        try:
            import openai
        except ImportError:  # pragma: no cover - openai is installed post-init
            return None
        if not isinstance(exc, openai.APIError):
            return None
        status = getattr(exc, "status_code", None)
        response = getattr(exc, "response", None)
        retry_after = self._retry_after_from_headers(getattr(response, "headers", None))
        return self._dataknobs_error_for_status(
            status,
            str(exc),
            retry_after=retry_after,
            code=getattr(exc, "code", None),
        )

    async def complete(
        self,
        messages: Union[str, List[LLMMessage]],
        config_overrides: Dict[str, Any] | None = None,
        tools: list[Any] | None = None,
        **kwargs: Any,
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
            messages = [LLMMessage(role="user", content=messages)]

        # Add system prompt if configured
        if runtime_config.system_prompt and messages[0].role != "system":
            messages.insert(0, LLMMessage(role="system", content=runtime_config.system_prompt))

        # Adapt messages and config (drops rejected params, clamps max_tokens,
        # applies the family's wire-param remaps — no-op for permissive models).
        adapted_messages = self.adapter.adapt_messages(messages)
        params = self._build_api_kwargs(runtime_config, kwargs)

        # Handle tools if provided
        if tools:
            params["tools"] = self.adapter.adapt_tools(tools)

        # Make API call
        response = await self._call_api(
            lambda: self._client.chat.completions.create(messages=adapted_messages, **params)
        )

        return self._analyze_response(self.adapter.adapt_response(response))

    async def stream_complete(
        self,
        messages: Union[str, List[LLMMessage]],
        config_overrides: Dict[str, Any] | None = None,
        tools: list[Any] | None = None,
        **kwargs: Any,
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
            messages = [LLMMessage(role="user", content=messages)]

        # Add system prompt if configured
        if runtime_config.system_prompt and messages[0].role != "system":
            messages.insert(0, LLMMessage(role="system", content=runtime_config.system_prompt))

        # Adapt messages and config (drops rejected params, clamps max_tokens,
        # applies the family's wire-param remaps — no-op for permissive models).
        adapted_messages = self.adapter.adapt_messages(messages)
        params = self._build_api_kwargs(runtime_config, kwargs)
        params["stream"] = True

        # Handle tools if provided
        if tools:
            params["tools"] = self.adapter.adapt_tools(tools)

        # Stream API call
        stream = await self._call_api(
            lambda: self._client.chat.completions.create(messages=adapted_messages, **params)
        )

        # Accumulate tool call deltas across chunks. OpenAI sends them
        # incrementally via delta.tool_calls[i].index.
        tool_call_accumulators: dict[int, dict[str, Any]] = {}

        # Iterate through the translating wrapper so a vendor error surfacing
        # mid-stream (rate limit, connection drop) is translated too — not
        # just the create() above.
        async for chunk in self._iter_translated(stream):
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
                            parameters=json.loads(acc["arguments"]) if acc["arguments"] else {},
                            id=acc["id"] or None,
                        )
                        for _, acc in sorted(tool_call_accumulators.items())
                    ]

                chunk_resp = LLMStreamResponse(
                    delta=content,
                    is_final=finish_reason is not None,
                    finish_reason=finish_reason,
                    truncated=finish_reason == "length",
                    tool_calls=accumulated_tool_calls,
                    model=runtime_config.model if finish_reason is not None else None,
                )
                if chunk_resp.is_final:
                    self._warn_if_truncated(chunk_resp)
                yield chunk_resp

    async def embed(
        self, texts: Union[str, List[str]], **kwargs: Any
    ) -> Union[List[float], List[List[float]]]:
        """Generate embeddings, at the requested width where the model allows it.

        The 3-series (``text-embedding-3-small`` / ``-3-large``) accepts a
        ``dimensions`` parameter and returns vectors of that length;
        ``text-embedding-ada-002`` does not and rejects the parameter, so the
        request is forwarded only for a model that advertises
        :attr:`~dataknobs_llm.llm.base.ModelCapability.EMBEDDING_DIMENSIONS`.
        Where it cannot be forwarded the answer is checked instead — one rule
        for both, so no stated width is silently dropped.

        Forwarding this is not a formality on OpenAI. A consumer asking
        ``text-embedding-3-large`` for 512 and receiving 3072 gets valid
        vectors, six times wider than requested, at six times the storage —
        and pays for the difference.

        Args:
            texts: A single text or a batch.
            **kwargs: ``dimensions`` (int) overrides ``LLMConfig.dimensions``
                for this call. Other keys are ignored.
        """
        if not self._is_initialized:
            await self.initialize()

        if isinstance(texts, str):
            texts = [texts]
            single = True
        else:
            single = False

        model = self.config.model or "text-embedding-ada-002"
        requested = self._requested_embedding_dimensions(kwargs)
        params: Dict[str, Any] = {"input": texts, "model": model}
        forwardable = self._forwardable_embedding_dimensions(kwargs)
        if forwardable is not None:
            params["dimensions"] = forwardable

        response = await self._call_api(lambda: self._client.embeddings.create(**params))

        embeddings = [e.embedding for e in response.data]
        self._check_embedding_width(embeddings, requested)
        return embeddings[0] if single else embeddings

    async def function_call(
        self, messages: List[LLMMessage], functions: List[Dict[str, Any]], **kwargs: Any
    ) -> LLMResponse:
        """Execute function calling."""
        warnings.warn(
            "function_call() is deprecated, use complete(tools=...) instead",
            DeprecationWarning,
            stacklevel=2,
        )
        if not self._is_initialized:
            await self.initialize()

        # Add system prompt if configured
        if self.config.system_prompt and messages[0].role != "system":
            messages.insert(0, LLMMessage(role="system", content=self.config.system_prompt))

        # Adapt messages and config (drops rejected params, clamps max_tokens,
        # applies the family's wire-param remaps — no-op for permissive models).
        adapted_messages = self.adapter.adapt_messages(messages)
        params = self._build_api_kwargs(self.config, kwargs)
        params["functions"] = functions
        params["function_call"] = kwargs.get("function_call", "auto")

        # Make API call
        response = await self._call_api(
            lambda: self._client.chat.completions.create(messages=adapted_messages, **params)
        )

        return self._analyze_response(self.adapter.adapt_response(response))
