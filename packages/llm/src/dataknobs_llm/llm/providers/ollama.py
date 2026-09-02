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

import asyncio
import json
import logging
import os
import re
from collections.abc import Iterable
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Dict, List, Union, AsyncIterator

from ..base import (
    LLMAdapter,
    LLMConfig,
    LLMMessage,
    LLMResponse,
    LLMStreamResponse,
    AsyncLLMProvider,
    ModelCapability,
    ToolCall,
    normalize_llm_config,
)
from ..model_profile import (
    CallableModelMetadataSource,
    ConfigOverrideSource,
    LayeredModelProfileResolver,
    LiveApiSource,
    ModelProfile,
)
from ..profile_detection import ProfileDetectionMixin
from ._aiohttp_shared import raise_for_status_with_body
from dataknobs_llm.prompts import AsyncPromptBuilder

if TYPE_CHECKING:
    from dataknobs_config.config import Config

logger = logging.getLogger(__name__)

# Seconds to sleep after aiohttp ClientSession.close() so that SSL transport
# callbacks scheduled via loop.call_soon() can drain before the event loop
# shuts down.  Without this, asyncio.run() hangs when session close is
# immediately followed by loop shutdown (e.g. error-path cleanup in
# DynaBot.from_config()).  This is the standard workaround recommended by the
# aiohttp documentation.
_AIOHTTP_DRAIN_SECS = 0.25


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
    return [m for m in available_models if m == base_model or m.startswith(base_model + ":")]


def ollama_match_key(model_lower: str, keys: Iterable[str]) -> str | None:
    """Resolve *model_lower* to the best-matching Ollama cache key, or ``None``.

    The ``name:tag``-aware matcher injected into the model-metadata
    :class:`~..model_profile.LiveApiSource` (its ``match=`` arg) so the live
    cache uses Ollama's own base-name-or-exact-tag semantics
    (:func:`_find_matching_models`) instead of the substrate default
    :func:`~..model_profile.match_family_key`. The default's pure-substring
    family-alias rule reintroduces the documented prefix-collision bug for
    Ollama ids — ``nomic-embed-text`` is a substring of
    ``nomic-embed-text-v2-moe:latest``, so it would false-resolve to the v2-moe
    model's profile — which :func:`_find_matching_models` exists to prevent.

    Exact id wins; otherwise a bare base name matches its tagged variants
    (``llama3.1`` → ``llama3.1:8b``) but never a different model that merely
    shares a prefix. Returns the first match (the cache holds one entry per
    installed model), or ``None`` when nothing matches.
    """
    matches = _find_matching_models(model_lower, list(keys))
    return matches[0] if matches else None


# Regex for <think>...</think> blocks emitted by reasoning models.
# DOTALL so '.' matches newlines inside the tag.
_THINK_TAG_RE = re.compile(r"^<think>(.*?)</think>\s*(.*)", re.DOTALL)


# ---------------------------------------------------------------------------
# Model-metadata binding (capabilities + context window + availability)
# ---------------------------------------------------------------------------
#
# Ollama is live-first: the server authoritatively reports each installed
# model's capabilities and context window via `/api/show`, so the binding
# sources those facets from the live API (per-provider LiveApiSource) with a
# name-based heuristic fallback for older servers — inverting the resource-first
# shape of the cloud providers. There is no pricing (local/free) and no output
# ceiling (num_predict: -1 = unlimited), so those facets stay unset by design.

#: Ollama `/api/show` `capabilities` vocabulary -> dataknobs ModelCapability.
#: The server reports these on modern Ollama; the mapping is applied live-first
#: by :func:`_ollama_caps_from_server` and mirrored by the name-based
#: :func:`_ollama_heuristic` fallback for servers that predate the field.
_OLLAMA_CAPABILITY_MAP: dict[str, frozenset[ModelCapability]] = {
    "completion": frozenset(
        {
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CHAT,
            ModelCapability.STREAMING,
        }
    ),
    "tools": frozenset({ModelCapability.FUNCTION_CALLING}),
    "vision": frozenset({ModelCapability.VISION}),
    "embedding": frozenset({ModelCapability.EMBEDDINGS}),
}

#: Ollama model-name substrings carrying function calling — the demoted family
#: knowledge from the pre-binding ``_detect_capabilities``, used only by the
#: heuristic fallback. The live ``/api/show`` ``tools`` capability supersedes
#: this whenever the server reports one.
_TOOL_CAPABLE_FAMILIES: tuple[str, ...] = (
    "llama3",
    "llama4",
    "mistral",
    "mixtral",
    "qwen",
    "command-r",
    "phi3",
    "phi4",
    "nemotron",
    "firefunction",
    "hermes",
    "gpt-oss",
)

#: Vision family substrings for the heuristic fallback (server `vision` wins).
_VISION_FAMILIES: tuple[str, ...] = ("llava", "bakllava")

#: Code family substrings (Ollama does not report CODE as a capability, so it
#: is always name-derived — on both the live and heuristic paths).
_CODE_FAMILIES: tuple[str, ...] = ("codellama", "codegemma")


def _name_has(model_lower: str, families: tuple[str, ...]) -> bool:
    """Whether *model_lower* contains any of *families* as a substring."""
    return any(fam in model_lower for fam in families)


def _is_embedding_only_name(model_lower: str) -> bool:
    """Whether a model *name* denotes a dedicated embedding model.

    Used only by the heuristic fallback (the live path reads the server's
    ``embedding`` / ``completion`` capabilities). A name carrying ``embed`` with
    no code marker is treated as embedding-only so it resolves an
    EMBEDDINGS-only disjoint set — matching how a modern server reports it.
    """
    return "embed" in model_lower and not _name_has(model_lower, _CODE_FAMILIES)


def _ollama_caps_from_server(model: str, reported: Iterable[str]) -> frozenset[ModelCapability]:
    """Map a server-reported `capabilities` array to the COMPLETE capability set.

    Because a present ``capabilities`` facet whole-set-overrides the lower
    heuristic layer (the substrate merge is first-non-``None``-per-facet, not a
    union), the live extractor must emit the full intended set, not a partial —
    else a live-classified model would lose the heuristic's JSON_MODE / CODE /
    broad EMBEDDINGS. So on top of the direct vocabulary mapping this adds, for a
    completion (chat) model:

    - JSON_MODE — Ollama's ``format: json`` is universal for chat models; the
      server does not report it as a capability.
    - EMBEDDINGS — ``/api/embeddings`` accepts completion models too, so the
      historical breadth is preserved; a pure embedding model (``embedding``
      without ``completion``) stays EMBEDDINGS-only-disjoint.
    - CODE by model name — the server does not report it.
    """
    reported_set = {str(c).lower() for c in reported}
    caps: set[ModelCapability] = set()
    for name in reported_set:
        caps |= _OLLAMA_CAPABILITY_MAP.get(name, frozenset())
    if "completion" in reported_set:
        caps.add(ModelCapability.JSON_MODE)
        caps.add(ModelCapability.EMBEDDINGS)
        if _name_has(model.lower(), _CODE_FAMILIES):
            caps.add(ModelCapability.CODE)
    return frozenset(caps)


def _ollama_heuristic(model: str) -> ModelProfile:
    """Name-based capability fallback (older servers / `/api/show` failure).

    The corrected, demoted form of the pre-binding inline
    ``_detect_capabilities``: produces the SAME complete-set shape as
    :func:`_ollama_caps_from_server` from the model name alone. A dedicated
    embedding model (name carrying ``embed``) resolves an EMBEDDINGS-only
    disjoint set; every other model resolves the base chat set plus JSON_MODE
    (universal ``format: json``) and broad EMBEDDINGS, plus FUNCTION_CALLING /
    VISION / CODE by family. Contributes only the ``capabilities`` facet.
    """
    model_lower = model.lower()
    if _is_embedding_only_name(model_lower):
        return ModelProfile(capabilities=frozenset({ModelCapability.EMBEDDINGS}))
    caps: set[ModelCapability] = {
        ModelCapability.TEXT_GENERATION,
        ModelCapability.CHAT,
        ModelCapability.STREAMING,
        ModelCapability.JSON_MODE,
        ModelCapability.EMBEDDINGS,
    }
    if _name_has(model_lower, _TOOL_CAPABLE_FAMILIES):
        caps.add(ModelCapability.FUNCTION_CALLING)
    if _name_has(model_lower, _VISION_FAMILIES):
        caps.add(ModelCapability.VISION)
    if _name_has(model_lower, _CODE_FAMILIES):
        caps.add(ModelCapability.CODE)
    return ModelProfile(capabilities=frozenset(caps))


def _extract_context_length(model_info: Any) -> int | None:
    """Read the input/context-window size from an `/api/show` `model_info` dict.

    Ollama serves ``model_info`` as a flat dict with architecture-prefixed keys
    (``{"general.architecture": "llama", "llama.context_length": 131072}``).
    Reads the architecture's ``<arch>.context_length`` first, falling back to
    any ``*.context_length`` key. Missing / non-integer → ``None`` (permissive).
    """
    if not isinstance(model_info, dict):
        return None
    arch = model_info.get("general.architecture")
    candidates: list[Any] = []
    if arch:
        candidates.append(model_info.get(f"{arch}.context_length"))
    candidates.extend(v for k, v in model_info.items() if k.endswith(".context_length"))
    for value in candidates:
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _ollama_live_extractor(entry: dict[str, Any]) -> ModelProfile:
    """Project one enriched `/api/tags`+`/api/show` entry into a partial profile.

    The ``extractor`` half of the Ollama live source. Each *entry* is a plain
    dict ``{name, capabilities, context_length}`` assembled by
    :meth:`OllamaProvider._list_ollama_models`. Sets:

    - ``available=True`` — the entry exists because the model is installed
      locally (from ``/api/tags``), which for Ollama *is* availability.
    - ``capabilities`` — the complete set from the server-reported array
      (:func:`_ollama_caps_from_server`), or ``None`` when the server did not
      report one (older Ollama) **or reported one that maps to nothing** (an
      empty or all-unrecognized array), so the heuristic fallback supplies the
      set per facet rather than an authoritative-empty set shadowing it away.
    - ``context_window`` — the server-reported input window, or ``None``.
    """
    reported = entry.get("capabilities")
    capabilities: frozenset[ModelCapability] | None = None
    if reported is not None:
        mapped = _ollama_caps_from_server(entry.get("name", ""), reported)
        # An empty mapped set means the server reported an array we recognize
        # none of — a proxy/gateway's ``[]``, or a future FIM/reasoning-only
        # capability not yet in ``_OLLAMA_CAPABILITY_MAP``. Under the substrate
        # merge (first-non-``None``-per-facet) a present empty ``frozenset()`` is
        # *authoritatively known* and would whole-set-shadow the heuristic to
        # zero capabilities (no CHAT). Degrade to ``None`` so the heuristic
        # fallback supplies the set instead of the model losing every capability.
        capabilities = mapped or None
    return ModelProfile(
        capabilities=capabilities,
        context_window=entry.get("context_length"),
        available=True,
    )


def _ollama_entry_model_id(entry: Any) -> str | None:
    """Read a live-source entry's model id (its ``name``) — the cache key.

    The entries are dicts, so the substrate's default attribute-based
    :func:`~..model_profile._default_model_id` does not apply.
    """
    value = entry.get("name") if isinstance(entry, dict) else None
    return str(value) if value is not None else None


#: The stateless heuristic Ollama model-metadata source — a module singleton
#: (a pure name-substring rule, no per-instance state). The live source is
#: per-provider (it owns its own ``/api/tags`` + ``/api/show`` cache) and the
#: config-override source is prepended per config, so both are composed in
#: :meth:`OllamaProvider._profile_resolver`. There is deliberately **no**
#: bundled-resource layer: Ollama's model space is open-ended and user-pulled,
#: and the server reports capabilities / context live (see the block comment).
_OLLAMA_HEURISTIC_SOURCE = CallableModelMetadataSource("ollama_heuristic", _ollama_heuristic)


def _coerce_bool(value: Any, *, default: bool) -> bool:
    """Coerce a config ``options`` value to ``bool`` (string-tolerant)."""
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes", "on")
    return bool(value)


def _coerce_ttl(value: Any, *, default: float) -> float:
    """Coerce a config ``options`` value to a non-negative TTL float."""
    try:
        ttl = float(value)
    except (TypeError, ValueError):
        return default
    return ttl if ttl >= 0 else default


#: Default TTL / refresh-timeout for the live model-metadata cache. It is
#: refreshed at most once per TTL per event loop (out-of-band, at the request
#: boundary) and a single poll is bounded by the timeout. Both are overridable
#: via ``LLMConfig.options`` (``model_metadata_ttl`` /
#: ``model_metadata_refresh_timeout``); the live source is enabled by default
#: (Ollama is local — no least-privilege concern) and disablable via
#: ``options["model_metadata_live"]=false``.
_DEFAULT_MODEL_METADATA_TTL: float = 3600.0

#: Max concurrent ``/api/show`` probes during one metadata poll. The per-model
#: shows are independent, so they run concurrently rather than sequentially: a
#: box with many installed models would otherwise pay N sequential round-trips
#: under the held refresh lock and could exhaust ``refresh_timeout`` (leaving the
#: cache empty → capabilities always heuristic). Bounded so a large install set
#: does not open an unbounded fan-out at the local server.
_MODEL_SHOW_CONCURRENCY: int = 8
_DEFAULT_MODEL_METADATA_REFRESH_TIMEOUT: float = 10.0


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
            system_prompt: Accepted for interface compatibility but
                ignored — Ollama passes system content as a normal message.

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

        Raises:
            ValidationError: If a tool call carries arguments that are not,
                and do not decode to, a JSON object. See
                :meth:`LLMAdapter.tool_call_parameters`.
        """
        message = data.get("message", {})
        content = message.get("content", "")
        raw_tool_calls = message.get("tool_calls", [])

        tool_calls = None
        if raw_tool_calls:
            tool_calls = []
            for tc in raw_tool_calls:
                function = tc.get("function", {})
                name = function.get("name", "")
                tool_calls.append(
                    ToolCall(
                        name=name,
                        parameters=self.tool_call_parameters(name, function.get("arguments")),
                        id=tc.get("id"),
                    )
                )

        usage = None
        if "eval_count" in data:
            usage = {
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
                "total_tokens": (data.get("prompt_eval_count", 0) + data.get("eval_count", 0)),
            }

        # Ollama reports a token-budget cut-off with done_reason == "length"
        # (the equivalent of Anthropic's max_tokens); done_reason == "stop"
        # is a clean finish. Older/streaming payloads may omit done_reason,
        # so fall back to the done flag.
        done_reason = data.get("done_reason")
        truncated = done_reason == "length"
        if tool_calls:
            finish_reason = "tool_calls"
        elif truncated:
            finish_reason = "length"
        elif data.get("done"):
            finish_reason = "stop"
        else:
            finish_reason = "length"

        return LLMResponse(
            content=content,
            model=data.get("model", ""),
            finish_reason=finish_reason,
            truncated=truncated,
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
        return [self._tool_to_dict(tool) for tool in tools]

    @staticmethod
    def _tool_to_dict(tool: Any) -> Dict[str, Any]:
        """Convert a single Tool or raw dict to Ollama format."""
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.schema if hasattr(tool, "schema") else {},
            },
        }


class OllamaProvider(ProfileDetectionMixin, AsyncLLMProvider):
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
        from dataknobs_llm import Tool

        class WeatherTool(Tool):
            def __init__(self):
                super().__init__("get_weather", "Get current weather")

            @property
            def schema(self):
                return {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                }

            async def execute(self, location: str):
                return f"sunny in {location}"

        response = await llm.complete(messages, tools=[WeatherTool()])
        for call in response.tool_calls or []:
            print(call.name, call.parameters)
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
        prompt_builder: AsyncPromptBuilder | None = None,
    ):
        # Normalize config first
        llm_config = normalize_llm_config(config)
        super().__init__(llm_config, prompt_builder=prompt_builder)

        self.adapter = OllamaAdapter()

        # Check for Docker environment and adjust URL accordingly
        default_url = "http://localhost:11434"
        if os.path.exists("/.dockerenv"):
            # Running in Docker, use host.docker.internal
            default_url = "http://host.docker.internal:11434"

        # Allow environment variable override
        self.base_url = llm_config.api_base or os.environ.get("OLLAMA_BASE_URL", default_url)

        # Live-first model-metadata source: capabilities + context window +
        # availability from GET /api/tags (installed set) enriched per-model by
        # POST /api/show (the server's authoritative `capabilities` array +
        # `model_info.<arch>.context_length`). Per-provider (owns its own cache);
        # refreshed out-of-band at the request boundary (TTL-gated, per-loop
        # locked) and read synchronously on the detect path. Enabled by default
        # (Ollama is local); disablable / tunable via LLMConfig.options. The
        # `name:tag`-aware matcher (`ollama_match_key`) is injected so the live
        # cache does not reintroduce the documented prefix-collision bug.
        self._live_source = LiveApiSource(
            self._list_ollama_models,
            _ollama_live_extractor,
            name="live_api",
            ttl=_coerce_ttl(
                llm_config.options.get("model_metadata_ttl"),
                default=_DEFAULT_MODEL_METADATA_TTL,
            ),
            refresh_timeout=_coerce_ttl(
                llm_config.options.get("model_metadata_refresh_timeout"),
                default=_DEFAULT_MODEL_METADATA_REFRESH_TIMEOUT,
            ),
            enabled=_coerce_bool(llm_config.options.get("model_metadata_live"), default=True),
            model_id=_ollama_entry_model_id,
            match=ollama_match_key,
        )

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
                # replace() copies every field (including ``truncated`` and any
                # field added to LLMResponse later), so extracting the visible
                # answer can never silently drop one — only ``content`` changes,
                # and ``metadata`` was already mutated in place above.
                response = replace(response, content=visible_text)
        return super()._analyze_response(response)

    def _messages_to_ollama(self, messages: List[LLMMessage]) -> List[Dict[str, Any]]:
        """Convert LLMMessage list to Ollama chat format.

        Delegates to the adapter.
        """
        return self.adapter.adapt_messages(messages)

    # Same finding as ``AsyncLLMProvider.initialize`` one level up, and the
    # same answer: ``LLMProvider`` declares the pair sync, so the whole async
    # subtree contradicts its own base. Resolving it moves the pair down into
    # ``SyncLLMProvider`` --- a public-ABC contract change needing consumer
    # verification, argued and deferred where the base declares it. Suppressed
    # here for that decision, not because this override is wrong.
    async def initialize(self) -> None:  # type: ignore[override]
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
                        models = [m["name"] for m in data.get("models", [])]
                        if models:
                            # Check if configured model is available
                            matching = _find_matching_models(self.config.model, models)
                            if matching and matching[0] != self.config.model:
                                # ``LLMConfig`` is a frozen ``StructuredConfig`` —
                                # replace the config via ``clone`` rather than
                                # mutating the (immutable) ``model`` field.
                                #
                                # ``self.config`` is a plain (mutable) instance
                                # attribute here, so reassigning it is fine. If
                                # this provider is ever migrated to
                                # ``StructuredConfigConsumer`` (where ``config`` is
                                # a read-only property), this site must rebind the
                                # backing ``_config`` instead — a plain
                                # ``self.config = ...`` would then raise at runtime.
                                self.config = self.config.clone(model=matching[0])
                                logger.info("Ollama: Using model %s", self.config.model)
                            elif not matching:
                                logger.warning(
                                    "Ollama: Model %s not found. Available: %s",
                                    self.config.model,
                                    models,
                                )
                        else:
                            logger.warning("Ollama: No models found. Please pull a model first.")
                    else:
                        logger.warning("Ollama: API returned status %s", response.status)
            except Exception as e:
                logger.warning("Ollama: Could not connect to %s: %s", self.base_url, e)

            self._is_initialized = True
        except ImportError as e:
            raise ImportError(
                "aiohttp package not installed. Install with: pip install 'dataknobs-llm[ollama]'"
            ) from e

    async def _close_client(self) -> None:
        """Close the aiohttp session."""
        if hasattr(self, "_session") and self._session:
            await self._session.close()
            await asyncio.sleep(_AIOHTTP_DRAIN_SECS)

    async def _list_ollama_models(self) -> list[dict[str, Any]]:
        """Collect installed models, enriched with live capabilities + context.

        The ``list_models`` callable for the provider's
        :class:`~..model_profile.LiveApiSource`. Queries ``GET /api/tags`` for
        the installed set, then ``POST /api/show`` per model to read the
        server's authoritative ``capabilities`` array and
        ``model_info.<arch>.context_length``. Returns one plain dict
        ``{name, capabilities, context_length}`` per installed model (the
        extractor's input). The source drives this out-of-band (TTL-gated,
        per-loop-locked, ``refresh_timeout``-bounded), so the N+1 shape (one
        tags call + N show calls, N = installed-model count) is off the request
        path. The N ``/api/show`` probes run **concurrently** (bounded by
        :data:`_MODEL_SHOW_CONCURRENCY`) rather than sequentially, so a box with
        many installed models does not exhaust ``refresh_timeout`` on serial
        round-trips. A per-model ``/api/show`` failure is tolerated — that model
        still contributes ``available=True`` (from ``/api/tags``) with unknown
        capabilities / context (the heuristic then supplies capabilities).
        """
        async with self._session.get(f"{self.base_url}/api/tags") as response:
            if response.status != 200:
                return []
            data = await response.json()
        names = [name for model in data.get("models", []) if (name := model.get("name"))]
        if not names:
            return []
        sem = asyncio.Semaphore(_MODEL_SHOW_CONCURRENCY)

        async def _enrich(name: str) -> dict[str, Any]:
            entry: dict[str, Any] = {
                "name": name,
                "capabilities": None,
                "context_length": None,
            }
            async with sem:
                try:
                    async with self._session.post(
                        f"{self.base_url}/api/show", json={"model": name}
                    ) as show_response:
                        if show_response.status == 200:
                            show = await show_response.json()
                            entry["capabilities"] = show.get("capabilities")
                            entry["context_length"] = _extract_context_length(
                                show.get("model_info")
                            )
                except Exception as exc:  # per-model best-effort — keep availability
                    logger.debug("Ollama /api/show failed for %s: %s", name, exc)
            return entry

        entries: list[dict[str, Any]] = await asyncio.gather(*(_enrich(name) for name in names))
        return entries

    def _profile_resolver(self, config: LLMConfig) -> LayeredModelProfileResolver:
        """Compose the Ollama model-profile resolver for *config*.

        Precedence (highest first): config override → live ``/api/show`` cache
        (the per-provider :class:`~..model_profile.LiveApiSource`) → name-based
        heuristic. Live-first (the server authoritatively reports capabilities +
        context for installed models), heuristic as the graceful-degradation
        fallback for older servers — there is **no** bundled-resource layer
        (Ollama's model space is open-ended and user-pulled) and **no** pricing /
        output-ceiling layer (Ollama is local/free with no output cap). A consumer's
        ``LLMConfig.model_profile_overrides`` wins over all of them per facet.
        """
        return LayeredModelProfileResolver(
            [
                ConfigOverrideSource(getattr(config, "model_profile_overrides", None)),
                self._live_source,
                _OLLAMA_HEURISTIC_SOURCE,
            ]
        )

    async def refresh_model_metadata(self) -> None:
        """Force an immediate refresh of the cached live model metadata.

        Public entry point for a consumer that prefers to drive freshness on
        their own schedule instead of relying on the TTL. Bypasses the TTL gate
        but honors ``options["model_metadata_live"]=false`` (a no-op then).
        Never raises — the underlying poll is best-effort.
        """
        if not self._is_initialized:
            await self.initialize()
        await self._live_source.force_refresh()

    async def validate_model(self) -> bool:
        """Validate model availability against the live ``available`` facet.

        Reads the resolved profile's ``available`` facet: the live source is the
        only source that sets it (``True`` for a model in the installed
        ``/api/tags`` set), so a model absent from the box resolves
        ``available=None`` → ``False`` — no bespoke direct-consult needed
        (unlike a provider whose lower resource layer sets a permissive
        ``available``). **Force-refreshes** the cache first (bypassing the TTL
        gate) so a model pulled since the last poll is seen immediately — an
        authoritative liveness check matching the pre-binding fresh-every-call
        ``/api/tags`` probe, not a value that can lag by up to a TTL. A cold
        cache against an unreachable server resolves ``False``; on a warm cache
        a transient poll failure leaves the last known state intact (a blip does
        not flip availability). Preserves the pre-binding behavior (installed →
        ``True``; not-installed / unreachable-cold → ``False``) as a resolved facet.
        """
        if not self._is_initialized or not hasattr(self, "_session"):
            return False
        await self._live_source.force_refresh()
        profile = self._profile_resolver(self.config).resolve(self.config.model)
        return bool(profile.available)

    def _build_shaped_options(self, config: LLMConfig) -> Dict[str, Any]:
        """Build Ollama ``options`` with the family's request-shape rules applied.

        Request-shaping choke point shared by ``complete`` and
        ``stream_complete``, mirroring the other providers' ``_build_api_kwargs``:
        delegates the request-shaping front-half to the shared
        :meth:`~..base.LLMProvider._shape_request_params` (resolves constraints
        once, drops family-rejected sampling params, clamps ``max_tokens`` to the
        ceiling — in canonical config space, independent of Ollama's nested
        ``options`` wire shape), builds the ``options`` dict, then applies any
        wire-level :meth:`~..base.LLMProvider._apply_param_remaps`. Ollama's
        auto-detected rules are empty by default (no ceiling, no rejected params,
        no remaps), so this is a byte-identical no-op in normal use — wired for the
        consumer-``constraints``-override / future-family path, symmetric with the
        other three providers.
        """
        shaped_config, _, constraints = self._shape_request_params(config)
        options = self._build_options(shaped_config)
        return self._apply_param_remaps(options, constraints.param_remaps)

    def _translate_api_error(self, exc: Exception) -> Exception | None:
        """Translate a raw aiohttp transport error into a dataknobs exception.

        Lets consumers catch by a dataknobs exception type instead of coupling
        to ``aiohttp``. Ollama has no SDK — it speaks HTTP over
        ``aiohttp``, so the gate is aiohttp's error hierarchy plus
        ``TimeoutError``. Extracts the status (from a
        ``ClientResponseError`` raised by ``raise_for_status()``; ``None`` for a
        connection error or timeout) and defers the status→type policy to
        :meth:`~dataknobs_llm.llm.base.LLMProvider._dataknobs_error_for_status`:

        - 429 → :class:`~dataknobs_common.exceptions.RateLimitError`,
        - 400 → :class:`~dataknobs_common.exceptions.ValidationError`,
        - 401/403 / other status / connection / timeout →
          :class:`~dataknobs_common.exceptions.OperationError`.

        Returns ``None`` for a non-transport exception so the caller re-raises
        it unchanged — this is what lets the domain-specific
        :class:`~dataknobs_llm.exceptions.ToolsNotSupportedError` (raised for a
        400 "does not support tools" body) pass through untranslated. The
        original error is preserved on ``__cause__`` — callers raise
        ``... from exc``.
        """
        import aiohttp

        if isinstance(exc, aiohttp.ClientResponseError):
            retry_after = self._retry_after_from_headers(getattr(exc, "headers", None))
            return self._dataknobs_error_for_status(exc.status, str(exc), retry_after=retry_after)
        if isinstance(exc, (aiohttp.ClientError, TimeoutError)):
            return self._dataknobs_error_for_status(None, str(exc))
        return None

    async def complete(
        self,
        messages: Union[str, List[LLMMessage]],
        config_overrides: Dict[str, Any] | None = None,
        tools: list[Any] | None = None,
        **kwargs: Any,
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
            messages = [LLMMessage(role="user", content=messages)]

        # Add system prompt if configured
        if runtime_config.system_prompt and (not messages or messages[0].role != "system"):
            messages = [LLMMessage(role="system", content=runtime_config.system_prompt)] + list(
                messages
            )

        # Convert to Ollama format
        ollama_messages = self._messages_to_ollama(messages)

        # Keep the live model-metadata cache fresh (TTL-gated, ≤1 poll per TTL
        # per loop) so the constraint reads in _build_shaped_options see current
        # context-window / (consumer-overridden) shaping rules.
        await self._live_source.refresh_if_stale()

        # Build payload for chat endpoint (options shaped by the model family's
        # constraints — a no-op by default; honors a consumer constraints override)
        payload = {
            "model": runtime_config.model,
            "messages": ollama_messages,
            "stream": False,
            "options": self._build_shaped_options(runtime_config),
        }

        # Add format if JSON mode requested
        if runtime_config.response_format == "json":
            payload["format"] = "json"

        # Handle tools if provided
        if tools:
            payload["tools"] = self.adapter.adapt_tools(tools)

        # Forward 'think' parameter for reasoning models (e.g. qwen3, deepseek-r1).
        # When True, the model emits <think>...</think> blocks before the answer.
        think = runtime_config.options.get("think")
        if think is not None:
            payload["think"] = bool(think)

        try:
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
                        logger.error(
                            "Ollama API error (status %s): %s", response.status, error_text
                        )
                        logger.error("Request payload: %s", json.dumps(payload, indent=2))
                        await raise_for_status_with_body(response, body=error_text)
                else:
                    data = await response.json()
        except Exception as exc:
            # ToolsNotSupportedError (non-transport) passes through untranslated.
            self._raise_translated(exc)

        parsed = self.adapter.adapt_response(data)
        # Override model with runtime config model (adapter uses response model)
        parsed.model = runtime_config.model
        return self._analyze_response(parsed)

    async def stream_complete(
        self,
        messages: Union[str, List[LLMMessage]],
        config_overrides: Dict[str, Any] | None = None,
        tools: list[Any] | None = None,
        **kwargs: Any,
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
            messages = [LLMMessage(role="user", content=messages)]

        # Add system prompt if configured
        if runtime_config.system_prompt and (not messages or messages[0].role != "system"):
            messages = [LLMMessage(role="system", content=runtime_config.system_prompt)] + list(
                messages
            )

        # Convert to Ollama format
        ollama_messages = self._messages_to_ollama(messages)

        # Keep the live model-metadata cache fresh before shaping (mirrors
        # complete()) — same choke point, TTL-gated.
        await self._live_source.refresh_if_stale()

        # Build payload for chat endpoint (mirrors complete())
        payload: Dict[str, Any] = {
            "model": runtime_config.model,
            "messages": ollama_messages,
            "stream": True,
            "options": self._build_shaped_options(runtime_config),
        }

        # Add format if JSON mode requested
        if runtime_config.response_format == "json":
            payload["format"] = "json"

        # Handle tools if provided
        if tools:
            payload["tools"] = self.adapter.adapt_tools(tools)

        # Forward 'think' parameter for reasoning models (mirrors complete())
        think = runtime_config.options.get("think")
        if think is not None:
            payload["think"] = bool(think)

        try:
            async with self._session.post(f"{self.base_url}/api/chat", json=payload) as response:
                await raise_for_status_with_body(response)

                async for line in response.content:
                    if line:
                        data = json.loads(line.decode("utf-8"))
                        msg = data.get("message", {})
                        done = data.get("done", False)

                        if done:
                            # Use adapter for final chunk parsing
                            parsed = self.adapter.adapt_response(data)
                            final_chunk = LLMStreamResponse(
                                delta=msg.get("content", ""),
                                is_final=True,
                                finish_reason=parsed.finish_reason,
                                truncated=parsed.truncated,
                                usage=parsed.usage,
                                tool_calls=parsed.tool_calls,
                                model=runtime_config.model,
                            )
                            self._warn_if_truncated(final_chunk)
                            yield final_chunk
                        else:
                            yield LLMStreamResponse(
                                delta=msg.get("content", ""),
                                is_final=False,
                            )
        except Exception as exc:
            self._raise_translated(exc)

    async def embed(
        self, texts: Union[str, List[str]], **kwargs: Any
    ) -> Union[List[float], List[List[float]]]:
        """Generate embeddings, checking any width the caller asked for.

        Ollama's ``/api/embeddings`` takes a model and a prompt: the width is
        the model's and there is no parameter to change it. So a stated
        ``dimensions`` cannot be forwarded, and the choice is between checking
        it and ignoring it. This checks — a config asking for 512 from a
        768-wide model used to receive 768 and say nothing, which is how a
        width promised in config and a width written to a store come apart.

        Declaring the width a model *does* produce stays valid and silent;
        the rule is that a stated width is never ignored, not that one may
        not be stated.

        Args:
            texts: A single text or a batch.
            **kwargs: ``dimensions`` (int) overrides ``LLMConfig.dimensions``
                for this call. Checked, not forwarded — see above.
        """
        if not self._is_initialized:
            await self.initialize()

        if isinstance(texts, str):
            texts = [texts]
            single = True
        else:
            single = False

        requested = self._requested_embedding_dimensions(kwargs)
        embeddings = []
        for text in texts:
            payload = {"model": self.config.model, "prompt": text}

            try:
                async with self._session.post(
                    f"{self.base_url}/api/embeddings", json=payload
                ) as response:
                    await raise_for_status_with_body(response)
                    data = await response.json()
                    embeddings.append(data["embedding"])
            except Exception as exc:
                self._raise_translated(exc)

        self._check_embedding_width(embeddings, requested)
        return embeddings[0] if single else embeddings

    def _build_prompt(self, messages: List[LLMMessage]) -> str:
        """Build prompt from messages."""
        prompt = ""
        for msg in messages:
            if msg.role == "system":
                prompt += f"System: {msg.content}\n\n"
            elif msg.role == "user":
                prompt += f"User: {msg.content}\n\n"
            elif msg.role == "assistant":
                prompt += f"Assistant: {msg.content}\n\n"
        return prompt
