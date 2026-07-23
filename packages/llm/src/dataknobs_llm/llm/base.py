"""Base LLM abstraction components.

This module provides the base abstractions for unified LLM operations across
different providers (OpenAI, Anthropic, Ollama, etc.). It defines standard
interfaces for completions, streaming, embeddings, and function calling.

The architecture follows a provider pattern where all LLM providers implement
common interfaces (AsyncLLMProvider or SyncLLMProvider) and use standardized
data structures (LLMMessage, LLMResponse, LLMConfig).

Key Components:
    - LLMProvider: Base provider interface with initialization and lifecycle
    - AsyncLLMProvider: Async provider with complete(), stream_complete(), embed()
    - SyncLLMProvider: Synchronous version for non-async applications
    - LLMMessage: Standard message format for conversations
    - LLMResponse: Standard response with content, usage, and cost tracking
    - LLMConfig: Comprehensive configuration with 20+ parameters
    - LLMAdapter: Format adapters for provider-specific APIs
    - LLMMiddleware: Request/response processing pipeline

Example:
    ```python
    from dataknobs_llm import create_llm_provider
    from dataknobs_llm.llm.base import LLMConfig, LLMMessage

    # Create provider with config
    config = LLMConfig(
        provider="openai",
        model="gpt-4",
        temperature=0.7,
        max_tokens=500
    )

    # Async usage
    async with create_llm_provider(config) as llm:
        # Simple completion
        response = await llm.complete("What is Python?")
        print(response.content)

        # Streaming
        async for chunk in llm.stream_complete("Tell me a story"):
            print(chunk.delta, end="", flush=True)

        # Multi-turn conversation
        messages = [
            LLMMessage(role="system", content="You are helpful"),
            LLMMessage(role="user", content="Hello!"),
        ]
        response = await llm.complete(messages)
    ```

See Also:
    - dataknobs_llm.llm.providers: Provider implementations
    - dataknobs_llm.conversations: Multi-turn conversation management
    - dataknobs_llm.prompts: Prompt rendering and RAG integration
"""

import asyncio
import logging
import types
from abc import ABC, abstractmethod
from collections.abc import Awaitable
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any, ClassVar, Coroutine, Dict, List, NoReturn, TypeVar, Union,
    AsyncIterator, Iterator, Callable, Protocol
)

from dataknobs_common.exceptions import (
    OperationError, RateLimitError, ResourceError, ValidationError,
)
from dataknobs_common.structured_config import StructuredConfig
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

# Import prompt builder types - clean one-way dependency (llm depends on prompts)
from dataknobs_llm.prompts import AsyncPromptBuilder, PromptBuilder
from dataknobs_config.config import Config

logger = logging.getLogger(__name__)

# Item value yielded by a vendor stream, threaded through _iter_translated.
_StreamItem = TypeVar("_StreamItem")


class CompletionMode(Enum):
    """LLM completion modes.

    Defines the operation mode for LLM requests. Different modes use
    different APIs and formatting requirements.

    Attributes:
        CHAT: Chat completion with conversational message history
        TEXT: Raw text completion (legacy models)
        INSTRUCT: Instruction-following mode
        EMBEDDING: Generate vector embeddings for semantic search
        FUNCTION: Function/tool calling mode

    Example:
        ```python
        from dataknobs_llm.llm.base import LLMConfig, CompletionMode

        # Chat mode (default for modern models)
        config = LLMConfig(
            provider="openai",
            model="gpt-4",
            mode=CompletionMode.CHAT
        )

        # Embedding mode for vector search
        embedding_config = LLMConfig(
            provider="openai",
            model="text-embedding-ada-002",
            mode=CompletionMode.EMBEDDING
        )
        ```
    """
    CHAT = "chat"  # Chat completion with message history
    TEXT = "text"  # Text completion
    INSTRUCT = "instruct"  # Instruction following
    EMBEDDING = "embedding"  # Generate embeddings
    FUNCTION = "function"  # Function calling


class ModelCapability(Enum):
    """Model capabilities.

    Enumerates the capabilities that different LLM models support.
    Providers use this to advertise what features are available for
    a specific model.

    Attributes:
        TEXT_GENERATION: Basic text generation
        CHAT: Multi-turn conversational interactions
        EMBEDDINGS: Vector embedding generation
        FUNCTION_CALLING: Tool/function calling support
        VISION: Image understanding capabilities
        CODE: Code generation and analysis
        JSON_MODE: Structured JSON output
        STREAMING: Incremental response streaming

    Example:
        ```python
        from dataknobs_llm import create_llm_provider
        from dataknobs_llm.llm.base import ModelCapability

        # Check model capabilities
        llm = create_llm_provider("openai", model="gpt-4")
        capabilities = llm.get_capabilities()

        if ModelCapability.STREAMING in capabilities:
            # Use streaming
            async for chunk in llm.stream_complete("Hello"):
                print(chunk.delta, end="")

        if ModelCapability.FUNCTION_CALLING in capabilities:
            # Use function calling
            response = await llm.function_call(messages, functions)
        ```
    """
    TEXT_GENERATION = "text_generation"
    CHAT = "chat"
    EMBEDDINGS = "embeddings"
    FUNCTION_CALLING = "function_calling"
    VISION = "vision"
    CODE = "code"
    JSON_MODE = "json_mode"
    STREAMING = "streaming"


# String-to-enum mapping for config-driven capability references.
CAPABILITY_NAMES: dict[str, ModelCapability] = {
    cap.value: cap for cap in ModelCapability
}


@dataclass(frozen=True)
class ModelConstraints:
    """Request-shape rules a model family enforces at its API boundary.

    Orthogonal to :class:`ModelCapability`. A *capability* is a feature a
    model supports (vision, function calling); a *constraint* is a rule the
    model family imposes on the **shape of a request** — which parameters it
    rejects, whether it accepts a system message inline, an upper bound on
    ``max_tokens``. The two are separate axes: there is no ``TEMPERATURE``
    capability, and rejection is per-parameter (finer than any capability
    member), so constraints are their own small structure rather than +/- on
    the capability set.

    Like the resolved ``List[ModelCapability]`` (and unlike the stored config
    fields), a ``ModelConstraints`` is a **runtime value**: a provider
    auto-detects it (:meth:`LLMProvider._detect_constraints`) and the base
    overlays any ``LLMConfig.constraints`` override
    (:meth:`LLMProvider._resolve_constraints`). It is never a stored typed
    config field — keeping it a runtime object means a consumer can declare or
    withdraw a constraint in config without a dataknobs release (the answer to
    "the family table goes stale"), and there is no serialization surface to
    round-trip.

    Attributes:
        rejected_params: Generation/sampling parameter names the family
            rejects at the request boundary (e.g. the Claude 5 family rejects
            ``temperature`` with a 400). A provider drops these before the
            call — always drop-and-warn, never silently.
        accepts_inline_system: Whether the family accepts a ``role="system"``
            message at a non-leading position in the message array. Anthropic
            hoists every system message into a top-level ``system`` param, so
            a mid-conversation system message cannot sit inline
            (``False``); providers that pass ``system`` through the message
            array leave this ``True``. Read by the mid-conversation
            system-message policy.
        max_tokens_ceiling: Hard upper bound on ``max_tokens`` for the family,
            or ``None`` when unconstrained. Reserved for future clamping; no
            provider currently populates it.
    """

    rejected_params: frozenset[str] = frozenset()
    accepts_inline_system: bool = True
    max_tokens_ceiling: int | None = None

    def with_overrides(self, overrides: Dict[str, Any]) -> "ModelConstraints":
        """Return a copy with the given loose-dict overrides overlaid.

        Overlays per field (an absent key leaves the detected value intact),
        so a config can adjust one constraint without redeclaring the rest.
        ``rejected_params`` is *replaced* (the override declares the full set
        the consumer wants rejected — pass ``[]`` to withdraw a stale rule);
        ``accepts_inline_system`` / ``max_tokens_ceiling`` are coerced/passed
        through. Mirrors the loose ``LLMConfig.capabilities`` override shape.

        Args:
            overrides: Loose mapping with any of ``"rejected_params"``
                (iterable of str, or ``None`` → empty), ``"accepts_inline_system"``
                (bool), ``"max_tokens_ceiling"`` (int or ``None``).

        Returns:
            A new ``ModelConstraints`` (the receiver is never mutated).
        """
        from dataclasses import replace

        changes: Dict[str, Any] = {}
        if "rejected_params" in overrides:
            raw = overrides["rejected_params"]
            changes["rejected_params"] = (
                frozenset(raw) if raw is not None else frozenset()
            )
        if "accepts_inline_system" in overrides:
            changes["accepts_inline_system"] = bool(
                overrides["accepts_inline_system"]
            )
        if "max_tokens_ceiling" in overrides:
            changes["max_tokens_ceiling"] = overrides["max_tokens_ceiling"]
        return replace(self, **changes)


@dataclass
class ToolCall:
    """Represents a tool call from the LLM.

    Used when the LLM wants to invoke a tool/function during reasoning.

    Attributes:
        name: Name of the tool to call
        parameters: Arguments to pass to the tool
        id: Optional unique identifier for the tool call
    """
    name: str
    parameters: Dict[str, Any]
    id: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to canonical dictionary format for storage/interchange.

        Returns:
            Dictionary with ``name``, ``parameters``, and optionally ``id``.
        """
        d: Dict[str, Any] = {"name": self.name, "parameters": self.parameters}
        if self.id is not None:
            d["id"] = self.id
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolCall":
        """Create a ToolCall from a dictionary.

        Args:
            data: Dictionary with ``name`` (required), ``parameters``, and ``id``.

        Returns:
            ToolCall instance.
        """
        return cls(
            name=data["name"],
            parameters=data.get("parameters", {}),
            id=data.get("id"),
        )


@dataclass
class LLMMessage:
    """Represents a message in LLM conversation.

    Standard message format used across all providers. Messages are the
    fundamental unit of LLM interactions, containing role-based content
    for multi-turn conversations.

    Attributes:
        role: Message role - 'system', 'user', 'assistant', 'tool', or 'function'
        content: Message content text
        name: Optional name for function/tool messages or multi-user scenarios
        tool_call_id: Provider-assigned ID for pairing tool results with
            invocations (required by OpenAI and Anthropic APIs)
        function_call: Function call data for tool-using models
        tool_calls: Tool calls made by the assistant in this message
        metadata: Additional metadata (timestamps, IDs, etc.)

    Example:
        ```python
        from dataknobs_llm.llm.base import LLMMessage

        # System message
        system_msg = LLMMessage(
            role="system",
            content="You are a helpful coding assistant."
        )

        # User message
        user_msg = LLMMessage(
            role="user",
            content="How do I reverse a list in Python?"
        )

        # Assistant message
        assistant_msg = LLMMessage(
            role="assistant",
            content="Use the reverse() method or [::-1] slicing."
        )

        # Function result message
        function_msg = LLMMessage(
            role="function",
            name="search_docs",
            content='{"result": "Found 3 examples"}'
        )

        # Build conversation
        messages = [system_msg, user_msg, assistant_msg]
        ```
    """
    role: str  # 'system', 'user', 'assistant', 'tool', 'function'
    content: str
    name: str | None = None  # For function/tool messages
    tool_call_id: str | None = None  # Provider-assigned tool call ID for pairing results
    function_call: Dict[str, Any] | None = None  # For function calling
    tool_calls: list[ToolCall] | None = None  # Tool calls from assistant
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to canonical dictionary format for storage/interchange.

        Only includes non-None/non-empty optional fields to keep output clean.

        Returns:
            Dictionary with ``role``, ``content``, and any present optional fields.
        """
        d: Dict[str, Any] = {
            "role": self.role,
            "content": self.content,
        }
        if self.name is not None:
            d["name"] = self.name
        if self.tool_call_id is not None:
            d["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            d["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        if self.function_call is not None:
            d["function_call"] = self.function_call
        if self.metadata:
            d["metadata"] = self.metadata
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LLMMessage":
        """Create an LLMMessage from a dictionary.

        Handles both the new canonical format (with ``tool_calls`` as list of
        dicts) and the legacy format (without ``tool_calls``/``function_call``).

        Args:
            data: Dictionary with ``role`` (required), ``content``, and
                optional ``name``, ``tool_calls``, ``function_call``, ``metadata``.

        Returns:
            LLMMessage instance.
        """
        tool_calls = None
        if data.get("tool_calls"):
            tool_calls = [ToolCall.from_dict(tc) for tc in data["tool_calls"]]
        return cls(
            role=data["role"],
            content=data.get("content", ""),
            name=data.get("name"),
            tool_call_id=data.get("tool_call_id"),
            tool_calls=tool_calls,
            function_call=data.get("function_call"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class LLMResponse:
    """Response from LLM.

    Standard response format returned by all LLM providers. Contains the
    generated content along with metadata about token usage, cost, and
    completion status.

    Attributes:
        content: Generated text content
        model: Model identifier that generated the response
        finish_reason: Why generation stopped, on the canonical vocabulary
            'stop' / 'length' / 'tool_calls' / 'function_call'. Every provider
            reports these tokens: OpenAI and Ollama emit them natively, and the
            Claude-family providers (Anthropic, Bedrock) normalize their raw
            stop reason onto them via ``normalize_claude_stop_reason`` — with
            the raw value preserved on ``metadata['raw_finish_reason']``.
        truncated: ``True`` when the provider cut generation off at the token
            budget (Anthropic ``stop_reason == "max_tokens"``, OpenAI/Ollama
            ``finish_reason``/``done_reason == "length"``, Bedrock
            ``stopReason == "max_tokens"``). A truncated response is
            **incomplete** — most dangerously, a truncated ``tool_calls`` turn
            carries partial/invalid arguments that look well-formed. Providers
            set this consistently so a consumer can honor it without knowing
            each provider's stop-reason vocabulary.
        usage: Token usage stats (prompt_tokens, completion_tokens, total_tokens)
        function_call: Function call data if model requested tool use
        metadata: Provider-specific metadata
        created_at: Response timestamp
        cost_usd: Estimated cost in USD for this request
        cumulative_cost_usd: Running total cost for conversation

    Example:
        ```python
        from dataknobs_llm import create_llm_provider

        llm = create_llm_provider("openai", model="gpt-4")
        response = await llm.complete("What is Python?")

        # Access response data
        print(response.content)
        # => "Python is a high-level programming language..."

        # Check token usage
        print(f"Tokens used: {response.usage['total_tokens']}")
        # => Tokens used: 87

        # Monitor costs
        if response.cost_usd:
            print(f"Cost: ${response.cost_usd:.4f}")
            print(f"Total: ${response.cumulative_cost_usd:.4f}")

        # Check completion status
        if response.finish_reason == "length":
            print("Response truncated due to max_tokens limit")
        ```

    See Also:
        LLMMessage: Request message format
        LLMStreamResponse: Streaming response format
    """
    content: str
    model: str
    finish_reason: str | None = None  # 'stop', 'length', 'function_call', 'tool_calls'
    truncated: bool = False  # provider cut generation off at the token budget
    usage: Dict[str, int] | None = None  # tokens used
    function_call: Dict[str, Any] | None = None  # Legacy single function call
    tool_calls: list["ToolCall"] | None = None  # List of tool calls (preferred)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    # Cost tracking (optional enhancement for DynaBot)
    cost_usd: float | None = None  # Estimated cost in USD
    cumulative_cost_usd: float | None = None  # Running total for conversation


@dataclass
class LLMStreamResponse:
    r"""Streaming response from LLM.

    Represents a single chunk in a streaming LLM response. Streaming
    allows displaying generated text incrementally as it's produced,
    providing better user experience for long responses.

    Attributes:
        delta: Incremental content for this chunk (not cumulative)
        is_final: True if this is the last chunk in the stream
        finish_reason: Why generation stopped (only set on final chunk)
        truncated: ``True`` on the final chunk when the provider cut generation
            off at the token budget (see :class:`LLMResponse.truncated`). Only
            set on the final chunk.
        usage: Token usage stats (only set on final chunk)
        tool_calls: Tool calls requested by the model (only set on final chunk)
        model: Model identifier (only set on final chunk)
        metadata: Additional chunk metadata

    Example:
        ```python
        from dataknobs_llm import create_llm_provider

        llm = create_llm_provider("openai", model="gpt-4")

        # Stream and display in real-time
        async for chunk in llm.stream_complete("Write a poem"):
            print(chunk.delta, end="", flush=True)

            if chunk.is_final:
                print(f"\n\nFinished: {chunk.finish_reason}")
                print(f"Tokens: {chunk.usage['total_tokens']}")

        # Accumulate full response
        full_text = ""
        chunks_received = 0

        async for chunk in llm.stream_complete("Explain Python"):
            full_text += chunk.delta
            chunks_received += 1

            # Optional: show progress
            if chunks_received % 10 == 0:
                print(f"Received {chunks_received} chunks...")

        print(f"\nComplete response ({len(full_text)} chars)")
        print(full_text)
        ```

    See Also:
        LLMResponse: Non-streaming response format
        AsyncLLMProvider.stream_complete: Streaming method
    """
    delta: str  # Incremental content
    is_final: bool = False
    finish_reason: str | None = None
    truncated: bool = False  # provider cut generation off at the token budget
    usage: Dict[str, int] | None = None
    tool_calls: list["ToolCall"] | None = None  # Only set on final chunk
    model: str | None = None  # Only set on final chunk
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LLMConfig(StructuredConfig):
    """Configuration for LLM operations.

    Comprehensive configuration for LLM providers with 20+ parameters
    controlling generation, rate limiting, function calling, and more.
    Works seamlessly with both direct instantiation and dataknobs Config objects.

    This class supports:
    - All major LLM providers (OpenAI, Anthropic, Ollama, HuggingFace)
    - Generation parameters (temperature, max_tokens, top_p, etc.)
    - Embedding configuration (dimensions)
    - Function/tool calling configuration
    - Streaming with callbacks
    - Rate limiting and retry logic
    - Provider-specific options via options dict

    Note:
        Generation parameters (``temperature``, ``top_p``, ``max_tokens``,
        etc.) default to ``None``, meaning "not set — let the provider API
        apply its own default."  Only explicitly supplied values are sent
        to the provider (see :meth:`generation_params`).

    Example:
        ```python
        from dataknobs_llm.llm.base import LLMConfig, CompletionMode

        # Basic configuration — temperature is explicitly set here;
        # omitting it would let the provider use its own default.
        config = LLMConfig(
            provider="openai",
            model="gpt-4",
            api_key="sk-...",
            temperature=0.7,
            max_tokens=500
        )

        # Creative writing config
        creative_config = LLMConfig(
            provider="anthropic",
            model="claude-3-sonnet",
            temperature=1.2,
            top_p=0.95,
            max_tokens=2000
        )

        # Deterministic config for testing
        test_config = LLMConfig(
            provider="openai",
            model="gpt-4",
            temperature=0.0,
            seed=42,  # Reproducible outputs
            max_tokens=100
        )

        # Function calling config
        function_config = LLMConfig(
            provider="openai",
            model="gpt-4",
            functions=[{
                "name": "search_docs",
                "description": "Search documentation",
                "parameters": {"type": "object", "properties": {...}}
            }],
            function_call="auto"
        )

        # Streaming with callback
        def on_chunk(chunk):
            print(chunk.delta, end="")

        streaming_config = LLMConfig(
            provider="openai",
            model="gpt-4",
            stream=True,
            stream_callback=on_chunk
        )

        # From dictionary (Config compatibility)
        config_dict = {
            "provider": "ollama",
            "model": "llama2",
            "type": "llm",  # Config metadata (ignored)
            "temperature": 0.8
        }
        config = LLMConfig.from_dict(config_dict)

        # Clone with overrides
        new_config = config.clone(temperature=1.0, max_tokens=1000)
        ```

    See Also:
        normalize_llm_config: Convert various formats to LLMConfig
        CompletionMode: Available completion modes
    """

    # ``api_key`` is the only credential field. Naming it here makes the
    # ``StructuredConfig`` redacting repr mask it as ``'***'`` so the key
    # never reaches logs via ``repr(config)`` or an f-string. ``api_base`` /
    # ``user_id`` are not secrets and are shown verbatim.
    #
    # Note: scalar redaction is by field *name*, not by parsing the value, so a
    # credential embedded in an ``api_base`` URL (``https://user:token@host``)
    # would NOT be masked — and ``register_sensitive_interior_key`` cannot help,
    # as it masks interior *dict keys*, not substrings of a scalar string. Put
    # the secret in ``api_key``; do not embed credentials in ``api_base``.
    _SENSITIVE_FIELDS: ClassVar[frozenset[str]] = frozenset({"api_key"})

    provider: str  # 'openai', 'anthropic', 'ollama', etc.
    model: str  # Model name/identifier
    api_key: str | None = None
    api_base: str | None = None  # Custom API endpoint

    # Generation parameters — None means "not set, use provider default"
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    stop_sequences: List[str] | None = None

    # Mode settings
    mode: CompletionMode = CompletionMode.CHAT
    system_prompt: str | None = None
    response_format: str | None = None  # 'text' or 'json'

    # Function calling
    functions: List[Dict[str, Any]] | None = None
    function_call: Union[str, Dict[str, str]] | None = None  # 'auto', 'none', or specific function

    # Streaming
    stream: bool = False
    stream_callback: Callable[[LLMStreamResponse], None] | None = None

    # Rate limiting
    rate_limit: int | None = None  # Requests per minute
    retry_count: int = 3
    retry_delay: float = 1.0
    timeout: float = 60.0

    # Advanced settings
    seed: int | None = None  # For reproducibility
    logit_bias: Dict[str, float] | None = None
    user_id: str | None = None

    # Provider-specific options
    options: Dict[str, Any] = field(default_factory=dict)

    # Embedding-specific settings
    dimensions: int | None = None  # Vector dimensions for embedding models

    # Capability overrides — when set, these override the provider's
    # auto-detected capabilities.  Accepts string values matching
    # ModelCapability enum names (e.g. "json_mode", "function_calling").
    capabilities: List[str] | None = None

    # Model-constraint overrides — when set, overlaid onto the provider's
    # auto-detected ModelConstraints (request-shape rules the model family
    # enforces).  Loose dict shape mirroring ``capabilities``: resolved to a
    # typed ModelConstraints at runtime, never stored typed.  Keys:
    # "rejected_params" (list[str]), "accepts_inline_system" (bool),
    # "max_tokens_ceiling" (int | None).  Lets a consumer declare/withdraw a
    # constraint without a dataknobs release (e.g. a future model family
    # gaining or dropping a rejected param).
    constraints: Dict[str, Any] | None = None

    # ``from_dict`` / ``to_dict`` are inherited from ``StructuredConfig``:
    #  - ``from_dict`` ignores unknown keys (so a dataknobs ``Config``'s
    #    ``type`` / ``name`` / ``factory`` are dropped automatically) and
    #    coerces the ``mode`` enum from its string value — exactly what the
    #    old hand-rolled version did, with no per-class code.
    #  - ``to_dict`` is the symmetric in-process projection (keeps ``Enum``
    #    members and round-trips). ``to_json_dict`` renders enums as their
    #    ``.value`` for a JSON-safe dict. Neither is redacted (display-only
    #    redaction lives in the repr), so a round-trip preserves ``api_key``.
    # ``clone`` and ``generation_params`` below are LLM-specific and retained.

    def clone(self, **overrides: Any) -> "LLMConfig":
        """Create a copy of this config with optional overrides.

        This method is useful for creating runtime configuration variations
        without mutating the original config. All dataclass fields can be
        overridden via keyword arguments.

        Args:
            **overrides: Field values to override in the cloned config

        Returns:
            New LLMConfig instance with overrides applied

        Example:
            >>> base_config = LLMConfig(provider="openai", model="gpt-4", temperature=0.7)
            >>> creative_config = base_config.clone(temperature=1.2, max_tokens=500)
        """
        from dataclasses import replace
        return replace(self, **overrides)

    def generation_params(self) -> Dict[str, Any]:
        """Return only explicitly-set generation parameters.

        Providers use this to build API requests without sending
        unnecessary defaults. Parameters left as None are omitted,
        letting each provider API apply its own default.

        Returns:
            Dictionary of generation parameter names to their values.
            Only includes parameters that were explicitly set (non-None).
        """
        params: Dict[str, Any] = {}
        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.top_p is not None:
            params["top_p"] = self.top_p
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens
        if self.frequency_penalty is not None:
            params["frequency_penalty"] = self.frequency_penalty
        if self.presence_penalty is not None:
            params["presence_penalty"] = self.presence_penalty
        if self.stop_sequences is not None:
            params["stop_sequences"] = self.stop_sequences
        if self.seed is not None:
            params["seed"] = self.seed
        return params


#: Maps the Claude-family provider-native stop-reason vocabulary onto the
#: canonical :attr:`LLMResponse.finish_reason` tokens (``'stop'`` / ``'length'``
#: / ``'tool_calls'``) the docstring advertises. The native Anthropic Messages
#: API and Bedrock Converse share this vocabulary **verbatim** (Bedrock runs
#: Claude), so both providers normalize through this one table and
#: ``finish_reason`` reads identically regardless of which endpoint served the
#: model. The raw value is preserved on ``metadata['raw_finish_reason']``. An
#: unmapped value passes through unchanged.
CLAUDE_STOP_REASON_NORMALIZATION: Dict[str, str] = {
    "max_tokens": "length",
    "end_turn": "stop",
    "stop_sequence": "stop",
    "tool_use": "tool_calls",
}

#: Claude-family stop-reason tokens that mean generation was cut off at the
#: token budget (the response is incomplete — see :attr:`LLMResponse.truncated`).
CLAUDE_TRUNCATION_STOP_REASONS: frozenset[str] = frozenset({"max_tokens"})


def normalize_claude_stop_reason(
    raw_stop_reason: str | None,
) -> tuple[str | None, bool, Dict[str, Any]]:
    """Normalize a Claude-family stop reason (Anthropic / Bedrock Converse).

    Anthropic and Bedrock (Claude-on-Bedrock) emit the identical stop-reason
    vocabulary, so both route through this single helper to normalize
    ``finish_reason`` onto the canonical tokens and detect token-budget
    truncation — keeping the two providers from drifting.

    Args:
        raw_stop_reason: The provider-native stop reason (Anthropic
            ``stop_reason`` / Bedrock ``stopReason``), or ``None``.

    Returns:
        ``(finish_reason, truncated, metadata)`` where ``finish_reason`` is the
        normalized canonical token (raw value passes through when unmapped),
        ``truncated`` is ``True`` for a token-budget cut-off, and ``metadata``
        carries ``raw_finish_reason`` **only** when normalization changed the
        value (so a caller needing the exact provider token can still read it).
    """
    if raw_stop_reason is None:
        return None, False, {}
    finish_reason = CLAUDE_STOP_REASON_NORMALIZATION.get(
        raw_stop_reason, raw_stop_reason
    )
    metadata: Dict[str, Any] = {}
    if raw_stop_reason != finish_reason:
        metadata["raw_finish_reason"] = raw_stop_reason
    truncated = raw_stop_reason in CLAUDE_TRUNCATION_STOP_REASONS
    return finish_reason, truncated, metadata


def normalize_llm_config(config: Union["LLMConfig", Config, Dict[str, Any]]) -> "LLMConfig":
    """Normalize various config formats to LLMConfig.

    This helper function accepts LLMConfig instances, dataknobs Config objects,
    or plain dictionaries and returns a standardized LLMConfig instance.

    Args:
        config: Configuration as LLMConfig, Config object, or dictionary

    Returns:
        LLMConfig instance

    Raises:
        TypeError: If config type is not supported
    """
    # Already an LLMConfig instance
    if isinstance(config, LLMConfig):
        return config

    # Dictionary (possibly from Config.get())
    if isinstance(config, dict):
        return LLMConfig.from_dict(config)

    # dataknobs Config object - try to get the config dict
    # We check for the get method to identify Config objects
    if hasattr(config, 'get') and hasattr(config, 'get_types'):
        # It's a Config object, extract the llm configuration
        # Try to get first llm config, or fall back to first available type
        try:
            config_dict = config.get('llm', 0)
        except Exception as e:
            # If no 'llm' type, try to get first available config of any type
            types = config.get_types()
            if types:
                config_dict = config.get(types[0], 0)
            else:
                raise ValueError("Config object has no configurations") from e

        return LLMConfig.from_dict(config_dict)

    raise TypeError(
        f"Unsupported config type: {type(config).__name__}. "
        f"Expected LLMConfig, Config, or dict."
    )


class LLMProvider(ABC):
    """Base LLM provider interface."""

    def __init__(
        self,
        config: Union[LLMConfig, Config, Dict[str, Any]],
        prompt_builder: Union[PromptBuilder, AsyncPromptBuilder] | None = None
    ):
        """Initialize provider with configuration.

        Args:
            config: Configuration as LLMConfig, dataknobs Config object, or dict
            prompt_builder: Optional prompt builder for integrated prompting
        """
        self.config = normalize_llm_config(config)
        self.prompt_builder = prompt_builder
        self._client = None
        self._is_initialized = False
        self._is_closing = False
        self._in_flight: set[asyncio.Task[Any]] = set()

    def _validate_prompt_builder(self, expected_type: type) -> None:
        """Validate that prompt builder is configured and of correct type.

        Args:
            expected_type: Expected builder type (PromptBuilder or AsyncPromptBuilder)

        Raises:
            ValueError: If prompt_builder not configured
            TypeError: If prompt_builder is wrong type
        """
        if not self.prompt_builder:
            raise ValueError(
                "No prompt_builder configured. Pass prompt_builder to __init__() "
                "or use complete() directly with pre-rendered messages."
            )

        if not isinstance(self.prompt_builder, expected_type):
            raise TypeError(
                f"{self.__class__.__name__} requires {expected_type.__name__}, "
                f"got {type(self.prompt_builder).__name__}"
            )

    def _validate_render_params(
        self,
        prompt_type: str
    ) -> None:
        """Validate render parameters.

        Args:
            prompt_type: Type of prompt to render

        Raises:
            ValueError: If prompt_type is invalid
        """
        if prompt_type not in ("system", "user", "both"):
            raise ValueError(
                f"Invalid prompt_type: {prompt_type}. "
                f"Must be 'system', 'user', or 'both'"
            )

    def _dataknobs_error_for_status(
        self,
        status: int | None,
        message: str,
        *,
        retry_after: float | None = None,
    ) -> Exception:
        """Map an HTTP-ish status to a dataknobs exception (pure dispatch).

        The single place the vendor-error status policy lives, shared by every
        provider's ``_translate_api_error``. Each provider does the SDK-specific
        gate + extraction (which is where the vendor coupling belongs) and then
        defers the status→type decision here, so the policy stays consistent
        across providers and this method stays import-free of any vendor SDK.

        - 429 → :class:`~dataknobs_common.exceptions.RateLimitError`
          (carrying ``retry_after`` when a provider could extract it),
        - 400 → :class:`~dataknobs_common.exceptions.ValidationError`,
        - 401/403 and anything else (other status, connection, timeout, or an
          unknown ``None`` status) →
          :class:`~dataknobs_common.exceptions.OperationError`.

        Args:
            status: HTTP status code, or ``None`` when the transport gave none
                (connection error, timeout).
            message: Human-readable error message for the raised exception.
            retry_after: Seconds to wait, when known (429 only).

        Returns:
            A dataknobs exception instance (not raised — the caller raises it
            ``from`` the original SDK error to preserve ``__cause__``).
        """
        if status == 429:
            return RateLimitError(message, retry_after=retry_after)
        if status == 400:
            return ValidationError(message)
        return OperationError(message)  # 401/403 and everything else

    def _translate_api_error(self, exc: Exception) -> Exception | None:
        """Translate a raw vendor SDK error into a dataknobs exception.

        Overridden per provider to do the SDK-specific gate (is this *my*
        SDK's error?) and status extraction, then defer the status→type
        policy to :meth:`_dataknobs_error_for_status`. The base default does
        no translation (returns ``None``), so a provider with no vendor error
        taxonomy inherits "never translate" without having to opt out.

        Returning ``None`` is the passthrough contract: a non-vendor exception
        (a bug in our own code, or a domain error like
        :class:`~dataknobs_llm.exceptions.ToolsNotSupportedError`) is left for
        the caller to re-raise unchanged, never masked as an API error.
        """
        return None

    def _raise_translated(self, exc: Exception) -> NoReturn:
        """Raise the dataknobs translation of *exc*, else re-raise it unchanged.

        The shared choke point behind every provider's vendor call sites: a
        vendor SDK error is raised as its dataknobs type ``from`` the original
        (preserving ``__cause__``); anything :meth:`_translate_api_error`
        does not recognize is re-raised as-is (the passthrough contract).
        Always call from inside an ``except`` block.
        """
        translated = self._translate_api_error(exc)
        if translated is None:
            raise exc
        raise translated from exc

    async def _call_api(self, factory: Callable[[], Awaitable[Any]]) -> Any:
        """Await a vendor SDK call, translating vendor errors (choke point).

        A single try/except around a ``create`` / HTTP await so the
        vendor→dataknobs translation lives in one place for the non-streaming
        call sites. Non-vendor errors propagate unchanged.
        """
        try:
            return await factory()
        except Exception as exc:
            self._raise_translated(exc)

    async def _iter_translated(
        self, stream: AsyncIterator[_StreamItem]
    ) -> AsyncIterator[_StreamItem]:
        """Yield from a vendor stream, translating errors raised mid-iteration.

        The streaming half of the choke point: a rate-limit, throttle, or
        connection drop surfacing *during* iteration — not just at stream
        creation — is translated to a dataknobs exception, so a consumer never
        sees a raw vendor SDK error from the streaming path. Non-vendor errors
        propagate unchanged.

        A transform error in the *consumer's* loop body is outside this
        ``try`` — it reaches the generator as ``GeneratorExit`` via ``aclose``,
        which is not an :class:`Exception` — so it is never mistranslated.
        """
        try:
            async for item in stream:
                yield item
        except Exception as exc:
            self._raise_translated(exc)

    @staticmethod
    def _retry_after_from_headers(headers: Any) -> float | None:
        """Best-effort ``retry-after`` seconds from a header mapping.

        Accepts anything with a ``.get(key)`` accessor (an SDK response's
        ``headers`` mapping — anthropic, openai, and aiohttp all expose one).
        Per RFC 7231 the value may be either a non-negative number of seconds
        or an HTTP-date; both forms are parsed (an HTTP-date is converted to
        seconds-from-now, floored at ``0.0``). Returns ``None`` when the
        mapping is absent, the header is missing, or the value parses as
        neither form.
        """
        if not headers:
            return None
        get = getattr(headers, "get", None)
        if get is None:
            return None
        raw = get("retry-after")
        if not raw:
            return None
        try:
            return float(raw)
        except (TypeError, ValueError):
            pass
        # RFC 7231 HTTP-date form, e.g. "Wed, 21 Oct 2025 07:28:00 GMT".
        try:
            when = parsedate_to_datetime(str(raw))
        except (TypeError, ValueError):
            return None
        if when is None:  # older Pythons return None instead of raising
            return None
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the LLM client."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the LLM client."""
        pass

    @abstractmethod
    async def validate_model(self) -> bool:
        """Validate that the model is available."""
        pass

    def get_capabilities(self) -> List[ModelCapability]:
        """Get model capabilities.

        Template method: calls :meth:`_detect_capabilities` (subclass hook),
        filters out ``None`` values, then applies config overrides via
        :meth:`_resolve_capabilities`.

        Subclasses override :meth:`_detect_capabilities` instead of this
        method.
        """
        detected = self._detect_capabilities()
        detected = [c for c in detected if c is not None]
        return self._resolve_capabilities(detected)

    @abstractmethod
    def _detect_capabilities(self) -> List[ModelCapability]:
        """Auto-detect capabilities for this provider/model.

        Subclasses return their best-effort capability list.
        The base class handles config overrides and None filtering.
        """
        pass

    def _resolve_capabilities(
        self, detected: List[ModelCapability]
    ) -> List[ModelCapability]:
        """Return config-declared capabilities if set, else *detected*.

        Providers should call this at the end of their
        ``get_capabilities()`` implementation so that environment configs
        can override auto-detected capabilities.
        """
        if self.config.capabilities is not None:
            resolved: list[ModelCapability] = []
            for name in self.config.capabilities:
                cap = CAPABILITY_NAMES.get(name)
                if cap is not None:
                    resolved.append(cap)
                else:
                    logger.warning("Unknown capability name in config: %s", name)
            return resolved
        return detected

    def get_constraints(
        self, config: LLMConfig | None = None
    ) -> ModelConstraints:
        """Get resolved request-shape constraints for this provider/model.

        Template method mirroring :meth:`get_capabilities`: calls
        :meth:`_detect_constraints` (provider hook), then applies any
        ``LLMConfig.constraints`` override via :meth:`_resolve_constraints`.
        Orthogonal to :meth:`get_capabilities` — capabilities are the feature
        set, constraints are the request-shape rules the model family enforces
        (see :class:`ModelConstraints`).

        Args:
            config: The config whose ``model`` + ``constraints`` drive
                detection and resolution. Defaults to ``self.config``. Pass a
                per-call runtime config (with ``config_overrides`` applied) so
                a call that overrides the model to a **different family** gets
                that family's constraints — the request-param drop is then
                computed for the model actually sent, not the configured
                default. (Unlike :meth:`get_capabilities`, which is a
                provider-level query keyed off ``self.config``; constraints are
                consumed per-request, so they honor the per-call model.)
        """
        cfg = config if config is not None else self.config
        return self._resolve_constraints(self._detect_constraints(cfg), cfg)

    def _detect_constraints(self, config: LLMConfig) -> ModelConstraints:
        """Auto-detect request-shape constraints for *config*'s model.

        Default: no constraints (permissive — the base assumes a family that
        accepts every sampling param and an inline system message).  Providers
        whose model families reject request params or forbid inline system
        messages override this with family string-matching on ``config.model``,
        mirroring :meth:`_detect_capabilities`.

        Args:
            config: The config whose ``model`` is matched. This is the per-call
                runtime config when called through :meth:`get_constraints`
                with a per-call override, so detection reflects the model
                actually being sent.
        """
        return ModelConstraints()

    def _resolve_constraints(
        self, detected: ModelConstraints, config: LLMConfig
    ) -> ModelConstraints:
        """Overlay ``config.constraints`` onto *detected*, if set.

        The config-override *mechanism* mirrors :meth:`_resolve_capabilities`
        (a loose ``LLMConfig`` field resolved at runtime, so a family rule can
        be declared or withdrawn without a dataknobs release). The *merge
        semantics differ deliberately*: capabilities **replace** the detected
        list wholesale, whereas constraints **overlay per field** — an absent
        override key keeps the detected value (see
        :meth:`ModelConstraints.with_overrides`). So
        ``{"accepts_inline_system": true}`` leaves the auto-detected
        ``rejected_params`` in force rather than resetting the whole
        structure. Per-field overlay is the better semantic for constraints: a
        consumer typically wants to adjust one rule, not restate every rule the
        family enforces.

        Args:
            detected: The auto-detected constraints for the model.
            config: The config whose ``constraints`` override is overlaid.
        """
        override = config.constraints
        if not override:
            return detected
        return detected.with_overrides(override)

    def _check_ready(self) -> None:
        """Raise if provider is not ready for requests.

        Raises:
            ResourceError: If provider is not initialized or is closing.
        """
        if not self._is_initialized:
            raise ResourceError("Provider not initialized. Call initialize() first.")
        if self._is_closing:
            raise ResourceError("Provider is closing. No new requests accepted.")

    def _analyze_response(self, response: "LLMResponse") -> "LLMResponse":
        """Post-process a response after provider builds it.

        Detects thinking-only responses (reasoning models that consume tokens
        on ``<think>`` blocks but return empty visible content) and annotates
        them with ``metadata["thinking_only"] = True``.

        Subclasses may override to add provider-specific analysis but should
        call ``super()._analyze_response(response)`` to preserve base checks.

        Args:
            response: The LLM response to analyze.

        Returns:
            The (possibly annotated) response.
        """
        # Detect thinking-only: empty visible content, no tool calls,
        # but significant completion token usage (> 50 tokens).
        if (
            not response.content
            and not response.tool_calls
            and response.usage
            and response.usage.get("completion_tokens", 0) > 50
        ):
            response.metadata["thinking_only"] = True
            logger.warning(
                "Thinking-only response detected: %d completion tokens, "
                "empty content (model: %s)",
                response.usage.get("completion_tokens", 0),
                response.model,
            )
        self._warn_if_truncated(response)
        return response

    def _warn_if_truncated(
        self, response: "LLMResponse | LLMStreamResponse"
    ) -> None:
        """Log when a response was cut off at the token budget.

        Shared by the complete path (via :meth:`_analyze_response`) and the
        streaming final-chunk assembly of every provider that populates
        :attr:`LLMResponse.truncated`, so the warning is emitted once and
        consistently regardless of provider. Detection of *whether* a response
        is truncated is per-provider (each knows its own stop-reason
        vocabulary); this only decides how loudly to surface the flag.

        A truncated **tool-call** turn is the dangerous case — the partial
        arguments look well-formed but are incomplete — so it warns; a plain
        truncated text turn logs at ``info``. Accepts either response type
        (both carry ``truncated``/``tool_calls``/``model``).
        """
        if not getattr(response, "truncated", False):
            return
        if response.tool_calls:
            logger.warning(
                "Response truncated at the token budget mid tool-call "
                "(model=%s): the tool call arguments are incomplete and will "
                "likely fail validation — raise max_tokens or shorten the "
                "request.",
                response.model,
            )
        else:
            logger.info(
                "Response truncated at the token budget (model=%s): the "
                "output is incomplete.",
                response.model,
            )

    @staticmethod
    def _attach_legacy_function_call(response: "LLMResponse") -> "LLMResponse":
        """Surface the first tool call as the legacy ``function_call`` dict.

        Backward-compat shim for the deprecated :meth:`function_call` entry
        point: the modern path returns ``tool_calls``, but legacy callers read
        the single ``function_call`` dict. Providers whose ``function_call``
        override needs this shape call it, then route the result through
        :meth:`_analyze_response` — the shared post-processing choke point that
        preserves ``truncated`` / ``metadata`` and fires
        :meth:`_warn_if_truncated`. Mutates and returns ``response``.
        """
        if response.tool_calls:
            tc = response.tool_calls[0]
            response.function_call = {"name": tc.name, "arguments": tc.parameters}
        return response

    @property
    def is_initialized(self) -> bool:
        """Check if provider is initialized."""
        return self._is_initialized

    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


class ConfigOverrideMixin:
    """Mixin providing config override functionality for LLM providers.

    This mixin provides shared functionality for handling per-request config
    overrides, presets, and callbacks. Both AsyncLLMProvider and SyncLLMProvider
    inherit from this mixin.

    Features:
        - Per-request config overrides (model, temperature, etc.)
        - Named presets for common override combinations
        - Callback hooks for logging/metrics
        - Options dict merging
    """

    # Supported fields for config overrides (base set)
    ALLOWED_CONFIG_OVERRIDES = {
        # Core generation parameters
        "model", "temperature", "max_tokens", "top_p", "stop_sequences", "seed",
        # Provider-specific parameters
        "presence_penalty", "frequency_penalty", "logit_bias", "response_format",
        # Function calling (dynamic)
        "functions", "function_call",
        # Provider-specific options dict
        "options",
    }

    # Override presets registry (class-level, shared across all providers)
    _override_presets: Dict[str, Dict[str, Any]] = {}

    # Override event callbacks (class-level)
    _override_callbacks: List[Callable[[Any, Dict[str, Any], LLMConfig], None]] = []

    @classmethod
    def register_preset(cls, name: str, overrides: Dict[str, Any]) -> None:
        """Register a named override preset.

        Presets allow you to define common override combinations that can be
        referenced by name instead of repeating the same overrides.

        Args:
            name: Preset name (e.g., "creative", "precise", "fast")
            overrides: Dictionary of override values

        Example:
            >>> AsyncLLMProvider.register_preset("creative", {
            ...     "temperature": 1.2,
            ...     "top_p": 0.95,
            ...     "presence_penalty": 0.5
            ... })
            >>> response = await provider.complete(
            ...     "Write a poem",
            ...     config_overrides={"preset": "creative"}
            ... )
        """
        cls._override_presets[name] = overrides.copy()

    @classmethod
    def on_override_applied(
        cls,
        callback: Callable[[Any, Dict[str, Any], LLMConfig], None]
    ) -> None:
        """Register a callback for when overrides are applied.

        Use this for logging, metrics collection, or auditing override usage.
        Callbacks receive the provider instance, the applied overrides dict,
        and the resulting runtime config.

        Args:
            callback: Function(provider, overrides, runtime_config) -> None

        Example:
            >>> def log_overrides(provider, overrides, runtime_config):
            ...     print(f"Overrides applied: {overrides}")
            ...     print(f"Runtime model: {runtime_config.model}")
            ...
            >>> AsyncLLMProvider.on_override_applied(log_overrides)
        """
        cls._override_callbacks.append(callback)

    @classmethod
    def clear_override_callbacks(cls) -> None:
        """Clear all registered override callbacks."""
        cls._override_callbacks.clear()

    @classmethod
    def get_preset(cls, name: str) -> Dict[str, Any] | None:
        """Get a registered override preset by name.

        Args:
            name: Preset name

        Returns:
            Preset overrides dict, or None if not found
        """
        return cls._override_presets.get(name)

    @classmethod
    def list_presets(cls) -> List[str]:
        """List all registered preset names.

        Returns:
            List of preset names
        """
        return list(cls._override_presets.keys())

    def _validate_config_overrides(
        self,
        overrides: Dict[str, Any] | None
    ) -> None:
        """Validate that config override fields are supported.

        Args:
            overrides: Dictionary of config overrides to validate

        Raises:
            ValueError: If overrides contains unsupported fields
        """
        if not overrides:
            return

        # Allow "preset" as a special key for named presets
        allowed = self.ALLOWED_CONFIG_OVERRIDES | {"preset"}
        invalid = set(overrides.keys()) - allowed
        if invalid:
            raise ValueError(
                f"Unsupported config overrides: {invalid}. "
                f"Allowed fields: {self.ALLOWED_CONFIG_OVERRIDES}"
            )

    def _expand_preset(
        self,
        overrides: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Expand preset reference to actual override values.

        If overrides contains a 'preset' key, replaces it with the
        registered preset values. Explicit overrides take precedence
        over preset values.

        Args:
            overrides: Override dict that may contain a preset reference

        Returns:
            Expanded overrides dict

        Raises:
            ValueError: If preset is not registered
        """
        if "preset" not in overrides:
            return overrides

        preset_name = overrides["preset"]
        preset_values = self.get_preset(preset_name)
        if preset_values is None:
            raise ValueError(
                f"Unknown preset: '{preset_name}'. "
                f"Available presets: {self.list_presets()}"
            )

        # Preset values as base, explicit overrides take precedence
        expanded = preset_values.copy()
        for key, value in overrides.items():
            if key != "preset":
                expanded[key] = value

        return expanded

    def _merge_options(
        self,
        base_options: Dict[str, Any],
        override_options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Deep merge options dicts.

        Args:
            base_options: Base options from config
            override_options: Override options to merge

        Returns:
            Merged options dict
        """
        merged = base_options.copy()
        merged.update(override_options)
        return merged

    def _notify_override_callbacks(
        self,
        overrides: Dict[str, Any],
        runtime_config: LLMConfig
    ) -> None:
        """Notify registered callbacks about applied overrides.

        Args:
            overrides: The overrides that were applied
            runtime_config: The resulting runtime config
        """
        for callback in self._override_callbacks:
            try:
                callback(self, overrides, runtime_config)
            except Exception:
                # Don't let callback errors break the main flow
                pass

    def _get_runtime_config(
        self,
        config_overrides: Dict[str, Any] | None = None
    ) -> LLMConfig:
        """Get runtime config, applying overrides if provided.

        Supports:
        - Direct field overrides (model, temperature, etc.)
        - Named presets via 'preset' key
        - Deep merging of 'options' dict
        - Override callback notifications for logging/metrics

        Args:
            config_overrides: Optional overrides to apply

        Returns:
            LLMConfig to use for this request (original or cloned with overrides)
        """
        if not config_overrides:
            return self.config  # type: ignore[attr-defined]

        self._validate_config_overrides(config_overrides)

        # Expand preset if present
        expanded = self._expand_preset(config_overrides)

        # Handle options merging specially
        if "options" in expanded and self.config.options:  # type: ignore[attr-defined]
            expanded["options"] = self._merge_options(
                self.config.options,  # type: ignore[attr-defined]
                expanded["options"]
            )

        runtime_config = self.config.clone(**expanded)  # type: ignore[attr-defined]

        # Notify callbacks
        self._notify_override_callbacks(config_overrides, runtime_config)

        return runtime_config


class AsyncLLMProvider(LLMProvider, ConfigOverrideMixin):
    """Async LLM provider interface."""

    @abstractmethod
    async def complete(
        self,
        messages: Union[str, List[LLMMessage]],
        config_overrides: Dict[str, Any] | None = None,
        tools: list[Any] | None = None,
        **kwargs
    ) -> LLMResponse:
        """Generate completion asynchronously.

        Primary method for getting LLM responses. Accepts either a simple
        string prompt or a list of LLMMessage objects for multi-turn
        conversations. This is the recommended async method for most use cases.

        Args:
            messages: Either a single string prompt or a list of LLMMessage
                objects for multi-turn conversations.
            config_overrides: Optional dict to override config fields for this
                request only. Supported fields: model, temperature, max_tokens,
                top_p, stop_sequences, seed. The original config is not modified.
            tools: Optional list of Tool objects available for this completion.
                Each tool should have ``name``, ``description``, and ``schema``
                attributes. When provided, the LLM may return tool calls in the
                response. Providers that do not support tools will raise
                ``ToolsNotSupportedError``.
            **kwargs: Additional provider-specific parameters. Common options:
                - temperature (float): Sampling temperature (0.0-2.0)
                - max_tokens (int): Maximum tokens to generate
                - top_p (float): Nucleus sampling parameter (0.0-1.0)
                - stop (List[str]): Stop sequences
                - presence_penalty (float): Presence penalty (-2.0 to 2.0)
                - frequency_penalty (float): Frequency penalty (-2.0 to 2.0)

        Returns:
            LLMResponse containing generated content, usage stats, and metadata

        Raises:
            ValueError: If messages format is invalid or config_overrides contains
                unsupported fields
            ConnectionError: If API connection fails
            TimeoutError: If request exceeds timeout

        Example:
            ```python
            from dataknobs_llm import create_llm_provider
            from dataknobs_llm.llm.base import LLMMessage

            llm = create_llm_provider("openai", model="gpt-4")

            # Simple string prompt
            response = await llm.complete("What is Python?")
            print(response.content)
            # => "Python is a high-level programming language..."

            # With config overrides (switch model per-request)
            response = await llm.complete(
                "Write a haiku about coding",
                config_overrides={"model": "gpt-4-turbo", "temperature": 0.9}
            )

            # Multi-turn conversation
            messages = [
                LLMMessage(role="system", content="You are a helpful tutor"),
                LLMMessage(role="user", content="Explain recursion"),
                LLMMessage(role="assistant", content="Recursion is when..."),
                LLMMessage(role="user", content="Can you give an example?")
            ]
            response = await llm.complete(messages)

            # Check token usage
            print(f"Tokens: {response.usage['total_tokens']}")
            print(f"Cost: ${response.cost_usd:.4f}")
            ```

        See Also:
            stream_complete: Streaming version
            render_and_complete: Complete with prompt rendering
        """
        pass

    async def render_and_complete(
        self,
        prompt_name: str,
        params: Dict[str, Any] | None = None,
        prompt_type: str = "user",
        index: int = 0,
        include_rag: bool = True,
        **llm_kwargs
    ) -> LLMResponse:
        """Render prompt from library and execute LLM completion.

        This is a convenience method for one-off interactions that combines
        prompt rendering with LLM execution. For multi-turn conversations,
        use ConversationManager instead.

        Args:
            prompt_name: Name of prompt in library
            params: Parameters for template rendering
            prompt_type: Type of prompt ("system", "user", or "both")
            index: Prompt variant index (for user prompts)
            include_rag: Whether to execute RAG searches
            **llm_kwargs: Additional arguments passed to complete()

        Returns:
            LLM response

        Raises:
            ValueError: If prompt_builder not configured or invalid prompt_type
            TypeError: If prompt_builder is not AsyncPromptBuilder

        Example:
            >>> llm = OpenAIProvider(config, prompt_builder=builder)
            >>> result = await llm.render_and_complete(
            ...     "analyze_code",
            ...     params={"code": code, "language": "python"}
            ... )
        """
        # Validate
        from dataknobs_llm.prompts import AsyncPromptBuilder
        self._validate_prompt_builder(AsyncPromptBuilder)
        self._validate_render_params(prompt_type)

        # Render messages
        messages = await self._render_messages(
            prompt_name, params, prompt_type, index, include_rag
        )

        # Execute LLM
        return await self.complete(messages, **llm_kwargs)

    async def render_and_stream(
        self,
        prompt_name: str,
        params: Dict[str, Any] | None = None,
        prompt_type: str = "user",
        index: int = 0,
        include_rag: bool = True,
        **llm_kwargs
    ) -> AsyncIterator[LLMStreamResponse]:
        """Render prompt and stream LLM response.

        Same as render_and_complete() but returns streaming response.

        Args:
            prompt_name: Name of prompt in library
            params: Parameters for template rendering
            prompt_type: Type of prompt ("system", "user", or "both")
            index: Prompt variant index
            include_rag: Whether to execute RAG searches
            **llm_kwargs: Additional arguments passed to stream_complete()

        Yields:
            Streaming response chunks

        Raises:
            ValueError: If prompt_builder not configured or invalid prompt_type
            TypeError: If prompt_builder is not AsyncPromptBuilder

        Example:
            >>> async for chunk in llm.render_and_stream("analyze_code", params={"code": code}):
            ...     print(chunk.delta, end="")
        """
        # Validate
        from dataknobs_llm.prompts import AsyncPromptBuilder
        self._validate_prompt_builder(AsyncPromptBuilder)
        self._validate_render_params(prompt_type)

        # Render messages
        messages = await self._render_messages(
            prompt_name, params, prompt_type, index, include_rag
        )

        # Stream LLM response
        async for chunk in self.stream_complete(messages, **llm_kwargs):
            yield chunk

    async def _render_messages(
        self,
        prompt_name: str,
        params: Dict[str, Any] | None,
        prompt_type: str,
        index: int,
        include_rag: bool
    ) -> List[LLMMessage]:
        """Render messages from prompt library (async version).

        Args:
            prompt_name: Name of prompt in library
            params: Parameters for template rendering
            prompt_type: Type of prompt ("system", "user", or "both")
            index: Prompt variant index
            include_rag: Whether to execute RAG searches

        Returns:
            List of rendered LLM messages
        """
        from dataknobs_llm.prompts import AsyncPromptBuilder
        builder: AsyncPromptBuilder = self.prompt_builder  # type: ignore

        messages: List[LLMMessage] = []
        params = params or {}

        if prompt_type in ("system", "both"):
            result = await builder.render_system_prompt(
                prompt_name, params=params, include_rag=include_rag
            )
            messages.append(LLMMessage(role="system", content=result.content))

        if prompt_type in ("user", "both"):
            result = await builder.render_user_prompt(
                prompt_name, index=index, params=params, include_rag=include_rag
            )
            messages.append(LLMMessage(role="user", content=result.content))

        return messages
        
    @abstractmethod
    async def stream_complete(
        self,
        messages: Union[str, List[LLMMessage]],
        config_overrides: Dict[str, Any] | None = None,
        tools: list[Any] | None = None,
        **kwargs
    ) -> AsyncIterator[LLMStreamResponse]:
        r"""Generate streaming completion asynchronously.

        Streams response chunks as they are generated, enabling real-time
        display of LLM output. Each chunk contains incremental content
        (delta), and the final chunk includes usage statistics and any
        tool calls requested by the model.

        Args:
            messages: Either a single string prompt or list of LLMMessage objects
            config_overrides: Optional dict to override config fields for this
                request only. Supported fields: model, temperature, max_tokens,
                top_p, stop_sequences, seed. The original config is not modified.
            tools: Optional list of Tool objects available for this completion.
            **kwargs: Provider-specific parameters (same as complete())

        Yields:
            LLMStreamResponse chunks containing incremental content. The final
            chunk has is_final=True and includes finish_reason and usage stats.

        Raises:
            ValueError: If messages format is invalid or config_overrides contains
                unsupported fields
            ConnectionError: If API connection fails
            TimeoutError: If request exceeds timeout

        Example:
            ```python
            from dataknobs_llm import create_llm_provider

            llm = create_llm_provider("openai", model="gpt-4")

            # Stream and display in real-time
            async for chunk in llm.stream_complete("Tell me a story"):
                print(chunk.delta, end="", flush=True)

                if chunk.is_final:
                    print(f"\n\nFinished: {chunk.finish_reason}")
                    print(f"Total tokens: {chunk.usage['total_tokens']}")

            # Stream with config overrides
            async for chunk in llm.stream_complete(
                "Write a poem",
                config_overrides={"model": "gpt-4-turbo", "temperature": 1.0}
            ):
                print(chunk.delta, end="", flush=True)

            # Accumulate full response
            full_text = ""
            chunk_count = 0

            async for chunk in llm.stream_complete("Explain quantum computing"):
                full_text += chunk.delta
                chunk_count += 1

            print(f"Received {chunk_count} chunks")
            print(f"Total length: {len(full_text)} characters")
            ```

        See Also:
            complete: Non-streaming version
            render_and_stream: Stream with prompt rendering
            LLMStreamResponse: Chunk data structure
        """
        pass
        
    @abstractmethod
    async def embed(
        self,
        texts: Union[str, List[str]],
        **kwargs
    ) -> Union[List[float], List[List[float]]]:
        """Generate embeddings asynchronously.

        Converts text into dense vector representations for semantic search,
        clustering, and similarity comparison. Returns high-dimensional
        vectors (typically 768-1536 dimensions depending on model).

        Args:
            texts: Single text string or list of texts to embed
            **kwargs: Provider-specific parameters:
                - model (str): Embedding model override
                - dimensions (int): Target dimensions (if supported)

        Returns:
            Single embedding vector (List[float]) if input is a string,
            or list of vectors (List[List[float]]) if input is a list

        Raises:
            ValueError: If texts is empty or invalid
            ConnectionError: If API connection fails

        Example:
            ```python
            from dataknobs_llm import create_llm_provider
            import numpy as np

            # Create embedding provider
            llm = create_llm_provider(
                "openai",
                model="text-embedding-ada-002"
            )

            # Single text embedding
            embedding = await llm.embed("What is machine learning?")
            print(f"Dimensions: {len(embedding)}")
            # => Dimensions: 1536

            # Batch embedding
            texts = [
                "Python is a programming language",
                "JavaScript is used for web development",
                "Machine learning uses statistical methods"
            ]
            embeddings = await llm.embed(texts)
            print(f"Generated {len(embeddings)} embeddings")

            # Compute similarity
            def cosine_similarity(v1, v2):
                return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

            query_emb = await llm.embed("Tell me about ML")
            similarities = [
                cosine_similarity(query_emb, emb)
                for emb in embeddings
            ]
            most_similar_idx = np.argmax(similarities)
            print(f"Most similar: {texts[most_similar_idx]}")
            # => Most similar: Machine learning uses statistical methods

            # Store in vector database
            from dataknobs_data import database_factory
            db = database_factory.create("vector_db")
            for text, emb in zip(texts, embeddings):
                db.create({"text": text, "embedding": emb})
            ```

        See Also:
            complete: Text generation method
        """
        pass
        
    @abstractmethod
    async def function_call(
        self,
        messages: List[LLMMessage],
        functions: List[Dict[str, Any]],
        **kwargs
    ) -> LLMResponse:
        """Execute function calling asynchronously.

        .. deprecated::
            Use ``complete(tools=...)`` instead. The ``function_call()`` method
            will be removed in a future major version. All providers now support
            the ``tools`` parameter on ``complete()`` for consistent tool
            handling.

        Enables the LLM to call external functions/tools. The model decides
        which function to call based on the conversation context, and returns
        the function name and arguments in a structured format.

        Args:
            messages: Conversation messages leading up to the function call
            functions: List of function definitions in JSON Schema format.
                Each function dict must have:
                - name (str): Function name
                - description (str): What the function does
                - parameters (dict): JSON Schema for parameters
            **kwargs: Provider-specific parameters:
                - function_call (str|dict): 'auto', 'none', or specific function
                - temperature (float): Sampling temperature
                - max_tokens (int): Maximum response tokens

        Returns:
            LLMResponse with function_call field populated containing:
            - name (str): Function to call
            - arguments (str): JSON string of arguments

        Raises:
            ValueError: If functions format is invalid
            ConnectionError: If API connection fails

        Example:
            ```python
            from dataknobs_llm import create_llm_provider
            from dataknobs_llm.llm.base import LLMMessage
            import json

            llm = create_llm_provider("openai", model="gpt-4")

            # Define available functions
            functions = [
                {
                    "name": "search_docs",
                    "description": "Search documentation for information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Max results"
                            }
                        },
                        "required": ["query"]
                    }
                },
                {
                    "name": "execute_code",
                    "description": "Execute Python code",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {"type": "string"}
                        },
                        "required": ["code"]
                    }
                }
            ]

            # Ask question that requires function
            messages = [
                LLMMessage(
                    role="user",
                    content="Search for information about async/await in Python"
                )
            ]

            # Model decides to call function
            response = await llm.function_call(messages, functions)

            if response.function_call:
                func_name = response.function_call["name"]
                func_args = json.loads(response.function_call["arguments"])

                print(f"Function: {func_name}")
                print(f"Arguments: {func_args}")
                # => Function: search_docs
                # => Arguments: {'query': 'async/await Python', 'limit': 5}

                # Execute function
                results = search_docs(**func_args)

                # Add function result to conversation
                messages.append(LLMMessage(
                    role="function",
                    name=func_name,
                    content=json.dumps(results)
                ))

                # Get final response
                final = await llm.complete(messages)
                print(final.content)
            ```

        See Also:
            complete: Standard completion without functions
            dataknobs_llm.tools: Tool abstraction framework
        """
        pass
        
    def __enter__(self) -> None:
        """Prevent sync context manager usage on async providers."""
        raise TypeError(
            "Use 'async with' for AsyncLLMProvider, not 'with'"
        )

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """Sync exit — unreachable since __enter__ raises."""

    async def initialize(self) -> None:
        """Initialize the async LLM client."""
        self._is_initialized = True

    async def close(self) -> None:
        """Close provider, cancelling in-flight requests.

        Safe to call multiple times (idempotent). Cancels any tracked
        in-flight requests before closing the underlying HTTP client.
        """
        if self._is_closing:
            return
        self._is_closing = True

        # Cancel in-flight requests
        if self._in_flight:
            logger.info("Cancelling %d in-flight requests", len(self._in_flight))
            for task in self._in_flight:
                task.cancel()
            await asyncio.gather(*self._in_flight, return_exceptions=True)
            self._in_flight.clear()

        # Close the HTTP client (subclass hook)
        await self._close_client()

        self._is_initialized = False
        self._is_closing = False

    async def _close_client(self) -> None:
        """Close the underlying HTTP client.

        Override in subclasses to close provider-specific HTTP clients.
        The base implementation is a no-op.
        """

    async def _tracked_call(self, coro: Coroutine[Any, Any, Any]) -> Any:
        """Execute a coroutine while tracking it as in-flight.

        Args:
            coro: The coroutine to execute.

        Returns:
            The coroutine's return value.
        """
        task = asyncio.current_task()
        if task:
            self._in_flight.add(task)
        try:
            return await coro
        finally:
            if task:
                self._in_flight.discard(task)

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """Async context manager exit."""
        await self.close()


class SyncLLMProvider(LLMProvider, ConfigOverrideMixin):
    """Synchronous LLM provider interface."""

    @abstractmethod
    def complete(
        self,
        messages: Union[str, List[LLMMessage]],
        config_overrides: Dict[str, Any] | None = None,
        tools: list[Any] | None = None,
        **kwargs
    ) -> LLMResponse:
        """Generate completion synchronously.

        Args:
            messages: Input messages or prompt
            config_overrides: Optional dict to override config fields for this
                request only. Supported fields: model, temperature, max_tokens,
                top_p, stop_sequences, seed. The original config is not modified.
            tools: Optional list of Tool objects available for this completion.
            **kwargs: Additional parameters

        Returns:
            LLM response
        """
        pass

    def render_and_complete(
        self,
        prompt_name: str,
        params: Dict[str, Any] | None = None,
        prompt_type: str = "user",
        index: int = 0,
        include_rag: bool = True,
        **llm_kwargs
    ) -> LLMResponse:
        """Render prompt from library and execute LLM completion.

        This is a convenience method for one-off interactions that combines
        prompt rendering with LLM execution. For multi-turn conversations,
        use ConversationManager instead.

        Args:
            prompt_name: Name of prompt in library
            params: Parameters for template rendering
            prompt_type: Type of prompt ("system", "user", or "both")
            index: Prompt variant index (for user prompts)
            include_rag: Whether to execute RAG searches
            **llm_kwargs: Additional arguments passed to complete()

        Returns:
            LLM response

        Raises:
            ValueError: If prompt_builder not configured or invalid prompt_type
            TypeError: If prompt_builder is not PromptBuilder

        Example:
            >>> llm = SyncOpenAIProvider(config, prompt_builder=builder)
            >>> result = llm.render_and_complete(
            ...     "analyze_code",
            ...     params={"code": code, "language": "python"}
            ... )
        """
        # Validate
        from dataknobs_llm.prompts import PromptBuilder
        self._validate_prompt_builder(PromptBuilder)
        self._validate_render_params(prompt_type)

        # Render messages
        messages = self._render_messages(
            prompt_name, params, prompt_type, index, include_rag
        )

        # Execute LLM
        return self.complete(messages, **llm_kwargs)

    def render_and_stream(
        self,
        prompt_name: str,
        params: Dict[str, Any] | None = None,
        prompt_type: str = "user",
        index: int = 0,
        include_rag: bool = True,
        **llm_kwargs
    ) -> Iterator[LLMStreamResponse]:
        """Render prompt and stream LLM response.

        Same as render_and_complete() but returns streaming response.

        Args:
            prompt_name: Name of prompt in library
            params: Parameters for template rendering
            prompt_type: Type of prompt ("system", "user", or "both")
            index: Prompt variant index
            include_rag: Whether to execute RAG searches
            **llm_kwargs: Additional arguments passed to stream_complete()

        Yields:
            Streaming response chunks

        Raises:
            ValueError: If prompt_builder not configured or invalid prompt_type
            TypeError: If prompt_builder is not PromptBuilder

        Example:
            >>> for chunk in llm.render_and_stream("analyze_code", params={"code": code}):
            ...     print(chunk.delta, end="")
        """
        # Validate
        from dataknobs_llm.prompts import PromptBuilder
        self._validate_prompt_builder(PromptBuilder)
        self._validate_render_params(prompt_type)

        # Render messages
        messages = self._render_messages(
            prompt_name, params, prompt_type, index, include_rag
        )

        # Stream LLM response
        for chunk in self.stream_complete(messages, **llm_kwargs):
            yield chunk

    def _render_messages(
        self,
        prompt_name: str,
        params: Dict[str, Any] | None,
        prompt_type: str,
        index: int,
        include_rag: bool
    ) -> List[LLMMessage]:
        """Render messages from prompt library (sync version).

        Args:
            prompt_name: Name of prompt in library
            params: Parameters for template rendering
            prompt_type: Type of prompt ("system", "user", or "both")
            index: Prompt variant index
            include_rag: Whether to execute RAG searches

        Returns:
            List of rendered LLM messages
        """
        from dataknobs_llm.prompts import PromptBuilder
        builder: PromptBuilder = self.prompt_builder  # type: ignore

        messages: List[LLMMessage] = []
        params = params or {}

        if prompt_type in ("system", "both"):
            result = builder.render_system_prompt(
                prompt_name, params=params, include_rag=include_rag
            )
            messages.append(LLMMessage(role="system", content=result.content))

        if prompt_type in ("user", "both"):
            result = builder.render_user_prompt(
                prompt_name, index=index, params=params, include_rag=include_rag
            )
            messages.append(LLMMessage(role="user", content=result.content))

        return messages

    @abstractmethod
    def stream_complete(
        self,
        messages: Union[str, List[LLMMessage]],
        config_overrides: Dict[str, Any] | None = None,
        tools: list[Any] | None = None,
        **kwargs
    ) -> Iterator[LLMStreamResponse]:
        """Generate streaming completion synchronously.

        Args:
            messages: Input messages or prompt
            config_overrides: Optional dict to override config fields for this
                request only. Supported fields: model, temperature, max_tokens,
                top_p, stop_sequences, seed. The original config is not modified.
            tools: Optional list of Tool objects available for this completion.
            **kwargs: Additional parameters

        Yields:
            Streaming response chunks
        """
        pass
        
    @abstractmethod
    def embed(
        self,
        texts: Union[str, List[str]],
        **kwargs
    ) -> Union[List[float], List[List[float]]]:
        """Generate embeddings synchronously.
        
        Args:
            texts: Input text(s)
            **kwargs: Additional parameters
            
        Returns:
            Embedding vector(s)
        """
        pass
        
    @abstractmethod
    def function_call(
        self,
        messages: List[LLMMessage],
        functions: List[Dict[str, Any]],
        **kwargs
    ) -> LLMResponse:
        """Execute function calling synchronously.
        
        Args:
            messages: Conversation messages
            functions: Available functions
            **kwargs: Additional parameters
            
        Returns:
            Response with function call
        """
        pass
        
    def initialize(self) -> None:
        """Initialize the sync LLM client."""
        self._is_initialized = True

    def close(self) -> None:
        """Close the sync LLM client.

        Safe to call multiple times (idempotent). Calls ``_close_client()``
        for subclass-specific resource cleanup.
        """
        if self._is_closing:
            return
        self._is_closing = True
        self._close_client()
        self._is_initialized = False
        self._is_closing = False

    def _close_client(self) -> None:
        """Close the underlying HTTP client.

        Override in subclasses to close provider-specific HTTP clients.
        The base implementation is a no-op.
        """


class LLMAdapter(ABC):
    """Base adapter for converting between different LLM formats.

    Adapters translate between the standard dataknobs LLM format
    (LLMMessage, LLMResponse, LLMConfig) and provider-specific formats
    (OpenAI, Anthropic, etc.). Providers with complex multi-format APIs
    (OpenAI, Anthropic, Ollama) have corresponding adapters. Simpler
    providers (EchoProvider, HuggingFaceProvider) do not require adapters.

    This enables provider-agnostic code that works across different
    LLM APIs without modification.

    Example:
        ```python
        from dataknobs_llm.llm.base import LLMAdapter, LLMMessage, LLMResponse
        from typing import Any, List, Dict

        class MyProviderAdapter(LLMAdapter):
            \"\"\"Adapter for custom LLM provider.\"\"\"

            def adapt_messages(
                self,
                messages: List[LLMMessage],
                system_prompt: str | None = None,
            ) -> List[Dict[str, str]]:
                \"\"\"Convert to provider format.\"\"\"
                return [
                    {"role": msg.role, "content": msg.content}
                    for msg in messages
                ]

            def adapt_response(
                self,
                response: Any
            ) -> LLMResponse:
                \"\"\"Convert from provider format.\"\"\"
                return LLMResponse(
                    content=response["text"],
                    model=response["model_id"],
                    usage={
                        "total_tokens": response["tokens_used"]
                    }
                )

            def adapt_config(
                self,
                config: LLMConfig
            ) -> Dict[str, Any]:
                \"\"\"Convert config to provider format.\"\"\"
                return {
                    "model_name": config.model,
                    "temp": config.temperature,
                    "max_length": config.max_tokens
                }

            def adapt_tools(
                self,
                tools: list[Any]
            ) -> list[Dict[str, Any]]:
                \"\"\"Convert tools to provider format.\"\"\"
                return [
                    {"name": t.name, "schema": t.schema}
                    for t in tools
                ]

        # Use adapter in provider
        adapter = MyProviderAdapter()
        provider_messages = adapter.adapt_messages(
            messages, system_prompt="You are helpful.",
        )
        ```

    See Also:
        LLMProvider: Base provider interface
        dataknobs_llm.llm.providers.OpenAIAdapter: Example implementation
    """

    @abstractmethod
    def adapt_messages(
        self,
        messages: List[LLMMessage],
        system_prompt: str | None = None,
    ) -> Any:
        """Adapt messages to provider format.

        Args:
            messages: Standard LLMMessage list
            system_prompt: Optional system prompt from provider config.
                Providers that require system content as a separate API
                parameter (e.g. Anthropic) merge this with any system
                messages found in the list. Other providers may ignore it.

        Returns:
            Provider-specific message format
        """
        pass

    @abstractmethod
    def adapt_response(
        self,
        response: Any
    ) -> LLMResponse:
        """Adapt provider response to standard format.

        Args:
            response: Provider-specific response object

        Returns:
            Standard LLMResponse
        """
        pass

    @abstractmethod
    def adapt_config(
        self,
        config: LLMConfig
    ) -> Dict[str, Any]:
        """Adapt configuration to provider format.

        Args:
            config: Standard LLMConfig

        Returns:
            Provider-specific config dict
        """
        pass

    @abstractmethod
    def adapt_tools(
        self,
        tools: list[Any]
    ) -> list[Dict[str, Any]]:
        """Adapt Tool objects to provider-specific tool format.

        Args:
            tools: List of Tool objects with ``name``, ``description``,
                and ``schema`` attributes.

        Returns:
            Provider-specific tool definitions.
        """
        pass


class LLMMiddleware(Protocol):
    """Protocol for LLM middleware.

    Middleware provides hooks to transform requests before they're sent
    to the LLM and responses before they're returned to the caller.
    Useful for logging, caching, content filtering, rate limiting, etc.

    Middleware can accept configuration as LLMConfig, dataknobs Config, or dict.

    Example:
        ```python
        from dataknobs_llm.llm.base import (
            LLMMiddleware, LLMMessage, LLMResponse, LLMConfig
        )
        from typing import List, Union, Dict, Any
        import logging

        class LoggingMiddleware:
            \"\"\"Logs all LLM requests and responses.\"\"\"

            def __init__(self):
                self.logger = logging.getLogger(__name__)

            async def process_request(
                self,
                messages: List[LLMMessage],
                config: Union[LLMConfig, Config, Dict[str, Any]]
            ) -> List[LLMMessage]:
                \"\"\"Log request before sending.\"\"\"
                self.logger.info(f"Request: {len(messages)} messages")
                for msg in messages:
                    self.logger.debug(f"  {msg.role}: {msg.content[:50]}...")
                return messages

            async def process_response(
                self,
                response: LLMResponse,
                config: Union[LLMConfig, Config, Dict[str, Any]]
            ) -> LLMResponse:
                \"\"\"Log response after receiving.\"\"\"
                self.logger.info(f"Response: {len(response.content)} chars")
                self.logger.info(f"Tokens: {response.usage['total_tokens']}")
                if response.cost_usd:
                    self.logger.info(f"Cost: ${response.cost_usd:.4f}")
                return response


        class ContentFilterMiddleware:
            \"\"\"Filters sensitive content.\"\"\"

            def __init__(self, blocked_words: List[str]):
                self.blocked_words = blocked_words

            async def process_request(
                self,
                messages: List[LLMMessage],
                config: Union[LLMConfig, Config, Dict[str, Any]]
            ) -> List[LLMMessage]:
                \"\"\"Filter input messages.\"\"\"
                filtered = []
                for msg in messages:
                    content = msg.content
                    for word in self.blocked_words:
                        content = content.replace(word, "***")
                    filtered.append(LLMMessage(
                        role=msg.role,
                        content=content,
                        name=msg.name,
                        function_call=msg.function_call,
                        metadata=msg.metadata
                    ))
                return filtered

            async def process_response(
                self,
                response: LLMResponse,
                config: Union[LLMConfig, Config, Dict[str, Any]]
            ) -> LLMResponse:
                \"\"\"Filter output.\"\"\"
                content = response.content
                for word in self.blocked_words:
                    content = content.replace(word, "***")

                from dataclasses import replace
                return replace(response, content=content)


        # Use with ConversationManager
        from dataknobs_llm.conversations import ConversationManager

        manager = await ConversationManager.create(
            llm=llm,
            prompt_builder=builder,
            middleware=[
                LoggingMiddleware(),
                ContentFilterMiddleware(["password", "secret"])
            ]
        )
        ```

    See Also:
        ConversationManager: Uses middleware for request/response processing
    """

    async def process_request(
        self,
        messages: List[LLMMessage],
        config: Union[LLMConfig, Config, Dict[str, Any]]
    ) -> List[LLMMessage]:
        """Process request before sending to LLM.

        Transform, log, validate, or filter messages before they are
        sent to the LLM provider.

        Args:
            messages: Input messages to be sent to LLM
            config: Configuration (LLMConfig, Config, or dict)

        Returns:
            Processed messages (can be modified, added to, or filtered)

        Raises:
            ValueError: If messages are invalid
        """
        ...

    async def process_response(
        self,
        response: LLMResponse,
        config: Union[LLMConfig, Config, Dict[str, Any]]
    ) -> LLMResponse:
        """Process response from LLM.

        Transform, log, validate, or filter the LLM response before
        returning to the caller.

        Args:
            response: LLM response to process
            config: Configuration (LLMConfig, Config, or dict)

        Returns:
            Processed response (can be modified)

        Raises:
            ValueError: If response is invalid
        """
        ...
