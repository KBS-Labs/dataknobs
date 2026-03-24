"""Testing utilities for dataknobs-llm.

This module provides convenience builders for creating test LLM responses,
designed for use with EchoProvider in unit and integration tests.

It also provides:
- Serialization helpers for LLMMessage/LLMResponse/ToolCall (to/from dict)
- CapturingProvider for recording real LLM interactions during test capture runs
- CapturedCall dataclass for individual captured LLM call records

Example:
    ```python
    from dataknobs_llm.llm.providers import EchoProvider
    from dataknobs_llm.testing import tool_call_response, text_response

    provider = EchoProvider({"provider": "echo", "model": "test"})
    provider.set_responses([
        tool_call_response("preview_config", {"format": "yaml"}),
        text_response("Here's your config preview!")
    ])
    ```

Capture example:
    ```python
    from dataknobs_llm.testing import CapturingProvider

    # Wrap a real provider to record all LLM calls
    capturing = CapturingProvider(real_provider, role="main")
    response = await capturing.complete(messages)

    # Inspect captured calls
    for call in capturing.captured_calls:
        print(f"Role: {call.role}, Messages: {len(call.messages)}")
    ```
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, List, Union

from .llm.base import (
    AsyncLLMProvider,
    LLMMessage,
    LLMResponse,
    LLMStreamResponse,
    ModelCapability,
    ToolCall,
)
from .extraction.schema_extractor import SimpleExtractionResult
from .llm.providers.echo import ErrorResponse

if TYPE_CHECKING:
    from .extraction.schema_extractor import SchemaExtractor
    from .llm.providers.echo import EchoProvider

logger = logging.getLogger(__name__)

__all__ = [
    "CallTracker",
    "CapturedCall",
    "CapturingProvider",
    "ConfigurableExtractor",
    "ErrorResponse",
    "ResponseSequenceBuilder",
    "SimpleExtractionResult",
    "extraction_response",
    "scripted_schema_extractor",
    "llm_message_from_dict",
    "llm_message_to_dict",
    "llm_response_from_dict",
    "llm_response_to_dict",
    "multi_tool_response",
    "text_response",
    "tool_call_from_dict",
    "tool_call_response",
    "tool_call_to_dict",
]


def text_response(
    content: str,
    *,
    model: str = "test-model",
    finish_reason: str = "stop",
    usage: dict[str, int] | None = None,
    metadata: dict[str, Any] | None = None,
) -> LLMResponse:
    """Create a simple text LLMResponse.

    Args:
        content: Response text content
        model: Model identifier (default: "test-model")
        finish_reason: Why generation stopped (default: "stop")
        usage: Optional token usage dict
        metadata: Optional metadata dict

    Returns:
        LLMResponse with text content

    Example:
        >>> response = text_response("Hello, world!")
        >>> response.content
        'Hello, world!'
    """
    return LLMResponse(
        content=content,
        model=model,
        finish_reason=finish_reason,
        usage=usage,
        metadata=metadata or {},
    )


def tool_call_response(
    tool_name: str,
    arguments: dict[str, Any] | None = None,
    *,
    tool_id: str | None = None,
    content: str = "",
    model: str = "test-model",
    additional_tools: list[tuple[str, dict[str, Any]]] | None = None,
) -> LLMResponse:
    """Create an LLMResponse with tool call(s).

    Args:
        tool_name: Name of the tool to call
        arguments: Arguments to pass to the tool (default: {})
        tool_id: Unique ID for the tool call (auto-generated if not provided)
        content: Optional text content alongside tool call
        model: Model identifier (default: "test-model")
        additional_tools: Additional tool calls as (name, args) tuples

    Returns:
        LLMResponse with tool_calls populated

    Example:
        >>> response = tool_call_response("get_weather", {"city": "NYC"})
        >>> response.tool_calls[0].name
        'get_weather'
        >>> response.tool_calls[0].parameters
        {'city': 'NYC'}

        # Multiple tool calls
        >>> response = tool_call_response(
        ...     "preview_config", {},
        ...     additional_tools=[("validate_config", {})]
        ... )
        >>> len(response.tool_calls)
        2
    """
    tools = [
        ToolCall(
            name=tool_name,
            parameters=arguments or {},
            id=tool_id or f"tc-{uuid.uuid4().hex[:8]}",
        )
    ]

    if additional_tools:
        for name, args in additional_tools:
            tools.append(
                ToolCall(
                    name=name,
                    parameters=args,
                    id=f"tc-{uuid.uuid4().hex[:8]}",
                )
            )

    return LLMResponse(
        content=content,
        model=model,
        finish_reason="tool_calls",
        tool_calls=tools,
    )


def multi_tool_response(
    tools: list[tuple[str, dict[str, Any]]],
    *,
    content: str = "",
    model: str = "test-model",
) -> LLMResponse:
    """Create an LLMResponse with multiple tool calls.

    Args:
        tools: List of (tool_name, arguments) tuples
        content: Optional text content alongside tool calls
        model: Model identifier (default: "test-model")

    Returns:
        LLMResponse with multiple tool_calls

    Example:
        >>> response = multi_tool_response([
        ...     ("preview_config", {}),
        ...     ("validate_config", {"strict": True}),
        ... ])
        >>> len(response.tool_calls)
        2
    """
    tool_calls = [
        ToolCall(
            name=name,
            parameters=args,
            id=f"tc-{uuid.uuid4().hex[:8]}",
        )
        for name, args in tools
    ]

    return LLMResponse(
        content=content,
        model=model,
        finish_reason="tool_calls",
        tool_calls=tool_calls,
    )


def extraction_response(
    data: dict[str, Any],
    *,
    model: str = "test-model",
) -> LLMResponse:
    """Create an LLMResponse for schema extraction.

    The data is JSON-encoded as the response content, mimicking
    how extraction LLMs return structured data.

    Args:
        data: Extracted data dict
        model: Model identifier (default: "test-model")

    Returns:
        LLMResponse with JSON content

    Example:
        >>> response = extraction_response({"name": "Math Tutor", "level": 5})
        >>> import json
        >>> json.loads(response.content)
        {'name': 'Math Tutor', 'level': 5}
    """
    return LLMResponse(
        content=json.dumps(data),
        model=model,
        finish_reason="stop",
    )


class ResponseSequenceBuilder:
    """Builder for creating sequences of LLM responses.

    Provides a fluent API for building test response sequences,
    useful for testing multi-turn conversations and ReAct loops.

    Example:
        ```python
        responses = (
            ResponseSequenceBuilder()
            .add_tool_call("list_templates", {})
            .add_tool_call("get_template", {"name": "quiz"})
            .add_text("I found the quiz template!")
            .build()
        )

        provider.set_responses(responses)
        ```
    """

    def __init__(self, model: str = "test-model"):
        """Initialize builder.

        Args:
            model: Model identifier for all responses
        """
        self._model = model
        self._responses: list[LLMResponse] = []

    def add_text(self, content: str) -> ResponseSequenceBuilder:
        """Add a text response to the sequence.

        Args:
            content: Response text

        Returns:
            Self for chaining
        """
        self._responses.append(text_response(content, model=self._model))
        return self

    def add_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> ResponseSequenceBuilder:
        """Add a tool call response to the sequence.

        Args:
            tool_name: Name of tool to call
            arguments: Tool arguments

        Returns:
            Self for chaining
        """
        self._responses.append(
            tool_call_response(tool_name, arguments, model=self._model)
        )
        return self

    def add_multi_tool(
        self,
        tools: list[tuple[str, dict[str, Any]]],
    ) -> ResponseSequenceBuilder:
        """Add a multi-tool call response to the sequence.

        Args:
            tools: List of (name, args) tuples

        Returns:
            Self for chaining
        """
        self._responses.append(multi_tool_response(tools, model=self._model))
        return self

    def add_extraction(
        self,
        data: dict[str, Any],
    ) -> ResponseSequenceBuilder:
        """Add an extraction response to the sequence.

        Args:
            data: Extracted data dict

        Returns:
            Self for chaining
        """
        self._responses.append(extraction_response(data, model=self._model))
        return self

    def add(self, response: LLMResponse) -> ResponseSequenceBuilder:
        """Add a custom LLMResponse to the sequence.

        Args:
            response: Custom response object

        Returns:
            Self for chaining
        """
        self._responses.append(response)
        return self

    def build(self) -> list[LLMResponse]:
        """Build and return the response sequence.

        Returns:
            List of LLMResponse objects
        """
        return list(self._responses)

    def configure(self, provider: EchoProvider) -> EchoProvider:
        """Configure an EchoProvider with this sequence.

        Args:
            provider: EchoProvider to configure

        Returns:
            The configured provider
        """
        provider.set_responses(self._responses)
        return provider


# =============================================================================
# Schema extraction testing constructs
# =============================================================================

# SimpleExtractionResult: canonical definition lives in
# extraction.schema_extractor alongside ExtractionResult.
# Re-exported here (imported at top of file) for backward compatibility.


class ConfigurableExtractor:
    """Test extractor that returns pre-configured results.

    Supports both single-result mode (same result every call) and
    sequence mode (different result per call).  After the sequence is
    exhausted, the last result is repeated.

    All calls are recorded in ``extract_calls`` for verification.

    Example — single result::

        extractor = ConfigurableExtractor(
            result_data={"name": "Alice"}, confidence=0.9,
        )

    Example — sequence of results::

        extractor = ConfigurableExtractor(results=[
            SimpleExtractionResult(data={"name": "Alice"}, confidence=0.5),
            SimpleExtractionResult(data={"topic": "math"}, confidence=0.9),
        ])

    Example — recording extractor (capture inputs, return empty)::

        extractor = ConfigurableExtractor(confidence=0.0)
        # ... run code that calls extractor.extract() ...
        assert extractor.extract_calls[0]["schema"] == expected_schema
    """

    def __init__(
        self,
        results: list[SimpleExtractionResult] | None = None,
        result_data: dict[str, Any] | None = None,
        confidence: float = 0.9,
    ) -> None:
        if results is not None:
            self._results = results
        else:
            self._results = [
                SimpleExtractionResult(
                    data=result_data or {}, confidence=confidence,
                )
            ]
        self.call_index = 0
        self.extract_calls: list[dict[str, Any]] = []

    async def extract(
        self,
        text: str,
        schema: dict[str, Any],
        context: dict[str, Any] | None = None,
        model: str | None = None,
    ) -> SimpleExtractionResult:
        """Return next result in sequence and record the call."""
        self.extract_calls.append({
            "text": text,
            "schema": schema,
            "context": context,
            "model": model,
        })
        idx = min(self.call_index, len(self._results) - 1)
        self.call_index += 1
        return self._results[idx]


def scripted_schema_extractor(
    responses: list[str] | None = None,
) -> tuple[SchemaExtractor, EchoProvider]:
    """Create a real SchemaExtractor backed by scripted EchoProvider responses.

    Unlike ``ConfigurableExtractor`` (which bypasses extraction entirely),
    this exercises the full ``SchemaExtractor`` pipeline: prompt building,
    LLM call, JSON parsing, and confidence scoring.  Use this for tests
    that need to verify extraction behavior end-to-end.

    Args:
        responses: JSON strings the EchoProvider will return in order.
            Each string should be the raw JSON that SchemaExtractor
            expects to parse (e.g. ``'{"name": "Alice"}'``).

    Returns:
        Tuple of ``(SchemaExtractor, EchoProvider)`` — the provider is
        returned so tests can verify call counts or queue additional
        responses.

    Example::

        extractor, ext_provider = scripted_schema_extractor([
            '{"name": "Alice", "topic": "math"}',
        ])
        # Use with WizardReasoning or any code that accepts a SchemaExtractor
        reasoning = WizardReasoning(wizard_fsm=fsm, extractor=extractor)
    """
    from dataknobs_llm.extraction.schema_extractor import SchemaExtractor
    from dataknobs_llm.llm.providers.echo import EchoProvider as _EchoProvider

    provider = _EchoProvider({"provider": "echo", "model": "echo-extraction"})
    if responses:
        provider.set_responses(responses)
    extractor = SchemaExtractor(provider=provider)
    return extractor, provider


# =============================================================================
# Serialization helpers — LLM types ↔ dict for capture/replay
# =============================================================================


def tool_call_to_dict(tc: ToolCall) -> dict[str, Any]:
    """Serialize a ToolCall to a JSON-compatible dict.

    Delegates to :meth:`ToolCall.to_dict`. Kept for backward compatibility.

    Args:
        tc: ToolCall to serialize

    Returns:
        Dictionary representation
    """
    return tc.to_dict()


def tool_call_from_dict(d: dict[str, Any]) -> ToolCall:
    """Deserialize a ToolCall from a dict.

    Delegates to :meth:`ToolCall.from_dict`. Kept for backward compatibility.

    Args:
        d: Dictionary representation

    Returns:
        ToolCall instance
    """
    return ToolCall.from_dict(d)


def llm_message_to_dict(msg: LLMMessage) -> dict[str, Any]:
    """Serialize an LLMMessage to a JSON-compatible dict.

    Delegates to :meth:`LLMMessage.to_dict`. Kept for backward compatibility.

    Args:
        msg: LLMMessage to serialize

    Returns:
        Dictionary representation (only non-None optional fields included)
    """
    return msg.to_dict()


def llm_message_from_dict(d: dict[str, Any]) -> LLMMessage:
    """Deserialize an LLMMessage from a dict.

    Delegates to :meth:`LLMMessage.from_dict`. Kept for backward compatibility.

    Args:
        d: Dictionary representation

    Returns:
        LLMMessage instance
    """
    return LLMMessage.from_dict(d)


def llm_response_to_dict(resp: LLMResponse) -> dict[str, Any]:
    """Serialize an LLMResponse to a JSON-compatible dict.

    Omits ``created_at`` and ``cumulative_cost_usd`` (runtime artifacts that
    make captures non-deterministic). Only includes non-None optional fields.

    Args:
        resp: LLMResponse to serialize

    Returns:
        Dictionary representation
    """
    d: dict[str, Any] = {"content": resp.content, "model": resp.model}
    if resp.finish_reason is not None:
        d["finish_reason"] = resp.finish_reason
    if resp.usage is not None:
        d["usage"] = resp.usage
    if resp.function_call is not None:
        d["function_call"] = resp.function_call
    if resp.tool_calls is not None:
        d["tool_calls"] = [tool_call_to_dict(tc) for tc in resp.tool_calls]
    if resp.metadata:
        d["metadata"] = resp.metadata
    if resp.cost_usd is not None:
        d["cost_usd"] = resp.cost_usd
    return d


def llm_response_from_dict(d: dict[str, Any]) -> LLMResponse:
    """Deserialize an LLMResponse from a dict.

    Args:
        d: Dictionary representation

    Returns:
        LLMResponse instance
    """
    tool_calls = None
    if "tool_calls" in d and d["tool_calls"] is not None:
        tool_calls = [tool_call_from_dict(tc) for tc in d["tool_calls"]]

    return LLMResponse(
        content=d["content"],
        model=d["model"],
        finish_reason=d.get("finish_reason"),
        usage=d.get("usage"),
        function_call=d.get("function_call"),
        tool_calls=tool_calls,
        metadata=d.get("metadata", {}),
        cost_usd=d.get("cost_usd"),
    )


# =============================================================================
# CapturingProvider — wraps a real provider, records all LLM calls
# =============================================================================


@dataclass
class CapturedCall:
    """Record of a single LLM call captured by CapturingProvider.

    Attributes:
        role: Provider role tag (e.g., "main", "extraction")
        messages: Serialized request messages (list of dicts)
        response: Serialized LLM response (dict)
        config_overrides: Config overrides passed to the call, if any
        tools: Tool definitions passed to the call, if any
        duration_seconds: Wall-clock duration of the call
        call_index: Per-instance call ordering (0-based)
    """

    role: str
    messages: list[dict[str, Any]]
    response: dict[str, Any]
    config_overrides: dict[str, Any] | None = None
    tools: list[Any] | None = None
    duration_seconds: float = 0.0
    call_index: int = 0


class CapturingProvider(AsyncLLMProvider):
    """Provider wrapper that records all LLM calls for capture-replay testing.

    Wraps a real ``AsyncLLMProvider`` delegate, forwarding all calls while
    recording request/response pairs as ``CapturedCall`` objects. The role
    tag (e.g., "main" or "extraction") enables replay to route responses
    to the correct EchoProvider.

    Args:
        delegate: Real provider to wrap
        role: Tag identifying this provider's role (default: "main")

    Example:
        ```python
        from dataknobs_llm.testing import CapturingProvider

        real_provider = OllamaProvider(config)
        capturing = CapturingProvider(real_provider, role="main")

        # Use normally — calls pass through to the real provider
        response = await capturing.complete(messages)

        # Inspect what was captured
        assert capturing.call_count == 1
        call = capturing.captured_calls[0]
        print(f"Sent {len(call.messages)} messages, got: {call.response['content'][:50]}")
        ```
    """

    def __init__(self, delegate: AsyncLLMProvider, role: str = "main") -> None:
        # Initialize with the delegate's config (satisfies LLMProvider.__init__)
        super().__init__(delegate.config)
        self._delegate = delegate
        self._role = role
        self._captured_calls: list[CapturedCall] = []

    @property
    def role(self) -> str:
        """Provider role tag."""
        return self._role

    @property
    def captured_calls(self) -> list[CapturedCall]:
        """All captured calls (read-only copy)."""
        return list(self._captured_calls)

    @property
    def call_count(self) -> int:
        """Number of captured calls."""
        return len(self._captured_calls)

    # -- Delegated lifecycle methods --

    async def initialize(self) -> None:
        """Delegate initialization to the wrapped provider."""
        await self._delegate.initialize()

    async def close(self) -> None:
        """Delegate close to the wrapped provider."""
        await self._delegate.close()

    async def validate_model(self) -> bool:
        """Delegate model validation to the wrapped provider."""
        if hasattr(self._delegate, "validate_model"):
            return await self._delegate.validate_model()
        return True

    def _detect_capabilities(self) -> List[ModelCapability]:
        """Delegate to the wrapped provider's full get_capabilities()."""
        return self._delegate.get_capabilities()

    def get_capabilities(self) -> List[ModelCapability]:
        """Delegate capability detection to the wrapped provider."""
        return self._delegate.get_capabilities()

    # -- Captured methods --

    async def complete(
        self,
        messages: Union[str, List[LLMMessage]],
        config_overrides: Dict[str, Any] | None = None,
        tools: list[Any] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Forward completion to delegate and capture the call.

        Args:
            messages: Input messages (string or list of LLMMessage)
            config_overrides: Optional per-request config overrides
            tools: Optional tool definitions
            **kwargs: Additional provider-specific parameters

        Returns:
            LLMResponse from the delegate (unchanged)
        """
        # Serialize messages for capture
        if isinstance(messages, str):
            serialized_msgs = [{"role": "user", "content": messages}]
        else:
            serialized_msgs = [llm_message_to_dict(m) for m in messages]

        start = time.monotonic()
        response = await self._delegate.complete(
            messages, config_overrides=config_overrides, tools=tools, **kwargs
        )
        duration = time.monotonic() - start

        self._captured_calls.append(
            CapturedCall(
                role=self._role,
                messages=serialized_msgs,
                response=llm_response_to_dict(response),
                config_overrides=config_overrides,
                tools=tools,
                duration_seconds=round(duration, 4),
                call_index=len(self._captured_calls),
            )
        )

        return response

    async def stream_complete(
        self,
        messages: Union[str, List[LLMMessage]],
        config_overrides: Dict[str, Any] | None = None,
        tools: list[Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[LLMStreamResponse]:
        """Forward streaming completion to delegate and capture the assembled response.

        Yields chunks to the caller in real time. After the stream completes,
        assembles the full response content and records a CapturedCall.

        Args:
            messages: Input messages
            config_overrides: Optional per-request config overrides
            tools: Optional tool definitions
            **kwargs: Additional provider-specific parameters

        Yields:
            LLMStreamResponse chunks from the delegate
        """
        if isinstance(messages, str):
            serialized_msgs = [{"role": "user", "content": messages}]
        else:
            serialized_msgs = [llm_message_to_dict(m) for m in messages]

        # Collect chunks while yielding them through
        assembled_content = ""
        final_chunk: LLMStreamResponse | None = None
        start = time.monotonic()

        async for chunk in self._delegate.stream_complete(
            messages, config_overrides=config_overrides, tools=tools, **kwargs
        ):
            assembled_content += chunk.delta
            if chunk.is_final:
                final_chunk = chunk
            yield chunk

        duration = time.monotonic() - start

        # Build assembled response for capture
        assembled_response = LLMResponse(
            content=assembled_content,
            model=final_chunk.model or self._delegate.config.model if final_chunk else self._delegate.config.model,
            finish_reason=final_chunk.finish_reason if final_chunk else None,
            usage=final_chunk.usage if final_chunk else None,
            tool_calls=final_chunk.tool_calls if final_chunk else None,
        )

        self._captured_calls.append(
            CapturedCall(
                role=self._role,
                messages=serialized_msgs,
                response=llm_response_to_dict(assembled_response),
                config_overrides=config_overrides,
                tools=tools,
                duration_seconds=round(duration, 4),
                call_index=len(self._captured_calls),
            )
        )

    async def embed(
        self,
        texts: Union[str, List[str]],
        **kwargs: Any,
    ) -> Union[List[float], List[List[float]]]:
        """Delegate embedding to the wrapped provider (not captured)."""
        return await self._delegate.embed(texts, **kwargs)

    async def function_call(
        self,
        messages: List[LLMMessage],
        functions: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> LLMResponse:
        """Delegate function calling to the wrapped provider and capture the call."""
        serialized_msgs = [llm_message_to_dict(m) for m in messages]

        start = time.monotonic()
        response = await self._delegate.function_call(messages, functions, **kwargs)
        duration = time.monotonic() - start

        self._captured_calls.append(
            CapturedCall(
                role=self._role,
                messages=serialized_msgs,
                response=llm_response_to_dict(response),
                tools=functions,
                duration_seconds=round(duration, 4),
                call_index=len(self._captured_calls),
            )
        )

        return response


# =============================================================================
# CallTracker — collect calls across multiple CapturingProviders
# =============================================================================


class CallTracker:
    """Collect new LLM calls across multiple CapturingProviders per turn.

    In multi-LLM bot scenarios (e.g. a main LLM and an extraction LLM),
    a single user turn may trigger calls to several providers.
    ``CallTracker`` collects calls from all registered providers since the
    last collection, assigns sequential global indices, and returns them
    in registration order.

    Args:
        **providers: Named ``CapturingProvider`` instances.  The keyword
            name is used only for ``get_provider()`` lookups.

    Example:
        ```python
        main = CapturingProvider(real_main, role="main")
        extraction = CapturingProvider(real_extraction, role="extraction")
        tracker = CallTracker(main=main, extraction=extraction)

        # ... run a bot turn that triggers LLM calls ...

        new_calls = tracker.collect_new_calls()
        for call in new_calls:
            print(f"[{call.call_index}] {call.role}: {call.response['content'][:40]}")
        ```
    """

    def __init__(self, **providers: CapturingProvider) -> None:
        self._providers: dict[str, CapturingProvider] = dict(providers)
        # Track how many calls we've already seen per provider
        self._cursors: dict[str, int] = {
            name: p.call_count for name, p in self._providers.items()
        }
        self._global_index: int = 0

    def get_provider(self, name: str) -> CapturingProvider | None:
        """Get a registered provider by name.

        Args:
            name: Provider registration name

        Returns:
            CapturingProvider or None if not found
        """
        return self._providers.get(name)

    @property
    def provider_names(self) -> list[str]:
        """Get list of registered provider names."""
        return list(self._providers.keys())

    @property
    def total_calls(self) -> int:
        """Total number of calls collected across all providers."""
        return self._global_index

    def collect_new_calls(self) -> list[CapturedCall]:
        """Collect calls made since the last collection.

        Returns new calls from all providers, sorted by provider
        registration order.  Each call receives a sequential
        ``call_index`` relative to all previously collected calls.

        Returns:
            List of new CapturedCall objects with global call_index values
        """
        new_calls: list[CapturedCall] = []

        for name, provider in self._providers.items():
            cursor = self._cursors[name]
            all_calls = provider.captured_calls
            provider_new = all_calls[cursor:]

            for call in provider_new:
                new_calls.append(
                    CapturedCall(
                        role=call.role,
                        messages=call.messages,
                        response=call.response,
                        config_overrides=call.config_overrides,
                        tools=call.tools,
                        duration_seconds=call.duration_seconds,
                        call_index=self._global_index,
                    )
                )
                self._global_index += 1

            self._cursors[name] = len(all_calls)

        return new_calls
