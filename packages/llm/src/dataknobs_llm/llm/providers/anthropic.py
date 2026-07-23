"""Anthropic Claude LLM provider implementation.

This module provides Anthropic Claude API integration for dataknobs-llm, supporting:
- Claude 3 (Opus, Sonnet, Haiku) and Claude 2 models
- Native tools API for function calling
- Vision capabilities (Claude 3+)
- Streaming responses
- Long context windows (up to 200k tokens)
- Advanced reasoning and coding capabilities

The AnthropicProvider uses the official Anthropic Python SDK and supports
all standard Anthropic API parameters including system prompts, temperature,
and token limits.

Example:
    ```python
    from dataknobs_llm.llm.providers import AnthropicProvider
    from dataknobs_llm.llm.base import LLMConfig

    # Create provider
    config = LLMConfig(
        provider="anthropic",
        model="claude-3-sonnet-20240229",
        api_key="sk-ant-...",  # or set ANTHROPIC_API_KEY env var
        temperature=0.7,
        max_tokens=1024
    )

    async with AnthropicProvider(config) as llm:
        # Simple completion
        response = await llm.complete("Explain quantum computing")
        print(response.content)

        # Streaming for real-time output
        async for chunk in llm.stream_complete("Write a story"):
            print(chunk.delta, end="", flush=True)

        # Tool use (Claude 3+)
        tools = [{
            "name": "calculator",
            "description": "Perform arithmetic",
            "input_schema": {
                "type": "object",
                "properties": {
                    "operation": {"type": "string"},
                    "x": {"type": "number"},
                    "y": {"type": "number"}
                }
            }
        }]

        response = await llm.function_call(messages, tools)
    ```

See Also:
    - Anthropic API Documentation: https://docs.anthropic.com/
    - anthropic Python package: https://github.com/anthropics/anthropic-sdk-python
"""

import json
import logging
import os
import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Union, AsyncIterator

from dataknobs_common.exceptions import (
    OperationError, RateLimitError, ValidationError,
)

from ..base import (
    LLMAdapter, LLMConfig, LLMMessage, LLMResponse, LLMStreamResponse,
    AsyncLLMProvider, ModelCapability, ModelConstraints, ToolCall,
    normalize_claude_stop_reason, normalize_llm_config
)
from dataknobs_llm.prompts import AsyncPromptBuilder

logger = logging.getLogger(__name__)

#: Claude model families that reject the ``temperature`` sampling parameter
#: at the request boundary (a hard 400). Matched as lowercase substrings of
#: the model id, distinguishing the Claude 5 generation
#: (``claude-sonnet-5``, ``claude-opus-5``, ...) from the Claude 4.x
#: generation (``claude-opus-4-8``, ``claude-haiku-4-5-...``), which still
#: accepts ``temperature``. This is the auto-detected default only — a
#: consumer can declare or withdraw the rule at runtime via
#: ``LLMConfig.constraints`` without a dataknobs release (see
#: :class:`~dataknobs_llm.llm.base.ModelConstraints`).
_CLAUDE_5_TEMPERATURE_REJECTORS: tuple[str, ...] = (
    "claude-5",
    "claude-sonnet-5",
    "claude-opus-5",
    "claude-haiku-5",
    "claude-fable-5",
)

#: Sampling parameters that ``adapt_config`` may forward and that a model
#: family might reject. Used by the 400-retry safety net to identify which
#: forwarded param an "unsupported parameter" 400 refers to.
_SAMPLING_PARAMS: tuple[str, ...] = (
    "temperature",
    "top_p",
    "stop_sequences",
    "frequency_penalty",
    "presence_penalty",
)

#: The Claude stop-reason normalization table + truncation detection live in
#: ``dataknobs_llm.llm.base`` (:func:`normalize_claude_stop_reason`), shared
#: verbatim with the Bedrock Converse adapter since Bedrock runs Claude.

#: Process-level cache of sampling params discovered — via a 400 at request
#: time — to be rejected by a given model, keyed by lowercased model id. The
#: safety net for families the static table doesn't know yet: the first request
#: pays one 400, the offending param is dropped and retried, and the discovery
#: is folded into :meth:`AnthropicProvider._detect_constraints` so subsequent
#: requests to the same model drop it up front (≤1 wasted round-trip per model
#: per process). A consumer can still pre-empt this entirely by declaring the
#: param in ``LLMConfig.constraints``.
_DISCOVERED_REJECTED_PARAMS: Dict[str, set[str]] = {}

#: Valid ``system_message_policy`` values governing how a **mid-conversation**
#: ``role="system"`` message is handled (a *leading* system prompt is always
#: hoisted into the top-level ``system`` param — Anthropic's Messages API has
#: no inline ``system`` role). See :meth:`AnthropicAdapter.adapt_messages`.
#:
#: - ``"inline"`` — convert the message to a ``user`` message at its position,
#:   consolidating content blocks so role-alternation and
#:   ``tool_use`` ↔ ``tool_result`` adjacency stay valid. **Default** —
#:   preserves the notice's positional, in-context meaning.
#: - ``"hoist"`` — merge into the top-level ``system`` param (legacy behavior;
#:   positionally lossy but a byte-for-byte back-compat escape hatch).
#: - ``"warn"`` — log a warning naming the message, then hoist (makes the
#:   lossy case visible without changing structure).
#: - ``"reject"`` — raise :class:`~dataknobs_common.exceptions.ValidationError`
#:   (treat a mid-conversation system message as a configuration error).
_SYSTEM_MESSAGE_POLICIES: frozenset[str] = frozenset(
    {"inline", "hoist", "warn", "reject"}
)

#: Default mid-conversation system-message policy. ``"inline"`` is the safe
#: default because the adapter's content-block consolidation keeps the adapted
#: request structurally valid (no consecutive same-role messages, tool pairing
#: preserved), so the more-correct positional semantics carry no alternation
#: risk.
_DEFAULT_SYSTEM_MESSAGE_POLICY: str = "inline"

#: Cap on how much of a mid-conversation system message body is echoed into a
#: ``warn``-policy log line — enough to identify the offending message without
#: dumping an arbitrarily large payload.
_WARN_CONTENT_PREVIEW_CHARS: int = 120

if TYPE_CHECKING:
    from dataknobs_config.config import Config


class AnthropicAdapter(LLMAdapter):
    """Adapter for Anthropic Messages API format.

    Converts between dataknobs standard types (LLMMessage, LLMResponse,
    LLMConfig) and Anthropic-specific formats. Key differences from other
    providers:

    - System messages are a top-level ``system`` parameter, not in the
      message list.
    - Assistant tool calls use ``content`` blocks with ``type="tool_use"``.
    - Tool results are ``role="user"`` messages with ``type="tool_result"``
      content blocks paired via ``tool_use_id``.
    """

    # Anthropic requires max_tokens on every request.
    DEFAULT_MAX_TOKENS: int = 1024

    def __init__(
        self,
        *,
        system_message_policy: str = _DEFAULT_SYSTEM_MESSAGE_POLICY,
        accepts_inline_system: bool = False,
    ) -> None:
        """Build the adapter.

        Args:
            system_message_policy: How a **mid-conversation** ``role="system"``
                message is handled — one of :data:`_SYSTEM_MESSAGE_POLICIES`
                (``"inline"``/``"hoist"``/``"warn"``/``"reject"``). A *leading*
                system prompt always hoists regardless. Read from
                ``LLMConfig.options["system_message_policy"]`` by
                :class:`AnthropicProvider`.
            accepts_inline_system: Whether the model family accepts an inline
                ``role="system"`` message in the ``messages`` array (the S1
                :class:`~dataknobs_llm.llm.base.ModelConstraints.accepts_inline_system`
                datum — ``False`` for Anthropic). When ``True``, a
                mid-conversation system message is left in place and the policy
                is not consulted (the family handles it natively).

        Raises:
            ValidationError: If ``system_message_policy`` is not a recognized
                policy — a configuration error, surfaced fail-closed at
                construction rather than silently defaulted.
        """
        super().__init__()
        if system_message_policy not in _SYSTEM_MESSAGE_POLICIES:
            raise ValidationError(
                f"Unknown system_message_policy {system_message_policy!r}. "
                f"Valid policies: {sorted(_SYSTEM_MESSAGE_POLICIES)}."
            )
        self.system_message_policy = system_message_policy
        self.accepts_inline_system = accepts_inline_system

    def adapt_messages(
        self,
        messages: List[LLMMessage],
        system_prompt: str | None = None,
    ) -> tuple[str, List[Dict[str, Any]]]:
        """Convert LLMMessages to Anthropic Messages API format.

        A **leading** ``role="system"`` message (before any non-system
        message) is always hoisted into the top-level ``system`` param —
        Anthropic's Messages API has no inline ``system`` role. A
        **mid-conversation** ``role="system"`` message is handled per the
        configured :attr:`system_message_policy` (unless
        :attr:`accepts_inline_system` is ``True``, in which case it is left in
        place). Under ``"inline"`` the message becomes a ``user`` message at
        its position; content-block consolidation
        (:meth:`_append_user_block`) keeps the adapted sequence structurally
        valid — no consecutive same-role messages, and every ``tool_result``
        stays adjacent/paired to its ``tool_use`` (the exact conditions the
        Anthropic API enforces).

        Args:
            messages: Standard LLMMessage list.
            system_prompt: Optional system prompt from provider config to
                merge with any system messages found in the list.

        Returns:
            Tuple of ``(system_content, anthropic_messages)`` where
            ``system_content`` should be passed as the ``system`` API
            parameter and ``anthropic_messages`` as ``messages``.

        Raises:
            ValidationError: Under ``system_message_policy="reject"`` when a
                mid-conversation system message is encountered.
        """
        anthropic_messages: List[Dict[str, Any]] = []
        system_content = system_prompt or ""
        seen_non_system = False

        for msg in messages:
            if msg.role == "system":
                if not seen_non_system:
                    # Leading system prompt — always hoist (correct + required).
                    system_content = self._merge_system(
                        system_content, msg.content
                    )
                elif self.accepts_inline_system:
                    # The family accepts an inline system message — leave it in
                    # place rather than hoisting or converting. RESERVED: no
                    # current Claude model accepts a role="system" entry in the
                    # Messages `messages` array (it is a 400), so this branch is
                    # dormant for every model AnthropicAdapter serves today. It
                    # exists only for a hypothetical future Claude family that
                    # sets accepts_inline_system=True via S1 constraints; do NOT
                    # force constraints={"accepts_inline_system": True} for a
                    # current model — use policy 'inline' instead.
                    anthropic_messages.append(
                        {"role": "system", "content": msg.content}
                    )
                elif self._mid_system_inlines(msg):
                    self._append_user_block(
                        anthropic_messages,
                        {"type": "text", "text": msg.content},
                    )
                else:
                    system_content = self._merge_system(
                        system_content, msg.content
                    )
                continue

            seen_non_system = True
            if msg.role == "assistant" and msg.tool_calls:
                content_blocks: List[Dict[str, Any]] = []
                if msg.content:
                    content_blocks.append({"type": "text", "text": msg.content})
                for tc in msg.tool_calls:
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tc.id or tc.name,
                        "name": tc.name,
                        "input": tc.parameters,
                    })
                anthropic_messages.append({
                    "role": "assistant",
                    "content": content_blocks,
                })
            elif msg.role == "tool":
                # Anthropic expects tool results as user messages with
                # tool_result content blocks paired by tool_use_id.
                # Consecutive tool results (and an inlined system notice) must
                # be consolidated into a single user message — the API rejects
                # consecutive messages with the same role.
                tool_use_id = msg.tool_call_id or msg.name or "unknown"
                self._append_user_block(
                    anthropic_messages,
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": msg.content,
                    },
                )
            elif (
                msg.role == "user"
                and anthropic_messages
                and anthropic_messages[-1]["role"] == "user"
            ):
                # A plain user turn directly following another user turn (e.g.
                # after an inlined system notice) must consolidate — the API
                # rejects consecutive same-role messages. A lone user turn
                # still passes through as a plain string (below).
                self._append_user_block(
                    anthropic_messages,
                    {"type": "text", "text": msg.content},
                )
            else:
                anthropic_messages.append({
                    "role": msg.role,
                    "content": msg.content,
                })

        return system_content, anthropic_messages

    @staticmethod
    def _merge_system(existing: str, addition: str) -> str:
        """Concatenate a system message into the accumulated system content."""
        return f"{existing}\n\n{addition}" if existing else addition

    def _mid_system_inlines(self, msg: LLMMessage) -> bool:
        """Apply the policy to a mid-conversation system message.

        Returns ``True`` when the message should be inlined as a ``user``
        message at its position, ``False`` when it should be hoisted into the
        top-level ``system`` param. ``"warn"`` logs then hoists; ``"reject"``
        raises. A *leading* system message never reaches this method.

        Raises:
            ValidationError: Under ``system_message_policy="reject"``.
        """
        policy = self.system_message_policy
        if policy == "inline":
            return True
        if policy == "hoist":
            return False
        if policy == "warn":
            preview = (msg.content or "")[:_WARN_CONTENT_PREVIEW_CHARS]
            logger.warning(
                'Mid-conversation role="system" message hoisted into the '
                "top-level system prompt (content: %r) — its positional, "
                "in-context meaning is lost. Set "
                "options.system_message_policy='inline' to preserve it at "
                "position, or 'reject' to treat it as a configuration error.",
                preview,
            )
            return False
        # policy == "reject"
        raise ValidationError(
            'Mid-conversation role="system" message rejected by '
            "system_message_policy='reject'. Anthropic's Messages API has no "
            "inline system role; move the system content to the leading "
            "position, or select policy 'inline'/'hoist'/'warn'."
        )

    @staticmethod
    def _append_user_block(
        anthropic_messages: List[Dict[str, Any]],
        block: Dict[str, Any],
    ) -> None:
        """Add a content block to the current ``user`` turn, order-safely.

        Consolidates into the preceding message when it is a ``user`` turn so
        the adapted sequence never has two consecutive ``user`` messages and a
        ``tool_result`` stays adjacent to the ``tool_use`` it answers. A
        plain-string ``user`` turn is promoted to a content-block list so the
        new block can join it; otherwise a fresh ``user`` message is started.

        Anthropic's Messages API additionally requires every ``tool_result``
        block to come **first** in a user turn's content array — any ``text``
        must follow all tool results, or the request is a 400. So a
        ``tool_result`` is spliced in *after the last existing* ``tool_result``
        (keeping the tool results grouped at the front, in first-seen order)
        rather than appended blindly onto the end past an inlined-notice
        ``text`` block; every other block appends at the end, already after the
        tool results. This keeps the headline case — a mid-conversation system
        notice inlined between an assistant ``tool_use`` and its following
        ``tool_result`` — a valid request instead of an invalid
        ``[text, tool_result]`` ordering.
        """
        if anthropic_messages and anthropic_messages[-1]["role"] == "user":
            last = anthropic_messages[-1]
            content = last["content"]
            if isinstance(content, str):
                content = (
                    [{"type": "text", "text": content}] if content else []
                )
                last["content"] = content
            if block.get("type") == "tool_result":
                insert_at = 0
                for i, existing in enumerate(content):
                    if existing.get("type") == "tool_result":
                        insert_at = i + 1
                content.insert(insert_at, block)
            else:
                content.append(block)
        else:
            anthropic_messages.append({"role": "user", "content": [block]})

    def adapt_response(self, response: Any) -> LLMResponse:
        """Parse Anthropic response into LLMResponse.

        Iterates content blocks to handle text, tool_use, and mixed
        responses. Builds ``ToolCall`` objects from ``tool_use`` blocks.

        Args:
            response: Anthropic ``Message`` object from the SDK.

        Returns:
            Standard ``LLMResponse`` with content, tool_calls, and usage.
        """
        content = ""
        tool_calls: list[ToolCall] = []

        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    name=block.name,
                    parameters=block.input if isinstance(block.input, dict) else {},
                    id=block.id,
                ))

        usage = None
        if hasattr(response, "usage"):
            usage = {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": (
                    response.usage.input_tokens + response.usage.output_tokens
                ),
            }

        finish_reason, truncated, metadata = normalize_claude_stop_reason(
            response.stop_reason
        )

        return LLMResponse(
            content=content,
            model=response.model,
            finish_reason=finish_reason,
            truncated=truncated,
            usage=usage,
            tool_calls=tool_calls if tool_calls else None,
            metadata=metadata,
        )

    def adapt_config(self, config: LLMConfig) -> Dict[str, Any]:
        """Build Anthropic API parameters from config.

        Shared by ``complete()``, ``stream_complete()``, and
        ``function_call()`` to prevent parameter drift between methods.

        Args:
            config: Standard LLMConfig.

        Returns:
            Dictionary of Anthropic API parameters.
        """
        gen = config.generation_params()
        params: Dict[str, Any] = {
            "model": config.model,
            "max_tokens": gen.get("max_tokens", self.DEFAULT_MAX_TOKENS),
        }
        if "temperature" in gen:
            params["temperature"] = gen["temperature"]
        if "top_p" in gen:
            params["top_p"] = gen["top_p"]
        if "stop_sequences" in gen:
            params["stop_sequences"] = gen["stop_sequences"]
        return params

    def adapt_tools(self, tools: list[Any]) -> list[Dict[str, Any]]:
        """Convert Tool objects to Anthropic tools format.

        Args:
            tools: List of Tool objects with ``name``, ``description``,
                and ``schema`` attributes.

        Returns:
            List of Anthropic tool definitions.
        """
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.schema if hasattr(tool, "schema") else {},
            }
            for tool in tools
        ]

    def adapt_raw_functions(
        self, functions: list[Dict[str, Any]],
    ) -> list[Dict[str, Any]]:
        """Convert raw function dicts to Anthropic tools format.

        Used by the deprecated ``function_call()`` method which receives
        raw dicts rather than Tool objects.

        Args:
            functions: List of raw function definition dicts with
                ``name``, ``description``, and ``parameters`` keys.

        Returns:
            List of Anthropic tool definitions.
        """
        return [
            {
                "name": func.get("name", ""),
                "description": func.get("description", ""),
                "input_schema": func.get("parameters", {
                    "type": "object",
                    "properties": {},
                    "required": [],
                }),
            }
            for func in functions
        ]


class AnthropicProvider(AsyncLLMProvider):
    r"""Anthropic Claude LLM provider with full API support.

    Provides async access to Anthropic's Claude models including Claude 3
    (Opus, Sonnet, Haiku) and Claude 2. Supports advanced features like
    native tool use, vision, and extended context windows.

    Features:
        - Claude 3 Opus/Sonnet/Haiku and Claude 2 models
        - Native tools API for function calling (Claude 3+)
        - Vision capabilities for image understanding (Claude 3+)
        - Streaming responses for real-time output
        - Long context windows (up to 200k tokens)
        - Advanced reasoning and coding capabilities
        - System prompts for behavior control
        - JSON output mode

    Example:
        ```python
        from dataknobs_llm.llm.providers import AnthropicProvider
        from dataknobs_llm.llm.base import LLMConfig, LLMMessage

        # Basic usage
        config = LLMConfig(
            provider="anthropic",
            model="claude-3-sonnet-20240229",
            api_key="sk-ant-...",
            temperature=0.7,
            max_tokens=1024
        )

        async with AnthropicProvider(config) as llm:
            # Simple completion
            response = await llm.complete("Explain machine learning")
            print(response.content)

            # With system prompt
            messages = [
                LLMMessage(
                    role="system",
                    content="You are an expert Python tutor"
                ),
                LLMMessage(
                    role="user",
                    content="How do I use decorators?"
                )
            ]
            response = await llm.complete(messages)

        # Long context processing (Claude 3+)
        long_config = LLMConfig(
            provider="anthropic",
            model="claude-3-opus-20240229",
            max_tokens=4096
        )

        llm = AnthropicProvider(long_config)
        await llm.initialize()

        # Process large document
        with open("large_doc.txt") as f:
            long_text = f.read()  # Up to 200k tokens!

        response = await llm.complete(
            f"Summarize this document:\n\n{long_text}"
        )

        # Tool use / function calling (Claude 3+)
        tools = [
            {
                "name": "web_search",
                "description": "Search the web for information",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        },
                        "num_results": {
                            "type": "integer",
                            "description": "Number of results"
                        }
                    },
                    "required": ["query"]
                }
            }
        ]

        messages = [
            LLMMessage(
                role="user",
                content="Search for latest AI news"
            )
        ]

        response = await llm.function_call(messages, tools)
        if response.function_call:
            import json
            tool_input = json.loads(response.function_call["arguments"])
            print(f"Tool: {response.function_call['name']}")
            print(f"Input: {tool_input}")
        ```

    Args:
        config: LLMConfig, dataknobs Config, or dict with provider settings
        prompt_builder: Optional AsyncPromptBuilder for prompt rendering

    Attributes:
        _client: Anthropic AsyncAnthropic client instance

    See Also:
        LLMConfig: Configuration options
        AsyncLLMProvider: Base provider interface
        Anthropic API Docs: https://docs.anthropic.com/
    """

    def __init__(
        self,
        config: Union[LLMConfig, "Config", Dict[str, Any]],
        prompt_builder: AsyncPromptBuilder | None = None
    ):
        # Normalize config first
        llm_config = normalize_llm_config(config)
        super().__init__(llm_config, prompt_builder=prompt_builder)
        # Mid-conversation system-message policy is a request-shape decision:
        # the policy comes from provider options, while whether the family even
        # accepts an inline system message is the S1 ModelConstraints datum
        # (False for Anthropic — resolved from self.config so a constraints
        # override is honored). Both are passed to the adapter, keeping the
        # LLMAdapter.adapt_messages ABC signature stable for other adapters.
        # Both are captured once here at construction rather than resolved
        # per-call: for Anthropic accepts_inline_system is model-invariant
        # (False for every model) and system_message_policy is not a per-call
        # config_overrides surface, so construction-time capture is correct. A
        # future family whose accepts_inline_system varied by model would need
        # this resolved per-call in adapt_messages instead.
        policy = str(
            llm_config.options.get(
                "system_message_policy", _DEFAULT_SYSTEM_MESSAGE_POLICY
            )
        )
        self.adapter = AnthropicAdapter(
            system_message_policy=policy,
            accepts_inline_system=self.get_constraints().accepts_inline_system,
        )

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

    async def _close_client(self) -> None:
        """Close the Anthropic client."""
        if self._client:
            await self._client.close()  # type: ignore[unreachable]

    async def validate_model(self) -> bool:
        """Validate model availability."""
        valid_models = [
            'claude-3-opus', 'claude-3-sonnet', 'claude-3-haiku',
            'claude-2.1', 'claude-2.0', 'claude-instant-1.2'
        ]
        return any(m in self.config.model for m in valid_models)

    def _detect_capabilities(self) -> List[ModelCapability]:
        """Auto-detect Anthropic model capabilities."""
        model = self.config.model.lower()
        capabilities = [
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CHAT,
            ModelCapability.STREAMING,
            ModelCapability.CODE,
        ]

        # Claude 3+ models support vision, tools, and JSON mode
        modern_models = [
            'claude-3', 'claude-3.5', 'claude-4',
            'claude-sonnet', 'claude-opus', 'claude-haiku',
        ]
        if any(m in model for m in modern_models):
            capabilities.extend([
                ModelCapability.VISION,
                ModelCapability.FUNCTION_CALLING,
                ModelCapability.JSON_MODE,
            ])

        return capabilities

    def _detect_constraints(self, config: LLMConfig) -> ModelConstraints:
        """Auto-detect Anthropic request-shape constraints for *config*'s model.

        Two Anthropic-specific rules, both string-matched by family
        (mirroring :meth:`_detect_capabilities`):

        - ``accepts_inline_system=False`` for **every** Anthropic model —
          Anthropic's Messages API has no inline ``system`` role; a system
          message is always a top-level ``system`` param. Read by the
          mid-conversation system-message policy.
        - ``rejected_params={"temperature"}`` for the Claude 5 family, which
          returns a 400 when ``temperature`` is supplied. The Claude 4.x
          family (``claude-opus-4-8``, ``claude-haiku-4-5-...``) still accepts
          it, so the match distinguishes the generations.

        Matches on ``config.model`` (not ``self.config.model``) so a per-call
        model override resolves the overriding family's constraints — see
        :meth:`~dataknobs_llm.llm.base.LLMProvider.get_constraints`.

        Any params discovered at runtime via the 400-retry safety net (see
        :meth:`_recover_rejected_param`) are folded in for the current process,
        so a family the static table doesn't know yet self-corrects after one
        request.

        Both are the auto-detected defaults; a consumer overrides either via
        ``LLMConfig.constraints`` (see :meth:`_resolve_constraints`).
        """
        model = config.model.lower()
        rejected: set[str] = set()
        if any(m in model for m in _CLAUDE_5_TEMPERATURE_REJECTORS):
            rejected.add("temperature")
        rejected |= _DISCOVERED_REJECTED_PARAMS.get(model, set())
        return ModelConstraints(
            rejected_params=frozenset(rejected),
            accepts_inline_system=False,
        )

    def _build_api_kwargs(self, config: LLMConfig) -> Dict[str, Any]:
        """Build Anthropic API params, dropping ones the family rejects.

        Single choke point over :meth:`AnthropicAdapter.adapt_config` shared
        by ``complete``/``stream_complete``/``function_call``: builds the
        request params, then drops any parameter named in the resolved
        :class:`~dataknobs_llm.llm.base.ModelConstraints.rejected_params`
        (e.g. ``temperature`` for the Claude 5 family) with a
        ``logger.warning`` naming the param and model — **drop-and-warn,
        never a silent drop**. Constraints resolve from the passed *runtime*
        ``config`` (not ``self.config``), so a per-call model or ``constraints``
        override is honored — the drop reflects the model actually being sent.

        Args:
            config: Runtime config (with any ``config_overrides`` applied).

        Returns:
            Anthropic API parameter dict with rejected params removed.
        """
        params = self.adapter.adapt_config(config)
        constraints = self.get_constraints(config)
        for param_name in constraints.rejected_params:
            if param_name in params:
                del params[param_name]
                logger.warning(
                    "Dropping request parameter %r rejected by Anthropic "
                    "model family (model=%r); override via "
                    "LLMConfig.constraints if this is incorrect.",
                    param_name,
                    config.model,
                )
        return params

    def _retry_after(self, exc: Exception) -> float | None:
        """Best-effort ``retry-after`` (seconds) from an Anthropic 429.

        Reads the ``retry-after`` response header if present; returns ``None``
        when absent or unparseable.
        """
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None)
        if not headers:
            return None
        raw = headers.get("retry-after")
        if not raw:
            return None
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None

    def _translate_api_error(self, exc: Exception) -> Exception | None:
        """Translate a raw Anthropic SDK error into a dataknobs exception.

        Lets consumers catch by a dataknobs exception type instead of coupling
        to the ``anthropic`` SDK's classes:

        - 429 → :class:`~dataknobs_common.exceptions.RateLimitError`
          (with ``retry_after`` when the header is present),
        - 400 → :class:`~dataknobs_common.exceptions.ValidationError`,
        - 401/403 → :class:`~dataknobs_common.exceptions.OperationError`
          (auth/permission),
        - any other Anthropic API error (other status, connection, timeout) →
          :class:`~dataknobs_common.exceptions.OperationError`.

        Returns ``None`` for a non-Anthropic exception so the caller re-raises
        it unchanged (a bug in our own code is never masked as an API error).
        The original SDK error is preserved on ``__cause__`` — callers raise
        ``... from exc``.
        """
        try:
            import anthropic
        except ImportError:  # pragma: no cover - anthropic is installed post-init
            return None
        if not isinstance(exc, anthropic.APIError):
            return None
        status = getattr(exc, "status_code", None)
        if status == 429:
            return RateLimitError(
                f"Anthropic rate limit exceeded: {exc}",
                retry_after=self._retry_after(exc),
            )
        if status == 400:
            return ValidationError(f"Anthropic rejected the request: {exc}")
        if status in (401, 403):
            return OperationError(
                f"Anthropic authentication/authorization error: {exc}"
            )
        return OperationError(f"Anthropic API error: {exc}")

    def _recover_rejected_param(
        self, exc: Exception, api_kwargs: Dict[str, Any]
    ) -> str | None:
        """Drop a sampling param an unexpected 400 identifies, for one retry.

        The safety net for a model family the static constraint table doesn't
        know yet. On a 400 whose message unambiguously names exactly one
        sampling param that is present in ``api_kwargs``, this **mutates**
        ``api_kwargs`` to drop that param, memoizes the discovery in
        :data:`_DISCOVERED_REJECTED_PARAMS` (so subsequent requests to the same
        model drop it up front — ≤1 wasted round-trip per model per process),
        and returns the dropped param name so the caller can retry once.

        Conservative by design — returns ``None`` (no retry) unless it can
        pin down exactly one offending param, so an ambiguous or unrelated 400
        is translated (S4) rather than blindly retried. Only genuine Anthropic
        400s (``status_code == 400``) are considered.

        Args:
            exc: The exception raised by ``messages.create``/``.stream``.
            api_kwargs: The request kwargs (mutated in place on a hit).

        Returns:
            The dropped param name, or ``None`` when no safe recovery applies.
        """
        if getattr(exc, "status_code", None) != 400:
            return None
        text = str(exc).lower()
        present = [p for p in _SAMPLING_PARAMS if p in api_kwargs]
        named = [p for p in present if p in text]
        if len(named) != 1:
            return None
        param = named[0]
        del api_kwargs[param]
        model = str(api_kwargs.get("model", "")).lower()
        _DISCOVERED_REJECTED_PARAMS.setdefault(model, set()).add(param)
        return param

    async def _create_message(self, api_kwargs: Dict[str, Any]) -> Any:
        """Call ``messages.create`` with 400-retry recovery and S4 wrapping.

        On an unexpected sampling-param 400, drops the offending param and
        retries once (:meth:`_recover_rejected_param`); any other Anthropic
        error is translated to a dataknobs exception
        (:meth:`_translate_api_error`); non-Anthropic errors propagate
        unchanged.
        """
        try:
            return await self._client.messages.create(**api_kwargs)
        except Exception as exc:
            dropped = self._recover_rejected_param(exc, api_kwargs)
            if dropped is not None:
                logger.warning(
                    "Anthropic rejected request parameter %r (model=%r); "
                    "dropped it and retrying once. Future requests to this "
                    "model will drop it up front.",
                    dropped,
                    api_kwargs.get("model"),
                )
                try:
                    return await self._client.messages.create(**api_kwargs)
                except Exception as retry_exc:
                    translated = self._translate_api_error(retry_exc)
                    if translated is None:
                        raise
                    raise translated from retry_exc
            translated = self._translate_api_error(exc)
            if translated is None:
                raise
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
            tools: Optional list of Tool objects for function calling
            **kwargs: Additional provider-specific parameters
        """
        if not self._is_initialized:
            await self.initialize()

        # Get runtime config (with overrides applied if provided)
        runtime_config = self._get_runtime_config(config_overrides)

        # Convert to Anthropic format
        if isinstance(messages, str):
            msg_list = [LLMMessage(role="user", content=messages)]
        else:
            msg_list = messages

        system_content, anthropic_messages = self.adapter.adapt_messages(
            msg_list, system_prompt=self.config.system_prompt,
        )

        # Build API call kwargs (drops params the model family rejects)
        api_kwargs = self._build_api_kwargs(runtime_config)
        api_kwargs["messages"] = anthropic_messages
        if system_content:
            api_kwargs["system"] = system_content

        # Handle tools if provided
        if tools:
            api_kwargs["tools"] = self.adapter.adapt_tools(tools)

        # Make API call (400-retry recovery + vendor-error translation)
        response = await self._create_message(api_kwargs)

        return self._analyze_response(self.adapter.adapt_response(response))

    async def stream_complete(
        self,
        messages: Union[str, List[LLMMessage]],
        config_overrides: Dict[str, Any] | None = None,
        tools: list[Any] | None = None,
        **kwargs: Any
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

        # Convert to Anthropic format
        if isinstance(messages, str):
            msg_list = [LLMMessage(role="user", content=messages)]
        else:
            msg_list = messages

        system_content, anthropic_messages = self.adapter.adapt_messages(
            msg_list, system_prompt=self.config.system_prompt,
        )

        # Build stream kwargs (drops params the model family rejects)
        stream_kwargs = self._build_api_kwargs(runtime_config)
        stream_kwargs["messages"] = anthropic_messages
        if system_content:
            stream_kwargs["system"] = system_content

        # Handle tools if provided
        if tools:
            stream_kwargs["tools"] = self.adapter.adapt_tools(tools)

        # Stream API call. A rejected-param 400 fails on stream entry, before
        # any chunk is yielded, so the 400-retry safety net can recover it once
        # without risk of double-yielding; any other Anthropic error (on entry
        # or mid-stream) is translated to a dataknobs exception. The retry is
        # gated on ``not yielded`` so a mid-stream failure is never retried.
        yielded = False
        retried = False
        while True:
            try:
                async with self._client.messages.stream(**stream_kwargs) as stream:
                    async for chunk in stream:
                        if chunk.type == 'content_block_delta':
                            if hasattr(chunk.delta, 'text'):
                                yielded = True
                                yield LLMStreamResponse(
                                    delta=chunk.delta.text,
                                    is_final=False
                                )

                    # Final message — use adapter to parse content blocks
                    message = await stream.get_final_message()
                    parsed = self.adapter.adapt_response(message)

                    final_chunk = LLMStreamResponse(
                        delta='',
                        is_final=True,
                        finish_reason=parsed.finish_reason,
                        truncated=parsed.truncated,
                        tool_calls=parsed.tool_calls,
                        model=runtime_config.model,
                    )
                    self._warn_if_truncated(final_chunk)
                    yielded = True
                    yield final_chunk
                return
            except Exception as exc:
                if not yielded and not retried:
                    dropped = self._recover_rejected_param(exc, stream_kwargs)
                    if dropped is not None:
                        retried = True
                        logger.warning(
                            "Anthropic rejected request parameter %r (model="
                            "%r); dropped it and retrying the stream once.",
                            dropped,
                            stream_kwargs.get("model"),
                        )
                        continue
                translated = self._translate_api_error(exc)
                if translated is None:
                    raise
                raise translated from exc

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
        **kwargs: Any
    ) -> LLMResponse:
        """Execute function calling with native Anthropic tools API (Claude 3+)."""
        warnings.warn(
            "function_call() is deprecated, use complete(tools=...) instead",
            DeprecationWarning,
            stacklevel=2,
        )
        if not self._is_initialized:
            await self.initialize()

        system_content, anthropic_messages = self.adapter.adapt_messages(
            messages, system_prompt=self.config.system_prompt,
        )

        # function_call() receives raw dicts, not Tool objects — delegate
        # to the adapter's raw function converter.
        tools = self.adapter.adapt_raw_functions(functions)

        try:
            fc_kwargs = self._build_api_kwargs(self.config)
            fc_kwargs["messages"] = anthropic_messages
            fc_kwargs["tools"] = tools
            if system_content:
                fc_kwargs["system"] = system_content
            # Route through the shared choke point so this path gets the same
            # 400-retry recovery and vendor-error translation as
            # complete()/stream_complete() — not a bare messages.create().
            response = await self._create_message(fc_kwargs)

            parsed = self.adapter.adapt_response(response)

            # Surface the first tool call as the legacy function_call dict and
            # route through the shared _analyze_response choke point, so this
            # path preserves truncated / raw_finish_reason and fires the
            # truncation warning — exactly like complete()/stream_complete().
            # Rebuilding a fresh LLMResponse here would silently drop them.
            return self._analyze_response(
                self._attach_legacy_function_call(parsed)
            )

        except ValidationError as e:
            # A 400 (translated to ValidationError by _create_message) means the
            # request was rejected — for older models that lack the native tools
            # API, that is the "tools unsupported" signal, so fall back to
            # prompt-based function calling. Rate-limit (429 → RateLimitError),
            # auth (401/403 → OperationError), and any other translated error
            # are NOT caught here: they propagate unchanged, so a rate-limited
            # or unauthenticated call is never masked as a "tools failed"
            # fallback that would issue a second call against the same endpoint.
            logger.warning(
                "Anthropic native tools unsupported (request rejected), falling "
                "back to prompt-based function calling: %s", e,
            )

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
                LLMMessage(role="system", content=system_prompt)
            ] + list(messages)

            response = await self.complete(messages_with_system, **kwargs)

            # Parse function call from response
            if "FUNCTION_CALL:" in response.content:
                try:
                    func_json = response.content.split("FUNCTION_CALL:")[1].strip()
                    function_call = json.loads(func_json)
                    response.function_call = function_call
                except (json.JSONDecodeError, IndexError):
                    pass

            return response
