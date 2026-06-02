"""Middleware system for conversation processing.

This module provides middleware capabilities for processing messages before they
are sent to the LLM and processing responses after they come back from the LLM.
Middleware can be used for logging, validation, content filtering, rate limiting,
metadata injection, and more.

Execution Model (Onion Pattern):
    Middleware wraps around LLM calls in an "onion" pattern:

    Request Flow:  MW0 → MW1 → MW2 → LLM
    Response Flow: LLM → MW2 → MW1 → MW0

    Example with 3 middleware [Logging, RateLimit, Validation]:

    ```
    1. Logging.process_request()      # Log incoming messages
    2.   RateLimit.process_request()  # Check rate limits
    3.     Validation.process_request()  # Validate request
    4.       → LLM Call →              # Actual LLM API call
    5.     Validation.process_response() # Validate LLM response
    6.   RateLimit.process_response() # Add rate limit info to response
    7. Logging.process_response()     # Log response details
    ```

    This ensures middleware can:
    - Time the full LLM call (start timer in process_request, stop in process_response)
    - Wrap operations symmetrically (open resources → LLM → close resources)
    - See the final state after inner middleware modifications

Performance Considerations:
    - **Middleware adds latency**: Each middleware's `process_request()` and
      `process_response()` adds to total response time. Keep middleware logic fast.

    - **Async is key**: All middleware methods are async. Use `await` for I/O
      operations (DB calls, network requests) to avoid blocking.

    - **Order matters**: Place expensive middleware (like ValidationMiddleware
      that makes additional LLM calls) at the end of the list to minimize
      wasted work if earlier middleware rejects the request.

    - **Memory usage**: RateLimitMiddleware keeps request history in memory.
      For high-traffic applications, consider external rate limiting (Redis, etc.).

Available Middleware:
    - **LoggingMiddleware**: Log requests and responses for debugging
    - **ContentFilterMiddleware**: Filter inappropriate content from responses
    - **ValidationMiddleware**: Validate responses with additional LLM call
    - **MetadataMiddleware**: Inject custom metadata into messages/responses
    - **RateLimitMiddleware**: Enforce rate limits with sliding window
    - **PromoteToPersistMiddleware**: Promote allowlisted flat ``response.metadata``
      keys into the ``_persist`` namespace so they flow onto the persisted
      assistant conversation node (no source changes to the original writers)

Example:
    ```python
    from dataknobs_llm.conversations import (
        ConversationManager,
        LoggingMiddleware,
        RateLimitMiddleware,
        ContentFilterMiddleware
    )
    import logging

    # Create middleware instances (order matters!)
    logger = logging.getLogger(__name__)
    logging_mw = LoggingMiddleware(logger)
    rate_limit_mw = RateLimitMiddleware(max_requests=10, window_seconds=60)
    filter_mw = ContentFilterMiddleware(
        filter_words=["inappropriate"],
        replacement="[FILTERED]"
    )

    # Create conversation with middleware stack
    # Execution: Logging → RateLimit → Filter → LLM → Filter → RateLimit → Logging
    manager = await ConversationManager.create(
        llm=llm,
        prompt_builder=builder,
        storage=storage,
        middleware=[logging_mw, rate_limit_mw, filter_mw]
    )

    # All requests will go through middleware pipeline
    await manager.add_message(role="user", content="Hello")
    response = await manager.complete()  # Middleware applied automatically
    ```

See Also:
    ConversationManager: Uses middleware for all LLM interactions
    ConversationMiddleware: Base class for custom middleware
"""

import dataclasses
import logging
import re
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Callable, Dict, List

from dataknobs_common.ratelimit import InMemoryRateLimiter, RateLimit, RateLimiterConfig
from dataknobs_common.exceptions import RateLimitError

from dataknobs_llm.llm import LLMMessage, LLMResponse
from dataknobs_llm.llm.providers import AsyncLLMProvider
from dataknobs_llm.conversations.history_redaction import (
    HistoryRedaction,
    apply_history_redactions,
    compile_history_redactions,
)
from dataknobs_llm.conversations.storage import ConversationState
from dataknobs_llm.prompts import AsyncPromptBuilder


class ConversationMiddleware(ABC):
    """Base class for conversation middleware.

    Middleware can process requests before LLM and responses after LLM.
    Middleware is executed in order for requests, and in reverse order
    for responses (onion pattern).

    Execution Order:
        Given middleware list [MW0, MW1, MW2]:

        - **Request**: MW0 → MW1 → MW2 → LLM
        - **Response**: LLM → MW2 → MW1 → MW0

        This allows MW0 to:
        1. Start a timer in `process_request()`
        2. See the LLM call complete
        3. Stop the timer in `process_response()` and log total time

    Use Cases:
        - **Logging**: Track request/response details
        - **Validation**: Verify request/response content
        - **Transformation**: Modify messages or responses
        - **Rate Limiting**: Enforce API usage limits
        - **Caching**: Store/retrieve responses
        - **Monitoring**: Collect metrics and analytics
        - **Security**: Filter sensitive information

    Per-turn Middleware:
        If the middleware's behavior depends on turn-specific content
        (e.g. a response post-processor that needs this turn's retrieval
        candidates), construct a fresh instance per turn and attach it
        via :meth:`ConversationManager.scoped_middleware` rather than
        via the permanent stack. The scoped API handles attachment,
        exception-safe detachment, and preserves onion ordering relative
        to the permanent middleware.

    Persisting Metadata:
        Middleware writes to ``response.metadata`` are ephemeral by
        default — they live on the
        :class:`~dataknobs_llm.llm.base.LLMResponse` for this call and do
        not flow to the persisted assistant conversation node. To persist
        audit data (a citation renderer's culling outcome, a validation
        middleware's verdict, a tool-selection middleware's rationale),
        write into the ``_persist`` sub-dictionary:

            >>> async def process_response(self, response, state):
            ...     if response.metadata is None:
            ...         response.metadata = {}
            ...     persist = response.metadata.setdefault("_persist", {})
            ...     persist["citation_audit"] = self._outcome_to_dict()
            ...     return response

        Keys inside ``_persist`` are merged into the assistant node's
        ``metadata`` by
        :meth:`ConversationManager._finalize_completion`. The
        ``_persist`` marker itself is not propagated to the node — only
        the keys inside it. Canonical framework fields (``usage``,
        ``model``, ``provider``, ``finish_reason``, cost, config
        overrides) and the caller's ``metadata=`` kwarg win over
        middleware-provided values of the same key — middleware audits,
        it does not replace.

        For third-party writers (providers, in-tree middleware) that
        write flat ``response.metadata`` keys you want persisted without
        modifying their source, use :class:`PromoteToPersistMiddleware`
        — register it at position ``[0]`` of the ``middleware`` list
        with a key allowlist, and it copies matching flat keys into
        ``_persist`` on each response.

    Example:
        ```python
        from dataknobs_llm.conversations import ConversationMiddleware
        import time

        class TimingMiddleware(ConversationMiddleware):
            '''Measure LLM call duration.'''

            async def process_request(self, messages, state):
                # Store start time in state metadata
                state.metadata["request_start"] = time.time()
                return messages

            async def process_response(self, response, state):
                # Calculate elapsed time
                start = state.metadata.get("request_start")
                if start:
                    elapsed = time.time() - start
                    if not response.metadata:
                        response.metadata = {}
                    response.metadata["llm_duration_seconds"] = elapsed
                    print(f"LLM call took {elapsed:.2f}s")
                return response

        # Use in conversation
        manager = await ConversationManager.create(
            llm=llm,
            middleware=[TimingMiddleware()]
        )
        ```

    Note:
        **Performance Tips**:

        - Keep `process_request()` and `process_response()` fast
        - Use async I/O (await) for external calls (DB, network)
        - Don't block the async loop with synchronous operations
        - For expensive operations, consider running them in background tasks
        - Store state in `state.metadata` not instance variables (thread safety)

    See Also:
        LoggingMiddleware: Example implementation
        ConversationManager.complete: Where middleware is executed
    """

    @abstractmethod
    async def process_request(
        self,
        messages: List[LLMMessage],
        state: ConversationState
    ) -> List[LLMMessage]:
        """Process messages before sending to LLM.

        Args:
            messages: Messages to send to LLM
            state: Current conversation state

        Returns:
            Processed messages (can modify, add, or remove messages)

        Example:
            >>> from datetime import datetime
            >>> async def process_request(self, messages, state):
            ...     # Add timestamp to metadata
            ...     for msg in messages:
            ...         if not msg.metadata:
            ...             msg.metadata = {}
            ...         msg.metadata["timestamp"] = datetime.now().isoformat()
            ...     return messages
        """
        pass

    @abstractmethod
    async def process_response(
        self,
        response: LLMResponse,
        state: ConversationState
    ) -> LLMResponse:
        """Process response from LLM.

        Args:
            response: LLM response
            state: Current conversation state

        Returns:
            Processed response (can modify content, metadata, etc.)

        Example:
            >>> from datetime import datetime
            >>> async def process_response(self, response, state):
            ...     # Add processing metadata
            ...     if not response.metadata:
            ...         response.metadata = {}
            ...     response.metadata["processed_at"] = datetime.now().isoformat()
            ...     return response
        """
        pass


class LoggingMiddleware(ConversationMiddleware):
    """Middleware that logs all requests and responses.

    This middleware is useful for debugging and monitoring conversations.
    It logs message counts, conversation IDs, and response metadata.

    Example:
        >>> import logging
        >>> logger = logging.getLogger(__name__)
        >>> logging.basicConfig(level=logging.INFO)
        >>>
        >>> middleware = LoggingMiddleware(logger)
        >>> manager = await ConversationManager.create(
        ...     llm=llm,
        ...     prompt_builder=builder,
        ...     storage=storage,
        ...     middleware=[middleware]
        ... )
    """

    def __init__(self, logger: logging.Logger | None = None):
        """Initialize logging middleware.

        Args:
            logger: Logger instance to use (defaults to module logger)
        """
        self.logger = logger or logging.getLogger(__name__)

    async def process_request(
        self,
        messages: List[LLMMessage],
        state: ConversationState
    ) -> List[LLMMessage]:
        """Log request details before sending to LLM."""
        self.logger.info(
            f"Conversation {state.conversation_id} - "
            f"Sending {len(messages)} messages to LLM"
        )
        self.logger.debug(
            f"Conversation {state.conversation_id} - "
            f"Message roles: {[msg.role for msg in messages]}"
        )
        return messages

    async def process_response(
        self,
        response: LLMResponse,
        state: ConversationState
    ) -> LLMResponse:
        """Log response details after receiving from LLM."""
        content_length = len(response.content) if response.content else 0
        self.logger.info(
            f"Conversation {state.conversation_id} - "
            f"Received response: {content_length} chars, "
            f"model={response.model}, finish_reason={response.finish_reason}"
        )
        if response.usage:
            self.logger.debug(
                f"Conversation {state.conversation_id} - "
                f"Token usage: {response.usage}"
            )
        return response


class ContentFilterMiddleware(ConversationMiddleware):
    """Middleware that filters inappropriate content from responses.

    This middleware can be used to redact or replace specific words or
    patterns in LLM responses. Useful for content moderation and compliance.

    Example:
        >>> # Filter specific words
        >>> middleware = ContentFilterMiddleware(
        ...     filter_words=["badword1", "badword2"],
        ...     replacement="[FILTERED]"
        ... )
        >>>
        >>> # Case-insensitive filtering
        >>> middleware = ContentFilterMiddleware(
        ...     filter_words=["sensitive"],
        ...     case_sensitive=False
        ... )
    """

    def __init__(
        self,
        filter_words: List[str],
        replacement: str = "[FILTERED]",
        case_sensitive: bool = True
    ):
        """Initialize content filter middleware.

        Args:
            filter_words: List of words/phrases to filter
            replacement: String to replace filtered content with
            case_sensitive: Whether filtering should be case-sensitive
        """
        self.filter_words = filter_words
        self.replacement = replacement
        self.case_sensitive = case_sensitive

    async def process_request(
        self,
        messages: List[LLMMessage],
        state: ConversationState
    ) -> List[LLMMessage]:
        """Pass through requests without filtering."""
        return messages

    async def process_response(
        self,
        response: LLMResponse,
        state: ConversationState
    ) -> LLMResponse:
        """Filter inappropriate content from response."""
        content = response.content

        for word in self.filter_words:
            if self.case_sensitive:
                content = content.replace(word, self.replacement)
            else:
                pattern = re.compile(re.escape(word), re.IGNORECASE)
                content = pattern.sub(self.replacement, content)

        # Track if any filtering occurred
        if content != response.content:
            if not response.metadata:
                response.metadata = {}
            response.metadata["content_filtered"] = True
            response.content = content

        return response


class HistoryRedactionMiddleware(ConversationMiddleware):
    r"""Rewrite assistant-role messages on their way into the LLM.

    Bots that emit structured citation tokens (``[bib:N · …]`` headers,
    bare ``bib:N`` references, footnote markers, …) carry those tokens
    forward in the conversation tree. On the next turn, the model sees
    its own prior citations as part of conversation history and will
    happily reuse them — even when this turn's retrieval no longer
    surfaces those sources. The result is a *citation carry-over leak*:
    bibs cited inline that aren't in the current kept set.

    This middleware applies regex rewrites to assistant-role message
    content in :meth:`process_request` — the prompt-feed direction.
    The persisted conversation tree is NOT mutated; only the
    in-memory ``LLMMessage`` list passed to the LLM provider is. The
    UI, exports, and `manager.messages` continue to see the original
    text. :meth:`process_response` is a passthrough.

    The matching set defaults to assistant-role messages only. Users
    rarely type bib codes; system prompts may intentionally reference
    bibs for the bibliography menu. Override ``redact_roles`` only
    when you need a broader rewrite (e.g. redacting tool-result
    messages that quote prior assistant output).

    The middleware accepts either the typed
    :class:`~dataknobs_llm.conversations.history_redaction.HistoryRedaction`
    sequence or the legacy ``dict`` shape — see :meth:`__init__`.

    Example:
        >>> # Block bib-N carry-over for an ASRM-style framework bot.
        >>> from dataknobs_llm.conversations import (
        ...     HistoryRedaction,
        ...     HistoryRedactionMiddleware,
        ... )
        >>> # Typed shape (preferred — reuses the list built for
        >>> # ``BufferMemoryConfig.history_redactions``):
        >>> mw = HistoryRedactionMiddleware(
        ...     redactions=[
        ...         # Bracketed header MUST come first (longer match).
        ...         HistoryRedaction(
        ...             pattern=r"\\[bib:\\d+[^\\]]*\\]",
        ...             replacement="[prior citation]",
        ...         ),
        ...         HistoryRedaction(
        ...             pattern=r"\\bbib:\\d+\\b",
        ...             replacement="[prior citation]",
        ...         ),
        ...     ],
        ... )
        >>> # Legacy dict shape (the config-spec path) is equivalent:
        >>> mw = HistoryRedactionMiddleware(
        ...     redactions=[
        ...         {"pattern": r"\\[bib:\\d+[^\\]]*\\]", "replacement": "[prior citation]"},
        ...         {"pattern": r"\\bbib:\\d+\\b", "replacement": "[prior citation]"},
        ...     ],
        ... )
        >>> manager = await ConversationManager.create(
        ...     llm=llm, prompt_builder=builder, storage=storage,
        ...     middleware=[mw],
        ... )

    Pattern order matters: callers list the more specific pattern
    (e.g. a bracketed header) before the more general bare token. If
    the bare-token rule ran first it would consume ``bib:N`` inside
    the bracket and leave a malformed ``[ · vendor · …]`` header.
    """

    def __init__(
        self,
        redactions: Sequence[HistoryRedaction] | Sequence[Mapping[str, str]],
        redact_roles: Iterable[str] = ("assistant",),
    ):
        """Initialize history-redaction middleware.

        Args:
            redactions: Ordered redaction patterns. Accepts either a
                sequence of
                :class:`~dataknobs_llm.conversations.history_redaction.HistoryRedaction`
                instances (preferred, typed-config-friendly) or a sequence
                of ``{"pattern": <regex>, "replacement": <str>}`` mappings
                (the original shape used by the config-spec path). Mixing
                the two shapes in a single call raises ``TypeError``.
                Patterns are compiled once at construction; an empty
                sequence ⇒ passthrough. A dict spec missing ``"pattern"``
                or carrying an empty pattern is rejected with ``ValueError``
                at construction so config typos surface at the config-load
                boundary rather than mid-loop on first request.
            redact_roles: Message roles whose content is rewritten.
                Defaults to ``("assistant",)`` — the dominant
                citation-leak source.

        Raises:
            TypeError: If typed and dict redactions are mixed in one call.
            ValueError: If a dict redaction spec is missing the
                ``"pattern"`` key or has an empty pattern.
            re.error: If a redaction spec's ``"pattern"`` is not a valid
                regular expression.
        """
        typed = self._normalize_redactions(redactions)
        self._compiled = compile_history_redactions(typed)
        self._redact_roles = frozenset(redact_roles)

    @staticmethod
    def _normalize_redactions(
        redactions: Sequence[HistoryRedaction] | Sequence[Mapping[str, str]],
    ) -> list[HistoryRedaction]:
        """Project the dual-shape input onto the typed sequence.

        First-element type drives the dispatch; the sequence must be
        homogeneous (mixing typed and dict shapes raises ``TypeError`` —
        a mixed list is a bug, not back-compat). In the dict branch every
        element is also re-checked against ``Mapping`` so a misuse like
        ``redactions=[("pattern", "x")]`` (a plausible "I meant a 2-tuple"
        typo) raises the same shape-mismatch ``TypeError`` rather than
        crashing inside the loop body with
        ``AttributeError: 'tuple' object has no attribute 'keys'`` far
        from the call site. The dict branch preserves the up-front
        "missing 'pattern' key" ``ValueError`` (with index + the keys
        actually present) so a config typo surfaces at the config-load
        boundary rather than as a generic empty-pattern error from
        ``HistoryRedaction.__post_init__``.
        """
        if not redactions:
            return []
        first = redactions[0]
        if isinstance(first, HistoryRedaction):
            for r in redactions:
                if not isinstance(r, HistoryRedaction):
                    raise TypeError(
                        "HistoryRedactionMiddleware: mixed typed/dict "
                        "redactions are not supported; pass either a "
                        "Sequence[HistoryRedaction] or a "
                        "Sequence[Mapping[str, str]]."
                    )
            return list(redactions)
        out: list[HistoryRedaction] = []
        for i, r in enumerate(redactions):
            if isinstance(r, HistoryRedaction):
                raise TypeError(
                    "HistoryRedactionMiddleware: mixed typed/dict "
                    "redactions are not supported; pass either a "
                    "Sequence[HistoryRedaction] or a "
                    "Sequence[Mapping[str, str]]."
                )
            if not isinstance(r, Mapping):
                raise TypeError(
                    "HistoryRedactionMiddleware: redactions["
                    f"{i}] is not a Mapping (got {type(r).__name__!r}); "
                    "pass either a Sequence[HistoryRedaction] or a "
                    "Sequence[Mapping[str, str]]."
                )
            if "pattern" not in r:
                raise ValueError(
                    f"HistoryRedactionMiddleware: redactions[{i}] is missing "
                    f"the required 'pattern' key (got keys: {sorted(r.keys())})"
                )
            # ``HistoryRedaction.__post_init__`` enforces the non-empty
            # pattern and eagerly compiles the regex — no separate
            # empty-pattern guard is needed here.
            out.append(
                HistoryRedaction(
                    pattern=r["pattern"],
                    replacement=r.get("replacement", ""),
                )
            )
        return out

    async def process_request(
        self,
        messages: List[LLMMessage],
        state: ConversationState | None,
    ) -> List[LLMMessage]:
        """Rewrite assistant content in the message list bound for the LLM.

        Returns new ``LLMMessage`` instances; the input list and its
        elements are not mutated. Non-content fields (``tool_calls``,
        ``tool_call_id``, ``name``, ``function_call``, ``metadata``) are
        preserved — agent/tool-use loops depend on the invocation and
        pairing fields surviving the rewrite, so the clone uses
        :func:`dataclasses.replace` rather than reconstructing field by
        field.

        The mutable container fields (``metadata`` dict, ``tool_calls``
        list, ``function_call`` dict) are shallow-copied onto the clone so
        a downstream middleware that does e.g.
        ``out_msg.metadata.update({"trace_id": ...})`` (see
        :class:`MetadataMiddleware`) does NOT alias-mutate the source
        ``LLMMessage`` through a shared reference. The defense is
        deliberately shallow: it neutralizes top-level mutations (the
        cited contract violation) without paying for a deep copy of
        nested values, which remains the caller's contract to avoid.
        ``ToolCall`` items inside ``tool_calls`` are shared by reference;
        callers that mutate a ``ToolCall.parameters`` dict in place still
        see the source mutate, but no in-tree middleware does that.

        Args:
            messages: Messages to send to LLM.
            state: Current conversation state (unused — the redactor
                is stateless and per-message).

        Returns:
            New message list with assistant content redacted per the
            configured patterns. Other roles pass through unchanged.
        """

        def _role_of(msg: LLMMessage) -> str:
            return msg.role

        def _content_of(msg: LLMMessage) -> str | None:
            return msg.content

        def _replace(msg: LLMMessage, new_content: str) -> LLMMessage:
            # ``dataclasses.replace`` clones the whole LLMMessage and
            # overrides only ``content`` — ``tool_calls``, ``tool_call_id``,
            # ``name``, ``function_call``, ``metadata`` (and any future
            # field) survive automatically. Manual reconstruction would
            # silently drop them and break agent / tool-use loops.
            #
            # The mutable containers are shallow-copied so downstream
            # mutations (e.g. ``out_msg.metadata.update(...)``) do not
            # alias-mutate the source ``LLMMessage``. See method docstring.
            return dataclasses.replace(
                msg,
                content=new_content,
                metadata=dict(msg.metadata),
                tool_calls=(
                    list(msg.tool_calls) if msg.tool_calls is not None else None
                ),
                function_call=(
                    dict(msg.function_call)
                    if msg.function_call is not None
                    else None
                ),
            )

        return apply_history_redactions(
            messages,
            self._compiled,
            role_of=_role_of,
            content_of=_content_of,
            replace_content=_replace,
            redact_roles=self._redact_roles,
        )

    async def process_response(
        self,
        response: LLMResponse,
        state: ConversationState | None,
    ) -> LLMResponse:
        """Passthrough — the LLM's response is not redacted.

        Redaction is intentionally scoped to the prompt-feed direction
        (see :meth:`process_request`); the response direction is a
        passthrough so the fresh LLM response keeps its full citation
        set for rendering. Only when that response RE-ENTERS the
        prompt as history on the next turn does the redactor strip it.
        """
        return response


class ValidationMiddleware(ConversationMiddleware):
    """Middleware that validates LLM responses using another LLM call.

    This middleware uses a validation prompt and a separate LLM call to check
    if responses meet certain criteria. Can optionally retry on validation failure.

    Example:
        >>> from dataknobs_llm.llm.providers import OpenAIProvider
        >>> from dataknobs_llm.llm.base import LLMConfig
        >>>
        >>> # Create validation middleware
        >>> config = LLMConfig(provider="openai", model="gpt-4")
        >>> validation_llm = OpenAIProvider(config)
        >>> middleware = ValidationMiddleware(
        ...     llm=validation_llm,
        ...     prompt_builder=builder,
        ...     validation_prompt="validate_response",
        ...     auto_retry=False  # Raise error instead of retrying
        ... )
        >>>
        >>> # Validation prompt should ask the LLM to respond with
        >>> # "VALID" or "INVALID" based on the response content
    """

    def __init__(
        self,
        llm: AsyncLLMProvider,
        prompt_builder: AsyncPromptBuilder,
        validation_prompt: str,
        auto_retry: bool = False,
        retry_limit: int = 3
    ):
        """Initialize validation middleware.

        Args:
            llm: LLM provider to use for validation (required)
            prompt_builder: Prompt builder for rendering validation prompt
            validation_prompt: Name of validation prompt template
            auto_retry: Whether to automatically retry on validation failure
            retry_limit: Maximum number of retries if auto_retry is True
        """
        self.llm: AsyncLLMProvider = llm
        self.builder: AsyncPromptBuilder = prompt_builder
        self.validation_prompt = validation_prompt
        self.auto_retry = auto_retry
        self.retry_limit = retry_limit

    async def process_request(
        self,
        messages: List[LLMMessage],
        state: ConversationState
    ) -> List[LLMMessage]:
        """Pass through requests without validation."""
        return messages

    async def process_response(
        self,
        response: LLMResponse,
        state: ConversationState
    ) -> LLMResponse:
        """Validate response by calling LLM with validation prompt."""
        # Render validation prompt with response content
        validation_prompt_result = await self.builder.render_user_prompt(
            self.validation_prompt,
            index=0,
            params={"response": response.content},
            include_rag=False  # Don't need RAG for validation
        )

        # Create message and call LLM to get validation judgment
        validation_message = LLMMessage(
            role="user",
            content=validation_prompt_result.content
        )
        validation_response = await self.llm.complete([validation_message])

        # Check if LLM says response is valid
        is_valid = self._check_validity(validation_response.content)

        if not is_valid:
            # Track validation failure
            if not response.metadata:
                response.metadata = {}
            response.metadata["validation_failed"] = True
            response.metadata["validation_response"] = validation_response.content

            if self.auto_retry:
                # Note: Actual retry logic would need to be implemented
                # at the ConversationManager level. This just marks the failure.
                response.metadata["retry_requested"] = True
            else:
                raise ValueError(
                    f"Response failed validation: {validation_response.content}"
                )

        return response

    def _check_validity(self, validation_response: str) -> bool:
        """Check if validation response indicates success.

        Args:
            validation_response: Content from validation prompt response

        Returns:
            True if valid, False otherwise
        """
        # Simple implementation: look for "VALID" in response
        # This can be customized based on validation prompt design
        return "VALID" in validation_response.upper()


class MetadataMiddleware(ConversationMiddleware):
    """Middleware that adds custom metadata to messages and responses.

    This middleware can inject metadata into both requests and responses,
    which is useful for tracking, analytics, and debugging.

    Example:
        >>> from datetime import datetime
        >>>
        >>> # Add environment info to all messages
        >>> middleware = MetadataMiddleware(
        ...     request_metadata={"environment": "production"},
        ...     response_metadata={"version": "1.0.0"}
        ... )
        >>>
        >>> # Add dynamic metadata via callback
        >>> def get_request_meta():
        ...     return {"timestamp": datetime.now().isoformat()}
        >>>
        >>> middleware = MetadataMiddleware(
        ...     request_metadata_fn=get_request_meta
        ... )
    """

    def __init__(
        self,
        request_metadata: Dict[str, Any] | None = None,
        response_metadata: Dict[str, Any] | None = None,
        request_metadata_fn: Callable[..., Dict[str, Any]] | None = None,
        response_metadata_fn: Callable[..., Dict[str, Any]] | None = None
    ):
        """Initialize metadata middleware.

        Args:
            request_metadata: Static metadata to add to requests
            response_metadata: Static metadata to add to responses
            request_metadata_fn: Callable that returns metadata for requests
            response_metadata_fn: Callable that returns metadata for responses
        """
        self.request_metadata = request_metadata or {}
        self.response_metadata = response_metadata or {}
        self.request_metadata_fn = request_metadata_fn
        self.response_metadata_fn = response_metadata_fn

    async def process_request(
        self,
        messages: List[LLMMessage],
        state: ConversationState
    ) -> List[LLMMessage]:
        """Add metadata to request messages."""
        # Collect metadata to add
        metadata_to_add = dict(self.request_metadata)

        # Add dynamic metadata if function provided
        if self.request_metadata_fn:
            dynamic_metadata = self.request_metadata_fn()
            metadata_to_add.update(dynamic_metadata)

        # Add metadata to each message
        if metadata_to_add:
            for msg in messages:
                if not msg.metadata:
                    msg.metadata = {}
                msg.metadata.update(metadata_to_add)

        return messages

    async def process_response(
        self,
        response: LLMResponse,
        state: ConversationState
    ) -> LLMResponse:
        """Add metadata to response."""
        # Collect metadata to add
        metadata_to_add = dict(self.response_metadata)

        # Add dynamic metadata if function provided
        if self.response_metadata_fn:
            dynamic_metadata = self.response_metadata_fn()
            metadata_to_add.update(dynamic_metadata)

        # Add metadata to response
        if metadata_to_add:
            if not response.metadata:
                response.metadata = {}
            response.metadata.update(metadata_to_add)

        return response


class RateLimitMiddleware(ConversationMiddleware):
    """Middleware that enforces rate limiting on LLM requests.

    This middleware tracks request rates per conversation or per client
    and raises an exception when the rate limit is exceeded. Delegates
    core rate limiting logic to ``InMemoryRateLimiter`` from
    ``dataknobs_common.ratelimit``.

    Example:
        >>> # Limit to 10 requests per minute
        >>> middleware = RateLimitMiddleware(
        ...     max_requests=10,
        ...     window_seconds=60
        ... )
        >>>
        >>> # Per-client rate limiting
        >>> middleware = RateLimitMiddleware(
        ...     max_requests=100,
        ...     window_seconds=3600,
        ...     scope="client_id"  # Rate limit per client
        ... )
        >>>
        >>> # With custom key function
        >>> def get_user_id(state):
        ...     return state.metadata.get("user_id")
        >>>
        >>> middleware = RateLimitMiddleware(
        ...     max_requests=50,
        ...     window_seconds=60,
        ...     key_fn=get_user_id
        ... )
    """

    def __init__(
        self,
        max_requests: int,
        window_seconds: int = 60,
        scope: str = "conversation",  # "conversation" or "client_id"
        key_fn: Callable[[ConversationState], str] | None = None
    ):
        """Initialize rate limiting middleware.

        Args:
            max_requests: Maximum number of requests allowed in window
            window_seconds: Time window in seconds for rate limiting
            scope: Scope for rate limiting ("conversation" or "client_id")
            key_fn: Optional custom function to extract rate limit key from state
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.scope = scope
        self.key_fn = key_fn

        # Delegate core rate limiting to common InMemoryRateLimiter
        config = RateLimiterConfig(
            default_rates=[RateLimit(limit=max_requests, interval=window_seconds)],
        )
        self._limiter = InMemoryRateLimiter(config)

    def _get_rate_limit_key(self, state: ConversationState) -> str:
        """Get the key to use for rate limiting.

        Args:
            state: Conversation state

        Returns:
            Rate limit key
        """
        if self.key_fn:
            return self.key_fn(state)
        elif self.scope == "client_id":
            return state.metadata.get("client_id", state.conversation_id)
        else:
            return state.conversation_id

    async def process_request(
        self,
        messages: List[LLMMessage],
        state: ConversationState
    ) -> List[LLMMessage]:
        """Check rate limit before allowing request through."""
        key = self._get_rate_limit_key(state)

        if not await self._limiter.try_acquire(key):
            status = await self._limiter.get_status(key)
            # Add rate limit info to state metadata for debugging
            if not state.metadata:
                state.metadata = {}
            state.metadata["rate_limit_exceeded"] = True
            state.metadata["rate_limit_count"] = status.current_count
            state.metadata["rate_limit_max"] = status.limit
            state.metadata["rate_limit_window"] = self.window_seconds

            raise RateLimitError(
                f"Rate limit exceeded: {status.current_count}/{status.limit} "
                f"requests in {self.window_seconds}s window",
                retry_after=status.reset_after,
            )

        # Add rate limit info to messages metadata
        status = await self._limiter.get_status(key)
        for msg in messages:
            if not msg.metadata:
                msg.metadata = {}
            msg.metadata["rate_limit_count"] = status.current_count
            msg.metadata["rate_limit_max"] = status.limit

        return messages

    async def process_response(
        self,
        response: LLMResponse,
        state: ConversationState
    ) -> LLMResponse:
        """Add rate limit info to response metadata."""
        key = self._get_rate_limit_key(state)
        status = await self._limiter.get_status(key)

        if not response.metadata:
            response.metadata = {}

        response.metadata["rate_limit_count"] = status.current_count
        response.metadata["rate_limit_max"] = status.limit
        response.metadata["rate_limit_remaining"] = status.remaining

        return response

    async def get_rate_limit_status(self, key: str) -> dict[str, Any]:
        """Get current rate limit status for a key.

        Args:
            key: Rate limit key

        Returns:
            Dictionary with rate limit status

        Example:
            >>> status = await middleware.get_rate_limit_status("client-abc")
            >>> print(status)
            {
                'current_count': 5,
                'max_requests': 10,
                'remaining': 5,
                'window_seconds': 60,
                'next_reset': 45.2
            }
        """
        status = await self._limiter.get_status(key)
        return {
            'current_count': status.current_count,
            'max_requests': status.limit,
            'remaining': status.remaining,
            'window_seconds': self.window_seconds,
            'next_reset': status.reset_after,
        }

    async def reset(self, key: str | None = None) -> None:
        """Reset rate limit for a specific key or all keys.

        Args:
            key: Key to reset. If None, resets all keys.

        Example:
            >>> # Reset specific client
            >>> await middleware.reset("client-abc")
            >>>
            >>> # Reset all
            >>> await middleware.reset()
        """
        await self._limiter.reset(key)


class PromoteToPersistMiddleware(ConversationMiddleware):
    """Promote allowlisted flat ``response.metadata`` keys into ``_persist``.

    By default, middleware and provider writes to ``response.metadata`` are
    ephemeral — they live on the :class:`~dataknobs_llm.llm.base.LLMResponse`
    for this call but do not flow to the persisted assistant conversation
    node (see :class:`ConversationMiddleware` "Persisting Metadata" for the
    ``_persist`` namespace contract).

    This middleware bridges the gap for writers that cannot (or should not)
    be modified at the source: third-party providers, in-tree middleware
    whose writes consumers want preserved per-deployment
    (``llm_duration_seconds`` from a custom timing middleware,
    ``rate_limit_count`` from :class:`RateLimitMiddleware`, Ollama's
    ``eval_duration``/``total_duration``, etc.).

    Example:
        ```python
        manager = await ConversationManager.create(
            llm=ollama_provider,
            prompt_builder=builder,
            middleware=[
                # Position-[0] — runs last on response, after other
                # middleware have written their flat keys.
                PromoteToPersistMiddleware(keys=[
                    "rate_limit_count",     # from RateLimitMiddleware
                    "eval_duration",        # from Ollama provider
                    "total_duration",       # from Ollama provider
                ]),
                RateLimitMiddleware(max_requests=10, window_seconds=60),
            ],
        )
        ```

    After the LLM call, each listed key present in ``response.metadata`` is
    copied into ``response.metadata["_persist"]``; the framework then merges
    ``_persist`` into the assistant node's metadata. Missing keys are
    silently skipped (the writer may not have run, or the value may be
    conditional).

    Ordering Requirement:
        Place this middleware at position ``[0]`` of the ``middleware`` list
        so it runs **last** on ``process_response`` — after every other
        middleware has written its flat keys. Onion-execution runs
        ``process_response`` in reverse order: ``middleware[-1]`` first,
        ``middleware[0]`` last. Placing the promoter at the end of the list
        would run it **first** on response, before other middleware has
        written — and it would capture nothing.

        Provider writes (e.g., Ollama's ``eval_duration``) are populated on
        ``response.metadata`` before any middleware runs, so position does
        not matter for them.

    Collision with an existing ``_persist`` write:
        If another middleware has already written a same-named key into
        ``_persist``, the promoter uses ``setdefault`` and does not
        overwrite — native ``_persist`` writes take precedence over passive
        promotion.

        Note: this is distinct from collision between two native
        ``_persist`` writers. Those use ``persist[key] = ...`` or
        ``persist.update(...)`` and thus follow onion ordering — the outer
        middleware (earlier in the ``middleware`` list) runs last on
        response and its write wins.

    Args:
        keys: List of flat ``response.metadata`` key names to promote into
            the ``_persist`` namespace. Order within the list is not
            significant.
    """

    def __init__(self, keys: list[str]) -> None:
        """Initialize with the list of flat metadata keys to promote.

        Args:
            keys: Flat ``response.metadata`` keys to copy into ``_persist``
                on each response.
        """
        self._keys = list(keys)

    async def process_request(
        self,
        messages: List[LLMMessage],
        state: ConversationState,
    ) -> List[LLMMessage]:
        """No-op. Promotion is a response-time concern."""
        return messages

    async def process_response(
        self,
        response: LLMResponse,
        state: ConversationState,
    ) -> LLMResponse:
        """Copy allowlisted flat keys into ``response.metadata['_persist']``."""
        if not self._keys:
            return response
        if response.metadata is None:
            response.metadata = {}
        persist = response.metadata.setdefault("_persist", {})
        for key in self._keys:
            if key == "_persist":
                continue  # Defensive: skip the namespace marker itself.
            if key in response.metadata:
                persist.setdefault(key, response.metadata[key])
        return response
