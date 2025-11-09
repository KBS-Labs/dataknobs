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

from abc import ABC, abstractmethod
from typing import List, Any, Dict, Callable
import logging

from dataknobs_llm.llm import LLMMessage, LLMResponse
from dataknobs_llm.llm.providers import AsyncLLMProvider
from dataknobs_llm.conversations.storage import ConversationState
from dataknobs_llm.prompts import AsyncPromptBuilder
from dataknobs_llm.exceptions import RateLimitError


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
                # Case-insensitive replacement
                import re
                pattern = re.compile(re.escape(word), re.IGNORECASE)
                content = pattern.sub(self.replacement, content)

        # Track if any filtering occurred
        if content != response.content:
            if not response.metadata:
                response.metadata = {}
            response.metadata["content_filtered"] = True
            response.content = content

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
    and raises an exception when the rate limit is exceeded. Rate limits
    are tracked in-memory using a sliding window algorithm.

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

        # In-memory storage: key -> list of request timestamps
        self._request_history: Dict[str, List[float]] = {}

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

    def _clean_old_requests(self, key: str, current_time: float) -> None:
        """Remove requests outside the time window.

        Args:
            key: Rate limit key
            current_time: Current timestamp
        """
        if key in self._request_history:
            cutoff_time = current_time - self.window_seconds
            self._request_history[key] = [
                ts for ts in self._request_history[key]
                if ts > cutoff_time
            ]

    def _check_rate_limit(self, key: str, current_time: float) -> tuple[bool, int]:
        """Check if request is within rate limit.

        Args:
            key: Rate limit key
            current_time: Current timestamp

        Returns:
            Tuple of (is_allowed, current_count)
        """
        # Clean old requests
        self._clean_old_requests(key, current_time)

        # Check current count
        if key not in self._request_history:
            self._request_history[key] = []

        current_count = len(self._request_history[key])
        is_allowed = current_count < self.max_requests

        return is_allowed, current_count

    def _record_request(self, key: str, current_time: float) -> None:
        """Record a new request.

        Args:
            key: Rate limit key
            current_time: Current timestamp
        """
        if key not in self._request_history:
            self._request_history[key] = []

        self._request_history[key].append(current_time)

    async def process_request(
        self,
        messages: List[LLMMessage],
        state: ConversationState
    ) -> List[LLMMessage]:
        """Check rate limit before allowing request through."""
        import time

        current_time = time.time()
        key = self._get_rate_limit_key(state)

        # Check rate limit
        is_allowed, current_count = self._check_rate_limit(key, current_time)

        if not is_allowed:
            # Add rate limit info to state metadata for debugging
            if not state.metadata:
                state.metadata = {}
            state.metadata["rate_limit_exceeded"] = True
            state.metadata["rate_limit_count"] = current_count
            state.metadata["rate_limit_max"] = self.max_requests
            state.metadata["rate_limit_window"] = self.window_seconds

            raise RateLimitError(
                f"Rate limit exceeded: {current_count}/{self.max_requests} "
                f"requests in {self.window_seconds}s window"
            )

        # Record this request
        self._record_request(key, current_time)

        # Add rate limit info to messages metadata
        for msg in messages:
            if not msg.metadata:
                msg.metadata = {}
            msg.metadata["rate_limit_count"] = current_count + 1
            msg.metadata["rate_limit_max"] = self.max_requests

        return messages

    async def process_response(
        self,
        response: LLMResponse,
        state: ConversationState
    ) -> LLMResponse:
        """Add rate limit info to response metadata."""
        key = self._get_rate_limit_key(state)

        if key in self._request_history:
            current_count = len(self._request_history[key])

            if not response.metadata:
                response.metadata = {}

            response.metadata["rate_limit_count"] = current_count
            response.metadata["rate_limit_max"] = self.max_requests
            response.metadata["rate_limit_remaining"] = self.max_requests - current_count

        return response

    def get_rate_limit_status(self, key: str) -> Dict[str, Any]:
        """Get current rate limit status for a key.

        Args:
            key: Rate limit key

        Returns:
            Dictionary with rate limit status

        Example:
            >>> status = middleware.get_rate_limit_status("client-abc")
            >>> print(status)
            {
                'current_count': 5,
                'max_requests': 10,
                'remaining': 5,
                'window_seconds': 60,
                'next_reset': 45.2  # seconds until oldest request expires
            }
        """
        import time

        current_time = time.time()
        self._clean_old_requests(key, current_time)

        if key not in self._request_history or not self._request_history[key]:
            return {
                'current_count': 0,
                'max_requests': self.max_requests,
                'remaining': self.max_requests,
                'window_seconds': self.window_seconds,
                'next_reset': 0
            }

        current_count = len(self._request_history[key])
        oldest_request = min(self._request_history[key])
        next_reset = max(0, (oldest_request + self.window_seconds) - current_time)

        return {
            'current_count': current_count,
            'max_requests': self.max_requests,
            'remaining': max(0, self.max_requests - current_count),
            'window_seconds': self.window_seconds,
            'next_reset': next_reset
        }

    def reset(self, key: str | None = None) -> None:
        """Reset rate limit for a specific key or all keys.

        Args:
            key: Key to reset. If None, resets all keys.

        Example:
            >>> # Reset specific client
            >>> middleware.reset("client-abc")
            >>>
            >>> # Reset all
            >>> middleware.reset()
        """
        if key is None:
            self._request_history.clear()
        elif key in self._request_history:
            del self._request_history[key]
