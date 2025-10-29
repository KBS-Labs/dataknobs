"""Middleware system for conversation processing.

This module provides middleware capabilities for processing messages before they
are sent to the LLM and processing responses after they come back from the LLM.
Middleware can be used for logging, validation, content filtering, and more.

Example:
    >>> from dataknobs_llm.conversations import (
    ...     ConversationManager,
    ...     LoggingMiddleware,
    ...     ValidationMiddleware
    ... )
    >>> import logging
    >>>
    >>> # Create middleware instances
    >>> logger = logging.getLogger(__name__)
    >>> logging_mw = LoggingMiddleware(logger)
    >>> validation_mw = ValidationMiddleware(
    ...     prompt_builder=builder,
    ...     validation_prompt="validate_response"
    ... )
    >>>
    >>> # Create conversation with middleware
    >>> manager = await ConversationManager.create(
    ...     llm=llm,
    ...     prompt_builder=builder,
    ...     storage=storage,
    ...     middleware=[logging_mw, validation_mw]
    ... )
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Any, Dict
import logging

from dataknobs_llm.llm import LLMMessage, LLMResponse
from dataknobs_llm.conversations.storage import ConversationState


class ConversationMiddleware(ABC):
    """Base class for conversation middleware.

    Middleware can process requests before LLM and responses after LLM.
    Middleware is executed in order for requests, and in reverse order
    for responses (like an onion).

    Example:
        >>> class CustomMiddleware(ConversationMiddleware):
        ...     async def process_request(self, messages, state):
        ...         # Add custom processing before LLM
        ...         return messages
        ...
        ...     async def process_response(self, response, state):
        ...         # Add custom processing after LLM
        ...         return response
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

    def __init__(self, logger: Optional[logging.Logger] = None):
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
        >>> # Create validation middleware
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
        llm: "AsyncLLMProvider",
        prompt_builder: "AsyncPromptBuilder",
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
        from dataknobs_llm.prompts import AsyncPromptBuilder
        from dataknobs_llm.llm import AsyncLLMProvider

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
        request_metadata: Optional[Dict[str, Any]] = None,
        response_metadata: Optional[Dict[str, Any]] = None,
        request_metadata_fn: Optional[callable] = None,
        response_metadata_fn: Optional[callable] = None
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
