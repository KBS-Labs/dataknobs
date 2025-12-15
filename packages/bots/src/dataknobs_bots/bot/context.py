"""Bot execution context."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BotContext:
    """Runtime context for bot execution.

    Supports dict-like access for dynamic attributes via request_metadata.
    Use `context["key"]` or `context.get("key")` for dynamic data.

    Attributes:
        conversation_id: Unique identifier for the conversation
        client_id: Identifier for the client/tenant
        user_id: Optional user identifier
        session_metadata: Metadata for the session
        request_metadata: Metadata for the current request (also used for dict-like access)
    """

    conversation_id: str
    client_id: str
    user_id: str | None = None
    session_metadata: dict[str, Any] = field(default_factory=dict)
    request_metadata: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        """Get item from request_metadata using dict-like access.

        Args:
            key: Key to retrieve

        Returns:
            Value from request_metadata

        Raises:
            KeyError: If key not found in request_metadata
        """
        return self.request_metadata[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """Set item in request_metadata using dict-like access.

        Args:
            key: Key to set
            value: Value to store
        """
        self.request_metadata[key] = value

    def __contains__(self, key: str) -> bool:
        """Check if key exists in request_metadata.

        Args:
            key: Key to check

        Returns:
            True if key exists in request_metadata
        """
        return key in self.request_metadata

    def get(self, key: str, default: Any = None) -> Any:
        """Get item from request_metadata with optional default.

        Args:
            key: Key to retrieve
            default: Default value if key not found

        Returns:
            Value from request_metadata or default
        """
        return self.request_metadata.get(key, default)

    def copy(self, **overrides: Any) -> "BotContext":
        """Create a copy of this context with optional field overrides.

        Creates shallow copies of session_metadata and request_metadata dicts
        to avoid mutation issues between the original and copy.

        Args:
            **overrides: Field values to override in the copy

        Returns:
            New BotContext instance with copied values

        Example:
            >>> ctx = BotContext(conversation_id="conv-1", client_id="client-1")
            >>> ctx2 = ctx.copy(conversation_id="conv-2")
            >>> ctx2.conversation_id
            'conv-2'
        """
        return BotContext(
            conversation_id=overrides.get("conversation_id", self.conversation_id),
            client_id=overrides.get("client_id", self.client_id),
            user_id=overrides.get("user_id", self.user_id),
            session_metadata=overrides.get(
                "session_metadata", dict(self.session_metadata)
            ),
            request_metadata=overrides.get(
                "request_metadata", dict(self.request_metadata)
            ),
        )
