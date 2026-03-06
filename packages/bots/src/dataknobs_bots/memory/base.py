"""Base memory interface for bot memory implementations."""

from abc import ABC, abstractmethod
from typing import Any


class Memory(ABC):
    """Abstract base class for memory implementations."""

    @abstractmethod
    async def add_message(
        self, content: str, role: str, metadata: dict[str, Any] | None = None
    ) -> None:
        """Add message to memory.

        Args:
            content: Message content
            role: Message role (user, assistant, system, etc.)
            metadata: Optional metadata for the message
        """
        pass

    @abstractmethod
    async def get_context(self, current_message: str) -> list[dict[str, Any]]:
        """Get relevant context for current message.

        Args:
            current_message: The current message to get context for

        Returns:
            List of relevant message dictionaries
        """
        pass

    @abstractmethod
    async def clear(self) -> None:
        """Clear all memory."""
        pass

    async def pop_messages(self, count: int = 2) -> list[dict[str, Any]]:
        """Remove and return the last N messages from memory.

        Used for conversation undo. The count is determined by the caller
        based on node depth difference (not a fixed 2).

        Args:
            count: Number of messages to remove from the end.

        Returns:
            The removed messages in the order they were stored.

        Raises:
            NotImplementedError: If the implementation does not support undo.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support pop_messages"
        )
