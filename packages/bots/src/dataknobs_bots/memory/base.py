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
