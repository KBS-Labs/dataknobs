"""Buffer memory implementation for simple FIFO message storage."""

from collections import deque
from typing import Any

from .base import Memory


class BufferMemory(Memory):
    """Simple buffer memory keeping last N messages.

    This implementation uses a fixed-size buffer that keeps the most recent
    messages in memory. When the buffer is full, the oldest messages are
    automatically removed.

    Attributes:
        max_messages: Maximum number of messages to keep in buffer
        messages: Deque containing the messages
    """

    def __init__(self, max_messages: int = 10):
        """Initialize buffer memory.

        Args:
            max_messages: Maximum number of messages to keep
        """
        self.max_messages = max_messages
        self.messages: deque[dict[str, Any]] = deque(maxlen=max_messages)

    async def add_message(
        self, content: str, role: str, metadata: dict[str, Any] | None = None
    ) -> None:
        """Add message to buffer.

        Args:
            content: Message content
            role: Message role
            metadata: Optional metadata
        """
        self.messages.append({"content": content, "role": role, "metadata": metadata or {}})

    async def get_context(self, current_message: str) -> list[dict[str, Any]]:
        """Get all messages in buffer.

        The current_message parameter is not used in buffer memory since
        we simply return all buffered messages in order.

        Args:
            current_message: Not used in buffer memory

        Returns:
            List of all buffered messages
        """
        return list(self.messages)

    async def clear(self) -> None:
        """Clear all messages from buffer."""
        self.messages.clear()
