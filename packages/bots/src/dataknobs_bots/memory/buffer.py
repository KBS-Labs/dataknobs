"""Buffer memory implementation for simple FIFO message storage."""

from collections import deque
from typing import Any, ClassVar

from dataknobs_common.structured_config import StructuredConfigConsumer

from .base import Memory, apply_history_redactions, compile_history_redactions
from .config import BufferMemoryConfig


class BufferMemory(StructuredConfigConsumer[BufferMemoryConfig], Memory):
    """Simple buffer memory keeping last N messages.

    This implementation uses a fixed-size buffer that keeps the most recent
    messages in memory. When the buffer is full, the oldest messages are
    automatically removed.

    Construct from config (``BufferMemory.from_config({"max_messages": 5})``)
    or directly with loose kwargs (``BufferMemory(max_messages=5)`` /
    ``BufferMemory()`` for the default of 10) — both route through the
    structured-config dispatch.

    Attributes:
        max_messages: Maximum number of messages to keep in buffer
        messages: Deque containing the messages
    """

    CONFIG_CLS: ClassVar[type[BufferMemoryConfig]] = BufferMemoryConfig

    def _setup(self) -> None:
        """Build the FIFO buffer from the validated config."""
        self.max_messages = self.config.max_messages
        self.messages: deque[dict[str, Any]] = deque(
            maxlen=self.config.max_messages
        )
        self._compiled_redactions = compile_history_redactions(
            self.config.history_redactions
        )

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
        we simply return all buffered messages in order. Configured
        ``history_redactions`` are applied to assistant-role messages
        in the returned copy; the underlying deque is not mutated.

        Args:
            current_message: Not used in buffer memory

        Returns:
            List of all buffered messages, with assistant content redacted
            per the configured ``history_redactions`` (passthrough when
            none are configured).
        """
        return apply_history_redactions(self.messages, self._compiled_redactions)

    async def clear(self) -> None:
        """Clear all messages from buffer."""
        self.messages.clear()

    async def pop_messages(self, count: int = 2) -> list[dict[str, Any]]:
        """Remove and return the last N messages from the buffer.

        Args:
            count: Number of messages to remove from the end.

        Returns:
            The removed messages in the order they were stored.

        Raises:
            ValueError: If count exceeds available messages or is < 1.
        """
        if count < 1:
            raise ValueError(f"count must be >= 1, got {count}")
        if count > len(self.messages):
            raise ValueError(
                f"Cannot pop {count} messages, only {len(self.messages)} available"
            )
        removed = []
        for _ in range(count):
            removed.append(self.messages.pop())
        removed.reverse()
        return removed
