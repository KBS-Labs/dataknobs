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

    def providers(self) -> dict[str, Any]:
        """Return LLM providers managed by this memory, keyed by role.

        Subsystems declare the providers they own so that the bot can
        register them in the provider catalog without reaching into
        private attributes.  The default returns an empty dict (no
        providers).

        Returns:
            Dict mapping provider role names to provider instances.
        """
        return {}

    def set_provider(self, role: str, provider: Any) -> bool:
        """Replace a provider managed by this memory.

        Called by ``inject_providers`` to wire a test provider into the
        actual subsystem, not just the registry catalog.  The default
        returns ``False`` (role not recognized).  Concrete subclasses
        override to accept their known roles.

        Args:
            role: Provider role name (e.g. ``PROVIDER_ROLE_MEMORY_EMBEDDING``).
            provider: Replacement provider instance.

        Returns:
            ``True`` if the role was recognized and the provider updated,
            ``False`` otherwise.
        """
        return False

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
