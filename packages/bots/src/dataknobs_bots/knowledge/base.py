"""Base knowledge base interface for DynaBot."""

from abc import ABC, abstractmethod
from typing import Any


class KnowledgeBase(ABC):
    """Abstract base class for knowledge base implementations.

    Defines the interface that DynaBot depends on for knowledge-backed
    context retrieval.  Mirrors the pattern established by ``Memory``
    and ``ReasoningStrategy`` — abstract methods for core operations,
    non-abstract methods with sensible defaults for provider registry
    integration.
    """

    @abstractmethod
    async def query(
        self,
        query: str,
        k: int = 5,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Query the knowledge base for relevant content.

        Args:
            query: Query text to search for.
            k: Number of results to return.
            **kwargs: Implementation-specific parameters.

        Returns:
            List of result dicts (at minimum containing ``"text"``).
        """

    @abstractmethod
    async def close(self) -> None:
        """Close the knowledge base and release resources."""

    def format_context(
        self,
        results: list[dict[str, Any]],
        wrap_in_tags: bool = True,
    ) -> str:
        """Format query results for inclusion in an LLM prompt.

        The default implementation joins result texts with newlines.
        Subclasses may override for richer formatting.

        Args:
            results: Query results from :meth:`query`.
            wrap_in_tags: Whether to wrap output in XML-style tags.

        Returns:
            Formatted context string.
        """
        texts = [r.get("text", "") for r in results]
        body = "\n\n".join(texts)
        if wrap_in_tags:
            return f"<knowledge_context>\n{body}\n</knowledge_context>"
        return body

    def providers(self) -> dict[str, Any]:
        """Return LLM providers managed by this knowledge base, keyed by role.

        Subsystems declare the providers they own so that the bot can
        register them in the provider catalog without reaching into
        private attributes.  The default returns an empty dict.

        Returns:
            Dict mapping provider role names to provider instances.
        """
        return {}

    def set_provider(self, role: str, provider: Any) -> bool:
        """Replace a provider managed by this knowledge base.

        Called by ``inject_providers`` to wire a test provider into the
        actual subsystem, not just the registry catalog.  The default
        returns ``False`` (role not recognized).

        Args:
            role: Provider role name.
            provider: Replacement provider instance.

        Returns:
            ``True`` if the role was recognized and the provider
            updated, ``False`` otherwise.
        """
        return False
