"""Resource adapter interfaces for pluggable data sources.

This module defines the adapter pattern for accessing external resources
(databases, vector stores, configuration systems, etc.) in both synchronous
and asynchronous contexts.

Key concepts:
- Separate sync (ResourceAdapter) and async (AsyncResourceAdapter) interfaces
- Shared base class (ResourceAdapterBase) for common functionality
- BaseSearchLogic for reusable search operations
- No mixing of sync/async - builders require matching adapter types
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List
import logging

logger = logging.getLogger(__name__)


class ResourceAdapterBase:
    """Base class with shared functionality for both sync and async adapters.

    This class provides:
    - Adapter name and metadata management
    - Metadata caching
    - Helper methods for type checking
    - Common initialization logic
    """

    def __init__(self, name: str = "adapter", metadata: Dict[str, Any] | None = None):
        """Initialize the resource adapter base.

        Args:
            name: Adapter identifier (used in logs and error messages)
            metadata: Optional metadata about this adapter
        """
        self._name = name
        self._metadata = metadata or {}
        self._metadata_cache: Dict[str, Any] | None = None

    @property
    def name(self) -> str:
        """Get the adapter name."""
        return self._name

    def is_async(self) -> bool:
        """Check if this is an async adapter.

        Returns:
            True if this adapter implements AsyncResourceAdapter
        """
        return isinstance(self, AsyncResourceAdapter)

    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about this adapter.

        Returns:
            Dictionary with adapter metadata
        """
        return {
            "name": self._name,
            "type": "async" if self.is_async() else "sync",
            "class": self.__class__.__name__,
            **self._metadata
        }

    def __repr__(self) -> str:
        """Return a string representation of this adapter."""
        adapter_type = "async" if self.is_async() else "sync"
        return f"{self.__class__.__name__}(name={self._name!r}, type={adapter_type})"


class ResourceAdapter(ResourceAdapterBase, ABC):
    """Synchronous resource adapter interface.

    Adapters implementing this interface provide synchronous access to
    external resources for parameter resolution and RAG searches.

    All methods are synchronous (blocking).
    """

    @abstractmethod
    def get_value(
        self,
        key: str,
        default: Any = None,
        context: Dict[str, Any] | None = None
    ) -> Any:
        """Retrieve a value by key from the resource.

        Args:
            key: The key to look up
            default: Default value if key not found
            context: Optional context for the lookup (e.g., user ID, session)

        Returns:
            The value associated with the key, or default if not found
        """
        pass

    @abstractmethod
    def search(
        self,
        query: str,
        k: int = 5,
        filters: Dict[str, Any] | None = None,
        **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """Perform a search query against the resource.

        Args:
            query: Search query string
            k: Number of results to return (default: 5)
            filters: Optional filters to apply to the search
            **kwargs: Additional adapter-specific search parameters

        Returns:
            List of search results as dictionaries
        """
        pass

    def batch_get_values(
        self,
        keys: List[str],
        default: Any = None,
        context: Dict[str, Any] | None = None
    ) -> Dict[str, Any]:
        """Retrieve multiple values by keys.

        Default implementation calls get_value() for each key.
        Adapters can override for more efficient batch operations.

        Args:
            keys: List of keys to look up
            default: Default value for keys not found
            context: Optional context for the lookup

        Returns:
            Dictionary mapping keys to their values
        """
        return {key: self.get_value(key, default, context) for key in keys}


class AsyncResourceAdapter(ResourceAdapterBase, ABC):
    """Asynchronous resource adapter interface.

    Adapters implementing this interface provide asynchronous access to
    external resources for parameter resolution and RAG searches.

    All methods are asynchronous (non-blocking).
    """

    @abstractmethod
    async def get_value(
        self,
        key: str,
        default: Any = None,
        context: Dict[str, Any] | None = None
    ) -> Any:
        """Retrieve a value by key from the resource (async).

        Args:
            key: The key to look up
            default: Default value if key not found
            context: Optional context for the lookup (e.g., user ID, session)

        Returns:
            The value associated with the key, or default if not found
        """
        pass

    @abstractmethod
    async def search(
        self,
        query: str,
        k: int = 5,
        filters: Dict[str, Any] | None = None,
        **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """Perform a search query against the resource (async).

        Args:
            query: Search query string
            k: Number of results to return (default: 5)
            filters: Optional filters to apply to the search
            **kwargs: Additional adapter-specific search parameters

        Returns:
            List of search results as dictionaries
        """
        pass

    async def batch_get_values(
        self,
        keys: List[str],
        default: Any = None,
        context: Dict[str, Any] | None = None
    ) -> Dict[str, Any]:
        """Retrieve multiple values by keys (async).

        Default implementation calls get_value() concurrently for each key.
        Adapters can override for more efficient batch operations.

        Args:
            keys: List of keys to look up
            default: Default value for keys not found
            context: Optional context for the lookup

        Returns:
            Dictionary mapping keys to their values
        """
        import asyncio
        tasks = [self.get_value(key, default, context) for key in keys]
        values = await asyncio.gather(*tasks)
        return dict(zip(keys, values, strict=True))


class BaseSearchLogic:
    """Shared search logic utilities for both sync and async adapters.

    This class provides helper methods for common search operations:
    - Result formatting and filtering
    - Score normalization
    - Result deduplication
    - Metadata extraction
    """

    @staticmethod
    def format_search_result(
        item: Any,
        score: float | None = None,
        metadata: Dict[str, Any] | None = None
    ) -> Dict[str, Any]:
        """Format a search result into a standardized dictionary.

        Args:
            item: The search result item (could be string, dict, or object)
            score: Optional relevance score
            metadata: Optional metadata about the result

        Returns:
            Formatted result dictionary with 'content', 'score', 'metadata'
        """
        result: Dict[str, Any] = {}

        # Extract content
        if isinstance(item, str):
            result["content"] = item
        elif isinstance(item, dict):
            # Try common content keys
            result["content"] = item.get("content") or item.get("text") or str(item)
            # Preserve other fields as metadata
            metadata = {**item, **(metadata or {})}
        else:
            result["content"] = str(item)

        # Add score if provided
        if score is not None:
            result["score"] = score

        # Add metadata if provided
        if metadata:
            result["metadata"] = metadata

        return result

    @staticmethod
    def filter_results(
        results: List[Dict[str, Any]],
        filters: Dict[str, Any] | None = None,
        min_score: float | None = None
    ) -> List[Dict[str, Any]]:
        """Filter search results based on criteria.

        Args:
            results: List of search result dictionaries
            filters: Dictionary of field filters (exact match)
            min_score: Minimum score threshold

        Returns:
            Filtered list of results
        """
        filtered = results

        # Apply score threshold
        if min_score is not None:
            filtered = [r for r in filtered if r.get("score", 0.0) >= min_score]

        # Apply field filters
        if filters:
            for key, value in filters.items():
                filtered = [
                    r for r in filtered
                    if r.get(key) == value or r.get("metadata", {}).get(key) == value
                ]

        return filtered

    @staticmethod
    def deduplicate_results(
        results: List[Dict[str, Any]],
        key: str = "content"
    ) -> List[Dict[str, Any]]:
        """Remove duplicate results based on a key.

        Args:
            results: List of search result dictionaries
            key: Key to use for deduplication (default: "content")

        Returns:
            Deduplicated list of results (preserves order, keeps first occurrence)
        """
        seen = set()
        deduplicated = []

        for result in results:
            value = result.get(key)
            if value not in seen:
                seen.add(value)
                deduplicated.append(result)

        return deduplicated
