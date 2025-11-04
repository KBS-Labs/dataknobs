"""In-memory resource adapters for testing and demos.

This module provides simple in-memory adapters that return predefined results.
Useful for testing, demos, and examples where you need predictable behavior
without external dependencies.
"""

from typing import Any, Dict, List
from .resource_adapter import ResourceAdapterBase, ResourceAdapter, AsyncResourceAdapter, BaseSearchLogic


class InMemoryAdapterBase(ResourceAdapterBase):
    """Base class with shared logic for in-memory adapters.

    This class contains all the core logic shared between sync and async
    in-memory adapters, following the DRY principle.
    """

    def __init__(
        self,
        search_results: List[Dict[str, Any]] | None = None,
        data: Dict[str, Any] | None = None,
        name: str = "inmemory"
    ):
        """Initialize in-memory adapter base.

        Args:
            search_results: List of results to return from search().
                Each result should be a dict with at least a content key.
                Optional keys: score, metadata, etc.
            data: Optional dictionary for key-value storage (used by get_value)
            name: Name identifier for this adapter
        """
        super().__init__(name=name)
        self._search_results = search_results or []
        self._data = data or {}
        self.search_count = 0  # Track number of search calls for testing

    def reset(self):
        """Reset search count. Useful for testing cache behavior."""
        self.search_count = 0

    def _get_value_impl(
        self,
        key: str,
        default: Any = None,
        context: Dict[str, Any] | None = None
    ) -> Any:
        """Shared implementation for get_value.

        Args:
            key: Key to look up
            default: Value to return if key is not found
            context: Optional context (unused in in-memory adapter)

        Returns:
            Value at the key, or default if not found
        """
        return self._data.get(key, default)

    def _search_impl(
        self,
        query: str,
        k: int = 5,
        filters: Dict[str, Any] | None = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Shared implementation for search.

        Args:
            query: Search query (unused - returns all configured results)
            k: Maximum number of results to return
            filters: Optional filters to apply via BaseSearchLogic
            **kwargs: Additional options:
                - min_score: Minimum score threshold

        Returns:
            List of search results (up to k items)
        """
        self.search_count += 1
        results = self._search_results.copy()

        # Ensure results have proper structure
        results = [
            BaseSearchLogic.format_search_result(
                r.get("content", ""),
                score=r.get("score", 1.0),
                metadata=r.get("metadata", {})
            )
            for r in results
        ]

        # Apply filters if provided
        if filters:
            results = BaseSearchLogic.filter_results(results, filters=filters)

        # Apply min_score filter if provided
        min_score = kwargs.get('min_score', 0.0)
        if min_score > 0:
            results = BaseSearchLogic.filter_results(results, min_score=min_score)

        return results[:k]


class InMemoryAdapter(InMemoryAdapterBase, ResourceAdapter):
    """Synchronous in-memory adapter with predefined search results.

    This adapter is designed for testing, demos, and examples. It returns
    predefined search results and optionally stores key-value pairs.

    Features:
    - Predefined search results for predictable testing
    - Optional key-value storage via get_value()
    - Simple and fast (no external I/O)
    - Search count tracking for testing

    Example:
        >>> # Simple usage with search results
        >>> adapter = InMemoryAdapter(
        ...     search_results=[
        ...         {'content': "Python is a programming language", 'score': 0.9},
        ...         {'content': "Python was created by Guido van Rossum", 'score': 0.8}
        ...     ],
        ...     name="docs"
        ... )
        >>> results = adapter.search("python")
        >>> len(results)
        2
        >>> first_result = results[0]
        >>> first_result.get('content')
        'Python is a programming language'
        >>> adapter.search_count  # Track how many times search was called
        1

        >>> # With key-value storage
        >>> adapter = InMemoryAdapter(
        ...     data={'language': "Python", 'version': "3.11"},
        ...     name="config"
        ... )
        >>> adapter.get_value('language')
        'Python'
    """

    def __init__(
        self,
        search_results: List[Dict[str, Any]] | None = None,
        data: Dict[str, Any] | None = None,
        name: str = "inmemory"
    ):
        """Initialize synchronous in-memory adapter.

        Args:
            search_results: List of results to return from search()
            data: Optional dictionary for key-value storage
            name: Name identifier for this adapter
        """
        super().__init__(search_results=search_results, data=data, name=name)

    def get_value(
        self,
        key: str,
        default: Any = None,
        context: Dict[str, Any] | None = None
    ) -> Any:
        """Retrieve a value by key from the in-memory data.

        Args:
            key: Key to look up
            default: Value to return if key is not found
            context: Optional context (unused in in-memory adapter)

        Returns:
            Value at the key, or default if not found
        """
        return self._get_value_impl(key, default, context)

    def search(
        self,
        query: str,
        k: int = 5,
        filters: Dict[str, Any] | None = None,
        **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """Return predefined search results.

        Args:
            query: Search query (unused - returns all configured results)
            k: Maximum number of results to return
            filters: Optional filters to apply via BaseSearchLogic
            **kwargs: Additional options:
                - min_score: Minimum score threshold

        Returns:
            List of search results (up to k items)
        """
        return self._search_impl(query, k, filters, **kwargs)


class InMemoryAsyncAdapter(InMemoryAdapterBase, AsyncResourceAdapter):
    """Asynchronous in-memory adapter with predefined search results.

    Async version of InMemoryAdapter. Useful for testing async code paths
    or maintaining consistency in async codebases.

    Example:
        >>> adapter = InMemoryAsyncAdapter(
        ...     search_results=[
        ...         {'content': "Result 1", 'score': 0.9},
        ...         {'content': "Result 2", 'score': 0.8}
        ...     ]
        ... )
        >>> results = await adapter.search("test")
        >>> len(results)
        2
        >>> adapter.search_count
        1
    """

    def __init__(
        self,
        search_results: List[Dict[str, Any]] | None = None,
        data: Dict[str, Any] | None = None,
        name: str = "inmemory_async"
    ):
        """Initialize asynchronous in-memory adapter.

        Args:
            search_results: List of results to return from search()
            data: Optional dictionary for key-value storage
            name: Name identifier for this adapter
        """
        super().__init__(search_results=search_results, data=data, name=name)

    async def get_value(
        self,
        key: str,
        default: Any = None,
        context: Dict[str, Any] | None = None
    ) -> Any:
        """Retrieve a value by key from the in-memory data (async).

        Args:
            key: Key to look up
            default: Value to return if key is not found
            context: Optional context (unused in in-memory adapter)

        Returns:
            Value at the key, or default if not found
        """
        return self._get_value_impl(key, default, context)

    async def search(
        self,
        query: str,
        k: int = 5,
        filters: Dict[str, Any] | None = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Return predefined search results (async).

        Args:
            query: Search query (unused - returns all configured results)
            k: Maximum number of results to return
            filters: Optional filters to apply via BaseSearchLogic
            **kwargs: Additional options:
                - min_score: Minimum score threshold

        Returns:
            List of search results (up to k items)
        """
        return self._search_impl(query, k, filters, **kwargs)
