"""Dictionary-based resource adapters.

This module provides adapters that wrap Python dictionaries, enabling them to be
used as resource providers in the prompt library system. Supports both flat and
nested dictionaries with dot-notation key access.
"""

from typing import Any, Dict, List
from .resource_adapter import ResourceAdapter, AsyncResourceAdapter, BaseSearchLogic


class DictResourceAdapter(ResourceAdapter):
    """Synchronous adapter for Python dictionary resources.

    Features:
    - Nested key access using dot notation (e.g., "user.name")
    - Simple text-based search across values
    - Optional case-insensitive search
    - Filtering and deduplication via BaseSearchLogic

    Example:
        >>> data = {
        ...     "user": {"name": "Alice", "age": 30},
        ...     "settings": {"theme": "dark"}
        ... }
        >>> adapter = DictResourceAdapter(data, name="config")
        >>> adapter.get_value("user.name")
        "Alice"
        >>> adapter.search("Alice")
        [{'content': "Alice", 'key': "user.name", 'score': 1.0}]
    """

    def __init__(
        self,
        data: Dict[str, Any],
        name: str = "dict_adapter",
        case_sensitive: bool = False
    ):
        """Initialize dictionary adapter.

        Args:
            data: Dictionary to wrap as a resource
            name: Name identifier for this adapter
            case_sensitive: Whether search should be case-sensitive (default: False)
        """
        super().__init__(name=name)
        self._data = data
        self._case_sensitive = case_sensitive

    def get_value(
        self,
        key: str,
        default: Any = None,
        context: Dict[str, Any] | None = None
    ) -> Any:
        """Retrieve a value by key from the dictionary.

        Supports nested key access using dot notation. Dot-separated keys
        traverse nested dictionaries (e.g., a.b.c accesses nested values).

        Args:
            key: Key to look up (supports dot notation for nested access)
            default: Value to return if key is not found
            context: Optional context (unused in dict adapter)

        Returns:
            Value at the key, or default if not found
        """
        # Handle dot notation for nested keys
        if '.' in key:
            parts = key.split('.')
            value = self._data
            for part in parts:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    return default
            return value
        else:
            return self._data.get(key, default)

    def search(
        self,
        query: str,
        k: int = 5,
        filters: Dict[str, Any] | None = None,
        **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """Perform text-based search across dictionary values.

        Searches through all values in the dictionary (including nested values)
        and returns items where the query string appears in the value.

        Args:
            query: Search query string
            k: Maximum number of results to return
            filters: Optional filters to apply (passed to BaseSearchLogic)
            **kwargs: Additional search options:
                - min_score: Minimum score threshold (default: 0.0)
                - deduplicate: Whether to deduplicate results (default: False)

        Returns:
            List of search results with structure:
            {
                'content': <value>,
                'key': <key path>,
                'score': <relevance score>,
                'metadata': {<additional metadata>}
            }
        """
        results = []

        # Normalize query for case-insensitive search
        search_query = query if self._case_sensitive else query.lower()

        # Flatten dictionary and search
        for key, value in self._flatten_dict(self._data).items():
            value_str = str(value)
            search_value = value_str if self._case_sensitive else value_str.lower()

            if search_query in search_value:
                # Simple scoring: exact match = 1.0, contains = 0.8
                score = 1.0 if search_query == search_value else 0.8

                result = BaseSearchLogic.format_search_result(
                    value_str,
                    score=score,
                    metadata={"key": key}
                )
                result["key"] = key  # Add key to top level for easier access
                results.append(result)

                if len(results) >= k:
                    break

        # Apply filters if provided
        if filters:
            results = BaseSearchLogic.filter_results(results, filters=filters)

        # Apply min_score filter if provided
        min_score = kwargs.get('min_score', 0.0)
        if min_score > 0:
            results = BaseSearchLogic.filter_results(results, min_score=min_score)

        # Deduplicate if requested
        if kwargs.get('deduplicate', False):
            results = BaseSearchLogic.deduplicate_results(results, key='content')

        return results[:k]

    def _flatten_dict(
        self,
        data: Dict[str, Any],
        parent_key: str = '',
        separator: str = '.'
    ) -> Dict[str, Any]:
        """Flatten nested dictionary with dot notation keys.

        Args:
            data: Dictionary to flatten
            parent_key: Parent key prefix
            separator: Separator for nested keys

        Returns:
            Flattened dictionary with dot-notation keys
        """
        items = []
        for key, value in data.items():
            new_key = f"{parent_key}{separator}{key}" if parent_key else key

            if isinstance(value, dict):
                items.extend(self._flatten_dict(value, new_key, separator).items())
            else:
                items.append((new_key, value))

        return dict(items)


class AsyncDictResourceAdapter(AsyncResourceAdapter):
    """Asynchronous adapter for Python dictionary resources.

    Provides the same functionality as DictResourceAdapter but with async methods.
    Useful for consistency in async codebases or when mixing with other async adapters.

    Example:
        >>> data = {"user": {"name": "Alice", "age": 30}}
        >>> adapter = AsyncDictResourceAdapter(data)
        >>> await adapter.get_value("user.name")
        "Alice"
    """

    def __init__(
        self,
        data: Dict[str, Any],
        name: str = "async_dict_adapter",
        case_sensitive: bool = False
    ):
        """Initialize async dictionary adapter.

        Args:
            data: Dictionary to wrap as a resource
            name: Name identifier for this adapter
            case_sensitive: Whether search should be case-sensitive (default: False)
        """
        super().__init__(name=name)
        self._data = data
        self._case_sensitive = case_sensitive

    async def get_value(
        self,
        key: str,
        default: Any = None,
        context: Dict[str, Any] | None = None
    ) -> Any:
        """Retrieve a value by key from the dictionary (async).

        See DictResourceAdapter.get_value for details.
        """
        # Handle dot notation for nested keys
        if '.' in key:
            parts = key.split('.')
            value = self._data
            for part in parts:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    return default
            return value
        else:
            return self._data.get(key, default)

    async def search(
        self,
        query: str,
        k: int = 5,
        filters: Dict[str, Any] | None = None,
        **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """Perform text-based search across dictionary values (async).

        See DictResourceAdapter.search for details.
        """
        results = []

        # Normalize query for case-insensitive search
        search_query = query if self._case_sensitive else query.lower()

        # Flatten dictionary and search
        for key, value in self._flatten_dict(self._data).items():
            value_str = str(value)
            search_value = value_str if self._case_sensitive else value_str.lower()

            if search_query in search_value:
                # Simple scoring: exact match = 1.0, contains = 0.8
                score = 1.0 if search_query == search_value else 0.8

                result = BaseSearchLogic.format_search_result(
                    value_str,
                    score=score,
                    metadata={"key": key}
                )
                result["key"] = key
                results.append(result)

                if len(results) >= k:
                    break

        # Apply filters if provided
        if filters:
            results = BaseSearchLogic.filter_results(results, filters=filters)

        # Apply min_score filter if provided
        min_score = kwargs.get('min_score', 0.0)
        if min_score > 0:
            results = BaseSearchLogic.filter_results(results, min_score=min_score)

        # Deduplicate if requested
        if kwargs.get('deduplicate', False):
            results = BaseSearchLogic.deduplicate_results(results, key='content')

        return results[:k]

    def _flatten_dict(
        self,
        data: Dict[str, Any],
        parent_key: str = '',
        separator: str = '.'
    ) -> Dict[str, Any]:
        """Flatten nested dictionary with dot notation keys.

        Args:
            data: Dictionary to flatten
            parent_key: Parent key prefix
            separator: Separator for nested keys

        Returns:
            Flattened dictionary with dot-notation keys
        """
        items = []
        for key, value in data.items():
            new_key = f"{parent_key}{separator}{key}" if parent_key else key

            if isinstance(value, dict):
                items.extend(self._flatten_dict(value, new_key, separator).items())
            else:
                items.append((new_key, value))

        return dict(items)
