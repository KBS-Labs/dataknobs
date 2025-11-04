"""Dataknobs backend resource adapters.

This module provides adapters that wrap dataknobs database backends, enabling them
to be used as resource providers in the prompt library system. Supports both sync
and async database backends.
"""

from typing import Any, Dict, List, TYPE_CHECKING

from .resource_adapter import ResourceAdapter, AsyncResourceAdapter, BaseSearchLogic

if TYPE_CHECKING:
    from dataknobs_data.database import SyncDatabase, AsyncDatabase


class DataknobsBackendAdapter(ResourceAdapter):
    """Synchronous adapter for dataknobs database backends.

    Wraps a dataknobs SyncDatabase instance to provide resource adapter functionality.

    Features:
    - Record retrieval by ID using database.read()
    - Search using database.search() with Query objects
    - Field extraction with dot-notation support
    - Score-based ranking from search results

    Example:
        >>> from dataknobs_data.backends import SyncMemoryDatabase
        >>> db = SyncMemoryDatabase()
        >>> # ... populate database ...
        >>> adapter = DataknobsBackendAdapter(db, name="memory")
        >>> record = adapter.get_value("record_id_123")
        >>> results = adapter.search("query text")
    """

    def __init__(
        self,
        database: "SyncDatabase",
        name: str = "dataknobs_backend",
        text_field: str = "content",
        metadata_field: str | None = None
    ):
        """Initialize dataknobs backend adapter.

        Args:
            database: SyncDatabase instance to wrap
            name: Name identifier for this adapter
            text_field: Field name to use as primary content (default: "content")
            metadata_field: Optional field to extract as metadata
        """
        super().__init__(name=name)
        self._database = database
        self._text_field = text_field
        self._metadata_field = metadata_field

    def get_value(
        self,
        key: str,
        default: Any = None,
        context: Dict[str, Any] | None = None
    ) -> Any:
        """Retrieve a record or field value by ID.

        Supports field extraction using dot notation:
        - Simple key: Returns entire record as dict
        - "record_id.field_name": Returns specific field value
        - "record_id.field.nested": Returns nested field value

        Args:
            key: Record ID or "record_id.field" notation
            default: Value to return if record/field not found
            context: Optional context with additional parameters

        Returns:
            Record dict, field value, or default if not found
        """
        try:
            # Parse key for potential field extraction
            if '.' in key:
                parts = key.split('.', 1)
                record_id = parts[0]
                field_path = parts[1]
            else:
                record_id = key
                field_path = None

            # Read record from database
            record = self._database.read(record_id)

            if record is None:
                return default

            # Extract field if specified
            if field_path:
                return record.get_value(field_path, default=default)
            else:
                # Return full record as dict
                return record.to_dict(include_metadata=True)

        except Exception:
            # Log error if needed, return default
            return default

    def search(
        self,
        query: str,
        k: int = 5,
        filters: Dict[str, Any] | None = None,
        **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """Perform search using database backend.

        Creates a Query object with LIKE filter for text search.
        Results are formatted according to BaseSearchLogic standards.

        Args:
            query: Search query string (searches text_field using LIKE)
            k: Maximum number of results to return
            filters: Optional additional filters for the search
            **kwargs: Additional search options:
                - min_score: Minimum relevance score (default: 0.0)
                - deduplicate: Whether to deduplicate results (default: False)

        Returns:
            List of search results with structure:
            {
                "content": <text content>,
                "score": <relevance score>,
                "metadata": {<record metadata>}
            }
        """
        try:
            from dataknobs_data.query import Query, Filter, Operator

            # Build filter for text search using LIKE operator
            # This searches for the query string anywhere in the text field
            search_filter = Filter(
                field=self._text_field,
                operator=Operator.LIKE,
                value=f"%{query}%"
            )

            # Build query object with filter and limit
            query_obj = Query(
                filters=[search_filter],
                limit_value=k
            )

            # Execute search
            records = self._database.search(query_obj)

            # Format results
            results = []
            for record in records:
                # Extract content field
                content = record.get_value(self._text_field, default="")

                # Get score from metadata if available
                score = record.metadata.get("score", record.metadata.get("_score", 1.0))

                # Extract metadata
                metadata = {}
                if self._metadata_field:
                    metadata_value = record.get_value(self._metadata_field)
                    if metadata_value is not None:
                        metadata["metadata_field"] = metadata_value

                # Add record ID and other metadata
                if hasattr(record, 'storage_id') and record.storage_id:
                    metadata["record_id"] = record.storage_id

                # Merge with record metadata
                metadata.update(record.metadata)

                # Format result
                result = BaseSearchLogic.format_search_result(
                    content,
                    score=score,
                    metadata=metadata
                )
                results.append(result)

            # Apply filters if provided
            if filters:
                results = BaseSearchLogic.filter_results(results, filters=filters)

            # Apply min_score filter
            min_score = kwargs.get('min_score', 0.0)
            if min_score > 0:
                results = BaseSearchLogic.filter_results(results, min_score=min_score)

            # Deduplicate if requested
            if kwargs.get('deduplicate', False):
                results = BaseSearchLogic.deduplicate_results(results, key='content')

            return results[:k]

        except Exception:
            # Log error if needed
            return []


class AsyncDataknobsBackendAdapter(AsyncResourceAdapter):
    """Asynchronous adapter for dataknobs database backends.

    Wraps a dataknobs AsyncDatabase instance to provide async resource adapter functionality.

    Example:
        >>> from dataknobs_data.backends import AsyncMemoryDatabase
        >>> db = AsyncMemoryDatabase()
        >>> adapter = AsyncDataknobsBackendAdapter(db)
        >>> record = await adapter.get_value("record_id_123")
        >>> results = await adapter.search("query text")
    """

    def __init__(
        self,
        database: "AsyncDatabase",
        name: str = "async_dataknobs_backend",
        text_field: str = "content",
        metadata_field: str | None = None
    ):
        """Initialize async dataknobs backend adapter.

        Args:
            database: AsyncDatabase instance to wrap
            name: Name identifier for this adapter
            text_field: Field name to use as primary content (default: "content")
            metadata_field: Optional field to extract as metadata
        """
        super().__init__(name=name)
        self._database = database
        self._text_field = text_field
        self._metadata_field = metadata_field

    async def get_value(
        self,
        key: str,
        default: Any = None,
        context: Dict[str, Any] | None = None
    ) -> Any:
        """Retrieve a record or field value by ID (async).

        See DataknobsBackendAdapter.get_value for details.
        """
        try:
            # Parse key for potential field extraction
            if '.' in key:
                parts = key.split('.', 1)
                record_id = parts[0]
                field_path = parts[1]
            else:
                record_id = key
                field_path = None

            # Read record from database
            record = await self._database.read(record_id)

            if record is None:
                return default

            # Extract field if specified
            if field_path:
                return record.get_value(field_path, default=default)
            else:
                # Return full record as dict
                return record.to_dict(include_metadata=True)

        except Exception:
            # Log error if needed, return default
            return default

    async def search(
        self,
        query: str,
        k: int = 5,
        filters: Dict[str, Any] | None = None,
        **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """Perform search using database backend (async).

        See DataknobsBackendAdapter.search for details.
        """
        try:
            from dataknobs_data.query import Query, Filter, Operator

            # Build filter for text search using LIKE operator
            search_filter = Filter(
                field=self._text_field,
                operator=Operator.LIKE,
                value=f"%{query}%"
            )

            # Build query object with filter and limit
            query_obj = Query(
                filters=[search_filter],
                limit_value=k
            )

            # Execute search
            records = await self._database.search(query_obj)

            # Format results
            results = []
            for record in records:
                # Extract content field
                content = record.get_value(self._text_field, default="")

                # Get score from metadata if available
                score = record.metadata.get("score", record.metadata.get("_score", 1.0))

                # Extract metadata
                metadata = {}
                if self._metadata_field:
                    metadata_value = record.get_value(self._metadata_field)
                    if metadata_value is not None:
                        metadata["metadata_field"] = metadata_value

                # Add record ID and other metadata
                if hasattr(record, 'storage_id') and record.storage_id:
                    metadata["record_id"] = record.storage_id

                # Merge with record metadata
                metadata.update(record.metadata)

                # Format result
                result = BaseSearchLogic.format_search_result(
                    content,
                    score=score,
                    metadata=metadata
                )
                results.append(result)

            # Apply filters if provided
            if filters:
                results = BaseSearchLogic.filter_results(results, filters=filters)

            # Apply min_score filter
            min_score = kwargs.get('min_score', 0.0)
            if min_score > 0:
                results = BaseSearchLogic.filter_results(results, min_score=min_score)

            # Deduplicate if requested
            if kwargs.get('deduplicate', False):
                results = BaseSearchLogic.deduplicate_results(results, key='content')

            return results[:k]

        except Exception:
            # Log error if needed
            return []
