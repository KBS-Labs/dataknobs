"""Database abstraction layer providing unified interfaces for data operations.

This module defines abstract base classes for synchronous and asynchronous database
operations, supporting CRUD, querying, streaming, and schema management across
different backend database implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

from .database_utils import ensure_record_id, process_search_results
from .query import Query
from .schema import DatabaseSchema, FieldSchema

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Iterator
    from .query_logic import ComplexQuery
    from .records import Record
    from .streaming import StreamConfig, StreamResult


class AsyncDatabase(ABC):
    """Abstract base class for async database implementations.

    Provides a unified async interface for CRUD operations, querying, and streaming
    across different backend databases. Supports schema validation, batch operations,
    and complex queries with boolean logic.

    Example:
        ```python
        from dataknobs_data import async_database_factory, Record, Query, Filter, Operator

        # Create async database
        db = async_database_factory("memory")

        # Use as async context manager
        async with db:
            # Create records
            id1 = await db.create(Record({"name": "Alice", "age": 30}))
            id2 = await db.create(Record({"name": "Bob", "age": 25}))

            # Query records
            query = Query(filters=[Filter("age", Operator.GT, 25)])
            results = await db.search(query)
            print(results)  # [Alice's record]

            # Update record
            await db.update(id1, Record({"name": "Alice", "age": 31}))

            # Stream large datasets
            async for record in db.stream_read():
                process_record(record)
        ```
    """

    def __init__(self, config: dict[str, Any] | None = None, schema: DatabaseSchema | None = None):
        """Initialize the database with optional configuration.

        Args:
            config: Backend-specific configuration parameters (may include 'schema' key)
            schema: Optional database schema (overrides config schema)

        Example:
            ```python
            from dataknobs_data import AsyncDatabase
            from dataknobs_data.schema import DatabaseSchema
            from dataknobs_data.fields import FieldType

            # With schema
            schema = DatabaseSchema.create(
                name=FieldType.STRING,
                age=FieldType.INTEGER
            )
            db = AsyncDatabase(config={"path": "data.db"}, schema=schema)
            ```
        """
        config = config or {}

        # Extract schema from config if present and no explicit schema provided
        if schema is None and "schema" in config:
            schema = self._extract_schema_from_config(config["schema"])
            # Remove schema from config so backends don't see it
            config = {k: v for k, v in config.items() if k != "schema"}

        self.config = config
        self.schema = schema or DatabaseSchema()
        self._initialize()

    @staticmethod
    def _extract_schema_from_config(schema_config: Any) -> DatabaseSchema | None:
        """Extract schema from configuration.
        
        Args:
            schema_config: Can be a DatabaseSchema, dict, or None
            
        Returns:
            DatabaseSchema instance or None
        """
        if isinstance(schema_config, DatabaseSchema):
            return schema_config
        elif isinstance(schema_config, dict):
            return DatabaseSchema.from_dict(schema_config)
        return None

    def _initialize(self) -> None:  # noqa: B027
        """Initialize the database backend. Override in subclasses if needed."""
        # Default implementation does nothing - backends can override if needed

    def _ensure_record_id(self, record: Record, record_id: str) -> Record:
        """Ensure a record has its ID set (delegates to utility function)."""
        return ensure_record_id(record, record_id)

    def _prepare_record_for_storage(self, record: Record) -> tuple[Record, str]:
        """Prepare a record for storage by ensuring it has a storage_id.
        
        Args:
            record: The record to prepare
            
        Returns:
            Tuple of (prepared_record_copy, storage_id)
        """
        import uuid
        # Make a copy to avoid modifying the original
        record_copy = record.copy(deep=True)

        # Generate storage ID if not present
        if not record_copy.has_storage_id():
            storage_id = str(uuid.uuid4())
            record_copy.storage_id = storage_id
        else:
            storage_id = record_copy.storage_id

        return record_copy, storage_id

    def _prepare_record_from_storage(self, record: Record | None, storage_id: str) -> Record | None:
        """Prepare a record retrieved from storage by ensuring storage_id is set.
        
        Args:
            record: The record retrieved from storage (or None)
            storage_id: The storage ID used to retrieve the record
            
        Returns:
            Record with storage_id set, or None if record was None
        """
        if record:
            record_copy = record.copy(deep=True)
            # Ensure storage_id is set
            if not record_copy.has_storage_id():
                record_copy.storage_id = storage_id
            return record_copy
        return None

    def _process_search_results(
        self,
        results: list[tuple[str, Record]],
        query: Query,
        deep_copy: bool = True
    ) -> list[Record]:
        """Process search results (delegates to utility function)."""
        return process_search_results(results, query, deep_copy)

    def set_schema(self, schema: DatabaseSchema) -> None:
        """Set the database schema.
        
        Args:
            schema: The database schema to use
        """
        self.schema = schema

    def add_field_schema(self, field_schema: FieldSchema) -> None:
        """Add a field to the database schema.
        
        Args:
            field_schema: The field schema to add
        """
        self.schema.add_field(field_schema)

    def with_schema(self, **field_definitions) -> AsyncDatabase:
        """Set schema using field definitions.
        
        Returns self for chaining.
        
        Examples:
            db = AsyncMemoryDatabase().with_schema(
                content=FieldType.TEXT,
                embedding=(FieldType.VECTOR, {"dimensions": 384, "source_field": "content"})
            )
        """
        self.schema = DatabaseSchema.create(**field_definitions)
        return self

    @abstractmethod
    async def create(self, record: Record) -> str:
        """Create a new record in the database.

        Args:
            record: The record to create

        Returns:
            The ID of the created record
        """
        raise NotImplementedError

    @abstractmethod
    async def read(self, id: str) -> Record | None:
        """Read a record by ID.

        Args:
            id: The record ID

        Returns:
            The record if found, None otherwise
        """
        raise NotImplementedError

    @abstractmethod
    async def update(self, id: str, record: Record) -> bool:
        """Update an existing record.

        Args:
            id: The record ID
            record: The updated record

        Returns:
            True if the record was updated, False if not found
        """
        raise NotImplementedError

    @abstractmethod
    async def delete(self, id: str) -> bool:
        """Delete a record by ID.

        Args:
            id: The record ID

        Returns:
            True if the record was deleted, False if not found
        """
        raise NotImplementedError

    @abstractmethod
    async def search(self, query: Query | ComplexQuery) -> list[Record]:
        """Search for records matching a query.

        Args:
            query: The search query (simple or complex)

        Returns:
            List of matching records
        """
        raise NotImplementedError

    async def all(self) -> list[Record]:
        """Get all records from the database.
        
        Returns:
            List of all records
        """
        # Default implementation using search with empty query
        from .query import Query
        return await self.search(Query())

    async def _search_with_complex_query(self, query: ComplexQuery) -> list[Record]:
        """Default implementation for ComplexQuery using in-memory filtering.
        
        Backends can override this for native boolean logic support.
        
        Args:
            query: Complex query with boolean logic
            
        Returns:
            List of matching records
        """
        # Try to convert to simple query if possible
        try:
            simple_query = query.to_simple_query()
            return await self.search(simple_query)
        except ValueError:
            # Can't convert - need to do in-memory filtering
            # Get all records (or use a base filter if possible)
            all_records = await self.search(Query())

            # Apply complex condition filtering
            results = []
            for record in all_records:
                if query.matches(record):
                    results.append(record)

            # Apply sorting
            if query.sort_specs:
                for sort_spec in reversed(query.sort_specs):
                    reverse = sort_spec.order.value == "desc"
                    results.sort(
                        key=lambda r: r.get_value(sort_spec.field, ""),
                        reverse=reverse
                    )

            # Apply offset and limit
            if query.offset_value:
                results = results[query.offset_value:]
            if query.limit_value:
                results = results[:query.limit_value]

            # Apply field projection
            if query.fields:
                results = [r.project(query.fields) for r in results]

            return results

    @abstractmethod
    async def exists(self, id: str) -> bool:
        """Check if a record exists.

        Args:
            id: The record ID

        Returns:
            True if the record exists, False otherwise
        """
        raise NotImplementedError

    async def upsert(self, id_or_record: str | Record, record: Record | None = None) -> str:
        """Update or insert a record.
        
        Can be called as:
        - upsert(id, record) - explicit ID and record
        - upsert(record) - extract ID from record using Record's built-in logic
        
        Args:
            id_or_record: Either an ID string or a Record
            record: The record to upsert (if first arg is ID)
            
        Returns:
            The record ID
        """
        import uuid
        
        # Determine ID and record based on arguments
        if isinstance(id_or_record, str):
            # Called with explicit ID: upsert(id, record)
            id = id_or_record
            if record is None:
                raise ValueError("Record required when ID is provided")
        else:
            # Called with just record: upsert(record)
            record = id_or_record
            # Use Record's built-in ID property which handles all the logic
            id = record.id
            
            if id is None:
                # Generate a new ID if none found
                id = str(uuid.uuid4())  # type: ignore[unreachable]
                # Set it on the record for future reference
                record.storage_id = id
        
        # Now perform the upsert
        if await self.exists(id):
            await self.update(id, record)
        else:
            # Ensure the record has the storage_id set for create
            if not record.storage_id:
                record.storage_id = id
            created_id = await self.create(record)
            # Return the created ID (might be different from what we provided)
            return created_id or id
        return id

    async def create_batch(self, records: list[Record]) -> list[str]:
        """Create multiple records in batch.

        Args:
            records: List of records to create

        Returns:
            List of created record IDs
        """
        ids = []
        for record in records:
            id = await self.create(record)
            ids.append(id)
        return ids

    async def read_batch(self, ids: list[str]) -> list[Record | None]:
        """Read multiple records by ID.

        Args:
            ids: List of record IDs

        Returns:
            List of records (None for not found)
        """
        records = []
        for id in ids:
            record = await self.read(id)
            records.append(record)
        return records

    async def delete_batch(self, ids: list[str]) -> list[bool]:
        """Delete multiple records by ID.

        Args:
            ids: List of record IDs

        Returns:
            List of deletion results
        """
        results = []
        for id in ids:
            result = await self.delete(id)
            results.append(result)
        return results

    async def update_batch(self, updates: list[tuple[str, Record]]) -> list[bool]:
        """Update multiple records.

        Default implementation calls update() for each ID/record pair.
        Override for better performance.

        Args:
            updates: List of (id, record) tuples to update

        Returns:
            List of success flags for each update
        """
        results = []
        for id, record in updates:
            result = await self.update(id, record)
            results.append(result)
        return results

    async def count(self, query: Query | None = None) -> int:
        """Count records matching a query.

        Args:
            query: Optional search query (counts all if None)

        Returns:
            Number of matching records
        """
        if query:
            results = await self.search(query)
            return len(results)
        else:
            return await self._count_all()

    @abstractmethod
    async def _count_all(self) -> int:
        """Count all records in the database."""
        raise NotImplementedError

    async def clear(self) -> int:
        """Clear all records from the database.

        Returns:
            Number of records deleted
        """
        raise NotImplementedError

    async def connect(self) -> None:  # noqa: B027
        """Connect to the database. Override in subclasses if needed."""
        # Default implementation does nothing - many backends don't need explicit connection

    async def close(self) -> None:  # noqa: B027
        """Close the database connection. Override in subclasses if needed."""
        # Default implementation does nothing - many backends don't need explicit closing

    async def disconnect(self) -> None:
        """Disconnect from the database (alias for close)."""
        await self.close()

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    @abstractmethod
    async def stream_read(
        self,
        query: Query | None = None,
        config: StreamConfig | None = None
    ) -> AsyncIterator[Record]:
        """Stream records from database.
        
        Yields records one at a time, fetching in batches internally.
        
        Args:
            query: Optional query to filter records
            config: Streaming configuration
            
        Yields:
            Records matching the query
        """
        raise NotImplementedError

    @abstractmethod
    async def stream_write(
        self,
        records: AsyncIterator[Record],
        config: StreamConfig | None = None
    ) -> StreamResult:
        """Stream records into database.
        
        Accepts an iterator and writes in batches.
        
        Args:
            records: Iterator of records to write
            config: Streaming configuration
            
        Returns:
            Result of the streaming operation
        """
        raise NotImplementedError

    async def stream_transform(
        self,
        query: Query | None = None,
        transform: Callable[[Record], Record | None] | None = None,
        config: StreamConfig | None = None
    ) -> AsyncIterator[Record]:
        """Stream records through a transformation.
        
        Default implementation, can be overridden for efficiency.
        
        Args:
            query: Optional query to filter records
            transform: Optional transformation function
            config: Streaming configuration
            
        Yields:
            Transformed records
        """
        async for record in self.stream_read(query, config):
            if transform:
                transformed = transform(record)
                if transformed:  # None means filter out
                    yield transformed
            else:
                yield record

    @classmethod
    async def from_backend(cls, backend: str, config: dict[str, Any] | None = None) -> AsyncDatabase:
        """Factory method to create and connect a database instance.

        Args:
            backend: The backend type ("memory", "file", "s3", "postgres", "elasticsearch")
            config: Backend-specific configuration

        Returns:
            Connected AsyncDatabase instance
        """
        from .backends import BACKEND_REGISTRY

        backend_class = BACKEND_REGISTRY.get(backend)
        if not backend_class:
            raise ValueError(
                f"Unknown backend: {backend}. Available: {list(BACKEND_REGISTRY.keys())}"
            )

        instance = backend_class(config)
        await instance.connect()
        return instance


class SyncDatabase(ABC):
    """Synchronous variant of the Database abstract base class.

    Provides a unified synchronous interface for CRUD operations, querying, and streaming
    across different backend databases. Supports schema validation, batch operations,
    and complex queries with boolean logic.

    Example:
        ```python
        from dataknobs_data import database_factory, Record, Query, Filter, Operator

        # Create database
        db = database_factory("memory")

        # Use as context manager
        with db:
            # Create records
            id1 = db.create(Record({"name": "Alice", "age": 30}))
            id2 = db.create(Record({"name": "Bob", "age": 25}))

            # Query records
            query = Query(filters=[Filter("age", Operator.GT, 25)])
            results = db.search(query)
            print(results)  # [Alice's record]

            # Update record
            db.update(id1, Record({"name": "Alice", "age": 31}))

            # Stream large datasets
            for record in db.stream_read():
                process_record(record)
        ```
    """

    def __init__(self, config: dict[str, Any] | None = None, schema: DatabaseSchema | None = None):
        """Initialize the database with optional configuration.

        Args:
            config: Backend-specific configuration parameters (may include 'schema' key)
            schema: Optional database schema (overrides config schema)

        Example:
            ```python
            from dataknobs_data import SyncDatabase
            from dataknobs_data.schema import DatabaseSchema
            from dataknobs_data.fields import FieldType

            # With schema
            schema = DatabaseSchema.create(
                name=FieldType.STRING,
                age=FieldType.INTEGER
            )
            db = SyncDatabase(config={"path": "data.db"}, schema=schema)
            ```
        """
        config = config or {}

        # Extract schema from config if present and no explicit schema provided
        if schema is None and "schema" in config:
            schema = AsyncDatabase._extract_schema_from_config(config["schema"])
            # Remove schema from config so backends don't see it
            config = {k: v for k, v in config.items() if k != "schema"}

        self.config = config
        self.schema = schema or DatabaseSchema()
        self._initialize()

    def _initialize(self) -> None:  # noqa: B027
        """Initialize the database backend. Override in subclasses if needed."""
        # Default implementation does nothing - backends can override if needed

    def _ensure_record_id(self, record: Record, record_id: str) -> Record:
        """Ensure a record has its ID set (delegates to utility function)."""
        return ensure_record_id(record, record_id)

    def _prepare_record_for_storage(self, record: Record) -> tuple[Record, str]:
        """Prepare a record for storage by ensuring it has a storage_id.
        
        Args:
            record: The record to prepare
            
        Returns:
            Tuple of (prepared_record_copy, storage_id)
        """
        import uuid
        # Make a copy to avoid modifying the original
        record_copy = record.copy(deep=True)

        # Generate storage ID if not present
        if not record_copy.has_storage_id():
            storage_id = str(uuid.uuid4())
            record_copy.storage_id = storage_id
        else:
            storage_id = record_copy.storage_id

        return record_copy, storage_id

    def _prepare_record_from_storage(self, record: Record | None, storage_id: str) -> Record | None:
        """Prepare a record retrieved from storage by ensuring storage_id is set.
        
        Args:
            record: The record retrieved from storage (or None)
            storage_id: The storage ID used to retrieve the record
            
        Returns:
            Record with storage_id set, or None if record was None
        """
        if record:
            record_copy = record.copy(deep=True)
            # Ensure storage_id is set
            if not record_copy.has_storage_id():
                record_copy.storage_id = storage_id
            return record_copy
        return None

    def _process_search_results(
        self,
        results: list[tuple[str, Record]],
        query: Query,
        deep_copy: bool = True
    ) -> list[Record]:
        """Process search results (delegates to utility function)."""
        return process_search_results(results, query, deep_copy)

    def set_schema(self, schema: DatabaseSchema) -> None:
        """Set the database schema.
        
        Args:
            schema: The database schema to use
        """
        self.schema = schema

    def add_field_schema(self, field_schema: FieldSchema) -> None:
        """Add a field to the database schema.
        
        Args:
            field_schema: The field schema to add
        """
        self.schema.add_field(field_schema)

    def with_schema(self, **field_definitions) -> SyncDatabase:
        """Set schema using field definitions.
        
        Returns self for chaining.
        
        Examples:
            db = SyncMemoryDatabase().with_schema(
                content=FieldType.TEXT,
                embedding=(FieldType.VECTOR, {"dimensions": 384, "source_field": "content"})
            )
        """
        self.schema = DatabaseSchema.create(**field_definitions)
        return self

    @abstractmethod
    def create(self, record: Record) -> str:
        """Create a new record in the database.

        Args:
            record: The record to create

        Returns:
            The ID of the created record

        Example:
            ```python
            record = Record({"name": "Alice", "age": 30})
            record_id = db.create(record)
            print(record_id)  # "550e8400-e29b-41d4-a716-446655440000"
            ```
        """
        raise NotImplementedError

    @abstractmethod
    def read(self, id: str) -> Record | None:
        """Read a record by ID.

        Args:
            id: The record ID

        Returns:
            The record if found, None otherwise

        Example:
            ```python
            record = db.read("550e8400-e29b-41d4-a716-446655440000")
            if record:
                print(record.get_value("name"))  # "Alice"
            ```
        """
        raise NotImplementedError

    @abstractmethod
    def update(self, id: str, record: Record) -> bool:
        """Update an existing record.

        Args:
            id: The record ID
            record: The updated record

        Returns:
            True if the record was updated, False if not found

        Example:
            ```python
            updated_record = Record({"name": "Alice", "age": 31})
            success = db.update(record_id, updated_record)
            print(success)  # True
            ```
        """
        raise NotImplementedError

    @abstractmethod
    def delete(self, id: str) -> bool:
        """Delete a record by ID.

        Args:
            id: The record ID

        Returns:
            True if the record was deleted, False if not found

        Example:
            ```python
            success = db.delete(record_id)
            print(success)  # True
            ```
        """
        raise NotImplementedError

    @abstractmethod
    def search(self, query: Query | ComplexQuery) -> list[Record]:
        """Search for records matching a query (simple or complex).

        Args:
            query: The search query

        Returns:
            List of matching records

        Example:
            ```python
            # Simple query
            query = Query(filters=[Filter("age", Operator.GT, 25)])
            results = db.search(query)

            # Complex query with boolean logic
            from dataknobs_data.query_logic import QueryBuilder, LogicOperator

            complex_query = (
                QueryBuilder()
                .where("age", Operator.GT, 25)
                .and_where("name", Operator.LIKE, "A%")
                .build()
            )
            results = db.search(complex_query)
            ```
        """
        raise NotImplementedError

    def all(self) -> list[Record]:
        """Get all records from the database.
        
        Returns:
            List of all records
        """
        # Default implementation using search with empty query
        from .query import Query
        return self.search(Query())

    def _search_with_complex_query(self, query: ComplexQuery) -> list[Record]:
        """Default implementation for ComplexQuery using in-memory filtering.
        
        Backends can override this for native boolean logic support.
        
        Args:
            query: Complex query with boolean logic
            
        Returns:
            List of matching records
        """
        # Try to convert to simple query if possible
        try:
            simple_query = query.to_simple_query()
            return self.search(simple_query)
        except ValueError:
            # Can't convert - need to do in-memory filtering
            # Get all records (or use a base filter if possible)
            all_records = self.search(Query())

            # Apply complex condition filtering
            results = []
            for record in all_records:
                if query.matches(record):
                    results.append(record)

            # Apply sorting
            if query.sort_specs:
                for sort_spec in reversed(query.sort_specs):
                    reverse = sort_spec.order.value == "desc"
                    results.sort(
                        key=lambda r: r.get_value(sort_spec.field, ""),
                        reverse=reverse
                    )

            # Apply offset and limit
            if query.offset_value:
                results = results[query.offset_value:]
            if query.limit_value:
                results = results[:query.limit_value]

            # Apply field projection
            if query.fields:
                results = [r.project(query.fields) for r in results]

            return results

    @abstractmethod
    def exists(self, id: str) -> bool:
        """Check if a record exists."""
        raise NotImplementedError

    def upsert(self, id_or_record: str | Record, record: Record | None = None) -> str:
        """Update or insert a record.
        
        Can be called as:
        - upsert(id, record) - explicit ID and record
        - upsert(record) - extract ID from record using Record's built-in logic
        
        Args:
            id_or_record: Either an ID string or a Record
            record: The record to upsert (if first arg is ID)
            
        Returns:
            The record ID
        """
        import uuid
        
        # Determine ID and record based on arguments
        if isinstance(id_or_record, str):
            # Called with explicit ID: upsert(id, record)
            id = id_or_record
            if record is None:
                raise ValueError("Record required when ID is provided")
        else:
            # Called with just record: upsert(record)
            record = id_or_record
            # Use Record's built-in ID property which handles all the logic
            id = record.id
            
            if id is None:
                # Generate a new ID if none found
                id = str(uuid.uuid4())  # type: ignore[unreachable]
                # Set it on the record for future reference
                record.storage_id = id
        
        # Now perform the upsert
        if self.exists(id):
            self.update(id, record)
        else:
            # Ensure the record has the storage_id set for create
            if not record.storage_id:
                record.storage_id = id
            created_id = self.create(record)
            # Return the created ID (might be different from what we provided)
            return created_id or id
        return id

    def create_batch(self, records: list[Record]) -> list[str]:
        """Create multiple records in batch."""
        ids = []
        for record in records:
            id = self.create(record)
            ids.append(id)
        return ids

    def read_batch(self, ids: list[str]) -> list[Record | None]:
        """Read multiple records by ID."""
        records = []
        for id in ids:
            record = self.read(id)
            records.append(record)
        return records

    def delete_batch(self, ids: list[str]) -> list[bool]:
        """Delete multiple records by ID."""
        results = []
        for id in ids:
            result = self.delete(id)
            results.append(result)
        return results

    def update_batch(self, updates: list[tuple[str, Record]]) -> list[bool]:
        """Update multiple records.

        Default implementation calls update() for each ID/record pair.
        Override for better performance.

        Args:
            updates: List of (id, record) tuples to update

        Returns:
            List of success flags for each update
        """
        results = []
        for id, record in updates:
            result = self.update(id, record)
            results.append(result)
        return results

    def count(self, query: Query | None = None) -> int:
        """Count records matching a query."""
        if query:
            results = self.search(query)
            return len(results)
        else:
            return self._count_all()

    @abstractmethod
    def _count_all(self) -> int:
        """Count all records in the database."""
        raise NotImplementedError

    def clear(self) -> int:
        """Clear all records from the database."""
        raise NotImplementedError

    def connect(self) -> None:  # noqa: B027
        """Connect to the database. Override in subclasses if needed."""
        # Default implementation does nothing - many backends don't need explicit connection

    def close(self) -> None:  # noqa: B027
        """Close the database connection. Override in subclasses if needed."""
        # Default implementation does nothing - many backends don't need explicit closing

    def disconnect(self) -> None:
        """Disconnect from the database (alias for close)."""
        self.close()

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    @abstractmethod
    def stream_read(
        self,
        query: Query | None = None,
        config: StreamConfig | None = None
    ) -> Iterator[Record]:
        """Stream records from database.
        
        Yields records one at a time, fetching in batches internally.
        
        Args:
            query: Optional query to filter records
            config: Streaming configuration
            
        Yields:
            Records matching the query
        """
        raise NotImplementedError

    @abstractmethod
    def stream_write(
        self,
        records: Iterator[Record],
        config: StreamConfig | None = None
    ) -> StreamResult:
        """Stream records into database.
        
        Accepts an iterator and writes in batches.
        
        Args:
            records: Iterator of records to write
            config: Streaming configuration
            
        Returns:
            Result of the streaming operation
        """
        raise NotImplementedError

    def stream_transform(
        self,
        query: Query | None = None,
        transform: Callable[[Record], Record | None] | None = None,
        config: StreamConfig | None = None
    ) -> Iterator[Record]:
        """Stream records through a transformation.
        
        Default implementation, can be overridden for efficiency.
        
        Args:
            query: Optional query to filter records
            transform: Optional transformation function
            config: Streaming configuration
            
        Yields:
            Transformed records
        """
        for record in self.stream_read(query, config):
            if transform:
                transformed = transform(record)
                if transformed:  # None means filter out
                    yield transformed
            else:
                yield record

    @classmethod
    def from_backend(cls, backend: str, config: dict[str, Any] | None = None) -> SyncDatabase:
        """Factory method to create and connect a synchronous database instance.

        Args:
            backend: The backend type ("memory", "file", "s3", "postgres", "elasticsearch")
            config: Backend-specific configuration

        Returns:
            Connected SyncDatabase instance
        """
        from .backends import SYNC_BACKEND_REGISTRY

        backend_class = SYNC_BACKEND_REGISTRY.get(backend)
        if not backend_class:
            raise ValueError(
                f"Unknown backend: {backend}. Available: {list(SYNC_BACKEND_REGISTRY.keys())}"
            )

        instance = backend_class(config)
        instance.connect()
        return instance
