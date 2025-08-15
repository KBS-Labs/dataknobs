from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Callable, Dict, Iterator, List, Optional

from .query import Query
from .records import Record
from .streaming import StreamConfig, StreamResult


class Database(ABC):
    """Abstract base class for database implementations."""

    def __init__(self, config: Dict[str, Any] | None = None):
        """Initialize the database with optional configuration.

        Args:
            config: Backend-specific configuration parameters
        """
        self.config = config or {}
        self._initialize()

    def _initialize(self) -> None:
        """Initialize the database backend. Override in subclasses."""
        pass

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
    async def search(self, query: Query) -> List[Record]:
        """Search for records matching a query.

        Args:
            query: The search query

        Returns:
            List of matching records
        """
        raise NotImplementedError

    @abstractmethod
    async def exists(self, id: str) -> bool:
        """Check if a record exists.

        Args:
            id: The record ID

        Returns:
            True if the record exists, False otherwise
        """
        raise NotImplementedError

    async def upsert(self, id: str, record: Record) -> str:
        """Update or insert a record.

        Args:
            id: The record ID
            record: The record to upsert

        Returns:
            The record ID
        """
        if await self.exists(id):
            await self.update(id, record)
        else:
            return await self.create(record)
        return id

    async def create_batch(self, records: List[Record]) -> List[str]:
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

    async def read_batch(self, ids: List[str]) -> List[Record | None]:
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

    async def delete_batch(self, ids: List[str]) -> List[bool]:
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

    async def close(self) -> None:
        """Close the database connection."""
        pass

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    @abstractmethod
    async def stream_read(
        self,
        query: Optional[Query] = None,
        config: Optional[StreamConfig] = None
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
        config: Optional[StreamConfig] = None
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
        query: Optional[Query] = None,
        transform: Optional[Callable[[Record], Optional[Record]]] = None,
        config: Optional[StreamConfig] = None
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
    def create(cls, backend: str, config: Dict[str, Any] | None = None) -> "Database":
        """Factory method to create a database instance.

        Args:
            backend: The backend type ("memory", "file", "s3", "postgres", "elasticsearch")
            config: Backend-specific configuration

        Returns:
            Database instance
        """
        from .backends import BACKEND_REGISTRY

        backend_class = BACKEND_REGISTRY.get(backend)
        if not backend_class:
            raise ValueError(
                f"Unknown backend: {backend}. Available: {list(BACKEND_REGISTRY.keys())}"
            )

        return backend_class(config)


class SyncDatabase(ABC):
    """Synchronous variant of the Database abstract base class."""

    def __init__(self, config: Dict[str, Any] | None = None):
        """Initialize the database with optional configuration.

        Args:
            config: Backend-specific configuration parameters
        """
        self.config = config or {}
        self._initialize()

    def _initialize(self) -> None:
        """Initialize the database backend. Override in subclasses."""
        pass

    @abstractmethod
    def create(self, record: Record) -> str:
        """Create a new record in the database."""
        raise NotImplementedError

    @abstractmethod
    def read(self, id: str) -> Record | None:
        """Read a record by ID."""
        raise NotImplementedError

    @abstractmethod
    def update(self, id: str, record: Record) -> bool:
        """Update an existing record."""
        raise NotImplementedError

    @abstractmethod
    def delete(self, id: str) -> bool:
        """Delete a record by ID."""
        raise NotImplementedError

    @abstractmethod
    def search(self, query: Query) -> List[Record]:
        """Search for records matching a query."""
        raise NotImplementedError

    @abstractmethod
    def exists(self, id: str) -> bool:
        """Check if a record exists."""
        raise NotImplementedError

    def upsert(self, id: str, record: Record) -> str:
        """Update or insert a record."""
        if self.exists(id):
            self.update(id, record)
        else:
            return self.create(record)
        return id

    def create_batch(self, records: List[Record]) -> List[str]:
        """Create multiple records in batch."""
        ids = []
        for record in records:
            id = self.create(record)
            ids.append(id)
        return ids

    def read_batch(self, ids: List[str]) -> List[Record | None]:
        """Read multiple records by ID."""
        records = []
        for id in ids:
            record = self.read(id)
            records.append(record)
        return records

    def delete_batch(self, ids: List[str]) -> List[bool]:
        """Delete multiple records by ID."""
        results = []
        for id in ids:
            result = self.delete(id)
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

    def close(self) -> None:
        """Close the database connection."""
        pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    @abstractmethod
    def stream_read(
        self,
        query: Optional[Query] = None,
        config: Optional[StreamConfig] = None
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
        config: Optional[StreamConfig] = None
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
        query: Optional[Query] = None,
        transform: Optional[Callable[[Record], Optional[Record]]] = None,
        config: Optional[StreamConfig] = None
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
    def create(cls, backend: str, config: Dict[str, Any] | None = None) -> "SyncDatabase":
        """Factory method to create a synchronous database instance.

        Args:
            backend: The backend type ("memory", "file", "s3", "postgres", "elasticsearch")
            config: Backend-specific configuration

        Returns:
            SyncDatabase instance
        """
        from .backends import SYNC_BACKEND_REGISTRY

        backend_class = SYNC_BACKEND_REGISTRY.get(backend)
        if not backend_class:
            raise ValueError(
                f"Unknown backend: {backend}. Available: {list(SYNC_BACKEND_REGISTRY.keys())}"
            )

        return backend_class(config)
