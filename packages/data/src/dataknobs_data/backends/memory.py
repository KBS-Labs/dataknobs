"""In-memory database backend implementation."""

import asyncio
import threading
import uuid
from collections import OrderedDict
from collections.abc import AsyncIterator, Iterator
from typing import Any

from dataknobs_config import ConfigurableBase

from ..database import AsyncDatabase, SyncDatabase
from ..query import Query
from ..query_logic import ComplexQuery
from ..records import Record
from ..streaming import AsyncStreamingMixin, StreamConfig, StreamingMixin, StreamResult


class AsyncMemoryDatabase(AsyncDatabase, AsyncStreamingMixin, ConfigurableBase):
    """Async in-memory database implementation."""

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self._storage: OrderedDict[str, Record] = OrderedDict()
        self._lock = asyncio.Lock()

    @classmethod
    def from_config(cls, config: dict) -> "AsyncMemoryDatabase":
        """Create from config dictionary."""
        return cls(config)

    async def connect(self) -> None:
        """Connect to the database (no-op for memory backend)."""
        pass

    def _generate_id(self) -> str:
        """Generate a unique ID for a record."""
        return str(uuid.uuid4())

    async def create(self, record: Record) -> str:
        """Create a new record in memory."""
        async with self._lock:
            # Use record's ID if it has one, otherwise generate a new one
            id = record.id if record.id else self._generate_id()
            self._storage[id] = record.copy(deep=True)
            return id

    async def read(self, id: str) -> Record | None:
        """Read a record from memory."""
        async with self._lock:
            record = self._storage.get(id)
            return record.copy(deep=True) if record else None

    async def update(self, id: str, record: Record) -> bool:
        """Update a record in memory."""
        async with self._lock:
            if id in self._storage:
                self._storage[id] = record.copy(deep=True)
                return True
            return False

    async def delete(self, id: str) -> bool:
        """Delete a record from memory."""
        async with self._lock:
            if id in self._storage:
                del self._storage[id]
                return True
            return False

    async def exists(self, id: str) -> bool:
        """Check if a record exists in memory."""
        async with self._lock:
            return id in self._storage

    async def upsert(self, id: str, record: Record) -> str:
        """Update or insert a record with the specified ID."""
        async with self._lock:
            self._storage[id] = record.copy(deep=True)
            return id

    async def search(self, query: Query | ComplexQuery) -> list[Record]:
        """Search for records matching the query."""
        # Handle ComplexQuery using base class implementation
        if isinstance(query, ComplexQuery):
            return await self._search_with_complex_query(query)

        async with self._lock:
            results = []

            for id, record in self._storage.items():
                # Apply filters
                matches = True
                for filter in query.filters:
                    field_value = record.get_value(filter.field)
                    if not filter.matches(field_value):
                        matches = False
                        break

                if matches:
                    results.append((id, record))

            # Use the helper method from base class
            return self._process_search_results(results, query, deep_copy=True)

    async def _count_all(self) -> int:
        """Count all records in memory."""
        async with self._lock:
            return len(self._storage)

    async def clear(self) -> int:
        """Clear all records from memory."""
        async with self._lock:
            count = len(self._storage)
            self._storage.clear()
            return count

    async def create_batch(self, records: list[Record]) -> list[str]:
        """Create multiple records efficiently."""
        async with self._lock:
            ids = []
            for record in records:
                # Use record's ID if it has one, otherwise generate a new one
                id = record.id if record.id else self._generate_id()
                self._storage[id] = record.copy(deep=True)
                ids.append(id)
            return ids

    async def read_batch(self, ids: list[str]) -> list[Record | None]:
        """Read multiple records efficiently."""
        async with self._lock:
            results = []
            for id in ids:
                record = self._storage.get(id)
                results.append(record.copy(deep=True) if record else None)
            return results

    async def delete_batch(self, ids: list[str]) -> list[bool]:
        """Delete multiple records efficiently."""
        async with self._lock:
            results = []
            for id in ids:
                if id in self._storage:
                    del self._storage[id]
                    results.append(True)
                else:
                    results.append(False)
            return results

    async def stream_read(
        self,
        query: Query | None = None,
        config: StreamConfig | None = None
    ) -> AsyncIterator[Record]:
        """Stream records from memory."""
        config = config or StreamConfig()

        # Get all matching records
        if query:
            records = await self.search(query)
        else:
            async with self._lock:
                # Ensure records have IDs when getting directly from storage
                records = []
                for record_id, record in self._storage.items():
                    record_copy = self._ensure_record_id(record, record_id)
                    records.append(record_copy)

        # Yield records in batches
        for i in range(0, len(records), config.batch_size):
            batch = records[i:i + config.batch_size]
            for record in batch:
                yield record.copy(deep=True)
                # Small yield to prevent blocking
                await asyncio.sleep(0)

    async def stream_write(
        self,
        records: AsyncIterator[Record],
        config: StreamConfig | None = None
    ) -> StreamResult:
        """Stream records into memory."""
        # Use the default implementation from mixin
        return await self._default_stream_write(records, config)


class SyncMemoryDatabase(SyncDatabase, StreamingMixin, ConfigurableBase):
    """Synchronous in-memory database implementation."""

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self._storage: OrderedDict[str, Record] = OrderedDict()
        self._lock = threading.RLock()

    @classmethod
    def from_config(cls, config: dict) -> "SyncMemoryDatabase":
        """Create from config dictionary."""
        return cls(config)

    def connect(self) -> None:
        """Connect to the database (no-op for memory backend)."""
        pass

    def _generate_id(self) -> str:
        """Generate a unique ID for a record."""
        return str(uuid.uuid4())

    def create(self, record: Record) -> str:
        """Create a new record in memory."""
        with self._lock:
            # Use record's ID if it has one, otherwise generate a new one
            id = record.id if record.id else self._generate_id()
            self._storage[id] = record.copy(deep=True)
            return id

    def read(self, id: str) -> Record | None:
        """Read a record from memory."""
        with self._lock:
            record = self._storage.get(id)
            return record.copy(deep=True) if record else None

    def update(self, id: str, record: Record) -> bool:
        """Update a record in memory."""
        with self._lock:
            if id in self._storage:
                self._storage[id] = record.copy(deep=True)
                return True
            return False

    def delete(self, id: str) -> bool:
        """Delete a record from memory."""
        with self._lock:
            if id in self._storage:
                del self._storage[id]
                return True
            return False

    def exists(self, id: str) -> bool:
        """Check if a record exists in memory."""
        with self._lock:
            return id in self._storage

    def upsert(self, id: str, record: Record) -> str:
        """Update or insert a record with the specified ID."""
        with self._lock:
            self._storage[id] = record.copy(deep=True)
            return id

    def search(self, query: Query | ComplexQuery) -> list[Record]:
        """Search for records matching the query."""
        # Handle ComplexQuery using base class implementation
        if isinstance(query, ComplexQuery):
            return self._search_with_complex_query(query)

        with self._lock:
            results = []

            for id, record in self._storage.items():
                # Apply filters
                matches = True
                for filter in query.filters:
                    field_value = record.get_value(filter.field)
                    if not filter.matches(field_value):
                        matches = False
                        break

                if matches:
                    results.append((id, record))

            # Use the helper method from base class
            return self._process_search_results(results, query, deep_copy=True)

    def _count_all(self) -> int:
        """Count all records in memory."""
        with self._lock:
            return len(self._storage)

    def clear(self) -> int:
        """Clear all records from memory."""
        with self._lock:
            count = len(self._storage)
            self._storage.clear()
            return count

    def create_batch(self, records: list[Record]) -> list[str]:
        """Create multiple records efficiently."""
        with self._lock:
            ids = []
            for record in records:
                # Use record's ID if it has one, otherwise generate a new one
                id = record.id if record.id else self._generate_id()
                self._storage[id] = record.copy(deep=True)
                ids.append(id)
            return ids

    def read_batch(self, ids: list[str]) -> list[Record | None]:
        """Read multiple records efficiently."""
        with self._lock:
            results = []
            for id in ids:
                record = self._storage.get(id)
                results.append(record.copy(deep=True) if record else None)
            return results

    def delete_batch(self, ids: list[str]) -> list[bool]:
        """Delete multiple records efficiently."""
        with self._lock:
            results = []
            for id in ids:
                if id in self._storage:
                    del self._storage[id]
                    results.append(True)
                else:
                    results.append(False)
            return results

    def stream_read(
        self,
        query: Query | None = None,
        config: StreamConfig | None = None
    ) -> Iterator[Record]:
        """Stream records from memory."""
        config = config or StreamConfig()

        # Get all matching records
        if query:
            records = self.search(query)
        else:
            with self._lock:
                # Ensure records have IDs when getting directly from storage
                records = []
                for record_id, record in self._storage.items():
                    record_copy = self._ensure_record_id(record, record_id)
                    records.append(record_copy)

        # Yield records in batches
        for i in range(0, len(records), config.batch_size):
            batch = records[i:i + config.batch_size]
            for record in batch:
                yield record.copy(deep=True)

    def stream_write(
        self,
        records: Iterator[Record],
        config: StreamConfig | None = None
    ) -> StreamResult:
        """Stream records into memory."""
        # Use the default implementation from mixin
        return self._default_stream_write(records, config)
