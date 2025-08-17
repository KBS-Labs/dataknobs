"""In-memory database backend implementation."""

import asyncio
import threading
import time
import uuid
from collections import OrderedDict
from typing import Any, AsyncIterator, Iterator, Optional

from dataknobs_config import ConfigurableBase

from ..database import AsyncDatabase, SyncDatabase
from ..query import Query
from ..records import Record
from ..streaming import AsyncStreamingMixin, StreamConfig, StreamResult, StreamingMixin


class AsyncMemoryDatabase(AsyncDatabase, ConfigurableBase):
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

    async def search(self, query: Query) -> list[Record]:
        """Search for records matching the query."""
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

            # Apply sorting
            if query.sort_specs:
                for sort_spec in reversed(query.sort_specs):
                    reverse = sort_spec.order.value == "desc"
                    results.sort(key=lambda x: x[1].get_value(sort_spec.field, ""), reverse=reverse)

            # Extract records
            records = [record for _, record in results]

            # Apply offset and limit
            if query.offset_value:
                records = records[query.offset_value :]
            if query.limit_value:
                records = records[: query.limit_value]

            # Apply field projection
            if query.fields:
                projected_records = []
                for record in records:
                    projected_records.append(record.project(query.fields))
                records = projected_records

            # Return deep copies
            return [record.copy(deep=True) for record in records]

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
        query: Optional[Query] = None,
        config: Optional[StreamConfig] = None
    ) -> AsyncIterator[Record]:
        """Stream records from memory."""
        config = config or StreamConfig()
        
        # Get all matching records
        if query:
            records = await self.search(query)
        else:
            async with self._lock:
                records = list(self._storage.values())
        
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
        config: Optional[StreamConfig] = None
    ) -> StreamResult:
        """Stream records into memory."""
        config = config or StreamConfig()
        result = StreamResult()
        start_time = time.time()
        
        batch = []
        async for record in records:
            batch.append(record)
            
            if len(batch) >= config.batch_size:
                # Write batch
                try:
                    ids = await self.create_batch(batch)
                    result.successful += len(ids)
                    result.total_processed += len(batch)
                except Exception as e:
                    result.failed += len(batch)
                    result.total_processed += len(batch)
                    if config.on_error:
                        for rec in batch:
                            if not config.on_error(e, rec):
                                result.add_error(None, e)
                                break
                    else:
                        result.add_error(None, e)
                
                batch = []
        
        # Write remaining batch
        if batch:
            try:
                ids = await self.create_batch(batch)
                result.successful += len(ids)
                result.total_processed += len(batch)
            except Exception as e:
                result.failed += len(batch)
                result.total_processed += len(batch)
                result.add_error(None, e)
        
        result.duration = time.time() - start_time
        return result


class SyncMemoryDatabase(SyncDatabase, ConfigurableBase):
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

    def search(self, query: Query) -> list[Record]:
        """Search for records matching the query."""
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

            # Apply sorting
            if query.sort_specs:
                for sort_spec in reversed(query.sort_specs):
                    reverse = sort_spec.order.value == "desc"
                    results.sort(key=lambda x: x[1].get_value(sort_spec.field, ""), reverse=reverse)

            # Extract records
            records = [record for _, record in results]

            # Apply offset and limit
            if query.offset_value:
                records = records[query.offset_value :]
            if query.limit_value:
                records = records[: query.limit_value]

            # Apply field projection
            if query.fields:
                projected_records = []
                for record in records:
                    projected_records.append(record.project(query.fields))
                records = projected_records

            # Return deep copies
            return [record.copy(deep=True) for record in records]

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
        query: Optional[Query] = None,
        config: Optional[StreamConfig] = None
    ) -> Iterator[Record]:
        """Stream records from memory."""
        config = config or StreamConfig()
        
        # Get all matching records
        if query:
            records = self.search(query)
        else:
            with self._lock:
                records = list(self._storage.values())
        
        # Yield records in batches
        for i in range(0, len(records), config.batch_size):
            batch = records[i:i + config.batch_size]
            for record in batch:
                yield record.copy(deep=True)
    
    def stream_write(
        self,
        records: Iterator[Record],
        config: Optional[StreamConfig] = None
    ) -> StreamResult:
        """Stream records into memory."""
        config = config or StreamConfig()
        result = StreamResult()
        start_time = time.time()
        
        batch = []
        for record in records:
            batch.append(record)
            
            if len(batch) >= config.batch_size:
                # Write batch
                try:
                    ids = self.create_batch(batch)
                    result.successful += len(ids)
                    result.total_processed += len(batch)
                except Exception as e:
                    result.failed += len(batch)
                    result.total_processed += len(batch)
                    if config.on_error:
                        for rec in batch:
                            if not config.on_error(e, rec):
                                result.add_error(None, e)
                                break
                    else:
                        result.add_error(None, e)
                
                batch = []
        
        # Write remaining batch
        if batch:
            try:
                ids = self.create_batch(batch)
                result.successful += len(ids)
                result.total_processed += len(batch)
            except Exception as e:
                result.failed += len(batch)
                result.total_processed += len(batch)
                result.add_error(None, e)
        
        result.duration = time.time() - start_time
        return result
