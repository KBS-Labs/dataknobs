"""In-memory database backend implementation."""

import asyncio
import threading
import uuid
from collections import OrderedDict
from typing import Any, Dict, List, Optional

from ..database import Database, SyncDatabase
from ..exceptions import RecordNotFoundError
from ..query import Query
from ..records import Record


class MemoryDatabase(Database):
    """Async in-memory database implementation."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._storage: OrderedDict[str, Record] = OrderedDict()
        self._lock = asyncio.Lock()
    
    def _generate_id(self) -> str:
        """Generate a unique ID for a record."""
        return str(uuid.uuid4())
    
    async def create(self, record: Record) -> str:
        """Create a new record in memory."""
        async with self._lock:
            id = self._generate_id()
            self._storage[id] = record.copy(deep=True)
            return id
    
    async def read(self, id: str) -> Optional[Record]:
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
    
    async def search(self, query: Query) -> List[Record]:
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
            if query.sort:
                for sort_spec in reversed(query.sort):
                    reverse = sort_spec.order.value == "desc"
                    results.sort(
                        key=lambda x: x[1].get_value(sort_spec.field, ""),
                        reverse=reverse
                    )
            
            # Extract records
            records = [record for _, record in results]
            
            # Apply offset and limit
            if query.offset:
                records = records[query.offset:]
            if query.limit:
                records = records[:query.limit]
            
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
    
    async def create_batch(self, records: List[Record]) -> List[str]:
        """Create multiple records efficiently."""
        async with self._lock:
            ids = []
            for record in records:
                id = self._generate_id()
                self._storage[id] = record.copy(deep=True)
                ids.append(id)
            return ids
    
    async def read_batch(self, ids: List[str]) -> List[Optional[Record]]:
        """Read multiple records efficiently."""
        async with self._lock:
            results = []
            for id in ids:
                record = self._storage.get(id)
                results.append(record.copy(deep=True) if record else None)
            return results
    
    async def delete_batch(self, ids: List[str]) -> List[bool]:
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


class SyncMemoryDatabase(SyncDatabase):
    """Synchronous in-memory database implementation."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._storage: OrderedDict[str, Record] = OrderedDict()
        self._lock = threading.RLock()
    
    def _generate_id(self) -> str:
        """Generate a unique ID for a record."""
        return str(uuid.uuid4())
    
    def create(self, record: Record) -> str:
        """Create a new record in memory."""
        with self._lock:
            id = self._generate_id()
            self._storage[id] = record.copy(deep=True)
            return id
    
    def read(self, id: str) -> Optional[Record]:
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
    
    def search(self, query: Query) -> List[Record]:
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
            if query.sort:
                for sort_spec in reversed(query.sort):
                    reverse = sort_spec.order.value == "desc"
                    results.sort(
                        key=lambda x: x[1].get_value(sort_spec.field, ""),
                        reverse=reverse
                    )
            
            # Extract records
            records = [record for _, record in results]
            
            # Apply offset and limit
            if query.offset:
                records = records[query.offset:]
            if query.limit:
                records = records[:query.limit]
            
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
    
    def create_batch(self, records: List[Record]) -> List[str]:
        """Create multiple records efficiently."""
        with self._lock:
            ids = []
            for record in records:
                id = self._generate_id()
                self._storage[id] = record.copy(deep=True)
                ids.append(id)
            return ids
    
    def read_batch(self, ids: List[str]) -> List[Optional[Record]]:
        """Read multiple records efficiently."""
        with self._lock:
            results = []
            for id in ids:
                record = self._storage.get(id)
                results.append(record.copy(deep=True) if record else None)
            return results
    
    def delete_batch(self, ids: List[str]) -> List[bool]:
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