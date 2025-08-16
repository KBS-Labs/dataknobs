"""Native async Elasticsearch backend implementation with connection pooling."""

import asyncio
import json
import logging
import time
import uuid
from typing import Any, AsyncIterator, Optional

from dataknobs_config import ConfigurableBase

from ..database import AsyncDatabase
from ..exceptions import DatabaseError
from ..pooling import ConnectionPoolManager
from ..pooling.elasticsearch import (
    ElasticsearchPoolConfig,
    create_async_elasticsearch_client,
    validate_elasticsearch_client,
    close_elasticsearch_client
)
from ..query import Operator, Query, SortOrder
from ..records import Record
from ..streaming import StreamConfig, StreamResult

logger = logging.getLogger(__name__)

# Global pool manager for Elasticsearch clients
_client_manager = ConnectionPoolManager()


class AsyncElasticsearchDatabase(AsyncDatabase, ConfigurableBase):
    """Native async Elasticsearch database backend with connection pooling."""
    
    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize async Elasticsearch database."""
        super().__init__(config)
        config = config or {}
        self._pool_config = ElasticsearchPoolConfig.from_dict(config)
        self.index_name = self._pool_config.index
        self.refresh = config.get("refresh", True)
        self._client = None
        self._connected = False
    
    @classmethod
    def from_config(cls, config: dict) -> "AsyncElasticsearchDatabase":
        """Create from config dictionary."""
        return cls(config)
    
    async def connect(self) -> None:
        """Connect to the Elasticsearch database."""
        if self._connected:
            return
        
        # Get or create client for current event loop
        self._client = await _client_manager.get_pool(
            self._pool_config,
            create_async_elasticsearch_client,
            validate_elasticsearch_client,
            close_elasticsearch_client
        )
        
        # Ensure index exists
        await self._ensure_index()
        self._connected = True
    
    async def close(self) -> None:
        """Close the database connection."""
        if self._connected:
            # Note: The client is managed by the pool manager, so we don't close it here
            # Just mark as disconnected
            self._client = None
            self._connected = False
    
    def _initialize(self) -> None:
        """Initialize is handled in connect."""
        pass
    
    async def _ensure_index(self) -> None:
        """Ensure the index exists with proper mappings."""
        if not self._client:
            raise RuntimeError("Database not connected. Call connect() first.")
        
        # Check if index exists
        if not await self._client.indices.exists(index=self.index_name):
            # Create index with mappings
            mappings = {
                "properties": {
                    "data": {"type": "object", "enabled": True},
                    "metadata": {"type": "object", "enabled": True},
                    "created_at": {"type": "date"},
                    "updated_at": {"type": "date"}
                }
            }
            
            await self._client.indices.create(
                index=self.index_name,
                mappings=mappings
            )
    
    def _check_connection(self) -> None:
        """Check if database is connected."""
        if not self._connected or not self._client:
            raise RuntimeError("Database not connected. Call connect() first.")
    
    def _record_to_doc(self, record: Record) -> dict[str, Any]:
        """Convert a Record to an Elasticsearch document."""
        doc = {
            "data": {},
            "metadata": record.metadata or {}
        }
        
        for field_name, field_obj in record.fields.items():
            doc["data"][field_name] = field_obj.value
        
        return doc
    
    def _doc_to_record(self, doc: dict[str, Any]) -> Record:
        """Convert an Elasticsearch document to a Record."""
        data = doc.get("_source", {}).get("data", {})
        metadata = doc.get("_source", {}).get("metadata", {})
        
        # Add document ID to metadata
        if "_id" in doc:
            metadata["id"] = doc["_id"]
        if "_score" in doc:
            metadata["_score"] = doc["_score"]
        
        return Record(data=data, metadata=metadata)
    
    async def create(self, record: Record) -> str:
        """Create a new record."""
        self._check_connection()
        doc = self._record_to_doc(record)
        
        # Create document
        response = await self._client.index(
            index=self.index_name,
            document=doc,
            refresh=self.refresh
        )
        
        return response["_id"]
    
    async def create_batch(self, records: list[Record]) -> list[str]:
        """Create multiple records in batch."""
        self._check_connection()
        
        ids = []
        operations = []
        
        for record in records:
            doc = self._record_to_doc(record)
            operations.append({"index": {"_index": self.index_name}})
            operations.append(doc)
        
        if operations:
            response = await self._client.bulk(
                operations=operations,
                refresh=self.refresh
            )
            
            # Extract IDs from response
            for item in response.get("items", []):
                if "index" in item and "_id" in item["index"]:
                    ids.append(item["index"]["_id"])
        
        return ids
    
    async def read(self, id: str) -> Record | None:
        """Read a record by ID."""
        self._check_connection()
        
        try:
            response = await self._client.get(
                index=self.index_name,
                id=id
            )
            return self._doc_to_record(response)
        except Exception:
            return None
    
    async def update(self, id: str, record: Record) -> bool:
        """Update an existing record."""
        self._check_connection()
        doc = self._record_to_doc(record)
        
        try:
            await self._client.update(
                index=self.index_name,
                id=id,
                doc=doc,
                refresh=self.refresh
            )
            return True
        except Exception:
            return False
    
    async def delete(self, id: str) -> bool:
        """Delete a record by ID."""
        self._check_connection()
        
        try:
            await self._client.delete(
                index=self.index_name,
                id=id,
                refresh=self.refresh
            )
            return True
        except Exception:
            return False
    
    async def exists(self, id: str) -> bool:
        """Check if a record exists."""
        self._check_connection()
        
        return await self._client.exists(
            index=self.index_name,
            id=id
        )
    
    async def upsert(self, id: str, record: Record) -> str:
        """Update or insert a record with a specific ID."""
        self._check_connection()
        doc = self._record_to_doc(record)
        
        await self._client.index(
            index=self.index_name,
            id=id,
            document=doc,
            refresh=self.refresh
        )
        
        return id
    
    async def search(self, query: Query) -> list[Record]:
        """Search for records matching the query."""
        self._check_connection()
        
        # Build Elasticsearch query
        es_query = {"bool": {"must": []}}
        
        for filter in query.filters:
            field_path = f"data.{filter.field}"
            
            if filter.operator == Operator.EQ:
                # For string values, use keyword field for exact matching
                if isinstance(filter.value, str):
                    field_path = f"{field_path}.keyword"
                es_query["bool"]["must"].append({"term": {field_path: filter.value}})
            elif filter.operator == Operator.NEQ:
                es_query["bool"]["must_not"] = es_query["bool"].get("must_not", [])
                es_query["bool"]["must_not"].append({"term": {field_path: filter.value}})
            elif filter.operator == Operator.GT:
                es_query["bool"]["must"].append({"range": {field_path: {"gt": filter.value}}})
            elif filter.operator == Operator.LT:
                es_query["bool"]["must"].append({"range": {field_path: {"lt": filter.value}}})
            elif filter.operator == Operator.GTE:
                es_query["bool"]["must"].append({"range": {field_path: {"gte": filter.value}}})
            elif filter.operator == Operator.LTE:
                es_query["bool"]["must"].append({"range": {field_path: {"lte": filter.value}}})
            elif filter.operator == Operator.LIKE:
                es_query["bool"]["must"].append({"wildcard": {field_path: f"*{filter.value}*"}})
            elif filter.operator == Operator.IN:
                es_query["bool"]["must"].append({"terms": {field_path: filter.value}})
            elif filter.operator == Operator.NOT_IN:
                es_query["bool"]["must_not"] = es_query["bool"].get("must_not", [])
                es_query["bool"]["must_not"].append({"terms": {field_path: filter.value}})
        
        # If no filters, use match_all
        if not es_query["bool"]["must"] and "must_not" not in es_query["bool"]:
            es_query = {"match_all": {}}
        
        # Build sort
        sort = []
        if query.sort_specs:
            for sort_spec in query.sort_specs:
                direction = "desc" if sort_spec.order == SortOrder.DESC else "asc"
                sort.append({f"data.{sort_spec.field}": {"order": direction}})
        
        # Build request body
        body = {"query": es_query}
        if sort:
            body["sort"] = sort
        
        # Add size and from for pagination
        size = query.limit_value if query.limit_value else 10000
        from_param = query.offset_value if query.offset_value else 0
        
        # Execute search
        response = await self._client.search(
            index=self.index_name,
            query=es_query,
            sort=sort if sort else None,
            size=size,
            from_=from_param
        )
        
        # Convert to records
        records = []
        for hit in response["hits"]["hits"]:
            record = self._doc_to_record(hit)
            
            # Apply field projection if specified
            if query.fields:
                record = record.project(query.fields)
            
            records.append(record)
        
        return records
    
    async def _count_all(self) -> int:
        """Count all records in the database."""
        self._check_connection()
        
        response = await self._client.count(index=self.index_name)
        return response["count"]
    
    async def clear(self) -> int:
        """Clear all records from the database."""
        self._check_connection()
        
        # Get count before deletion
        count = await self._count_all()
        
        # Delete by query - delete all documents
        response = await self._client.delete_by_query(
            index=self.index_name,
            query={"match_all": {}},
            refresh=self.refresh
        )
        
        return response.get("deleted", count)
    
    async def stream_read(
        self,
        query: Optional[Query] = None,
        config: Optional[StreamConfig] = None
    ) -> AsyncIterator[Record]:
        """Stream records from Elasticsearch using scroll API."""
        self._check_connection()
        config = config or StreamConfig()
        
        # Build query
        es_query = {"match_all": {}}
        if query and query.filters:
            es_query = {"bool": {"must": []}}
            for filter in query.filters:
                field_path = f"data.{filter.field}"
                if filter.operator == Operator.EQ:
                    es_query["bool"]["must"].append({"term": {field_path: filter.value}})
        
        # Initial search with scroll
        response = await self._client.search(
            index=self.index_name,
            query=es_query,
            scroll="2m",
            size=config.batch_size
        )
        
        scroll_id = response["_scroll_id"]
        hits = response["hits"]["hits"]
        
        try:
            while hits:
                for hit in hits:
                    record = self._doc_to_record(hit)
                    if query and query.fields:
                        record = record.project(query.fields)
                    yield record
                
                # Get next batch
                response = await self._client.scroll(
                    scroll_id=scroll_id,
                    scroll="2m"
                )
                hits = response["hits"]["hits"]
        finally:
            # Clear scroll
            await self._client.clear_scroll(scroll_id=scroll_id)
    
    async def stream_write(
        self,
        records: AsyncIterator[Record],
        config: Optional[StreamConfig] = None
    ) -> StreamResult:
        """Stream records into Elasticsearch using bulk API."""
        self._check_connection()
        config = config or StreamConfig()
        result = StreamResult()
        start_time = time.time()
        
        batch = []
        async for record in records:
            batch.append(record)
            
            if len(batch) >= config.batch_size:
                # Write batch using bulk API
                try:
                    await self._write_batch(batch)
                    result.successful += len(batch)
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
                await self._write_batch(batch)
                result.successful += len(batch)
                result.total_processed += len(batch)
            except Exception as e:
                result.failed += len(batch)
                result.total_processed += len(batch)
                result.add_error(None, e)
        
        result.duration = time.time() - start_time
        return result
    
    async def _write_batch(self, records: list[Record]) -> None:
        """Write a batch of records using bulk API."""
        if not records:
            return
        
        # Build bulk operations
        operations = []
        for record in records:
            doc = self._record_to_doc(record)
            operations.append({"index": {"_index": self.index_name}})
            operations.append(doc)
        
        # Execute bulk
        await self._client.bulk(
            operations=operations,
            refresh=self.refresh
        )