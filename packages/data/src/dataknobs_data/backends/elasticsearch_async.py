"""Native async Elasticsearch backend implementation with connection pooling."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, cast

from dataknobs_config import ConfigurableBase

from ..database import AsyncDatabase
from ..pooling import ConnectionPoolManager
from ..pooling.elasticsearch import (
    ElasticsearchPoolConfig,
    close_elasticsearch_client,
    create_async_elasticsearch_client,
    validate_elasticsearch_client,
)
from ..query import Operator, Query, SortOrder
from ..streaming import StreamConfig, StreamResult, async_process_batch_with_fallback
from ..vector.mixins import VectorOperationsMixin
from ..vector.types import DistanceMetric, VectorSearchResult
from .elasticsearch_mixins import (
    ElasticsearchBaseConfig,
    ElasticsearchErrorHandler,
    ElasticsearchIndexManager,
    ElasticsearchQueryBuilder,
    ElasticsearchRecordSerializer,
    ElasticsearchVectorSupport,
)

if TYPE_CHECKING:
    import numpy as np
    from collections.abc import AsyncIterator, Callable, Awaitable
    from ..records import Record

logger = logging.getLogger(__name__)

# Global pool manager for Elasticsearch clients
_client_manager = ConnectionPoolManager()


class AsyncElasticsearchDatabase(
    AsyncDatabase,
    ConfigurableBase,
    VectorOperationsMixin,
    ElasticsearchBaseConfig,
    ElasticsearchIndexManager,
    ElasticsearchVectorSupport,
    ElasticsearchErrorHandler,
    ElasticsearchRecordSerializer,
    ElasticsearchQueryBuilder,
):
    """Native async Elasticsearch database backend with connection pooling."""

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize async Elasticsearch database."""
        super().__init__(config)

        # Initialize vector support
        self.vector_fields = {}  # field_name -> dimensions
        self.vector_enabled = False

        config = config or {}
        self._pool_config = ElasticsearchPoolConfig.from_dict(config)
        self.index_name = self._pool_config.index
        self.refresh = config.get("refresh", True)
        self._client = None
        self._connected = False

    @classmethod
    def from_config(cls, config: dict) -> AsyncElasticsearchDatabase:
        """Create from config dictionary."""
        return cls(config)

    async def connect(self) -> None:
        """Connect to the Elasticsearch database."""
        if self._connected:
            return

        # Get or create client for current event loop
        from ..pooling import BasePoolConfig
        self._client = await _client_manager.get_pool(
            self._pool_config,
            cast("Callable[[BasePoolConfig], Awaitable[Any]]", create_async_elasticsearch_client),
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
        if not await self._client.indices.exists(index=self.index_name):  # type: ignore[unreachable]
            # Get mappings with vector field support
            mappings = self.get_index_mappings(self.vector_fields)

            # Get settings optimized for KNN if we have vector fields
            settings = self.get_knn_index_settings() if self.vector_fields else {
                "number_of_shards": 1,
                "number_of_replicas": 0,
            }

            await self._client.indices.create(
                index=self.index_name,
                mappings=mappings,
                settings=settings
            )

            if self.vector_fields:
                self.vector_enabled = True
                logger.info(f"Created index '{self.index_name}' with vector support")

    def _check_connection(self) -> None:
        """Check if database is connected."""
        if not self._connected or not self._client:
            raise RuntimeError("Database not connected. Call connect() first.")

    def _record_to_doc(self, record: Record) -> dict[str, Any]:
        """Convert a Record to an Elasticsearch document."""
        # Update vector tracking if needed
        if self._has_vector_fields(record):
            self._update_vector_tracking(record)

            # Add vector field metadata to record metadata
            if "vector_fields" not in record.metadata:
                record.metadata["vector_fields"] = {}

            for field_name in self.vector_fields:
                if field_name in record.fields:
                    field = record.fields[field_name]
                    if hasattr(field, "source_field"):
                        record.metadata["vector_fields"][field_name] = {
                            "type": "vector",
                            "dimensions": self.vector_fields[field_name],
                            "source_field": field.source_field,
                            "model": getattr(field, "model_name", None),
                            "model_version": getattr(field, "model_version", None),
                        }

        return self._record_to_document(record)

    def _doc_to_record(self, doc: dict[str, Any]) -> Record:
        """Convert an Elasticsearch document to a Record."""
        doc_id = doc.get("_id")
        record = self._document_to_record(doc, doc_id)

        # Add score if present
        if "_score" in doc:
            record.metadata["_score"] = doc["_score"]

        return record

    async def create(self, record: Record) -> str:
        """Create a new record."""
        self._check_connection()
        doc = self._record_to_doc(record)

        # Create document with explicit ID if record has one
        kwargs = {
            "index": self.index_name,
            "document": doc,
            "refresh": self.refresh
        }
        if record.id:
            kwargs["id"] = record.id

        response = await self._client.index(**kwargs)

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
        except Exception as e:
            # Log the error for debugging
            logger.debug(f"Error reading document {id}: {e}")
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

    async def upsert(self, id_or_record: str | Record, record: Record | None = None) -> str:
        """Update or insert a record.
        
        Can be called as:
        - upsert(id, record) - explicit ID and record
        - upsert(record) - extract ID from record using Record's built-in logic
        """
        self._check_connection()
        
        # Determine ID and record based on arguments
        if isinstance(id_or_record, str):
            id = id_or_record
            if record is None:
                raise ValueError("Record required when ID is provided")
        else:
            record = id_or_record
            id = record.id
            if id is None:
                import uuid  # type: ignore[unreachable]
                id = str(uuid.uuid4())
                record.storage_id = id
        
        doc = self._record_to_doc(record)

        await self._client.index(
            index=self.index_name,
            id=id,
            document=doc,
            refresh=self.refresh
        )

        return id

    async def update_batch(self, updates: list[tuple[str, Record]]) -> list[bool]:
        """Update multiple records efficiently using the bulk API.
        
        Uses AsyncElasticsearch's bulk API for efficient batch updates.
        
        Args:
            updates: List of (id, record) tuples to update
            
        Returns:
            List of success flags for each update
        """
        if not updates:
            return []

        self._check_connection()

        # Build bulk operations for AsyncElasticsearch
        operations: list[dict[str, Any]] = []
        for record_id, record in updates:
            # Add update operation
            operations.append({
                "update": {
                    "_index": self.index_name,
                    "_id": record_id
                }
            })
            # Add document data
            doc = self._record_to_doc(record)
            operations.append({
                "doc": doc,
                "doc_as_upsert": False  # Don't create if doesn't exist
            })

        try:
            # Execute bulk update using AsyncElasticsearch
            response = await self._client.bulk(
                operations=operations,
                refresh=self.refresh
            )

            # Process the response to determine which updates succeeded
            results = []
            if response.get("items"):
                for item in response["items"]:
                    if "update" in item:
                        update_result = item["update"]
                        # Check if update was successful (status 200) or not found (404)
                        results.append(update_result.get("status") == 200)
                    else:
                        results.append(False)
            else:
                # If no items in response, mark all as failed
                results = [False] * len(updates)

            return results

        except Exception as e:
            # If bulk operation fails, mark all as failed
            import logging
            logging.error(f"Bulk update failed: {e}")
            return [False] * len(updates)

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
            elif filter.operator == Operator.BETWEEN:
                # Use Elasticsearch's native range query for efficient BETWEEN
                if isinstance(filter.value, (list, tuple)) and len(filter.value) == 2:
                    lower, upper = filter.value
                    es_query["bool"]["must"].append({
                        "range": {
                            field_path: {
                                "gte": lower,
                                "lte": upper
                            }
                        }
                    })
            elif filter.operator == Operator.NOT_BETWEEN:
                # NOT BETWEEN using must_not with range
                if isinstance(filter.value, (list, tuple)) and len(filter.value) == 2:
                    lower, upper = filter.value
                    es_query["bool"]["must_not"] = es_query["bool"].get("must_not", [])
                    es_query["bool"]["must_not"].append({
                        "range": {
                            field_path: {
                                "gte": lower,
                                "lte": upper
                            }
                        }
                    })

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
        query: Query | None = None,
        config: StreamConfig | None = None
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
        config: StreamConfig | None = None
    ) -> StreamResult:
        """Stream records into Elasticsearch using bulk API."""
        self._check_connection()
        config = config or StreamConfig()
        result = StreamResult()
        start_time = time.time()
        quitting = False

        batch = []
        async for record in records:
            batch.append(record)

            if len(batch) >= config.batch_size:
                # Write batch with graceful fallback
                async def batch_func(b):
                    await self._write_batch(b)
                    return [r.id for r in b]

                continue_processing = await async_process_batch_with_fallback(
                    batch,
                    batch_func,
                    self.create,
                    result,
                    config
                )

                if not continue_processing:
                    quitting = True
                    break

                batch = []

        # Write remaining batch
        if batch and not quitting:
            async def batch_func(b):
                await self._write_batch(b)
                return [r.id for r in b]

            await async_process_batch_with_fallback(
                batch,
                batch_func,
                self.create,
                result,
                config
            )

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

    async def vector_search(
        self,
        query_vector: np.ndarray | list[float],
        vector_field: str = "embedding",
        k: int = 10,
        metric: DistanceMetric = DistanceMetric.COSINE,
        filter: Query | None = None,
        include_source: bool = True,
        score_threshold: float | None = None,
    ) -> list[VectorSearchResult]:
        """Search for similar vectors using Elasticsearch KNN.
        
        Args:
            query_vector: The vector to search for
            vector_field: Name of the vector field to search
            k: Number of results to return
            metric: Distance metric to use
            filter: Optional query filter to apply before vector search
            include_source: Whether to include source document in results
            score_threshold: Optional minimum similarity score
            
        Returns:
            List of search results ordered by similarity
        """
        self._check_connection()

        # Import vector utilities
        from ..vector.elasticsearch_utils import (
            build_knn_query,
        )

        # Build filter query if provided
        filter_query = self._build_filter_query(filter) if filter else None

        # Build KNN query
        query = build_knn_query(
            query_vector=query_vector,
            field_name=vector_field,
            k=k,
            filter_query=filter_query,
        )

        # Execute search
        try:
            response = await self._client.search(
                index=self.index_name,
                **query,  # Unpack the query dict directly
                size=k,
                _source=include_source,
            )
        except Exception as e:
            self._handle_elasticsearch_error(e, "vector search")
            return []

        # Process results
        results = []
        for hit in response.get("hits", {}).get("hits", []):
            score = hit.get("_score", 0.0)

            # Apply score threshold if specified
            if score_threshold is not None and score < score_threshold:
                continue

            # Convert document to record if source included
            record = None
            if include_source:
                record = self._doc_to_record(hit)

            # Set the storage ID on the record if we have one
            if record and not record.has_storage_id():
                record.storage_id = hit["_id"]

            # Skip if no record (shouldn't happen if include_source is True)
            if record is None:
                continue

            results.append(VectorSearchResult(
                record=record,
                score=score,
                vector_field=vector_field,
                metadata={
                    "index": self.index_name,
                    "metric": metric.value,
                    "doc_id": hit["_id"],
                },
            ))

        return results

    async def bulk_embed_and_store(
        self,
        records: list[Record],
        text_field: str | list[str],
        vector_field: str = "embedding",
        embedding_fn: Any | None = None,
        batch_size: int = 100,
        model_name: str | None = None,
        model_version: str | None = None,
    ) -> list[str]:
        """Embed text fields and store vectors with records.
        
        Args:
            records: Records to process
            text_field: Field name(s) containing text to embed
            vector_field: Field name to store vectors in
            embedding_fn: Function to generate embeddings
            batch_size: Number of records to process at once
            model_name: Name of the embedding model
            model_version: Version of the embedding model
            
        Returns:
            List of record IDs that were processed
        """
        # This is a stub implementation
        # Full implementation would require an actual embedding function
        logger.warning("bulk_embed_and_store is not fully implemented for Elasticsearch")
        return []

    async def create_vector_index(
        self,
        vector_field: str = "embedding",
        dimensions: int | None = None,
        metric: DistanceMetric = DistanceMetric.COSINE,
        index_type: str = "auto",
        **kwargs: Any,
    ) -> bool:
        """Create or update index mapping for vector field.
        
        Args:
            vector_field: Name of the vector field to index
            dimensions: Number of dimensions
            metric: Distance metric for the index
            index_type: Type of index (ignored for ES, always uses HNSW)
            **kwargs: Additional index parameters
            
        Returns:
            True if index was created/updated successfully
        """
        self._check_connection()

        if not dimensions:
            if vector_field not in self.vector_fields:
                raise ValueError(f"Unknown dimensions for field '{vector_field}'")
            dimensions = self.vector_fields[vector_field]

        # Import vector utilities
        from ..vector.elasticsearch_utils import (
            get_similarity_for_metric,
            get_vector_mapping,
        )

        # Get similarity function for metric
        similarity = get_similarity_for_metric(metric)

        # Build mapping for the vector field
        mapping = get_vector_mapping(dimensions, similarity)

        # Update index mapping
        try:
            await self._client.indices.put_mapping(
                index=self.index_name,
                properties={
                    f"data.{vector_field}": mapping
                }
            )

            # Track the vector field
            self.vector_fields[vector_field] = dimensions
            self.vector_enabled = True

            logger.info(f"Created vector mapping for field '{vector_field}' with {dimensions} dimensions")
            return True

        except Exception as e:
            self._handle_elasticsearch_error(e, "create vector index")
            return False
