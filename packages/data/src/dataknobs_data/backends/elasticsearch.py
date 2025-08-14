"""Elasticsearch backend implementation for the data package."""

import asyncio
import uuid
from typing import Any

from dataknobs_utils.elasticsearch_utils import (
    ElasticsearchIndex,
    TableSettings,
)

from ..database import Database, SyncDatabase
from ..exceptions import DatabaseError
from ..query import Operator, Query, SortOrder
from ..records import Record


class SyncElasticsearchDatabase(SyncDatabase):
    """Synchronous Elasticsearch database backend."""

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize Elasticsearch database.

        Args:
            config: Configuration with the following optional keys:
                - host: Elasticsearch host (default: localhost)
                - port: Elasticsearch port (default: 9200)
                - index: Index name (default: "records")
                - refresh: Whether to refresh after write operations (default: True)
                - settings: Index settings dict
                - mappings: Index mappings dict
        """
        super().__init__(config)

    def _initialize(self) -> None:
        """Initialize the Elasticsearch connection and index."""
        config = self.config.copy()

        # Extract configuration
        self.host = config.pop("host", "localhost")
        self.port = config.pop("port", 9200)
        self.index_name = config.pop("index", "records")
        self.refresh = config.pop("refresh", True)

        # Create index settings
        settings = config.pop("settings", {
            "number_of_shards": 1,
            "number_of_replicas": 0,
        })

        # Create mappings for flexible schema
        mappings = config.pop("mappings", {
            "properties": {
                "id": {"type": "keyword"},
                "data": {"type": "object", "enabled": True},
                "metadata": {"type": "object", "enabled": True},
                "created_at": {"type": "date"},
                "updated_at": {"type": "date"},
            }
        })

        # Initialize the Elasticsearch index
        table_settings = TableSettings(
            settings=settings,
            mappings=mappings,
        )

        self.es_index = ElasticsearchIndex(
            index_name=self.index_name,
            host=self.host,
            port=self.port,
            table_settings=table_settings,
        )

        # Ensure index exists
        if not self.es_index.exists():
            self.es_index.create()

    def _record_to_doc(self, record: Record, id: str | None = None) -> dict[str, Any]:
        """Convert a Record to an Elasticsearch document."""
        data = {}
        for field_name, field_obj in record.fields.items():
            data[field_name] = field_obj.value

        doc = {
            "id": id or str(uuid.uuid4()),
            "data": data,
            "metadata": record.metadata or {},
        }

        return doc

    def _doc_to_record(self, doc: dict[str, Any]) -> Record:
        """Convert an Elasticsearch document to a Record."""
        data = doc.get("data", {})
        metadata = doc.get("metadata", {})

        return Record(data=data, metadata=metadata)

    def create(self, record: Record) -> str:
        """Create a new record."""
        id = str(uuid.uuid4())
        doc = self._record_to_doc(record, id)

        # Index the document
        response = self.es_index.index(
            doc_id=id,
            body=doc,
            refresh=self.refresh,
        )

        if not response.succeeded:
            raise DatabaseError(f"Failed to create record: {response.text}")

        return id

    def read(self, id: str) -> Record | None:
        """Read a record by ID."""
        response = self.es_index.get(doc_id=id)

        if not response.succeeded:
            if "404" in str(response.status_code):
                return None
            raise DatabaseError(f"Failed to read record: {response.text}")

        doc = response.json().get("_source", {})
        return self._doc_to_record(doc)

    def update(self, id: str, record: Record) -> bool:
        """Update an existing record."""
        doc = self._record_to_doc(record, id)

        # Update the document
        response = self.es_index.update(
            doc_id=id,
            body={"doc": doc},
            refresh=self.refresh,
        )

        if not response.succeeded:
            if "404" in str(response.status_code):
                return False
            raise DatabaseError(f"Failed to update record: {response.text}")

        return True

    def delete(self, id: str) -> bool:
        """Delete a record by ID."""
        response = self.es_index.delete(
            doc_id=id,
            refresh=self.refresh,
        )

        if not response.succeeded:
            if "404" in str(response.status_code):
                return False
            raise DatabaseError(f"Failed to delete record: {response.text}")

        return True

    def exists(self, id: str) -> bool:
        """Check if a record exists."""
        response = self.es_index.exists(doc_id=id)
        return response.succeeded and response.json()

    def search(self, query: Query) -> list[Record]:
        """Search for records matching a query."""
        # Build Elasticsearch query from Query object
        es_query = {"bool": {"must": []}}

        # Apply filters
        for filter_obj in query.filters:
            field_path = f"data.{filter_obj.field}"

            if filter_obj.operator == Operator.EQ:
                es_query["bool"]["must"].append({"term": {field_path: filter_obj.value}})
            elif filter_obj.operator == Operator.NEQ:
                es_query["bool"]["must"].append({"bool": {"must_not": {"term": {field_path: filter_obj.value}}}})
            elif filter_obj.operator == Operator.GT:
                es_query["bool"]["must"].append({"range": {field_path: {"gt": filter_obj.value}}})
            elif filter_obj.operator == Operator.GTE:
                es_query["bool"]["must"].append({"range": {field_path: {"gte": filter_obj.value}}})
            elif filter_obj.operator == Operator.LT:
                es_query["bool"]["must"].append({"range": {field_path: {"lt": filter_obj.value}}})
            elif filter_obj.operator == Operator.LTE:
                es_query["bool"]["must"].append({"range": {field_path: {"lte": filter_obj.value}}})
            elif filter_obj.operator == Operator.LIKE:
                # Convert SQL LIKE pattern to Elasticsearch wildcard
                pattern = filter_obj.value.replace("%", "*").replace("_", "?")
                es_query["bool"]["must"].append({"wildcard": {field_path: pattern}})
            elif filter_obj.operator == Operator.IN:
                es_query["bool"]["must"].append({"terms": {field_path: filter_obj.value}})
            elif filter_obj.operator == Operator.NOT_IN:
                es_query["bool"]["must"].append({"bool": {"must_not": {"terms": {field_path: filter_obj.value}}}})
            elif filter_obj.operator == Operator.EXISTS:
                es_query["bool"]["must"].append({"exists": {"field": field_path}})
            elif filter_obj.operator == Operator.NOT_EXISTS:
                es_query["bool"]["must"].append({"bool": {"must_not": {"exists": {"field": field_path}}}})
            elif filter_obj.operator == Operator.REGEX:
                es_query["bool"]["must"].append({"regexp": {field_path: filter_obj.value}})

        # If no filters, match all
        if not es_query["bool"]["must"]:
            es_query = {"match_all": {}}

        # Build sort
        sort = []
        if query.sort:
            for sort_spec in query.sort:
                field_path = f"data.{sort_spec.field}"
                order = "desc" if sort_spec.order == SortOrder.DESC else "asc"
                sort.append({field_path: {"order": order}})

        # Build search body
        search_body = {"query": es_query}
        if sort:
            search_body["sort"] = sort
        if query.limit:
            search_body["size"] = query.limit
        if query.offset:
            search_body["from"] = query.offset

        # Execute search
        response = self.es_index.search(body=search_body)

        if not response.succeeded:
            raise DatabaseError(f"Failed to search records: {response.text}")

        # Parse results
        records = []
        hits = response.json().get("hits", {}).get("hits", [])
        for hit in hits:
            doc = hit.get("_source", {})
            records.append(self._doc_to_record(doc))

        # Apply field projection if specified
        if query.fields:
            for record in records:
                # Keep only specified fields
                field_names = list(record.fields.keys())
                for field_name in field_names:
                    if field_name not in query.fields:
                        del record.fields[field_name]

        return records

    def _count_all(self) -> int:
        """Count all records in the database."""
        response = self.es_index.count()

        if not response.succeeded:
            raise DatabaseError(f"Failed to count records: {response.text}")

        return response.json().get("count", 0)

    def clear(self) -> int:
        """Clear all records from the database."""
        # Get count before deletion
        count = self._count_all()

        # Delete by query - delete all documents
        response = self.es_index.delete_by_query(
            body={"query": {"match_all": {}}},
            refresh=self.refresh,
        )

        if not response.succeeded:
            raise DatabaseError(f"Failed to clear records: {response.text}")

        return count

    def close(self) -> None:
        """Close the database connection."""
        # ElasticsearchIndex manages its own connections
        pass


class ElasticsearchDatabase(Database):
    """Asynchronous Elasticsearch database backend."""

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize async Elasticsearch database."""
        # Create sync database for delegation
        self._sync_db = SyncElasticsearchDatabase(config)
        super().__init__(config)

    def _initialize(self) -> None:
        """Initialize is handled by sync database."""
        pass

    async def create(self, record: Record) -> str:
        """Create a new record asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._sync_db.create, record)

    async def read(self, id: str) -> Record | None:
        """Read a record by ID asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._sync_db.read, id)

    async def update(self, id: str, record: Record) -> bool:
        """Update a record asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._sync_db.update, id, record)

    async def delete(self, id: str) -> bool:
        """Delete a record asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._sync_db.delete, id)

    async def search(self, query: Query) -> list[Record]:
        """Search for records asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._sync_db.search, query)

    async def exists(self, id: str) -> bool:
        """Check if a record exists asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._sync_db.exists, id)

    async def _count_all(self) -> int:
        """Count all records asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._sync_db._count_all)

    async def clear(self) -> int:
        """Clear all records asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._sync_db.clear)

    async def close(self) -> None:
        """Close the database connection asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._sync_db.close)
