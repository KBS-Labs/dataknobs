"""Elasticsearch backend implementation for the data package."""

import asyncio
import uuid
from typing import Any

from dataknobs_utils.elasticsearch_utils import SimplifiedElasticsearchIndex

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
        self.es_index = SimplifiedElasticsearchIndex(
            index_name=self.index_name,
            host=self.host,
            port=self.port,
            settings=settings,
            mappings=mappings,
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
            body=doc,
            doc_id=id,
            refresh=self.refresh,
        )

        if not response.get("_id"):
            raise DatabaseError(f"Failed to create record: {response}")

        return response["_id"]

    def read(self, id: str) -> Record | None:
        """Read a record by ID."""
        response = self.es_index.get(doc_id=id)

        if not response:
            return None

        doc = response.get("_source", {})
        return self._doc_to_record(doc)

    def update(self, id: str, record: Record) -> bool:
        """Update an existing record."""
        doc = self._record_to_doc(record, id)

        # Update the document
        success = self.es_index.update(
            doc_id=id,
            body={"doc": doc},
            refresh=self.refresh,
        )

        return success

    def delete(self, id: str) -> bool:
        """Delete a record by ID."""
        success = self.es_index.delete(doc_id=id)
        
        # Refresh if needed
        if success and self.refresh:
            self.es_index.refresh()
        
        return success

    def exists(self, id: str) -> bool:
        """Check if a record exists."""
        return self.es_index.exists(doc_id=id)
    
    def create_batch(self, records: list[Record]) -> list[str]:
        """Create multiple records in batch with a single refresh."""
        ids = []
        for record in records:
            # Generate ID
            id = str(uuid.uuid4())
            doc = self._record_to_doc(record, id)
            
            # Index without refresh (we'll refresh once at the end)
            response = self.es_index.index(body=doc, doc_id=id, refresh=False)
            
            if response.get("_id"):
                ids.append(id)
            else:
                ids.append(None)
        
        # Single refresh after all documents are indexed
        if self.refresh and any(ids):
            self.es_index.refresh()
        
        return ids
    
    def read_batch(self, ids: list[str]) -> list[Record | None]:
        """Read multiple records in batch."""
        records = []
        for id in ids:
            record = self.read(id)
            records.append(record)
        return records
    
    def delete_batch(self, ids: list[str]) -> list[bool]:
        """Delete multiple records in batch with a single refresh."""
        results = []
        for id in ids:
            # Delete without refresh (we'll refresh once at the end)
            success = self.es_index.delete(doc_id=id)
            results.append(success)
        
        # Single refresh after all documents are deleted
        if self.refresh and any(results):
            self.es_index.refresh()
        
        return results

    def search(self, query: Query) -> list[Record]:
        """Search for records matching a query."""
        # Build Elasticsearch query from Query object
        es_query = {"bool": {"must": []}}

        # Apply filters
        for filter_obj in query.filters:
            field_path = f"data.{filter_obj.field}"
            
            # For string fields in exact match queries, use .keyword suffix
            # LIKE and REGEX need to use the text field, not keyword
            if filter_obj.operator in [Operator.EQ, Operator.NEQ, Operator.IN, Operator.NOT_IN]:
                if isinstance(filter_obj.value, str) or (
                    isinstance(filter_obj.value, list) and 
                    filter_obj.value and 
                    isinstance(filter_obj.value[0], str)
                ):
                    field_path = f"{field_path}.keyword"
            elif filter_obj.operator == Operator.LIKE:
                # Wildcard needs .keyword for proper matching
                if isinstance(filter_obj.value, str):
                    field_path = f"{field_path}.keyword"

            if filter_obj.operator == Operator.EQ:
                # Handle boolean values correctly
                value = str(filter_obj.value).lower() if isinstance(filter_obj.value, bool) else filter_obj.value
                es_query["bool"]["must"].append({"term": {field_path: value}})
            elif filter_obj.operator == Operator.NEQ:
                value = str(filter_obj.value).lower() if isinstance(filter_obj.value, bool) else filter_obj.value
                es_query["bool"]["must"].append({"bool": {"must_not": {"term": {field_path: value}}}})
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
                # Wildcard queries should use the keyword field for exact matching
                pattern = filter_obj.value.replace("%", "*").replace("_", "?")
                # Use the base field path for LIKE (already has .keyword added above if string)
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
        if query.sort_specs:
            for sort_spec in query.sort_specs:
                field_path = f"data.{sort_spec.field}"
                # Don't add .keyword if user already specified it or for common numeric fields
                # This is a heuristic - ideally we'd check the mapping
                numeric_fields = ['age', 'salary', 'balance', 'count', 'score', 'amount', 'price']
                if (not sort_spec.field.endswith('.keyword') and 
                    not sort_spec.field.endswith('.raw') and
                    sort_spec.field.lower() not in numeric_fields):
                    # Likely a text field, add .keyword for sorting
                    field_path = f"data.{sort_spec.field}.keyword"
                order = "desc" if sort_spec.order == SortOrder.DESC else "asc"
                sort.append({field_path: {"order": order}})

        # Build search body
        search_body = {"query": es_query}
        if sort:
            search_body["sort"] = sort
        if query.limit_value:
            search_body["size"] = query.limit_value
        if query.offset_value:
            search_body["from"] = query.offset_value

        # Execute search
        response = self.es_index.search(body=search_body)

        if not response.succeeded:
            raise DatabaseError(f"Failed to search records: {response.text}")

        # Parse results
        records = []
        hits = response.json.get("hits", {}).get("hits", [])
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
        return self.es_index.count()
    
    def count(self, query: Query | None = None) -> int:
        """Count records matching a query using efficient Elasticsearch count.
        
        Args:
            query: Optional search query (counts all if None)
            
        Returns:
            Number of matching records
        """
        if not query or not query.filters:
            return self._count_all()
        
        # Build Elasticsearch query from Query object (same as search)
        es_query = {"bool": {"must": []}}
        
        for filter_obj in query.filters:
            field_path = f"data.{filter_obj.field}"
            
            # For string fields in exact match queries, use .keyword suffix
            # LIKE and REGEX need different handling
            if filter_obj.operator in [Operator.EQ, Operator.NEQ, Operator.IN, Operator.NOT_IN]:
                if isinstance(filter_obj.value, str) or (
                    isinstance(filter_obj.value, list) and 
                    filter_obj.value and 
                    isinstance(filter_obj.value[0], str)
                ):
                    field_path = f"{field_path}.keyword"
            elif filter_obj.operator == Operator.LIKE:
                # Wildcard needs .keyword for proper matching
                if isinstance(filter_obj.value, str):
                    field_path = f"{field_path}.keyword"
            
            if filter_obj.operator == Operator.EQ:
                # Handle boolean values correctly
                value = str(filter_obj.value).lower() if isinstance(filter_obj.value, bool) else filter_obj.value
                es_query["bool"]["must"].append({"term": {field_path: value}})
            elif filter_obj.operator == Operator.NEQ:
                value = str(filter_obj.value).lower() if isinstance(filter_obj.value, bool) else filter_obj.value
                es_query["bool"]["must"].append({"bool": {"must_not": {"term": {field_path: value}}}})
            elif filter_obj.operator == Operator.GT:
                es_query["bool"]["must"].append({"range": {field_path: {"gt": filter_obj.value}}})
            elif filter_obj.operator == Operator.GTE:
                es_query["bool"]["must"].append({"range": {field_path: {"gte": filter_obj.value}}})
            elif filter_obj.operator == Operator.LT:
                es_query["bool"]["must"].append({"range": {field_path: {"lt": filter_obj.value}}})
            elif filter_obj.operator == Operator.LTE:
                es_query["bool"]["must"].append({"range": {field_path: {"lte": filter_obj.value}}})
            elif filter_obj.operator == Operator.LIKE:
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
        
        # If no filters were added, use match_all
        if not es_query["bool"]["must"]:
            es_query = {"match_all": {}}
        
        # Count with the query
        return self.es_index.count(body={"query": es_query})

    def clear(self) -> int:
        """Clear all records from the database."""
        # Get count before deletion
        count = self._count_all()

        # Delete by query - delete all documents
        response = self.es_index.delete_by_query(
            body={"query": {"match_all": {}}}
        )
        
        # Refresh if needed
        if self.refresh:
            self.es_index.refresh()
        
        return response.get("deleted", count)

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
    
    async def count(self, query: Query | None = None) -> int:
        """Count records matching a query asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._sync_db.count, query)

    async def clear(self) -> int:
        """Clear all records asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._sync_db.clear)

    async def close(self) -> None:
        """Close the database connection asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._sync_db.close)
