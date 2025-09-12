"""Elasticsearch backend implementation for the data package."""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Any

from dataknobs_config import ConfigurableBase

from dataknobs_utils.elasticsearch_utils import SimplifiedElasticsearchIndex

from ..database import SyncDatabase
from ..exceptions import DatabaseError
from ..query import Operator, Query, SortOrder
from ..query_logic import ComplexQuery
from ..streaming import StreamConfig, StreamingMixin, StreamResult
from ..vector.types import DistanceMetric, VectorSearchResult
from .elasticsearch_mixins import (
    ElasticsearchBaseConfig,
    ElasticsearchErrorHandler,
    ElasticsearchIndexManager,
    ElasticsearchQueryBuilder,
    ElasticsearchRecordSerializer,
    ElasticsearchVectorSupport,
)
from .vector_config_mixin import VectorConfigMixin

if TYPE_CHECKING:
    import numpy as np
    from collections.abc import Iterator
    from ..records import Record

logger = logging.getLogger(__name__)


class SyncElasticsearchDatabase(
    SyncDatabase,
    StreamingMixin,
    ConfigurableBase,
    VectorConfigMixin,
    ElasticsearchBaseConfig,
    ElasticsearchIndexManager,
    ElasticsearchVectorSupport,
    ElasticsearchErrorHandler,
    ElasticsearchRecordSerializer,
    ElasticsearchQueryBuilder,
):
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

        # Parse vector configuration using the mixin
        self._parse_vector_config(config)

        # Initialize vector support
        self.vector_fields = {}  # field_name -> dimensions

        self.es_index = None  # Will be initialized in connect()
        self._connected = False

    @classmethod
    def from_config(cls, config: dict) -> SyncElasticsearchDatabase:
        """Create from config dictionary."""
        return cls(config)

    def connect(self) -> None:
        """Connect to the Elasticsearch database."""
        if self._connected:
            return  # Already connected

        # Initialize the Elasticsearch connection and index
        config = self.config.copy()

        # Extract configuration
        self.host = config.pop("host", "localhost")
        self.port = config.pop("port", 9200)
        self.index_name = config.pop("index", "records")
        self.refresh = config.pop("refresh", True)

        # If vector is enabled but no vector fields defined yet, set up default
        if self._vector_enabled and not self.vector_fields:
            # Set a default embedding field with configurable dimensions
            default_dimensions = config.pop("vector_dimensions", 1536)  # Common default
            default_field = config.pop("default_vector_field", "embedding")
            self.vector_fields[default_field] = default_dimensions

        # Get mappings with vector field support
        base_mappings = self.get_index_mappings(self.vector_fields)

        # Allow custom mappings to override
        custom_mappings = config.pop("mappings", None)
        if custom_mappings:
            mappings = custom_mappings
        else:
            mappings = base_mappings

        # Get settings optimized for KNN if we have vector fields
        settings = config.pop("settings", None)
        if not settings:
            settings = self.get_knn_index_settings() if (self.vector_fields or self._vector_enabled) else {
                "number_of_shards": 1,
                "number_of_replicas": 0,
            }

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

        # Create an Elasticsearch client for bulk operations
        from elasticsearch import Elasticsearch
        self.es_client = Elasticsearch([f"http://{self.host}:{self.port}"])

        self._connected = True

    def close(self) -> None:
        """Close the database connection."""
        if self.es_index:
            # ElasticsearchIndex manages its own connections
            self._connected = False  # type: ignore[unreachable]

    def _initialize(self) -> None:
        """Initialize method - connection setup moved to connect()."""
        # Configuration parsing stays here if needed
        pass

    def _check_connection(self) -> None:
        """Check if database is connected."""
        if not self._connected or not self.es_index:
            raise RuntimeError("Database not connected. Call connect() first.")

    def _record_to_doc(self, record: Record, id: str | None = None) -> dict[str, Any]:
        """Convert a Record to an Elasticsearch document."""
        # Create a copy of the record to avoid modifying the original
        record_copy = record.copy(deep=True)

        # Update vector tracking if needed
        if self._has_vector_fields(record_copy):
            self._update_vector_tracking(record_copy)

            # Add vector field metadata to copied record metadata
            if "vector_fields" not in record_copy.metadata:
                record_copy.metadata["vector_fields"] = {}

            for field_name in self.vector_fields:
                if field_name in record_copy.fields:
                    field = record_copy.fields[field_name]
                    if hasattr(field, "source_field"):
                        record_copy.metadata["vector_fields"][field_name] = {
                            "type": "vector",
                            "dimensions": self.vector_fields[field_name],
                            "source_field": field.source_field,
                            "model": getattr(field, "model_name", None),
                            "model_version": getattr(field, "model_version", None),
                        }

        doc = self._record_to_document(record_copy)
        if id:
            doc["id"] = id
        elif not doc.get("id"):
            doc["id"] = str(uuid.uuid4())

        return doc

    def _doc_to_record(self, doc: dict[str, Any]) -> Record:
        """Convert an Elasticsearch document to a Record."""
        # Handle both direct documents and search results
        if "_source" in doc:
            source_doc = doc
        else:
            source_doc = {"_source": doc}

        record = self._document_to_record(source_doc)

        # Add score if present
        if "_score" in doc:
            record.metadata["_score"] = doc.get("_score")

        return record

    def create(self, record: Record) -> str:
        """Create a new record."""
        # Use record's ID if it has one, otherwise generate a new one
        id = record.id if record.id else str(uuid.uuid4())
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

    def upsert(self, id_or_record: str | Record, record: Record | None = None) -> str:
        """Update or insert a record.
        
        Can be called as:
        - upsert(id, record) - explicit ID and record
        - upsert(record) - extract ID from record using Record's built-in logic
        """
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
        
        doc = self._record_to_doc(record, id)
        response = self.es_index.index(body=doc, doc_id=id, refresh=self.refresh)

        if response.get("_id"):
            return id
        else:
            raise DatabaseError(f"Failed to upsert record {id}: {response}")

    def create_batch(self, records: list[Record]) -> list[str]:
        """Create multiple records efficiently using the bulk API.
        
        Uses Elasticsearch's bulk API for efficient batch creation.
        
        Args:
            records: List of records to create
            
        Returns:
            List of created record IDs
        """
        if not records:
            return []

        # Build bulk operations
        bulk_operations = []
        ids = []

        for record in records:
            # Generate ID
            record_id = str(uuid.uuid4())
            ids.append(record_id)

            # Create action dict for bulk operation
            doc = self._record_to_doc(record, record_id)
            action = {
                "_op_type": "index",
                "_index": self.es_index.index_name,
                "_id": record_id,
                "_source": doc
            }
            bulk_operations.append(action)

        # Execute bulk create
        from elasticsearch import helpers

        try:
            # Use the bulk helper for creation
            # Note: helpers.BulkIndexError may be raised if raise_on_error=True
            _success_count, errors = helpers.bulk(
                self.es_client,
                bulk_operations,
                refresh=self.refresh,
                raise_on_error=False,
                stats_only=False
            )
            # Process results to return actual IDs
            if errors:
                # Some operations failed - need to check which ones
                error_dict = {}
                for err in errors:
                    # Error dict can have 'index', 'create', 'update', or 'delete' keys
                    for op_type in ['index', 'create']:
                        if op_type in err:
                            error_dict[err[op_type].get('_id')] = err
                            break

                result_ids = []
                for record_id in ids:
                    if record_id not in error_dict:
                        result_ids.append(record_id)
                    # Skip failed records
                return result_ids
            else:
                # All succeeded
                return ids

        except Exception as e:
            # Check if this is a BulkIndexError from the helpers module
            if hasattr(e, 'errors'):
                # Extract which operations succeeded
                failed_ids = {err.get('index', {}).get('_id') for err in e.errors}
                result_ids = []
                for record_id in ids:
                    if record_id not in failed_ids:
                        result_ids.append(record_id)
                    # Skip failed records
                return result_ids
            else:
                # Complete failure - return empty list
                return []

    def read_batch(self, ids: list[str]) -> list[Record | None]:
        """Read multiple records in batch."""
        records = []
        for id in ids:
            record = self.read(id)
            records.append(record)
        return records

    def delete_batch(self, ids: list[str]) -> list[bool]:
        """Delete multiple records efficiently using the bulk API.
        
        Uses Elasticsearch's bulk API for efficient batch deletion.
        
        Args:
            ids: List of record IDs to delete
            
        Returns:
            List of success flags for each deletion
        """
        if not ids:
            return []

        # Build bulk operations
        bulk_operations = []
        for record_id in ids:
            # Create action dict for bulk delete
            action = {
                "_op_type": "delete",
                "_index": self.es_index.index_name,
                "_id": record_id
            }
            bulk_operations.append(action)

        # Execute bulk delete
        from elasticsearch import helpers

        try:
            # Use the bulk helper for deletion
            _success_count, errors = helpers.bulk(
                self.es_client,
                bulk_operations,
                refresh=self.refresh,
                raise_on_error=False,
                stats_only=False
            )

            # Process results to determine which deletes succeeded
            results = []
            if errors:
                error_dict = {}
                for err in errors:
                    if 'delete' in err:
                        error_dict[err['delete'].get('_id')] = err

                for record_id in ids:
                    if record_id in error_dict:
                        # Check if error was "not found" (404) - that's still a successful delete
                        error = error_dict[record_id]
                        status = error.get('delete', {}).get('status')
                        results.append(status == 200 or status == 404)
                    else:
                        results.append(True)
            else:
                # All operations completed (either deleted or not found)
                results = [True] * len(ids)

            return results

        except Exception as e:
            # Check if this is a BulkIndexError from the helpers module
            if hasattr(e, 'errors'):
                # Extract which operations failed
                results = []
                failed_ids = {err.get('delete', {}).get('_id') for err in e.errors}

                for record_id in ids:
                    results.append(record_id not in failed_ids)

                return results
            else:
                # If bulk operation completely fails, mark all as failed
                return [False] * len(ids)

    def update_batch(self, updates: list[tuple[str, Record]]) -> list[bool]:
        """Update multiple records efficiently using the bulk API.
        
        Uses Elasticsearch's bulk API for efficient batch updates.
        
        Args:
            updates: List of (id, record) tuples to update
            
        Returns:
            List of success flags for each update
        """
        if not updates:
            return []

        # Build bulk operations
        bulk_operations = []
        for record_id, record in updates:
            # Create action dict for bulk update
            doc = self._record_to_doc(record, record_id)
            action = {
                "_op_type": "update",
                "_index": self.es_index.index_name,
                "_id": record_id,
                "doc": doc,
                "doc_as_upsert": False  # Don't create if doesn't exist
            }
            bulk_operations.append(action)

        # Execute bulk update
        from elasticsearch import helpers

        try:
            # Use the bulk helper for the update
            _success_count, errors = helpers.bulk(
                self.es_client,
                bulk_operations,
                refresh=self.refresh,
                raise_on_error=False,
                stats_only=False
            )

            # Process results to determine which updates succeeded
            results = []
            error_dict = {}
            if errors:
                for err in errors:
                    if 'update' in err:
                        error_dict[err['update']['_id']] = err

            for record_id, _ in updates:
                # Check if this ID had an error
                if record_id in error_dict:
                    error = error_dict[record_id]
                    # If error is 404 (not found), mark as failed
                    status = error.get('update', {}).get('status')
                    results.append(status == 200)  # Only 200 is success for update
                else:
                    results.append(True)

            return results

        except Exception as e:
            # Check if this is a BulkIndexError from the helpers module
            if hasattr(e, 'errors'):
                # Extract which operations failed
                results = []
                failed_ids = {err['update']['_id'] for err in e.errors}

                for record_id, _ in updates:
                    results.append(record_id not in failed_ids)

                return results
            else:
                # If bulk operation completely fails, mark all as failed
                return [False] * len(updates)

    def _build_complex_es_query(self, condition: Any) -> dict[str, Any]:
        """Build Elasticsearch query from complex boolean logic conditions.
        
        Args:
            condition: The Condition object (LogicCondition or FilterCondition)
            
        Returns:
            Elasticsearch query dict
        """
        from ..query_logic import FilterCondition, LogicCondition, LogicOperator

        # Handle FilterCondition (leaf node)
        if isinstance(condition, FilterCondition):
            return self._build_filter_es_query(condition.filter)

        # Handle LogicCondition (branch node)
        elif isinstance(condition, LogicCondition):
            if condition.operator == LogicOperator.AND:
                # Build AND query with must clauses
                must_clauses = []
                for sub_condition in condition.conditions:
                    sub_query = self._build_complex_es_query(sub_condition)
                    if sub_query:
                        must_clauses.append(sub_query)

                if not must_clauses:
                    return {"match_all": {}}
                elif len(must_clauses) == 1:
                    return must_clauses[0]
                else:
                    return {"bool": {"must": must_clauses}}

            elif condition.operator == LogicOperator.OR:
                # Build OR query with should clauses
                should_clauses = []
                for sub_condition in condition.conditions:
                    sub_query = self._build_complex_es_query(sub_condition)
                    if sub_query:
                        should_clauses.append(sub_query)

                if not should_clauses:
                    return {"match_all": {}}
                elif len(should_clauses) == 1:
                    return should_clauses[0]
                else:
                    return {"bool": {"should": should_clauses, "minimum_should_match": 1}}

            elif condition.operator == LogicOperator.NOT:
                # Build NOT query with must_not
                if condition.conditions:
                    sub_query = self._build_complex_es_query(condition.conditions[0])
                    if sub_query:
                        return {"bool": {"must_not": sub_query}}

                return {"match_all": {}}

        return {"match_all": {}}

    def _build_filter_es_query(self, filter_obj: Any) -> dict[str, Any]:
        """Build Elasticsearch query for a single filter.
        
        Args:
            filter_obj: The Filter object
            
        Returns:
            Elasticsearch query dict for the filter
        """
        # Special handling for 'id' field - use _id in Elasticsearch
        if filter_obj.field == 'id':
            field_path = "_id"
            # _id field doesn't need .keyword suffix
        else:
            field_path = f"data.{filter_obj.field}"

            # For string fields in exact match queries, use .keyword suffix
            if filter_obj.operator in [Operator.EQ, Operator.NEQ, Operator.IN, Operator.NOT_IN]:
                if isinstance(filter_obj.value, str) or (
                    isinstance(filter_obj.value, list) and
                    filter_obj.value and
                    isinstance(filter_obj.value[0], str)
                ):
                    field_path = f"{field_path}.keyword"
            elif filter_obj.operator in [Operator.LIKE, Operator.NOT_LIKE]:
                # Wildcard needs .keyword for proper matching
                if isinstance(filter_obj.value, str):
                    field_path = f"{field_path}.keyword"

        if filter_obj.operator == Operator.EQ:
            value = str(filter_obj.value).lower() if isinstance(filter_obj.value, bool) else filter_obj.value
            return {"term": {field_path: value}}
        elif filter_obj.operator == Operator.NEQ:
            value = str(filter_obj.value).lower() if isinstance(filter_obj.value, bool) else filter_obj.value
            return {"bool": {"must_not": {"term": {field_path: value}}}}
        elif filter_obj.operator == Operator.GT:
            return {"range": {field_path: {"gt": filter_obj.value}}}
        elif filter_obj.operator == Operator.GTE:
            return {"range": {field_path: {"gte": filter_obj.value}}}
        elif filter_obj.operator == Operator.LT:
            return {"range": {field_path: {"lt": filter_obj.value}}}
        elif filter_obj.operator == Operator.LTE:
            return {"range": {field_path: {"lte": filter_obj.value}}}
        elif filter_obj.operator == Operator.LIKE:
            pattern = filter_obj.value.replace("%", "*").replace("_", "?")
            return {"wildcard": {field_path: pattern}}
        elif filter_obj.operator == Operator.NOT_LIKE:
            pattern = filter_obj.value.replace("%", "*").replace("_", "?")
            return {"bool": {"must_not": {"wildcard": {field_path: pattern}}}}
        elif filter_obj.operator == Operator.IN:
            # Special handling for _id field - use ids query instead of terms
            if filter_obj.field == 'id':
                return {"ids": {"values": filter_obj.value}}
            else:
                return {"terms": {field_path: filter_obj.value}}
        elif filter_obj.operator == Operator.NOT_IN:
            # Special handling for _id field
            if filter_obj.field == 'id':
                return {"bool": {"must_not": {"ids": {"values": filter_obj.value}}}}
            else:
                return {"bool": {"must_not": {"terms": {field_path: filter_obj.value}}}}
        elif filter_obj.operator == Operator.EXISTS:
            return {"exists": {"field": field_path}}
        elif filter_obj.operator == Operator.NOT_EXISTS:
            return {"bool": {"must_not": {"exists": {"field": field_path}}}}
        elif filter_obj.operator == Operator.REGEX:
            return {"regexp": {field_path: filter_obj.value}}
        elif filter_obj.operator == Operator.BETWEEN:
            if isinstance(filter_obj.value, (list, tuple)) and len(filter_obj.value) == 2:
                lower, upper = filter_obj.value
                return {"range": {field_path: {"gte": lower, "lte": upper}}}
        elif filter_obj.operator == Operator.NOT_BETWEEN:
            if isinstance(filter_obj.value, (list, tuple)) and len(filter_obj.value) == 2:
                lower, upper = filter_obj.value
                return {"bool": {"must_not": {"range": {field_path: {"gte": lower, "lte":
upper}}}}}

        return {"match_all": {}}


    def search(self, query: Query | ComplexQuery) -> list[Record]:
        """Search for records matching a query."""
        # Handle ComplexQuery with native Elasticsearch bool queries
        if isinstance(query, ComplexQuery):
            if query.condition:
                es_query = self._build_complex_es_query(query.condition)
            else:
                es_query = {"match_all": {}}
        else:
            # Build Elasticsearch query from simple Query object
            es_query = {"bool": {"must": []}}

            # Apply filters
            for filter_obj in query.filters:
                # Special handling for 'id' field - use _id in Elasticsearch
                if filter_obj.field == 'id':
                    field_path = "_id"
                    # _id field doesn't need .keyword suffix
                else:
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
                    elif filter_obj.operator in [Operator.LIKE, Operator.NOT_LIKE]:
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
                elif filter_obj.operator == Operator.NOT_LIKE:
                    pattern = filter_obj.value.replace("%", "*").replace("_", "?")
                    es_query["bool"]["must"].append({"bool": {"must_not": {"wildcard": {field_path: pattern}}}})
                elif filter_obj.operator == Operator.IN:
                    # Special handling for _id field - use ids query instead of terms
                    if filter_obj.field == 'id':
                        es_query["bool"]["must"].append({"ids": {"values": filter_obj.value}})
                    else:
                        es_query["bool"]["must"].append({"terms": {field_path: filter_obj.value}})
                elif filter_obj.operator == Operator.NOT_IN:
                    # Special handling for _id field
                    if filter_obj.field == 'id':
                        es_query["bool"]["must"].append({"bool": {"must_not": {"ids": {"values": filter_obj.value}}}})
                    else:
                        es_query["bool"]["must"].append({"bool": {"must_not": {"terms": {field_path: filter_obj.value}}}})
                elif filter_obj.operator == Operator.EXISTS:
                    es_query["bool"]["must"].append({"exists": {"field": field_path}})
                elif filter_obj.operator == Operator.NOT_EXISTS:
                    es_query["bool"]["must"].append({"bool": {"must_not": {"exists": {"field": field_path}}}})
                elif filter_obj.operator == Operator.REGEX:
                    es_query["bool"]["must"].append({"regexp": {field_path: filter_obj.value}})
                elif filter_obj.operator == Operator.BETWEEN:
                    # Use Elasticsearch's native range query for efficient BETWEEN
                    if isinstance(filter_obj.value, (list, tuple)) and len(filter_obj.value) == 2:
                        lower, upper = filter_obj.value
                        es_query["bool"]["must"].append({
                            "range": {
                                field_path: {
                                    "gte": lower,
                                    "lte": upper
                                }
                            }
                        })
                elif filter_obj.operator == Operator.NOT_BETWEEN:
                    # NOT BETWEEN using bool must_not with range
                    if isinstance(filter_obj.value, (list, tuple)) and len(filter_obj.value) == 2:
                        lower, upper = filter_obj.value
                        es_query["bool"]["must"].append({
                            "bool": {
                                "must_not": {
                                    "range": {
                                        field_path: {
                                            "gte": lower,
                                            "lte": upper
                                        }
                                    }
                                }
                            }
                        })

        # If no filters, match all
        if not es_query["bool"]["must"]:
            es_query = {"match_all": {}}

        # Build sort
        sort = []
        if query.sort_specs:
            for sort_spec in query.sort_specs:
                # Special handling for 'id' field - sort by the id field in source data
                # We can't sort by _id directly as it requires fielddata which is disabled by default
                # The id field is already of type keyword, so no .keyword suffix needed
                if sort_spec.field == 'id':
                    field_path = "id"
                else:
                    field_path = f"data.{sort_spec.field}"
                    # Don't add .keyword if user already specified it or for common numeric fields
                    # This is a heuristic - ideally we'd check the mapping
                    numeric_fields = ['age', 'salary', 'balance', 'count', 'score', 'amount', 'price', 'index', 'number', 'total', 'quantity']
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

        # Check if the response is valid (has the expected structure)
        # An empty result set is still a valid response
        if not hasattr(response, 'json') or response.json is None:
            raise DatabaseError(f"Invalid search response: {response}")

        # Check for actual errors in the response
        if 'error' in response.json:
            raise DatabaseError(f"Failed to search records: {response.json['error']}")

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
        self._check_connection()
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
            # Special handling for 'id' field - use _id in Elasticsearch
            if filter_obj.field == 'id':
                field_path = "_id"
                # _id field doesn't need .keyword suffix
            else:
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
                elif filter_obj.operator in [Operator.LIKE, Operator.NOT_LIKE]:
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
            elif filter_obj.operator == Operator.NOT_LIKE:
                pattern = filter_obj.value.replace("%", "*").replace("_", "?")
                es_query["bool"]["must"].append({"bool": {"must_not": {"wildcard": {field_path: pattern}}}})
            elif filter_obj.operator == Operator.IN:
                # Special handling for _id field - use ids query instead of terms
                if filter_obj.field == 'id':
                    es_query["bool"]["must"].append({"ids": {"values": filter_obj.value}})
                else:
                    es_query["bool"]["must"].append({"terms": {field_path: filter_obj.value}})
            elif filter_obj.operator == Operator.NOT_IN:
                # Special handling for _id field
                if filter_obj.field == 'id':
                    es_query["bool"]["must"].append({"bool": {"must_not": {"ids": {"values": filter_obj.value}}}})
                else:
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
        self._check_connection()
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

    def stream_read(
        self,
        query: Query | None = None,
        config: StreamConfig | None = None
    ) -> Iterator[Record]:
        """Stream records from Elasticsearch."""
        config = config or StreamConfig()

        # Use search to get all matching records
        if query:
            records = self.search(query)
        else:
            records = self.search(Query())

        # Yield records in batches for consistency
        for i in range(0, len(records), config.batch_size):
            batch = records[i:i + config.batch_size]
            for record in batch:
                yield record

    def stream_write(
        self,
        records: Iterator[Record],
        config: StreamConfig | None = None
    ) -> StreamResult:
        """Stream records into Elasticsearch."""
        # Use the default implementation from mixin
        return self._default_stream_write(records, config)

    def vector_search(
        self,
        query_vector: np.ndarray | list[float],
        field_name: str = "embedding",
        k: int = 10,
        metric: DistanceMetric = DistanceMetric.COSINE,
        filter: Query | None = None,
        include_source: bool = True,
        score_threshold: float | None = None,
    ) -> list[VectorSearchResult]:
        """Search for similar vectors using Elasticsearch KNN.
        
        Note: This is a synchronous wrapper around the async implementation.
        For production use, consider using the async version for better performance.
        
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
            field_name=field_name,
            k=k,
            filter_query=filter_query,
        )

        # Execute search using the es_client
        try:
            response = self.es_client.search(
                index=self.index_name,
                **query,
                size=k,
                source=include_source,
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
                if not record.has_storage_id():
                    record.storage_id = hit["_id"]

            # Skip if no record (shouldn't happen if include_source is True)
            if record is None:
                continue

            results.append(VectorSearchResult(
                record=record,
                score=score,
                vector_field=field_name,
                metadata={
                    "index": self.index_name,
                    "metric": metric.value,
                    "doc_id": hit["_id"],
                },
            ))

        return results

    def create_vector_index(
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

        # Update index mapping using the es_client
        try:
            self.es_client.indices.put_mapping(
                index=self.index_name,
                properties={
                    f"data.{vector_field}": mapping
                }
            )

            # Track the vector field
            self.vector_fields[vector_field] = dimensions
            self._vector_enabled = True

            logger.info(f"Created vector mapping for field '{vector_field}' with {dimensions} dimensions")
            return True

        except Exception as e:
            self._handle_elasticsearch_error(e, "create vector index")
            return False


# Import the native async implementation
