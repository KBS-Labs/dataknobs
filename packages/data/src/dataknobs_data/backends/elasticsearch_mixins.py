"""Shared mixins for Elasticsearch backend implementations."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from ..fields import Field, FieldType, VectorField
from ..records import Record

if TYPE_CHECKING:
    from ..query import Query

logger = logging.getLogger(__name__)


class ElasticsearchBaseConfig:
    """Mixin for parsing Elasticsearch configuration."""

    def _parse_elasticsearch_config(self, config: dict[str, Any]) -> tuple[str, int, str, dict]:
        """Parse Elasticsearch configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Tuple of (host, port, index_name, extra_config)
        """
        host = config.get("host", "localhost")
        port = config.get("port", 9200)
        index = config.get("index", "records")

        # Extract other config options
        extra_config = {
            "refresh": config.get("refresh", True),
            "settings": config.get("settings", {
                "number_of_shards": 1,
                "number_of_replicas": 0,
            }),
            "mappings": config.get("mappings"),
        }

        return host, port, index, extra_config


class ElasticsearchIndexManager:
    """Mixin for Elasticsearch index management."""

    @staticmethod
    def get_index_mappings(vector_fields: dict[str, int] | None = None) -> dict:
        """Get index mappings with vector field support.
        
        Args:
            vector_fields: Dict mapping vector field names to dimensions
            
        Returns:
            Elasticsearch mappings dictionary
        """
        mappings = {
            "properties": {
                "id": {"type": "keyword"},
                "data": {
                    "type": "object",
                    "properties": {}
                },
                "metadata": {"type": "object", "enabled": True},
                "created_at": {"type": "date"},
                "updated_at": {"type": "date"},
            }
        }

        # Add vector field mappings if specified
        if vector_fields:
            for field_name, dimensions in vector_fields.items():
                # Use dense_vector type for vector fields nested under data
                data_props = mappings["properties"]["data"]["properties"]  # type: ignore[index]
                data_props[field_name] = {
                    "type": "dense_vector",
                    "dims": dimensions,
                    "index": True,
                    "similarity": "cosine"  # Default similarity
                }

        return mappings

    @staticmethod
    def get_knn_index_settings() -> dict:
        """Get index settings optimized for KNN search.
        
        Returns:
            Index settings dictionary
        """
        return {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            # Note: "knn" setting is not needed for standard Elasticsearch
            # KNN is enabled by having dense_vector fields with index=true
        }


class ElasticsearchVectorSupport:
    """Mixin for vector field detection and tracking."""

    def __init__(self):
        """Initialize vector support tracking."""
        self.vector_fields: dict[str, int] = {}  # field_name -> dimensions
        self.vector_enabled = False

    def _detect_vector_fields(self, record: Record) -> dict[str, int]:
        """Detect vector fields in a record.
        
        Args:
            record: Record to examine
            
        Returns:
            Dict mapping field names to dimensions
        """
        vector_fields = {}

        for field_name, field_obj in record.fields.items():
            if field_obj.type in (FieldType.VECTOR, FieldType.SPARSE_VECTOR):
                if isinstance(field_obj, VectorField) and field_obj.value is not None:
                    # Get dimensions from the vector value
                    if isinstance(field_obj.value, (list, np.ndarray)):
                        dims = len(field_obj.value) if isinstance(field_obj.value, list) else field_obj.value.shape[0]
                        vector_fields[field_name] = dims
                        logger.debug(f"Detected vector field '{field_name}' with {dims} dimensions")

        return vector_fields

    def _has_vector_fields(self, record: Record) -> bool:
        """Check if a record has vector fields.
        
        Args:
            record: Record to check
            
        Returns:
            True if record has vector fields
        """
        return len(self._detect_vector_fields(record)) > 0

    def _update_vector_tracking(self, record: Record) -> None:
        """Update tracking of vector fields from a record.
        
        Args:
            record: Record to examine
        """
        detected = self._detect_vector_fields(record)
        for field_name, dims in detected.items():
            if field_name not in self.vector_fields:
                self.vector_fields[field_name] = dims
                logger.info(f"Tracking new vector field '{field_name}' with {dims} dimensions")


class ElasticsearchErrorHandler:
    """Mixin for consistent error handling."""

    @staticmethod
    def _handle_elasticsearch_error(error: Exception, operation: str) -> None:
        """Handle Elasticsearch errors consistently.
        
        Args:
            error: The exception that occurred
            operation: Description of the operation that failed
        """
        from elasticsearch import (
            ConnectionError,
            NotFoundError,
            RequestError,
            TransportError,
        )

        if isinstance(error, ConnectionError):
            logger.error(f"Connection error during {operation}: {error}")
            raise RuntimeError(f"Failed to connect to Elasticsearch: {error}") from error
        elif isinstance(error, NotFoundError):
            logger.warning(f"Resource not found during {operation}: {error}")
            raise ValueError(f"Resource not found: {error}") from error
        elif isinstance(error, RequestError):
            logger.error(f"Bad request during {operation}: {error}")
            raise ValueError(f"Invalid request: {error}") from error
        elif isinstance(error, TransportError):
            logger.error(f"Transport error during {operation}: {error}")
            raise RuntimeError(f"Elasticsearch transport error: {error}") from error
        else:
            logger.error(f"Unexpected error during {operation}: {error}")
            raise error


class ElasticsearchRecordSerializer:
    """Mixin for record serialization with vector field handling."""

    @staticmethod
    def _record_to_document(record: Record) -> dict[str, Any]:
        """Convert a record to an Elasticsearch document.
        
        Args:
            record: Record to convert
            
        Returns:
            Document dictionary for Elasticsearch
        """
        # Serialize the record data
        data_dict = {}

        for field_name, field_obj in record.fields.items():
            if isinstance(field_obj, VectorField) and field_obj.value is not None:
                # Convert numpy arrays to lists for JSON serialization
                if isinstance(field_obj.value, np.ndarray):
                    data_dict[field_name] = field_obj.value.tolist()
                else:
                    data_dict[field_name] = field_obj.value
            else:
                data_dict[field_name] = field_obj.value

        # Create the document
        doc = {
            "data": data_dict,
            "metadata": record.metadata,
        }

        # Add timestamps if they exist as attributes
        if hasattr(record, "created_at") and record.created_at:
            doc["created_at"] = record.created_at.isoformat()
        if hasattr(record, "updated_at") and record.updated_at:
            doc["updated_at"] = record.updated_at.isoformat()

        # Add ID if present
        if record.id:
            doc["id"] = record.id

        return doc

    @staticmethod
    def _document_to_record(doc: dict[str, Any], doc_id: str | None = None) -> Record:
        """Convert an Elasticsearch document to a record.
        
        Args:
            doc: Document from Elasticsearch
            doc_id: Document ID from Elasticsearch
            
        Returns:
            Record instance
        """
        # Get the source data
        source = doc.get("_source", doc)

        # Extract data and metadata
        data = source.get("data", {})
        metadata = source.get("metadata", {})

        # Create fields
        fields = {}
        for field_name, value in data.items():
            # Check if this is a vector field based on metadata
            field_meta = metadata.get("vector_fields", {}).get(field_name, {})

            if field_meta.get("type") == "vector" or (
                isinstance(value, list) and len(value) > 0 and
                all(isinstance(v, (int, float)) for v in value)
            ):
                # This looks like a vector field
                vector_value = np.array(value, dtype=np.float32) if value else np.array([], dtype=np.float32)
                fields[field_name] = VectorField(
                    name=field_name,
                    value=vector_value,
                    source_field=field_meta.get("source_field"),
                    model_name=field_meta.get("model"),
                    model_version=field_meta.get("model_version"),
                )
            else:
                # Regular field - infer type from value
                field_type = FieldType.STRING  # default
                if isinstance(value, bool):
                    field_type = FieldType.BOOLEAN
                elif isinstance(value, int):
                    field_type = FieldType.INTEGER
                elif isinstance(value, float):
                    field_type = FieldType.FLOAT
                elif isinstance(value, dict) or (isinstance(value, (list, tuple)) and not all(isinstance(v, (int, float)) for v in value)):
                    field_type = FieldType.JSON

                fields[field_name] = Field(
                    name=field_name,
                    value=value,
                    type=field_type,
                )

        # Create the record - pass fields as OrderedDict since they're Field objects
        from collections import OrderedDict
        record = Record(data=OrderedDict(fields), metadata=metadata)

        # Set ID from document
        if doc_id:
            record.id = doc_id
        elif "_id" in doc:
            record.id = doc["_id"]
        elif "id" in source:
            record.id = source["id"]

        # Set timestamps if available (as attributes, not fields)
        if source.get("created_at"):
            from datetime import datetime
            record.created_at = datetime.fromisoformat(source["created_at"])

        if source.get("updated_at"):
            from datetime import datetime
            record.updated_at = datetime.fromisoformat(source["updated_at"])

        return record


class ElasticsearchQueryBuilder:
    """Mixin for building Elasticsearch queries."""

    @staticmethod
    def _build_filter_query(filter_query: Query | None) -> dict[str, Any] | None:
        """Build Elasticsearch filter query from Query object.
        
        Args:
            filter_query: Query object to convert
            
        Returns:
            Elasticsearch query dict or None
        """
        if not filter_query:
            return None

        # TODO: Implement full query translation
        # For now, just support simple field equality
        from ..query import Operator

        must_clauses = []

        if filter_query.filters:
            for filter_item in filter_query.filters:
                field_path = f"data.{filter_item.field}"

                if filter_item.operator == Operator.EQ:
                    # Use match query for text fields to handle analyzed text
                    must_clauses.append({
                        "match": {field_path: filter_item.value}
                    })
                elif filter_item.operator == Operator.IN:
                    must_clauses.append({
                        "terms": {field_path: filter_item.value}
                    })
                elif filter_item.operator == Operator.GT:
                    must_clauses.append({
                        "range": {field_path: {"gt": filter_item.value}}
                    })
                elif filter_item.operator == Operator.LT:
                    must_clauses.append({
                        "range": {field_path: {"lt": filter_item.value}}
                    })

        if must_clauses:
            return {"bool": {"must": must_clauses}}

        return None
