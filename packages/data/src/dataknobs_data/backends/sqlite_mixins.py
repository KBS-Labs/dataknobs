"""SQLite-specific mixins for vector support and other functionality."""

import json
import logging
from typing import Any, TYPE_CHECKING

import numpy as np

from ..records import Record
from ..fields import Field, FieldType, VectorField
from ..vector.types import DistanceMetric

if TYPE_CHECKING:
    from ..vector.types import VectorSearchResult

logger = logging.getLogger(__name__)


class SQLiteVectorSupport:
    """Vector support for SQLite using JSON storage and Python-based similarity."""
    
    def __init__(self):
        """Initialize vector support tracking."""
        self._vector_dimensions = {}
        self._vector_fields = {}
    
    def _has_vector_fields(self, record: Record) -> bool:
        """Check if record has vector fields.
        
        Args:
            record: Record to check
            
        Returns:
            True if record has vector fields
        """
        return any(isinstance(field, VectorField) 
                   for field in record.fields.values())
    
    def _extract_vector_dimensions(self, record: Record) -> dict[str, int]:
        """Extract dimensions from vector fields in a record.
        
        Args:
            record: Record containing potential vector fields
            
        Returns:
            Dictionary mapping field names to dimensions
        """
        dimensions = {}
        for name, field in record.fields.items():
            if isinstance(field, VectorField):
                if field.value is not None:
                    if isinstance(field.value, np.ndarray):
                        dimensions[name] = field.value.shape[0]
                    elif isinstance(field.value, list):
                        dimensions[name] = len(field.value)
                elif field.dimensions:
                    dimensions[name] = field.dimensions
        return dimensions
    
    def _update_vector_dimensions(self, record: Record) -> None:
        """Update tracked vector dimensions from a record.
        
        Args:
            record: Record containing vector fields
        """
        dimensions = self._extract_vector_dimensions(record)
        self._vector_dimensions.update(dimensions)
        
        # Track which fields are vectors
        for name, field in record.fields.items():
            if isinstance(field, VectorField):
                self._vector_fields[name] = {
                    "dimensions": dimensions.get(name),
                    "source_field": field.source_field,
                    "model_name": field.model_name,
                    "model_version": field.model_version,
                }
    
    def _serialize_vector(self, vector: np.ndarray | list) -> str:
        """Serialize a vector to JSON string for storage.
        
        Args:
            vector: Vector as numpy array or list
            
        Returns:
            JSON string representation
        """
        if isinstance(vector, np.ndarray):
            vector = vector.tolist()
        return json.dumps(vector)
    
    def _deserialize_vector(self, vector_str: str) -> np.ndarray:
        """Deserialize a vector from JSON string.
        
        Args:
            vector_str: JSON string representation
            
        Returns:
            Numpy array
        """
        if not vector_str:
            return None
        try:
            vector_list = json.loads(vector_str)
            return np.array(vector_list, dtype=np.float32)
        except (json.JSONDecodeError, TypeError, ValueError):
            return None
    
    def _compute_similarity(
        self, 
        vec1: np.ndarray, 
        vec2: np.ndarray, 
        metric: DistanceMetric = DistanceMetric.COSINE
    ) -> float:
        """Compute similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            metric: Distance metric to use
            
        Returns:
            Similarity score (higher is more similar)
        """
        if vec1 is None or vec2 is None:
            return 0.0
        
        # Ensure vectors are numpy arrays
        if not isinstance(vec1, np.ndarray):
            vec1 = np.array(vec1, dtype=np.float32)
        if not isinstance(vec2, np.ndarray):
            vec2 = np.array(vec2, dtype=np.float32)
        
        # Check dimensions match
        if vec1.shape != vec2.shape:
            raise ValueError(f"Vector dimensions don't match: {vec1.shape} vs {vec2.shape}")
        
        if metric == DistanceMetric.COSINE:
            # Cosine similarity
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return float(np.dot(vec1, vec2) / (norm1 * norm2))
        
        elif metric == DistanceMetric.EUCLIDEAN:
            # Convert Euclidean distance to similarity (inverse)
            distance = np.linalg.norm(vec1 - vec2)
            return 1.0 / (1.0 + distance)
        
        elif metric == DistanceMetric.DOT_PRODUCT:
            # Dot product similarity
            return float(np.dot(vec1, vec2))
        
        else:
            raise ValueError(f"Unsupported metric: {metric}")


class SQLiteRecordSerializer:
    """Handle record serialization for SQLite with vector support."""
    
    def _record_to_row(self, record: Record) -> dict[str, Any]:
        """Convert a record to a SQLite row with JSON-serialized data.
        
        SQLite uses a generic table structure with 'data' and 'metadata' JSON columns.
        
        Args:
            record: Record to convert
            
        Returns:
            Dictionary representing a database row
        """
        # Build the data dictionary
        data_dict = {}
        
        # Handle fields
        for field_name, field_obj in record.fields.items():
            if isinstance(field_obj, VectorField):
                # Serialize vector as list in JSON
                if field_obj.value is not None:
                    if isinstance(field_obj.value, np.ndarray):
                        data_dict[field_name] = field_obj.value.tolist()
                    else:
                        data_dict[field_name] = field_obj.value
                # Skip None vectors - don't include in data
            else:
                # Regular field
                data_dict[field_name] = field_obj.value
        
        # Build metadata including vector field info
        metadata = record.metadata.copy() if record.metadata else {}
        
        # Track vector field metadata
        vector_fields_meta = {}
        for name, field in record.fields.items():
            if isinstance(field, VectorField):
                vector_fields_meta[name] = {
                    "type": "vector",
                    "dimensions": field.dimensions,
                    "source_field": field.source_field,
                    "model": field.model_name,
                    "model_version": field.model_version,
                }
        
        if vector_fields_meta:
            metadata["vector_fields"] = vector_fields_meta
        
        # Build the row for SQLite's generic table structure
        row = {
            "data": json.dumps(data_dict),
            "metadata": json.dumps(metadata) if metadata else None
        }
        
        # Add ID if present
        if record.id:
            row["id"] = record.id
        else:
            import uuid
            row["id"] = str(uuid.uuid4())
        
        return row
    
    def _row_to_record(self, row: dict[str, Any]) -> Record:
        """Convert a SQLite row to a record with vector deserialization.
        
        Args:
            row: Database row as dictionary
            
        Returns:
            Record instance
        """
        # Parse metadata
        metadata = {}
        if row.get("metadata"):
            try:
                metadata = json.loads(row["metadata"])
            except json.JSONDecodeError:
                metadata = {}
        
        # Get vector field info from metadata
        vector_fields_meta = metadata.get("vector_fields", {})
        
        # Create fields
        from collections import OrderedDict
        fields = OrderedDict()
        
        for col_name, value in row.items():
            if col_name in ["id", "metadata", "created_at", "updated_at"]:
                continue  # Skip system columns
            
            # Check if this is a vector field
            if col_name in vector_fields_meta or col_name in self._vector_fields:
                field_meta = vector_fields_meta.get(col_name, self._vector_fields.get(col_name, {}))
                
                # Deserialize vector
                vector_value = None
                if value:
                    try:
                        vector_list = json.loads(value)
                        vector_value = np.array(vector_list, dtype=np.float32)
                    except (json.JSONDecodeError, TypeError, ValueError):
                        vector_value = None
                
                fields[col_name] = VectorField(
                    name=col_name,
                    value=vector_value,
                    dimensions=field_meta.get("dimensions"),
                    source_field=field_meta.get("source_field"),
                    model_name=field_meta.get("model"),
                    model_version=field_meta.get("model_version"),
                )
            else:
                # Regular field - infer type
                field_type = FieldType.STRING  # default
                if isinstance(value, bool):
                    field_type = FieldType.BOOLEAN
                elif isinstance(value, int):
                    field_type = FieldType.INTEGER
                elif isinstance(value, float):
                    field_type = FieldType.FLOAT
                elif isinstance(value, (dict, list)):
                    field_type = FieldType.JSON
                
                fields[col_name] = Field(
                    name=col_name,
                    value=value,
                    type=field_type,
                )
        
        # Create the record
        record = Record(data=fields, metadata=metadata)
        
        # Set ID
        if "id" in row:
            record.id = row["id"]
        
        return record