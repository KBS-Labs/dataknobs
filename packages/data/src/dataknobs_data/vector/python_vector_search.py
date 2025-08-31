"""Python-based vector search implementation for databases without native vector support."""

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..records import Record
    from .types import VectorSearchResult

logger = logging.getLogger(__name__)


class PythonVectorSearchMixin:
    """Mixin providing Python-based vector similarity search.
    
    This mixin can be used by database backends that don't have native vector
    search capabilities (like SQLite) to provide vector search functionality
    using Python/NumPy calculations.
    
    The backend must provide:
    - A way to fetch all records (with optional filtering)
    - A method to extract vector data from records
    - The _compute_similarity method (or inherit from a mixin that provides it)
    """

    async def python_vector_search_async(
        self,
        query_vector,
        vector_field: str = "embedding",
        k: int = 10,
        filter=None,
        metric=None,
        fetch_all_method: str = "search",
        fetch_filtered_method: str = "search",
        **kwargs
    ) -> list["VectorSearchResult"]:
        """Perform async vector search using Python calculations.
        
        Args:
            query_vector: Query vector
            vector_field: Name of the vector field to search
            k: Number of results to return
            filter: Optional filter conditions
            metric: Distance metric to use
            fetch_all_method: Name of method to fetch all records
            fetch_filtered_method: Name of method to fetch filtered records
            **kwargs: Additional arguments
            
        Returns:
            List of VectorSearchResult objects
        """
        import numpy as np

        from ..query import Query
        from ..records import Record
        from .types import DistanceMetric, VectorSearchResult

        # Get metric from parameter or instance default
        if metric is None:
            metric = getattr(self, 'vector_metric', DistanceMetric.COSINE)
        if isinstance(metric, str):
            metric = DistanceMetric(metric)

        # Ensure query vector is numpy array
        if not isinstance(query_vector, np.ndarray):
            query_vector = np.array(query_vector, dtype=np.float32)

        # Fetch records using search method with proper Query
        if filter:
            records = await getattr(self, fetch_filtered_method)(filter)
        else:
            records = await getattr(self, fetch_all_method)(Query())

        # Calculate similarities
        results = []
        for record_data in records:
            # Handle different record formats and keep original for later use
            original_record = record_data
            if isinstance(record_data, dict):
                data = self._extract_record_data(record_data)
            elif isinstance(record_data, Record):
                # If we already have a Record object, use it directly
                data = record_data.data
            else:
                data = record_data

            # Check if the record has the vector field
            if isinstance(data, dict) and vector_field in data and data[vector_field] is not None:
                stored_vector = data[vector_field]

                # Handle VectorField dict format (from to_dict())
                if isinstance(stored_vector, dict) and 'value' in stored_vector:
                    stored_vector = stored_vector['value']

                # Convert to numpy array if needed
                if not isinstance(stored_vector, np.ndarray):
                    stored_vector = np.array(stored_vector, dtype=np.float32)

                # Calculate similarity
                score = self._compute_similarity(query_vector, stored_vector, metric)

                # Create Record object for result
                if isinstance(original_record, Record):
                    record = original_record
                else:
                    record = self._create_record_from_data(original_record, data)

                # Create result
                result = VectorSearchResult(
                    record=record,
                    score=float(score),
                    vector_field=vector_field
                )
                results.append(result)

        # Sort by score (descending) and return top k
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:k]

    def python_vector_search_sync(
        self,
        query_vector,
        vector_field: str = "embedding",
        k: int = 10,
        filter=None,
        metric=None,
        fetch_all_method: str = "search",
        fetch_filtered_method: str = "search",
        **kwargs
    ) -> list["VectorSearchResult"]:
        """Perform sync vector search using Python calculations.
        
        Args:
            query_vector: Query vector
            vector_field: Name of the vector field to search
            k: Number of results to return
            filter: Optional filter conditions
            metric: Distance metric to use
            fetch_all_method: Name of method to fetch all records
            fetch_filtered_method: Name of method to fetch filtered records
            **kwargs: Additional arguments
            
        Returns:
            List of VectorSearchResult objects
        """
        import numpy as np

        from ..query import Query
        from ..records import Record
        from .types import DistanceMetric, VectorSearchResult

        # Get metric from parameter or instance default
        if metric is None:
            metric = getattr(self, 'vector_metric', DistanceMetric.COSINE)
        if isinstance(metric, str):
            metric = DistanceMetric(metric)

        # Ensure query vector is numpy array
        if not isinstance(query_vector, np.ndarray):
            query_vector = np.array(query_vector, dtype=np.float32)

        # Fetch records using search method with proper Query
        if filter:
            records = getattr(self, fetch_filtered_method)(filter)
        else:
            records = getattr(self, fetch_all_method)(Query())

        # Calculate similarities
        results = []
        for record_data in records:
            # Handle different record formats and keep original for later use
            original_record = record_data
            if isinstance(record_data, dict):
                data = self._extract_record_data(record_data)
            elif isinstance(record_data, Record):
                # If we already have a Record object, use it directly
                data = record_data.data
            else:
                data = record_data

            # Check if the record has the vector field
            if isinstance(data, dict) and vector_field in data and data[vector_field] is not None:
                stored_vector = data[vector_field]

                # Handle VectorField dict format (from to_dict())
                if isinstance(stored_vector, dict) and 'value' in stored_vector:
                    stored_vector = stored_vector['value']

                # Convert to numpy array if needed
                if not isinstance(stored_vector, np.ndarray):
                    stored_vector = np.array(stored_vector, dtype=np.float32)

                # Calculate similarity
                score = self._compute_similarity(query_vector, stored_vector, metric)

                # Create Record object for result
                if isinstance(original_record, Record):
                    record = original_record
                else:
                    record = self._create_record_from_data(original_record, data)

                # Create result
                result = VectorSearchResult(
                    record=record,
                    score=float(score),
                    vector_field=vector_field
                )
                results.append(result)

        # Sort by score (descending) and return top k
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:k]

    def _extract_record_data(self, record_dict: dict[str, Any]) -> dict[str, Any]:
        """Extract the actual data from a record dictionary.
        
        Handles different storage formats like:
        - Direct data storage
        - Data in a 'data' column (JSON)
        - Double-nested data structures
        
        Args:
            record_dict: Raw record dictionary from database
            
        Returns:
            Extracted data dictionary
        """
        import json

        # Check if there's a 'data' column (common in generic table structures)
        if 'data' in record_dict:
            data = record_dict['data']

            # Parse JSON if needed
            if isinstance(data, str):
                data = json.loads(data)

            # Handle double-nested data structure
            if isinstance(data, dict) and 'data' in data:
                data = data['data']

            return data

        # Direct storage
        return record_dict

    def _create_record_from_data(self, record_dict: dict[str, Any], data: dict[str, Any]) -> "Record":
        """Create a Record object from raw data.
        
        Args:
            record_dict: Original record dictionary (may contain metadata)
            data: Extracted data dictionary
            
        Returns:
            Record object
        """
        import json

        from ..records import Record

        # Extract metadata if present
        metadata = record_dict.get('metadata', {})
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata) if metadata else {}
            except json.JSONDecodeError:
                metadata = {}

        # Create Record with proper initialization
        record = Record(data=data, id=record_dict.get('id'), metadata=metadata)

        return record
