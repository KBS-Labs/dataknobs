"""Test sync Elasticsearch vector functionality."""

import os
import time
import pytest
import numpy as np

from dataknobs_data import Record
from dataknobs_data.backends.elasticsearch import SyncElasticsearchDatabase
from dataknobs_data.fields import VectorField
from dataknobs_data.vector.types import DistanceMetric

# Skip tests if Elasticsearch integration testing is not enabled
pytestmark = pytest.mark.skipif(
    os.environ.get("TEST_ELASTICSEARCH") != "true",
    reason="Elasticsearch integration tests not enabled"
)


def test_sync_vector_search(elasticsearch_test_index):
    """Test vector search with sync Elasticsearch backend."""
    db = SyncElasticsearchDatabase(elasticsearch_test_index)
    db.connect()
    
    try:
        # Create vector index
        db.create_vector_index(
            vector_field="embedding",
            dimensions=8,
            metric=DistanceMetric.COSINE
        )
        
        # Create test records with vectors
        vec1 = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        vec2 = np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        vec3 = np.array([0.7, 0.7, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        
        records = []
        for i, vec in enumerate([vec1, vec2, vec3]):
            rec = Record({"name": f"Doc{i}"})
            rec.fields["embedding"] = VectorField(vec, name="embedding")
            records.append(rec)
        
        # Create records
        ids = []
        for rec in records:
            id = db.create(rec)
            ids.append(id)
        
        # Wait for indexing
        time.sleep(2)
        
        # Search for similar vectors
        results = db.vector_search(
            query_vector=vec1,
            field_name="embedding",
            k=3,
            metric=DistanceMetric.COSINE
        )
        
        assert len(results) > 0
        assert results[0].record.id == ids[0]  # First doc should match
        assert results[0].score > 0.99  # Should be very similar
        
        # Test that vector field is in result
        assert "embedding" in results[0].record.fields
        embedding = results[0].record.fields["embedding"]
        assert isinstance(embedding, VectorField)
        assert embedding.value is not None
        
    finally:
        db.close()
