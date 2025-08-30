"""Integration tests for Elasticsearch vector functionality."""

import os
import time
import pytest
import numpy as np

from dataknobs_data import Record
from dataknobs_data.backends.elasticsearch_async import AsyncElasticsearchDatabase
from dataknobs_data.fields import VectorField
from dataknobs_data.query import Query, Operator
from dataknobs_data.vector.types import DistanceMetric

# Skip tests if Elasticsearch integration testing is not enabled
pytestmark = pytest.mark.skipif(
    os.environ.get("TEST_ELASTICSEARCH") != "true",
    reason="Elasticsearch integration tests not enabled"
)


class TestElasticsearchVectorIntegration:
    """Test Elasticsearch vector functionality with real backend."""
    
    @pytest.fixture
    async def vector_test_index(self, elasticsearch_test_index):
        """Provide an index configuration for vector tests."""
        # Add vector-specific configuration
        config = elasticsearch_test_index.copy()
        config["vector_enabled"] = True
        return config
    
    async def test_vector_field_storage_and_retrieval(self, vector_test_index):
        """Test storing and retrieving vector fields."""
        db = AsyncElasticsearchDatabase(vector_test_index)
        await db.connect()
        
        try:
            # Create a record with a vector field
            vector = np.random.rand(128).astype(np.float32)
            record = Record({
                "title": "Test Document",
                "content": "This is a test document for vector storage",
            })
            
            # Add vector field
            record.fields["embedding"] = VectorField(
                value=vector,
                name="embedding",
                source_field="content",
                model_name="test-model",
                model_version="1.0"
            )
            
            # Store the record
            record_id = await db.create(record)
            assert record_id is not None
            
            # Wait for indexing (Elasticsearch near real-time)
            time.sleep(1)
            
            # Retrieve the record
            retrieved = await db.read(record_id)
            assert retrieved is not None
            assert "embedding" in retrieved.fields
            
            # Check vector field
            embedding_field = retrieved.fields["embedding"]
            assert isinstance(embedding_field, VectorField)
            assert embedding_field.value is not None
            assert isinstance(embedding_field.value, np.ndarray)
            assert embedding_field.value.shape == (128,)
            
            # Check vector values match (within tolerance)
            np.testing.assert_allclose(
                embedding_field.value,
                vector,
                rtol=1e-5
            )
            
            # Check metadata preserved
            assert embedding_field.source_field == "content"
            
        finally:
            await db.close()
    
    async def test_vector_similarity_search(self, vector_test_index):
        """Test vector similarity search."""
        db = AsyncElasticsearchDatabase(vector_test_index)
        await db.connect()
        
        try:
            # Create vector index first
            await db.create_vector_index(
                vector_field="embedding",
                dimensions=128,
                metric=DistanceMetric.COSINE
            )
            
            # Create test vectors
            base_vector = np.random.rand(128).astype(np.float32)
            similar_vector = base_vector + np.random.randn(128) * 0.1
            different_vector = np.random.rand(128).astype(np.float32)
            
            # Create records with vectors
            records = [
                Record({
                    "title": "Similar Document",
                    "embedding": VectorField(similar_vector, name="embedding")
                }),
                Record({
                    "title": "Different Document", 
                    "embedding": VectorField(different_vector, name="embedding")
                }),
                Record({
                    "title": "Base Document",
                    "embedding": VectorField(base_vector, name="embedding")
                }),
            ]
            
            # Store records
            ids = []
            for record in records:
                record_id = await db.create(record)
                ids.append(record_id)
            
            # Wait for indexing
            time.sleep(2)
            
            # Search for similar vectors
            results = await db.vector_search(
                query_vector=base_vector,
                vector_field="embedding",
                k=3,
                metric=DistanceMetric.COSINE
            )
            
            assert len(results) > 0
            assert results[0].record.id == ids[2]  # Base document should be first
            assert results[0].score > 0.99  # Should be very close to 1.0
            
            # The similar document should rank higher than different
            similar_idx = next((i for i, r in enumerate(results) if r.record.id == ids[0]), -1)
            different_idx = next((i for i, r in enumerate(results) if r.record.id == ids[1]), -1)
            
            if similar_idx != -1 and different_idx != -1:
                assert similar_idx < different_idx
            
        finally:
            await db.close()
    
    async def test_filtered_vector_search(self, vector_test_index):
        """Test vector search with filters."""
        db = AsyncElasticsearchDatabase(vector_test_index)
        await db.connect()
        
        try:
            # Create vector index
            await db.create_vector_index(
                vector_field="embedding",
                dimensions=64,
                metric=DistanceMetric.COSINE
            )
            
            # Create test data
            vectors = [np.random.rand(64).astype(np.float32) for _ in range(5)]
            categories = ["A", "B", "A", "B", "C"]
            
            # Create and store records
            ids = []
            for i, (vec, cat) in enumerate(zip(vectors, categories)):
                record = Record({
                    "title": f"Document {i}",
                    "category": cat,
                    "embedding": VectorField(vec, name="embedding")
                })
                record_id = await db.create(record)
                ids.append(record_id)
            
            # Wait for indexing
            time.sleep(2)
            
            # Search with filter for category A
            filter_query = Query().filter("category", Operator.EQ, "A")
            results = await db.vector_search(
                query_vector=vectors[0],
                vector_field="embedding",
                k=5,
                filter=filter_query
            )
            
            # Should only return category A documents
            assert len(results) == 2
            for result in results:
                assert result.record.get_value("category") == "A"
            
        finally:
            await db.close()
    
    async def test_different_distance_metrics(self, vector_test_index):
        """Test vector search with different distance metrics."""
        db = AsyncElasticsearchDatabase(vector_test_index)
        await db.connect()
        
        try:
            # Test vectors
            vec1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            vec2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            vec3 = np.array([0.707, 0.707, 0.0], dtype=np.float32)  # 45 degrees from vec1
            
            # Test COSINE similarity
            await db.create_vector_index(
                vector_field="embedding_cosine",
                dimensions=3,
                metric=DistanceMetric.COSINE
            )
            
            # Create records
            for i, vec in enumerate([vec1, vec2, vec3]):
                record = Record({
                    "id": f"vec_{i}",
                    "embedding_cosine": VectorField(vec, name="embedding_cosine")
                })
                await db.create(record)
            
            # Wait for indexing
            time.sleep(2)
            
            # Search with cosine similarity
            results = await db.vector_search(
                query_vector=vec1,
                vector_field="embedding_cosine",
                k=3,
                metric=DistanceMetric.COSINE
            )
            
            assert len(results) > 0
            # vec1 should be first (exact match), vec3 should be second (45 degrees)
            assert results[0].record.get_value("id") == "vec_0"
            if len(results) > 1:
                assert results[1].record.get_value("id") == "vec_2"
            
        finally:
            await db.close()
    
    async def test_batch_vector_operations(self, vector_test_index):
        """Test batch operations with vector fields."""
        db = AsyncElasticsearchDatabase(vector_test_index)
        await db.connect()
        
        try:
            # Create vector index
            await db.create_vector_index(
                vector_field="embedding",
                dimensions=32,
                metric=DistanceMetric.COSINE
            )
            
            # Create batch of records with vectors
            batch_size = 10
            records = []
            
            for i in range(batch_size):
                vec = np.random.rand(32).astype(np.float32)
                record = Record({
                    "batch_id": i,
                    "text": f"Batch document {i}",
                    "embedding": VectorField(vec, name="embedding")
                })
                records.append(record)
            
            # Batch create
            ids = await db.create_batch(records)
            assert len(ids) == batch_size
            
            # Wait for indexing
            time.sleep(2)
            
            # Verify all records searchable
            query_vec = records[0].fields["embedding"].value
            results = await db.vector_search(
                query_vector=query_vec,
                vector_field="embedding",
                k=batch_size
            )
            
            assert len(results) == batch_size
            
        finally:
            await db.close()
    
    async def test_vector_field_metadata_persistence(self, vector_test_index):
        """Test that vector field metadata is preserved."""
        db = AsyncElasticsearchDatabase(vector_test_index)
        await db.connect()
        
        try:
            # Create record with detailed vector metadata
            vec = np.random.rand(64).astype(np.float32)
            record = Record({
                "document": "Test content"
            })
            
            record.fields["embedding"] = VectorField(
                value=vec,
                name="embedding",
                source_field="document",
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_version="2.0.0"
            )
            
            # Store and retrieve
            record_id = await db.create(record)
            time.sleep(1)
            
            retrieved = await db.read(record_id)
            assert retrieved is not None
            
            # Check metadata
            embedding_field = retrieved.fields["embedding"]
            assert isinstance(embedding_field, VectorField)
            assert embedding_field.source_field == "document"
            
            # Check vector field metadata in record metadata
            if "vector_fields" in retrieved.metadata:
                vec_meta = retrieved.metadata["vector_fields"].get("embedding", {})
                assert vec_meta.get("source_field") == "document"
                assert vec_meta.get("model") == "sentence-transformers/all-MiniLM-L6-v2"
                assert vec_meta.get("model_version") == "2.0.0"
            
        finally:
            await db.close()