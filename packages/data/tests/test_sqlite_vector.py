"""Test SQLite vector support implementation."""

import numpy as np
import pytest
import tempfile
import os

from dataknobs_data.backends.sqlite import SyncSQLiteDatabase
from dataknobs_data.records import Record
from dataknobs_data.fields import VectorField
from dataknobs_data.vector.types import DistanceMetric
from collections import OrderedDict


class TestSQLiteVectorSupport:
    """Test SQLite vector support functionality."""
    
    @pytest.fixture
    def db_path(self):
        """Create a temporary database file."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        yield path
        # Cleanup
        if os.path.exists(path):
            os.unlink(path)
    
    @pytest.fixture
    def db(self, db_path):
        """Create a SQLite database with vector support."""
        db = SyncSQLiteDatabase({
            "path": db_path,
            "table": "test_vectors",
            "vector_enabled": True,
            "vector_metric": "cosine"
        })
        db.connect()
        yield db
        db.close()
    
    def test_vector_support_detection(self, db):
        """Test that SQLite reports no native vector support."""
        assert db.has_vector_support() is False  # No native support
        assert db.enable_vector_support() is True  # But can be enabled (Python-based)
        assert db.vector_enabled is True
    
    def test_create_record_with_vector(self, db):
        """Test creating a record with a vector field."""
        # Create a vector field
        vector = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        vector_field = VectorField(
            name="embedding",
            value=vector,
            dimensions=4
        )
        
        # Create a record
        record = Record(
            data=OrderedDict({"embedding": vector_field}),
            metadata={"test": "value"}
        )
        
        # Save the record
        record_id = db.create(record)
        assert record_id is not None
        
        # Read it back
        retrieved = db.read(record_id)
        assert retrieved is not None
        assert "embedding" in retrieved.fields
        
        # Check the vector was preserved
        retrieved_vector = retrieved.fields["embedding"].value
        assert isinstance(retrieved_vector, np.ndarray)
        np.testing.assert_array_almost_equal(retrieved_vector, vector)
    
    def test_vector_search(self, db):
        """Test vector similarity search."""
        # Add some test vectors
        vectors = []
        for i in range(5):
            vec = np.random.randn(8).astype(np.float32)
            vec = vec / np.linalg.norm(vec)  # Normalize
            vectors.append(vec)
        
        # Add vectors to database
        ids = db.add_vectors(
            vectors=vectors,
            metadata=[{"index": i} for i in range(5)],
            field_name="embedding"
        )
        
        assert len(ids) == 5
        
        # Search for similar vectors
        query_vec = vectors[0] + np.random.randn(8) * 0.01  # Slightly perturbed first vector
        query_vec = query_vec / np.linalg.norm(query_vec)
        
        results = db.vector_search(
            query_vector=query_vec,
            field_name="embedding",
            k=3
        )
        
        assert len(results) <= 3
        assert results[0].score > 0.9  # Should be very similar to first vector
        # The best match should have index 0 in metadata
        assert results[0].record.metadata["index"] == 0
    
    def test_vector_search_with_filter(self, db):
        """Test vector search with metadata filter."""
        from dataknobs_data.query import Query, Filter, Operator
        
        # Add vectors with categories
        vectors = []
        categories = ["A", "B", "A", "B", "A"]
        
        for i in range(5):
            vec = np.random.randn(4).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            vectors.append(vec)
        
        ids = []
        for i, (vec, cat) in enumerate(zip(vectors, categories)):
            vector_field = VectorField(
                name="embedding",
                value=vec,
                dimensions=4
            )
            
            record = Record(
                data=OrderedDict({
                    "embedding": vector_field,
                    "category": cat
                }),
                metadata={"index": i}
            )
            
            ids.append(db.create(record))
        
        # Search with filter for category A
        query_vec = np.random.randn(4).astype(np.float32)
        query_vec = query_vec / np.linalg.norm(query_vec)
        
        # Create a Query object for filtering
        filter_query = Query(filters=[
            Filter(field="category", operator=Operator.EQ, value="A")
        ])
        
        results = db.vector_search(
            query_vector=query_vec,
            field_name="embedding",
            k=10,
            filter=filter_query
        )
        
        # Should only return category A records
        assert all(r.record.fields["category"].value == "A" for r in results)
        assert len(results) <= 3  # Only 3 records have category A
    
    def test_different_distance_metrics(self, db):
        """Test different distance metrics for vector search."""
        # Create two vectors
        vec1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        vec2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        
        # Add them to database
        ids = db.add_vectors(
            vectors=[vec1, vec2],
            metadata=[{"name": "vec1"}, {"name": "vec2"}]
        )
        
        # Test cosine similarity
        results_cosine = db.vector_search(
            query_vector=vec1,
            k=2,
            metric=DistanceMetric.COSINE
        )
        assert results_cosine[0].record.metadata["name"] == "vec1"  # vec1 should be first
        
        # Test Euclidean distance (converted to similarity)
        results_euclidean = db.vector_search(
            query_vector=vec1,
            k=2,
            metric=DistanceMetric.EUCLIDEAN
        )
        assert results_euclidean[0].record.metadata["name"] == "vec1"  # vec1 should be first
        
        # Test dot product
        results_dot = db.vector_search(
            query_vector=vec1,
            k=2,
            metric=DistanceMetric.DOT_PRODUCT
        )
        assert results_dot[0].record.metadata["name"] == "vec1"  # vec1 should be first
    
    def test_update_record_with_vector(self, db):
        """Test updating a record with a vector field."""
        # Create initial record
        vec1 = np.array([1.0, 0.0], dtype=np.float32)
        vector_field = VectorField(name="embedding", value=vec1)
        
        record = Record(
            data=OrderedDict({"embedding": vector_field}),
            metadata={"version": 1}
        )
        record_id = db.create(record)
        
        # Update with new vector
        vec2 = np.array([0.0, 1.0], dtype=np.float32)
        updated_field = VectorField(name="embedding", value=vec2)
        
        updated_record = Record(
            data=OrderedDict({"embedding": updated_field}),
            metadata={"version": 2}
        )
        
        success = db.update(record_id, updated_record)
        assert success
        
        # Verify update
        retrieved = db.read(record_id)
        np.testing.assert_array_almost_equal(
            retrieved.fields["embedding"].value,
            vec2
        )
        assert retrieved.metadata["version"] == 2
    
    def test_batch_operations_with_vectors(self, db):
        """Test batch create with vector fields."""
        # Create multiple records with vectors
        records = []
        for i in range(10):
            vec = np.random.randn(4).astype(np.float32)
            vector_field = VectorField(
                name="embedding",
                value=vec,
                dimensions=4
            )
            
            record = Record(
                data=OrderedDict({"embedding": vector_field}),
                metadata={"batch_index": i}
            )
            record.id = f"batch_{i}"
            records.append(record)
        
        # Batch create
        ids = db.create_batch(records)
        assert len(ids) == 10
        
        # Verify all records were created
        for i, record_id in enumerate(ids):
            retrieved = db.read(record_id)
            assert retrieved is not None
            assert retrieved.metadata["batch_index"] == i
    
    def test_empty_vector_handling(self, db):
        """Test handling of records without vectors."""
        # Create record without a vector field
        # This simulates a record that hasn't been vectorized yet
        from dataknobs_data.fields import Field
        
        record = Record(
            data=OrderedDict({
                "text": Field(name="text", value="Some text without a vector")
            }),
            metadata={"has_vector": False}
        )
        
        record_id = db.create(record)
        
        # Also create a record with a vector for comparison
        vec = np.random.randn(4).astype(np.float32)
        vector_field = VectorField(
            name="embedding",
            value=vec,
            dimensions=4
        )
        
        record_with_vector = Record(
            data=OrderedDict({"embedding": vector_field}),
            metadata={"has_vector": True}
        )
        db.create(record_with_vector)
        
        # Search should only return the record with a vector
        query_vec = np.random.randn(4).astype(np.float32)
        results = db.vector_search(
            query_vector=query_vec,
            field_name="embedding",
            k=10
        )
        
        # Should not include the record without a vector field
        result_ids = [r.record.id for r in results]
        assert record_id not in result_ids
        assert len(results) == 1  # Only the record with vector