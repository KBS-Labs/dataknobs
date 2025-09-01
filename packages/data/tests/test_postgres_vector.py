"""Tests for PostgreSQL vector integration.

These tests use the in-memory backend to test vector functionality
without requiring a real PostgreSQL instance with pgvector.
"""

import os
import pytest

from dataknobs_data import Record, VectorField
from dataknobs_data.backends.memory import AsyncMemoryDatabase
from dataknobs_data.vector import DistanceMetric, VectorSearchResult

# Skip tests if numpy is not available
np = pytest.importorskip("numpy")

# Skip all tests if PostgreSQL is not available
pytestmark = pytest.mark.skipif(
    not os.environ.get("TEST_POSTGRES", "").lower() == "true",
    reason="PostgreSQL vector tests require TEST_POSTGRES=true and a running PostgreSQL instance with pgvector"
)


@pytest.mark.asyncio
class TestVectorFieldIntegration:
    """Test vector field integration with real database backends."""

    @pytest.fixture
    async def db(self):
        """Create an in-memory database for testing."""
        db = AsyncMemoryDatabase()
        await db.connect()
        
        yield db
        
        await db.close()

    async def test_create_record_with_vector(self, db):
        """Test creating a record with vector field."""
        # Create record with vector
        vector = np.array([0.1, 0.2, 0.3])
        record = Record(data={"text": "sample text"})
        record.fields["embedding"] = VectorField(
            vector,
            name="embedding",
            source_field="text",
            model_name="test-model"
        )
        
        record_id = await db.create(record)
        assert record_id is not None
        
        # Read it back
        retrieved = await db.read(record_id)
        assert retrieved is not None
        assert "text" in retrieved.fields
        assert "embedding" in retrieved.fields
        
        # The vector should be stored as a list in JSON
        embedding_value = retrieved.fields["embedding"].value
        if isinstance(embedding_value, list):
            assert np.allclose(embedding_value, vector.tolist())

    async def test_vector_field_serialization(self, db):
        """Test vector field serialization and deserialization."""
        vector = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        record = Record(data={"title": "Test Document"})
        record.fields["features"] = VectorField(
            vector,
            name="features",
            dimensions=5,
            source_field="title",
            model_name="bert",
            model_version="1.0"
        )
        
        # Store and retrieve
        record_id = await db.create(record)
        retrieved = await db.read(record_id)
        
        assert retrieved is not None
        assert "features" in retrieved.fields
        
        # Check if vector data is preserved
        features_value = retrieved.fields["features"].value
        if isinstance(features_value, list):
            assert len(features_value) == 5
            assert np.allclose(features_value, vector.tolist())

    async def test_search_with_vector_fields(self, db):
        """Test searching records that contain vector fields."""
        # Create multiple records with vectors
        vectors = [
            np.array([0.1, 0.2, 0.3]),
            np.array([0.4, 0.5, 0.6]),
            np.array([0.7, 0.8, 0.9])
        ]
        
        record_ids = []
        for i, vec in enumerate(vectors):
            record = Record(data={
                "text": f"document {i}",
                "category": "test"
            })
            record.fields["embedding"] = VectorField(vec, name="embedding")
            record_id = await db.create(record)
            record_ids.append(record_id)
        
        # Search by regular field
        from dataknobs_data import Query, Filter, Operator
        query = Query(filters=[
            Filter(field="category", operator=Operator.EQ, value="test")
        ])
        
        results = await db.search(query)
        assert len(results) == 3
        
        # Verify all records have embedding fields
        for record in results:
            assert "embedding" in record.fields

    async def test_update_record_with_vector(self, db):
        """Test updating a record that contains a vector field."""
        # Create initial record
        initial_vector = np.array([1.0, 0.0, 0.0])
        record = Record(data={"text": "initial"})
        record.fields["embedding"] = VectorField(initial_vector, name="embedding")
        
        record_id = await db.create(record)
        
        # Update the record with new vector
        updated_vector = np.array([0.0, 1.0, 0.0])
        updated_record = Record(data={"text": "updated"})
        updated_record.fields["embedding"] = VectorField(updated_vector, name="embedding")
        
        success = await db.update(record_id, updated_record)
        assert success is True
        
        # Verify update
        retrieved = await db.read(record_id)
        assert retrieved.fields["text"].value == "updated"
        embedding_value = retrieved.fields["embedding"].value
        if isinstance(embedding_value, list):
            assert np.allclose(embedding_value, updated_vector.tolist())

    async def test_batch_operations_with_vectors(self, db):
        """Test batch operations with vector fields."""
        records = []
        for i in range(5):
            record = Record(data={"index": i})
            record.fields["vector"] = VectorField(
                np.random.rand(3),  # Random 3D vector
                name="vector"
            )
            records.append(record)
        
        # Batch create
        ids = await db.create_batch(records)
        assert len(ids) == 5
        
        # Verify all were created
        for record_id in ids:
            retrieved = await db.read(record_id)
            assert retrieved is not None
            assert "vector" in retrieved.fields


class TestVectorOperations:
    """Test vector-specific operations."""

    def test_vector_field_cosine_similarity(self):
        """Test cosine similarity computation between vector fields."""
        vec1 = VectorField(np.array([1.0, 0.0, 0.0]), name="v1")
        vec2 = VectorField(np.array([1.0, 0.0, 0.0]), name="v2")
        vec3 = VectorField(np.array([0.0, 1.0, 0.0]), name="v3")
        
        # Same vectors should have similarity 1
        similarity = vec1.cosine_similarity(vec2)
        assert np.isclose(similarity, 1.0)
        
        # Orthogonal vectors should have similarity 0
        similarity = vec1.cosine_similarity(vec3)
        assert np.isclose(similarity, 0.0)

    def test_vector_field_euclidean_distance(self):
        """Test Euclidean distance computation."""
        vec1 = VectorField(np.array([0.0, 0.0]), name="v1")
        vec2 = VectorField(np.array([3.0, 4.0]), name="v2")
        
        # Distance should be 5 (3-4-5 triangle)
        distance = vec1.euclidean_distance(vec2)
        assert np.isclose(distance, 5.0)

    def test_vector_field_with_metadata(self):
        """Test vector field with full metadata."""
        vector = np.array([0.1, 0.2, 0.3])
        field = VectorField(
            value=vector,
            name="embedding",
            dimensions=3,
            source_field="text",
            model_name="bert-base",
            model_version="1.0.0",
            metadata={"custom_key": "custom_value"}
        )
        
        assert field.dimensions == 3
        assert field.source_field == "text"
        assert field.model_name == "bert-base"
        assert field.model_version == "1.0.0"
        assert field.metadata["custom_key"] == "custom_value"
        
        # Convert to dict and back
        data = field.to_dict()
        restored = VectorField.from_dict(data)
        
        assert restored.dimensions == field.dimensions
        assert restored.source_field == field.source_field
        assert restored.model_name == field.model_name
        assert np.allclose(restored.value, field.value)


class TestPostgresVectorUtilities:
    """Test PostgreSQL vector utility functions."""

    def test_format_vector_for_postgres(self):
        """Test formatting vectors for PostgreSQL."""
        from dataknobs_data.backends.postgres_vector import format_vector_for_postgres
        
        # Test numpy array
        vector = np.array([0.1, 0.2, 0.3])
        result = format_vector_for_postgres(vector)
        assert result == "[0.1,0.2,0.3]"
        
        # Test list
        vector = [1.0, 2.0, 3.0]
        result = format_vector_for_postgres(vector)
        assert result == "[1.0,2.0,3.0]"

    def test_parse_postgres_vector(self):
        """Test parsing PostgreSQL vector strings."""
        from dataknobs_data.backends.postgres_vector import parse_postgres_vector
        
        result = parse_postgres_vector("[0.1,0.2,0.3]")
        assert result == [0.1, 0.2, 0.3]
        
        result = parse_postgres_vector("[]")
        assert result == []

    def test_get_vector_operator(self):
        """Test getting correct PostgreSQL operators."""
        from dataknobs_data.backends.postgres_vector import get_vector_operator
        
        assert get_vector_operator("cosine") == "<=>"
        assert get_vector_operator("euclidean") == "<->"
        assert get_vector_operator("inner_product") == "<#>"
        assert get_vector_operator("l2") == "<->"
        assert get_vector_operator("unknown") == "<=>"  # Default

    def test_get_optimal_index_type(self):
        """Test optimal index selection based on dataset size."""
        from dataknobs_data.backends.postgres_vector import get_optimal_index_type
        
        # Small dataset
        index_type, params = get_optimal_index_type(1000)
        assert index_type == "ivfflat"
        assert params["lists"] == 100
        
        # Medium dataset
        index_type, params = get_optimal_index_type(100000)
        assert index_type == "ivfflat"
        assert params["lists"] > 100
        
        # Large dataset
        index_type, params = get_optimal_index_type(10000000)
        assert index_type == "hnsw"
        assert "m" in params
        assert "ef_construction" in params

    def test_build_vector_index_sql(self):
        """Test building vector index SQL."""
        from dataknobs_data.backends.postgres_vector import build_vector_index_sql
        
        # IVFFlat index
        sql = build_vector_index_sql(
            "records", "public", "vector_embedding", 768,
            metric="cosine", index_type="ivfflat", index_params={"lists": 100}
        )
        assert "CREATE INDEX" in sql
        assert "USING ivfflat" in sql
        assert "vector_cosine_ops" in sql  # Cosine operator class
        assert "lists = 100" in sql
        
        # HNSW index
        sql = build_vector_index_sql(
            "records", "public", "vector_embedding", 768,
            metric="euclidean", index_type="hnsw",
            index_params={"m": 16, "ef_construction": 200}
        )
        assert "USING hnsw" in sql
        assert "vector_l2_ops" in sql  # L2 operator class
        assert "m = 16" in sql
        assert "ef_construction = 200" in sql