"""Integration tests for PostgreSQL with pgvector extension using real database."""

import os
import uuid

import pytest

from dataknobs_data import AsyncDatabase, Query, Record, SyncDatabase, VectorField
from dataknobs_data.query import Filter, Operator
from dataknobs_data.vector import DistanceMetric, VectorSearchResult

# Skip all tests if PostgreSQL is not available or numpy is not installed
pytestmark = pytest.mark.skipif(
    not os.environ.get("TEST_POSTGRES", "").lower() == "true",
    reason="PostgreSQL tests require TEST_POSTGRES=true and a running PostgreSQL instance with pgvector"
)

# Also skip if numpy is not available
np = pytest.importorskip("numpy")


class TestPostgresVectorIntegration:
    """Integration tests for PostgreSQL with pgvector extension."""

    @pytest.fixture
    def vector_test_db(self, postgres_test_db):
        """Enhanced test database configuration with vector support."""
        # Add vector-specific configuration
        config = postgres_test_db.copy()
        config["vector_enabled"] = True
        return config

    def test_pgvector_extension_detection(self, vector_test_db):
        """Test automatic pgvector extension detection and installation."""
        db = SyncDatabase.from_backend("postgres", vector_test_db)
        
        try:
            # The database should detect and enable pgvector automatically
            # Create a record with a vector field to trigger detection
            vector = np.array([0.1, 0.2, 0.3, 0.4])
            record = Record({
                "text": "Test document",
                "category": "test"
            })
            record.fields["embedding"] = VectorField(
                vector,
                name="embedding",
                source_field="text",
                model_name="test-model"
            )
            
            # This should succeed if pgvector is available
            record_id = db.create(record)
            assert record_id is not None
            
            # Verify we can read it back
            retrieved = db.read(record_id)
            assert retrieved is not None
            assert "embedding" in retrieved.fields
            
            # Cleanup
            db.delete(record_id)
        finally:
            db.close()

    def test_vector_field_storage_and_retrieval(self, vector_test_db):
        """Test storing and retrieving vector fields with PostgreSQL."""
        db = SyncDatabase.from_backend("postgres", vector_test_db)
        
        try:
            # Create multiple records with vectors
            vectors = [
                np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
                np.array([0.5, 0.4, 0.3, 0.2, 0.1]),
                np.array([0.3, 0.3, 0.3, 0.3, 0.3]),
            ]
            
            record_ids = []
            for i, vec in enumerate(vectors):
                record = Record({
                    "title": f"Document {i}",
                    "content": f"This is test document number {i}",
                })
                record.fields["embedding"] = VectorField(
                    vec,
                    name="embedding",
                    dimensions=5,
                    source_field="content",
                    model_name="test-model",
                    model_version="1.0"
                )
                record_id = db.create(record)
                record_ids.append(record_id)
            
            # Retrieve and verify each record
            for i, record_id in enumerate(record_ids):
                retrieved = db.read(record_id)
                assert retrieved is not None
                assert retrieved.get_value("title") == f"Document {i}"
                assert "embedding" in retrieved.fields
                
                # Check vector values (should be stored as list in JSON)
                embedding_value = retrieved.fields["embedding"].value
                if isinstance(embedding_value, list):
                    assert np.allclose(embedding_value, vectors[i].tolist())
            
            # Cleanup
            for record_id in record_ids:
                db.delete(record_id)
        finally:
            db.close()

    def test_vector_similarity_search(self, vector_test_db):
        """Test vector similarity search with real PostgreSQL backend."""
        from dataknobs_data.backends.postgres import SyncPostgresDatabase
        
        # Use SyncPostgresDatabase directly for vector operations
        db = SyncPostgresDatabase(vector_test_db)
        db.connect()
        
        try:
            # Create test data with embeddings
            documents = [
                ("The cat sat on the mat", [0.1, 0.8, 0.2, 0.1]),
                ("The dog played in the park", [0.3, 0.2, 0.7, 0.1]),
                ("A bird flew over the tree", [0.2, 0.1, 0.3, 0.8]),
                ("The cat and dog are friends", [0.2, 0.5, 0.4, 0.1]),
                ("Trees in the park are green", [0.1, 0.1, 0.4, 0.6]),
            ]
            
            record_ids = []
            for text, embedding in documents:
                record = Record({"text": text, "type": "document"})
                record.fields["embedding"] = VectorField(
                    np.array(embedding, dtype=np.float32),
                    name="embedding",
                    source_field="text"
                )
                record_id = db.create(record)
                record_ids.append(record_id)
            
            # Perform vector similarity search - looking for cat-related documents
            query_vector = np.array([0.15, 0.65, 0.3, 0.1], dtype=np.float32)
            
            # Test cosine similarity search
            results = db.vector_search(
                query_vector=query_vector,
                field_name="embedding",
                k=3,
                metric="cosine"
            )
            
            assert len(results) <= 3
            assert all(isinstance(r, VectorSearchResult) for r in results)
            
            # Results should be ordered by similarity (descending)
            if len(results) > 1:
                for i in range(len(results) - 1):
                    assert results[i].score >= results[i + 1].score
            
            # The most similar should be cat-related documents
            if results:
                top_result = results[0]
                assert top_result.record is not None
                text = top_result.record.get_value("text")
                assert "cat" in text.lower() or "dog" in text.lower()
            
            # Test with filters
            from dataknobs_data.query import Query
            filter_query = Query(filters=[Filter(field="type", operator=Operator.EQ, value="document")])
            filtered_results = db.vector_search(
                query_vector=query_vector,
                field_name="embedding",
                k=5,
                filter=filter_query,
                metric="cosine"
            )
            
            assert all(r.record.get_value("type") == "document" for r in filtered_results)
            
            # Test Euclidean distance
            euclidean_results = db.vector_search(
                query_vector=query_vector,
                field_name="embedding",
                k=3,
                metric="euclidean"
            )
            
            assert len(euclidean_results) <= 3
            
            # Cleanup
            for record_id in record_ids:
                db.delete(record_id)
        finally:
            db.close()

    def test_batch_vector_operations(self, vector_test_db):
        """Test batch operations with vector fields."""
        db = SyncDatabase.from_backend("postgres", vector_test_db)
        
        try:
            # Create batch of records with vectors
            records = []
            for i in range(10):
                record = Record({
                    "index": i,
                    "description": f"Item number {i}"
                })
                # Generate a random vector
                record.fields["features"] = VectorField(
                    np.random.rand(8).astype(np.float32),
                    name="features",
                    source_field="description"
                )
                records.append(record)
            
            # Batch create
            ids = db.create_batch(records)
            assert len(ids) == 10
            
            # Read batch
            retrieved = db.read_batch(ids)
            assert len(retrieved) == 10
            
            # Verify all have vector fields
            for record in retrieved:
                assert record is not None
                assert "features" in record.fields
                assert isinstance(record.fields["features"].value, (list, np.ndarray))
            
            # Update batch with new vectors
            for record in retrieved:
                record.fields["features"] = VectorField(
                    np.random.rand(8).astype(np.float32),
                    name="features",
                    source_field="description"
                )
            
            success = db.update_batch(list(zip(ids, retrieved)))
            assert all(success)
            
            # Cleanup
            success = db.delete_batch(ids)
            assert all(success)
        finally:
            db.close()

    def test_vector_field_metadata_persistence(self, vector_test_db):
        """Test that vector field metadata is preserved through storage."""
        db = SyncDatabase.from_backend("postgres", vector_test_db)
        
        try:
            # Create record with detailed vector metadata
            vector = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
            record = Record({"content": "Test document for metadata"})
            record.fields["embedding"] = VectorField(
                vector,
                name="embedding",
                dimensions=8,
                source_field="content",
                model_name="sentence-transformer",
                model_version="2.3.0",
                metadata={"temperature": 0.7, "max_tokens": 512}
            )
            
            record_id = db.create(record)
            
            # Retrieve and check metadata
            retrieved = db.read(record_id)
            assert retrieved is not None
            assert "embedding" in retrieved.fields
            
            embedding_field = retrieved.fields["embedding"]
            assert embedding_field.metadata.get("dimensions") == 8
            assert embedding_field.metadata.get("source_field") == "content"
            assert embedding_field.metadata.get("model", {}).get("name") == "sentence-transformer"
            assert embedding_field.metadata.get("model", {}).get("version") == "2.3.0"
            assert embedding_field.metadata.get("temperature") == 0.7
            assert embedding_field.metadata.get("max_tokens") == 512
            
            # Cleanup
            db.delete(record_id)
        finally:
            db.close()


@pytest.mark.asyncio
class TestAsyncPostgresVectorIntegration:
    """Async integration tests for PostgreSQL with pgvector."""

    @pytest.fixture
    def vector_test_db(self, postgres_test_db):
        """Enhanced test database configuration with vector support."""
        config = postgres_test_db.copy()
        config["vector_enabled"] = True
        return config

    async def test_async_vector_operations(self, vector_test_db):
        """Test async vector operations with PostgreSQL."""
        from dataknobs_data.backends.postgres import AsyncPostgresDatabase
        
        db = AsyncPostgresDatabase(vector_test_db)
        await db.connect()
        
        try:
            # Create test data
            vectors = [
                np.array([1.0, 0.0, 0.0, 0.0]),
                np.array([0.0, 1.0, 0.0, 0.0]),
                np.array([0.0, 0.0, 1.0, 0.0]),
                np.array([0.0, 0.0, 0.0, 1.0]),
            ]
            
            record_ids = []
            for i, vec in enumerate(vectors):
                record = Record({
                    "name": f"Vector {i}",
                    "category": "test"
                })
                record.fields["vector"] = VectorField(vec, name="vector")
                record_id = await db.create(record)
                record_ids.append(record_id)
            
            # Search for similar vectors
            query = np.array([0.9, 0.1, 0.0, 0.0])
            results = await db.vector_search(
                query_vector=query,
                field_name="vector",
                k=2,
                metric="cosine"
            )
            
            assert len(results) <= 2
            if results:
                # First result should be closest to [1, 0, 0, 0]
                assert results[0].record.get_value("name") == "Vector 0"
            
            # Cleanup
            for record_id in record_ids:
                await db.delete(record_id)
        finally:
            await db.close()

    async def test_async_batch_embed_and_store(self, vector_test_db):
        """Test batch embedding and storage (mock embedding function)."""
        from dataknobs_data.backends.postgres import AsyncPostgresDatabase
        
        db = AsyncPostgresDatabase(vector_test_db)
        await db.connect()
        
        try:
            # Create test records without vectors
            records = []
            for i in range(5):
                record = Record({
                    "title": f"Document {i}",
                    "content": f"This is the content of document {i}",
                })
                records.append(record)
            
            # Mock embedding function
            async def mock_embed(texts: list[str]) -> np.ndarray:
                # Return random embeddings for testing
                return np.random.rand(len(texts), 4).astype(np.float32)
            
            # This should use the bulk_embed_and_store method
            ids = await db.bulk_embed_and_store(
                records=records,
                text_field="content",
                vector_field="content_vector",
                embedding_fn=mock_embed,
                batch_size=2,
                model_name="mock-model",
                model_version="1.0"
            )
            
            assert len(ids) == 5
            
            # Verify vectors were added
            for record_id in ids:
                retrieved = await db.read(record_id)
                assert retrieved is not None
                assert "content_vector" in retrieved.fields
                assert retrieved.fields["content_vector"].metadata.get("source_field") == "content"
                assert retrieved.fields["content_vector"].metadata.get("model", {}).get("name") == "mock-model"
            
            # Cleanup
            for record_id in ids:
                await db.delete(record_id)
        finally:
            await db.close()

    async def test_vector_index_creation(self, vector_test_db):
        """Test creating vector indices in PostgreSQL."""
        from dataknobs_data.backends.postgres import AsyncPostgresDatabase
        import asyncpg

        db = AsyncPostgresDatabase(vector_test_db)
        await db.connect()

        try:
            # Create some records with vectors first
            for i in range(100):
                record = Record({"index": i})
                record.fields["embedding"] = VectorField(
                    np.random.rand(16).astype(np.float32),
                    name="embedding"
                )
                await db.create(record)

            # Create vector index
            try:
                success = await db.create_vector_index(
                    vector_field="embedding",
                    dimensions=16,
                    metric=DistanceMetric.COSINE,
                    index_type="ivfflat",
                    lists=10
                )

                assert success is True

                # Get index stats
                stats = await db.get_vector_index_stats("embedding")
                assert stats["field"] == "embedding"
                assert stats.get("indexed") is True or stats.get("vector_count", 0) > 0
            except asyncpg.exceptions.InsufficientPrivilegeError as e:
                if "_fsm" in str(e):
                    pytest.skip(
                        "Skipping due to PostgreSQL FSM file permission issue in Docker. "
                        "This is a known issue with pgvector index creation in certain "
                        "containerized environments."
                    )
                else:
                    raise
        finally:
            await db.close()
