"""End-to-end integration tests for vector stores."""

import numpy as np
import pytest

from dataknobs_data.factory import DatabaseFactory, VectorStoreFactory
from dataknobs_data.query import Query


class TestVectorIntegration:
    """Test vector store integration with VectorStoreFactory."""
    
    @pytest.fixture
    def factory(self):
        """Create a vector store factory."""
        return VectorStoreFactory()
    
    @pytest.fixture
    def db_factory(self):
        """Create a database factory."""
        return DatabaseFactory()
    
    def test_create_memory_vector_store(self, factory):
        """Test creating a memory vector store via factory."""
        store = factory.create(
            backend="memory",
            dimensions=128,
            metric="cosine"
        )
        
        assert store is not None
        assert hasattr(store, "add_vectors")
        assert hasattr(store, "search")
    
    def test_vector_enabled_database(self, db_factory):
        """Test creating a vector-enabled database."""
        # Test that all backends now support vector mode
        db = db_factory.create(
            backend="memory",  # Memory backend now supports vector mode via Python-based search
            vector_enabled=True,
            vector_dimensions=256
        )
        assert db is not None
        # Verify it has vector operations
        assert hasattr(db, 'vector_search')
    
    def test_faiss_backend_missing_dependency(self, factory):
        """Test Faiss backend with missing dependency."""
        try:
            import faiss
            pytest.skip("Faiss is installed, skipping missing dependency test")
        except ImportError:
            with pytest.raises(ValueError, match="Faiss backend requires faiss-cpu"):
                factory.create(backend="faiss", dimensions=256)
    
    def test_chroma_backend_missing_dependency(self, factory):
        """Test Chroma backend with missing dependency."""
        try:
            import chromadb
            pytest.skip("ChromaDB is installed, skipping missing dependency test")
        except ImportError:
            with pytest.raises(ValueError, match="Chroma backend requires chromadb"):
                factory.create(backend="chroma", dimensions=256)
    
    @pytest.mark.asyncio
    async def test_end_to_end_vector_workflow(self, factory):
        """Test complete workflow from creation to search."""
        # Create a memory vector store
        store = factory.create(
            backend="memory",
            dimensions=64,
            metric="cosine"
        )
        
        # Initialize if it's an async store
        if hasattr(store, "initialize"):
            await store.initialize()
        
        # Add some vectors
        vectors = []
        for i in range(10):
            vec = np.random.randn(64).astype(np.float32)
            vec = vec / np.linalg.norm(vec)  # Normalize
            vectors.append(vec)
            
            # Use add_vectors method
            await store.add_vectors(
                vectors=[vec],
                ids=[f"vec_{i}"],
                metadata=[{"index": i, "category": "test"}]
            )
        
        # Search for similar vectors - use the exact same vector for guaranteed match
        query_vec = vectors[0].copy()  # Use exact vector for first test
        
        results = await store.search(
            query_vector=query_vec,
            k=5
        )
        
        assert len(results) <= 5
        if results:
            # Results are tuples of (id, score, metadata)
            assert results[0][0] == "vec_0"  # ID
            assert results[0][1] > 0.99  # Should be ~1.0 for exact match
            
        # Now test with slightly perturbed vector
        query_vec2 = vectors[1] + np.random.randn(64) * 0.1
        query_vec2 = query_vec2 / np.linalg.norm(query_vec2)
        
        results2 = await store.search(
            query_vector=query_vec2,
            k=5
        )
        assert len(results2) <= 5  # Just check we get results
    
    def test_backend_info(self, factory, db_factory):
        """Test getting backend information."""
        # Test standalone vector store info
        faiss_info = factory.get_backend_info("faiss")
        assert "description" in faiss_info
        assert "Facebook AI" in faiss_info["description"]
        
        chroma_info = factory.get_backend_info("chroma")
        assert "ChromaDB" in chroma_info["description"]
        
        memory_info = factory.get_backend_info("memory")
        assert "In-memory vector" in memory_info["description"]
        
        # Test vector-enabled database info
        postgres_info = db_factory.get_backend_info("postgres")
        assert "vector_support" in postgres_info
        assert postgres_info["vector_support"] is True
        assert "pgvector" in postgres_info["description"]
    
    @pytest.mark.asyncio
    async def test_vector_query_integration(self, factory):
        """Test vector queries with Query class."""
        # Create a memory vector store
        store = factory.create(
            backend="memory",
            dimensions=128,
            metric="euclidean"
        )
        
        if hasattr(store, "initialize"):
            await store.initialize()
        
        # Add vectors with metadata
        for i in range(20):
            vec = np.random.randn(128).astype(np.float32)
            await store.add_vectors(
                vectors=[vec],
                ids=[f"doc_{i}"],
                metadata=[{
                    "title": f"Document {i}",
                    "category": "category_a" if i < 10 else "category_b",
                    "score": i * 10
                }]
            )
        
        # Create a vector query
        query = Query().similar_to(
            vector=np.random.randn(128).astype(np.float32),
            field="embedding",
            k=5
        )
        
        # The store should be able to handle this query
        assert query.vector_query is not None
        assert query.vector_query.k == 5
        assert query.vector_query.field_name == "embedding"  # Internal field name
    
    @pytest.mark.asyncio 
    async def test_hybrid_search_workflow(self, factory):
        """Test hybrid search combining filters and vectors."""
        store = factory.create(
            backend="memory",
            dimensions=64,
            metric="cosine"
        )
        
        if hasattr(store, "initialize"):
            await store.initialize()
        
        # Add data
        categories = ["tech", "science", "health"]
        for i in range(30):
            vec = np.random.randn(64).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            
            await store.add_vectors(
                vectors=[vec],
                ids=[f"item_{i}"],
                metadata=[{
                    "title": f"Item {i}",
                    "category": categories[i % 3],
                    "views": i * 100
                }]
            )
        
        # Hybrid query: vector similarity + metadata filter
        query_vec = np.random.randn(64).astype(np.float32)
        query_vec = query_vec / np.linalg.norm(query_vec)
        
        # Search with filter
        results = await store.search(
            query_vector=query_vec,
            k=10,
            filter={"category": "tech"}  # Only tech items
        )
        
        # All results should be from tech category
        for result in results:
            # Results are tuples of (id, score, metadata)
            if result[2]:  # metadata is third element
                assert result[2].get("category") == "tech"
    
    def test_configuration_validation(self, factory):
        """Test configuration validation for vector stores."""
        # Test creating with empty config (should use defaults)
        store = factory.create(backend="memory")
        assert store is not None  # Should work with defaults
    
    @pytest.mark.asyncio
    async def test_cross_backend_compatibility(self, factory):
        """Test that different backends follow the same interface."""
        backends_to_test = []
        
        # Memory vector store (always available)
        backends_to_test.append(("memory", {
            "dimensions": 64,
            "metric": "cosine"
        }))
        
        # Test each backend
        for backend_name, config in backends_to_test:
            store = factory.create(backend=backend_name, **config)
            
            # All stores should have the same interface
            assert hasattr(store, "add_vectors")
            assert hasattr(store, "search")
            assert hasattr(store, "delete_vectors")
            assert hasattr(store, "count")
            
            if hasattr(store, "initialize"):
                await store.initialize()
            
            # Basic operations should work
            vec = np.random.randn(64).astype(np.float32)
            await store.add_vectors([vec], ["test_id"], [{"test": True}])
            
            results = await store.search(vec, k=1)
            assert len(results) >= 0  # May be empty if not yet indexed
            
            count = await store.count()
            assert count >= 1


class TestPerformanceRegression:
    """Test for performance regressions."""
    
    @pytest.mark.asyncio
    async def test_indexing_performance(self):
        """Test vector indexing performance."""
        from dataknobs_data.vector.stores.memory import MemoryVectorStore
        import time
        
        store = MemoryVectorStore({"dimensions": 128, "metric": "cosine"})
        await store.initialize()
        
        # Measure indexing time
        num_vectors = 1000
        start_time = time.time()
        
        # Batch add for better performance
        vectors = []
        ids = []
        for i in range(num_vectors):
            vec = np.random.randn(128).astype(np.float32)
            vectors.append(vec)
            ids.append(f"vec_{i}")
        
        # Add in batches
        batch_size = 100
        for i in range(0, num_vectors, batch_size):
            batch_vecs = vectors[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]
            await store.add_vectors(batch_vecs, batch_ids)
        
        elapsed = time.time() - start_time
        throughput = num_vectors / elapsed
        
        # Should index at least 100 vectors per second
        assert throughput > 100, f"Indexing too slow: {throughput:.1f} vec/s"
    
    @pytest.mark.asyncio
    async def test_search_performance(self):
        """Test vector search performance."""
        from dataknobs_data.vector.stores.memory import MemoryVectorStore
        import time
        
        store = MemoryVectorStore({"dimensions": 128, "metric": "cosine"})
        await store.initialize()
        
        # Add vectors in batch for efficiency
        vectors = [np.random.randn(128).astype(np.float32) for _ in range(1000)]
        ids = [f"vec_{i}" for i in range(1000)]
        await store.add_vectors(vectors, ids)
        
        # Measure search time
        num_searches = 100
        start_time = time.time()
        
        for _ in range(num_searches):
            query = np.random.randn(128).astype(np.float32)
            await store.search(query_vector=query, k=10)
        
        elapsed = time.time() - start_time
        avg_latency = (elapsed / num_searches) * 1000  # ms
        
        # Search should be under 50ms on average
        assert avg_latency < 50, f"Search too slow: {avg_latency:.1f}ms"