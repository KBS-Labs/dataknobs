"""Tests for specialized vector stores."""

import os
import tempfile
from typing import Any
from uuid import uuid4

import numpy as np
import pytest

from dataknobs_data.vector.stores import VectorStore, VectorStoreFactory
from dataknobs_data.vector.stores.memory import MemoryVectorStore
from dataknobs_data.vector.types import DistanceMetric


class TestMemoryVectorStore:
    """Test the in-memory vector store."""
    
    @pytest.fixture
    def store(self):
        """Create a memory vector store."""
        config = {
            "dimensions": 128,
            "metric": "cosine",
        }
        return MemoryVectorStore(config)
    
    @pytest.mark.asyncio
    async def test_initialize(self, store):
        """Test store initialization."""
        await store.initialize()
        assert store._initialized
        assert store.dimensions == 128
        assert store.metric == DistanceMetric.COSINE
    
    @pytest.mark.asyncio
    async def test_add_vectors(self, store):
        """Test adding vectors."""
        await store.initialize()
        
        # Create test vectors
        vectors = np.random.rand(5, 128).astype(np.float32)
        ids = [str(uuid4()) for _ in range(5)]
        metadata = [{"index": i} for i in range(5)]
        
        # Add vectors
        result_ids = await store.add_vectors(vectors, ids=ids, metadata=metadata)
        
        assert result_ids == ids
        assert len(store.vectors) == 5
        assert len(store.metadata_store) == 5
    
    @pytest.mark.asyncio
    async def test_get_vectors(self, store):
        """Test retrieving vectors."""
        await store.initialize()
        
        # Add vectors
        vector = np.random.rand(128).astype(np.float32)
        vector_id = str(uuid4())
        metadata = {"test": "value"}
        
        await store.add_vectors(vector, ids=[vector_id], metadata=[metadata])
        
        # Get vector
        results = await store.get_vectors([vector_id], include_metadata=True)
        
        assert len(results) == 1
        retrieved_vector, retrieved_metadata = results[0]
        assert np.allclose(retrieved_vector, vector.reshape(1, -1)[0])
        assert retrieved_metadata == metadata
    
    @pytest.mark.asyncio
    async def test_delete_vectors(self, store):
        """Test deleting vectors."""
        await store.initialize()
        
        # Add vectors
        vectors = np.random.rand(3, 128).astype(np.float32)
        ids = [str(uuid4()) for _ in range(3)]
        
        await store.add_vectors(vectors, ids=ids)
        assert len(store.vectors) == 3
        
        # Delete vectors
        deleted = await store.delete_vectors(ids[:2])
        assert deleted == 2
        assert len(store.vectors) == 1
        assert ids[2] in store.vectors
    
    @pytest.mark.asyncio
    async def test_search_cosine(self, store):
        """Test vector search with cosine similarity."""
        await store.initialize()
        
        # Add known vectors
        vectors = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.0],
        ], dtype=np.float32)
        
        # Normalize for cosine similarity
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / norms
        
        ids = ["x", "y", "z", "xy"]
        metadata = [{"axis": name} for name in ids]
        
        config = {
            "dimensions": 3,
            "metric": "cosine",
        }
        store = MemoryVectorStore(config)
        await store.initialize()
        await store.add_vectors(vectors, ids=ids, metadata=metadata)
        
        # Search for vector similar to x-axis
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        results = await store.search(query, k=2)
        
        assert len(results) == 2
        assert results[0][0] == "x"  # Exact match
        assert results[0][1] > 0.99  # High similarity
    
    @pytest.mark.asyncio
    async def test_search_euclidean(self, store):
        """Test vector search with Euclidean distance."""
        config = {
            "dimensions": 3,
            "metric": "euclidean",
        }
        store = MemoryVectorStore(config)
        await store.initialize()
        
        # Add vectors
        vectors = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ], dtype=np.float32)
        
        ids = ["origin", "x", "y", "xy"]
        await store.add_vectors(vectors, ids=ids)
        
        # Search for vector near origin
        query = np.array([0.1, 0.1, 0.0], dtype=np.float32)
        results = await store.search(query, k=2)
        
        assert len(results) == 2
        assert results[0][0] == "origin"  # Closest to origin
    
    @pytest.mark.asyncio
    async def test_search_with_filter(self, store):
        """Test vector search with metadata filter."""
        await store.initialize()
        
        # Add vectors with metadata
        vectors = np.random.rand(10, 128).astype(np.float32)
        ids = [str(i) for i in range(10)]
        metadata = [{"category": "A" if i < 5 else "B", "index": i} for i in range(10)]
        
        await store.add_vectors(vectors, ids=ids, metadata=metadata)
        
        # Search with filter
        query = vectors[0]
        results = await store.search(
            query, k=3, filter={"category": "B"}
        )
        
        # Should only return vectors from category B
        assert all(int(result[0]) >= 5 for result in results)
    
    @pytest.mark.asyncio
    async def test_update_metadata(self, store):
        """Test updating vector metadata."""
        await store.initialize()
        
        # Add vectors
        vectors = np.random.rand(3, 128).astype(np.float32)
        ids = ["a", "b", "c"]
        metadata = [{"version": 1} for _ in range(3)]
        
        await store.add_vectors(vectors, ids=ids, metadata=metadata)
        
        # Update metadata
        new_metadata = [{"version": 2, "updated": True} for _ in range(2)]
        updated = await store.update_metadata(ids[:2], new_metadata)
        
        assert updated == 2
        assert store.metadata_store["a"]["version"] == 2
        assert store.metadata_store["c"]["version"] == 1
    
    @pytest.mark.asyncio
    async def test_count(self, store):
        """Test counting vectors."""
        await store.initialize()
        
        assert await store.count() == 0
        
        # Add vectors
        vectors = np.random.rand(5, 128).astype(np.float32)
        metadata = [{"type": "A" if i < 3 else "B"} for i in range(5)]
        await store.add_vectors(vectors, metadata=metadata)
        
        assert await store.count() == 5
        assert await store.count(filter={"type": "A"}) == 3
        assert await store.count(filter={"type": "B"}) == 2
    
    @pytest.mark.asyncio
    async def test_clear(self, store):
        """Test clearing all vectors."""
        await store.initialize()
        
        # Add vectors
        vectors = np.random.rand(5, 128).astype(np.float32)
        await store.add_vectors(vectors)
        assert await store.count() == 5
        
        # Clear
        await store.clear()
        assert await store.count() == 0
        assert len(store.vectors) == 0
        assert len(store.metadata_store) == 0


# Conditional test classes for optional dependencies
try:
    from dataknobs_data.vector.stores.faiss import FaissVectorStore, FAISS_AVAILABLE
    
    if FAISS_AVAILABLE:
        class TestFaissVectorStore:
            """Test the Faiss vector store."""
            
            @pytest.fixture
            def store(self):
                """Create a Faiss vector store."""
                config = {
                    "dimensions": 128,
                    "metric": "cosine",
                    "index_type": "flat",
                }
                return FaissVectorStore(config)
            
            @pytest.fixture
            def persistent_store(self, tmp_path):
                """Create a persistent Faiss vector store."""
                config = {
                    "dimensions": 128,
                    "metric": "euclidean",
                    "index_type": "flat",
                    "persist_path": str(tmp_path / "faiss.index"),
                }
                return FaissVectorStore(config)
            
            @pytest.mark.asyncio
            async def test_initialize(self, store):
                """Test Faiss store initialization."""
                await store.initialize()
                assert store._initialized
                assert store.index is not None
                assert store.dimensions == 128
            
            @pytest.mark.asyncio
            async def test_add_and_search(self, store):
                """Test adding and searching vectors."""
                await store.initialize()
                
                # Add vectors
                vectors = np.random.rand(100, 128).astype(np.float32)
                ids = [str(i) for i in range(100)]
                metadata = [{"index": i} for i in range(100)]
                
                await store.add_vectors(vectors, ids=ids, metadata=metadata)
                
                # Search
                query = vectors[0]
                results = await store.search(query, k=5)
                
                assert len(results) == 5
                assert results[0][0] == "0"  # First result should be exact match
                assert results[0][1] > 0.99  # High similarity
            
            @pytest.mark.asyncio
            async def test_ivf_index(self):
                """Test IVF index type."""
                config = {
                    "dimensions": 64,
                    "metric": "euclidean",
                    "index_type": "ivfflat",
                    "index_params": {"nlist": 10},
                }
                store = FaissVectorStore(config)
                await store.initialize()
                
                # Add enough vectors to train IVF
                vectors = np.random.rand(100, 64).astype(np.float32)
                await store.add_vectors(vectors)
                
                # Search
                query = vectors[0]
                results = await store.search(query, k=5)
                assert len(results) <= 5
            
            @pytest.mark.asyncio
            async def test_persistence(self, persistent_store):
                """Test saving and loading index."""
                await persistent_store.initialize()
                
                # Add vectors
                vectors = np.random.rand(10, 128).astype(np.float32)
                ids = [str(i) for i in range(10)]
                metadata = [{"value": i} for i in range(10)]
                
                await persistent_store.add_vectors(vectors, ids=ids, metadata=metadata)
                
                # Save
                await persistent_store.save()
                
                # Create new store and load
                new_store = FaissVectorStore({
                    "dimensions": 128,
                    "metric": "euclidean",
                    "persist_path": persistent_store.persist_path,
                })
                await new_store.initialize()
                
                # Verify loaded data
                assert new_store.index.ntotal == 10
                assert len(new_store.id_map) == 10
                assert len(new_store.metadata_store) == 10
                
                # Search should work
                results = await new_store.search(vectors[0], k=1)
                assert results[0][0] == "0"

except ImportError:
    pass


try:
    from dataknobs_data.vector.stores.chroma import ChromaVectorStore, CHROMA_AVAILABLE
    
    if CHROMA_AVAILABLE:
        class TestChromaVectorStore:
            """Test the Chroma vector store."""
            
            @pytest.fixture
            def store(self):
                """Create a Chroma vector store."""
                config = {
                    "dimensions": 384,
                    "metric": "cosine",
                    "collection_name": "test_vectors",
                }
                return ChromaVectorStore(config)
            
            @pytest.mark.asyncio
            async def test_initialize(self, store):
                """Test Chroma store initialization."""
                await store.initialize()
                assert store._initialized
                assert store.client is not None
                assert store.collection is not None
            
            @pytest.mark.asyncio
            async def test_add_and_search(self, store):
                """Test adding and searching vectors."""
                await store.initialize()
                
                # Add vectors
                vectors = np.random.rand(50, 384).astype(np.float32)
                ids = [str(uuid4()) for _ in range(50)]
                metadata = [{"index": i} for i in range(50)]
                
                await store.add_vectors(vectors, ids=ids, metadata=metadata)
                
                # Search
                query = vectors[0]
                results = await store.search(query, k=5)
                
                assert len(results) <= 5
                assert results[0][0] == ids[0]  # First result should be exact match
            
            @pytest.mark.asyncio
            async def test_metadata_filtering(self, store):
                """Test searching with metadata filters."""
                await store.initialize()
                
                # Add vectors with categories
                vectors = np.random.rand(20, 384).astype(np.float32)
                ids = [str(i) for i in range(20)]
                metadata = [
                    {"category": "A" if i < 10 else "B", "value": i}
                    for i in range(20)
                ]
                
                await store.add_vectors(vectors, ids=ids, metadata=metadata)
                
                # Search with filter
                query = vectors[0]
                results = await store.search(
                    query, k=5, filter={"category": "B"}
                )
                
                # Should only return category B vectors
                for _, _, meta in results:
                    assert meta["category"] == "B"
            
            @pytest.mark.asyncio
            async def test_delete_vectors(self, store):
                """Test deleting vectors."""
                await store.initialize()
                
                # Add vectors
                vectors = np.random.rand(5, 384).astype(np.float32)
                ids = [str(i) for i in range(5)]
                
                await store.add_vectors(vectors, ids=ids)
                
                # Delete some
                deleted = await store.delete_vectors(ids[:3])
                assert deleted == 3
                
                # Verify remaining
                remaining = await store.get_vectors(ids)
                assert remaining[0][0] is None  # Deleted
                assert remaining[3][0] is not None  # Still exists

except ImportError:
    pass


# Check what's available for the factory tests
try:
    import faiss
    FAISS_AVAILABLE_FOR_FACTORY = True
except ImportError:
    FAISS_AVAILABLE_FOR_FACTORY = False

try:
    import chromadb
    CHROMA_AVAILABLE_FOR_FACTORY = True
except ImportError:
    CHROMA_AVAILABLE_FOR_FACTORY = False


class TestVectorStoreFactory:
    """Test the vector store factory."""
    
    def test_create_memory_store(self):
        """Test creating memory vector store."""
        factory = VectorStoreFactory()
        store = factory.create(
            backend="memory",
            dimensions=128,
            metric="cosine"
        )
        assert isinstance(store, MemoryVectorStore)
        assert store.dimensions == 128
    
    def test_create_faiss_store(self):
        """Test creating Faiss vector store."""
        factory = VectorStoreFactory()
        if not FAISS_AVAILABLE_FOR_FACTORY:
            # Test that it raises the appropriate error when not installed
            with pytest.raises(ValueError, match="Faiss backend requires faiss-cpu"):
                factory.create(
                    backend="faiss",
                    dimensions=256,
                    index_type="flat"
                )
        else:
            # Test normal creation when installed
            from dataknobs_data.vector.stores.faiss import FaissVectorStore
            store = factory.create(
                backend="faiss",
                dimensions=256,
                index_type="flat"
            )
            assert isinstance(store, FaissVectorStore)
            assert store.dimensions == 256
    
    def test_create_chroma_store(self):
        """Test creating Chroma vector store."""
        factory = VectorStoreFactory()
        if not CHROMA_AVAILABLE_FOR_FACTORY:
            # Test that it raises the appropriate error when not installed
            with pytest.raises(ValueError, match="Chroma backend requires chromadb"):
                factory.create(
                    backend="chroma",
                    collection_name="test"
                )
        else:
            # Test normal creation when installed
            from dataknobs_data.vector.stores.chroma import ChromaVectorStore
            store = factory.create(
                backend="chroma",
                collection_name="test"
            )
            assert isinstance(store, ChromaVectorStore)
    
    def test_unknown_backend(self):
        """Test creating store with unknown backend."""
        factory = VectorStoreFactory()
        with pytest.raises(ValueError, match="Unknown backend"):
            factory.create(backend="unknown")
    
    def test_get_backend_info(self):
        """Test getting backend information."""
        factory = VectorStoreFactory()
        
        info = factory.get_backend_info("memory")
        assert "description" in info
        assert info["persistent"] is False
        
        info = factory.get_backend_info("faiss")
        assert "description" in info
        assert info["persistent"] is True
        assert "pip install" in info["requires_install"]