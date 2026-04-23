"""Tests for specialized vector stores."""

import asyncio
import os
import pickle
import tempfile
from datetime import datetime, timezone
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

    @pytest.mark.asyncio
    async def test_metadata_fields_empty(self, store):
        """Test metadata_fields returns empty set for empty store."""
        await store.initialize()
        fields = await store.metadata_fields()
        assert fields == set()

    @pytest.mark.asyncio
    async def test_metadata_fields_with_data(self, store):
        """Test metadata_fields returns union of all field names."""
        await store.initialize()
        vectors = np.random.rand(3, 128).astype(np.float32)
        metadata = [
            {"headings": ["A"], "heading_levels": [1], "source": "doc.md"},
            {"headings": ["B"], "category": "test"},
            {"heading_levels": [2], "author": "alice"},
        ]
        await store.add_vectors(vectors, metadata=metadata)
        fields = await store.metadata_fields()
        assert fields == {"headings", "heading_levels", "source", "category", "author"}

    @pytest.mark.asyncio
    async def test_metadata_fields_after_delete(self, store):
        """Test metadata_fields reflects current state after deletion."""
        await store.initialize()
        vectors = np.random.rand(2, 128).astype(np.float32)
        ids = ["v1", "v2"]
        metadata = [
            {"field_a": 1},
            {"field_b": 2},
        ]
        await store.add_vectors(vectors, ids=ids, metadata=metadata)
        await store.delete_vectors(["v1"])
        fields = await store.metadata_fields()
        assert "field_a" not in fields
        assert "field_b" in fields


class TestMemoryVectorStoreTimestamps:
    """Phase 4: timestamp tracking + include_timestamps exposure on MVS."""

    @pytest.fixture
    def store(self):
        """Default config: 4 dims, cosine, ISO format timestamps."""
        return MemoryVectorStore({"dimensions": 4, "metric": "cosine"})

    @pytest.mark.asyncio
    async def test_add_vectors_tracks_timestamps(self, store):
        """Fresh add populates self.timestamps; created == updated."""
        await store.initialize()
        vec = np.random.rand(4).astype(np.float32)
        await store.add_vectors([vec], ids=["t1"], metadata=[{"k": "v"}])

        assert "t1" in store.timestamps
        created, updated = store.timestamps["t1"]
        assert isinstance(created, datetime)
        assert isinstance(updated, datetime)
        assert created.tzinfo is timezone.utc
        # Fresh add: created == updated (same now() call).
        assert created == updated

    @pytest.mark.asyncio
    async def test_upsert_refreshes_updated_only(self, store):
        """Re-add same ID: created preserved, updated advances."""
        await store.initialize()
        vec1 = np.random.rand(4).astype(np.float32)
        vec2 = np.random.rand(4).astype(np.float32)

        await store.add_vectors([vec1], ids=["t1"])
        created_1, updated_1 = store.timestamps["t1"]

        await asyncio.sleep(0.01)

        await store.add_vectors([vec2], ids=["t1"])
        created_2, updated_2 = store.timestamps["t1"]

        assert created_2 == created_1, "created must be preserved on upsert"
        assert updated_2 > updated_1, "updated must advance on upsert"

    @pytest.mark.asyncio
    async def test_update_metadata_refreshes_updated(self, store):
        """update_metadata advances updated, preserves created."""
        await store.initialize()
        vec = np.random.rand(4).astype(np.float32)
        await store.add_vectors([vec], ids=["t1"], metadata=[{"k": "v"}])
        created_1, updated_1 = store.timestamps["t1"]

        await asyncio.sleep(0.01)

        await store.update_metadata(["t1"], [{"k": "v2"}])
        created_2, updated_2 = store.timestamps["t1"]

        assert created_2 == created_1, "created preserved on update_metadata"
        assert updated_2 > updated_1, "updated advances on update_metadata"

    @pytest.mark.asyncio
    async def test_delete_vectors_removes_timestamp(self, store):
        """delete_vectors removes the timestamp entry."""
        await store.initialize()
        vec = np.random.rand(4).astype(np.float32)
        await store.add_vectors([vec], ids=["t1"])
        assert "t1" in store.timestamps

        await store.delete_vectors(["t1"])
        assert "t1" not in store.timestamps

    @pytest.mark.asyncio
    async def test_clear_removes_timestamps(self, store):
        """clear() empties the timestamps dict."""
        await store.initialize()
        vectors = np.random.rand(3, 4).astype(np.float32)
        await store.add_vectors(vectors, ids=["a", "b", "c"])
        assert len(store.timestamps) == 3

        await store.clear()
        assert len(store.timestamps) == 0

    @pytest.mark.asyncio
    async def test_save_load_round_trip_preserves_timestamps(self, tmp_path):
        """Pickle persistence preserves timestamps across save/load."""
        persist_path = str(tmp_path / "mvs.pkl")
        store = MemoryVectorStore({
            "dimensions": 4,
            "metric": "cosine",
            "persist_path": persist_path,
        })
        await store.initialize()

        vec = np.random.rand(4).astype(np.float32)
        await store.add_vectors([vec], ids=["t1"], metadata=[{"k": "v"}])
        original_created, original_updated = store.timestamps["t1"]

        await store.save()

        # Create a fresh store pointing at the same file and load.
        new_store = MemoryVectorStore({
            "dimensions": 4,
            "metric": "cosine",
            "persist_path": persist_path,
        })
        await new_store.initialize()

        assert "t1" in new_store.timestamps
        loaded_created, loaded_updated = new_store.timestamps["t1"]
        assert loaded_created == original_created
        assert loaded_updated == original_updated

    @pytest.mark.asyncio
    async def test_get_vectors_include_timestamps(self, store):
        """include_timestamps=True injects _created_at / _updated_at."""
        await store.initialize()
        vec = np.random.rand(4).astype(np.float32)
        await store.add_vectors([vec], ids=["t1"], metadata=[{"k": "v"}])

        results = await store.get_vectors(["t1"], include_timestamps=True)
        _vector, meta = results[0]

        assert meta is not None
        assert meta["k"] == "v"
        assert "_created_at" in meta
        assert "_updated_at" in meta
        # Default ISO format -> string.
        assert isinstance(meta["_created_at"], str)
        assert isinstance(meta["_updated_at"], str)

        # Default path unchanged — no timestamp keys.
        default_results = await store.get_vectors(["t1"])
        _, default_meta = default_results[0]
        assert "_created_at" not in default_meta
        assert "_updated_at" not in default_meta

    @pytest.mark.asyncio
    async def test_search_include_timestamps(self, store):
        """Same include_timestamps semantics on search."""
        await store.initialize()
        vectors = np.random.rand(3, 4).astype(np.float32)
        await store.add_vectors(
            vectors,
            ids=["a", "b", "c"],
            metadata=[{"i": 0}, {"i": 1}, {"i": 2}],
        )

        query = vectors[0]
        results = await store.search(query, k=3, include_timestamps=True)

        assert len(results) == 3
        for _id, _score, meta in results:
            assert meta is not None
            assert "_created_at" in meta
            assert "_updated_at" in meta
            assert isinstance(meta["_created_at"], str)

    @pytest.mark.asyncio
    async def test_legacy_pickle_load_has_empty_timestamps(self, tmp_path):
        """Pre-Item-36 pickle files load cleanly; legacy rows have no tracked ts."""
        persist_path = str(tmp_path / "legacy.pkl")
        # Write a legacy pickle file (no "timestamps" key).
        legacy = {
            "vectors": {"legacy-id": [0.1, 0.2, 0.3, 0.4]},
            "metadata_store": {"legacy-id": {"k": "v"}},
            "config": {"dimensions": 4, "metric": "cosine"},
        }
        with open(persist_path, "wb") as f:
            pickle.dump(legacy, f)

        store = MemoryVectorStore({
            "dimensions": 4,
            "metric": "cosine",
            "persist_path": persist_path,
        })
        await store.initialize()

        # Vector + metadata loaded correctly.
        assert "legacy-id" in store.vectors
        # No tracked timestamp for legacy rows.
        assert "legacy-id" not in store.timestamps

        # include_timestamps=True surfaces keys-present-with-None values
        # (analogous to pgvector pre-migration NULL rows).
        results = await store.get_vectors(
            ["legacy-id"], include_timestamps=True
        )
        _vector, meta = results[0]
        assert meta is not None
        assert meta["k"] == "v"
        assert "_created_at" in meta
        assert "_updated_at" in meta
        assert meta["_created_at"] is None
        assert meta["_updated_at"] is None


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

            @pytest.mark.asyncio
            async def test_metadata_fields_empty(self, store):
                """Test metadata_fields returns empty set for empty store."""
                await store.initialize()
                fields = await store.metadata_fields()
                assert fields == set()

            @pytest.mark.asyncio
            async def test_metadata_fields_with_data(self, store):
                """Test metadata_fields returns union of all field names."""
                await store.initialize()
                vectors = np.random.rand(3, 128).astype(np.float32)
                ids = ["f1", "f2", "f3"]
                metadata = [
                    {"headings": ["A"], "source": "doc.md"},
                    {"headings": ["B"], "category": "test"},
                    {"author": "alice"},
                ]
                await store.add_vectors(vectors, ids=ids, metadata=metadata)
                fields = await store.metadata_fields()
                assert fields == {"headings", "source", "category", "author"}

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

            @pytest.mark.asyncio
            async def test_metadata_fields_empty(self, store):
                """Test metadata_fields returns empty set for empty collection."""
                await store.initialize()
                fields = await store.metadata_fields()
                assert fields == set()

            @pytest.mark.asyncio
            async def test_metadata_fields_with_data(self, store):
                """Test metadata_fields returns union of all field names."""
                await store.initialize()
                vectors = np.random.rand(3, 384).astype(np.float32)
                ids = [str(uuid4()) for _ in range(3)]
                metadata = [
                    {"headings": "A", "source": "doc.md"},
                    {"headings": "B", "category": "test"},
                    {"author": "alice"},
                ]
                await store.add_vectors(vectors, ids=ids, metadata=metadata)
                fields = await store.metadata_fields()
                assert fields == {"headings", "source", "category", "author"}

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