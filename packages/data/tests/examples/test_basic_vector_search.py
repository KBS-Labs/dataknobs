"""Tests for the basic vector search example."""

import os
import pytest
import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock
from typing import List, Dict, Any

# Skip all tests if PostgreSQL is not available
pytestmark = pytest.mark.skipif(
    not os.environ.get("TEST_POSTGRES", "").lower() == "true",
    reason="Vector search tests require TEST_POSTGRES=true and a running PostgreSQL instance with pgvector"
)

# Add examples to path
examples_path = Path(__file__).parent.parent.parent / "examples"
sys.path.insert(0, str(examples_path))

from basic_vector_search import VectorSearchExample
from dataknobs_data import Record, VectorField


class MockEmbeddingModel:
    """Mock embedding model for testing."""
    
    def encode(self, text: str):
        """Generate deterministic fake embeddings based on text hash."""
        import numpy as np
        # Create a simple deterministic embedding based on text
        hash_val = hash(text) % 1000
        # Return 384-dimensional vector (matching all-MiniLM-L6-v2)
        embedding = [float((hash_val + i) % 100) / 100.0 for i in range(384)]
        return np.array(embedding)


@pytest.fixture
def mock_embedding_model():
    """Provide a mock embedding model."""
    return MockEmbeddingModel()


@pytest.fixture
async def vector_example(mock_embedding_model):
    """Create a VectorSearchExample instance with mocked model."""
    example = VectorSearchExample(verbose=False)
    
    # Mock the SentenceTransformer
    with patch('basic_vector_search.SentenceTransformer') as mock_st:
        mock_st.return_value = mock_embedding_model
        example.load_model()
    
    yield example
    
    # Cleanup
    if example.db:
        await example.cleanup()


class TestVectorSearchExample:
    """Test cases for VectorSearchExample class."""
    
    def test_initialization(self):
        """Test VectorSearchExample initialization."""
        example = VectorSearchExample(verbose=False)
        assert example.verbose is False
        assert example.model is None
        assert example.db is None
        assert example.model_name == 'all-MiniLM-L6-v2'
    
    def test_log_verbose(self, capsys):
        """Test logging in verbose mode."""
        example = VectorSearchExample(verbose=True)
        example.log("Test message")
        captured = capsys.readouterr()
        assert "Test message" in captured.out
    
    def test_log_silent(self, capsys):
        """Test logging in silent mode."""
        example = VectorSearchExample(verbose=False)
        example.log("Test message")
        captured = capsys.readouterr()
        assert captured.out == ""
    
    def test_generate_embedding(self, mock_embedding_model):
        """Test embedding generation."""
        example = VectorSearchExample(verbose=False)
        
        with patch('basic_vector_search.SentenceTransformer') as mock_st:
            mock_st.return_value = mock_embedding_model
            
            embedding = example.generate_embedding("test text")
            
            assert isinstance(embedding, list)
            assert len(embedding) == 384
            assert all(isinstance(x, float) for x in embedding)
    
    def test_get_sample_documents(self):
        """Test sample documents generation."""
        example = VectorSearchExample(verbose=False)
        docs = example.get_sample_documents()
        
        assert len(docs) == 6
        assert all('title' in doc for doc in docs)
        assert all('content' in doc for doc in docs)
        assert all('category' in doc for doc in docs)
        assert all('level' in doc for doc in docs)
        
        # Check categories
        categories = {doc['category'] for doc in docs}
        assert 'AI' in categories
        assert 'Programming' in categories
    
    @pytest.mark.asyncio
    async def test_setup_database(self, vector_example):
        """Test database setup."""
        db = await vector_example.setup_database()
        
        assert db is not None
        assert vector_example.db is not None
        assert db == vector_example.db
    
    @pytest.mark.asyncio
    async def test_create_documents_with_embeddings(self, vector_example):
        """Test document creation with embeddings."""
        await vector_example.setup_database()
        
        # Create documents
        record_ids, records = await vector_example.create_documents_with_embeddings()
        
        assert len(record_ids) == 6
        assert len(records) == 6
        
        # Check that each record has an embedding
        for record in records:
            assert 'embedding' in record.fields
            assert isinstance(record.fields['embedding'], VectorField)
            assert record.fields['embedding'].dimensions == 384
    
    @pytest.mark.asyncio
    async def test_create_custom_documents(self, vector_example):
        """Test creating custom documents."""
        await vector_example.setup_database()
        
        custom_docs = [
            {
                "title": "Custom Document",
                "content": "This is a custom test document.",
                "category": "Test",
                "level": "basic"
            }
        ]
        
        record_ids, records = await vector_example.create_documents_with_embeddings(custom_docs)
        
        assert len(record_ids) == 1
        assert records[0].data['title'] == "Custom Document"
    
    @pytest.mark.asyncio
    async def test_perform_vector_search(self, vector_example):
        """Test vector similarity search."""
        await vector_example.setup_database()
        await vector_example.create_documents_with_embeddings()
        
        # Perform search
        results = await vector_example.perform_vector_search("machine learning", k=3)
        
        assert len(results) <= 3
        assert all(hasattr(r, 'record') for r in results)
        assert all(hasattr(r, 'score') for r in results)
        
        # Scores should be in descending order
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)
    
    @pytest.mark.asyncio
    async def test_perform_filtered_search(self, vector_example):
        """Test filtered vector search."""
        await vector_example.setup_database()
        await vector_example.create_documents_with_embeddings()
        
        # Search with filter
        results = await vector_example.perform_filtered_search(
            "neural networks",
            filter_category="AI",
            k=2
        )
        
        assert len(results) <= 2
        # All results should be from AI category
        assert all(r.record['category'] == 'AI' for r in results)
    
    @pytest.mark.asyncio
    async def test_error_without_database(self):
        """Test that operations fail without database setup."""
        example = VectorSearchExample(verbose=False)
        
        with pytest.raises(RuntimeError, match="Database not initialized"):
            await example.create_documents_with_embeddings()
        
        with pytest.raises(RuntimeError, match="Database not initialized"):
            await example.perform_vector_search("test")
        
        with pytest.raises(RuntimeError, match="Database not initialized"):
            await example.perform_filtered_search("test", "AI")
    
    @pytest.mark.asyncio
    async def test_cleanup(self, vector_example):
        """Test cleanup functionality."""
        await vector_example.setup_database()
        assert vector_example.db is not None
        
        await vector_example.cleanup()
        # After cleanup, db should be closed (we can't easily test connection state)
        # but at least cleanup should not raise an error
    
    @pytest.mark.asyncio
    async def test_full_workflow(self, vector_example):
        """Test the complete workflow."""
        # Setup
        await vector_example.setup_database()
        
        # Create documents
        record_ids, records = await vector_example.create_documents_with_embeddings()
        assert len(record_ids) == 6
        
        # Search
        results = await vector_example.perform_vector_search("deep learning AI", k=3)
        assert len(results) > 0
        
        # Filtered search
        filtered = await vector_example.perform_filtered_search("programming", "Programming", k=2)
        assert all(r.record['category'] == 'Programming' for r in filtered)
        
        # Cleanup
        await vector_example.cleanup()


class TestIntegrationWithRealModel:
    """Integration tests with real model (optional, requires sentence-transformers)."""
    
    @pytest.mark.asyncio
    async def test_with_real_model(self):
        """Test with actual SentenceTransformer model."""
        # Skip if sentence_transformers is not available
        try:
            import sentence_transformers
        except ImportError:
            pytest.skip("sentence-transformers not installed")
        
        example = VectorSearchExample(model_name='all-MiniLM-L6-v2', verbose=False)
        
        try:
            # This will use the real model
            await example.setup_database()
            
            # Create just a few documents to keep test fast
            small_docs = [
                {
                    "title": "Machine Learning",
                    "content": "ML is about learning from data.",
                    "category": "AI",
                    "level": "basic"
                },
                {
                    "title": "Web Development",
                    "content": "Building websites with HTML and CSS.",
                    "category": "Web",
                    "level": "basic"
                }
            ]
            
            record_ids, records = await example.create_documents_with_embeddings(small_docs)
            assert len(record_ids) == 2
            
            # Search should return results with valid scores
            results = await example.perform_vector_search("machine learning algorithms", k=2)
            assert len(results) == 2
            # Verify we got valid results with proper structure
            assert all(hasattr(r, 'score') for r in results)
            assert all(0 <= r.score <= 1 for r in results)
            assert all(r.record.get('title') is not None for r in results)
            
        finally:
            await example.cleanup()


@pytest.mark.asyncio
async def test_example_main_function():
    """Test the main function runs without errors."""
    # Mock the VectorSearchExample to avoid loading real model
    with patch('basic_vector_search.VectorSearchExample') as MockExample:
        mock_instance = AsyncMock()
        MockExample.return_value = mock_instance
        
        # Mock all the methods
        mock_instance.setup_database = AsyncMock()
        mock_instance.create_documents_with_embeddings = AsyncMock(
            return_value=(["id1", "id2"], [MagicMock(), MagicMock()])
        )
        mock_instance.perform_vector_search = AsyncMock(return_value=[])
        mock_instance.perform_filtered_search = AsyncMock(return_value=[])
        mock_instance.cleanup = AsyncMock()
        mock_instance.db = MagicMock()
        mock_instance.db.vector_search = AsyncMock(return_value=[])
        mock_instance.db.find = AsyncMock(return_value=[])
        mock_instance.generate_embedding = MagicMock(return_value=[0.1] * 384)
        mock_instance.log = MagicMock()
        
        # Import and run main
        from basic_vector_search import main
        await main()
        
        # Verify key methods were called
        mock_instance.setup_database.assert_called_once()
        mock_instance.create_documents_with_embeddings.assert_called_once()
        mock_instance.perform_vector_search.assert_called()
        mock_instance.cleanup.assert_called_once()