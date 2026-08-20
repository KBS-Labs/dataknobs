"""Tests for the basic vector search example."""

import pytest
import sys
import zlib
from pathlib import Path
from unittest.mock import patch

from dataknobs_common.testing import requires_package
from dataknobs_data import VectorField

# Add examples to path
examples_path = Path(__file__).parent.parent.parent / "examples"
sys.path.insert(0, str(examples_path))

from basic_vector_search import VectorSearchExample  # noqa: E402 - must follow the sys.path.insert above


class MockEmbeddingModel:
    """Mock embedding model for testing."""

    def encode(self, text: str):
        """Generate deterministic fake embeddings based on text hash."""
        import numpy as np

        # Create a simple deterministic embedding based on text. crc32, not
        # hash(): str.__hash__ is salted per process (PYTHONHASHSEED), so
        # hash() would make these "deterministic" embeddings differ every run.
        hash_val = zlib.crc32(text.encode()) % 1000
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
    with patch("basic_vector_search.SentenceTransformer") as mock_st:
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
        assert example.model_name == "all-MiniLM-L6-v2"

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

        with patch("basic_vector_search.SentenceTransformer") as mock_st:
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
        assert all("title" in doc for doc in docs)
        assert all("content" in doc for doc in docs)
        assert all("category" in doc for doc in docs)
        assert all("level" in doc for doc in docs)

        # Check categories
        categories = {doc["category"] for doc in docs}
        assert "AI" in categories
        assert "Programming" in categories

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
            assert "embedding" in record.fields
            assert isinstance(record.fields["embedding"], VectorField)
            assert record.fields["embedding"].dimensions == 384

    @pytest.mark.asyncio
    async def test_create_custom_documents(self, vector_example):
        """Test creating custom documents."""
        await vector_example.setup_database()

        custom_docs = [
            {
                "title": "Custom Document",
                "content": "This is a custom test document.",
                "category": "Test",
                "level": "basic",
            }
        ]

        record_ids, records = await vector_example.create_documents_with_embeddings(custom_docs)

        assert len(record_ids) == 1
        assert records[0].data["title"] == "Custom Document"

    @pytest.mark.asyncio
    async def test_perform_vector_search(self, vector_example):
        """Test vector similarity search."""
        await vector_example.setup_database()
        await vector_example.create_documents_with_embeddings()

        # Perform search
        results = await vector_example.perform_vector_search("machine learning", k=3)

        assert len(results) <= 3
        assert all(hasattr(r, "record") for r in results)
        assert all(hasattr(r, "score") for r in results)

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
            "neural networks", filter_category="AI", k=2
        )

        assert len(results) <= 2
        # All results should be from AI category
        assert all(r.record["category"] == "AI" for r in results)

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
        record_ids, _records = await vector_example.create_documents_with_embeddings()
        assert len(record_ids) == 6

        # Search
        results = await vector_example.perform_vector_search("deep learning AI", k=3)
        assert len(results) > 0

        # Filtered search
        filtered = await vector_example.perform_filtered_search("programming", "Programming", k=2)
        assert all(r.record["category"] == "Programming" for r in filtered)

        # Cleanup
        await vector_example.cleanup()


class TestIntegrationWithRealModel:
    """Integration tests with real model (optional, requires sentence-transformers)."""

    @pytest.mark.asyncio
    @requires_package("sentence_transformers")
    async def test_with_real_model(self):
        """Test with actual SentenceTransformer model."""
        example = VectorSearchExample(model_name="all-MiniLM-L6-v2", verbose=False)

        try:
            # This will use the real model
            await example.setup_database()

            # Create just a few documents to keep test fast
            small_docs = [
                {
                    "title": "Machine Learning",
                    "content": "ML is about learning from data.",
                    "category": "AI",
                    "level": "basic",
                },
                {
                    "title": "Web Development",
                    "content": "Building websites with HTML and CSS.",
                    "category": "Web",
                    "level": "basic",
                },
            ]

            record_ids, _records = await example.create_documents_with_embeddings(small_docs)
            assert len(record_ids) == 2

            # Search should return results with valid scores
            results = await example.perform_vector_search("machine learning algorithms", k=2)
            assert len(results) == 2
            # Verify we got valid results with proper structure
            assert all(hasattr(r, "score") for r in results)
            assert all(0 <= r.score <= 1 for r in results)
            assert all(r.record.get("title") is not None for r in results)

        finally:
            await example.cleanup()


@pytest.mark.asyncio
async def test_example_main_function(capsys):
    """Run the example's ``main()`` for real, against a real database.

    Nothing stands in for ``VectorSearchExample`` here, and that is the
    whole point. The previous version patched the class out and drove
    ``main()`` against auto-generated attributes, so the mock *invented*
    every API the example got wrong: a ``db.find()`` no database class has
    ever defined, and a ``records[0].data["embedding"].vector`` reaching
    for a wrapper that ``.data`` has already unwrapped. Both passed here
    for as long as they existed, because a mock answers to any name.

    Only the embedding model is substituted, and only because downloading
    one is not something a test may do -- ``MockEmbeddingModel`` above is a
    real class with a real ``encode``, not a stand-in that agrees with
    whatever it is asked. Everything else is the shipped code path over a
    real in-memory SQLite database, which runs in about a second.
    """
    from basic_vector_search import main

    with patch("basic_vector_search.SentenceTransformer", lambda _name: MockEmbeddingModel()):
        await main()

    out = capsys.readouterr().out
    assert "Example completed successfully" in out
    # Each numbered step prints its own heading; assert the run reached the
    # last one rather than stopping early on a swallowed error.
    for heading in (
        "1. Setting up vector-enabled database",
        "3. Performing vector similarity search",
        "5. Finding similar documents",
        "7. Using Query builder methods",
    ):
        assert heading in out, heading
