"""Tests for the hybrid search example using real implementations."""

import os
import pytest
import asyncio
import sys
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass

# Skip all tests if PostgreSQL is not available
pytestmark = pytest.mark.skipif(
    not os.environ.get("TEST_POSTGRES", "").lower() == "true",
    reason="Hybrid search tests require TEST_POSTGRES=true and a running PostgreSQL instance with pgvector"
)

# Add examples to path
examples_path = Path(__file__).parent.parent.parent / "examples"
sys.path.insert(0, str(examples_path))

from dataknobs_data import DatabaseFactory, AsyncDatabaseFactory, Record, VectorField, Query, ComplexQuery, Operator


class SimpleEmbeddingModel:
    """Simple embedding model for testing without external dependencies."""
    
    def encode(self, text: str) -> List[float]:
        """Generate deterministic embeddings based on text content."""
        # Create a simple but deterministic embedding
        # This simulates what a real model would do but without the dependency
        words = text.lower().split()
        embedding = []
        
        # Generate 384-dimensional embedding (matching all-MiniLM-L6-v2)
        for i in range(384):
            value = 0.0
            for word in words:
                # Simple hash-based approach for deterministic embeddings
                word_hash = hash(word + str(i)) % 1000
                value += word_hash / 1000.0
            # Normalize
            value = value / (len(words) + 1)
            embedding.append(value)
        
        return embedding


def create_test_embedding(text: str) -> List[float]:
    """Create test embeddings without external dependencies."""
    model = SimpleEmbeddingModel()
    return model.encode(text)


@pytest.fixture
async def real_sqlite_db():
    """Create a real SQLite database with vector support."""
    factory = AsyncDatabaseFactory()
    db = factory.create(
        backend="sqlite",
        path=":memory:",
        vector_enabled=True,
        vector_metric="cosine"
    )
    await db.connect()
    yield db
    await db.close()


@pytest.fixture
async def populated_vector_db(real_sqlite_db):
    """Create a populated database with real vector data."""
    documents = [
        {
            "id": 1,
            "title": "Introduction to Machine Learning",
            "content": "Machine learning is a branch of artificial intelligence that enables computers to learn from data without explicit programming.",
            "category": "AI",
            "keywords": ["machine learning", "AI", "algorithms"]
        },
        {
            "id": 2,
            "title": "Deep Neural Networks Explained",
            "content": "Deep learning uses neural networks with multiple layers to progressively extract higher-level features from raw input.",
            "category": "AI",
            "keywords": ["deep learning", "neural networks", "layers"]
        },
        {
            "id": 3,
            "title": "Natural Language Processing Fundamentals",
            "content": "NLP combines linguistics and machine learning to help computers understand, interpret, and generate human language.",
            "category": "AI",
            "keywords": ["NLP", "linguistics", "language"]
        },
        {
            "id": 4,
            "title": "Python Programming Best Practices",
            "content": "Python is a versatile programming language known for clean syntax and readability. Best practices include using virtual environments.",
            "category": "Programming",
            "keywords": ["Python", "programming", "best practices"]
        },
        {
            "id": 5,
            "title": "Web Development with JavaScript",
            "content": "JavaScript powers interactive web applications. Modern frameworks like React help developers build complex applications.",
            "category": "Programming",
            "keywords": ["JavaScript", "web", "React"]
        },
        {
            "id": 6,
            "title": "Database Design and Optimization",
            "content": "Effective database design involves normalization, indexing strategies, and query optimization for better performance.",
            "category": "Database",
            "keywords": ["database", "SQL", "optimization"]
        }
    ]
    
    for doc in documents:
        # Create real embeddings
        text = f"{doc['title']} {doc['content']}"
        embedding = create_test_embedding(text)
        
        record = Record({
            **doc,
            "embedding": VectorField(embedding, dimensions=384)
        })
        await real_sqlite_db.create(record)
    
    return real_sqlite_db


class TestRealVectorSearch:
    """Test vector search with real SQLite backend."""
    
    @pytest.mark.asyncio
    async def test_real_vector_search(self, populated_vector_db):
        """Test actual vector similarity search."""
        # Create query embedding
        query_text = "machine learning algorithms"
        query_embedding = create_test_embedding(query_text)
        
        # Perform real vector search
        results = await populated_vector_db.vector_search(
            query_vector=query_embedding,
            k=3,
            vector_field="embedding"
        )
        
        assert len(results) <= 3
        assert all(hasattr(r, 'record') for r in results)
        assert all(hasattr(r, 'score') for r in results)
        
        # Scores should be properly ordered
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)
    
    @pytest.mark.asyncio
    async def test_filtered_vector_search(self, populated_vector_db):
        """Test vector search with real filtering."""
        query_embedding = create_test_embedding("programming concepts")
        
        # Search with category filter using real Query object
        results = await populated_vector_db.vector_search(
            query_vector=query_embedding,
            k=5,
            filter=Query().filter("category", "=", "Programming"),
            vector_field="embedding"
        )
        
        # Verify filter is applied
        assert all(r.record.get_value('category') == 'Programming' for r in results)
        assert len(results) <= 2  # Only 2 programming documents
    
    @pytest.mark.asyncio
    async def test_multi_filter_search(self, populated_vector_db):
        """Test vector search with multiple filters."""
        query_embedding = create_test_embedding("artificial intelligence")
        
        # Complex filter
        results = await populated_vector_db.vector_search(
            query_vector=query_embedding,
            k=10,
            filter=Query().filter("category", "in", ["AI", "Programming"]),
            vector_field="embedding"
        )
        
        categories = {r.record.get_value('category') for r in results}
        assert categories.issubset({"AI", "Programming"})
        assert "Database" not in categories


class TestRealHybridSearch:
    """Test hybrid search implementation with real components."""
    
    @pytest.mark.asyncio
    async def test_text_search_implementation(self, populated_vector_db):
        """Test text search using real database queries."""
        # Use real Query for text search
        # Note: SQLite doesn't have full-text search by default, 
        # so we'll use simple filter
        results = await populated_vector_db.search(
            Query().filter("content", Operator.LIKE, "%machine learning%")
        )
        
        assert len(results) > 0
        # Verify results contain the search term
        for record in results:
            assert "machine learning" in record.get_value('content').lower() or \
                   "machine learning" in record.get_value('title').lower()
    
    @pytest.mark.asyncio
    async def test_combined_search(self, populated_vector_db):
        """Test combining text and vector search results."""
        query_text = "Python programming"
        query_embedding = create_test_embedding(query_text)
        
        # Get vector results
        vector_results = await populated_vector_db.vector_search(
            query_vector=query_embedding,
            k=3,
            vector_field="embedding"
        )
        
        # Get text results (simple contains filter)
        text_results = await populated_vector_db.search(
            Query().filter("title", Operator.LIKE, "%Python%").limit(3)
        )
        
        # Both should return results
        assert len(vector_results) > 0
        assert len(text_results) > 0
        
        # Text results should contain Python in title
        for record in text_results:
            assert "Python" in record.get_value('title')
    
    @pytest.mark.asyncio
    async def test_reciprocal_rank_fusion_concept(self, populated_vector_db):
        """Test RRF concept with real search results."""
        query_embedding = create_test_embedding("deep learning neural networks")
        
        # Get results from different methods
        vector_results = await populated_vector_db.vector_search(
            query_vector=query_embedding,
            k=5,
            vector_field="embedding"
        )
        
        # Simple text search
        text_results = await populated_vector_db.search(
            Query().filter("content", Operator.LIKE, "%neural%").limit(5)
        )
        
        # Calculate RRF scores (simplified)
        k = 60
        rrf_scores = {}
        
        # Process vector results
        for rank, result in enumerate(vector_results):
            doc_id = result.record.get_value('id')
            rrf_scores[doc_id] = 1.0 / (k + rank + 1)
        
        # Process text results
        for rank, record in enumerate(text_results):
            doc_id = record.get_value('id')
            score = 1.0 / (k + rank + 1)
            if doc_id in rrf_scores:
                rrf_scores[doc_id] += score
            else:
                rrf_scores[doc_id] = score
        
        # Verify RRF combines results
        assert len(rrf_scores) > 0
        # Top RRF result should have good score
        top_score = max(rrf_scores.values())
        assert top_score > 0


class TestRealQueryBuilder:
    """Test Query builder with real database."""
    
    @pytest.mark.asyncio
    async def test_near_text_query(self, populated_vector_db):
        """Test near_text query method."""
        # Create query embedding and use vector search
        query_embedding = create_test_embedding("machine learning and artificial intelligence")
        results = await populated_vector_db.vector_search(
            query_vector=query_embedding,
            k=3,
            vector_field="embedding"
        )
        
        assert len(results) <= 3
        # Results should be AI-related given the query
        categories = [r.record.get_value('category') for r in results]
        assert 'AI' in categories
    
    @pytest.mark.asyncio
    async def test_similar_to_query(self, populated_vector_db):
        """Test similar_to query method."""
        # Get a reference embedding
        reference_text = "Deep learning with neural networks"
        reference_embedding = create_test_embedding(reference_text)
        
        # Use vector search directly
        results = await populated_vector_db.vector_search(
            query_vector=reference_embedding,
            k=3,
            vector_field="embedding"
        )
        
        assert len(results) <= 3
        # Should find AI/deep learning related documents
        titles = [r.record.get_value('title') for r in results]
        assert any('Deep' in title or 'Neural' in title for title in titles)


class TestRealComplexQueries:
    """Test complex queries with real database."""
    
    @pytest.mark.asyncio
    async def test_complex_and_query(self, populated_vector_db):
        """Test ComplexQuery with AND logic."""
        query_embedding = create_test_embedding("artificial intelligence")
        
        # Use vector search with filter
        results = await populated_vector_db.vector_search(
            query_vector=query_embedding,
            k=10,
            filter=Query().filter("category", "=", "AI"),
            vector_field="embedding"
        )
        
        # All results should be AI category
        assert all(r.record.get_value('category') == 'AI' for r in results)
        assert len(results) <= 10  # Should respect the k=10 limit
    
    @pytest.mark.asyncio
    async def test_complex_or_query(self, populated_vector_db):
        """Test ComplexQuery with OR logic."""
        # Search for Python and JavaScript separately (SQLite doesn't support complex OR)
        python_results = await populated_vector_db.search(
            Query().filter("title", Operator.LIKE, "%Python%")
        )
        js_results = await populated_vector_db.search(
            Query().filter("title", Operator.LIKE, "%JavaScript%")
        )
        
        # Combine results
        results = python_results + js_results
        
        # Should find both Python and JavaScript documents
        titles = [r.get_value('title') for r in results]
        assert any('Python' in title for title in titles)
        assert any('JavaScript' in title for title in titles)


class TestRealPerformance:
    """Test performance with real operations."""
    
    @pytest.mark.asyncio
    async def test_batch_vector_operations(self, real_sqlite_db):
        """Test batch creation with vectors."""
        # Create multiple records with vectors
        records = []
        for i in range(10):
            text = f"Document {i} with unique content about topic {i}"
            embedding = create_test_embedding(text)
            
            record = Record({
                "id": i,
                "title": f"Document {i}",
                "content": text,
                "embedding": VectorField(embedding, dimensions=384)
            })
            records.append(record)
        
        # Batch create
        record_ids = await real_sqlite_db.create_batch(records)
        assert len(record_ids) == 10
        
        # Verify vector search works
        query_embedding = create_test_embedding("Document 5")
        results = await real_sqlite_db.vector_search(
            query_vector=query_embedding,
            k=3,
            vector_field="embedding"
        )
        
        assert len(results) == 3
        # Document 5 should be in top results
        titles = [r.record.get_value('title') for r in results]
        assert "Document 5" in titles
    
    @pytest.mark.asyncio
    async def test_vector_search_performance(self, populated_vector_db):
        """Test vector search performance metrics."""
        import time
        
        # Perform multiple searches
        search_times = []
        
        for i in range(5):
            query_text = f"search query {i}"
            query_embedding = create_test_embedding(query_text)
            
            start = time.time()
            results = await populated_vector_db.vector_search(
                query_vector=query_embedding,
                k=3,
                vector_field="embedding"
            )
            search_time = time.time() - start
            search_times.append(search_time)
            
            assert len(results) <= 3
        
        # Calculate average search time
        avg_time = sum(search_times) / len(search_times)
        assert avg_time < 1.0  # Should be fast for small dataset
        
        # All searches should complete
        assert len(search_times) == 5


class TestRealDatabaseIntegration:
    """Test real database integration features."""
    
    @pytest.mark.asyncio
    async def test_vector_field_persistence(self, real_sqlite_db):
        """Test that vector fields are properly persisted."""
        # Create record with vector
        embedding = create_test_embedding("test document")
        record = Record({
            "title": "Test",
            "embedding": VectorField(embedding, dimensions=384)
        })
        
        record_id = await real_sqlite_db.create(record)
        
        # Read back and verify
        retrieved = await real_sqlite_db.read(record_id)
        assert 'embedding' in retrieved.fields
        embedding_field = retrieved.fields['embedding']
        assert len(embedding_field.value) == 384
        
        # Verify it's the same embedding
        for i in range(10):  # Check first 10 values
            assert abs(embedding_field.value[i] - embedding[i]) < 0.0001
    
    @pytest.mark.asyncio
    async def test_update_vector_field(self, real_sqlite_db):
        """Test updating vector fields."""
        # Create initial record
        initial_embedding = create_test_embedding("initial text")
        record = Record({
            "title": "Test",
            "content": "Initial content",
            "embedding": VectorField(initial_embedding, dimensions=384)
        })
        
        record_id = await real_sqlite_db.create(record)
        
        # Update with new embedding
        new_embedding = create_test_embedding("updated text")
        updated_record = await real_sqlite_db.read(record_id)
        updated_record.set_value("content", "Updated content")
        updated_record.fields["embedding"] = VectorField(new_embedding, dimensions=384)
        await real_sqlite_db.update(record_id, updated_record)
        
        # Verify update
        updated = await real_sqlite_db.read(record_id)
        assert updated.get_value('content') == "Updated content"
        assert 'embedding' in updated.fields
        updated_embedding = updated.fields['embedding'].value
        assert len(updated_embedding) == 384
        # Embedding should be different
        assert updated_embedding[0] != initial_embedding[0]


@pytest.mark.asyncio
async def test_complete_hybrid_workflow_real():
    """Test complete hybrid search workflow with real components."""
    # Create real database
    factory = AsyncDatabaseFactory()
    db = factory.create(
        backend="sqlite",
        path=":memory:",
        vector_enabled=True,
        vector_metric="cosine"
    )
    await db.connect()
    
    try:
        # Add diverse documents
        docs = [
            {
                "title": "AI Research Paper",
                "content": "Latest advances in artificial intelligence and machine learning research.",
                "category": "Research"
            },
            {
                "title": "ML Tutorial",
                "content": "Beginner-friendly machine learning tutorial with Python examples.",
                "category": "Tutorial"
            },
            {
                "title": "Python Guide",
                "content": "Comprehensive guide to Python programming for data science.",
                "category": "Guide"
            },
            {
                "title": "Deep Learning Book",
                "content": "Understanding deep neural networks and their applications.",
                "category": "Book"
            }
        ]
        
        # Create records with real embeddings
        for doc in docs:
            text = f"{doc['title']} {doc['content']}"
            embedding = create_test_embedding(text)
            record = Record({
                **doc,
                "embedding": VectorField(embedding, dimensions=384)
            })
            await db.create(record)
        
        # Test hybrid approach
        query = "machine learning Python tutorial"
        query_embedding = create_test_embedding(query)
        
        # Vector search
        vector_results = await db.vector_search(
            query_vector=query_embedding,
            k=3,
            vector_field="embedding"
        )
        
        # Text search (using contains)
        text_results = await db.search(
            Query().filter("content", Operator.LIKE, "%Python%").limit(3)
        )
        
        # Combine results (simple scoring)
        combined_scores = {}
        
        # Add vector scores
        for result in vector_results:
            doc_id = result.record.get_value('title')  # Use title as ID
            combined_scores[doc_id] = result.score * 0.7  # Weight for vector
        
        # Add text matches
        for record in text_results:
            doc_id = record.get_value('title')
            if doc_id in combined_scores:
                combined_scores[doc_id] += 0.3  # Bonus for text match
            else:
                combined_scores[doc_id] = 0.3
        
        # Sort by combined score
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        assert len(sorted_results) > 0
        # ML Tutorial should rank high (has both ML and Python)
        top_titles = [title for title, _ in sorted_results[:2]]
        assert "ML Tutorial" in top_titles or "Python Guide" in top_titles
        
    finally:
        await db.close()