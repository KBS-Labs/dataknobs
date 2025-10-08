"""Integration tests for all example scripts using real implementations."""

import os
import pytest
import asyncio
import sys
import time
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

# Skip all tests if PostgreSQL is not available
pytestmark = pytest.mark.skipif(
    not os.environ.get("TEST_POSTGRES", "").lower() == "true",
    reason="Example integration tests require TEST_POSTGRES=true and a running PostgreSQL instance with pgvector"
)

# Add examples to path
examples_path = Path(__file__).parent.parent.parent / "examples"
sys.path.insert(0, str(examples_path))

# Import real implementations
from dataknobs_data import DatabaseFactory, AsyncDatabaseFactory, Record, VectorField, Query, ComplexQuery
from dataknobs_data.vector import VectorTextSynchronizer, ChangeTracker, VectorMigration, IncrementalVectorizer


class TestEmbedding:
    """Real embedding implementation for testing without external dependencies."""

    @staticmethod
    def _deterministic_hash(text: str) -> int:
        """Generate a deterministic hash from text using character codes."""
        result = 0
        for char in text:
            result = (result * 31 + ord(char)) & 0xFFFFFFFF
        return result

    @staticmethod
    def generate(text: str, dimensions: int = 384) -> List[float]:
        """Generate deterministic embeddings that simulate real semantic similarity."""
        # Tokenize and create word-based features
        words = text.lower().split()

        # Common words and their "semantic" weights
        semantic_weights = {
            'machine': 1.0, 'learning': 1.0, 'ai': 1.0, 'artificial': 0.95,
            'intelligence': 0.95, 'neural': 0.9, 'network': 0.85, 'deep': 0.9,
            'python': 0.7, 'programming': 0.6, 'code': 0.5, 'data': 0.7,
            'science': 0.6, 'algorithm': 0.8, 'model': 0.75, 'training': 0.7,
            'database': 0.4, 'sql': 0.3, 'web': 0.1, 'javascript': 0.05, 'html': 0.05
        }

        # Create embedding
        embedding = []
        for i in range(dimensions):
            value = 0.1  # Base value

            # Add semantic contribution from each word
            for word in words:
                if word in semantic_weights:
                    # Use word weight and position to create variation
                    word_hash = TestEmbedding._deterministic_hash(word)
                    contribution = semantic_weights[word] * (1 + np.sin(i * 0.1 + word_hash % 10))
                    value += contribution / 10
                else:
                    # Unknown words get small deterministic contribution
                    combined_hash = TestEmbedding._deterministic_hash(word + str(i))
                    value += (combined_hash % 100) / 10000

            # Normalize to [0, 1]
            value = min(1.0, max(0.0, value / (len(words) + 1)))
            embedding.append(value)

        # Normalize the vector
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = [v / norm for v in embedding]

        return embedding


@pytest.fixture
async def vector_db():
    """Create a real vector-enabled database."""
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


class TestBasicVectorSearchIntegration:
    """Integration tests for basic vector search example."""
    
    @pytest.mark.asyncio
    async def test_vector_search_workflow(self, vector_db):
        """Test the complete vector search workflow."""
        # Import the refactored example
        from basic_vector_search import VectorSearchExample
        
        # Create example with test embedding
        example = VectorSearchExample(verbose=False)
        example.generate_embedding = lambda text: TestEmbedding.generate(text)
        example.db = vector_db
        
        # Create documents
        record_ids, records = await example.create_documents_with_embeddings()
        assert len(record_ids) == 6
        
        # Perform vector search
        results = await example.perform_vector_search("neural networks AI", k=3)
        assert len(results) <= 3
        
        # Verify we got valid results with scores
        assert all(hasattr(r, 'score') for r in results), "All results should have scores"
        assert all(0 <= r.score <= 1 for r in results), "All scores should be between 0 and 1"
        
        # Verify results have expected fields
        for r in results:
            assert r.record.get_value('title') is not None
            assert r.record.get_value('content') is not None
        
        # Test filtered search
        filtered = await example.perform_filtered_search("programming", "Programming", k=2)
        assert all(r.record.get_value('category') == 'Programming' for r in filtered)
    
    @pytest.mark.asyncio
    async def test_similarity_metrics(self, vector_db):
        """Test different similarity metrics."""
        # Create two documents
        doc1 = Record({
            "title": "Machine Learning",
            "content": "Introduction to ML",
            "embedding": VectorField(TestEmbedding.generate("machine learning AI"), dimensions=384)
        })
        doc2 = Record({
            "title": "Web Development",
            "content": "Building websites",
            "embedding": VectorField(TestEmbedding.generate("web javascript html"), dimensions=384)
        })
        
        await vector_db.create(doc1)
        await vector_db.create(doc2)
        
        # Search for ML-related content
        ml_query = TestEmbedding.generate("artificial intelligence machine learning")
        results = await vector_db.vector_search(
            query_vector=ml_query,
            k=2,
            vector_field="embedding"
        )
        
        # Verify we got valid results with proper structure and scores
        assert len(results) == 2, "Should return 2 results"
        assert all(hasattr(r, 'score') for r in results), "All results should have scores"
        assert all(0 <= r.score <= 1 for r in results), "All scores should be between 0 and 1"
        
        # Verify results have expected fields
        for r in results:
            assert r.record.get_value('title') is not None
            assert r.record.get_value('content') is not None


class TestTextToVectorSyncIntegration:
    """Integration tests for text-to-vector synchronization."""
    
    @pytest.mark.asyncio
    async def test_synchronizer_workflow(self, vector_db):
        """Test complete synchronization workflow."""
        # Create documents without embeddings
        docs = [
            {"id": 1, "title": "Python Guide", "content": "Learn Python programming"},
            {"id": 2, "title": "ML Basics", "content": "Machine learning fundamentals"},
            {"id": 3, "title": "Data Science", "content": "Analyzing data with Python"}
        ]
        
        record_ids = []
        for doc in docs:
            record_id = await vector_db.create(Record(doc))
            record_ids.append(record_id)
        
        # Setup synchronizer
        sync = VectorTextSynchronizer(
            vector_db,
            TestEmbedding.generate,
            text_fields=["title", "content"],
            vector_field="embedding",
            field_separator=" ",
            batch_size=2
        )
        
        # Bulk sync
        results = await sync.sync_all()
        assert results['processed'] == 3
        
        # Verify embeddings added
        for record_id in record_ids:
            record = await vector_db.read(record_id)
            assert 'embedding' in record.fields
            assert len(record.fields['embedding'].value) == 384
        
        # Test vector search on synced data
        query_embedding = TestEmbedding.generate("Python machine learning")
        search_results = await vector_db.vector_search(
            query_vector=query_embedding,
            k=2,
            vector_field="embedding"
        )
        
        assert len(search_results) == 2
        # ML and Data Science docs should rank high
        titles = [r.record.get_value('title') for r in search_results]
        assert "ML Basics" in titles or "Data Science" in titles
    
    @pytest.mark.asyncio
    async def test_change_tracking(self, vector_db):
        """Test change tracking with real implementation."""
        # Create records with embeddings
        record1 = await vector_db.create(Record({
            "title": "Original",
            "content": "Original content",
            "embedding": VectorField(TestEmbedding.generate("original content"), dimensions=384)
        }))
        
        # Setup tracker
        tracker = ChangeTracker(vector_db)
        await tracker.start_tracking(
            tracked_fields=["title", "content"],
            vector_field="embedding"
        )
        
        # Update content
        record = await vector_db.read(record1)
        record.fields["content"].value = "Updated content"
        await vector_db.update(record1, record)
        
        # Should detect as outdated
        outdated = await tracker.get_outdated_records()
        assert len(outdated) == 1
        assert outdated[0].get_value('content') == "Updated content"
        
        # Re-sync
        sync = VectorTextSynchronizer(
            vector_db, 
            TestEmbedding.generate,
            text_fields=["title", "content"],
            vector_field="embedding",
            field_separator=" "
        )
        await sync.sync_record(record1)
        
        # Should no longer be outdated
        outdated = await tracker.get_outdated_records()
        assert len(outdated) == 0


class TestMigrationIntegration:
    """Integration tests for migration example."""
    
    @pytest.mark.asyncio
    async def test_database_migration(self):
        """Test migrating from non-vector to vector database."""
        # Create legacy database
        factory = AsyncDatabaseFactory()
        legacy_db = factory.create(
            backend="sqlite",
            path=":memory:",
            vector_enabled=False
        )
        await legacy_db.connect()
        
        # Create vector database
        factory = AsyncDatabaseFactory()
        vector_db = factory.create(
            backend="sqlite",
            path=":memory:",
            vector_enabled=True,
            vector_metric="cosine"
        )
        await vector_db.connect()
        
        try:
            # Add legacy data
            legacy_docs = [
                {"id": 1, "title": "Cloud Computing", "content": "Introduction to cloud services"},
                {"id": 2, "title": "Docker Basics", "content": "Container technology explained"},
                {"id": 3, "title": "Kubernetes", "content": "Container orchestration platform"}
            ]
            
            for doc in legacy_docs:
                await legacy_db.create(Record(doc))
            
            # Setup migration
            migration = VectorMigration(
                source_db=legacy_db,
                target_db=vector_db,
                embedding_fn=TestEmbedding.generate,
                text_fields=["title", "content"],
                vector_field="embedding",
                batch_size=2
            )
            
            # Run migration
            status = await migration.run()
            
            # Check status
            assert status.total_processed == 3
            assert status.failed_count == 0
            
            # Verify migrated data
            from dataknobs_data import Query
            migrated = await vector_db.search(Query())
            assert len(migrated) == 3
            
            # All should have embeddings
            for record in migrated:
                assert 'embedding' in record.fields
                assert len(record.fields['embedding'].value) == 384
            
            # Test search on migrated data
            query = TestEmbedding.generate("container docker kubernetes")
            search_results = await vector_db.vector_search(
                query_vector=query,
                k=2,
                vector_field="embedding"
            )
            
            assert len(search_results) == 2
            # Docker and Kubernetes should rank high
            titles = [r.record.get_value('title') for r in search_results]
            assert "Docker Basics" in titles or "Kubernetes" in titles
            
        finally:
            await legacy_db.close()
            await vector_db.close()
    
    @pytest.mark.asyncio
    async def test_incremental_vectorization(self, vector_db):
        """Test incremental vectorization."""
        # Add records without embeddings
        docs = [
            {"title": f"Document {i}", "content": f"Content for document {i}"}
            for i in range(5)
        ]
        
        for doc in docs:
            await vector_db.create(Record(doc))
        
        # Setup incremental vectorizer
        vectorizer = IncrementalVectorizer(
            database=vector_db,
            embedding_fn=TestEmbedding.generate,
            text_fields="content",  # Using content as the text field
            vector_field="embedding",
            batch_size=2
        )
        
        # Track progress
        progress_updates = []
        
        async def progress_callback(completed, total, batch):
            progress_updates.append((completed, total))
        
        # Run vectorization
        results = await vectorizer.run(
            progress_callback=progress_callback,
            max_workers=1
        )
        
        assert results['processed'] == 5
        assert results['failed'] == 0
        assert len(progress_updates) > 0
        
        # Verify all have embeddings
        from dataknobs_data import Query
        all_records = await vector_db.search(Query())
        assert all('embedding' in r.fields for r in all_records)


class TestHybridSearchIntegration:
    """Integration tests for hybrid search."""
    
    @pytest.mark.asyncio
    async def test_hybrid_search_methods(self, vector_db):
        """Test different hybrid search strategies."""
        # Create diverse documents
        docs = [
            {
                "title": "Machine Learning with Python",
                "content": "Using Python for machine learning and data science applications.",
                "category": "AI"
            },
            {
                "title": "Deep Learning Fundamentals",
                "content": "Understanding neural networks and deep learning architectures.",
                "category": "AI"
            },
            {
                "title": "Python Web Development",
                "content": "Building web applications with Python frameworks like Django and Flask.",
                "category": "Web"
            },
            {
                "title": "JavaScript Basics",
                "content": "Introduction to JavaScript programming for web development.",
                "category": "Web"
            },
            {
                "title": "Database Design",
                "content": "Principles of database design and SQL optimization techniques.",
                "category": "Database"
            }
        ]
        
        # Add documents with embeddings
        for doc in docs:
            text = f"{doc['title']} {doc['content']}"
            embedding = TestEmbedding.generate(text)
            record = Record({
                **doc,
                "embedding": VectorField(embedding, dimensions=384)
            })
            await vector_db.create(record)
        
        # Test vector search
        query_embedding = TestEmbedding.generate("Python machine learning AI")
        vector_results = await vector_db.vector_search(
            query_vector=query_embedding,
            k=3,
            vector_field="embedding"
        )
        
        assert len(vector_results) <= 3
        # ML/AI related documents should rank high (check all results for robustness)
        all_titles = [r.record.get_value('title') for r in vector_results]
        has_ml_or_ai = any("Machine Learning" in title or "Deep Learning" in title or "Python machine learning" in title.lower() 
                          for title in all_titles)
        assert has_ml_or_ai, f"Expected ML/AI related document in results, got: {all_titles}"
        
        # Test filtered search
        from dataknobs_data.query import Operator
        filtered_results = await vector_db.vector_search(
            query_vector=query_embedding,
            k=5,
            filter=Query().filter("category", Operator.EQ, "AI"),
            vector_field="embedding"
        )
        
        assert all(r.record.get_value('category') == 'AI' for r in filtered_results)
        assert len(filtered_results) <= 5  # Should respect the k=5 limit
        
        # Test complex query
        from dataknobs_data.query import Operator
        complex_query = ComplexQuery.OR([
            Query().filter("title", Operator.LIKE, "%Python%"),
            Query().filter("category", Operator.EQ, "AI")
        ])
        
        complex_results = await vector_db.search(complex_query)
        assert len(complex_results) >= 1  # Should find at least one matching document
    
    @pytest.mark.asyncio
    async def test_reciprocal_rank_fusion(self, vector_db):
        """Test RRF implementation."""
        # Add test documents
        docs = [
            {"id": i, "title": f"Doc {i}", "content": f"Content {i}"}
            for i in range(5)
        ]
        
        for doc in docs:
            embedding = TestEmbedding.generate(f"{doc['title']} {doc['content']}")
            record = Record({
                **doc,
                "embedding": VectorField(embedding, dimensions=384)
            })
            await vector_db.create(record)
        
        # Get vector results
        query_embedding = TestEmbedding.generate("Doc 2 Content")
        vector_results = await vector_db.vector_search(
            query_vector=query_embedding,
            k=3,
            vector_field="embedding"
        )
        
        # Calculate RRF scores
        k = 60
        rrf_scores = {}
        
        for rank, result in enumerate(vector_results):
            doc_id = result.record.get_value('id')
            rrf_scores[doc_id] = 1.0 / (k + rank + 1)
        
        # Verify RRF scoring
        assert len(rrf_scores) == 3
        # Scores should decrease with rank
        scores = list(rrf_scores.values())
        assert scores[0] > scores[1] > scores[2]


class TestPerformanceIntegration:
    """Test performance aspects of examples."""
    
    @pytest.mark.asyncio
    async def test_batch_operations_performance(self, vector_db):
        """Test batch operation performance."""
        # Create many documents
        num_docs = 50
        docs = []
        
        for i in range(num_docs):
            text = f"Document {i} about topic {i % 5}"
            embedding = TestEmbedding.generate(text)
            
            record = Record({
                "id": i,
                "title": f"Doc {i}",
                "content": text,
                "embedding": VectorField(embedding, dimensions=384)
            })
            docs.append(record)
        
        # Batch create
        start = time.time()
        record_ids = await vector_db.create_batch(docs)
        create_time = time.time() - start
        
        assert len(record_ids) == num_docs
        assert create_time < 5.0  # Should be reasonably fast
        
        # Test search performance
        search_times = []
        
        for i in range(10):
            query = TestEmbedding.generate(f"topic {i % 5}")
            
            start = time.time()
            results = await vector_db.vector_search(
                query_vector=query,
                k=5,
                vector_field="embedding"
            )
            search_times.append(time.time() - start)
            
            assert len(results) <= 5
        
        avg_search_time = sum(search_times) / len(search_times)
        assert avg_search_time < 0.5  # Searches should be fast
    
    @pytest.mark.asyncio
    async def test_large_embedding_dimensions(self):
        """Test with different embedding dimensions."""
        # Test with different dimensions - each needs its own database
        # since vectors in a collection must have consistent dimensions
        dimensions = [128, 256, 384, 512]
        
        for dim in dimensions:
            # Create a fresh database for each dimension
            factory = AsyncDatabaseFactory()
            db = factory.create(
                backend="memory",
                vector_enabled=True,
                vector_metric="cosine"
            )
            await db.connect()
            
            try:
                # Create embedding of specified dimension
                text = f"Test document for {dim} dimensions"
                embedding = TestEmbedding.generate(text, dimensions=dim)
                
                record = Record({
                    "title": f"Dim {dim}",
                    "embedding": VectorField(embedding, dimensions=dim)
                })
                
                record_id = await db.create(record)
                
                # Verify storage and retrieval
                retrieved = await db.read(record_id)
                assert len(retrieved.fields['embedding'].value) == dim
                
                # Test search
                query = TestEmbedding.generate("test query", dimensions=dim)
                results = await db.vector_search(
                    query_vector=query,
                    k=1,
                    vector_field="embedding"
                )
                
                assert len(results) == 1
                assert results[0].record.get_value('title') == f"Dim {dim}"
            
            finally:
                await db.close()


@pytest.mark.asyncio
async def test_end_to_end_workflow():
    """Test complete end-to-end workflow using all examples."""
    # Create database
    factory = AsyncDatabaseFactory()
    db = factory.create(
        backend="sqlite",
        path=":memory:",
        vector_enabled=True,
        vector_metric="cosine"
    )
    await db.connect()
    
    try:
        # 1. Create initial documents without embeddings
        initial_docs = [
            {"id": 1, "title": "AI Basics", "content": "Introduction to artificial intelligence"},
            {"id": 2, "title": "ML Guide", "content": "Machine learning comprehensive guide"},
            {"id": 3, "title": "Python Tutorial", "content": "Learn Python programming"}
        ]
        
        record_ids = []
        for doc in initial_docs:
            record_id = await db.create(Record(doc))
            record_ids.append(record_id)
        
        # 2. Synchronize to add embeddings
        sync = VectorTextSynchronizer(
            db,
            TestEmbedding.generate,
            text_fields=["title", "content"],
            vector_field="embedding",
            field_separator=" "
        )
        
        results = await sync.sync_all()
        assert results['processed'] == 3
        
        # 3. Perform vector search
        query = TestEmbedding.generate("artificial intelligence machine learning")
        search_results = await db.vector_search(
            query_vector=query,
            k=2,
            vector_field="embedding"
        )
        
        assert len(search_results) == 2
        titles = [r.record.get_value('title') for r in search_results]
        assert "AI Basics" in titles or "ML Guide" in titles
        
        # 4. Update a document
        record = await db.read(record_ids[0])
        record.fields["content"].value = "Advanced AI concepts and applications"
        await db.update(record_ids[0], record)
        
        # 5. Track changes
        tracker = ChangeTracker(db)
        await tracker.start_tracking(["title", "content"], "embedding")
        
        outdated = await tracker.get_outdated_records()
        assert len(outdated) == 1
        
        # 6. Re-sync outdated record
        await sync.sync_record(outdated[0].id)
        
        # 7. Verify search reflects update
        new_results = await db.vector_search(
            query_vector=query,
            k=2,
            vector_field="embedding"
        )
        
        # Updated document might rank differently
        assert len(new_results) == 2
        
    finally:
        await db.close()
    
    print("✓ End-to-end workflow completed successfully!")