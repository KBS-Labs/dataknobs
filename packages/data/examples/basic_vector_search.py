#!/usr/bin/env python3
"""
Basic Vector Search Example

This example demonstrates:
1. Setting up a vector-enabled database
2. Creating records with vector embeddings
3. Performing vector similarity search
4. Filtering vector search results

Requirements:
    pip install dataknobs-data sentence-transformers
"""

import asyncio
from typing import List, Optional, Dict, Any
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    # Provide a mock for testing
    class SentenceTransformer:
        def __init__(self, model_name):
            self.model_name = model_name
        
        def encode(self, text):
            # Simple hash-based embedding for testing
            import hashlib
            hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
            return np.array([float((hash_val + i) % 100) / 100.0 for i in range(384)])

from dataknobs_data import AsyncDatabaseFactory, DatabaseFactory, Record, VectorField, Query


class VectorSearchExample:
    """Encapsulates vector search functionality for testing."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', verbose: bool = True):
        """Initialize with optional model and verbosity settings."""
        self.verbose = verbose
        self.model = None
        self.model_name = model_name
        self.db = None
        
    def log(self, message: str):
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(message)
    
    def load_model(self):
        """Load the embedding model."""
        if not self.model:
            self.log("Loading embedding model...")
            if not HAS_SENTENCE_TRANSFORMERS:
                self.log("Note: Using mock embeddings (sentence-transformers not installed)")
            self.model = SentenceTransformer(self.model_name)
        return self.model
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a text string."""
        if not self.model:
            self.load_model()
        embedding = self.model.encode(text)
        return embedding.tolist()
    
    def get_sample_documents(self) -> List[Dict[str, Any]]:
        """Return sample documents for testing."""
        return [
            {
                "title": "Introduction to Machine Learning",
                "content": "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
                "category": "AI",
                "level": "beginner"
            },
            {
                "title": "Deep Learning Fundamentals",
                "content": "Deep learning uses neural networks with multiple layers to progressively extract features from raw input.",
                "category": "AI",
                "level": "intermediate"
            },
            {
                "title": "Natural Language Processing",
                "content": "NLP combines computational linguistics with machine learning to process and analyze human language.",
                "category": "AI",
                "level": "intermediate"
            },
            {
                "title": "Python Programming Basics",
                "content": "Python is a high-level programming language known for its simplicity and readability.",
                "category": "Programming",
                "level": "beginner"
            },
            {
                "title": "Data Structures and Algorithms",
                "content": "Understanding data structures and algorithms is fundamental to writing efficient code.",
                "category": "Programming",
                "level": "intermediate"
            },
            {
                "title": "Web Development with JavaScript",
                "content": "JavaScript is the programming language of the web, enabling interactive and dynamic websites.",
                "category": "Programming",
                "level": "beginner"
            }
        ]
    
    async def setup_database(self, backend: str = "sqlite", database: str = ":memory:") -> Any:
        """Set up and return a vector-enabled database."""
        self.log("\n1. Setting up vector-enabled database...")
        
        # Use AsyncDatabaseFactory for async operations
        factory = AsyncDatabaseFactory()
        self.db = factory.create(
            backend=backend,
            database=database,
            vector_enabled=True,
            vector_metric="cosine"
        )
        
        await self.db.connect()  # Use connect instead of initialize
        self.log("✓ Database initialized with vector support")
        return self.db
    
    async def create_documents_with_embeddings(self, documents: Optional[List[Dict]] = None) -> List[str]:
        """Create documents with embeddings in the database."""
        if not self.db:
            raise RuntimeError("Database not initialized. Call setup_database first.")
        
        if documents is None:
            documents = self.get_sample_documents()
        
        self.log(f"\n2. Creating {len(documents)} documents with embeddings...")
        
        records = []
        for doc in documents:
            text = f"{doc['title']} {doc['content']}"
            embedding = self.generate_embedding(text)
            
            record = Record({
                **doc,
                "embedding": VectorField(embedding)  # Simplified - name and dimensions auto-detected
            })
            records.append(record)
        
        record_ids = await self.db.create_batch(records)
        self.log(f"✓ Created {len(record_ids)} documents with embeddings")
        
        return record_ids, records
    
    async def perform_vector_search(self, query_text: str, k: int = 3) -> List[Any]:
        """Perform vector similarity search."""
        if not self.db:
            raise RuntimeError("Database not initialized.")
        
        self.log(f"\n3. Performing vector similarity search...")
        self.log(f"Query: '{query_text}'")
        
        query_embedding = self.generate_embedding(query_text)
        
        results = await self.db.vector_search(
            query_vector=query_embedding,
            k=k,
            vector_field="embedding"
        )
        
        if self.verbose:
            self.log("\nTop results:")
            for i, result in enumerate(results, 1):
                self.log(f"{i}. {result.record['title']}")
                self.log(f"   Score: {result.score:.3f}")
                self.log(f"   Category: {result.record['category']}")
        
        return results
    
    async def perform_filtered_search(self, query_text: str, filter_category: str, k: int = 3) -> List[Any]:
        """Perform vector search with category filtering."""
        if not self.db:
            raise RuntimeError("Database not initialized.")
        
        self.log(f"\n4. Vector search with category filter...")
        
        query_embedding = self.generate_embedding(query_text)
        
        filtered_results = await self.db.vector_search(
            query_vector=query_embedding,
            k=k,
            filter=Query().filter("category", "=", filter_category),
            vector_field="embedding"
        )
        
        if self.verbose:
            self.log(f"\nTop results in '{filter_category}' category:")
            for i, result in enumerate(filtered_results, 1):
                self.log(f"{i}. {result.record['title']} (Score: {result.score:.3f})")
        
        return filtered_results
    
    async def cleanup(self):
        """Clean up database connection."""
        if self.db:
            await self.db.close()
            self.log("\n✓ Database connection closed")


async def main():
    """Run the basic vector search example."""
    
    # Create example instance
    example = VectorSearchExample(verbose=True)
    
    try:
        # Setup database
        await example.setup_database()
        
        # Create documents
        record_ids, records = await example.create_documents_with_embeddings()
        
        # Perform searches
        query_text = "How do neural networks work in AI?"
        results = await example.perform_vector_search(query_text)
        
        # Filtered search
        filtered_results = await example.perform_filtered_search(query_text, "AI")
        
        # Find similar documents
        example.log("\n5. Finding similar documents...")
        first_doc = records[0].data
        reference_embedding = first_doc["embedding"].vector
        
        similar_results = await example.db.vector_search(
            query_vector=reference_embedding,
            k=3,
            vector_field="embedding"
        )
        
        example.log(f"\nDocuments similar to '{first_doc['title']}':")
        for i, result in enumerate(similar_results, 1):
            if result.record['title'] != first_doc['title']:
                example.log(f"{i}. {result.record['title']} (Score: {result.score:.3f})")
        
        # Demonstrate similarity metrics
        example.log("\n6. Testing different similarity metrics...")
        vec1 = np.array(records[0].data["embedding"].vector)
        vec2 = np.array(records[1].data["embedding"].vector)
        
        # Calculate cosine similarity with safe division
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            cosine_sim = 0.0  # Define similarity as 0 when either vector is zero
        else:
            cosine_sim = np.dot(vec1, vec2) / (norm1 * norm2)
        example.log(f"Cosine similarity between first two docs: {cosine_sim:.3f}")
        
        euclidean_dist = np.linalg.norm(vec1 - vec2)
        example.log(f"Euclidean distance between first two docs: {euclidean_dist:.3f}")
        
        # Query builder methods
        example.log("\n7. Using Query builder methods...")
        
        query = Query().near_text(
            text="machine learning and artificial intelligence",
            embedding_fn=example.generate_embedding,
            field="embedding",
            k=2
        )
        
        near_text_results = await example.db.find(query)
        example.log(f"\nNear text search results:")
        for result in near_text_results:
            example.log(f"- {result['title']}")
        
        example.log("\n✓ Example completed successfully!")
        
    finally:
        # Cleanup
        await example.cleanup()


if __name__ == "__main__":
    asyncio.run(main())