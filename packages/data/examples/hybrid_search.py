#!/usr/bin/env python3
"""
Hybrid Search Example - Combining Text and Vector Search

This example demonstrates:
1. Setting up hybrid search (text + vector)
2. Balancing between keyword and semantic search
3. Using different search strategies
4. Comparing pure text, pure vector, and hybrid results
5. Optimizing hybrid search parameters

Requirements:
    pip install dataknobs-data sentence-transformers
"""

import asyncio
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import time

from dataknobs_data import AsyncDatabaseFactory, Record, VectorField, Query


# Initialize embedding model
print("Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')


def generate_embedding(text: str) -> List[float]:
    """Generate embedding for a text string."""
    embedding = model.encode(text)
    return embedding.tolist()


@dataclass
class SearchResult:
    """Store search results with metadata."""
    record: Dict[str, Any]
    score: float
    method: str  # 'text', 'vector', or 'hybrid'
    rank: int


class HybridSearchDemo:
    """Demonstrate different hybrid search strategies."""
    
    def __init__(self, db):
        self.db = db
        
    async def text_search(self, query: str, limit: int = 5) -> List[SearchResult]:
        """Perform traditional text search."""
        # Simple text matching (in real apps, use full-text search)
        results = await self.db.find(
            Query().filter("content", "contains", query).limit(limit)
        )
        
        return [
            SearchResult(
                record=r,
                score=1.0,  # Simple scoring for demo
                method="text",
                rank=i
            )
            for i, r in enumerate(results)
        ]
    
    async def vector_search(self, query: str, limit: int = 5) -> List[SearchResult]:
        """Perform pure vector similarity search."""
        query_embedding = generate_embedding(query)
        
        results = await self.db.vector_search(
            query_vector=query_embedding,
            k=limit,
            vector_field="embedding"
        )
        
        return [
            SearchResult(
                record=r.record,
                score=r.score,
                method="vector",
                rank=i
            )
            for i, r in enumerate(results)
        ]
    
    async def hybrid_search(
        self,
        query: str,
        alpha: float = 0.5,
        limit: int = 5
    ) -> List[SearchResult]:
        """
        Perform hybrid search combining text and vector.
        
        Args:
            query: Search query
            alpha: Weight between text (0) and vector (1) search
            limit: Number of results to return
        """
        query_embedding = generate_embedding(query)
        
        # Get both text and vector results
        text_results = await self.text_search(query, limit * 2)
        vector_results = await self.vector_search(query, limit * 2)
        
        # Combine and re-rank results
        combined_scores = {}
        all_records = {}
        
        # Add text search scores
        for result in text_results:
            record_id = result.record.get('id')
            combined_scores[record_id] = (1 - alpha) * result.score
            all_records[record_id] = result.record
        
        # Add vector search scores
        for result in vector_results:
            record_id = result.record.get('id')
            if record_id in combined_scores:
                combined_scores[record_id] += alpha * result.score
            else:
                combined_scores[record_id] = alpha * result.score
                all_records[record_id] = result.record
        
        # Sort by combined score
        sorted_results = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:limit]
        
        return [
            SearchResult(
                record=all_records[record_id],
                score=score,
                method="hybrid",
                rank=i
            )
            for i, (record_id, score) in enumerate(sorted_results)
        ]
    
    async def reciprocal_rank_fusion(
        self,
        query: str,
        k: int = 60,
        limit: int = 5
    ) -> List[SearchResult]:
        """
        Reciprocal Rank Fusion (RRF) for combining search results.
        More robust than simple weighted combination.
        """
        # Get both result sets
        text_results = await self.text_search(query, limit * 2)
        vector_results = await self.vector_search(query, limit * 2)
        
        # Calculate RRF scores
        rrf_scores = {}
        all_records = {}
        
        # Process text results
        for result in text_results:
            record_id = result.record.get('id')
            rrf_scores[record_id] = 1.0 / (k + result.rank + 1)
            all_records[record_id] = result.record
        
        # Process vector results
        for result in vector_results:
            record_id = result.record.get('id')
            score = 1.0 / (k + result.rank + 1)
            
            if record_id in rrf_scores:
                rrf_scores[record_id] += score
            else:
                rrf_scores[record_id] = score
                all_records[record_id] = result.record
        
        # Sort by RRF score
        sorted_results = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:limit]
        
        return [
            SearchResult(
                record=all_records[record_id],
                score=score,
                method="rrf",
                rank=i
            )
            for i, (record_id, score) in enumerate(sorted_results)
        ]


async def create_sample_database():
    """Create a database with sample documents."""
    
    print("\n1. Creating sample database...")
    
    factory = AsyncDatabaseFactory()
    db = factory.create(
        backend="sqlite",
        database=":memory:",
        vector_enabled=True,
        vector_metric="cosine"
    )
    
    await db.connect()
    
    # Sample documents covering different topics
    documents = [
        {
            "id": 1,
            "title": "Introduction to Machine Learning",
            "content": "Machine learning is a branch of artificial intelligence that enables computers to learn from data without explicit programming. It uses algorithms to identify patterns and make decisions.",
            "category": "AI",
            "keywords": ["machine learning", "AI", "algorithms", "data science"]
        },
        {
            "id": 2,
            "title": "Deep Neural Networks Explained",
            "content": "Deep learning uses neural networks with multiple layers to progressively extract higher-level features from raw input. Convolutional neural networks are particularly effective for image recognition.",
            "category": "AI",
            "keywords": ["deep learning", "neural networks", "CNN", "layers"]
        },
        {
            "id": 3,
            "title": "Natural Language Processing Fundamentals",
            "content": "NLP combines linguistics and machine learning to help computers understand, interpret, and generate human language. Applications include chatbots, translation, and sentiment analysis.",
            "category": "AI",
            "keywords": ["NLP", "linguistics", "chatbots", "language"]
        },
        {
            "id": 4,
            "title": "Python Programming Best Practices",
            "content": "Python is a versatile programming language known for clean syntax and readability. Best practices include using virtual environments, following PEP 8, and writing comprehensive tests.",
            "category": "Programming",
            "keywords": ["Python", "programming", "PEP 8", "testing"]
        },
        {
            "id": 5,
            "title": "Web Development with JavaScript",
            "content": "JavaScript powers interactive web applications. Modern frameworks like React, Vue, and Angular help developers build complex single-page applications with component-based architecture.",
            "category": "Programming",
            "keywords": ["JavaScript", "web development", "React", "frontend"]
        },
        {
            "id": 6,
            "title": "Database Design and Optimization",
            "content": "Effective database design involves normalization, indexing strategies, and query optimization. Understanding execution plans and avoiding N+1 queries improves application performance.",
            "category": "Database",
            "keywords": ["database", "SQL", "optimization", "indexing"]
        },
        {
            "id": 7,
            "title": "Cloud Computing Architecture",
            "content": "Cloud computing provides scalable infrastructure through services like AWS, Azure, and GCP. Key concepts include virtualization, containerization, and serverless computing.",
            "category": "Cloud",
            "keywords": ["cloud", "AWS", "Azure", "serverless"]
        },
        {
            "id": 8,
            "title": "Cybersecurity Fundamentals",
            "content": "Security best practices include encryption, authentication, and regular updates. Common threats include SQL injection, XSS attacks, and social engineering.",
            "category": "Security",
            "keywords": ["security", "encryption", "authentication", "threats"]
        },
        {
            "id": 9,
            "title": "Data Science with Python",
            "content": "Python libraries like pandas, numpy, and scikit-learn enable powerful data analysis. Data scientists use statistical methods and machine learning to extract insights from data.",
            "category": "Data Science",
            "keywords": ["data science", "pandas", "numpy", "statistics"]
        },
        {
            "id": 10,
            "title": "Computer Vision Applications",
            "content": "Computer vision enables machines to interpret visual information. Applications include facial recognition, object detection, and autonomous vehicles using deep learning models.",
            "category": "AI",
            "keywords": ["computer vision", "image processing", "object detection", "deep learning"]
        }
    ]
    
    # Add embeddings to documents
    for doc in documents:
        text = f"{doc['title']} {doc['content']}"
        embedding = generate_embedding(text)
        
        record = Record({
            **doc,
            "embedding": VectorField(embedding)  # Simplified - dimensions auto-detected
        })
        
        await db.create(record)
    
    print(f"✓ Created database with {len(documents)} documents")
    
    return db


async def compare_search_methods(db, query: str):
    """Compare different search methods for the same query."""
    
    print(f"\n🔍 Query: '{query}'")
    print("=" * 60)
    
    demo = HybridSearchDemo(db)
    
    # 1. Text-only search
    print("\n📝 Text Search Results:")
    start = time.time()
    text_results = await demo.text_search(query)
    text_time = time.time() - start
    
    for result in text_results[:3]:
        print(f"  {result.rank + 1}. {result.record['title']}")
        print(f"     Category: {result.record['category']}")
    print(f"  ⏱ Time: {text_time*1000:.2f}ms")
    
    # 2. Vector-only search
    print("\n🧮 Vector Search Results:")
    start = time.time()
    vector_results = await demo.vector_search(query)
    vector_time = time.time() - start
    
    for result in vector_results[:3]:
        print(f"  {result.rank + 1}. {result.record['title']} (Score: {result.score:.3f})")
        print(f"     Category: {result.record['category']}")
    print(f"  ⏱ Time: {vector_time*1000:.2f}ms")
    
    # 3. Hybrid search with different alpha values
    for alpha in [0.3, 0.5, 0.7]:
        print(f"\n🔀 Hybrid Search (alpha={alpha}):")
        start = time.time()
        hybrid_results = await demo.hybrid_search(query, alpha=alpha)
        hybrid_time = time.time() - start
        
        for result in hybrid_results[:3]:
            print(f"  {result.rank + 1}. {result.record['title']} (Score: {result.score:.3f})")
        print(f"  ⏱ Time: {hybrid_time*1000:.2f}ms")
    
    # 4. Reciprocal Rank Fusion
    print("\n🎯 Reciprocal Rank Fusion:")
    start = time.time()
    rrf_results = await demo.reciprocal_rank_fusion(query)
    rrf_time = time.time() - start
    
    for result in rrf_results[:3]:
        print(f"  {result.rank + 1}. {result.record['title']} (Score: {result.score:.4f})")
        print(f"     Category: {result.record['category']}")
    print(f"  ⏱ Time: {rrf_time*1000:.2f}ms")


async def test_query_expansion(db):
    """Demonstrate query expansion for better results."""
    
    print("\n\n🔄 Query Expansion Example")
    print("=" * 60)
    
    original_query = "AI"
    expanded_query = "AI artificial intelligence machine learning deep learning"
    
    demo = HybridSearchDemo(db)
    
    print(f"Original query: '{original_query}'")
    original_results = await demo.vector_search(original_query, limit=3)
    
    print("\nOriginal results:")
    for r in original_results:
        print(f"  - {r.record['title']}")
    
    print(f"\nExpanded query: '{expanded_query}'")
    expanded_results = await demo.vector_search(expanded_query, limit=3)
    
    print("\nExpanded results:")
    for r in expanded_results:
        print(f"  - {r.record['title']}")


async def test_filtered_hybrid_search(db):
    """Demonstrate hybrid search with filters."""
    
    print("\n\n🎛️ Filtered Hybrid Search")
    print("=" * 60)
    
    query = "programming best practices"
    query_embedding = generate_embedding(query)
    
    # Hybrid search with category filter
    print(f"Query: '{query}'")
    print("Filter: category IN ['Programming', 'AI']")
    
    # Perform filtered vector search
    filtered_results = await db.vector_search(
        query_vector=query_embedding,
        k=5,
        filter=Query().filter("category", "in", ["Programming", "AI"]),
        vector_field="embedding"
    )
    
    print("\nFiltered results:")
    for i, result in enumerate(filtered_results, 1):
        print(f"  {i}. {result.record['title']}")
        print(f"     Category: {result.record['category']}")
        print(f"     Score: {result.score:.3f}")


async def analyze_alpha_impact(db):
    """Analyze the impact of alpha parameter on hybrid search."""
    
    print("\n\n📊 Alpha Parameter Analysis")
    print("=" * 60)
    
    query = "neural networks for image recognition"
    demo = HybridSearchDemo(db)
    
    print(f"Query: '{query}'")
    print("\nTesting different alpha values (0=text only, 1=vector only):")
    
    alpha_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    results_by_alpha = {}
    
    for alpha in alpha_values:
        results = await demo.hybrid_search(query, alpha=alpha, limit=3)
        results_by_alpha[alpha] = results
        
        print(f"\nAlpha = {alpha}:")
        for r in results:
            print(f"  - {r.record['title'][:40]}... (Score: {r.score:.3f})")
    
    # Find optimal alpha (highest average score)
    avg_scores = {}
    for alpha, results in results_by_alpha.items():
        avg_score = sum(r.score for r in results) / len(results) if results else 0
        avg_scores[alpha] = avg_score
    
    optimal_alpha = max(avg_scores, key=avg_scores.get)
    print(f"\n✨ Optimal alpha for this query: {optimal_alpha}")
    print(f"   Average score: {avg_scores[optimal_alpha]:.3f}")


async def main():
    """Run the hybrid search example."""
    
    print("\n" + "="*60)
    print("Hybrid Search Example")
    print("="*60)
    
    # Create sample database
    db = await create_sample_database()
    
    # Test different queries
    test_queries = [
        "machine learning algorithms",
        "python programming",
        "neural networks deep learning",
        "database optimization SQL"
    ]
    
    # Compare search methods for each query
    for query in test_queries:
        await compare_search_methods(db, query)
    
    # Additional demonstrations
    await test_query_expansion(db)
    await test_filtered_hybrid_search(db)
    await analyze_alpha_impact(db)
    
    # Performance comparison summary
    print("\n\n📈 Performance Summary")
    print("=" * 60)
    print("Method          | Pros                          | Cons")
    print("-" * 60)
    print("Text Search     | Fast, exact matches           | Misses semantic similarity")
    print("Vector Search   | Semantic understanding        | Slower, needs embeddings")
    print("Hybrid (α=0.5)  | Balanced accuracy             | Requires tuning")
    print("RRF             | Robust, no tuning needed      | More complex")
    
    # Best practices
    print("\n\n💡 Best Practices for Hybrid Search")
    print("=" * 60)
    print("1. Start with α=0.5 and adjust based on your data")
    print("2. Use RRF when you don't want to tune parameters")
    print("3. Pre-filter with metadata for better performance")
    print("4. Cache embeddings for frequently searched queries")
    print("5. Monitor and log search quality metrics")
    print("6. Consider query expansion for short queries")
    print("7. Use different strategies for different query types")
    
    # Cleanup
    await db.close()
    print("\n✓ Hybrid search example completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())