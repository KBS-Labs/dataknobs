# Getting Started with DataKnobs Vector Store

This guide will help you quickly get started with vector search capabilities in DataKnobs. Vector search enables semantic similarity search, allowing you to find records based on meaning rather than exact keyword matches.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Backend Selection](#backend-selection)
- [Basic Operations](#basic-operations)
- [Text Synchronization](#text-synchronization)
- [Query Methods](#query-methods)
- [Migration Guide](#migration-guide)
- [Performance Tips](#performance-tips)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### Required Dependencies

```bash
# Core DataKnobs package
pip install dataknobs-data

# For embedding generation (choose one)
pip install openai          # For OpenAI embeddings
pip install sentence-transformers  # For local embeddings
pip install cohere          # For Cohere embeddings
```

### Backend-Specific Requirements

#### PostgreSQL with pgvector
```bash
# Install PostgreSQL and pgvector extension
brew install postgresql
brew install pgvector  # macOS

# Or using Docker
docker run -d \
  --name pgvector \
  -e POSTGRES_PASSWORD=password \
  -p 5432:5432 \
  ankane/pgvector
```

#### Elasticsearch
```bash
# Using Docker
docker run -d \
  --name elasticsearch \
  -p 9200:9200 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  elasticsearch:8.11.0
```

#### SQLite
No additional setup required - SQLite vector support is built-in using Python.

#### Specialized Vector Stores
```bash
# For Faiss
pip install faiss-cpu  # or faiss-gpu for GPU support

# For Chroma
pip install chromadb
```

## Quick Start

### 1. Create a Vector-Enabled Database

```python
from dataknobs_data import DatabaseFactory, Record, VectorField
import numpy as np

# Create database with vector support
db = await DatabaseFactory.create_async(
    backend="postgres",
    host="localhost",
    database="myapp",
    user="user",
    password="password",
    vector_enabled=True,  # Enable vector support
    vector_metric="cosine"  # Distance metric
)

# Initialize the database
await db.initialize()
```

### 2. Define Your Schema with Vectors

```python
from dataknobs_data import Field, VectorField

# Define a document schema with vector field
class Document:
    id = Field(primary_key=True)
    title = Field(type="text", required=True)
    content = Field(type="text")
    category = Field(type="text", indexed=True)
    embedding = VectorField(dimensions=768)  # For BERT/OpenAI embeddings
    metadata = Field(type="json")
```

### 3. Generate Embeddings

```python
# Using OpenAI
import openai

def generate_embedding(text: str) -> list[float]:
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response['data'][0]['embedding']

# Using Sentence Transformers (local)
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_embedding(text: str) -> list[float]:
    embedding = model.encode(text)
    return embedding.tolist()
```

### 4. Store Documents with Vectors

```python
# Create and store a document with vector
text = "Machine learning is a subset of artificial intelligence"
embedding = generate_embedding(text)

record = Record({
    "title": "Introduction to ML",
    "content": text,
    "category": "technology",
    "embedding": VectorField(embedding, dimensions=768),
    "metadata": {"author": "John Doe", "date": "2024-01-15"}
})

record_id = await db.create(record)
print(f"Created record: {record_id}")
```

### 5. Search by Vector Similarity

```python
# Search for similar documents
query_text = "AI and neural networks"
query_embedding = generate_embedding(query_text)

# Find top 5 similar documents
results = await db.vector_search(
    query_vector=query_embedding,
    k=5,
    vector_field="embedding"
)

for result in results:
    print(f"Score: {result.score:.3f} - {result.record['title']}")
```

## Backend Selection

### Choosing the Right Backend

| Backend | Best For | Pros | Cons |
|---------|----------|------|------|
| **PostgreSQL + pgvector** | Production apps with existing PostgreSQL | - ACID compliant<br>- SQL queries<br>- Reliable | - Requires extension<br>- Limited index types |
| **Elasticsearch** | Full-text + vector hybrid search | - Powerful text search<br>- Distributed<br>- Fast | - Resource intensive<br>- Complex setup |
| **SQLite** | Development, testing, small datasets | - Zero setup<br>- Embedded<br>- Simple | - Python-based search<br>- Not for large scale |
| **Faiss** | High-performance similarity search | - Very fast<br>- Multiple index types<br>- GPU support | - Vector-only<br>- No filtering |
| **Chroma** | RAG applications, prototyping | - Built for LLMs<br>- Metadata filtering<br>- Simple API | - Limited scalability<br>- Newer, less mature |

### Configuration Examples

#### PostgreSQL Configuration
```python
config = {
    "backend": "postgres",
    "host": "localhost",
    "port": 5432,
    "database": "vectors_db",
    "user": "postgres",
    "password": "password",
    "vector_enabled": True,
    "vector_metric": "cosine",  # cosine, euclidean, or dot_product
    "vector_index_type": "ivfflat",  # or hnsw
    "vector_index_lists": 100  # for ivfflat
}
db = await DatabaseFactory.create_async(**config)
```

#### Elasticsearch Configuration
```python
config = {
    "backend": "elasticsearch",
    "hosts": ["localhost:9200"],
    "index": "documents",
    "vector_enabled": True,
    "vector_dimensions": 768,
    "vector_similarity": "cosine",
    "vector_index_options": {
        "type": "hnsw",
        "m": 16,
        "ef_construction": 200
    }
}
db = await DatabaseFactory.create_async(**config)
```

#### Faiss Configuration
```python
from dataknobs_data.vector import FaissVectorStore

store = FaissVectorStore(
    dimension=768,
    index_type="HNSW",  # or "Flat", "IVFFlat"
    metric="cosine",
    index_params={"M": 32, "efConstruction": 200}
)
await store.initialize()
```

## Basic Operations

### Creating Records with Vectors

```python
# Single record creation
record = Record({
    "title": "Understanding Neural Networks",
    "content": "Neural networks are computing systems...",
    "embedding": VectorField(generate_embedding(content), dimensions=768)
})
record_id = await db.create(record)

# Batch creation for better performance
records = []
for doc in documents:
    embedding = generate_embedding(doc['content'])
    records.append(Record({
        "title": doc['title'],
        "content": doc['content'],
        "embedding": VectorField(embedding, dimensions=768)
    }))

record_ids = await db.create_many(records)
```

### Updating Vectors

```python
# Update vector for existing record
new_content = "Updated content about deep learning..."
new_embedding = generate_embedding(new_content)

await db.update(
    record_id,
    {
        "content": new_content,
        "embedding": VectorField(new_embedding, dimensions=768)
    }
)
```

### Deleting Records with Vectors

```python
# Delete by ID
await db.delete(record_id)

# Delete by query
from dataknobs_data import Query

query = Query().filter("category", "=", "outdated")
await db.delete_many(query)
```

## Text Synchronization

### Automatic Vector Updates

Keep vectors in sync with text changes automatically:

```python
from dataknobs_data.vector import VectorTextSynchronizer

# Create synchronizer with embedding function
sync = VectorTextSynchronizer(
    database=db,
    embedding_function=generate_embedding
)

# Configure synchronization
await sync.setup(
    text_fields=["title", "content"],  # Fields to concatenate
    vector_field="embedding",
    separator=" "  # Join text fields with space
)

# Bulk synchronize existing records
await sync.bulk_sync(batch_size=100)

# Enable automatic sync on updates
sync.enable_auto_sync()
```

### Change Tracking

Monitor which records need vector updates:

```python
from dataknobs_data.vector import ChangeTracker

tracker = ChangeTracker(db)

# Track changes to text fields
await tracker.start_tracking(
    tracked_fields=["title", "content"],
    vector_field="embedding"
)

# Get records needing updates
outdated_records = await tracker.get_outdated_records()
print(f"Records needing vector updates: {len(outdated_records)}")

# Update vectors for outdated records
for record in outdated_records:
    text = f"{record['title']} {record['content']}"
    embedding = generate_embedding(text)
    await db.update(record['id'], {"embedding": embedding})
    await tracker.mark_updated(record['id'])
```

## Query Methods

### Vector Similarity Search

```python
from dataknobs_data import Query

# Basic vector search
results = await db.vector_search(
    query_vector=query_embedding,
    k=10,
    vector_field="embedding"
)

# With metadata filtering
results = await db.vector_search(
    query_vector=query_embedding,
    k=10,
    filter=Query().filter("category", "=", "technology"),
    vector_field="embedding"
)

# With distance threshold
results = await db.vector_search(
    query_vector=query_embedding,
    k=10,
    min_score=0.7,  # Only return results with cosine similarity > 0.7
    vector_field="embedding"
)
```

### Query Builder Methods

```python
# Similar to a specific vector
query = Query().similar_to(
    vector=reference_embedding,
    k=5,
    vector_field="embedding"
)
results = await db.find(query)

# Near text (automatic embedding)
query = Query().near_text(
    text="machine learning algorithms",
    k=5,
    embedding_function=generate_embedding
)
results = await db.find(query)

# Hybrid search (text + vector)
query = Query().hybrid(
    text_query="neural network",
    text_fields=["title", "content"],
    vector=query_embedding,
    vector_field="embedding",
    alpha=0.5  # Balance between text (0) and vector (1) search
)
results = await db.find(query)
```

### Complex Queries

```python
from dataknobs_data import ComplexQuery

# Combine vector search with multiple filters
complex_query = ComplexQuery.AND([
    Query().similar_to(query_embedding, k=20),
    Query().filter("category", "in", ["AI", "ML", "DL"]),
    Query().filter("metadata.year", ">=", 2020)
])
results = await db.find(complex_query)

# OR logic with vector search
complex_query = ComplexQuery.OR([
    Query().similar_to(embedding_a, k=10),
    Query().similar_to(embedding_b, k=10)
])
results = await db.find(complex_query)
```

## Migration Guide

### Migrating Existing Data

Add vectors to existing records:

```python
from dataknobs_data.vector import VectorMigration

# Create migration
migration = VectorMigration(
    source_db=db,
    embedding_function=generate_embedding
)

# Configure migration
await migration.configure(
    text_fields=["title", "content"],
    vector_field="embedding",
    dimensions=768
)

# Run migration
await migration.run(
    batch_size=100,
    progress_callback=lambda done, total: print(f"Progress: {done}/{total}")
)
```

### Incremental Vectorization

Process large datasets incrementally:

```python
from dataknobs_data.vector import IncrementalVectorizer

vectorizer = IncrementalVectorizer(
    database=db,
    embedding_function=generate_embedding
)

# Start background vectorization
await vectorizer.start(
    text_fields=["content"],
    vector_field="embedding",
    batch_size=50,
    rate_limit=10  # Max 10 batches per minute
)

# Check progress
status = await vectorizer.get_status()
print(f"Vectorized: {status['completed']}/{status['total']}")

# Stop when needed
await vectorizer.stop()
```

## Performance Tips

### 1. Index Management

```python
# Create vector index for PostgreSQL
await db.execute_raw("""
    CREATE INDEX ON documents 
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
""")

# For Elasticsearch, configure during creation
config = {
    "vector_index_options": {
        "type": "hnsw",
        "m": 16,  # Higher = better recall, more memory
        "ef_construction": 200  # Higher = better quality, slower indexing
    }
}
```

### 2. Batch Operations

```python
# Batch embedding generation
texts = [doc['content'] for doc in documents]
embeddings = model.encode(texts, batch_size=32)  # Process in batches

# Batch insertion
records_with_vectors = []
for doc, embedding in zip(documents, embeddings):
    records_with_vectors.append(Record({
        **doc,
        "embedding": VectorField(embedding.tolist(), dimensions=768)
    }))

await db.create_many(records_with_vectors, batch_size=100)
```

### 3. Caching Strategies

```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=1000)
def cached_embedding(text: str) -> tuple:
    # Cache embeddings for repeated texts
    embedding = generate_embedding(text)
    return tuple(embedding)  # Convert to tuple for hashing

# Use with cache
text = "Frequently searched text"
embedding = list(cached_embedding(text))
```

### 4. Dimension Reduction

```python
from sklearn.decomposition import PCA

# Reduce dimensions for faster search
pca = PCA(n_components=256)  # Reduce from 768 to 256
reduced_embeddings = pca.fit_transform(embeddings)

# Store reduced embeddings
record = Record({
    "content": text,
    "embedding_full": VectorField(full_embedding, dimensions=768),
    "embedding_reduced": VectorField(reduced_embedding, dimensions=256)
})
```

## Troubleshooting

### Common Issues and Solutions

#### 1. "pgvector extension not found"
```bash
# Install pgvector extension
CREATE EXTENSION vector;

# Or in your Python code
await db.execute_raw("CREATE EXTENSION IF NOT EXISTS vector;")
```

#### 2. "Dimension mismatch"
```python
# Ensure consistent dimensions
assert len(embedding) == 768, f"Expected 768 dimensions, got {len(embedding)}"

# Validate before storing
if len(embedding) != expected_dimensions:
    raise ValueError(f"Embedding dimension mismatch")
```

#### 3. "Slow vector search"
```python
# Check if index exists
indexes = await db.execute_raw("""
    SELECT indexname FROM pg_indexes 
    WHERE tablename = 'documents';
""")

# Create index if missing
if not any('embedding' in idx for idx in indexes):
    await db.execute_raw("""
        CREATE INDEX ON documents 
        USING ivfflat (embedding vector_cosine_ops);
    """)
```

#### 4. "Out of memory with large datasets"
```python
# Process in smaller batches
async def process_large_dataset(records, batch_size=50):
    for i in range(0, len(records), batch_size):
        batch = records[i:i+batch_size]
        embeddings = generate_embeddings_batch(batch)
        await store_batch(batch, embeddings)
        
        # Optional: Clear cache
        import gc
        gc.collect()
```

### Debug Helpers

```python
# Check vector field configuration
from dataknobs_data import inspect_schema

schema = await inspect_schema(db)
vector_fields = [
    field for field in schema.fields 
    if isinstance(field, VectorField)
]
print(f"Vector fields: {vector_fields}")

# Verify vector dimensions
sample_record = await db.find_one()
if 'embedding' in sample_record:
    print(f"Vector dimensions: {len(sample_record['embedding'])}")

# Test similarity calculation
from dataknobs_data.vector import cosine_similarity

similarity = cosine_similarity(vector1, vector2)
print(f"Similarity score: {similarity}")
```

## Next Steps

- [API Reference](./API_REFERENCE.md) - Detailed API documentation
- [Example Scripts](../examples/) - Complete working examples
- [Performance Benchmarks](./BENCHMARKS.md) - Compare backend performance
- [Best Practices](./BEST_PRACTICES.md) - Production deployment guide

## Support

For issues or questions:
- GitHub Issues: [dataknobs/issues](https://github.com/dataknobs/dataknobs/issues)
- Documentation: [docs.dataknobs.com](https://docs.dataknobs.com)
- Community: [Discord](https://discord.gg/dataknobs)