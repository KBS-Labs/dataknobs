# pgvector Backend

## Overview

The pgvector backend provides production-ready vector similarity search using PostgreSQL with the [pgvector](https://github.com/pgvector/pgvector) extension. It combines the reliability and features of PostgreSQL with efficient approximate nearest neighbor (ANN) search capabilities.

## Features

- **Production-ready** - Built on battle-tested PostgreSQL infrastructure
- **Multiple index types** - HNSW and IVFFlat for different use cases
- **Flexible configuration** - Custom column mappings, schemas, and ID types
- **Multi-tenant support** - Domain-based isolation for SaaS applications
- **Connection pooling** - Built-in async connection pool management
- **Auto table creation** - Automatically creates tables and indexes
- **Distributed-safe** - Index management works correctly in multi-instance deployments

## Installation

```bash
# Install the data package with asyncpg
pip install dataknobs-data asyncpg

# Or with the postgres extra
pip install "dataknobs-data[postgres]"
```

### PostgreSQL Setup

Ensure pgvector is installed on your PostgreSQL server:

```sql
-- Connect as superuser and create the extension
CREATE EXTENSION IF NOT EXISTS vector;
```

## Quick Start

```python
from dataknobs_data.vector.stores import VectorStoreFactory
import numpy as np

# Create store using factory
factory = VectorStoreFactory()
store = factory.create(
    backend="pgvector",
    connection_string="postgresql://user:pass@localhost:5432/mydb",
    dimensions=768,
    metric="cosine"
)

# Initialize connection
await store.initialize()

# Add vectors
vectors = np.random.rand(100, 768).astype(np.float32)
metadata = [{"source": f"doc_{i}"} for i in range(100)]
ids = await store.add_vectors(vectors, metadata=metadata)

# Search
query = np.random.rand(768).astype(np.float32)
results = await store.search(query, k=5)

for id, score, meta in results:
    print(f"ID: {id}, Score: {score:.4f}, Source: {meta['source']}")

# Cleanup
await store.close()
```

## Configuration Options

### Basic Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `connection_string` | str | None | PostgreSQL connection URL. Falls back to `DATABASE_URL` env var |
| `dimensions` | int | Required | Vector dimensions (e.g., 768 for sentence-transformers) |
| `metric` | str | `"cosine"` | Distance metric: `cosine`, `euclidean`, `inner_product` |
| `schema` | str | `"edubot"` | Database schema name |
| `table_name` | str | `"knowledge_embeddings"` | Table name for vectors |

### Connection Pool Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `pool_min_size` | int | 2 | Minimum connections in pool |
| `pool_max_size` | int | 10 | Maximum connections in pool |

### Table Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `auto_create_table` | bool | `True` | Create table if it doesn't exist |
| `id_type` | str | `"uuid"` | ID column type: `uuid` or `text` |
| `columns` | dict | See below | Column name mappings |

### Index Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `index_type` | str | `"none"` | Index type: `none`, `hnsw`, or `ivfflat` |
| `auto_create_index` | bool | `False` | Automatically create index when conditions are met |
| `min_rows_for_index` | int | 1000 | Minimum rows before auto-creating IVFFlat index |
| `index_params` | dict | `{}` | Index-specific parameters |

### Multi-tenant Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `domain_id` | str | None | Domain ID for filtering (multi-tenant isolation) |

## Index Types

### No Index (Default)

Best for small datasets (<1,000 vectors) where exact search is acceptable:

```python
store = factory.create(
    backend="pgvector",
    connection_string="postgresql://...",
    dimensions=768,
    index_type="none"  # Default, explicit for clarity
)
```

### HNSW Index

Hierarchical Navigable Small World graphs. Best for most production use cases:

- Works with empty tables (can be created immediately)
- Good balance of speed and recall
- Higher memory usage

```python
store = factory.create(
    backend="pgvector",
    connection_string="postgresql://...",
    dimensions=768,
    index_type="hnsw",
    auto_create_index=True,
    index_params={
        "m": 16,              # Connections per layer (higher = better recall, more memory)
        "ef_construction": 64  # Build quality (higher = better recall, slower build)
    }
)
```

### IVFFlat Index

Inverted File with Flat quantization. Best for very large datasets:

- Requires existing data for clustering
- Lower memory usage
- Faster index creation

```python
store = factory.create(
    backend="pgvector",
    connection_string="postgresql://...",
    dimensions=768,
    index_type="ivfflat",
    auto_create_index=True,
    min_rows_for_index=1000,  # Create index when this threshold is reached
    index_params={
        "lists": 100  # Number of clusters (sqrt(n) is a good starting point)
    }
)
```

### Explicit Index Creation

For more control, create indexes explicitly:

```python
# Initialize without auto-index
store = factory.create(
    backend="pgvector",
    connection_string="postgresql://...",
    dimensions=768,
    auto_create_index=False
)
await store.initialize()

# Add data first (required for IVFFlat)
await store.add_vectors(vectors, metadata=metadata)

# Create index with custom parameters
await store.create_index(
    index_type="ivfflat",
    params={"lists": 200}
)
```

## Column Mappings

Customize column names to match your existing schema:

```python
store = factory.create(
    backend="pgvector",
    connection_string="postgresql://...",
    dimensions=768,
    table_name="product_embeddings",
    columns={
        "id": "product_id",
        "embedding": "vector_data",
        "content": "description",
        "metadata": "attributes",
        "domain_id": "category",
        "document_id": "source_file",
        "chunk_index": "segment_num",
        "created_at": "indexed_at"
    },
    id_type="text",
    auto_create_table=True
)
```

Default column names:

| Logical Name | Default Column | Description |
|--------------|----------------|-------------|
| `id` | `id` | Primary key |
| `embedding` | `embedding` | Vector data |
| `content` | `content` | Source text |
| `metadata` | `metadata` | JSONB metadata |
| `domain_id` | `domain_id` | Multi-tenant domain |
| `document_id` | `document_id` | Source document reference |
| `chunk_index` | `chunk_index` | Chunk sequence number |
| `created_at` | `created_at` | Timestamp |

## Multi-tenant Usage

Isolate data by domain for SaaS applications:

```python
# Create store for a specific tenant
tenant_store = factory.create(
    backend="pgvector",
    connection_string="postgresql://...",
    dimensions=768,
    domain_id="tenant_123"  # All operations filtered to this domain
)
await tenant_store.initialize()

# Add vectors (automatically tagged with domain_id)
await tenant_store.add_vectors(vectors, metadata=metadata)

# Search only returns vectors from this domain
results = await tenant_store.search(query, k=10)

# Count only vectors in this domain
count = await tenant_store.count()

# Clear only clears vectors in this domain
await tenant_store.clear()
```

## Usage Examples

### Basic CRUD Operations

```python
# Add vectors
vectors = np.random.rand(10, 768).astype(np.float32)
ids = await store.add_vectors(
    vectors,
    ids=["doc_1", "doc_2", ...],  # Optional, auto-generated if not provided
    metadata=[{"source": "file.pdf"} for _ in range(10)]
)

# Get vectors by ID
results = await store.get_vectors(["doc_1", "doc_2"], include_metadata=True)
for vec, meta in results:
    if vec is not None:
        print(f"Vector shape: {vec.shape}, Metadata: {meta}")

# Update metadata
await store.update_metadata(
    ["doc_1", "doc_2"],
    [{"source": "updated.pdf"}, {"source": "new.pdf"}]
)

# Delete vectors
deleted_count = await store.delete_vectors(["doc_1", "doc_2"])

# Count vectors
total = await store.count()
filtered = await store.count(filter={"category": "science"})

# Clear all vectors
await store.clear()
```

### Search with Filters

```python
# Search with metadata filters
results = await store.search(
    query_vector,
    k=10,
    filter={"category": "technology", "year": "2024"},
    include_metadata=True
)

for id, score, metadata in results:
    print(f"ID: {id}")
    print(f"Score: {score:.4f}")
    print(f"Content: {metadata.get('content', '')[:100]}...")
```

### Using with Context Manager

```python
from dataknobs_data.vector.stores.pgvector import PgVectorStore

config = {
    "connection_string": "postgresql://...",
    "dimensions": 768,
    "metric": "cosine"
}

store = PgVectorStore(config)
try:
    await store.initialize()
    # Use store...
finally:
    await store.close()
```

## Distance Metrics

| Metric | Operator | Use Case |
|--------|----------|----------|
| `cosine` | `<=>` | Normalized embeddings, semantic similarity |
| `euclidean` | `<->` | Spatial data, when magnitude matters |
| `inner_product` | `<#>` | Dot product similarity (MaxSim) |

The score returned by `search()` is converted to a similarity score:

- **Cosine**: `1 - distance` (0 = different, 1 = identical)
- **Euclidean**: `1 / (1 + distance)` (0 = far, 1 = close)
- **Inner Product**: Raw inner product value

## Performance Optimization

### Index Selection Guide

| Dataset Size | Recommended Index | Parameters |
|--------------|-------------------|------------|
| < 1,000 | None | Exact search is fast enough |
| 1,000 - 100,000 | HNSW | `m=16, ef_construction=64` |
| 100,000 - 1,000,000 | HNSW or IVFFlat | HNSW: `m=32, ef_construction=128` |
| > 1,000,000 | IVFFlat | `lists=sqrt(n)` |

### Connection Pool Tuning

```python
# For high-throughput applications
store = factory.create(
    backend="pgvector",
    connection_string="postgresql://...",
    dimensions=768,
    pool_min_size=5,   # Keep more connections ready
    pool_max_size=20   # Allow more concurrent operations
)
```

### Batch Operations

```python
# Add vectors in batches for better performance
batch_size = 1000
for i in range(0, len(all_vectors), batch_size):
    batch = all_vectors[i:i+batch_size]
    batch_meta = all_metadata[i:i+batch_size]
    await store.add_vectors(batch, metadata=batch_meta)
```

## Production Considerations

### Environment Variables

```bash
# Connection string via environment
export DATABASE_URL="postgresql://user:pass@host:5432/db"
```

```python
# Store will use DATABASE_URL if connection_string not provided
store = factory.create(
    backend="pgvector",
    dimensions=768
)
```

### Index Management in Distributed Environments

The pgvector backend safely handles index creation in multi-instance deployments by querying the PostgreSQL `pg_indexes` catalog rather than using in-memory state:

```python
# Safe for AWS ECS, Kubernetes, etc.
store = factory.create(
    backend="pgvector",
    connection_string="postgresql://...",
    dimensions=768,
    index_type="ivfflat",
    auto_create_index=True,
    min_rows_for_index=1000
)
# Multiple instances can safely run - only one will create the index
```

### Schema Migration

The auto-created table schema:

```sql
CREATE TABLE IF NOT EXISTS {schema}.{table_name} (
    {id} UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    {domain_id} VARCHAR(100),
    {document_id} VARCHAR(255),
    {chunk_index} INTEGER,
    {content} TEXT,
    {embedding} vector({dimensions}),
    {metadata} JSONB DEFAULT '{}',
    {created_at} TIMESTAMP DEFAULT NOW()
);
```

## Troubleshooting

### Common Issues

**"pgvector extension not installed"**

```sql
-- Run as superuser
CREATE EXTENSION vector;
```

**"asyncpg is not installed"**

```bash
pip install asyncpg
```

**Poor search recall with IVFFlat**

- Increase `lists` parameter (try `sqrt(n)` where n is row count)
- Ensure index was created after sufficient data was loaded
- Consider switching to HNSW for better recall

**Slow index creation**

- HNSW: Lower `ef_construction` (trades build time for recall)
- IVFFlat: Fewer `lists` (trades recall for speed)

## See Also

- [Backends Overview](backends.md)
- [PostgreSQL Backend](postgres-backend.md)
- [Performance Tuning](performance-tuning.md)
- [pgvector Documentation](https://github.com/pgvector/pgvector)
