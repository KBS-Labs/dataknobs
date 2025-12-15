# pgvector Backend

## Overview

The pgvector backend provides production-ready vector similarity search using PostgreSQL with the [pgvector](https://github.com/pgvector/pgvector) extension. It combines PostgreSQL's reliability with efficient approximate nearest neighbor (ANN) search.

**Location**: `dataknobs_data.vector.stores.pgvector.PgVectorStore`

## Installation

```bash
pip install asyncpg
```

PostgreSQL setup:
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

## Quick Start

```python
from dataknobs_data.vector.stores import VectorStoreFactory
import numpy as np

factory = VectorStoreFactory()
store = factory.create(
    backend="pgvector",
    connection_string="postgresql://user:pass@localhost:5432/mydb",
    dimensions=768,
    metric="cosine"
)

await store.initialize()

# Add vectors
vectors = np.random.rand(100, 768).astype(np.float32)
ids = await store.add_vectors(vectors, metadata=[{"source": f"doc_{i}"} for i in range(100)])

# Search
results = await store.search(np.random.rand(768).astype(np.float32), k=5)

await store.close()
```

## Configuration

### Connection & Basic Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `connection_string` | str | `DATABASE_URL` env | PostgreSQL connection URL |
| `dimensions` | int | Required | Vector dimensions |
| `metric` | str | `"cosine"` | Distance metric: `cosine`, `euclidean`, `inner_product` |
| `schema` | str | `"edubot"` | Database schema |
| `table_name` | str | `"knowledge_embeddings"` | Table name |
| `pool_min_size` | int | 2 | Min connection pool size |
| `pool_max_size` | int | 10 | Max connection pool size |

### Table Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `auto_create_table` | bool | `True` | Auto-create table if missing |
| `id_type` | str | `"uuid"` | ID type: `uuid` or `text` |
| `columns` | dict | Default mappings | Column name overrides |
| `domain_id` | str | None | Multi-tenant domain filter |

### Index Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `index_type` | str | `"none"` | Index: `none`, `hnsw`, `ivfflat` |
| `auto_create_index` | bool | `False` | Auto-create index |
| `min_rows_for_index` | int | 1000 | Min rows for IVFFlat auto-creation |
| `index_params` | dict | `{}` | Index parameters |

## Index Types

### HNSW

Best for most production use cases. Works with empty tables.

```python
store = factory.create(
    backend="pgvector",
    dimensions=768,
    index_type="hnsw",
    auto_create_index=True,
    index_params={
        "m": 16,               # Connections per layer
        "ef_construction": 64  # Build quality
    }
)
```

### IVFFlat

Best for very large datasets. Requires existing data for clustering.

```python
store = factory.create(
    backend="pgvector",
    dimensions=768,
    index_type="ivfflat",
    auto_create_index=True,
    min_rows_for_index=1000,
    index_params={"lists": 100}  # Number of clusters
)
```

### Explicit Index Creation

```python
await store.create_index("hnsw", {"m": 32, "ef_construction": 128})
# or
await store.create_index("ivfflat", {"lists": 200})
```

## Column Mappings

Default columns with customization:

```python
store = factory.create(
    backend="pgvector",
    dimensions=768,
    columns={
        "id": "product_id",
        "embedding": "vector_data",
        "content": "description",
        "metadata": "attributes"
    },
    id_type="text"
)
```

Default column names: `id`, `embedding`, `content`, `metadata`, `domain_id`, `document_id`, `chunk_index`, `created_at`

## Multi-tenant Usage

```python
tenant_store = factory.create(
    backend="pgvector",
    dimensions=768,
    domain_id="tenant_123"  # All operations filtered to this domain
)
```

## API Reference

### Core Methods

```python
# Initialize connection
await store.initialize()

# Add vectors
ids = await store.add_vectors(
    vectors,                    # np.ndarray
    ids=None,                   # Optional list[str]
    metadata=None               # Optional list[dict]
)

# Search
results = await store.search(
    query_vector,               # np.ndarray
    k=10,                       # Number of results
    filter=None,                # Metadata filter dict
    include_metadata=True
)  # Returns list[tuple[id, score, metadata]]

# Get vectors
results = await store.get_vectors(
    ids,                        # list[str]
    include_metadata=True
)  # Returns list[tuple[vector, metadata]]

# Delete vectors
count = await store.delete_vectors(ids)

# Update metadata
count = await store.update_metadata(ids, metadata_list)

# Count
total = await store.count(filter=None)

# Clear
await store.clear()

# Create index
created = await store.create_index(
    index_type="hnsw",          # or "ivfflat"
    params={"m": 16},           # Index parameters
    if_not_exists=True
)

# Close
await store.close()
```

## Distance Metrics

| Metric | pgvector Op | Score Conversion |
|--------|-------------|------------------|
| `cosine` | `<=>` | `1 - distance` |
| `euclidean` | `<->` | `1 / (1 + distance)` |
| `inner_product` | `<#>` | Raw value |

## Distributed Environments

The backend queries PostgreSQL's `pg_indexes` catalog for index existence checks, making it safe for multi-instance deployments (AWS ECS, Kubernetes, etc.).

## Auto-Created Schema

```sql
CREATE TABLE IF NOT EXISTS {schema}.{table_name} (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    domain_id VARCHAR(100),
    document_id VARCHAR(255),
    chunk_index INTEGER,
    content TEXT,
    embedding vector({dimensions}),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW()
);
```

## See Also

- [Vector Store Design](history/vector-implementation/VECTOR_STORE_DESIGN_V2.md)
- [API Reference](API_REFERENCE.md)
