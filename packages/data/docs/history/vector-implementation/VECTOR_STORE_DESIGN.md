# Vector Store Support - Design Plan

## Executive Summary

This design document outlines the approach for adding vector store capabilities to the DataKnobs data package. We propose a hybrid architecture that extends existing backends with vector capabilities where natural (PostgreSQL pgvector, Elasticsearch) while providing a dedicated vector module for specialized stores (Faiss, AWS OpenSearch).

## Design Goals

1. **Maintain Simplicity**: Avoid unnecessary complexity while adding powerful vector search capabilities
2. **Leverage Existing Infrastructure**: Reuse connection pooling, configuration, and streaming systems
3. **Support Multiple Use Cases**:
   - Embedding storage and retrieval
   - Similarity search (k-NN, ANN)
   - Hybrid queries (structured + vector)
   - Vector indexing strategies
4. **Follow DRY Principle**: Maximize code reuse between vector and traditional backends

## Architecture Overview

### Two-Tier Approach

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────────┐  ┌─────────────────────────┐ │
│  │  Enhanced Backends        │  │  Vector Store Module    │ │
│  │  (Postgres, Elasticsearch)│  │  (Faiss, OpenSearch)    │ │
│  └──────────────────────────┘  └─────────────────────────┘ │
│           │                              │                   │
│  ┌────────▼──────────────────────────────▼─────────────┐   │
│  │           Vector Operations Interface                │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │      Existing Infrastructure (Pooling, Config)       │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Implementation Phases

### Phase 1: Core Vector Abstractions

#### 1.1 Vector Field Type
```python
# src/dataknobs_data/fields.py
class FieldType(Enum):
    # Existing types...
    VECTOR = "vector"  # New type for vector/embedding fields

class VectorField(Field):
    """Specialized field for vector/embedding data."""
    
    def __init__(self, 
                 value: np.ndarray | List[float],
                 dimensions: int,
                 metadata: Dict[str, Any] = None):
        self.dimensions = dimensions
        self.value = self._validate_vector(value, dimensions)
        super().__init__(value, FieldType.VECTOR, metadata)
    
    def _validate_vector(self, value, dimensions):
        """Validate and normalize vector input."""
        if isinstance(value, list):
            value = np.array(value)
        if value.shape[0] != dimensions:
            raise ValueError(f"Vector must have {dimensions} dimensions")
        return value.astype(np.float32)  # Standard for most vector stores
```

#### 1.2 Vector Operations Interface
```python
# src/dataknobs_data/vector/operations.py
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum

class DistanceMetric(Enum):
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"

class VectorOperations(ABC):
    """Interface for vector-specific operations."""
    
    @abstractmethod
    async def index_vectors(self, 
                           records: List[Record],
                           vector_field: str,
                           index_params: Dict[str, Any] = None) -> bool:
        """Index vectors for efficient similarity search."""
        pass
    
    @abstractmethod
    async def search_similar(self,
                            query_vector: np.ndarray,
                            k: int = 10,
                            metric: DistanceMetric = DistanceMetric.COSINE,
                            filter: Query = None) -> List[Tuple[Record, float]]:
        """Search for k most similar vectors."""
        pass
    
    @abstractmethod
    async def update_vector(self,
                           id: str,
                           vector_field: str,
                           new_vector: np.ndarray) -> bool:
        """Update a specific vector."""
        pass
    
    @abstractmethod
    async def delete_from_index(self, ids: List[str]) -> bool:
        """Remove vectors from the index."""
        pass
```

### Phase 2: Enhanced Backend Implementation

#### 2.1 PostgreSQL with pgvector
```python
# src/dataknobs_data/backends/postgres_vector.py
from .postgres import AsyncPostgresDatabase
from ..vector.operations import VectorOperations, DistanceMetric

class AsyncPostgresVectorDatabase(AsyncPostgresDatabase, VectorOperations):
    """PostgreSQL backend with pgvector extension support."""
    
    async def _initialize_vector_extension(self):
        """Ensure pgvector extension is available."""
        async with self.pool.acquire() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    
    async def _create_vector_table(self, vector_dim: int):
        """Create table with vector column."""
        query = """
        CREATE TABLE IF NOT EXISTS {table_name} (
            id TEXT PRIMARY KEY,
            data JSONB,
            embedding vector({dim}),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """.format(table_name=self.table_name, dim=vector_dim)
        
        async with self.pool.acquire() as conn:
            await conn.execute(query)
            # Create indexes for different distance metrics
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS {self.table_name}_embedding_cosine_idx 
                ON {self.table_name} 
                USING ivfflat (embedding vector_cosine_ops)
            """)
    
    async def index_vectors(self, records: List[Record], 
                          vector_field: str,
                          index_params: Dict[str, Any] = None) -> bool:
        """Index vectors using pgvector's ivfflat or hnsw."""
        # Implementation using COPY for bulk insert
        # Similar to existing bulk operations but with vector handling
        pass
    
    async def search_similar(self, query_vector: np.ndarray,
                           k: int = 10,
                           metric: DistanceMetric = DistanceMetric.COSINE,
                           filter: Query = None) -> List[Tuple[Record, float]]:
        """Vector similarity search with optional filtering."""
        operator_map = {
            DistanceMetric.COSINE: "<=>",
            DistanceMetric.EUCLIDEAN: "<->",
            DistanceMetric.DOT_PRODUCT: "<#>"
        }
        
        query = f"""
        SELECT id, data, embedding, 
               embedding {operator_map[metric]} $1 as distance
        FROM {self.table_name}
        {self._build_where_clause(filter)}
        ORDER BY distance
        LIMIT $2
        """
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, query_vector, k)
            return [(self._row_to_record(row), row['distance']) for row in rows]
```

#### 2.2 Elasticsearch with dense_vector
```python
# src/dataknobs_data/backends/elasticsearch_vector.py
from .elasticsearch_async import AsyncElasticsearchDatabase
from ..vector.operations import VectorOperations

class AsyncElasticsearchVectorDatabase(AsyncElasticsearchDatabase, VectorOperations):
    """Elasticsearch backend with dense_vector support."""
    
    async def _create_vector_mapping(self, vector_dim: int):
        """Create index mapping with dense_vector field."""
        mapping = {
            "mappings": {
                "properties": {
                    "id": {"type": "keyword"},
                    "data": {"type": "object", "enabled": True},
                    "embedding": {
                        "type": "dense_vector",
                        "dims": vector_dim,
                        "index": True,
                        "similarity": "cosine"  # or "l2_norm", "dot_product"
                    },
                    "created_at": {"type": "date"},
                    "updated_at": {"type": "date"}
                }
            }
        }
        
        await self.client.indices.create(
            index=self.index_name,
            body=mapping,
            ignore=400  # Ignore if already exists
        )
    
    async def search_similar(self, query_vector: np.ndarray,
                           k: int = 10,
                           metric: DistanceMetric = DistanceMetric.COSINE,
                           filter: Query = None) -> List[Tuple[Record, float]]:
        """KNN search using Elasticsearch's vector search."""
        
        # Build the KNN query
        knn_query = {
            "field": "embedding",
            "query_vector": query_vector.tolist(),
            "k": k,
            "num_candidates": k * 10  # For better recall
        }
        
        # Add filtering if provided
        if filter:
            knn_query["filter"] = self._build_es_filter(filter)
        
        body = {
            "knn": knn_query,
            "_source": ["id", "data", "embedding"]
        }
        
        response = await self.client.search(
            index=self.index_name,
            body=body
        )
        
        results = []
        for hit in response['hits']['hits']:
            record = self._hit_to_record(hit)
            score = hit['_score']  # ES provides similarity score
            results.append((record, score))
        
        return results
```

### Phase 3: Dedicated Vector Store Module

#### 3.1 Faiss Backend
```python
# src/dataknobs_data/vector/backends/faiss.py
import faiss
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

from ...database import AsyncDatabase
from ...records import Record
from ..operations import VectorOperations, DistanceMetric

class FaissVectorStore(AsyncDatabase, VectorOperations):
    """Specialized Faiss vector store implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.dimension = config.get('dimension', 768)
        self.index_type = config.get('index_type', 'IVF')
        self.metric = config.get('metric', DistanceMetric.COSINE)
        self.index_path = Path(config.get('index_path', './faiss_index'))
        
        self.index = None
        self.id_map = {}  # Map internal Faiss IDs to record IDs
        self.records = {}  # Store full records
        
    async def connect(self):
        """Initialize or load Faiss index."""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.index_path.exists():
            await self._load_index()
        else:
            await self._create_index()
    
    async def _create_index(self):
        """Create a new Faiss index based on configuration."""
        metric_map = {
            DistanceMetric.COSINE: faiss.METRIC_INNER_PRODUCT,
            DistanceMetric.EUCLIDEAN: faiss.METRIC_L2,
            DistanceMetric.DOT_PRODUCT: faiss.METRIC_INNER_PRODUCT
        }
        
        if self.index_type == 'Flat':
            self.index = faiss.IndexFlatIP(self.dimension)
        elif self.index_type == 'IVF':
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(
                quantizer, self.dimension, 
                100,  # nlist - number of clusters
                metric_map[self.metric]
            )
        elif self.index_type == 'HNSW':
            self.index = faiss.IndexHNSWFlat(
                self.dimension, 32,  # M - number of connections
                metric_map[self.metric]
            )
        
        # Wrap with IDMap to maintain external IDs
        self.index = faiss.IndexIDMap(self.index)
    
    async def index_vectors(self, records: List[Record], 
                          vector_field: str,
                          index_params: Dict[str, Any] = None) -> bool:
        """Add vectors to Faiss index."""
        vectors = []
        ids = []
        
        for record in records:
            if vector_field in record.fields:
                vector = record.fields[vector_field].value
                if self.metric == DistanceMetric.COSINE:
                    # Normalize for cosine similarity
                    vector = vector / np.linalg.norm(vector)
                vectors.append(vector)
                
                # Generate or use existing ID
                record_id = record.id or str(uuid.uuid4())
                faiss_id = len(self.id_map)
                
                ids.append(faiss_id)
                self.id_map[faiss_id] = record_id
                self.records[record_id] = record
        
        if vectors:
            vectors_array = np.array(vectors).astype('float32')
            ids_array = np.array(ids)
            
            # Train index if needed (for IVF)
            if hasattr(self.index, 'is_trained') and not self.index.is_trained:
                self.index.train(vectors_array)
            
            self.index.add_with_ids(vectors_array, ids_array)
            await self._save_index()
            
        return True
    
    async def search_similar(self, query_vector: np.ndarray,
                           k: int = 10,
                           metric: DistanceMetric = DistanceMetric.COSINE,
                           filter: Query = None) -> List[Tuple[Record, float]]:
        """Search for similar vectors in Faiss index."""
        if self.metric == DistanceMetric.COSINE:
            query_vector = query_vector / np.linalg.norm(query_vector)
        
        query_vector = query_vector.reshape(1, -1).astype('float32')
        
        # Search in index
        distances, indices = self.index.search(query_vector, k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1:  # Valid result
                record_id = self.id_map.get(idx)
                if record_id and record_id in self.records:
                    record = self.records[record_id]
                    
                    # Apply post-filtering if needed
                    if filter and not self._matches_filter(record, filter):
                        continue
                    
                    results.append((record, float(dist)))
        
        return results
    
    async def _save_index(self):
        """Persist Faiss index and metadata to disk."""
        faiss.write_index(self.index, str(self.index_path))
        
        metadata_path = self.index_path.with_suffix('.meta')
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'id_map': self.id_map,
                'records': self.records,
                'dimension': self.dimension,
                'metric': self.metric
            }, f)
    
    async def _load_index(self):
        """Load Faiss index and metadata from disk."""
        self.index = faiss.read_index(str(self.index_path))
        
        metadata_path = self.index_path.with_suffix('.meta')
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
            self.id_map = metadata['id_map']
            self.records = metadata['records']
            self.dimension = metadata['dimension']
            self.metric = metadata['metric']
```

### Phase 4: Query System Extension

#### 4.1 Vector Query Support
```python
# src/dataknobs_data/query.py (additions)
from dataclasses import dataclass
from typing import Optional, List, Any
import numpy as np

@dataclass
class VectorQuery:
    """Represents a vector similarity search query."""
    
    vector: np.ndarray
    vector_field: str = "embedding"
    k: int = 10
    metric: str = "cosine"
    filter: Optional[Query] = None
    include_scores: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for backend consumption."""
        return {
            "vector": self.vector.tolist(),
            "vector_field": self.vector_field,
            "k": self.k,
            "metric": self.metric,
            "filter": self.filter.to_dict() if self.filter else None,
            "include_scores": self.include_scores
        }

class Query:
    """Extended Query class with vector support."""
    
    # Existing code...
    
    def with_vector_search(self, 
                          vector: np.ndarray,
                          field: str = "embedding",
                          k: int = 10,
                          metric: str = "cosine") -> 'Query':
        """Add vector similarity search to the query."""
        self.vector_query = VectorQuery(
            vector=vector,
            vector_field=field,
            k=k,
            metric=metric,
            filter=self  # Use existing filters
        )
        return self
```

### Phase 5: Factory and Configuration

#### 5.1 Vector Store Factory
```python
# src/dataknobs_data/vector/factory.py
from typing import Dict, Any, Type
from ..factory import DatabaseFactory
from .backends.faiss import FaissVectorStore
from ..backends.postgres_vector import AsyncPostgresVectorDatabase
from ..backends.elasticsearch_vector import AsyncElasticsearchVectorDatabase

class VectorStoreFactory:
    """Factory for creating vector store instances."""
    
    VECTOR_BACKENDS = {
        'faiss': FaissVectorStore,
        'postgres_vector': AsyncPostgresVectorDatabase,
        'pgvector': AsyncPostgresVectorDatabase,
        'elasticsearch_vector': AsyncElasticsearchVectorDatabase,
        'es_vector': AsyncElasticsearchVectorDatabase,
    }
    
    @classmethod
    async def create(cls, backend: str, config: Dict[str, Any]) -> VectorOperations:
        """Create a vector store instance."""
        
        # Check if it's a specialized vector backend
        if backend in cls.VECTOR_BACKENDS:
            store_class = cls.VECTOR_BACKENDS[backend]
            store = store_class(config)
            await store.connect()
            return store
        
        # Check if standard backend supports vectors
        db = await DatabaseFactory.create(backend, config)
        if isinstance(db, VectorOperations):
            return db
        
        raise ValueError(f"Backend '{backend}' does not support vector operations")
```

#### 5.2 Configuration Examples
```python
# Example configurations for different vector stores

# PostgreSQL with pgvector
postgres_config = {
    "backend": "postgres_vector",
    "host": "localhost",
    "port": 5432,
    "database": "vectors",
    "table": "embeddings",
    "pool": {
        "min_size": 5,
        "max_size": 20
    },
    "vector": {
        "dimension": 768,
        "index_type": "ivfflat",  # or "hnsw"
        "lists": 100  # for ivfflat
    }
}

# Elasticsearch with dense_vector
es_config = {
    "backend": "elasticsearch_vector",
    "hosts": ["localhost:9200"],
    "index": "embeddings",
    "vector": {
        "dimension": 768,
        "similarity": "cosine"
    }
}

# Faiss specialized store
faiss_config = {
    "backend": "faiss",
    "dimension": 768,
    "index_type": "HNSW",  # or "IVF", "Flat"
    "metric": "cosine",
    "index_path": "./data/faiss_index",
    "hnsw_m": 32,  # HNSW connections
    "ivf_nlist": 100  # IVF clusters
}
```

### Phase 6: Integration & Bridge

#### 6.1 Unified Interface
```python
# src/dataknobs_data/vector/bridge.py
from typing import List, Tuple, Optional, Union
from ..database import AsyncDatabase
from .operations import VectorOperations
from ..records import Record

class AsyncHybridVectorDatabase:
    """Bridge between structured data and vector operations."""
    
    def __init__(self, 
                 data_backend: AsyncDatabase,
                 vector_backend: VectorOperations):
        self.data_backend = data_backend
        self.vector_backend = vector_backend
    
    async def store_with_embedding(self, 
                                  record: Record,
                                  embedding: np.ndarray,
                                  vector_field: str = "embedding") -> str:
        """Store structured data and vector together."""
        # Add vector to record
        record.fields[vector_field] = VectorField(
            value=embedding,
            dimensions=len(embedding)
        )
        
        # Store in data backend
        record_id = await self.data_backend.create(record)
        
        # Index in vector backend
        await self.vector_backend.index_vectors(
            [record], vector_field
        )
        
        return record_id
    
    async def hybrid_search(self,
                          vector: np.ndarray,
                          structured_filter: Query,
                          k: int = 10) -> List[Tuple[Record, float]]:
        """Combine vector similarity with structured filtering."""
        
        # First apply structured filter
        filtered_records = await self.data_backend.search(structured_filter)
        
        # Then perform vector search on filtered results
        if hasattr(self.vector_backend, 'search_similar'):
            return await self.vector_backend.search_similar(
                vector, k=k, filter=structured_filter
            )
        
        # Fallback: post-filter vector results
        all_results = await self.vector_backend.search_similar(vector, k=k*10)
        filtered = []
        for record, score in all_results:
            if self._matches_filter(record, structured_filter):
                filtered.append((record, score))
                if len(filtered) >= k:
                    break
        
        return filtered
```

## Testing Strategy

### Unit Tests
```python
# tests/test_vector_operations.py
import pytest
import numpy as np
from dataknobs_data.vector import VectorField, VectorQuery
from dataknobs_data.vector.backends import FaissVectorStore

@pytest.mark.asyncio
async def test_vector_field_validation():
    """Test vector field creation and validation."""
    # Valid vector
    vector = np.random.rand(768)
    field = VectorField(vector, dimensions=768)
    assert field.dimensions == 768
    assert field.type == FieldType.VECTOR
    
    # Invalid dimensions
    with pytest.raises(ValueError):
        VectorField(np.random.rand(512), dimensions=768)

@pytest.mark.asyncio
async def test_faiss_indexing():
    """Test Faiss vector indexing."""
    store = FaissVectorStore({
        'dimension': 128,
        'index_type': 'Flat',
        'metric': 'cosine'
    })
    await store.connect()
    
    # Create test records with vectors
    records = []
    for i in range(100):
        record = Record({
            'title': f'Document {i}',
            'embedding': VectorField(np.random.rand(128), 128)
        })
        records.append(record)
    
    # Index vectors
    success = await store.index_vectors(records, 'embedding')
    assert success
    
    # Search
    query_vector = np.random.rand(128)
    results = await store.search_similar(query_vector, k=5)
    assert len(results) <= 5
```

### Integration Tests
```python
# tests/integration/test_vector_backends.py
@pytest.mark.integration
@pytest.mark.asyncio
async def test_postgres_vector_search():
    """Test PostgreSQL pgvector integration."""
    db = await AsyncPostgresVectorDatabase.create({
        'host': 'localhost',
        'database': 'test_vectors',
        'table': 'embeddings'
    })
    
    # Create records with embeddings
    records = generate_test_embeddings(100, dim=768)
    
    # Bulk insert
    for record in records:
        await db.create(record)
    
    # Vector search
    query = np.random.rand(768)
    results = await db.search_similar(query, k=10)
    
    assert len(results) == 10
    # Results should be sorted by distance
    distances = [score for _, score in results]
    assert distances == sorted(distances)
```

## Migration Path

### For Existing Users

1. **No Breaking Changes**: Existing code continues to work
2. **Opt-in Vector Support**: Enable vectors by using vector-enabled backends
3. **Gradual Migration**: Can migrate backend-by-backend

### Migration Example
```python
# Before: Standard Postgres backend
db = await AsyncPostgresDatabase.create(config)

# After: Postgres with vector support (backward compatible)
db = await AsyncPostgresVectorDatabase.create(config)
# All existing operations still work
record = await db.read(record_id)

# New vector operations available
results = await db.search_similar(query_vector, k=10)
```

## Performance Considerations

### Indexing Strategies

1. **Flat Index** (Exact Search)
   - Best for < 10K vectors
   - 100% recall
   - O(n) search time

2. **IVF** (Inverted File)
   - Good for 10K - 1M vectors
   - 90-95% recall typical
   - O(sqrt(n)) search time

3. **HNSW** (Hierarchical Navigable Small World)
   - Best for > 100K vectors
   - 95-99% recall typical
   - O(log n) search time

### Optimization Tips

1. **Batch Operations**: Index vectors in batches of 1000-10000
2. **Preprocessing**: Normalize vectors for cosine similarity
3. **Index Training**: For IVF, train on representative sample
4. **Connection Pooling**: Reuse existing pool infrastructure
5. **Caching**: Cache frequently accessed vectors

## Security Considerations

1. **Vector Privacy**: Embeddings can leak information
2. **Access Control**: Extend existing RBAC to vector operations
3. **Rate Limiting**: Vector search can be compute-intensive
4. **Input Validation**: Validate vector dimensions and values

## Future Enhancements

1. **Streaming Vector Updates**: Real-time index updates
2. **Quantization**: Reduce memory usage with product quantization
3. **Multi-vector Search**: Search with multiple query vectors
4. **Hybrid Ranking**: Combine vector and keyword scores
5. **Auto-indexing**: Automatic index type selection based on size
6. **Vector Versioning**: Track embedding model versions

## Conclusion

This design provides a flexible, performant way to add vector store capabilities while maintaining the simplicity and consistency of the existing data package. The hybrid approach allows us to leverage existing infrastructure while supporting specialized vector stores, giving users the best of both worlds.

The implementation can be done incrementally:
1. Start with vector field types and basic operations
2. Add PostgreSQL pgvector support (most requested)
3. Add Elasticsearch dense_vector support
4. Implement Faiss for specialized use cases
5. Create bridge and hybrid search capabilities

This approach follows the established patterns in the codebase while adding powerful new capabilities for modern AI/ML workloads.
