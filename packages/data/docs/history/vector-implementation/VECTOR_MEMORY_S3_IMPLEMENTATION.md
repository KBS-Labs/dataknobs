# Vector Support Implementation for Memory, File, and S3 Backends

## Overview

This document tracks the implementation of vector support for the Memory, File, and S3 backends using the consolidated `PythonVectorSearchMixin` approach that was successfully applied to SQLite backends.

## Current State

### Backends with Vector Support
- ✅ **PostgreSQL** (native via pgvector)
- ✅ **Elasticsearch** (native via KNN)
- ✅ **SQLite** (Python-based via PythonVectorSearchMixin)

### Backends Needing Vector Support
- ❌ **Memory** (sync and async)
- ❌ **S3** (sync and async)
- ❌ **File** (sync and async)

## Implementation Strategy

### Core Components to Reuse

1. **PythonVectorSearchMixin** (`src/dataknobs_data/vector/python_vector_search.py`)
   - Provides `python_vector_search_async()` and `python_vector_search_sync()`
   - Handles Record extraction and creation
   - Manages VectorField serialization
   - Implements similarity scoring and sorting

2. **SQLiteVectorSupport** (`src/dataknobs_data/backends/sqlite_mixins.py`)
   - Provides `_compute_similarity()` method
   - Supports cosine, euclidean, and dot product metrics
   - Handles vector serialization/deserialization

3. **VectorConfigMixin** (`src/dataknobs_data/backends/vector_config_mixin.py`)
   - Parses vector configuration from config dict
   - Sets up vector_enabled, vector_dimensions, vector_metric

4. **VectorOperationsMixin** (`src/dataknobs_data/vector/mixins.py`)
   - Provides standard vector operation interfaces
   - Includes bulk_embed_and_store, add_vectors, etc.

## Implementation Plan

### Phase 1: Memory Backend (Async)

**File**: `src/dataknobs_data/backends/memory.py`

```python
# Update imports
from ..vector import VectorOperationsMixin
from ..vector.bulk_embed_mixin import BulkEmbedMixin
from ..vector.python_vector_search import PythonVectorSearchMixin
from .sqlite_mixins import SQLiteVectorSupport
from .vector_config_mixin import VectorConfigMixin

# Update class definition
class AsyncMemoryDatabase(
    AsyncDatabase,
    AsyncStreamingMixin,
    ConfigurableBase,
    VectorConfigMixin,           # Parse vector config
    SQLiteVectorSupport,          # Provides _compute_similarity
    PythonVectorSearchMixin,      # Provides python_vector_search_async
    BulkEmbedMixin,              # Bulk embedding operations
    VectorOperationsMixin        # Standard vector interface
):
    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self._storage: OrderedDict[str, Record] = OrderedDict()
        self._lock = asyncio.Lock()
        
        # Initialize vector support
        self._parse_vector_config(config or {})
        self._init_vector_state()  # From SQLiteVectorSupport
    
    async def vector_search(
        self,
        query_vector,
        vector_field: str = "embedding",
        k: int = 10,
        filter=None,
        metric=None,
        **kwargs
    ):
        """Perform vector similarity search using Python calculations."""
        return await self.python_vector_search_async(
            query_vector=query_vector,
            vector_field=vector_field,
            k=k,
            filter=filter,
            metric=metric,
            **kwargs
        )
```

### Phase 2: Memory Backend (Sync)

```python
class SyncMemoryDatabase(
    SyncDatabase,
    StreamingMixin,
    ConfigurableBase,
    VectorConfigMixin,
    SQLiteVectorSupport,
    PythonVectorSearchMixin,
    BulkEmbedMixin,
    VectorOperationsMixin
):
    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self._storage: OrderedDict[str, Record] = OrderedDict()
        self._lock = threading.Lock()
        
        # Initialize vector support
        self._parse_vector_config(config or {})
        self._init_vector_state()
    
    def vector_search(
        self,
        query_vector,
        vector_field: str = "embedding",
        k: int = 10,
        filter=None,
        metric=None,
        **kwargs
    ):
        """Perform vector similarity search using Python calculations."""
        return self.python_vector_search_sync(
            query_vector=query_vector,
            vector_field=vector_field,
            k=k,
            filter=filter,
            metric=metric,
            **kwargs
        )
```

### Phase 3: S3 Backend (Async)

**File**: `src/dataknobs_data/backends/s3_async.py`

```python
class AsyncS3Database(
    AsyncDatabase,
    ConfigurableBase,
    VectorConfigMixin,
    SQLiteVectorSupport,
    PythonVectorSearchMixin,
    BulkEmbedMixin,
    VectorOperationsMixin
):
    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        # ... existing S3 setup ...
        
        # Initialize vector support
        self._parse_vector_config(config or {})
        self._init_vector_state()
    
    async def vector_search(
        self,
        query_vector,
        vector_field: str = "embedding",
        k: int = 10,
        filter=None,
        metric=None,
        **kwargs
    ):
        """
        Perform vector similarity search using Python calculations.
        
        WARNING: This implementation downloads all records from S3 to perform
        the search locally. This is inefficient for large datasets. Consider
        using a vector-enabled backend like PostgreSQL or Elasticsearch for
        production use with large datasets.
        
        Future optimization: Override this method to use AWS OpenSearch or
        similar vector-enabled service when available.
        """
        return await self.python_vector_search_async(
            query_vector=query_vector,
            vector_field=vector_field,
            k=k,
            filter=filter,
            metric=metric,
            **kwargs
        )
```

### Phase 4: S3 Backend (Sync)

**File**: `src/dataknobs_data/backends/s3.py`

```python
class SyncS3Database(
    SyncDatabase,
    ConfigurableBase,
    VectorConfigMixin,
    SQLiteVectorSupport,
    PythonVectorSearchMixin,
    BulkEmbedMixin,
    VectorOperationsMixin
):
    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        # ... existing S3 setup ...
        
        # Initialize vector support
        self._parse_vector_config(config or {})
        self._init_vector_state()
    
    def vector_search(
        self,
        query_vector,
        vector_field: str = "embedding",
        k: int = 10,
        filter=None,
        metric=None,
        **kwargs
    ):
        """
        Perform vector similarity search using Python calculations.
        
        WARNING: This implementation downloads all records from S3 to perform
        the search locally. This is inefficient for large datasets. Consider
        using a vector-enabled backend like PostgreSQL or Elasticsearch for
        production use with large datasets.
        """
        return self.python_vector_search_sync(
            query_vector=query_vector,
            vector_field=vector_field,
            k=k,
            filter=filter,
            metric=metric,
            **kwargs
        )
```

### Phase 5: File Backend (Async)

**File**: `src/dataknobs_data/backends/file.py`

```python
class AsyncFileDatabase(
    AsyncDatabase,
    ConfigurableBase,
    VectorConfigMixin,
    SQLiteVectorSupport,
    PythonVectorSearchMixin,
    BulkEmbedMixin,
    VectorOperationsMixin
):
    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        # ... existing file setup ...
        
        # Initialize vector support
        self._parse_vector_config(config or {})
        self._init_vector_state()
    
    async def vector_search(
        self,
        query_vector,
        vector_field: str = "embedding",
        k: int = 10,
        filter=None,
        metric=None,
        **kwargs
    ):
        """
        Perform vector similarity search using Python calculations.
        
        Note: This implementation reads all records from disk to perform
        the search locally. For better performance with large datasets,
        consider using SQLite or a dedicated vector database.
        """
        return await self.python_vector_search_async(
            query_vector=query_vector,
            vector_field=vector_field,
            k=k,
            filter=filter,
            metric=metric,
            **kwargs
        )
```

### Phase 6: File Backend (Sync)

**File**: `src/dataknobs_data/backends/file.py`

```python
class SyncFileDatabase(
    SyncDatabase,
    ConfigurableBase,
    VectorConfigMixin,
    SQLiteVectorSupport,
    PythonVectorSearchMixin,
    BulkEmbedMixin,
    VectorOperationsMixin
):
    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        # ... existing file setup ...
        
        # Initialize vector support
        self._parse_vector_config(config or {})
        self._init_vector_state()
    
    def vector_search(
        self,
        query_vector,
        vector_field: str = "embedding",
        k: int = 10,
        filter=None,
        metric=None,
        **kwargs
    ):
        """
        Perform vector similarity search using Python calculations.
        
        Note: This implementation reads all records from disk to perform
        the search locally. For better performance with large datasets,
        consider using SQLite or a dedicated vector database.
        """
        return self.python_vector_search_sync(
            query_vector=query_vector,
            vector_field=vector_field,
            k=k,
            filter=filter,
            metric=metric,
            **kwargs
        )
```

### Phase 7: Factory Updates

**File**: `src/dataknobs_data/factory.py`

```python
# In DatabaseFactory.create()
if backend_type == "memory":
    config["vector_enabled"] = vector_enabled
    config["vector_dimensions"] = vector_dimensions
    config["vector_metric"] = vector_metric
    # ... rest of memory setup

if backend_type == "s3":
    config["vector_enabled"] = vector_enabled
    config["vector_dimensions"] = vector_dimensions
    config["vector_metric"] = vector_metric
    # ... rest of S3 setup

if backend_type == "file":
    config["vector_enabled"] = vector_enabled
    config["vector_dimensions"] = vector_dimensions
    config["vector_metric"] = vector_metric
    # ... rest of file setup

# Similar updates for AsyncDatabaseFactory (memory, s3, file)
```

## Testing Requirements

### Testing Infrastructure

**Environment Variables:**
- `TEST_S3=true` - Run S3 tests with existing localstack service
- `TEST_POSTGRES=true` - Run PostgreSQL tests (existing)
- `TEST_ELASTICSEARCH=true` - Run Elasticsearch tests (existing)

**Test Decorators:**
```python
import pytest
import os

# S3 test decorator (follows existing pattern)
skip_without_s3 = pytest.mark.skipif(
    os.getenv("TEST_S3") != "true",
    reason="S3 tests require TEST_S3=true and localstack running"
)

@skip_without_s3
class TestS3VectorSearch:
    # ... S3 vector tests
```

### Unit Tests

1. **Memory Vector Tests** (`tests/test_memory_vector.py`)
   - Test vector_search with different metrics
   - Test filtering during vector search
   - Test with empty database
   - Test with non-vector records
   - Test bulk_embed_and_store

2. **S3 Vector Tests** (`tests/test_s3_vector.py`)
   - Similar to memory tests
   - Add performance warning test
   - Use docker localstack service for S3 operations
   - Run tests only when `TEST_S3=true` environment variable is set
   - Follow same pattern as PostgreSQL and Elasticsearch tests

3. **File Vector Tests** (`tests/test_file_vector.py`)
   - Test vector_search with different metrics
   - Test filtering during vector search
   - Test with temporary directory
   - Test concurrent file access handling
   - Test bulk_embed_and_store

### Integration Tests

1. **Factory Integration** (`tests/test_factory_vector_integration.py`)
   - Add Memory backend to existing test suite
   - Add S3 backend with localstack (when TEST_S3=true)
   - Add File backend to existing test suite

2. **Example Scripts** (`examples/`)
   - Update `basic_vector_search.py` to show Memory backend
   - Create `s3_vector_search.py` with performance warning
   - Create `file_vector_search.py` for local file storage example

## Documentation Updates

### User-Facing Documentation

1. **Getting Started Guide** (`docs/VECTOR_GETTING_STARTED.md`)
   - Add Memory backend example (good for prototyping)
   - Add S3 backend with clear performance warnings
   - Add File backend example

2. **Backend Comparison** (`docs/VECTOR_BACKENDS.md`)
   ```markdown
   | Backend | Vector Support | Method | Performance | Use Case |
   |---------|---------------|---------|-------------|----------|
   | PostgreSQL | ✅ Native | pgvector | Excellent | Production |
   | Elasticsearch | ✅ Native | KNN | Excellent | Production |
   | SQLite | ✅ Python | Local | Good | Development |
   | Memory | ✅ Python | In-memory | Excellent | Testing/Prototyping |
   | File | ✅ Python | Local files | Moderate | Small-medium datasets |
   | S3 | ✅ Python | Download all | Poor | Small datasets only |
   ```

3. **S3 Optimization Guide** (`docs/S3_VECTOR_OPTIMIZATION.md`)
   - Explain current limitations
   - Provide override example using AWS OpenSearch
   - Suggest alternatives for large datasets

## Implementation Checklist

### Memory Backend
- [ ] Update AsyncMemoryDatabase with mixins
- [ ] Implement vector_search delegation
- [ ] Update SyncMemoryDatabase with mixins
- [ ] Implement vector_search delegation
- [ ] Add unit tests
- [ ] Add integration tests

### S3 Backend
- [ ] Update AsyncS3Database with mixins
- [ ] Implement vector_search with warning
- [ ] Update SyncS3Database with mixins
- [ ] Implement vector_search with warning
- [ ] Add unit tests with localstack (TEST_S3=true)
- [ ] Add integration tests with localstack
- [ ] Add pytest skip decorators for TEST_S3 environment variable

### File Backend
- [ ] Update AsyncFileDatabase with mixins
- [ ] Implement async vector_search delegation
- [ ] Update SyncFileDatabase with mixins
- [ ] Implement sync vector_search delegation
- [ ] Add unit tests
- [ ] Add integration tests

### Factory
- [ ] Update DatabaseFactory for Memory vector config
- [ ] Update DatabaseFactory for S3 vector config
- [ ] Update DatabaseFactory for File vector config
- [ ] Update AsyncDatabaseFactory for Memory vector config
- [ ] Update AsyncDatabaseFactory for S3 vector config
- [ ] Update AsyncDatabaseFactory for File vector config
- [ ] Add factory tests for all three backends

### Documentation
- [ ] Update Getting Started guide
- [ ] Create backend comparison table
- [ ] Write S3 optimization guide
- [ ] Update API documentation

## Performance Considerations

### Memory Backend
- **Pros**: 
  - Fastest possible Python-based search
  - No I/O overhead
  - Great for testing and prototyping
- **Cons**:
  - Limited by available RAM
  - No persistence

### File Backend
- **Pros**:
  - Local disk storage with persistence
  - Reasonable performance for small-medium datasets
  - No network overhead
  - Simple deployment (no external services)
- **Cons**:
  - Must read all files for each search
  - Performance degrades with dataset size
  - Limited by disk I/O speed
  - No concurrent write optimization

### S3 Backend
- **Current Implementation**:
  - Downloads all records for each search
  - O(n) network calls for n records
  - Suitable only for small datasets (<1000 records)
  
- **Future Optimizations**:
  - Override vector_search to use AWS OpenSearch
  - Implement caching layer for frequently accessed vectors
  - Use S3 Select to filter before downloading

## Code Quality Standards

1. **Type Hints**: All new methods must have complete type hints
2. **Docstrings**: Detailed docstrings with warnings for S3
3. **Tests**: Minimum 80% coverage for new code
4. **Examples**: Working examples for each backend

## Migration Path for Users

For users currently without vector support on Memory/S3:

1. **Immediate**: Update to new version, vector support works automatically
2. **Optimization**: For S3 users with large datasets:
   ```python
   class OptimizedS3Database(AsyncS3Database):
       async def vector_search(self, query_vector, **kwargs):
           # Use AWS OpenSearch or similar
           return await self._opensearch_vector_search(query_vector, **kwargs)
   ```

## Success Criteria

1. ✅ All Memory/S3 backends pass vector integration tests
2. ✅ Factory correctly configures vector support
3. ✅ Documentation clearly explains performance implications
4. ✅ Example code demonstrates proper usage
5. ✅ S3 implementation includes clear performance warnings

## Notes

- The `PythonVectorSearchMixin` makes this implementation straightforward
- Most complexity is already handled by the mixins
- Main effort is testing and documentation
- S3 inefficiency is acceptable for compatibility, with clear upgrade path
