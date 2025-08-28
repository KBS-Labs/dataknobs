# Vector Store Implementation Plan

## Overview
This document provides a detailed, context-preserving implementation plan for the Vector Store Design V2, with comprehensive checklists for tracking progress through each phase.

## Context & Architecture Summary

### Current State
- **Data Package Location**: `packages/data/`
- **Core Classes**: 
  - `AsyncDatabase` / `SyncDatabase` (base classes)
  - `AsyncPostgresDatabase` / `SyncPostgresDatabase` 
  - `AsyncElasticsearchDatabase` / `SyncElasticsearchDatabase`
  - `Field` / `FieldType` (data types)
  - `Record` (data container)
  - `Query` / `ComplexQuery` (query builders)

### Target Architecture
- Enhanced existing backends with automatic vector detection
- New `VectorField` extending `Field`
- Mixins for vector operations
- Synchronization and migration tools
- Preserved backward compatibility

## Prerequisites & Dependencies

### Required Packages
```bash
# Core dependencies
numpy>=1.24.0
asyncpg>=0.29.0  # For PostgreSQL async support
psycopg2-binary>=2.9.0  # For PostgreSQL sync support
elasticsearch>=8.0.0  # For Elasticsearch support

# Vector-specific dependencies
pgvector>=0.2.0  # PostgreSQL vector extension
faiss-cpu>=1.7.4  # For Faiss backend (optional)
chromadb>=0.4.0  # For Chroma backend (optional)

# Development dependencies
pytest>=7.0.0
pytest-asyncio>=0.21.0
pytest-mock>=3.10.0
```

### Database Prerequisites
- PostgreSQL 12+ with ability to install extensions
- Elasticsearch 8.0+ with dense_vector support
- SQLite 3.35+ (for vector extension support)

## Implementation Phases

### Phase 1: Core Infrastructure (Days 1-3) ✅ COMPLETED

#### Context
Start with minimal changes to existing codebase, adding new types and protocols without modifying existing backends.

#### Files to Create/Modify
1. `src/dataknobs_data/fields.py` - Add VectorField
2. `src/dataknobs_data/vector/__init__.py` - New module
3. `src/dataknobs_data/vector/types.py` - Core types
4. `src/dataknobs_data/vector/mixins.py` - Vector mixins
5. `src/dataknobs_data/vector/operations.py` - Vector operations

#### Implementation Checklist

- [x] **1.1 Add Vector Field Type**
  ```python
  # src/dataknobs_data/fields.py
  - [x] Add VECTOR to FieldType enum
  - [x] Add SPARSE_VECTOR to FieldType enum
  - [x] Create VectorField class
  - [x] Add vector validation logic
  - [x] Add dimension validation
  - [x] Add source_field tracking
  - [x] Add model metadata support
  - [x] Update Field.from_dict() for vectors
  - [x] Update Field.to_dict() for vectors
  ```

- [x] **1.2 Create Vector Module Structure**
  ```
  src/dataknobs_data/vector/
  - [x] __init__.py (exports)
  - [x] types.py (DistanceMetric, VectorSearchResult)
  - [x] mixins.py (VectorCapable, VectorOperationsMixin)
  - [x] operations.py (base vector operations)
  - [x] exceptions.py (vector-specific exceptions)
  ```

- [x] **1.3 Define Core Types**
  ```python
  # src/dataknobs_data/vector/types.py
  - [x] DistanceMetric enum (COSINE, EUCLIDEAN, DOT_PRODUCT, L1, L2)
  - [x] VectorSearchResult dataclass
  - [x] VectorConfig dataclass
  - [x] VectorIndexConfig dataclass
  - [x] VectorMetadata dataclass
  ```

- [x] **1.4 Create Vector Mixins**
  ```python
  # src/dataknobs_data/vector/mixins.py
  - [x] VectorCapable protocol
    - [x] has_vector_support()
    - [x] enable_vector_support()
    - [x] detect_vector_fields()
  - [x] VectorOperationsMixin abstract class
    - [x] vector_search()
    - [x] bulk_embed_and_store()
    - [x] update_vector()
    - [x] delete_from_index()
  ```

- [x] **1.5 Unit Tests for Core Components**
  ```python
  # tests/test_vector_fields.py
  - [x] Test VectorField creation
  - [x] Test dimension validation
  - [x] Test numpy array conversion
  - [x] Test metadata preservation
  - [x] Test serialization/deserialization
  - [x] Test cosine similarity and euclidean distance methods
  - [x] Fixed floating-point precision issues with np.allclose()
  ```

### Phase 2: PostgreSQL Integration (Days 4-6) ✅ COMPLETED

#### Context
Enhanced PostgreSQL backend with automatic vector detection while maintaining backward compatibility.

#### Files to Modify
1. `src/dataknobs_data/backends/postgres.py`
2. `src/dataknobs_data/backends/postgres_vector.py` (new)
3. `tests/test_postgres_vector.py` (new)
4. `tests/integration/test_postgres_vector_integration.py` (new)

#### Implementation Checklist

- [x] **2.1 Enhance PostgreSQL Backend**
  ```python
  # src/dataknobs_data/backends/postgres.py
  - [x] Add VectorOperationsMixin to class inheritance
  - [x] Add vector_enabled flag
  - [x] Add vector_dimensions tracking dict
  - [x] Modify connect() to check for vectors (_detect_vector_support)
  - [x] Add _detect_vector_support() method
  - [x] Add has_vector_support() method (from VectorCapable)
  - [x] Add enable_vector_support() method
  - [x] Override _record_to_row() to handle VectorField serialization
  - [x] Implement vector_search() method
  - [x] Implement bulk_embed_and_store() stub (abstract requirement)
  ```

- [x] **2.2 PostgreSQL Vector Utilities**
  ```python
  # src/dataknobs_data/backends/postgres_vector.py
  - [x] pgvector installation checker (check_pgvector_extension)
  - [x] Auto-install pgvector (install_pgvector_extension)
  - [x] Index type selector (get_optimal_index_type)
  - [x] Index creation SQL builder (build_vector_index_sql)
  - [x] Vector column DDL generator (get_vector_column_ddl)
  - [x] Distance operator mapper (get_vector_operator)
  - [x] Vector formatting for PostgreSQL (format_vector_for_postgres)
  - [x] Vector parsing from PostgreSQL (parse_postgres_vector)
  ```

- [x] **2.3 PostgreSQL Vector Operations**
  ```python
  # Enhanced methods in postgres.py
  - [x] vector_search() with filters
  - [x] bulk_embed_and_store() full implementation
  - [x] create_vector_index()
  - [ ] drop_vector_index()
  - [ ] optimize_vector_index()
  - [ ] get_vector_index_stats()
  ```

- [x] **2.4 Auto-Detection Logic**
  ```python
  - [x] Check pg_extension for pgvector
  - [x] Auto-install pgvector if needed and permissions allow
  - [x] Handle permission errors gracefully
  - [x] Log vector support status
  - [x] Query information_schema for existing vector columns
  - [x] Check config for enable_vectors flag
  ```

- [x] **2.5 PostgreSQL Vector Tests**
  ```python
  # tests/test_postgres_vector.py & test_postgres_vector_integration.py
  - [x] Test pgvector extension detection (PASSING)
  - [x] Test vector field storage and retrieval (PASSING)
  - [x] Test VectorField JSON serialization fix
  - [x] Test vector search (all metrics) - PASSING
  - [x] Test filtered vector search - PASSING
  - [x] Test bulk vector operations - PASSING
  - [x] Test index creation/optimization - PASSING
  - [x] Test backward compatibility (no breaking changes)
  ```

- [x] **2.6 Additional Infrastructure**
  ```bash
  # Docker and scripts
  - [x] Updated docker-compose.override.yml to use pgvector/pgvector:pg15
  - [x] Created bin/ensure-pgvector.sh script for extension verification
  - [x] Script works with docker-compose exec (no psql needed on host)
  - [x] Integration with existing test infrastructure (TEST_POSTGRES=true)
  ```

- [x] **2.7 Code Quality Improvements**
  ```python
  # Refactoring completed 2025-08-28
  - [x] Created shared mixins to eliminate ~150-200 lines of duplicated code
  - [x] PostgresBaseConfig: Centralized configuration parsing
  - [x] PostgresTableManager: Shared table management SQL
  - [x] PostgresVectorSupport: Vector field detection and tracking
  - [x] PostgresConnectionValidator: Connection validation logic
  - [x] PostgresErrorHandler: Consistent error handling
  - [x] SQLRecordSerializer: Centralized vector field JSON serialization
  - [x] Fixed connection pool cleanup and resource leaks
  - [x] Fixed all metadata handling test failures
  - [x] Reduced default pool sizes for better test stability
  ```

### Phase 3: Elasticsearch Integration (Days 7-9) - ✅ COMPLETED

#### Context
Enhance Elasticsearch backend with dense_vector support and KNN search.

#### Lessons from PostgreSQL Implementation to Apply
1. **Use Shared Mixins**: Create elasticsearch_mixins.py for common code between any ES variants
2. **Centralize Serialization**: Extend SQLRecordSerializer or create ESRecordSerializer for vector handling
3. **Vector Field Detection**: Implement early detection of vector fields in records
4. **Connection Management**: Ensure proper cleanup of ES clients and connection pools
5. **Test with Real Backend**: Use actual Elasticsearch with docker-compose, not mocks
6. **Operator/Query Mapping**: Create centralized mapping for distance metrics to ES query types
7. **Bulk Operations**: Implement efficient batch processing from the start
8. **Error Handling**: Create consistent error handling patterns like PostgresErrorHandler

#### Files to Modify
1. `src/dataknobs_data/backends/elasticsearch_async.py`
2. `src/dataknobs_data/vector/elasticsearch_utils.py` (new)
3. `tests/test_backends/test_elasticsearch_vector.py` (new)

#### Implementation Checklist

- [x] **3.1 Enhance Elasticsearch Backend**
  ```python
  # src/dataknobs_data/backends/elasticsearch_async.py
  - [x] Add VectorOperationsMixin
  - [x] Add vector field detection (like _has_vector_fields in postgres)
  - [x] Modify mapping creation for dense_vector
  - [x] Override _record_to_doc() for vector serialization (like _record_to_row)
  - [x] Override _doc_to_record() for vector deserialization (like _row_to_record)
  - [x] Implement vector_search() with KNN
  - [x] Add hybrid search support
  - [x] Handle index refresh for vectors
  - [x] Ensure proper client cleanup in close()
  ```

- [x] **3.2 Elasticsearch Vector Utilities**
  ```python
  # src/dataknobs_data/vector/elasticsearch_utils.py
  - [x] Mapping generator for dense_vector
  - [x] KNN query builder with distance metric mapping
  - [x] Hybrid query combiner
  - [x] Index settings optimizer
  - [x] Similarity metric converter (cosine -> cosine, euclidean -> l2_norm, etc.)
  - [x] Bulk indexing for vectors
  - [x] Vector formatting functions (like format_vector_for_postgres)
  - [x] Vector parsing utilities (like parse_postgres_vector)
  ```

- [x] **3.3 Elasticsearch Vector Operations**
  ```python
  - [x] KNN search implementation
  - [x] Approximate vs exact search
  - [x] Script score queries
  - [x] Hybrid text + vector search
  - [x] Vector aggregations
  - [x] Index optimization for vectors
  ```

- [x] **3.4 Elasticsearch Vector Tests**
  ```python
  # tests/test_backends/test_elasticsearch_vector.py
  - [x] Test dense_vector mapping
  - [x] Test KNN search
  - [x] Test filtered KNN
  - [x] Test hybrid search
  - [x] Test similarity metrics
  - [x] Test bulk vector indexing
  - [x] Test index refresh behavior
  - [x] Test with real ES backend (docker-compose)
  - [x] Test vector field metadata persistence
  - [x] Test backward compatibility
  ```

- [x] **3.5 Shared Elasticsearch Infrastructure (NEW - from PostgreSQL learnings)**
  ```python
  # src/dataknobs_data/backends/elasticsearch_mixins.py (new)
  - [x] ElasticsearchBaseConfig: Centralized config parsing
  - [x] ElasticsearchIndexManager: Shared index management
  - [x] ElasticsearchVectorSupport: Vector field detection and tracking
  - [x] ElasticsearchErrorHandler: Consistent error handling
  - [x] ElasticsearchRecordSerializer: Vector field JSON handling
  ```

### Phase 4: Synchronization & Migration (Days 10-12)

#### Context
Build tools for keeping vectors synchronized with text and migrating existing data.

#### Files to Create
1. `src/dataknobs_data/vector/sync.py`
2. `src/dataknobs_data/vector/migration.py`
3. `src/dataknobs_data/vector/tracker.py`
4. `tests/test_vector_sync.py`
5. `tests/test_vector_migration.py`

#### Implementation Checklist

- [ ] **4.1 Vector-Text Synchronizer**
  ```python
  # src/dataknobs_data/vector/sync.py
  - [ ] VectorTextSynchronizer class
    - [ ] __init__ with database and embedding_fn
    - [ ] sync_record() method
    - [ ] bulk_sync() method
    - [ ] _has_current_vector() checker
    - [ ] _needs_update() logic
  - [ ] Configuration management
    - [ ] auto_embed_on_create flag
    - [ ] auto_update_on_text_change flag
    - [ ] batch_size setting
    - [ ] track_model_version flag
  ```

- [ ] **4.2 Change Tracking**
  ```python
  # src/dataknobs_data/vector/tracker.py
  - [ ] ChangeTracker class
    - [ ] Track field dependencies
    - [ ] on_update() hook
    - [ ] on_delete() hook
    - [ ] Queue management
  - [ ] Field dependency mapping
  - [ ] Update detection logic
  - [ ] Batch update processor
  ```

- [ ] **4.3 Migration Tools**
  ```python
  # src/dataknobs_data/vector/migration.py
  - [ ] VectorMigration class
    - [ ] add_vectors_to_existing()
    - [ ] migrate_between_backends()
    - [ ] verify_migration()
  - [ ] IncrementalVectorizer class
    - [ ] Queue management
    - [ ] Background processing
    - [ ] Error handling
  - [ ] Progress tracking
  - [ ] Error recovery
  - [ ] Rollback support
  ```

- [ ] **4.4 Synchronization Tests**
  ```python
  # tests/test_vector_sync.py
  - [ ] Test single record sync
  - [ ] Test bulk sync
  - [ ] Test update detection
  - [ ] Test model version tracking
  - [ ] Test change tracking
  - [ ] Test auto-update on text change
  - [ ] Test error handling
  ```

- [ ] **4.5 Migration Tests**
  ```python
  # tests/test_vector_migration.py
  - [ ] Test adding vectors to existing data
  - [ ] Test incremental vectorization
  - [ ] Test cross-backend migration
  - [ ] Test progress tracking
  - [ ] Test error recovery
  - [ ] Test rollback
  ```

### Phase 5: Query Enhancement (Days 13-14)

#### Context
Enhance the Query class to support vector operations seamlessly.

#### Files to Modify
1. `src/dataknobs_data/query.py`
2. `src/dataknobs_data/query_logic.py`
3. `tests/test_vector_queries.py` (new)

#### Implementation Checklist

- [ ] **5.1 Query Class Enhancement**
  ```python
  # src/dataknobs_data/query.py
  - [ ] Add VectorQuery dataclass
  - [ ] Add vector_query field to Query
  - [ ] Implement similar_to() method
  - [ ] Implement near_text() method
  - [ ] Implement hybrid() method
  - [ ] Add score_threshold support
  - [ ] Add reranking support
  ```

- [ ] **5.2 Query Builder Updates**
  ```python
  - [ ] Vector query serialization
  - [ ] Combined filter + vector queries
  - [ ] Hybrid query generation
  - [ ] Score combination logic
  - [ ] Result reranking
  ```

- [ ] **5.3 Query Tests**
  ```python
  # tests/test_vector_queries.py
  - [ ] Test vector query creation
  - [ ] Test combined filters + vectors
  - [ ] Test hybrid queries
  - [ ] Test score thresholds
  - [ ] Test result ordering
  - [ ] Test query serialization
  ```

### Phase 6: Specialized Vector Stores (Days 15-17)

#### Context
Add support for specialized vector databases like Faiss and Chroma.

#### Files to Create
1. `src/dataknobs_data/vector/backends/faiss.py`
2. `src/dataknobs_data/vector/backends/chroma.py`
3. `src/dataknobs_data/vector/factory.py`
4. `tests/test_vector_backends.py`

#### Implementation Checklist

- [ ] **6.1 Faiss Backend**
  ```python
  # src/dataknobs_data/vector/backends/faiss.py
  - [ ] FaissVectorStore class
  - [ ] Index type selection
  - [ ] Index creation/loading
  - [ ] Vector CRUD operations
  - [ ] Search implementation
  - [ ] Persistence support
  - [ ] Memory management
  ```

- [ ] **6.2 Chroma Backend**
  ```python
  # src/dataknobs_data/vector/backends/chroma.py
  - [ ] ChromaVectorStore class
  - [ ] Collection management
  - [ ] Embedding functions
  - [ ] Metadata handling
  - [ ] Query implementation
  - [ ] Persistence options
  ```

- [ ] **6.3 Vector Store Factory**
  ```python
  # src/dataknobs_data/vector/factory.py
  - [ ] VectorStoreFactory class
  - [ ] Backend registration
  - [ ] Configuration validation
  - [ ] Auto-detection logic
  - [ ] Fallback strategies
  ```

- [ ] **6.4 Specialized Store Tests**
  ```python
  # tests/test_vector_backends.py
  - [ ] Test Faiss operations
  - [ ] Test Chroma operations
  - [ ] Test factory creation
  - [ ] Test configuration
  - [ ] Test persistence
  - [ ] Test memory usage
  ```

### Phase 7: Optimization & Performance (Days 18-19)

#### Context
Implement backend-specific optimizations and performance enhancements.

#### Files to Create
1. `src/dataknobs_data/vector/optimizations.py`
2. `src/dataknobs_data/vector/benchmarks.py`
3. `tests/test_vector_performance.py`

#### Implementation Checklist

- [ ] **7.1 PostgreSQL Optimizations**
  ```python
  - [ ] Automatic index type selection
  - [ ] Index parameter tuning
  - [ ] Batch size optimization
  - [ ] Connection pool tuning
  - [ ] Query plan analysis
  - [ ] Halfvec support
  ```

- [ ] **7.2 Elasticsearch Optimizations**
  ```python
  - [ ] Shard configuration
  - [ ] Refresh interval tuning
  - [ ] Bulk indexing optimization
  - [ ] Query cache usage
  - [ ] Circuit breaker config
  ```

- [ ] **7.3 Performance Benchmarks**
  ```python
  # src/dataknobs_data/vector/benchmarks.py
  - [ ] Indexing speed tests
  - [ ] Search latency tests
  - [ ] Memory usage profiling
  - [ ] Scalability tests
  - [ ] Comparison framework
  ```

- [ ] **7.4 Performance Tests**
  ```python
  # tests/test_vector_performance.py
  - [ ] Benchmark different index types
  - [ ] Test batch size impacts
  - [ ] Measure memory usage
  - [ ] Test concurrent operations
  - [ ] Profile hot paths
  ```

### Phase 8: Integration & Documentation (Days 20-21)

#### Context
Final integration, comprehensive testing, and documentation.

#### Files to Create/Update
1. `src/dataknobs_data/factory.py` (update)
2. `docs/vector_store_guide.md` (new)
3. `examples/vector_examples.py` (new)
4. `tests/integration/test_vector_integration.py` (new)

#### Implementation Checklist

- [ ] **8.1 Factory Integration**
  ```python
  # src/dataknobs_data/factory.py
  - [ ] Update DatabaseFactory
  - [ ] Add vector backend detection
  - [ ] Configuration validation
  - [ ] Backward compatibility checks
  ```

- [ ] **8.2 Integration Tests**
  ```python
  # tests/integration/test_vector_integration.py
  - [ ] End-to-end vector workflows
  - [ ] Cross-backend compatibility
  - [ ] Migration scenarios
  - [ ] Synchronization workflows
  - [ ] Error recovery tests
  - [ ] Performance regression tests
  ```

- [ ] **8.3 Documentation**
  ```markdown
  # docs/vector_store_guide.md
  - [ ] Getting started guide
  - [ ] Configuration reference
  - [ ] API documentation
  - [ ] Migration guide
  - [ ] Best practices
  - [ ] Troubleshooting
  - [ ] Performance tuning
  ```

- [ ] **8.4 Examples**
  ```python
  # examples/vector_examples.py
  - [ ] Basic vector storage
  - [ ] Vector search
  - [ ] Hybrid search
  - [ ] Migration example
  - [ ] Synchronization example
  - [ ] Multi-backend example
  ```

## Testing Strategy

### Unit Test Coverage Goals
- Core components: 95%+
- Backend integrations: 90%+
- Migration tools: 85%+
- Overall package: 90%+

### Integration Test Matrix

| Backend | Vector Search | Sync | Migration | Performance |
|---------|--------------|------|-----------|-------------|
| PostgreSQL | ✓ | ✓ | ✓ | ✓ |
| Elasticsearch | ✓ | ✓ | ✓ | ✓ |
| SQLite | ✓ | ✓ | ✓ | ✓ |
| Faiss | ✓ | ✓ | ✓ | ✓ |
| Chroma | ✓ | ✓ | ✓ | ✓ |

### Performance Benchmarks

| Operation | Target Latency | Target Throughput |
|-----------|---------------|-------------------|
| Single vector index | < 10ms | > 100/sec |
| Batch vector index (100) | < 100ms | > 10K/sec |
| Vector search (k=10) | < 50ms | > 20/sec |
| Hybrid search | < 100ms | > 10/sec |
| Sync check | < 5ms | > 200/sec |

## Validation Milestones

### Milestone 1: Core Infrastructure (Day 3)
- [ ] All vector field types working
- [ ] Mixins properly defined
- [ ] Base tests passing
- [ ] No breaking changes to existing code

### Milestone 2: PostgreSQL Integration (Day 6)
- [ ] Auto-detection working
- [ ] pgvector operations functional
- [ ] Performance acceptable
- [ ] Tests passing

### Milestone 3: Elasticsearch Integration (Day 9)
- [ ] KNN search working
- [ ] Hybrid search functional
- [ ] Performance optimized
- [ ] Tests passing

### Milestone 4: Sync & Migration (Day 12)
- [ ] Synchronization working
- [ ] Migration tools functional
- [ ] Change tracking operational
- [ ] Tests passing

### Milestone 5: Full Integration (Day 17)
- [ ] All backends integrated
- [ ] Query system enhanced
- [ ] Specialized stores working
- [ ] Integration tests passing

### Milestone 6: Production Ready (Day 21)
- [ ] All tests passing
- [ ] Documentation complete
- [ ] Performance validated
- [ ] Examples working
- [ ] Backward compatibility confirmed

## Risk Management

### Technical Risks
1. **pgvector installation failures**
   - Mitigation: Graceful fallback, clear error messages
   - Alternative: In-memory vector operations

2. **Performance degradation**
   - Mitigation: Comprehensive benchmarking
   - Alternative: Async processing, caching

3. **Breaking changes**
   - Mitigation: Extensive backward compatibility tests
   - Alternative: Version flags, migration tools

### Dependencies Risks
1. **External package conflicts**
   - Mitigation: Optional dependencies, version ranges
   - Alternative: Vendored implementations

2. **Database version requirements**
   - Mitigation: Feature detection, version checks
   - Alternative: Polyfills, compatibility layers

## Success Metrics

### Functional Metrics
- 100% backward compatibility
- All test suites passing
- Zero critical bugs
- Complete documentation

### Performance Metrics
- < 10% overhead for non-vector operations
- Meeting all latency targets
- Efficient memory usage
- Scalable to 10M+ vectors

### Adoption Metrics
- Clear migration path
- Working examples
- Positive user feedback
- Community contributions

## Next Steps

1. **Review and approve design**
2. **Set up development environment**
3. **Create feature branch**
4. **Begin Phase 1 implementation**
5. **Set up CI/CD for vector tests**
6. **Schedule review checkpoints**

## Appendix: Command Reference

### Development Commands
```bash
# Install development dependencies
pip install -e ".[dev,postgres,elasticsearch,vector]"

# Run vector tests
pytest tests/test_vector_* -v

# Run integration tests
pytest tests/integration/test_vector_integration.py -v

# Run benchmarks
python -m dataknobs_data.vector.benchmarks

# Check coverage
pytest --cov=dataknobs_data.vector --cov-report=html
```

### Database Setup
```bash
# PostgreSQL with pgvector
docker run -d --name pgvector \
  -e POSTGRES_PASSWORD=password \
  -p 5432:5432 \
  ankane/pgvector

# Elasticsearch
docker run -d --name elasticsearch \
  -p 9200:9200 -p 9300:9300 \
  -e "discovery.type=single-node" \
  elasticsearch:8.11.0
```

### Migration Commands
```python
# Add vectors to existing data
python -m dataknobs_data.vector.migrate \
  --source postgres://localhost/mydb \
  --text-fields title,content \
  --vector-field embedding \
  --batch-size 100

# Sync vectors
python -m dataknobs_data.vector.sync \
  --database postgres://localhost/mydb \
  --check-updates \
  --auto-fix
```

---

This implementation plan provides a complete roadmap with preserved context at each phase, ensuring smooth development and integration of vector store capabilities.