# DataKnobs Data Package Documentation

## Overview
The DataKnobs Data Package provides a unified data abstraction layer for consistent database operations across multiple storage technologies.

**Version**: 0.1.0  
**Status**: Released to PyPI  
**Python**: 3.12+

## Quick Links

### Core Documentation
- [Architecture](architecture.md) - System design and components
- [Record ID Architecture](record-id-architecture.md) - Dual ID system design
- [Record Serialization](record-serialization.md) - Serialization architecture for complex types
- [API Reference](api-reference.md) - Complete API documentation
- [Design Plan](design-plan.md) - Original design architecture

### Vector Store Backends
- [pgvector Backend](pgvector-backend.md) - PostgreSQL pgvector vector store

### Feature Guides
- [Boolean Logic Operators](boolean-logic-operators.md) - Complex query construction
- [Batch Processing](batch-processing-guide.md) - Efficient bulk operations

### Active Development
(Currently no active development checklists)

## Development History

### Initial Package Design (August 17, 2025)
Foundational architecture and feature planning:
- [Feature Summary](history/initial-design/feature-summary.md) - Original feature overview
- [Implementation Status](history/initial-design/implementation-status.md) - Initial implementation tracking
- [Progress Checklist](history/initial-design/progress-checklist.md) - Development milestones
- [Range Operators Implementation](history/initial-design/range-operators-implementation.md) - Range query design
- [Next Steps](history/initial-design/next-steps.md) - Future enhancements planned
- [API Improvements](history/api-improvements/api-improvements.md) - API enhancement proposals

### Vector Store Implementation (August 17-29, 2025)
Comprehensive vector search capability development:

#### Planning Phase (August 17)
- [Phase 6 Plan](history/vector-implementation/phase6-plan.md) - Core vector implementation
- [Phase 7 Plan](history/vector-implementation/phase7-plan.md) - Advanced vector features
- [Phase 8 Documentation Plan](history/vector-implementation/phase8-documentation-plan.md) - Documentation strategy
- [Redesign Plan](history/vector-implementation/redesign-plan.md) - Architecture redesign
- [Redesign Checklist](history/vector-implementation/redesign-checklist.md) - Implementation tasks

#### Design Evolution (August 26)
- [Vector Store Design V1](history/vector-implementation/vector-store-design.md) - Initial design
- [Vector Store Design V2](history/vector-implementation/vector-store-design-v2.md) - Refined architecture

#### Implementation & Tracking (August 28-29)
- [Getting Started Guide](history/vector-implementation/vector-getting-started.md) - User guide
- [Implementation Summary](history/vector-implementation/vector-implementation-summary.md) - Technical details
- [Phase 7 Progress](history/vector-implementation/vector-progress-tracker-phase7.md) - Phase 7 tracking
- [Phase 8 Progress](history/vector-implementation/vector-progress-tracker-phase8.md) - Phase 8 tracking
- [Memory/S3 Implementation](history/vector-implementation/vector-memory-s3-implementation.md) - Backend specifics
- [API Refactoring](history/vector-implementation/vector-api-refactoring.md) - API improvements
- [Implementation Plan](history/vector-implementation/vector-implementation-plan.md) - Detailed roadmap
- [Progress Tracker](history/vector-implementation/vector-progress-tracker.md) - Overall progress
- [Remaining Work](history/vector-implementation/vector-remaining-work.md) - Outstanding tasks

### PostgreSQL Refactoring (August 27, 2025)
Backend optimization and code consolidation:
- [Shared Code Analysis](history/postgres-refactoring/analysis-postgres-shared-code.md) - Code analysis
- [Refactoring Summary](history/postgres-refactoring/postgres-refactoring-summary.md) - Changes made

### Linting & Type Checking Improvements (August 30-31, 2025)
Comprehensive code quality improvements:
- [Linting Errors Checklist](history/linting-and-type-checking/linting-errors-checklist.md) - Complete tracking of linting and type checking fixes
  - Reduced Ruff errors from ~1500 to 0 (with configuration)
  - Reduced MyPy errors from 774 to 41 (focused mode)
  - Fixed Python 3.9 compatibility issues
  - Added VectorStoreFactory for proper separation of concerns

## Installation

```bash
pip install dataknobs-data

# With specific backends
pip install "dataknobs-data[postgres]"
pip install "dataknobs-data[elasticsearch]"
pip install "dataknobs-data[s3]"
pip install "dataknobs-data[all]"
```

## Quick Start

```python
from dataknobs_data import AsyncDatabaseFactory, Record, Query, Operator

# Create a database connection
factory = AsyncDatabaseFactory()
async with factory.create(backend="memory") as db:
    # Create a record
    record = Record({
        "name": "Alice",
        "age": 30,
        "email": "alice@example.com"
    })
    id = await db.create(record)
    
    # Query records
    query = Query().filter("age", Operator.GTE, 25)
    results = await db.search(query)
    
    # Use ergonomic field access
    for r in results:
        print(f"{r.name}: {r['email']}")
```

## Supported Backends

| Backend | Status | Use Case | Performance |
|---------|---------|----------|-------------|
| Memory | ✅ Stable | Testing, caching | Very High |
| File | ✅ Stable | Local persistence | Medium |
| SQLite | ✅ Stable | Embedded SQL, transactions | High |
| DuckDB | ✅ Stable | Analytics, OLAP | High |
| PostgreSQL | ✅ Stable | Relational data | High |
| Elasticsearch | ✅ Stable | Search, analytics | High |
| S3 | ✅ Stable | Cloud storage | Medium |

## Key Features

### ✅ Completed
- Unified API across all backends
- Async/await support with sync fallbacks
- Complex boolean queries (AND, OR, NOT)
- Range operators (BETWEEN, IN, NOT_IN)
- Ergonomic field access (dict-like and attribute)
- Batch operations for efficiency
- Streaming API for large datasets
- Schema validation
- Data migration utilities
- Pandas integration
- Comprehensive test coverage

### 🚧 In Progress
- Additional backend implementations
- Performance optimizations
- Advanced caching layer

### 📋 Planned
- Vector search support
- GraphQL query translation
- Time-series specialization
- Multi-backend replication

## Documentation Structure

### Development History
These documents track the evolution of the package:
- Phase 6-8 Plans: Historical development phases
- Redesign documents: Architecture improvements
- Vector Store Design: Future enhancement planning

### Current Documentation
The package documentation is integrated into the main DataKnobs documentation at `/docs/packages/data/`.

## Contributing

See the main DataKnobs [contributing guide](../../../docs/development/contributing.md) for guidelines.

## License

Part of the DataKnobs project. See [LICENSE](../../../LICENSE) for details.
