# DataKnobs Data Package Documentation

## Overview
The DataKnobs Data Package provides a unified data abstraction layer for consistent database operations across multiple storage technologies.

**Version**: 0.1.0  
**Status**: Released to PyPI  
**Python**: 3.10+

## Quick Links

### Core Documentation
- [Architecture](ARCHITECTURE.md) - System design and components
- [Record ID Architecture](RECORD_ID_ARCHITECTURE.md) - Dual ID system design
- [Record Serialization](RECORD_SERIALIZATION.md) - Serialization architecture for complex types
- [API Reference](API_REFERENCE.md) - Complete API documentation
- [Design Plan](DESIGN_PLAN.md) - Original design architecture

### Feature Guides
- [Boolean Logic Operators](BOOLEAN_LOGIC_OPERATORS.md) - Complex query construction
- [Batch Processing](BATCH_PROCESSING_GUIDE.md) - Efficient bulk operations

### Active Development
(Currently no active development checklists)

## Development History

### Initial Package Design (August 17, 2025)
Foundational architecture and feature planning:
- [Feature Summary](history/initial-design/FEATURE_SUMMARY.md) - Original feature overview
- [Implementation Status](history/initial-design/IMPLEMENTATION_STATUS.md) - Initial implementation tracking
- [Progress Checklist](history/initial-design/PROGRESS_CHECKLIST.md) - Development milestones
- [Range Operators Implementation](history/initial-design/RANGE_OPERATORS_IMPLEMENTATION.md) - Range query design
- [Next Steps](history/initial-design/NEXT_STEPS.md) - Future enhancements planned
- [API Improvements](history/api-improvements/API_IMPROVEMENTS.md) - API enhancement proposals

### Vector Store Implementation (August 17-29, 2025)
Comprehensive vector search capability development:

#### Planning Phase (August 17)
- [Phase 6 Plan](history/vector-implementation/PHASE6_PLAN.md) - Core vector implementation
- [Phase 7 Plan](history/vector-implementation/PHASE7_PLAN.md) - Advanced vector features
- [Phase 8 Documentation Plan](history/vector-implementation/PHASE8_DOCUMENTATION_PLAN.md) - Documentation strategy
- [Redesign Plan](history/vector-implementation/REDESIGN_PLAN.md) - Architecture redesign
- [Redesign Checklist](history/vector-implementation/REDESIGN_CHECKLIST.md) - Implementation tasks

#### Design Evolution (August 26)
- [Vector Store Design V1](history/vector-implementation/VECTOR_STORE_DESIGN.md) - Initial design
- [Vector Store Design V2](history/vector-implementation/VECTOR_STORE_DESIGN_V2.md) - Refined architecture

#### Implementation & Tracking (August 28-29)
- [Getting Started Guide](history/vector-implementation/VECTOR_GETTING_STARTED.md) - User guide
- [Implementation Summary](history/vector-implementation/VECTOR_IMPLEMENTATION_SUMMARY.md) - Technical details
- [Phase 7 Progress](history/vector-implementation/VECTOR_PROGRESS_TRACKER_PHASE7.md) - Phase 7 tracking
- [Phase 8 Progress](history/vector-implementation/VECTOR_PROGRESS_TRACKER_PHASE8.md) - Phase 8 tracking
- [Memory/S3 Implementation](history/vector-implementation/VECTOR_MEMORY_S3_IMPLEMENTATION.md) - Backend specifics
- [API Refactoring](history/vector-implementation/VECTOR_API_REFACTORING.md) - API improvements
- [Implementation Plan](history/vector-implementation/VECTOR_IMPLEMENTATION_PLAN.md) - Detailed roadmap
- [Progress Tracker](history/vector-implementation/VECTOR_PROGRESS_TRACKER.md) - Overall progress
- [Remaining Work](history/vector-implementation/VECTOR_REMAINING_WORK.md) - Outstanding tasks

### PostgreSQL Refactoring (August 27, 2025)
Backend optimization and code consolidation:
- [Shared Code Analysis](history/postgres-refactoring/analysis_postgres_shared_code.md) - Code analysis
- [Refactoring Summary](history/postgres-refactoring/POSTGRES_REFACTORING_SUMMARY.md) - Changes made

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
from dataknobs_data import Database, Record, Query, Operator

# Create a database connection
async with Database.create("memory") as db:
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

See the main DataKnobs [CONTRIBUTING.md](../../../CONTRIBUTING.md) for guidelines.

## License

Part of the DataKnobs project. See [LICENSE](../../../LICENSE) for details.
