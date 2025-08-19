# DataKnobs Data Package Documentation

## Overview
The DataKnobs Data Package provides a unified data abstraction layer for consistent database operations across multiple storage technologies.

**Version**: 0.1.0  
**Status**: Released to PyPI  
**Python**: 3.10+

## Quick Links

### Core Documentation
- [Architecture](ARCHITECTURE.md) - System design and components
- [API Reference](API_REFERENCE.md) - Complete API documentation
- [Progress Checklist](PROGRESS_CHECKLIST.md) - Development status

### Feature Guides
- [Boolean Logic Operators](BOOLEAN_LOGIC_OPERATORS.md) - Complex query construction
- [Range Operators](RANGE_OPERATORS_IMPLEMENTATION.md) - Range and set operations
- [Batch Processing](BATCH_PROCESSING_GUIDE.md) - Efficient bulk operations
- [Feature Summary](FEATURE_SUMMARY.md) - Complete feature overview

### Evolutionary Planning Documents
- [Design Plan](DESIGN_PLAN.md) - Original design document
- [Implementation Status](IMPLEMENTATION_STATUS.md) - Intermediary implementation state
- [Next Steps](NEXT_STEPS.md) -  Follow-up enhancements

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
