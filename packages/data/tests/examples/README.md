# Vector Store Example Tests

This directory contains comprehensive tests for the DataKnobs vector store example scripts.

## Test Files

- `test_basic_vector_search.py` - Tests for basic vector search functionality
- `test_text_to_vector_sync.py` - Tests for text-to-vector synchronization
- `test_migrate_existing_data.py` - Tests for migration workflows
- `test_hybrid_search.py` - Tests for hybrid search strategies
- `test_examples_integration.py` - Integration tests using real implementations

## Running Tests

### Run All Tests
```bash
# From the package root
uv run pytest tests/examples/

# Or use the test runner
uv run python tests/examples/run_example_tests.py
```

### Run Specific Test File
```bash
uv run pytest tests/examples/test_basic_vector_search.py -v
```

### Run with Coverage
```bash
uv run pytest tests/examples/ --cov=examples --cov-report=html
```

### Run Test Runner
```bash
# Run all tests
uv run python tests/examples/run_example_tests.py

# Run with verbose output
uv run python tests/examples/run_example_tests.py --verbose

# Filter tests
uv run python tests/examples/run_example_tests.py --filter hybrid

# Validate examples only
uv run python tests/examples/run_example_tests.py --validate-only
```

## Test Strategy

The tests use a combination of real implementations and lightweight test utilities:

1. **Real Database Backends**: Tests use actual SQLite with vector support
2. **Test Embeddings**: Simple deterministic embeddings that simulate semantic similarity
3. **Real Vector Operations**: Actual vector search, filtering, and scoring
4. **Integration Testing**: Complete workflows from the examples

## Key Features Tested

### Basic Vector Search
- Database setup with vector support
- Document creation with embeddings
- Vector similarity search
- Filtered vector search
- Query builder methods (near_text, similar_to)

### Text-to-Vector Synchronization
- Bulk synchronization
- Change tracking
- Single record sync
- Auto-sync functionality
- Update detection

### Migration
- Legacy to vector database migration
- Incremental vectorization
- Batch processing
- Error handling and retries
- Migration verification

### Hybrid Search
- Text search
- Vector search
- Combined scoring (weighted, RRF)
- Complex queries
- Performance optimization

## Test Utilities

### TestEmbedding Class
A lightweight embedding generator that:
- Creates deterministic embeddings
- Simulates semantic similarity
- Supports variable dimensions
- No external dependencies

### Real Components Used
- `DatabaseFactory` - Real database creation
- `VectorField` - Actual vector field implementation
- `Query` and `ComplexQuery` - Real query builders
- `VectorTextSynchronizer` - Real synchronization
- `ChangeTracker` - Real change tracking
- `VectorMigration` - Real migration tools

## Performance Considerations

Tests are designed to be fast while still being comprehensive:
- Use in-memory SQLite databases
- Small test datasets (5-50 records)
- Lightweight embedding generation
- Parallel test execution supported

## Continuous Integration

These tests can be integrated into CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run Vector Example Tests
  run: |
    uv run python tests/examples/run_example_tests.py --verbose
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure the examples directory is in the Python path
2. **Database Errors**: SQLite vector support is built-in, no extensions needed
3. **Test Failures**: Check that all vector store components are properly installed

### Debug Mode

Run tests with pytest debug flags:
```bash
uv run pytest tests/examples/ -vv --tb=long --pdb
```

## Contributing

When adding new examples:
1. Create the example script in `examples/`
2. Refactor for testability (separate logic into classes/functions)
3. Create corresponding test in `tests/examples/test_<example_name>.py`
4. Use real implementations where possible
5. Add to test runner in `run_example_tests.py`
6. Update this README