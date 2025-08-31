# Integration Tests for DataKnobs Data Package

This directory contains integration tests that verify the data package works correctly with real database services (PostgreSQL, Elasticsearch, and S3/LocalStack).

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Running Tests Locally](#running-tests-locally)
- [CI/CD Integration](#cicd-integration)
- [Test Structure](#test-structure)
- [Writing New Tests](#writing-new-tests)
- [Troubleshooting](#troubleshooting)

## Overview

Integration tests ensure that our database backends work correctly with real services:

- **PostgreSQL**: Tests JSONB storage, query translation, transactions
- **Elasticsearch**: Tests document storage, full-text search, aggregations
- **S3/LocalStack**: Tests object storage operations (coming soon)

## Prerequisites

### Docker Services

The integration tests require the following services to be running:

1. **PostgreSQL 15**: Database server
2. **Elasticsearch 8.11**: Search and analytics engine
3. **LocalStack**: AWS S3 emulation (optional, for S3 tests)

### Python Dependencies

```bash
# Install the data package with dev dependencies
uv pip install -e packages/data[dev]

# Install test dependencies
uv pip install pytest pytest-asyncio pytest-cov pytest-html
```

## Running Tests Locally

### Method 1: Using the Test Runner Script (Recommended)

The easiest way to run integration tests is using the provided script:

```bash
# Run all integration tests
./scripts/run-integration-tests.sh

# Run tests in verbose mode
./scripts/run-integration-tests.sh --verbose

# Run specific test file
./scripts/run-integration-tests.sh packages/data/tests/integration/test_postgres_integration.py

# Keep services running after tests (for debugging)
KEEP_SERVICES=true ./scripts/run-integration-tests.sh
```

The script will:
1. Start Docker services (PostgreSQL, Elasticsearch, LocalStack)
2. Wait for services to be healthy
3. Run the integration tests
4. Clean up services afterwards

### Method 2: Using Docker Compose Manually

```bash
# Step 1: Start services
docker-compose -f docker-compose.yml -f docker-compose.override.yml up -d postgres elasticsearch localstack

# Step 2: Wait for services (check health)
docker-compose ps

# Step 3: Set environment variables
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432
export POSTGRES_USER=postgres
export POSTGRES_PASSWORD=postgres
export POSTGRES_DB=dataknobs_test

export ELASTICSEARCH_HOST=localhost
export ELASTICSEARCH_PORT=9200

# Step 4: Run tests
uv run pytest packages/data/tests/integration/ -v -m integration

# Step 5: Clean up
docker-compose -f docker-compose.yml -f docker-compose.override.yml down
```

### Method 3: Using Existing Services

If you already have PostgreSQL and Elasticsearch running:

```bash
# Configure connection via environment variables
export POSTGRES_HOST=your-postgres-host
export POSTGRES_PORT=5432
export POSTGRES_USER=your-user
export POSTGRES_PASSWORD=your-password
export POSTGRES_DB=test_db

export ELASTICSEARCH_HOST=your-es-host
export ELASTICSEARCH_PORT=9200

# Run tests
uv run pytest packages/data/tests/integration/ -v
```

## CI/CD Integration

### GitHub Actions

Integration tests run automatically on:
- Push to `main` or `develop` branches
- Pull requests
- Daily schedule (2 AM UTC)
- Manual workflow dispatch

The workflow:
1. Sets up test services using GitHub Actions service containers
2. Runs tests across multiple Python versions (3.10, 3.11, 3.12)
3. Uploads test results and coverage reports
4. Comments results on pull requests

### Running Specific Test Suites

```bash
# PostgreSQL tests only
uv run pytest packages/data/tests/integration/test_postgres_integration.py -v

# Elasticsearch tests only
uv run pytest packages/data/tests/integration/test_elasticsearch_integration.py -v

# Async tests only
uv run pytest packages/data/tests/integration/ -k "Async" -v

# Specific test class
uv run pytest packages/data/tests/integration/test_postgres_integration.py::TestPostgresIntegration -v

# Specific test method
uv run pytest packages/data/tests/integration/test_postgres_integration.py::TestPostgresIntegration::test_full_crud_cycle -v
```

## Test Structure

### Test Files

- `conftest.py`: Fixtures and test configuration
- `test_postgres_integration.py`: PostgreSQL backend tests
- `test_elasticsearch_integration.py`: Elasticsearch backend tests
- `test_s3_integration.py`: S3 backend tests (coming soon)

### Key Fixtures

```python
# Database connection fixtures
postgres_test_db          # Provides clean PostgreSQL database
elasticsearch_test_index   # Provides clean Elasticsearch index

# Test data fixtures
sample_records            # Sample dataset for testing

# Service readiness fixtures
ensure_postgres_ready     # Waits for PostgreSQL
ensure_elasticsearch_ready # Waits for Elasticsearch
```

### Test Categories

1. **Connection Tests**: Verify service connectivity
2. **CRUD Tests**: Create, Read, Update, Delete operations
3. **Query Tests**: Complex queries, filtering, sorting
4. **Batch Tests**: Bulk operations
5. **Concurrent Tests**: Thread/async safety
6. **Edge Cases**: Special characters, large data, etc.

## Writing New Tests

### Test Template

```python
import pytest
from dataknobs_data import SyncDatabase, Record, Query

@pytest.mark.integration
class TestNewFeature:
    """Test new feature with real database."""
    
    def test_feature_with_postgres(self, postgres_test_db):
        """Test feature with PostgreSQL."""
        db = SyncDatabase.from_backend("postgres", postgres_test_db)
        
        # Your test logic here
        record = Record({"test": "data"})
        id = db.create(record)
        
        # Assertions
        assert db.exists(id)
        
        # Cleanup
        db.delete(id)
        db.close()
    
    def test_feature_with_elasticsearch(self, elasticsearch_test_index):
        """Test feature with Elasticsearch."""
        db = SyncDatabase.from_backend("elasticsearch", elasticsearch_test_index)
        
        # Test logic...
        
        db.close()
```

### Best Practices

1. **Use Fixtures**: Leverage provided fixtures for clean test databases
2. **Clean Up**: Always clean up test data (fixtures handle this automatically)
3. **Wait for Indexing**: Add delays after Elasticsearch writes (`time.sleep(0.5)`)
4. **Test Isolation**: Each test should be independent
5. **Meaningful Names**: Use descriptive test names
6. **Documentation**: Add docstrings explaining what's being tested

## Troubleshooting

### Common Issues

#### 1. Services Not Starting

```bash
# Check service status
docker-compose ps

# View service logs
docker-compose logs postgres
docker-compose logs elasticsearch

# Restart services
docker-compose restart postgres elasticsearch
```

#### 2. Connection Refused

```bash
# Verify services are listening
netstat -an | grep -E "5432|9200"

# Check firewall/security groups
# Ensure localhost connections are allowed
```

#### 3. Elasticsearch Timeout

```bash
# Elasticsearch may take longer to start
# Increase wait time or check logs
curl http://localhost:9200/_cluster/health?pretty
```

#### 4. Test Database Already Exists

```bash
# Drop test database
docker-compose exec postgres psql -U postgres -c "DROP DATABASE IF EXISTS dataknobs_test;"
```

### Debug Mode

```bash
# Run tests with full output
uv run pytest packages/data/tests/integration/ -vvs --tb=long

# Run with debug logging
uv run pytest packages/data/tests/integration/ -v --log-cli-level=DEBUG

# Keep failed test data for inspection
uv run pytest packages/data/tests/integration/ --pdb --pdbcls=IPython.terminal.debugger:Pdb
```

### Performance Tips

1. **Parallel Execution**: Tests can run in parallel with `pytest-xdist`
   ```bash
   uv pip install pytest-xdist
   uv run pytest packages/data/tests/integration/ -n auto
   ```

2. **Selective Testing**: Run only changed tests
   ```bash
   uv run pytest packages/data/tests/integration/ --lf  # Last failed
   uv run pytest packages/data/tests/integration/ --ff  # Failed first
   ```

3. **Coverage Reports**: Generate HTML coverage reports
   ```bash
   uv run pytest packages/data/tests/integration/ --cov=packages/data --cov-report=html
   open htmlcov/index.html
   ```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `POSTGRES_HOST` | PostgreSQL host | `localhost` |
| `POSTGRES_PORT` | PostgreSQL port | `5432` |
| `POSTGRES_USER` | PostgreSQL user | `postgres` |
| `POSTGRES_PASSWORD` | PostgreSQL password | `postgres` |
| `POSTGRES_DB` | Test database name | `dataknobs_test` |
| `ELASTICSEARCH_HOST` | Elasticsearch host | `localhost` |
| `ELASTICSEARCH_PORT` | Elasticsearch port | `9200` |
| `AWS_ENDPOINT_URL` | LocalStack endpoint | `http://localhost:4566` |
| `AWS_ACCESS_KEY_ID` | AWS access key | `test` |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key | `test` |

## Contributing

When adding new integration tests:

1. Follow the existing test structure
2. Use appropriate fixtures
3. Add documentation for new test scenarios
4. Ensure tests pass locally before submitting PR
5. Update this README if adding new test categories

## Resources

- [pytest Documentation](https://docs.pytest.org/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [PostgreSQL Testing Best Practices](https://www.postgresql.org/docs/current/regress.html)
- [Elasticsearch Testing Guide](https://www.elastic.co/guide/en/elasticsearch/reference/current/testing.html)