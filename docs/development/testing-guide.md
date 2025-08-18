# Testing Guide

This guide explains how to run tests for the DataKnobs project.

## Quick Start

```bash
# Run all tests (unit and integration)
bin/dev.sh test

# Run only unit tests
bin/dev.sh test -t unit

# Run only integration tests
bin/dev.sh test -t integration

# Run tests for a specific package
bin/dev.sh test data
bin/dev.sh test -t unit config
```

## Test Types

### Unit Tests
Unit tests are fast, isolated tests that don't require external services. They test individual components in isolation.

```bash
# Run unit tests for all packages
bin/test.sh -t unit

# Run unit tests for specific package
bin/test.sh -t unit -p data
```

### Integration Tests
Integration tests verify that components work correctly with external services like databases and cloud storage.

```bash
# Run integration tests (starts services automatically)
bin/test.sh -t integration

# Run integration tests without starting services
bin/test.sh -t integration -n

# Only start services without running tests
bin/run-integration-tests.sh -s
```

## Available Commands

### bin/dev.sh test
The main test command with automatic detection of test types.

```bash
bin/dev.sh test [OPTIONS]
```

Options are passed through to the improved test runner.

### bin/test.sh
Advanced test runner with fine-grained control.

```bash
bin/test.sh [OPTIONS] [PACKAGE]
```

Options:
- `-t, --type TYPE` - Test type: unit, integration, or both (default: both)
- `-p, --package PACKAGE` - Package to test (e.g., data, config, structures)
- `-s, --services` - Start services for integration tests (auto by default)
- `-n, --no-services` - Don't start services (assume they're already running)
- `-v, --verbose` - Run tests in verbose mode
- `-k EXPRESSION` - Only run tests matching the expression (pytest -k)
- `-x, --exitfirst` - Exit on first failure
- `-h, --help` - Show help message

### bin/run-integration-tests.sh
Dedicated integration test runner with Docker service management.

```bash
bin/run-integration-tests.sh [OPTIONS] [PACKAGE]
```

Options:
- `-p, --package PACKAGE` - Package to test
- `-v, --verbose` - Run tests in verbose mode
- `-s, --services-only` - Only start services, don't run tests
- `-n, --no-services` - Don't start services
- `-k, --keep-services` - Keep services running after tests
- `-h, --help` - Show help message

## Examples

### Development Workflow

```bash
# Start your development session by running unit tests
bin/dev.sh test -t unit

# When ready to test with real services
bin/dev.sh test -t integration data

# Keep services running for debugging
bin/run-integration-tests.sh -k data

# Run specific test
bin/dev.sh test -k test_s3_backend

# Run tests verbosely for debugging
bin/dev.sh test -v -k test_query
```

### Continuous Integration

```bash
# Run all unit tests first (fast)
bin/test.sh -t unit

# Then run integration tests if unit tests pass
bin/test.sh -t integration
```

### Testing Specific Packages

```bash
# Test only the data package
bin/dev.sh test data

# Test only config package unit tests
bin/dev.sh test -t unit config

# Test structures package with verbose output
bin/dev.sh test -v structures
```

## Services for Integration Tests

Integration tests require the following services:
- **PostgreSQL** - SQL database backend
- **Elasticsearch** - Search engine backend
- **LocalStack** - AWS S3 simulation

These services are automatically started when running integration tests. They can be managed manually:

```bash
# Start services only
bin/run-integration-tests.sh -s

# Stop services
docker-compose down -v

# Check service status
docker-compose ps
```

## Environment Variables

The following environment variables are automatically set for integration tests:

```bash
# PostgreSQL
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=dataknobs_test

# Elasticsearch
ELASTICSEARCH_HOST=localhost
ELASTICSEARCH_PORT=9200

# LocalStack (S3)
AWS_ENDPOINT_URL=http://localhost:4566
AWS_ACCESS_KEY_ID=test
AWS_SECRET_ACCESS_KEY=test
AWS_DEFAULT_REGION=us-east-1
LOCALSTACK_ENDPOINT=http://localhost:4566
```

## Writing Tests

### Unit Tests
Place unit tests in `packages/<package>/tests/`:

```python
# packages/data/tests/test_query.py
def test_query_creation():
    query = Query()
    assert query.filters == []
```

### Integration Tests
Place integration tests in `packages/<package>/tests/integration/`:

```python
# packages/data/tests/integration/test_s3_backend.py
@pytest.mark.integration
def test_s3_operations(s3_backend):
    record = Record({"data": "test"})
    record_id = s3_backend.create(record)
    assert s3_backend.exists(record_id)
```

## Troubleshooting

### Services Won't Start
```bash
# Check if services are already running
docker-compose ps

# Stop and clean up existing services
docker-compose down -v

# Try starting services again
bin/run-integration-tests.sh -s
```

### Tests Can't Connect to Services
```bash
# Verify services are healthy
docker-compose ps

# Check service logs
docker-compose logs postgres
docker-compose logs elasticsearch
docker-compose logs localstack
```

### Permission Errors
```bash
# Make scripts executable
chmod +x bin/test.sh
chmod +x bin/run-integration-tests.sh
```

## Best Practices

1. **Run unit tests frequently** - They're fast and catch most issues
2. **Run integration tests before committing** - Ensure everything works with real services
3. **Use verbose mode for debugging** - Add `-v` to see detailed test output
4. **Keep services running during development** - Use `-k` flag to avoid restart delays
5. **Clean up services when done** - Run `docker-compose down -v` to free resources