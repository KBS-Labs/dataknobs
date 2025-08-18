# Integration Testing and CI/CD Quality Gates

## Overview

This document explains how integration tests are incorporated into the DataKnobs quality assurance process and CI/CD pipeline, ensuring code quality before merging pull requests.

## Quality Check Flow

### 1. Local Development (Pre-PR)

Before creating a pull request, developers **must** run:

```bash
./bin/run-quality-checks.sh
```

This script:
- ✅ Starts Docker services (PostgreSQL, Elasticsearch, LocalStack)
- ✅ Runs linting and style checks
- ✅ Executes unit tests (no external dependencies)
- ✅ **Executes integration tests** (with real services)
- ✅ Generates coverage reports
- ✅ Creates artifacts in `.quality-artifacts/`

The artifacts **must be committed** with your code changes.

### 2. Pull Request Validation

When you create a PR, the following workflows run:

#### A. Quality Artifact Validation (`quality-validation.yml`)
- **Purpose**: Verify that quality checks were run locally
- **Checks**:
  - ✅ Artifacts exist in `.quality-artifacts/`
  - ✅ Artifacts are recent (< 24 hours old)
  - ✅ Unit tests passed
  - ✅ **Integration tests passed**
  - ✅ Coverage meets minimum threshold (70%)
- **Result**: PR blocked if validation fails

#### B. Integration Tests Workflow (`integration-tests.yml`)
- **Purpose**: Run integration tests in CI environment
- **Triggers**: PR, push to main/develop, schedule
- **Services**: PostgreSQL, Elasticsearch, LocalStack
- **Test Matrix**: Python 3.10, 3.11, 3.12
- **Output**: Test results, coverage reports, PR comments

#### C. Standard CI (`ci.yml`)
- **Purpose**: Basic tests and builds
- **Runs**: Unit tests, linting, package builds
- **Note**: Focuses on fast checks without external services

## Integration Test Coverage

### What Gets Tested

1. **PostgreSQL Backend**
   - CRUD operations with real database
   - Query translation (filters, sorting, pagination)
   - Transaction isolation
   - Concurrent operations
   - JSONB storage and retrieval
   - Special characters and Unicode

2. **Elasticsearch Backend**
   - Document indexing and retrieval
   - Full-text search capabilities
   - Complex queries and aggregations
   - Bulk operations
   - Real-time indexing behavior
   - Nested object handling

3. **S3/LocalStack Backend** (when implemented)
   - Object storage operations
   - Metadata handling
   - Batch uploads/downloads
   - Cost optimization features

### Test Execution

Integration tests are marked with `@pytest.mark.integration` and can be:

```bash
# Run only integration tests
pytest -m integration

# Run all except integration tests
pytest -m "not integration"

# Run specific backend tests
pytest tests/integration/test_postgres_integration.py
pytest tests/integration/test_elasticsearch_integration.py
```

## Quality Gates Summary

| Check | Local Required | PR Blocking | Where Validated |
|-------|---------------|-------------|-----------------|
| Unit Tests | ✅ Yes | ✅ Yes | Artifacts + CI |
| **Integration Tests** | ✅ Yes | ✅ Yes | Artifacts + CI |
| Linting | ✅ Yes | ⚠️ Warning | Artifacts |
| Style | ✅ Yes | ⚠️ Warning | Artifacts |
| Coverage | ✅ Yes | ⚠️ Warning | Artifacts |
| Artifacts Fresh | N/A | ✅ Yes | CI Validation |

## PR Merge Requirements

A PR can only be merged when:

1. ✅ **Local quality checks passed** (including integration tests)
2. ✅ **Artifacts are committed** and validated
3. ✅ **CI workflows pass** (including integration test workflow)
4. ✅ **Code review approved**

## Integration Test Output

### In Artifacts (`.quality-artifacts/`)

```
integration-test-results.xml    # JUnit XML test results
integration-test-output.txt     # Full test output
coverage-integration.xml        # Coverage from integration tests
quality-summary.json           # Overall status including integration tests
```

### In CI/CD

- **GitHub Actions Artifacts**: Test results, coverage reports
- **PR Comments**: Test summary and failures
- **Status Checks**: Pass/fail indicators on PR
- **CodeCov Integration**: Coverage tracking

## Running Integration Tests

### Method 1: Full Quality Checks (Recommended)
```bash
./bin/run-quality-checks.sh
```

### Method 2: Integration Tests Only
```bash
./scripts/run-integration-tests.sh
```

### Method 3: Manual with Docker
```bash
# Start services
docker-compose up -d postgres elasticsearch

# Run tests
pytest packages/data/tests/integration/ -v

# Stop services
docker-compose down
```

## Troubleshooting

### Common Issues

1. **"Artifacts missing or outdated"**
   - Run `./bin/run-quality-checks.sh` locally
   - Commit the `.quality-artifacts/` directory

2. **Integration tests fail locally**
   - Check Docker is running: `docker ps`
   - Verify services are healthy: `docker-compose ps`
   - Check logs: `docker-compose logs postgres elasticsearch`

3. **Integration tests fail in CI**
   - Check workflow logs for service startup issues
   - Verify environment variables are set correctly
   - Look for timeout issues (services may take longer in CI)

### Debug Commands

```bash
# Check what tests will run
pytest --collect-only -m integration

# Run with verbose output
pytest -vvs -m integration

# Run specific test
pytest tests/integration/test_postgres_integration.py::TestPostgresIntegration::test_full_crud_cycle

# Keep services running after tests
KEEP_SERVICES=true ./scripts/run-integration-tests.sh
```

## Best Practices

1. **Always run quality checks before pushing**
   ```bash
   ./bin/run-quality-checks.sh
   git add .quality-artifacts/
   git commit -m "Update quality artifacts"
   ```

2. **Keep artifacts up to date**
   - Re-run checks after significant changes
   - Don't manually edit artifact files

3. **Write integration tests for new backends**
   - Follow existing test patterns
   - Test real service behavior
   - Include edge cases and error conditions

4. **Monitor CI feedback**
   - Check PR comments for test results
   - Review failed test logs
   - Address issues before requesting review

## Configuration

### Environment Variables

Integration tests use these environment variables:

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

# AWS/LocalStack
AWS_ENDPOINT_URL=http://localhost:4566
AWS_ACCESS_KEY_ID=test
AWS_SECRET_ACCESS_KEY=test
```

### Docker Services

Required services are defined in `docker-compose.override.yml`:
- PostgreSQL 15
- Elasticsearch 8.11.0
- LocalStack 3.0 (for S3)

## Summary

Integration tests are a **mandatory** part of the quality checks and are validated at multiple levels:

1. **Locally**: Via `run-quality-checks.sh` before creating PR
2. **Artifacts**: Results must be committed and fresh
3. **CI/CD**: Tests run again in GitHub Actions
4. **PR Gate**: Cannot merge without passing integration tests

This multi-layer approach ensures that:
- Code works with real services
- No regressions are introduced
- Quality is maintained across all backends
- PRs are only merged when fully tested