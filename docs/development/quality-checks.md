# Quality Checks Process for DataKnobs

## Overview

DataKnobs uses a **developer-driven quality assurance process** where developers run comprehensive tests locally before creating pull requests. This approach ensures code quality while keeping CI/CD pipelines fast and cost-effective.

**Key Principle:** All tests, including integration tests with real services (PostgreSQL, Elasticsearch, LocalStack), must pass locally before a PR can be merged.

## Quick Start

Before creating a pull request:

```bash
# Run all quality checks (required for PRs to main)
./bin/run-quality-checks.sh
```

This single command will:
1. ✅ Start all required Docker services
2. ✅ Run unit tests
3. ✅ Run integration tests with real services
4. ✅ Check code style and linting
5. ✅ Generate coverage reports
6. ✅ Create artifacts in `.quality-artifacts/`

**Important:** The artifacts must be committed with your PR!

## Detailed Process

### 1. Development Workflow

During development, you can run tests incrementally:

```bash
# Run only unit tests (fast, no services needed)
uv run pytest packages/*/tests/ -v -m "not integration"

# Run specific package tests
uv run pytest packages/data/tests/ -v

# Run with coverage
uv run pytest packages/*/tests/ -v --cov=packages --cov-report=term
```

### 2. Pre-PR Quality Checks (Required)

Before creating a PR to `main` or `develop`:

```bash
# Ensure Docker is running
docker info

# Run the complete quality check suite
./bin/run-quality-checks.sh
```

The script will:
- Start PostgreSQL, Elasticsearch, and LocalStack containers
- Wait for services to be healthy
- Run all tests (unit and integration)
- Perform linting and style checks
- Generate artifacts in `.quality-artifacts/`

**Expected output:**
```
═══════════════════════════════════════════════════════════════
                       Quality Check Summary                     
═══════════════════════════════════════════════════════════════
  Linting:           ✓ PASSED
  Style Check:       ✓ PASSED
  Unit Tests:        ✓ PASSED
  Integration Tests: ✓ PASSED

✓ All critical checks passed!
  Artifacts saved to: .quality-artifacts/
  You can now create your pull request.
```

### 3. Commit the Artifacts

After running quality checks successfully:

```bash
# Add the quality artifacts to your commit
git add .quality-artifacts/

# Commit with your code changes
git commit -m "feat: implement new feature with passing quality checks"

# Push and create PR
git push origin your-branch
```

### 4. CI Validation

When you create a PR, GitHub Actions will:
1. **Validate artifacts exist** - Checks for required files
2. **Verify freshness** - Ensures artifacts are < 24 hours old
3. **Confirm tests passed** - Validates all tests show "PASS" status
4. **Check coverage** - Ensures minimum coverage threshold (70%)

The CI validation is **fast** (< 30 seconds) since it only validates artifacts, not re-running tests.

## Required Services

The quality checks require these Docker services:

| Service | Purpose | Port | Test Type |
|---------|---------|------|-----------|
| PostgreSQL | Database storage | 5432 | Integration |
| Elasticsearch | Search functionality | 9200 | Integration |
| LocalStack | S3-compatible storage | 4566 | Integration |

Services are automatically started by `run-quality-checks.sh`.

### Manual Service Management

If you need to manage services manually:

```bash
# Start all services
docker-compose up -d postgres elasticsearch localstack

# Check service health
docker-compose ps

# View service logs
docker-compose logs elasticsearch

# Stop services
docker-compose down
```

## Artifact Structure

Quality checks generate these artifacts in `.quality-artifacts/`:

```
.quality-artifacts/
├── quality-summary.json       # Overall pass/fail status
├── environment.json           # Python version, OS, git info
├── unit-test-results.xml      # JUnit format test results
├── integration-test-results.xml
├── coverage.xml              # Coverage report
├── lint-report.json          # Pylint results
├── style-check.json          # Ruff style check results
└── signature.sha256          # Integrity checksum
```

## Test Organization

Tests should be organized with pytest markers:

```python
# Unit test (no external services)
def test_data_model():
    assert DataModel().validate() == True

# Integration test (requires services)
@pytest.mark.integration
def test_elasticsearch_query():
    es = Elasticsearch(['localhost:9200'])
    result = es.search(index='test')
    assert result['hits']['total']['value'] >= 0
```

## Troubleshooting

### Services Won't Start

```bash
# Check if ports are already in use
lsof -i :5432  # PostgreSQL
lsof -i :9200  # Elasticsearch
lsof -i :4566  # LocalStack

# Reset Docker services
docker-compose down -v
docker-compose up -d
```

### Tests Pass Locally but CI Rejects Artifacts

Common causes:
- **Artifacts too old** - Re-run `./bin/run-quality-checks.sh`
- **Forgot to commit artifacts** - Run `git add .quality-artifacts/`
- **Modified artifacts** - Don't edit files in `.quality-artifacts/`

### Integration Tests Fail

```bash
# Check service connectivity
curl http://localhost:9200/_cluster/health  # Elasticsearch
psql postgresql://postgres:postgres@localhost:5432/dataknobs  # PostgreSQL
curl http://localhost:4566/_localstack/health  # LocalStack

# View service logs
docker-compose logs postgres
docker-compose logs elasticsearch
docker-compose logs localstack
```

### Out of Disk Space

```bash
# Clean up old Docker data
docker system prune -a --volumes

# Remove old test data
rm -rf ~/dataknobs_postgres_data
rm -rf ~/dataknobs_elasticsearch_data
rm -rf ~/dataknobs_localstack_data
```

## Configuration

### Environment Variables

The quality check scripts respect these environment variables:

```bash
# Maximum age of artifacts for CI validation (hours)
export MAX_AGE_HOURS=24

# Minimum required code coverage (percentage)
export REQUIRED_COVERAGE=70

# Custom pytest markers
export PYTEST_MARKERS="not slow"
```

### Customizing Checks

To add custom checks, modify `bin/run-quality-checks.sh`:

```bash
# Add your custom check
print_status "Running custom security scan..."
if run_security_scan; then
    print_success "Security scan passed"
else
    print_error "Security scan failed"
    OVERALL_STATUS="FAIL"
fi
```

## Benefits of This Approach

1. **Fast CI/CD** - GitHub Actions runs in < 30 seconds vs 5-10 minutes
2. **Real Integration Testing** - Tests run against actual services, not mocks
3. **Cost Effective** - No cloud service costs for every PR
4. **Developer Ownership** - Developers verify their code works before PR
5. **Audit Trail** - Artifacts provide evidence of test execution

## FAQ

**Q: Why not run tests in CI?**
A: Running PostgreSQL, Elasticsearch, and LocalStack for every PR is expensive and slow. Local testing with artifact validation is faster and more cost-effective.

**Q: What if I don't have Docker?**
A: Docker is required for integration tests. You can still run unit tests without Docker: `uv run pytest -m "not integration"`

**Q: Can I skip integration tests?**
A: For PRs to feature branches, you might skip integration tests. For PRs to `main`, they're required.

**Q: How do I add a new service?**
A: Add it to `docker-compose.override.yml`, update `bin/run-quality-checks.sh` to wait for it, and document it here.

**Q: What if artifacts are accidentally modified?**
A: The signature check will detect this. Re-run `./bin/run-quality-checks.sh` to regenerate valid artifacts.

## Summary

1. **Before PR:** Run `./bin/run-quality-checks.sh`
2. **Commit:** Include `.quality-artifacts/` in your commit
3. **CI Validates:** Artifacts are checked automatically
4. **Merge:** Only if all checks pass

This process ensures high code quality while keeping CI/CD fast and economical!