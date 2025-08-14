#!/bin/bash
set -e

# Quality Checks Script for DataKnobs
# This script runs all quality checks including unit tests, integration tests,
# linting, and code coverage. Results are saved as artifacts for CI validation.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ARTIFACTS_DIR="$PROJECT_ROOT/.quality-artifacts"
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# Check if we're inside a Docker container
if [ -f /.dockerenv ] || [ -n "$DOCKER_CONTAINER" ]; then
    echo "Detected Docker container environment."
    echo "Switching to internal version of quality checks..."
    exec "$SCRIPT_DIR/run-quality-checks-internal.sh" "$@"
    exit $?
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}         DataKnobs Quality Checks - Starting                      ${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""

# Function to print status
print_status() {
    echo -e "${BLUE}▶${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Create artifacts directory
mkdir -p "$ARTIFACTS_DIR"

# Ensure all packages are installed
print_status "Ensuring all packages are installed..."
if uv sync --all-packages >/dev/null 2>&1; then
    print_success "Packages synced"
else
    print_warning "Package sync had issues, continuing anyway"
fi

# Check if Docker is running
print_status "Checking Docker status..."
if ! docker info >/dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker and try again."
    exit 1
fi
print_success "Docker is running"

# Start services with docker-compose
print_status "Starting required services (PostgreSQL, Elasticsearch, LocalStack)..."
cd "$PROJECT_ROOT"
docker-compose -f docker-compose.yml -f docker-compose.override.yml up -d postgres elasticsearch localstack 2>/dev/null || {
    print_warning "Some services may already be running"
}

# Wait for services to be healthy
print_status "Waiting for services to be ready..."
MAX_WAIT=60
WAITED=0

# Wait for PostgreSQL
while ! docker-compose exec -T postgres pg_isready -U postgres >/dev/null 2>&1; do
    sleep 1
    WAITED=$((WAITED + 1))
    if [ $WAITED -gt $MAX_WAIT ]; then
        print_error "PostgreSQL failed to start within $MAX_WAIT seconds"
        exit 1
    fi
    printf "."
done
echo ""
print_success "PostgreSQL is ready"

# Wait for Elasticsearch
while ! curl -s http://localhost:9200/_cluster/health >/dev/null 2>&1; do
    sleep 1
    WAITED=$((WAITED + 1))
    if [ $WAITED -gt $MAX_WAIT ]; then
        print_error "Elasticsearch failed to start within $MAX_WAIT seconds"
        exit 1
    fi
    printf "."
done
echo ""
print_success "Elasticsearch is ready"

# Wait for LocalStack
while ! curl -s http://localhost:4566/_localstack/health >/dev/null 2>&1; do
    sleep 1
    WAITED=$((WAITED + 1))
    if [ $WAITED -gt $MAX_WAIT ]; then
        print_error "LocalStack failed to start within $MAX_WAIT seconds"
        exit 1
    fi
    printf "."
done
echo ""
print_success "LocalStack is ready"

# Capture environment information
print_status "Capturing environment information..."
cat > "$ARTIFACTS_DIR/environment.json" <<EOF
{
  "timestamp": "$TIMESTAMP",
  "python_version": "$(uv run python --version 2>&1)",
  "uv_version": "$(uv --version 2>&1)",
  "os": "$(uname -s)",
  "os_version": "$(uname -r)",
  "git_branch": "$(git branch --show-current 2>/dev/null || echo 'unknown')",
  "git_commit": "$(git rev-parse HEAD 2>/dev/null || echo 'unknown')",
  "docker_compose_services": "postgres, elasticsearch, localstack"
}
EOF
print_success "Environment captured"

# Run linting
print_status "Running linting checks..."
LINT_OUTPUT=""
LINT_STATUS=0
if uv run pylint packages/*/src --rcfile=.pylintrc --output-format=json > "$ARTIFACTS_DIR/lint-report.json" 2>&1; then
    print_success "Linting passed"
else
    LINT_STATUS=$?
    if [ $LINT_STATUS -eq 2 ] || [ $LINT_STATUS -eq 4 ] || [ $LINT_STATUS -eq 8 ] || [ $LINT_STATUS -eq 16 ]; then
        print_warning "Linting found issues (exit code: $LINT_STATUS)"
    else
        print_error "Linting failed with error"
    fi
fi

# Run ruff for style checking
print_status "Running style checks with ruff..."
STYLE_STATUS=0
if uv run ruff check packages/*/src --output-format=json > "$ARTIFACTS_DIR/style-check.json" 2>&1; then
    print_success "Style checks passed"
else
    STYLE_STATUS=$?
    print_warning "Style check found issues"
fi

# Run unit tests (without external services)
print_status "Running unit tests..."
UNIT_TEST_STATUS=0

# First check if pytest is available and tests exist
if ! uv run python -c "import pytest" 2>/dev/null; then
    print_error "pytest not installed - run: uv sync --all-packages"
    UNIT_TEST_STATUS=1
else
    # Count tests first
    TEST_COUNT=$(uv run pytest packages/*/tests/ --co -q 2>/dev/null | grep -c "test" || echo "0")
    
    if [ "$TEST_COUNT" -eq "0" ]; then
        print_warning "No unit tests found"
        echo '<?xml version="1.0" encoding="utf-8"?><testsuites><testsuite name="pytest" tests="0" failures="0" errors="0" time="0.0"></testsuite></testsuites>' > "$ARTIFACTS_DIR/unit-test-results.xml"
        echo '<?xml version="1.0" encoding="utf-8"?><coverage version="1" line-rate="0"><packages></packages></coverage>' > "$ARTIFACTS_DIR/coverage-unit.xml"
    else
        # Run tests and capture exit code properly
        set +e  # Temporarily disable exit on error
        # Run tests, excluding only those explicitly marked as integration
        # Clean up any existing coverage data
        rm -f .coverage .coverage.*
        uv run pytest packages/*/tests/ -v \
            -m "not integration" \
            --junit-xml="$ARTIFACTS_DIR/unit-test-results.xml" \
            --cov=packages \
            --cov-report=xml:"$ARTIFACTS_DIR/coverage-unit.xml" \
            --cov-report=term 2>&1 | tee "$ARTIFACTS_DIR/unit-test-output.txt"
        UNIT_TEST_STATUS=${PIPESTATUS[0]}  # Get pytest exit code, not tee's
        # Save the coverage database file
        if [ -f .coverage ]; then
            cp .coverage "$ARTIFACTS_DIR/.coverage.unit"
        fi
        set -e  # Re-enable exit on error
        
        if [ $UNIT_TEST_STATUS -eq 0 ]; then
            print_success "Unit tests passed"
        elif [ $UNIT_TEST_STATUS -eq 5 ]; then
            print_warning "No unit tests collected (check test discovery)"
        else
            print_error "Unit tests failed (exit code: $UNIT_TEST_STATUS)"
        fi
    fi
fi

# Run integration tests (with external services)
print_status "Running integration tests..."
INTEGRATION_TEST_STATUS=0

# Set environment variables for integration tests
export DATABASE_URL="postgresql://postgres:postgres@localhost:5432/dataknobs"
export ELASTICSEARCH_URL="http://localhost:9200"
export AWS_ENDPOINT_URL="http://localhost:4566"
export AWS_ACCESS_KEY_ID="test"
export AWS_SECRET_ACCESS_KEY="test"
export AWS_DEFAULT_REGION="us-east-1"
export S3_BUCKET="dataknobs-local"

# Check for integration tests
INT_TEST_COUNT=$(uv run pytest packages/*/tests/ -m "integration" --co -q 2>/dev/null | grep -c "test" || echo "0")

if [ "$INT_TEST_COUNT" -eq "0" ]; then
    print_warning "No integration tests found"
    echo '<?xml version="1.0" encoding="utf-8"?><testsuites><testsuite name="pytest" tests="0" failures="0" errors="0" time="0.0"></testsuite></testsuites>' > "$ARTIFACTS_DIR/integration-test-results.xml"
    echo '<?xml version="1.0" encoding="utf-8"?><coverage version="1" line-rate="0"><packages></packages></coverage>' > "$ARTIFACTS_DIR/coverage-integration.xml"
else
    # Run tests and capture exit code properly
    set +e  # Temporarily disable exit on error
    # Clean up any existing coverage data
    rm -f .coverage .coverage.*
    uv run pytest packages/*/tests/ -v -m "integration" \
        --junit-xml="$ARTIFACTS_DIR/integration-test-results.xml" \
        --cov=packages \
        --cov-report=xml:"$ARTIFACTS_DIR/coverage-integration.xml" \
        --cov-report=term 2>&1 | tee "$ARTIFACTS_DIR/integration-test-output.txt"
    INTEGRATION_TEST_STATUS=${PIPESTATUS[0]}  # Get pytest exit code, not tee's
    # Save the coverage database file
    if [ -f .coverage ]; then
        cp .coverage "$ARTIFACTS_DIR/.coverage.integration"
    fi
    set -e  # Re-enable exit on error
    
    if [ $INTEGRATION_TEST_STATUS -eq 0 ]; then
        print_success "Integration tests passed"
    elif [ $INTEGRATION_TEST_STATUS -eq 5 ]; then
        print_warning "No integration tests collected"
    else
        print_error "Integration tests failed (exit code: $INTEGRATION_TEST_STATUS)"
    fi
fi

# Combine coverage reports
print_status "Processing coverage reports..."
if [ -f "$ARTIFACTS_DIR/.coverage.unit" ] && [ -f "$ARTIFACTS_DIR/.coverage.integration" ]; then
    # Combine the coverage database files
    cd "$ARTIFACTS_DIR"
    uv run coverage combine .coverage.unit .coverage.integration 2>/dev/null || {
        print_warning "Could not combine coverage databases, using unit coverage only"
        cp .coverage.unit .coverage 2>/dev/null || true
    }
    # Generate combined XML report
    if [ -f .coverage ]; then
        uv run coverage xml -o coverage.xml 2>/dev/null || {
            print_warning "Could not generate combined XML, using existing XML"
            if [ ! -f coverage.xml ] && [ -f coverage-unit.xml ]; then
                cp coverage-unit.xml coverage.xml
            fi
        }
    fi
    cd "$PROJECT_ROOT"
    print_success "Coverage reports processed"
elif [ -f "$ARTIFACTS_DIR/.coverage.unit" ]; then
    # Only unit test coverage available
    cd "$ARTIFACTS_DIR"
    cp .coverage.unit .coverage
    uv run coverage xml -o coverage.xml 2>/dev/null || {
        if [ -f coverage-unit.xml ]; then
            cp coverage-unit.xml coverage.xml
        fi
    }
    cd "$PROJECT_ROOT"
    print_success "Coverage report created from unit tests"
elif [ -f "$ARTIFACTS_DIR/.coverage.integration" ]; then
    # Only integration test coverage available
    cd "$ARTIFACTS_DIR"
    cp .coverage.integration .coverage
    uv run coverage xml -o coverage.xml 2>/dev/null || {
        if [ -f coverage-integration.xml ]; then
            cp coverage-integration.xml coverage.xml
        fi
    }
    cd "$PROJECT_ROOT"
    print_success "Coverage report created from integration tests"
elif [ -f "$ARTIFACTS_DIR/coverage-unit.xml" ] || [ -f "$ARTIFACTS_DIR/coverage-integration.xml" ]; then
    # Fallback: use existing XML reports if no database files
    if [ -f "$ARTIFACTS_DIR/coverage-unit.xml" ] && [ -f "$ARTIFACTS_DIR/coverage-integration.xml" ]; then
        # Both XML files exist, just copy unit as we can't merge XML
        cp "$ARTIFACTS_DIR/coverage-unit.xml" "$ARTIFACTS_DIR/coverage.xml"
        print_warning "Using unit test coverage XML (cannot merge XML files)"
    elif [ -f "$ARTIFACTS_DIR/coverage-unit.xml" ]; then
        cp "$ARTIFACTS_DIR/coverage-unit.xml" "$ARTIFACTS_DIR/coverage.xml"
    else
        cp "$ARTIFACTS_DIR/coverage-integration.xml" "$ARTIFACTS_DIR/coverage.xml"
    fi
    print_success "Coverage report created from existing XML"
else
    # Create empty coverage report if no tests ran
    echo '<?xml version="1.0" encoding="utf-8"?><coverage version="1" line-rate="0"><packages></packages></coverage>' > "$ARTIFACTS_DIR/coverage.xml"
    print_warning "No coverage data available"
fi

# Generate summary
print_status "Generating quality summary..."
OVERALL_STATUS="PASS"
if [ $UNIT_TEST_STATUS -ne 0 ] || [ $INTEGRATION_TEST_STATUS -ne 0 ]; then
    OVERALL_STATUS="FAIL"
fi

cat > "$ARTIFACTS_DIR/quality-summary.json" <<EOF
{
  "timestamp": "$TIMESTAMP",
  "overall_status": "$OVERALL_STATUS",
  "checks": {
    "lint": {
      "status": $([ $LINT_STATUS -eq 0 ] && echo '"pass"' || echo '"warning"'),
      "exit_code": $LINT_STATUS
    },
    "style": {
      "status": $([ $STYLE_STATUS -eq 0 ] && echo '"pass"' || echo '"warning"'),
      "exit_code": $STYLE_STATUS
    },
    "unit_tests": {
      "status": $([ $UNIT_TEST_STATUS -eq 0 ] && echo '"pass"' || echo '"fail"'),
      "exit_code": $UNIT_TEST_STATUS
    },
    "integration_tests": {
      "status": $([ $INTEGRATION_TEST_STATUS -eq 0 ] && echo '"pass"' || echo '"fail"'),
      "exit_code": $INTEGRATION_TEST_STATUS
    }
  }
}
EOF

# Generate signature of artifacts
print_status "Generating artifact signature..."
cd "$ARTIFACTS_DIR"
find . -type f -name "*.json" -o -name "*.xml" | sort | xargs sha256sum > signature.sha256
cd "$PROJECT_ROOT"
print_success "Artifact signature generated"

# Print summary
echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}                        Quality Check Summary                     ${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""

if [ $LINT_STATUS -eq 0 ]; then
    echo -e "  Linting:           ${GREEN}✓ PASSED${NC}"
else
    echo -e "  Linting:           ${YELLOW}⚠ WARNINGS${NC}"
fi

if [ $STYLE_STATUS -eq 0 ]; then
    echo -e "  Style Check:       ${GREEN}✓ PASSED${NC}"
else
    echo -e "  Style Check:       ${YELLOW}⚠ WARNINGS${NC}"
fi

if [ $UNIT_TEST_STATUS -eq 0 ]; then
    echo -e "  Unit Tests:        ${GREEN}✓ PASSED${NC}"
else
    echo -e "  Unit Tests:        ${RED}✗ FAILED${NC}"
fi

if [ $INTEGRATION_TEST_STATUS -eq 0 ]; then
    echo -e "  Integration Tests: ${GREEN}✓ PASSED${NC}"
else
    echo -e "  Integration Tests: ${RED}✗ FAILED${NC}"
fi

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"

if [ "$OVERALL_STATUS" = "PASS" ]; then
    echo ""
    echo -e "${GREEN}✓ All critical checks passed!${NC}"
    echo -e "${GREEN}  Artifacts saved to: .quality-artifacts/${NC}"
    echo -e "${GREEN}  You can now create your pull request.${NC}"
    echo ""
    exit 0
else
    echo ""
    echo -e "${RED}✗ Some checks failed!${NC}"
    echo -e "${RED}  Please fix the issues and run this script again.${NC}"
    echo -e "${RED}  Check the output files in .quality-artifacts/ for details.${NC}"
    echo ""
    exit 1
fi