#!/bin/bash
set -e

# Quality Checks Script - Internal Version (runs inside Docker container)
# This version is for running quality checks from within the dataknobs-dev container

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ARTIFACTS_DIR="$PROJECT_ROOT/.quality-artifacts"
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}    DataKnobs Quality Checks - Internal (Docker) Version          ${NC}"
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
if "$SCRIPT_DIR/sync-packages.sh" >/dev/null 2>&1; then
    print_success "Packages synced"
else
    print_warning "Package sync had issues, continuing anyway"
fi

# Check if we're inside a container
if [ -f /.dockerenv ] || [ -n "$DOCKER_CONTAINER" ]; then
    print_success "Running inside Docker container"
    
    # Use network service names for connections
    export DATABASE_URL="postgresql://postgres:postgres@postgres:5432/dataknobs"
    export ELASTICSEARCH_URL="http://elasticsearch:9200"
    export AWS_ENDPOINT_URL="http://localstack:4566"
else
    print_warning "Not running inside Docker container - use bin/run-quality-checks.sh instead"
    exit 1
fi

# Set AWS credentials for LocalStack
export AWS_ACCESS_KEY_ID="test"
export AWS_SECRET_ACCESS_KEY="test"
export AWS_DEFAULT_REGION="us-east-1"
export S3_BUCKET="dataknobs-local"

# Test service connectivity
print_status "Testing service connectivity..."

# Test PostgreSQL
if python3 -c "import psycopg2; psycopg2.connect('$DATABASE_URL')" 2>/dev/null; then
    print_success "PostgreSQL is accessible"
else
    print_warning "PostgreSQL not accessible - integration tests may fail"
fi

# Test Elasticsearch
if curl -s "$ELASTICSEARCH_URL/_cluster/health" >/dev/null 2>&1; then
    print_success "Elasticsearch is accessible"
else
    print_warning "Elasticsearch not accessible - integration tests may fail"
fi

# Test LocalStack
if curl -s "$AWS_ENDPOINT_URL/_localstack/health" >/dev/null 2>&1; then
    print_success "LocalStack is accessible"
else
    print_warning "LocalStack not accessible - integration tests may fail"
fi

# Capture environment information
print_status "Capturing environment information..."
cat > "$ARTIFACTS_DIR/environment.json" <<EOF
{
  "timestamp": "$TIMESTAMP",
  "python_version": "$(python3 --version 2>&1)",
  "uv_version": "$(uv --version 2>&1 || echo 'uv not found')",
  "os": "$(uname -s)",
  "os_version": "$(uname -r)",
  "git_branch": "$(git branch --show-current 2>/dev/null || echo 'unknown')",
  "git_commit": "$(git rev-parse HEAD 2>/dev/null || echo 'unknown')",
  "docker_container": "true",
  "docker_services": "postgres, elasticsearch, localstack"
}
EOF
print_success "Environment captured"

# Initialize test tracking
LINT_STATUS=0
STYLE_STATUS=0
UNIT_TEST_STATUS=0
INTEGRATION_TEST_STATUS=0

# Run linting
print_status "Running linting checks..."
if uv run pylint packages/*/src --rcfile=.pylintrc --output-format=json > "$ARTIFACTS_DIR/lint-report.json" 2>&1; then
    print_success "Linting passed"
    LINT_STATUS=0
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
if uv run ruff check packages/*/src --output-format=json > "$ARTIFACTS_DIR/style-check.json" 2>&1; then
    print_success "Style checks passed"
    STYLE_STATUS=0
else
    STYLE_STATUS=$?
    print_warning "Style check found issues"
fi

# Run unit tests (without external services)
print_status "Running unit tests..."

# First check if tests exist
if uv run python -c "import pytest" 2>/dev/null; then
    # Count tests first
    TEST_COUNT=$(uv run pytest packages/*/tests/ --co -q 2>/dev/null | grep -c "test" || echo "0")
    
    if [ "$TEST_COUNT" -eq "0" ]; then
        print_warning "No tests found"
        echo '<?xml version="1.0" encoding="utf-8"?><testsuites><testsuite name="pytest" tests="0" failures="0" errors="0" time="0.0"></testsuite></testsuites>' > "$ARTIFACTS_DIR/unit-test-results.xml"
        echo '<?xml version="1.0" encoding="utf-8"?><coverage version="1" line-rate="0"><packages></packages></coverage>' > "$ARTIFACTS_DIR/coverage-unit.xml"
        UNIT_TEST_STATUS=0
    else
        # Run unit tests and capture exit code properly
        set +e  # Temporarily disable exit on error
        uv run pytest packages/*/tests/ -v -m "not integration" \
            --junit-xml="$ARTIFACTS_DIR/unit-test-results.xml" \
            --cov=packages \
            --cov-report=xml:"$ARTIFACTS_DIR/coverage-unit.xml" \
            --cov-report=term 2>&1 | tee "$ARTIFACTS_DIR/unit-test-output.txt"
        UNIT_TEST_STATUS=${PIPESTATUS[0]}  # Get pytest exit code, not tee's
        set -e  # Re-enable exit on error
        
        if [ $UNIT_TEST_STATUS -eq 0 ]; then
            print_success "Unit tests passed"
        elif [ $UNIT_TEST_STATUS -eq 5 ]; then
            print_warning "No unit tests collected (check test discovery)"
        else
            print_error "Unit tests failed (exit code: $UNIT_TEST_STATUS)"
        fi
    fi
else
    print_error "pytest not installed"
    UNIT_TEST_STATUS=1
fi

# Run integration tests (with external services)
print_status "Running integration tests..."

# Check for integration tests
INT_TEST_COUNT=$(uv run pytest packages/*/tests/ -m "integration" --co -q 2>/dev/null | grep -c "test" || echo "0")

if [ "$INT_TEST_COUNT" -eq "0" ]; then
    print_warning "No integration tests found"
    echo '<?xml version="1.0" encoding="utf-8"?><testsuites><testsuite name="pytest" tests="0" failures="0" errors="0" time="0.0"></testsuite></testsuites>' > "$ARTIFACTS_DIR/integration-test-results.xml"
    echo '<?xml version="1.0" encoding="utf-8"?><coverage version="1" line-rate="0"><packages></packages></coverage>' > "$ARTIFACTS_DIR/coverage-integration.xml"
    INTEGRATION_TEST_STATUS=0
else
    # Run tests and capture exit code properly
    set +e  # Temporarily disable exit on error
    uv run pytest packages/*/tests/ -v -m "integration" \
        --junit-xml="$ARTIFACTS_DIR/integration-test-results.xml" \
        --cov=packages \
        --cov-report=xml:"$ARTIFACTS_DIR/coverage-integration.xml" \
        --cov-report=term 2>&1 | tee "$ARTIFACTS_DIR/integration-test-output.txt"
    INTEGRATION_TEST_STATUS=${PIPESTATUS[0]}  # Get pytest exit code, not tee's
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
if [ -f "$ARTIFACTS_DIR/coverage-unit.xml" ] && [ -f "$ARTIFACTS_DIR/coverage-integration.xml" ]; then
    # Try to combine, but don't fail if it doesn't work
    uv run coverage combine "$ARTIFACTS_DIR/coverage-unit.xml" "$ARTIFACTS_DIR/coverage-integration.xml" 2>/dev/null || true
    uv run coverage xml -o "$ARTIFACTS_DIR/coverage.xml" 2>/dev/null || true
    
    if [ ! -f "$ARTIFACTS_DIR/coverage.xml" ]; then
        cp "$ARTIFACTS_DIR/coverage-unit.xml" "$ARTIFACTS_DIR/coverage.xml" 2>/dev/null || true
    fi
    print_success "Coverage reports processed"
else
    if [ -f "$ARTIFACTS_DIR/coverage-unit.xml" ]; then
        cp "$ARTIFACTS_DIR/coverage-unit.xml" "$ARTIFACTS_DIR/coverage.xml" 2>/dev/null || true
    else
        echo '<?xml version="1.0" encoding="utf-8"?><coverage version="1" line-rate="0"><packages></packages></coverage>' > "$ARTIFACTS_DIR/coverage.xml"
    fi
fi

# Generate summary
print_status "Generating quality summary..."
OVERALL_STATUS="PASS"

# Check for actual test failures
if [ $UNIT_TEST_STATUS -ne 0 ] || [ $INTEGRATION_TEST_STATUS -ne 0 ]; then
    OVERALL_STATUS="FAIL"
fi

cat > "$ARTIFACTS_DIR/quality-summary.json" <<EOF
{
  "timestamp": "$TIMESTAMP",
  "overall_status": "$OVERALL_STATUS",
  "environment": "docker-internal",
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