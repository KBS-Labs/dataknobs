#!/bin/bash
set -e

# Quality Checks Script for DataKnobs
# This script runs all quality checks including unit tests, integration tests,
# linting, and code coverage. Results are saved as artifacts for CI validation.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ARTIFACTS_DIR="$PROJECT_ROOT/.quality-artifacts"
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# Default values
PACKAGES=""
SKIP_STYLE="no"
SKIP_TESTS="no"
PYTEST_ARGS=""
KEEP_SERVICES="false"
PR_MODE="auto"  # auto, yes, no

# Check if we're inside a Docker container
IN_DOCKER=false
if [ -f /.dockerenv ] || [ -n "${DOCKER_CONTAINER:-}" ]; then
    IN_DOCKER=true
fi

# Colors for output (check if terminal supports colors)
if [ -t 1 ] && [ -n "${TERM:-}" ] && [ "${TERM}" != "dumb" ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    CYAN='\033[0;36m'
    NC='\033[0m' # No Color
else
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    CYAN=''
    NC=''
fi

# Function to show usage
show_usage() {
    cat << EOF
${CYAN}DataKnobs Quality Checks Runner${NC}

Usage: $0 [OPTIONS] [PACKAGE...] [-- PYTEST_ARGS]

Run quality checks (linting, style, tests) for DataKnobs packages.

${YELLOW}Options:${NC}
    -p, --package PACKAGE    Package to check (can be specified multiple times)
                            If not specified, checks all packages
    --pr                    PR mode: Run full quality checks with artifacts
                            (unit and integration tests separately)
    --dev                   Dev mode: Run quick checks without artifacts
                            (combined tests, no artifact pollution)
    --skip-style            Skip style checks (ruff)
    --skip-tests            Skip test execution
    --keep-services         Keep services running after completion
    -h, --help              Show this help message

${YELLOW}Advanced Usage:${NC}
    Any arguments after -- are passed directly to pytest:
    $0 data -- -xvs --tb=short
    
${YELLOW}Examples:${NC}
    $0                      # PR mode: Full checks for all packages with artifacts
    $0 --pr                 # Explicit PR mode for all packages
    $0 --dev data           # Dev mode: Quick checks for data package
    $0 data config          # Dev mode: Check specific packages (no artifacts)
    $0 --pr data            # PR mode for data package only
    $0 --skip-style         # Run all checks except style checks
    $0 data -- -x           # Run data package with pytest -x flag

${YELLOW}Environment:${NC}
    Running in: $([ "$IN_DOCKER" = true ] && echo "Docker container" || echo "Host system")
    
${YELLOW}Output:${NC}
    All artifacts are saved to: .quality-artifacts/
    - environment.json: System information
    - lint-report.json: Linting results
    - style-check.json: Style check results
    - *-test-results.xml: Test results in JUnit format
    - coverage*.xml: Coverage reports
    - quality-summary.json: Overall summary

EOF
    exit 0
}

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

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--package)
            PACKAGES="$PACKAGES $2"
            shift 2
            ;;
        --pr)
            PR_MODE="yes"
            shift
            ;;
        --dev)
            PR_MODE="no"
            shift
            ;;
        --skip-style)
            SKIP_STYLE="yes"
            shift
            ;;
        --skip-tests)
            SKIP_TESTS="yes"
            shift
            ;;
        --keep-services)
            KEEP_SERVICES="true"
            shift
            ;;
        -h|--help)
            show_usage
            ;;
        --)
            shift
            PYTEST_ARGS="$@"
            break
            ;;
        *)
            # Assume it's a package name
            PACKAGES="$PACKAGES $1"
            shift
            ;;
    esac
done

# Auto-detect mode if not explicitly set
if [ "$PR_MODE" = "auto" ]; then
    if [ -n "$PACKAGES" ]; then
        # If specific packages are selected, assume dev mode
        PR_MODE="no"
    else
        # If running all packages, assume PR mode
        PR_MODE="yes"
    fi
fi

# Function to set environment variables based on context
set_environment_vars() {
    if [ "$IN_DOCKER" = true ]; then
        # Inside Docker container - use service names
        export DATABASE_URL="postgresql://postgres:postgres@postgres:5432/dataknobs"
        export ELASTICSEARCH_URL="http://elasticsearch:9200"
        export AWS_ENDPOINT_URL="http://localstack:4566"
        export LOCALSTACK_ENDPOINT="http://localstack:4566"
        
        # Individual host/port variables for test fixtures
        export POSTGRES_HOST="postgres"
        export POSTGRES_PORT=5432
        export POSTGRES_USER="postgres"
        export POSTGRES_PASSWORD="postgres"
        export POSTGRES_DB="dataknobs_test"
        
        export ELASTICSEARCH_HOST="elasticsearch"
        export ELASTICSEARCH_PORT=9200
    else
        # On host system - use localhost
        export DATABASE_URL="postgresql://postgres:postgres@localhost:5432/dataknobs"
        export ELASTICSEARCH_URL="http://localhost:9200"
        export AWS_ENDPOINT_URL="http://localhost:4566"
        export LOCALSTACK_ENDPOINT="http://localhost:4566"
        
        # Individual host/port variables for test fixtures
        export POSTGRES_HOST="localhost"
        export POSTGRES_PORT=5432
        export POSTGRES_USER="postgres"
        export POSTGRES_PASSWORD="postgres"
        export POSTGRES_DB="dataknobs_test"
        
        export ELASTICSEARCH_HOST="localhost"
        export ELASTICSEARCH_PORT=9200
    fi
    
    # AWS credentials for LocalStack (same regardless of environment)
    export AWS_ACCESS_KEY_ID="test"
    export AWS_SECRET_ACCESS_KEY="test"
    export AWS_DEFAULT_REGION="us-east-1"
    export S3_BUCKET="dataknobs-local"
    
    # Enable test flags for integration tests
    export TEST_S3="true"
    export TEST_ELASTICSEARCH="true"
    export TEST_POSTGRES="true"
    export TEST_OLLAMA="true"
}

# Function to cleanup resources
cleanup() {
    if [ "$IN_DOCKER" = false ]; then
        # Only cleanup if manage-services.sh indicates we started them
        if [ -f "/tmp/.dataknobs_services_started_$$" ]; then
            if [ "${KEEP_SERVICES}" != "true" ]; then
                echo ""
                print_status "Cleaning up services..."
                "$SCRIPT_DIR/manage-services.sh" stop
            else
                echo ""
                print_status "Services are still running. To stop them, run:"
                echo "$SCRIPT_DIR/manage-services.sh stop"
            fi
        fi
    fi
}

# Set trap for cleanup on exit
trap cleanup EXIT INT TERM

echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
if [ "$PR_MODE" = "yes" ]; then
    echo -e "${BLUE}       DataKnobs Quality Checks - PR Mode                         ${NC}"
else
    echo -e "${BLUE}       DataKnobs Quality Checks - Developer Mode                  ${NC}"
fi
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""

# Display environment
if [ "$IN_DOCKER" = true ]; then
    print_status "Environment: Docker Container"
else
    print_status "Environment: Host System"
fi

# Display mode and packages
if [ "$PR_MODE" = "yes" ]; then
    print_status "Mode: PR (full checks with artifacts)"
else
    print_status "Mode: Developer (quick checks, no artifacts)"
fi

# Display packages to check
if [ -n "$PACKAGES" ]; then
    print_status "Packages: $PACKAGES"
else
    print_status "Packages: All"
fi

if [ -n "$PYTEST_ARGS" ]; then
    print_status "Pytest args: $PYTEST_ARGS"
fi
echo ""

# Create artifacts directory only in PR mode
if [ "$PR_MODE" = "yes" ]; then
    mkdir -p "$ARTIFACTS_DIR"
fi

# Ensure all packages are installed
print_status "Ensuring all packages are installed..."
if [ "$IN_DOCKER" = true ]; then
    # Use sync-packages.sh if available in Docker
    if [ -f "$SCRIPT_DIR/sync-packages.sh" ]; then
        "$SCRIPT_DIR/sync-packages.sh" >/dev/null 2>&1 || print_warning "Package sync had issues"
    else
        uv sync --all-packages >/dev/null 2>&1 || print_warning "Package sync had issues"
    fi
else
    uv sync --all-packages >/dev/null 2>&1 || print_warning "Package sync had issues"
fi
print_success "Packages synced"

# Start services if needed (only on host)
if [ "$IN_DOCKER" = false ] && [ "$SKIP_TESTS" != "yes" ]; then
    print_status "Ensuring test services are running..."
    if ! "$SCRIPT_DIR/manage-services.sh" ensure; then
        print_error "Failed to start services"
        exit 1
    fi
    print_success "Services are ready"
elif [ "$IN_DOCKER" = true ] && [ "$SKIP_TESTS" != "yes" ]; then
    print_status "Testing service connectivity..."
    
    # Set environment for connectivity tests
    set_environment_vars
    
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
fi

# Set environment variables for all checks
set_environment_vars

# Capture environment information (PR mode only)
if [ "$PR_MODE" = "yes" ]; then
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
  "docker_container": "$IN_DOCKER",
  "docker_compose_services": "postgres, elasticsearch, localstack"
}
EOF
    print_success "Environment captured"
fi

# Initialize status tracking
STYLE_STATUS=0
DOCS_STATUS=0
DOCS_VERSIONS_STATUS=0
TEST_STATUS=0
UNIT_TEST_STATUS=0
INTEGRATION_TEST_STATUS=0

# Determine package glob pattern
if [ -n "$PACKAGES" ]; then
    # Build glob pattern for specific packages
    PACKAGE_PATTERN=""
    for pkg in $PACKAGES; do
        if [ -d "packages/$pkg" ]; then
            if [ -z "$PACKAGE_PATTERN" ]; then
                PACKAGE_PATTERN="packages/$pkg/src"
            else
                PACKAGE_PATTERN="$PACKAGE_PATTERN packages/$pkg/src"
            fi
        else
            print_warning "Package not found: $pkg"
        fi
    done
    
    if [ -z "$PACKAGE_PATTERN" ]; then
        print_error "No valid packages specified"
        exit 1
    fi
else
    PACKAGE_PATTERN="packages/*/src"
fi

# Validate package references
print_status "Validating package references across codebase..."
if uv run python "$SCRIPT_DIR/validate-package-references.py" > "$ARTIFACTS_DIR/package-validation.log" 2>&1; then
    print_success "Package references are consistent"
else
    print_error "Package validation failed - see $ARTIFACTS_DIR/package-validation.log"
    cat "$ARTIFACTS_DIR/package-validation.log"
    exit 1
fi

# Build documentation (PR mode only)
if [ "$PR_MODE" = "yes" ]; then
    print_status "Building documentation (checking for errors)..."
    if env NO_MKDOCS_2_WARNING=1 uv run mkdocs build --strict > "$ARTIFACTS_DIR/docs-build.log" 2>&1; then
        print_success "Documentation builds without errors or warnings"
    else
        DOCS_STATUS=$?
        print_error "Documentation build failed - see $ARTIFACTS_DIR/docs-build.log"
        echo ""
        echo -e "${YELLOW}Documentation errors:${NC}"
        cat "$ARTIFACTS_DIR/docs-build.log"
        echo ""
    fi

    # Check documentation versions are in sync with packages.json
    print_status "Checking documentation versions..."
    if "$SCRIPT_DIR/docs-update-versions.sh" --check > "$ARTIFACTS_DIR/docs-versions.log" 2>&1; then
        print_success "Documentation versions are in sync"
    else
        DOCS_VERSIONS_STATUS=$?
        print_error "Documentation versions are out of sync"
        echo ""
        echo -e "${YELLOW}Version sync errors:${NC}"
        cat "$ARTIFACTS_DIR/docs-versions.log"
        echo ""
        echo -e "${CYAN}Run 'bin/docs-update-versions.sh' to fix${NC}"
        echo ""
    fi
fi

# Run style checks (using ruff for linting and style)
if [ "$SKIP_STYLE" != "yes" ]; then
    print_status "Running style checks with ruff..."
    if [ "$PR_MODE" = "yes" ]; then
        # PR mode: save output to artifacts (use config to respect suppressions)
        if uv run ruff check $PACKAGE_PATTERN --output-format=json --config "$PROJECT_ROOT/pyproject.toml" > "$ARTIFACTS_DIR/style-check.json" 2>&1; then
            print_success "Style checks passed"
        else
            STYLE_STATUS=$?
            print_warning "Style check found issues"
        fi
    else
        # Dev mode: show output directly (use config to respect suppressions)
        if uv run ruff check $PACKAGE_PATTERN --config "$PROJECT_ROOT/pyproject.toml"; then
            print_success "Style checks passed"
        else
            STYLE_STATUS=$?
            print_warning "Style check found issues"
        fi
    fi
else
    print_status "Skipping style checks"
fi

# Run tests using the test.sh script
if [ "$SKIP_TESTS" != "yes" ]; then
    print_status "Running tests..."
    
    # Build test command
    TEST_CMD="$SCRIPT_DIR/test.sh"
    
    if [ "$PR_MODE" = "yes" ]; then
        # PR mode: Run unit and integration tests separately with artifacts
        
        # Run unit tests
        print_status "Running unit tests..."
        if [ -n "$PACKAGES" ]; then
            # Run unit tests for specific packages
            for pkg in $PACKAGES; do
                if [ -d "packages/$pkg" ]; then
                    if [ -n "$PYTEST_ARGS" ]; then
                        $TEST_CMD "$pkg" -t unit --cov-report xml -- $PYTEST_ARGS 2>&1 | tee "$ARTIFACTS_DIR/unit-test-output-$pkg.txt"
                    else
                        $TEST_CMD "$pkg" -t unit --cov-report xml 2>&1 | tee "$ARTIFACTS_DIR/unit-test-output-$pkg.txt"
                    fi
                    
                    pkg_status=${PIPESTATUS[0]}
                    if [ $pkg_status -ne 0 ]; then
                        UNIT_TEST_STATUS=$pkg_status
                    fi
                    if [ -f "coverage.xml" ]; then
                        mv coverage.xml "$ARTIFACTS_DIR/coverage-unit-$pkg.xml"
                    fi
                    # Also save the .coverage data file for combining later
                    if [ -f ".coverage" ]; then
                        cp .coverage "$ARTIFACTS_DIR/.coverage.unit.$pkg"
                    fi
                fi
            done
        else
            # Run all unit tests - but do it per package to preserve individual coverage
            for pkg_dir in "$PROJECT_ROOT"/packages/*/; do
                if [ -d "$pkg_dir" ]; then
                    pkg_name=$(basename "$pkg_dir")
                    # Skip packages without tests
                    if [ -d "$pkg_dir/tests" ]; then
                        print_status "Running unit tests for $pkg_name..."
                        if [ -n "$PYTEST_ARGS" ]; then
                            $TEST_CMD "$pkg_name" -t unit --cov-report xml -- $PYTEST_ARGS 2>&1 | tee -a "$ARTIFACTS_DIR/unit-test-output.txt"
                        else
                            $TEST_CMD "$pkg_name" -t unit --cov-report xml 2>&1 | tee -a "$ARTIFACTS_DIR/unit-test-output.txt"
                        fi
                        
                        pkg_status=${PIPESTATUS[0]}
                        if [ $pkg_status -ne 0 ]; then
                            UNIT_TEST_STATUS=$pkg_status
                        fi
                        
                        # Save individual package coverage
                        if [ -f "coverage.xml" ]; then
                            mv coverage.xml "$ARTIFACTS_DIR/coverage-unit-$pkg_name.xml"
                        fi
                        if [ -f ".coverage" ]; then
                            mv .coverage "$ARTIFACTS_DIR/.coverage.unit.$pkg_name"
                        fi
                    fi
                fi
            done
        fi
        
        # Run integration tests
        print_status "Running integration tests..."
        if [ -n "$PACKAGES" ]; then
            # Run integration tests for specific packages
            for pkg in $PACKAGES; do
                if [ -d "packages/$pkg" ]; then
                    if [ -n "$PYTEST_ARGS" ]; then
                        $TEST_CMD "$pkg" -t integration --cov-report xml -- $PYTEST_ARGS 2>&1 | tee "$ARTIFACTS_DIR/integration-test-output-$pkg.txt"
                    else
                        $TEST_CMD "$pkg" -t integration --cov-report xml 2>&1 | tee "$ARTIFACTS_DIR/integration-test-output-$pkg.txt"
                    fi
                    
                    pkg_status=${PIPESTATUS[0]}
                    if [ $pkg_status -ne 0 ]; then
                        INTEGRATION_TEST_STATUS=$pkg_status
                    fi
                    if [ -f "coverage.xml" ]; then
                        mv coverage.xml "$ARTIFACTS_DIR/coverage-integration-$pkg.xml"
                    fi
                    # Also save the .coverage data file for combining later
                    if [ -f ".coverage" ]; then
                        cp .coverage "$ARTIFACTS_DIR/.coverage.integration.$pkg"
                    fi
                fi
            done
        else
            # Run all integration tests - per package that has them
            for pkg_dir in "$PROJECT_ROOT"/packages/*/; do
                if [ -d "$pkg_dir" ]; then
                    pkg_name=$(basename "$pkg_dir")
                    # Check if integration tests exist
                    if [ -d "$pkg_dir/tests/integration" ]; then
                        print_status "Running integration tests for $pkg_name..."
                        if [ -n "$PYTEST_ARGS" ]; then
                            $TEST_CMD "$pkg_name" -t integration --cov-report xml -- $PYTEST_ARGS 2>&1 | tee -a "$ARTIFACTS_DIR/integration-test-output.txt"
                        else
                            $TEST_CMD "$pkg_name" -t integration --cov-report xml 2>&1 | tee -a "$ARTIFACTS_DIR/integration-test-output.txt"
                        fi
                        
                        pkg_status=${PIPESTATUS[0]}
                        if [ $pkg_status -ne 0 ]; then
                            INTEGRATION_TEST_STATUS=$pkg_status
                        fi
                        
                        # Save individual package coverage
                        if [ -f "coverage.xml" ]; then
                            mv coverage.xml "$ARTIFACTS_DIR/coverage-integration-$pkg_name.xml"
                        fi
                        if [ -f ".coverage" ]; then
                            mv .coverage "$ARTIFACTS_DIR/.coverage.integration.$pkg_name"
                        fi
                    fi
                fi
            done
        fi
        
        # Set overall test status
        if [ $UNIT_TEST_STATUS -ne 0 ] || [ $INTEGRATION_TEST_STATUS -ne 0 ]; then
            TEST_STATUS=1
        else
            TEST_STATUS=0
        fi
        
        if [ $UNIT_TEST_STATUS -eq 0 ]; then
            print_success "Unit tests passed"
        else
            print_error "Unit tests failed"
        fi
        
        if [ $INTEGRATION_TEST_STATUS -eq 0 ]; then
            print_success "Integration tests passed"
        else
            print_error "Integration tests failed"
        fi
    else
        # Dev mode: Run combined tests without polluting artifacts
        if [ -n "$PACKAGES" ]; then
            # Run tests for each package
            for pkg in $PACKAGES; do
                if [ -d "packages/$pkg" ]; then
                    print_status "Testing package: $pkg"
                    
                    if [ -n "$PYTEST_ARGS" ]; then
                        $TEST_CMD "$pkg" -- $PYTEST_ARGS
                    else
                        $TEST_CMD "$pkg"
                    fi
                    
                    pkg_status=$?
                    if [ $pkg_status -ne 0 ]; then
                        TEST_STATUS=$pkg_status
                        print_error "Tests failed for package: $pkg"
                    else
                        print_success "Tests passed for package: $pkg"
                    fi
                fi
            done
        else
            # This shouldn't happen in dev mode (auto-detection), but handle it
            if [ -n "$PYTEST_ARGS" ]; then
                $TEST_CMD -- $PYTEST_ARGS
            else
                $TEST_CMD
            fi
            
            TEST_STATUS=$?
            if [ $TEST_STATUS -eq 0 ]; then
                print_success "All tests passed"
            else
                print_error "Some tests failed"
            fi
        fi
    fi
    
    # Create test results XML files for CI systems (PR mode only)
    if [ "$PR_MODE" = "yes" ]; then
        if [ ! -f "$ARTIFACTS_DIR/unit-test-results.xml" ]; then
            echo '<?xml version="1.0" encoding="utf-8"?><testsuites></testsuites>' > "$ARTIFACTS_DIR/unit-test-results.xml"
        fi
        if [ ! -f "$ARTIFACTS_DIR/integration-test-results.xml" ]; then
            echo '<?xml version="1.0" encoding="utf-8"?><testsuites></testsuites>' > "$ARTIFACTS_DIR/integration-test-results.xml"
        fi
    fi
else
    print_status "Skipping tests"
fi

# Process coverage and generate artifacts (PR mode only)
if [ "$PR_MODE" = "yes" ]; then
    # Combine coverage data files if they exist
    if ls "$ARTIFACTS_DIR"/.coverage.* >/dev/null 2>&1; then
        print_status "Combining coverage reports..."
        
        # Change to artifacts directory for coverage operations
        cd "$ARTIFACTS_DIR"
        
        # Combine all .coverage.* files
        if uv run coverage combine .coverage.* 2>/dev/null; then
            print_success "Coverage data combined"
            
            # Generate combined XML report
            if uv run coverage xml -o coverage.xml 2>/dev/null; then
                print_success "Combined coverage XML generated"
                
                # Also generate per-package XML reports
                for pkg_dir in "$PROJECT_ROOT"/packages/*/; do
                    if [ -d "$pkg_dir" ]; then
                        pkg_name=$(basename "$pkg_dir")
                        # Special case: legacy package is named "dataknobs" not "dataknobs_legacy"
                        if [ "$pkg_name" = "legacy" ]; then
                            src_name="dataknobs"
                        else
                            src_name="dataknobs_${pkg_name}"
                        fi
                        if [ -d "$pkg_dir/src/${src_name}" ]; then
                            # Generate package-specific XML
                            uv run coverage xml -o "coverage-${pkg_name}-combined.xml" --include="*/${src_name}/*" 2>/dev/null || true
                        fi
                    fi
                done
                print_success "Per-package coverage XML generated"
            else
                print_warning "Could not generate combined XML"
                # Fall back to using individual XML files
                if [ -f "coverage-unit.xml" ]; then
                    cp coverage-unit.xml coverage.xml
                elif [ -f "coverage-integration.xml" ]; then
                    cp coverage-integration.xml coverage.xml
                fi
            fi
            
            # Generate combined HTML report for easier viewing
            if uv run coverage html -d htmlcov 2>/dev/null; then
                print_success "Combined coverage HTML generated in .quality-artifacts/htmlcov/"
            fi
            
            # Generate terminal report to show combined coverage
            echo "" >> test-coverage-summary.txt
            echo "Combined Coverage Report:" >> test-coverage-summary.txt
            echo "=========================" >> test-coverage-summary.txt
            uv run coverage report >> test-coverage-summary.txt 2>&1 || true
            
            # Generate per-package coverage summary
            echo "" >> test-coverage-summary.txt
            echo "Coverage by Package:" >> test-coverage-summary.txt
            echo "====================" >> test-coverage-summary.txt
            
            # Extract package-specific coverage from the combined data
            for pkg_dir in "$PROJECT_ROOT"/packages/*/; do
                if [ -d "$pkg_dir" ]; then
                    pkg_name=$(basename "$pkg_dir")
                    # Special case: legacy package is named "dataknobs" not "dataknobs_legacy"
                    if [ "$pkg_name" = "legacy" ]; then
                        src_name="dataknobs"
                    else
                        src_name="dataknobs_${pkg_name}"
                    fi
                    # Skip packages without source code
                    if [ -d "$pkg_dir/src/${src_name}" ]; then
                        echo "" >> test-coverage-summary.txt
                        echo "Package: $pkg_name" >> test-coverage-summary.txt
                        echo "--------" >> test-coverage-summary.txt
                        # Generate coverage report filtered to this package (from artifacts dir where .coverage is)
                        uv run coverage report --data-file="$ARTIFACTS_DIR/.coverage" --include="*/${src_name}/*" 2>/dev/null >> test-coverage-summary.txt || echo "  No coverage data for $pkg_name" >> test-coverage-summary.txt
                    fi
                fi
            done
            
            # Generate JSON coverage summary for each package
            echo "{" > coverage-by-package.json
            echo '  "generated": "'$(date -u +"%Y-%m-%dT%H:%M:%SZ")'",' >> coverage-by-package.json
            echo '  "packages": {' >> coverage-by-package.json
            
            first_pkg=true
            for pkg_dir in "$PROJECT_ROOT"/packages/*/; do
                if [ -d "$pkg_dir" ]; then
                    pkg_name=$(basename "$pkg_dir")
                    # Special case: legacy package is named "dataknobs" not "dataknobs_legacy"
                    if [ "$pkg_name" = "legacy" ]; then
                        src_name="dataknobs"
                    else
                        src_name="dataknobs_${pkg_name}"
                    fi
                    if [ -d "$pkg_dir/src/${src_name}" ]; then
                        # Get coverage percentage for this package (from artifacts dir where .coverage is)
                        coverage_output=$(uv run coverage report --data-file="$ARTIFACTS_DIR/.coverage" --include="*/${src_name}/*" 2>/dev/null | tail -1)
                        if echo "$coverage_output" | grep -q "TOTAL"; then
                            # Extract coverage percentage from the TOTAL line
                            coverage_pct=$(echo "$coverage_output" | awk '{print $(NF)}' | sed 's/%//')
                            statements=$(echo "$coverage_output" | awk '{print $2}')
                            missing=$(echo "$coverage_output" | awk '{print $3}')
                            
                            if [ "$first_pkg" = false ]; then
                                echo "," >> coverage-by-package.json
                            fi
                            first_pkg=false
                            
                            echo -n '    "'$pkg_name'": {' >> coverage-by-package.json
                            echo -n '"statements": '$statements', ' >> coverage-by-package.json
                            echo -n '"missing": '$missing', ' >> coverage-by-package.json
                            echo -n '"coverage": "'$coverage_pct'%"}' >> coverage-by-package.json
                        fi
                    fi
                fi
            done
            
            echo "" >> coverage-by-package.json
            echo "  }" >> coverage-by-package.json
            echo "}" >> coverage-by-package.json
            
            print_success "Package coverage summary saved to coverage-by-package.json"
            
        else
            print_warning "Could not combine coverage data, using individual reports"
            # Fall back to XML files if combine fails
            if [ -f "coverage-unit.xml" ] && [ -f "coverage-integration.xml" ]; then
                # Use unit coverage as base (usually has more coverage)
                cp coverage-unit.xml coverage.xml
                print_warning "Using unit test coverage as primary report"
            elif [ -f "coverage-unit.xml" ]; then
                cp coverage-unit.xml coverage.xml
            elif [ -f "coverage-integration.xml" ]; then
                cp coverage-integration.xml coverage.xml
            fi
        fi
        
        # Return to project root
        cd "$PROJECT_ROOT"
    elif ls "$ARTIFACTS_DIR"/coverage*.xml >/dev/null 2>&1; then
        # No .coverage data files but we have XML files
        print_status "Processing coverage XML files..."
        coverage_files=($ARTIFACTS_DIR/coverage-*.xml)
        if [ ${#coverage_files[@]} -gt 1 ]; then
            cp "${coverage_files[0]}" "$ARTIFACTS_DIR/coverage.xml"
            print_warning "Multiple XML files found, using ${coverage_files[0]##*/}"
        elif [ ! -f "$ARTIFACTS_DIR/coverage.xml" ]; then
            # If coverage.xml doesn't exist but other coverage files do
            if [ -f "$ARTIFACTS_DIR/coverage-unit.xml" ]; then
                cp "$ARTIFACTS_DIR/coverage-unit.xml" "$ARTIFACTS_DIR/coverage.xml"
            elif [ -f "$ARTIFACTS_DIR/coverage-integration.xml" ]; then
                cp "$ARTIFACTS_DIR/coverage-integration.xml" "$ARTIFACTS_DIR/coverage.xml"
            fi
        fi
    else
        # Create empty coverage report if no tests ran
        echo '<?xml version="1.0" encoding="utf-8"?><coverage version="1" line-rate="0"><packages></packages></coverage>' > "$ARTIFACTS_DIR/coverage.xml"
    fi
    
    # Generate summary
    print_status "Generating quality summary..."
    OVERALL_STATUS="PASS"

    # Check for failures
    if [ $DOCS_STATUS -ne 0 ] || [ $DOCS_VERSIONS_STATUS -ne 0 ] || [ $TEST_STATUS -ne 0 ]; then
        OVERALL_STATUS="FAIL"
    fi
    
    cat > "$ARTIFACTS_DIR/quality-summary.json" <<EOF
{
  "timestamp": "$TIMESTAMP",
  "overall_status": "$OVERALL_STATUS",
  "environment": "$([ "$IN_DOCKER" = true ] && echo "docker" || echo "host")",
  "packages": "$([ -n "$PACKAGES" ] && echo "$PACKAGES" || echo "all")",
  "checks": {
    "documentation": {
      "status": $([ $DOCS_STATUS -eq 0 ] && echo '"pass"' || echo '"fail"'),
      "exit_code": $DOCS_STATUS,
      "tool": "mkdocs"
    },
    "documentation_versions": {
      "status": $([ $DOCS_VERSIONS_STATUS -eq 0 ] && echo '"pass"' || echo '"fail"'),
      "exit_code": $DOCS_VERSIONS_STATUS,
      "tool": "docs-update-versions.sh"
    },
    "style": {
      "status": $([ $STYLE_STATUS -eq 0 ] && echo '"pass"' || echo '"warning"'),
      "exit_code": $STYLE_STATUS,
      "skipped": $([ "$SKIP_STYLE" = "yes" ] && echo "true" || echo "false"),
      "tool": "ruff"
    },
    "unit_tests": {
      "status": $([ $UNIT_TEST_STATUS -eq 0 ] && echo '"pass"' || echo '"fail"'),
      "exit_code": $UNIT_TEST_STATUS,
      "skipped": $([ "$SKIP_TESTS" = "yes" ] && echo "true" || echo "false")
    },
    "integration_tests": {
      "status": $([ $INTEGRATION_TEST_STATUS -eq 0 ] && echo '"pass"' || echo '"fail"'),
      "exit_code": $INTEGRATION_TEST_STATUS,
      "skipped": $([ "$SKIP_TESTS" = "yes" ] && echo "true" || echo "false")
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
fi

# Print summary
echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}                        Quality Check Summary                     ${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""

if [ "$PR_MODE" = "yes" ]; then
    # Show documentation build status (only in PR mode)
    if [ $DOCS_STATUS -eq 0 ]; then
        echo -e "  Documentation:      ${GREEN}✓ PASSED${NC}"
    else
        echo -e "  Documentation:      ${RED}✗ FAILED${NC}"
    fi

    # Show documentation versions status
    if [ $DOCS_VERSIONS_STATUS -eq 0 ]; then
        echo -e "  Doc Versions:       ${GREEN}✓ PASSED${NC}"
    else
        echo -e "  Doc Versions:       ${RED}✗ FAILED${NC}"
    fi
fi

if [ "$SKIP_STYLE" = "yes" ]; then
    echo -e "  Style Check (ruff): ${CYAN}⊘ SKIPPED${NC}"
elif [ $STYLE_STATUS -eq 0 ]; then
    echo -e "  Style Check (ruff): ${GREEN}✓ PASSED${NC}"
else
    echo -e "  Style Check (ruff): ${YELLOW}⚠ WARNINGS${NC}"
fi

if [ "$PR_MODE" = "yes" ]; then
    # PR mode: Show unit and integration tests separately
    if [ "$SKIP_TESTS" = "yes" ]; then
        echo -e "  Unit Tests:        ${CYAN}⊘ SKIPPED${NC}"
        echo -e "  Integration Tests: ${CYAN}⊘ SKIPPED${NC}"
    else
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
    fi
else
    # Dev mode: Show combined test status
    if [ "$SKIP_TESTS" = "yes" ]; then
        echo -e "  Tests:             ${CYAN}⊘ SKIPPED${NC}"
    elif [ $TEST_STATUS -eq 0 ]; then
        echo -e "  Tests:             ${GREEN}✓ PASSED${NC}"
    else
        echo -e "  Tests:             ${RED}✗ FAILED${NC}"
    fi
fi

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"

# Determine overall status
OVERALL_STATUS="PASS"
if [ $DOCS_STATUS -ne 0 ] || [ $DOCS_VERSIONS_STATUS -ne 0 ] || [ $TEST_STATUS -ne 0 ]; then
    OVERALL_STATUS="FAIL"
fi

if [ "$OVERALL_STATUS" = "PASS" ]; then
    echo ""
    echo -e "${GREEN}✓ All critical checks passed!${NC}"
    if [ "$PR_MODE" = "yes" ]; then
        echo -e "${GREEN}  Artifacts saved to: .quality-artifacts/${NC}"
        echo -e "${GREEN}  You can now create your pull request.${NC}"
    fi
    echo ""
    exit 0
else
    echo ""
    echo -e "${RED}✗ Some checks failed!${NC}"
    echo -e "${RED}  Please fix the issues and run this script again.${NC}"
    
    # Show quick diagnostic info for failures
    echo ""
    echo -e "${YELLOW}Quick Diagnostics:${NC}"
    
    if [ "$PR_MODE" = "yes" ]; then
        # In PR mode, show specific commands to investigate failures
        if [ $DOCS_STATUS -ne 0 ] && [ -f "$ARTIFACTS_DIR/docs-build.log" ]; then
            echo -e "  ${CYAN}Documentation Build Failures:${NC}"
            echo "    View documentation errors:"
            echo "      cat $ARTIFACTS_DIR/docs-build.log"
            echo ""
        fi

        if [ $DOCS_VERSIONS_STATUS -ne 0 ] && [ -f "$ARTIFACTS_DIR/docs-versions.log" ]; then
            echo -e "  ${CYAN}Documentation Version Mismatch:${NC}"
            echo "    View version differences:"
            echo "      cat $ARTIFACTS_DIR/docs-versions.log"
            echo "    To fix:"
            echo "      bin/docs-update-versions.sh"
            echo ""
        fi

        if [ $UNIT_TEST_STATUS -ne 0 ] || [ $INTEGRATION_TEST_STATUS -ne 0 ]; then
            echo -e "  ${CYAN}Test Failures:${NC}"
            
            # Find test output files with failures
            if [ $UNIT_TEST_STATUS -ne 0 ] && [ -f "$ARTIFACTS_DIR/unit-test-output.txt" ]; then
                echo "    View unit test failures:"
                echo "      grep -E '(FAILED|ERROR|AssertionError)' $ARTIFACTS_DIR/unit-test-output.txt"
            fi
            
            if [ $INTEGRATION_TEST_STATUS -ne 0 ] && [ -f "$ARTIFACTS_DIR/integration-test-output.txt" ]; then
                echo "    View integration test failures:"
                echo "      grep -E '(FAILED|ERROR|AssertionError)' $ARTIFACTS_DIR/integration-test-output.txt"
            fi
            
            # Check for individual package test outputs
            for output_file in "$ARTIFACTS_DIR"/*-test-output-*.txt; do
                if [ -f "$output_file" ]; then
                    if grep -q "FAILED" "$output_file" 2>/dev/null; then
                        pkg_name=$(basename "$output_file" | sed 's/.*-test-output-\(.*\)\.txt/\1/')
                        echo "    Package $pkg_name has failures:"
                        echo "      grep -E '(FAILED|ERROR)' $output_file"
                    fi
                fi
            done
        fi
        
        if [ $LINT_STATUS -ne 0 ] && [ -f "$ARTIFACTS_DIR/lint-report.json" ]; then
            echo -e "  ${CYAN}Linting Issues:${NC}"
            echo "    View linting report:"
            echo "      jq -r '.[] | \"\\(.path):\\(.line):\\(.column): \\(.message)\"' $ARTIFACTS_DIR/lint-report.json | head -10"
            echo "    Or with python:"
            echo "      python -m json.tool $ARTIFACTS_DIR/lint-report.json | grep -A2 '\"message\"' | head -20"
        fi
        
        if [ $STYLE_STATUS -ne 0 ] && [ -f "$ARTIFACTS_DIR/style-check.json" ]; then
            echo -e "  ${CYAN}Style Issues:${NC}"
            echo "    View style violations:"
            echo "      jq -r '.[] | \"\\(.filename):\\(.location.row): \\(.message) [\\(.code)]\"' $ARTIFACTS_DIR/style-check.json | head -10"
            echo "    Or with python:"
            echo "      python -m json.tool $ARTIFACTS_DIR/style-check.json | grep -A2 '\"message\"' | head -20"
        fi
        
        echo ""
        echo -e "  ${CYAN}Full artifacts in:${NC} .quality-artifacts/"
        echo -e "  ${CYAN}View summary:${NC} cat .quality-artifacts/quality-summary.json | python -m json.tool"
    else
        # In dev mode, suggest re-running with specific focus
        if [ $TEST_STATUS -ne 0 ]; then
            echo -e "  ${CYAN}To re-run only failed tests:${NC}"
            echo "    $0 $PACKAGES -- --lf    # Run last failed tests"
            echo "    $0 $PACKAGES -- -x      # Stop on first failure"
            echo "    $0 $PACKAGES -- -vvs    # Verbose output with stdout"
        fi
        
        if [ $LINT_STATUS -ne 0 ]; then
            echo -e "  ${CYAN}To see linting details:${NC}"
            echo "    uv run pylint $PACKAGE_PATTERN --rcfile=.pylintrc"
        fi
        
        if [ $STYLE_STATUS -ne 0 ]; then
            echo -e "  ${CYAN}To see style issues:${NC}"
            echo "    uv run ruff check $PACKAGE_PATTERN"
            echo -e "  ${CYAN}To auto-fix style issues:${NC}"
            echo "    uv run ruff check --fix $PACKAGE_PATTERN"
        fi
    fi
    
    echo ""
    exit 1
fi

