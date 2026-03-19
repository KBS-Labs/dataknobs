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
RUN_MODE=""     # pr, all, full (set after argument parsing)
BASE_REF="main" # Git ref for changed-package detection

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
    --pr                    PR mode (default): Only test changed packages + dependents.
                            Uses parallel execution, quiet output, XML-only coverage.
                            Skips docs build if no docs changed.
    --all                   All mode: Test all packages with parallel execution
                            and optimized coverage (no HTML reports)
    --full                  Full mode: Legacy behavior — all packages, sequential,
                            verbose output, all coverage reports (HTML + XML + term)
    --dev                   Dev mode: Run quick checks without artifacts
                            (combined tests, no artifact pollution)
    --base-ref REF          Git ref for change detection (default: main)
    --skip-style            Skip style checks (ruff)
    --skip-tests            Skip test execution
    --keep-services         Keep services running after completion
    -h, --help              Show this help message

${YELLOW}Advanced Usage:${NC}
    Any arguments after -- are passed directly to pytest:
    $0 data -- -xvs --tb=short

${YELLOW}Examples:${NC}
    $0 --pr                 # PR mode: Only changed packages (default)
    $0 --all                # All packages, parallel, optimized
    $0 --full               # Legacy: all packages, sequential, verbose
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
            RUN_MODE="pr"
            shift
            ;;
        --all)
            PR_MODE="yes"
            RUN_MODE="all"
            shift
            ;;
        --full)
            PR_MODE="yes"
            RUN_MODE="full"
            shift
            ;;
        --base-ref)
            BASE_REF="$2"
            shift 2
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

# Default RUN_MODE based on PR_MODE
if [ -z "$RUN_MODE" ]; then
    if [ "$PR_MODE" = "yes" ]; then
        RUN_MODE="pr"
    else
        RUN_MODE="dev"
    fi
fi

# Changed-package detection (pr mode only, when no explicit packages given)
DOCS_CHANGED="true"
TESTED_PACKAGES_JSON="[]"
if [ "$RUN_MODE" = "pr" ] && [ -z "$PACKAGES" ]; then
    print_status "Detecting changed packages..."
    CHANGED_INFO=$(uv run python "$SCRIPT_DIR/changed-packages.py" --base-ref "$BASE_REF" 2>/dev/null) || {
        print_warning "Change detection failed — testing all packages"
        CHANGED_INFO=""
    }

    if [ -n "$CHANGED_INFO" ]; then
        CHANGED_PACKAGES=$(echo "$CHANGED_INFO" | python3 -c "import sys, json; d=json.load(sys.stdin); print(' '.join(d['packages']))" 2>/dev/null || echo "")
        DOCS_CHANGED=$(echo "$CHANGED_INFO" | python3 -c "import sys, json; d=json.load(sys.stdin); print(str(d['docs_changed']).lower())" 2>/dev/null || echo "true")
        CHANGE_MODE=$(echo "$CHANGED_INFO" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('mode', 'all'))" 2>/dev/null || echo "all")

        if [ -n "$CHANGED_PACKAGES" ]; then
            PACKAGES="$CHANGED_PACKAGES"
            TESTED_PACKAGES_JSON=$(echo "$CHANGED_INFO" | python3 -c "import sys, json; d=json.load(sys.stdin); print(json.dumps(d['packages']))" 2>/dev/null || echo "[]")
            print_success "Changed packages: $PACKAGES"
            if [ "$CHANGE_MODE" = "all" ]; then
                print_status "Global files changed — testing all packages"
            fi
        else
            print_success "No package changes detected — skipping tests"
            SKIP_TESTS="yes"
        fi
    fi

    if [ "$DOCS_CHANGED" = "true" ]; then
        print_status "Documentation changes detected"
    else
        print_status "No documentation changes detected — skipping docs build"
    fi
fi

# Determine test runner flags based on mode
TEST_PARALLEL_FLAG=""
TEST_VERBOSITY_FLAG=""
TEST_COV_REPORT="xml"
case "$RUN_MODE" in
    pr)
        TEST_PARALLEL_FLAG="--parallel"
        TEST_VERBOSITY_FLAG="--quiet"
        TEST_COV_REPORT="xml"
        ;;
    all)
        TEST_PARALLEL_FLAG="--parallel"
        TEST_VERBOSITY_FLAG=""
        TEST_COV_REPORT="xml"
        ;;
    full)
        TEST_PARALLEL_FLAG="--no-parallel"
        TEST_VERBOSITY_FLAG="--verbose"
        TEST_COV_REPORT="term-missing,html,xml"
        ;;
    dev)
        TEST_PARALLEL_FLAG=""
        TEST_VERBOSITY_FLAG=""
        TEST_COV_REPORT="term-missing"
        ;;
esac

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
case "$RUN_MODE" in
    pr)   echo -e "${BLUE}       DataKnobs Quality Checks - PR Mode (changed packages)     ${NC}" ;;
    all)  echo -e "${BLUE}       DataKnobs Quality Checks - All Packages                   ${NC}" ;;
    full) echo -e "${BLUE}       DataKnobs Quality Checks - Full Mode (legacy)             ${NC}" ;;
    *)    echo -e "${BLUE}       DataKnobs Quality Checks - Developer Mode                 ${NC}" ;;
esac
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""

# Display environment
if [ "$IN_DOCKER" = true ]; then
    print_status "Environment: Docker Container"
else
    print_status "Environment: Host System"
fi

# Display mode and packages
case "$RUN_MODE" in
    pr)   print_status "Mode: PR (changed packages, parallel, quiet)" ;;
    all)  print_status "Mode: All (all packages, parallel)" ;;
    full) print_status "Mode: Full (all packages, sequential, verbose)" ;;
    *)    print_status "Mode: Developer (quick checks, no artifacts)" ;;
esac

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
VALIDATION_STATUS=0
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

# Build documentation (PR mode only, skip if no docs changes in pr mode)
if [ "$PR_MODE" = "yes" ]; then
    if [ "$DOCS_CHANGED" = "true" ] || [ "$RUN_MODE" != "pr" ]; then
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
    else
        print_status "Skipping docs build (no documentation changes detected)"
    fi
fi

# Run code validation (syntax, ruff, imports, mypy, print statements)
if [ "$SKIP_STYLE" != "yes" ]; then
    print_status "Running code validation (syntax, ruff, imports, mypy, print statements)..."

    # Build package args for validate.sh
    VALIDATE_ARGS=""
    if [ -n "$PACKAGES" ]; then
        VALIDATE_ARGS="$PACKAGES"
    fi

    # Skip if no packages to validate in PR mode (e.g., only docs changed)
    if [ -n "$VALIDATE_ARGS" ] || [ "$RUN_MODE" != "pr" ]; then
        if [ "$PR_MODE" = "yes" ]; then
            # Also generate ruff JSON artifact for diagnostics
            uv run ruff check $PACKAGE_PATTERN --output-format=json --config "$PROJECT_ROOT/pyproject.toml" > "$ARTIFACTS_DIR/style-check.json" 2>&1 || true

            if "$SCRIPT_DIR/validate.sh" $VALIDATE_ARGS > "$ARTIFACTS_DIR/validation.log" 2>&1; then
                print_success "Code validation passed"
            else
                VALIDATION_STATUS=$?
                print_error "Code validation failed - see $ARTIFACTS_DIR/validation.log"
                cat "$ARTIFACTS_DIR/validation.log"
            fi
        else
            # Dev mode: show output directly
            if "$SCRIPT_DIR/validate.sh" $VALIDATE_ARGS; then
                print_success "Code validation passed"
            else
                VALIDATION_STATUS=$?
                print_error "Code validation failed"
            fi
        fi
    else
        print_status "Skipping code validation (no package changes)"
    fi
else
    print_status "Skipping code validation"
fi

# Run tests using the test.sh script
if [ "$SKIP_TESTS" != "yes" ]; then
    print_status "Running tests..."
    
    # Build test command (skip service management — already handled above)
    TEST_CMD="$SCRIPT_DIR/test.sh -n"
    
    if [ "$PR_MODE" = "yes" ]; then
        # PR/All/Full mode: Run unit and integration tests separately with artifacts

        # Build common test flags
        TEST_FLAGS="$TEST_PARALLEL_FLAG $TEST_VERBOSITY_FLAG"

        # Always capture failure and skip reasons in output files for the summary
        # -rf = show FAILED lines, -rs = show SKIPPED reasons
        if [ -z "$PYTEST_ARGS" ]; then
            PYTEST_ARGS="-rfs"
        else
            PYTEST_ARGS="-rfs $PYTEST_ARGS"
        fi

        # Helper: run unit tests for a single package, saving artifacts.
        # Uses per-package coverage filenames to avoid race conditions in concurrent mode.
        # Args: $1=package name, $2=cov_report_type (optional override)
        run_unit_for_pkg() {
            local pkg=$1
            local cov_report="${2:-$TEST_COV_REPORT}"
            print_status "Running unit tests for $pkg..."
            local test_exit=0

            # Set unique coverage file to avoid collisions in concurrent mode
            export COVERAGE_FILE="$PROJECT_ROOT/.coverage.unit.$pkg"

            if [ -n "$PYTEST_ARGS" ]; then
                $TEST_CMD "$pkg" -t unit --cov-report "$cov_report" $TEST_FLAGS -- $PYTEST_ARGS > "$ARTIFACTS_DIR/unit-test-output-$pkg.txt" 2>&1 || test_exit=$?
            else
                $TEST_CMD "$pkg" -t unit --cov-report "$cov_report" $TEST_FLAGS > "$ARTIFACTS_DIR/unit-test-output-$pkg.txt" 2>&1 || test_exit=$?
            fi

            # Save coverage artifacts
            if [ -f "coverage.xml" ]; then
                mv coverage.xml "$ARTIFACTS_DIR/coverage-unit-$pkg.xml"
            fi
            if [ -f "$PROJECT_ROOT/.coverage.unit.$pkg" ]; then
                mv "$PROJECT_ROOT/.coverage.unit.$pkg" "$ARTIFACTS_DIR/.coverage.unit.$pkg"
            elif [ -f ".coverage" ]; then
                mv .coverage "$ARTIFACTS_DIR/.coverage.unit.$pkg"
            fi

            # Unset to avoid leaking
            unset COVERAGE_FILE

            if [ $test_exit -ne 0 ] && [ $test_exit -ne 5 ]; then
                print_error "Unit tests failed for $pkg"
                # Show failed test names inline (strip ANSI codes first)
                if [ -f "$ARTIFACTS_DIR/unit-test-output-$pkg.txt" ]; then
                    sed 's/\x1b\[[0-9;]*m//g' "$ARTIFACTS_DIR/unit-test-output-$pkg.txt" 2>/dev/null | \
                        grep -E '^FAILED ' | sed 's/^/    /' || true
                fi
                return $test_exit
            else
                print_success "Unit tests passed for $pkg"
                return 0
            fi
        }

        # Determine packages to test
        PACKAGES_TO_TEST=""
        if [ -n "$PACKAGES" ]; then
            PACKAGES_TO_TEST="$PACKAGES"
        else
            for pkg_dir in "$PROJECT_ROOT"/packages/*/; do
                if [ -d "$pkg_dir" ]; then
                    pkg_name=$(basename "$pkg_dir")
                    if [ -d "$pkg_dir/tests" ]; then
                        PACKAGES_TO_TEST="$PACKAGES_TO_TEST $pkg_name"
                    fi
                fi
            done
        fi

        # Run unit tests
        print_status "Running unit tests..."
        if [ "$RUN_MODE" = "full" ]; then
            # Full mode: sequential execution
            for pkg in $PACKAGES_TO_TEST; do
                if [ -d "packages/$pkg" ]; then
                    run_unit_for_pkg "$pkg" || UNIT_TEST_STATUS=$?
                fi
            done
        else
            # PR/All mode: concurrent execution of independent packages
            # Each package gets its own COVERAGE_FILE and output file to avoid races.
            # Skip per-package XML reports to avoid write collisions — XML is generated
            # from combined .coverage data in the coverage combining phase.
            # Limit concurrency to avoid CPU contention breaking timing-sensitive tests.
            MAX_CONCURRENT=3
            unit_pids=()
            unit_pkgs=()
            for pkg in $PACKAGES_TO_TEST; do
                if [ -d "packages/$pkg" ]; then
                    run_unit_for_pkg "$pkg" "none" &
                    unit_pids+=($!)
                    unit_pkgs+=("$pkg")

                    # Throttle: when we hit MAX_CONCURRENT, wait for one to finish
                    if [ ${#unit_pids[@]} -ge $MAX_CONCURRENT ]; then
                        # Wait for the oldest job
                        wait "${unit_pids[0]}" 2>/dev/null || {
                            UNIT_TEST_STATUS=1
                        }
                        unit_pids=("${unit_pids[@]:1}")
                        unit_pkgs=("${unit_pkgs[@]:1}")
                    fi
                fi
            done

            # Wait for remaining jobs
            for idx in "${!unit_pids[@]}"; do
                wait "${unit_pids[$idx]}" 2>/dev/null || {
                    UNIT_TEST_STATUS=1
                }
            done
        fi

        # Run integration tests (always sequential — shared external services)
        print_status "Running integration tests..."
        for pkg in $PACKAGES_TO_TEST; do
            if [ -d "packages/$pkg" ] && [ -d "packages/$pkg/tests/integration" ]; then
                print_status "Running integration tests for $pkg..."
                local_exit=0

                # Use per-package coverage file
                export COVERAGE_FILE="$PROJECT_ROOT/.coverage.integration.$pkg"

                if [ -n "$PYTEST_ARGS" ]; then
                    $TEST_CMD "$pkg" -t integration --cov-report "$TEST_COV_REPORT" $TEST_FLAGS -- $PYTEST_ARGS > "$ARTIFACTS_DIR/integration-test-output-$pkg.txt" 2>&1 || local_exit=$?
                else
                    $TEST_CMD "$pkg" -t integration --cov-report "$TEST_COV_REPORT" $TEST_FLAGS > "$ARTIFACTS_DIR/integration-test-output-$pkg.txt" 2>&1 || local_exit=$?
                fi

                unset COVERAGE_FILE

                if [ $local_exit -ne 0 ] && [ $local_exit -ne 5 ]; then
                    INTEGRATION_TEST_STATUS=$local_exit
                    print_error "Integration tests failed for $pkg"
                    # Show failed test names inline (strip ANSI codes first)
                    if [ -f "$ARTIFACTS_DIR/integration-test-output-$pkg.txt" ]; then
                        sed 's/\x1b\[[0-9;]*m//g' "$ARTIFACTS_DIR/integration-test-output-$pkg.txt" 2>/dev/null | \
                            grep -E '^FAILED ' | sed 's/^/    /' || true
                    fi
                else
                    print_success "Integration tests passed for $pkg"
                fi

                if [ -f "coverage.xml" ]; then
                    mv coverage.xml "$ARTIFACTS_DIR/coverage-integration-$pkg.xml"
                fi
                if [ -f "$PROJECT_ROOT/.coverage.integration.$pkg" ]; then
                    mv "$PROJECT_ROOT/.coverage.integration.$pkg" "$ARTIFACTS_DIR/.coverage.integration.$pkg"
                elif [ -f ".coverage" ]; then
                    mv .coverage "$ARTIFACTS_DIR/.coverage.integration.$pkg"
                fi
            fi
        done

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

        # Surface test failure details from output files
        if [ $TEST_STATUS -ne 0 ]; then
            echo ""
            echo -e "${RED}── Test Failure Details ──${NC}"

            # Collect all FAILED lines from unit and integration output files
            # Strip ANSI codes before matching since pytest uses colored output
            for output_file in "$ARTIFACTS_DIR"/unit-test-output-*.txt "$ARTIFACTS_DIR"/integration-test-output-*.txt; do
                if [ -f "$output_file" ]; then
                    failed_lines=$(sed 's/\x1b\[[0-9;]*m//g' "$output_file" 2>/dev/null | grep -E '^FAILED ' || true)
                    if [ -n "$failed_lines" ]; then
                        pkg_label=$(basename "$output_file" .txt | sed 's/.*-output-//')
                        test_type=$(basename "$output_file" .txt | sed 's/-test-output-.*//')
                        echo -e "  ${YELLOW}$test_type ($pkg_label):${NC}"
                        echo "$failed_lines" | sed 's/^/    /'
                    fi
                fi
            done

            echo -e "${RED}──────────────────────────${NC}"
            echo ""
        fi

        # Surface skip summary from output files (unique reasons with counts)
        skip_summary=""
        for output_file in "$ARTIFACTS_DIR"/unit-test-output-*.txt "$ARTIFACTS_DIR"/integration-test-output-*.txt; do
            if [ -f "$output_file" ]; then
                skips=$(sed 's/\x1b\[[0-9;]*m//g' "$output_file" 2>/dev/null | grep -E '^SKIPPED ' || true)
                if [ -n "$skips" ]; then
                    pkg_label=$(basename "$output_file" .txt | sed 's/.*-output-//')
                    test_type=$(basename "$output_file" .txt | sed 's/-test-output-.*//')
                    # Extract unique skip reasons (part after the last ": ")
                    unique_reasons=$(echo "$skips" | sed 's/.*: //' | sort | uniq -c | sort -rn)
                    if [ -n "$unique_reasons" ]; then
                        skip_summary="${skip_summary}\n  ${YELLOW}${test_type} (${pkg_label}):${NC}\n$(echo "$unique_reasons" | sed 's/^ */    /')\n"
                    fi
                fi
            fi
        done

        if [ -n "$skip_summary" ]; then
            echo ""
            echo -e "${YELLOW}── Skipped Tests ──${NC}"
            echo -e "$skip_summary"
            echo -e "${YELLOW}───────────────────${NC}"
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
            
            # Generate combined HTML report (full mode only — slow)
            if [ "$RUN_MODE" = "full" ]; then
                if uv run coverage html -d htmlcov 2>/dev/null; then
                    print_success "Combined coverage HTML generated in .quality-artifacts/htmlcov/"
                fi
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
    if [ $VALIDATION_STATUS -ne 0 ] || [ $DOCS_STATUS -ne 0 ] || [ $DOCS_VERSIONS_STATUS -ne 0 ] || [ $TEST_STATUS -ne 0 ]; then
        OVERALL_STATUS="FAIL"
    fi
    
    cat > "$ARTIFACTS_DIR/quality-summary.json" <<EOF
{
  "timestamp": "$TIMESTAMP",
  "overall_status": "$OVERALL_STATUS",
  "run_mode": "$RUN_MODE",
  "environment": "$([ "$IN_DOCKER" = true ] && echo "docker" || echo "host")",
  "packages": "$([ -n "$PACKAGES" ] && echo "$PACKAGES" || echo "all")",
  "tested_packages": $TESTED_PACKAGES_JSON,
  "checks": {
    "documentation": {
      "status": $([ $DOCS_STATUS -eq 0 ] && echo '"pass"' || echo '"fail"'),
      "exit_code": $DOCS_STATUS,
      "skipped": $([ "$DOCS_CHANGED" = "true" ] || [ "$RUN_MODE" != "pr" ] && echo "false" || echo "true"),
      "tool": "mkdocs"
    },
    "documentation_versions": {
      "status": $([ $DOCS_VERSIONS_STATUS -eq 0 ] && echo '"pass"' || echo '"fail"'),
      "exit_code": $DOCS_VERSIONS_STATUS,
      "tool": "docs-update-versions.sh"
    },
    "validation": {
      "status": $([ $VALIDATION_STATUS -eq 0 ] && echo '"pass"' || echo '"fail"'),
      "exit_code": $VALIDATION_STATUS,
      "skipped": $([ "$SKIP_STYLE" = "yes" ] && echo "true" || echo "false"),
      "tool": "validate.sh"
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
    if command -v sha256sum >/dev/null 2>&1; then
        find . -type f \( -name "*.json" -o -name "*.xml" \) | sort | xargs sha256sum > signature.sha256
    else
        find . -type f \( -name "*.json" -o -name "*.xml" \) | sort | xargs shasum -a 256 > signature.sha256
    fi
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
    echo -e "  Code Validation:    ${CYAN}⊘ SKIPPED${NC}"
elif [ $VALIDATION_STATUS -eq 0 ]; then
    echo -e "  Code Validation:    ${GREEN}✓ PASSED${NC}"
else
    echo -e "  Code Validation:    ${RED}✗ FAILED${NC}"
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
        
        if [ $VALIDATION_STATUS -ne 0 ] && [ -f "$ARTIFACTS_DIR/validation.log" ]; then
            echo -e "  ${CYAN}Code Validation Failures:${NC}"
            echo "    View full validation output:"
            echo "      cat $ARTIFACTS_DIR/validation.log"
            echo "    To auto-fix what's possible:"
            echo "      bin/validate.sh -f"
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
        
        if [ $VALIDATION_STATUS -ne 0 ]; then
            echo -e "  ${CYAN}To auto-fix validation issues:${NC}"
            echo "    bin/validate.sh -f"
        fi
    fi
    
    echo ""
    exit 1
fi

