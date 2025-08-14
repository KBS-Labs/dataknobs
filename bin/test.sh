#!/bin/bash
# Enhanced test runner with flexible pytest options

set -e

# Get the script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Colors for output (check if we're in a terminal that supports colors)
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

# Default values
TEST_TYPE="both"
PACKAGE=""
START_SERVICES="auto"
COVERAGE="yes"
PYTEST_ARGS=""
CUSTOM_PYTEST_ARGS=""
COV_REPORT="term-missing"
VERBOSE_LEVEL=""
TB_STYLE=""
MARKERS=""

# Check if we're running in a Docker container
IN_DOCKER=false
if [ -f /.dockerenv ] || [ -n "${DOCKER_CONTAINER:-}" ]; then
    IN_DOCKER=true
fi

# Function to show usage
show_usage() {
    cat << EOF
${CYAN}DataKnobs Test Runner${NC}

Usage: $0 [OPTIONS] [PACKAGE] [-- PYTEST_ARGS]

Run unit and/or integration tests for DataKnobs packages with flexible pytest options.

${YELLOW}Options:${NC}
    -t, --type TYPE          Test type: unit, integration, or both (default: both)
    -p, --package PACKAGE    Package to test (e.g., data, config, structures)
                            If not specified, tests all packages
    -s, --services          Start services for integration tests (auto by default)
    -n, --no-services       Don't start services (assume they're already running)
    --no-cov                Disable coverage reporting
    --cov-report TYPE       Coverage report type: term, term-missing, html, xml, or combinations
                           (default: term-missing, use comma to combine: term-missing,html,xml)
    
${YELLOW}Pytest Options:${NC}
    -v, -vv, -vvv           Verbosity level (can stack for more verbosity)
    -x                      Exit on first failure (--exitfirst)
    -s                      No capture, show print statements (--capture=no)
    --tb=STYLE              Traceback style: auto, short, line, no, native, long
    -k EXPRESSION           Only run tests matching the expression
    -m MARKERS              Only run tests with specified markers
    --lf                    Rerun only failures from last run
    --ff                    Run failures first, then other tests
    --pdb                   Drop into debugger on failures
    --pdbcls                Drop into debugger at start of test
    --maxfail=N             Stop after N failures
    
    -h, --help              Show this help message

${YELLOW}Advanced Usage:${NC}
    Any arguments after -- are passed directly to pytest:
    $0 data -- -xvs --tb=short --pdb
    
${YELLOW}Examples:${NC}
    $0                                    # Run all tests with default settings
    $0 data                               # Test data package
    $0 -t unit data                       # Unit tests only for data package
    $0 data -xvs                          # Exit on first failure, verbose, no capture
    $0 data -vv --tb=short                # Very verbose with short tracebacks
    $0 data --tb=no                       # No tracebacks
    $0 data -k "test_s3"                  # Run only tests matching "test_s3"
    $0 data -m "slow"                     # Run only tests marked as slow
    $0 data --lf                          # Rerun only last failures
    $0 data -- --pdb --maxfail=3          # Custom pytest args
    $0 -n data                            # Run without starting services (Docker)

${YELLOW}Docker/Container Notes:${NC}
    - Services are automatically detected when running in Docker
    - Use -n/--no-services if services are already running
    - Coverage reports are saved to project root for persistence

EOF
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--type)
            TEST_TYPE="$2"
            shift 2
            ;;
        -p|--package)
            PACKAGE="$2"
            shift 2
            ;;
        -s|--services)
            START_SERVICES="yes"
            shift
            ;;
        -n|--no-services)
            START_SERVICES="no"
            shift
            ;;
        --no-cov)
            COVERAGE="no"
            shift
            ;;
        --cov-report)
            COV_REPORT="$2"
            shift 2
            ;;
        # Pytest verbosity options
        -vvv)
            VERBOSE_LEVEL="-vvv"
            shift
            ;;
        -vv)
            VERBOSE_LEVEL="-vv"
            shift
            ;;
        -v|--verbose)
            VERBOSE_LEVEL="-v"
            shift
            ;;
        # Pytest capture options
        -s)
            PYTEST_ARGS="$PYTEST_ARGS --capture=no"
            shift
            ;;
        # Pytest failure options
        -x|--exitfirst)
            PYTEST_ARGS="$PYTEST_ARGS --exitfirst"
            shift
            ;;
        --maxfail)
            PYTEST_ARGS="$PYTEST_ARGS --maxfail=$2"
            shift 2
            ;;
        # Pytest traceback options
        --tb=*)
            TB_STYLE="${1#--tb=}"
            PYTEST_ARGS="$PYTEST_ARGS --tb=$TB_STYLE"
            shift
            ;;
        --tb)
            PYTEST_ARGS="$PYTEST_ARGS --tb=$2"
            shift 2
            ;;
        # Pytest selection options
        -k)
            PYTEST_ARGS="$PYTEST_ARGS -k '$2'"
            shift 2
            ;;
        -m)
            MARKERS="$2"
            PYTEST_ARGS="$PYTEST_ARGS -m '$2'"
            shift 2
            ;;
        # Pytest failure rerun options
        --lf|--last-failed)
            PYTEST_ARGS="$PYTEST_ARGS --lf"
            shift
            ;;
        --ff|--failed-first)
            PYTEST_ARGS="$PYTEST_ARGS --ff"
            shift
            ;;
        # Pytest debugging options
        --pdb)
            PYTEST_ARGS="$PYTEST_ARGS --pdb"
            shift
            ;;
        --pdbcls)
            PYTEST_ARGS="$PYTEST_ARGS --pdbcls"
            shift
            ;;
        # Help
        -h|--help)
            show_usage
            ;;
        # Separator for custom pytest args
        --)
            shift
            CUSTOM_PYTEST_ARGS="$@"
            break
            ;;
        *)
            # Assume it's a package name if no package specified yet
            if [ -z "$PACKAGE" ]; then
                PACKAGE="$1"
            else
                echo -e "${RED}Unknown option: $1${NC}"
                echo "Use -- to pass custom arguments to pytest"
                show_usage
            fi
            shift
            ;;
    esac
done

# Add verbose level to pytest args
if [ -n "$VERBOSE_LEVEL" ]; then
    PYTEST_ARGS="$VERBOSE_LEVEL $PYTEST_ARGS"
fi

# Append any custom pytest args
if [ -n "$CUSTOM_PYTEST_ARGS" ]; then
    PYTEST_ARGS="$PYTEST_ARGS $CUSTOM_PYTEST_ARGS"
fi

# Validate test type
if [[ "$TEST_TYPE" != "unit" && "$TEST_TYPE" != "integration" && "$TEST_TYPE" != "both" ]]; then
    echo -e "${RED}Invalid test type: $TEST_TYPE${NC}"
    echo "Must be one of: unit, integration, both"
    exit 1
fi

# Function to discover packages with tests
discover_test_packages() {
    local test_type=$1
    local packages=()
    
    for pkg_dir in "$ROOT_DIR"/packages/*/; do
        if [ -d "$pkg_dir" ]; then
            pkg_name=$(basename "$pkg_dir")
            
            # Skip legacy package for tests
            if [ "$pkg_name" = "legacy" ]; then
                continue
            fi
            
            # Check for test directories based on type
            if [ "$test_type" = "unit" ] || [ "$test_type" = "both" ]; then
                if [ -d "$pkg_dir/tests" ] && [ "$(find "$pkg_dir/tests" -name "test_*.py" -type f 2>/dev/null | head -1)" ]; then
                    packages+=("$pkg_name")
                    continue
                fi
            fi
            
            if [ "$test_type" = "integration" ] || [ "$test_type" = "both" ]; then
                if [ -d "$pkg_dir/tests/integration" ]; then
                    packages+=("$pkg_name")
                    continue
                fi
            fi
        fi
    done
    
    echo "${packages[@]}"
}

# Function to run unit tests
run_unit_tests() {
    local package=$1
    echo -e "${YELLOW}Running unit tests for package: $package${NC}"
    
    local test_path="packages/$package/tests"
    
    # Exclude integration tests if they're in a subdirectory
    local exclude_args=""
    if [ -d "packages/$package/tests/integration" ]; then
        exclude_args="--ignore=packages/$package/tests/integration"
    fi
    
    # Build coverage args if enabled
    local cov_args=""
    if [ "$COVERAGE" = "yes" ]; then
        cov_args="--cov=packages/$package/src/dataknobs_${package}"
        # Add coverage report types
        IFS=',' read -ra REPORT_TYPES <<< "$COV_REPORT"
        for report_type in "${REPORT_TYPES[@]}"; do
            case $report_type in
                term|term-missing)
                    cov_args="$cov_args --cov-report=$report_type"
                    ;;
                html)
                    cov_args="$cov_args --cov-report=html:htmlcov"
                    ;;
                xml)
                    cov_args="$cov_args --cov-report=xml:coverage.xml"
                    ;;
            esac
        done
    fi
    
    # Run tests
    local cmd="pytest $test_path $exclude_args $cov_args $PYTEST_ARGS --color=yes"
    echo -e "${CYAN}Command: $cmd${NC}"
    
    if command -v uv &> /dev/null; then
        eval "uv run $cmd" || return $?
    else
        eval "$cmd" || return $?
    fi
}

# Function to run integration tests
run_integration_tests() {
    local package=$1
    echo -e "${YELLOW}Running integration tests for package: $package${NC}"
    
    local test_path="packages/$package/tests/integration"
    
    # Check if integration tests exist
    if [ ! -d "$test_path" ]; then
        echo -e "${BLUE}No integration tests found for package: $package${NC}"
        return 0
    fi
    
    # Start services if needed (and not in Docker with services already running)
    if [ "$START_SERVICES" = "auto" ] || [ "$START_SERVICES" = "yes" ]; then
        if [ "$IN_DOCKER" = true ]; then
            echo -e "${BLUE}Running in Docker container, checking for services...${NC}"
            # In Docker, services might be running in the same container or linked containers
            if [ "$START_SERVICES" = "auto" ]; then
                # Check if we can connect to services
                if nc -z localhost 5432 2>/dev/null && nc -z localhost 9200 2>/dev/null; then
                    echo -e "${BLUE}Services detected, skipping startup${NC}"
                    START_SERVICES="no"
                else
                    echo -e "${YELLOW}Services not detected, will attempt to start${NC}"
                    START_SERVICES="yes"
                fi
            fi
        else
            if [ "$START_SERVICES" = "auto" ]; then
                # Check if services are already running
                if docker ps 2>/dev/null | grep -q "dataknobs.*postgres\|dataknobs.*elasticsearch\|dataknobs.*localstack"; then
                    echo -e "${BLUE}Services appear to be running, skipping startup${NC}"
                else
                    START_SERVICES="yes"
                fi
            fi
        fi
        
        if [ "$START_SERVICES" = "yes" ] && [ -f "$SCRIPT_DIR/run-integration-tests.sh" ]; then
            echo -e "${YELLOW}Starting services for integration tests...${NC}"
            "$SCRIPT_DIR/run-integration-tests.sh" --services-only || {
                echo -e "${RED}Failed to start services${NC}"
                return 1
            }
        fi
    fi
    
    # Set environment variables for tests
    export POSTGRES_HOST="${POSTGRES_HOST:-localhost}"
    export POSTGRES_PORT="${POSTGRES_PORT:-5432}"
    export POSTGRES_USER="${POSTGRES_USER:-postgres}"
    export POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-postgres}"
    export POSTGRES_DB="${POSTGRES_DB:-dataknobs_test}"
    export ELASTICSEARCH_HOST="${ELASTICSEARCH_HOST:-localhost}"
    export ELASTICSEARCH_PORT="${ELASTICSEARCH_PORT:-9200}"
    export AWS_ENDPOINT_URL="${AWS_ENDPOINT_URL:-http://localhost:4566}"
    export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-test}"
    export AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-test}"
    export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"
    export LOCALSTACK_ENDPOINT="${LOCALSTACK_ENDPOINT:-http://localhost:4566}"
    
    # Build coverage args if enabled
    local cov_args=""
    if [ "$COVERAGE" = "yes" ]; then
        cov_args="--cov=packages/$package/src/dataknobs_${package} --cov-append"
        # Add coverage report types
        IFS=',' read -ra REPORT_TYPES <<< "$COV_REPORT"
        for report_type in "${REPORT_TYPES[@]}"; do
            case $report_type in
                term|term-missing)
                    cov_args="$cov_args --cov-report=$report_type"
                    ;;
                html)
                    cov_args="$cov_args --cov-report=html:htmlcov"
                    ;;
                xml)
                    cov_args="$cov_args --cov-report=xml:coverage.xml"
                    ;;
            esac
        done
    fi
    
    # Run tests
    local cmd="pytest $test_path $cov_args $PYTEST_ARGS --color=yes"
    echo -e "${CYAN}Command: $cmd${NC}"
    
    if command -v uv &> /dev/null; then
        eval "uv run $cmd" || return $?
    else
        eval "$cmd" || return $?
    fi
}

# Function to run combined tests with coverage
run_combined_tests() {
    local package=$1
    echo -e "${YELLOW}Running all tests for package: $package${NC}"
    
    local test_path="packages/$package/tests"
    
    # Check if we need to start services for integration tests
    if [ -d "$test_path/integration" ]; then
        # Start services if needed
        if [ "$START_SERVICES" = "auto" ] || [ "$START_SERVICES" = "yes" ]; then
            if [ "$IN_DOCKER" = true ]; then
                echo -e "${BLUE}Running in Docker container, checking for services...${NC}"
                if [ "$START_SERVICES" = "auto" ]; then
                    if nc -z localhost 5432 2>/dev/null && nc -z localhost 9200 2>/dev/null; then
                        echo -e "${BLUE}Services detected, skipping startup${NC}"
                    else
                        echo -e "${YELLOW}Services not detected in container${NC}"
                    fi
                fi
            else
                if [ "$START_SERVICES" = "auto" ]; then
                    if docker ps 2>/dev/null | grep -q "dataknobs.*postgres\|dataknobs.*elasticsearch\|dataknobs.*localstack"; then
                        echo -e "${BLUE}Services appear to be running, skipping startup${NC}"
                    else
                        START_SERVICES="yes"
                    fi
                fi
                
                if [ "$START_SERVICES" = "yes" ] && [ -f "$SCRIPT_DIR/run-integration-tests.sh" ]; then
                    echo -e "${YELLOW}Starting services for integration tests...${NC}"
                    "$SCRIPT_DIR/run-integration-tests.sh" --services-only || {
                        echo -e "${RED}Failed to start services${NC}"
                        echo -e "${YELLOW}Integration tests may fail without services${NC}"
                    }
                fi
            fi
        fi
        
        # Set environment variables for integration tests
        export POSTGRES_HOST="${POSTGRES_HOST:-localhost}"
        export POSTGRES_PORT="${POSTGRES_PORT:-5432}"
        export POSTGRES_USER="${POSTGRES_USER:-postgres}"
        export POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-postgres}"
        export POSTGRES_DB="${POSTGRES_DB:-dataknobs_test}"
        export ELASTICSEARCH_HOST="${ELASTICSEARCH_HOST:-localhost}"
        export ELASTICSEARCH_PORT="${ELASTICSEARCH_PORT:-9200}"
        export AWS_ENDPOINT_URL="${AWS_ENDPOINT_URL:-http://localhost:4566}"
        export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-test}"
        export AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-test}"
        export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"
        export LOCALSTACK_ENDPOINT="${LOCALSTACK_ENDPOINT:-http://localhost:4566}"
    fi
    
    # Build coverage args if enabled
    local cov_args=""
    if [ "$COVERAGE" = "yes" ]; then
        cov_args="--cov=packages/$package/src/dataknobs_${package}"
        # Add coverage report types
        IFS=',' read -ra REPORT_TYPES <<< "$COV_REPORT"
        for report_type in "${REPORT_TYPES[@]}"; do
            case $report_type in
                term|term-missing)
                    cov_args="$cov_args --cov-report=$report_type"
                    ;;
                html)
                    cov_args="$cov_args --cov-report=html:htmlcov"
                    ;;
                xml)
                    cov_args="$cov_args --cov-report=xml:coverage.xml"
                    ;;
            esac
        done
    fi
    
    # Run all tests together for combined coverage
    local cmd="pytest $test_path $cov_args $PYTEST_ARGS --color=yes"
    echo -e "${CYAN}Command: $cmd${NC}"
    
    if command -v uv &> /dev/null; then
        eval "uv run $cmd" || return $?
    else
        eval "$cmd" || return $?
    fi
}

# Main execution
echo -e "${GREEN}DataKnobs Test Runner${NC}"
echo "======================================"
echo -e "Test type: ${BLUE}$TEST_TYPE${NC}"
if [ "$IN_DOCKER" = true ]; then
    echo -e "Environment: ${CYAN}Docker Container${NC}"
fi

# Determine which packages to test
if [ -n "$PACKAGE" ]; then
    # Test specific package
    if [ ! -d "$ROOT_DIR/packages/$PACKAGE" ]; then
        echo -e "${RED}Package not found: $PACKAGE${NC}"
        exit 1
    fi
    PACKAGES=("$PACKAGE")
    echo -e "Package: ${BLUE}$PACKAGE${NC}"
else
    # Discover packages based on test type
    PACKAGES=($(discover_test_packages "$TEST_TYPE"))
    echo -e "Packages: ${BLUE}${PACKAGES[*]}${NC}"
fi

if [ -n "$PYTEST_ARGS" ] || [ -n "$CUSTOM_PYTEST_ARGS" ]; then
    echo -e "Pytest args: ${CYAN}$PYTEST_ARGS $CUSTOM_PYTEST_ARGS${NC}"
fi
echo ""

# Track overall test result
OVERALL_RESULT=0

# Run tests for each package
for pkg in "${PACKAGES[@]}"; do
    echo -e "\n${GREEN}Testing package: $pkg${NC}"
    echo "----------------------------------------"
    
    case "$TEST_TYPE" in
        unit)
            run_unit_tests "$pkg" || OVERALL_RESULT=$?
            ;;
        integration)
            run_integration_tests "$pkg" || OVERALL_RESULT=$?
            ;;
        both)
            run_combined_tests "$pkg" || OVERALL_RESULT=$?
            ;;
    esac
done

# Summary
echo ""
echo "======================================"
if [ $OVERALL_RESULT -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed!${NC}"
else
    echo -e "${RED}✗ Some tests failed${NC}"
fi

exit $OVERALL_RESULT