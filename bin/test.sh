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
TEST_PATH=""  # For direct file/directory paths
START_SERVICES="auto"
COVERAGE="yes"
PYTEST_ARGS=""
COV_REPORT="term-missing"
SKIP_INTEGRATION="false"  # If true, sets TEST_*=false
ONLY_INTEGRATION="false"  # If true, only runs integration tests

# Check if we're running in a Docker container
IN_DOCKER=false
if [ -f /.dockerenv ] || [ -n "${DOCKER_CONTAINER:-}" ]; then
    IN_DOCKER=true
fi

# Consolidated function to execute pytest with proper error handling
execute_pytest() {
    local cmd="$1"
    local context="${2:-tests}"  # Optional context for warning message
    
    echo -e "${CYAN}Command: $cmd${NC}"
    
    local test_result
    if command -v uv &> /dev/null; then
        eval "uv run $cmd"
        test_result=$?
    else
        eval "$cmd"
        test_result=$?
    fi
    
    # Exit code 5 means no tests were collected - treat as success with warning
    if [ $test_result -eq 5 ]; then
        echo -e "${YELLOW}Warning: No $context found${NC}"
        return 0
    fi
    
    return $test_result
}

# Function to set environment variables for integration tests
set_integration_env_vars() {
    if [ "$IN_DOCKER" = true ]; then
        # Use Docker network hostnames when inside container
        export POSTGRES_HOST=postgres
        export ELASTICSEARCH_HOST=elasticsearch
        export AWS_ENDPOINT_URL=http://localstack:4566
        export LOCALSTACK_ENDPOINT=http://localstack:4566
    else
        # Use localhost when running on host
        export POSTGRES_HOST="${POSTGRES_HOST:-localhost}"
        export ELASTICSEARCH_HOST="${ELASTICSEARCH_HOST:-localhost}"
        export AWS_ENDPOINT_URL="${AWS_ENDPOINT_URL:-http://localhost:4566}"
        export LOCALSTACK_ENDPOINT="${LOCALSTACK_ENDPOINT:-http://localhost:4566}"
    fi
    export POSTGRES_PORT="${POSTGRES_PORT:-5432}"
    export POSTGRES_USER="${POSTGRES_USER:-postgres}"
    export POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-postgres}"
    export POSTGRES_DB="${POSTGRES_DB:-dataknobs_test}"
    export ELASTICSEARCH_PORT="${ELASTICSEARCH_PORT:-9200}"
    export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-test}"
    export AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-test}"
    export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"
    
    # Enable test flags based on environment or defaults
    # These control whether integration tests are skipped or run
    # Can be overridden by setting these before calling the script
    if [ "$SKIP_INTEGRATION" = "true" ]; then
        # Explicitly skip integration tests
        export TEST_S3="false"
        export TEST_ELASTICSEARCH="false"
        export TEST_POSTGRES="false"
        export TEST_OLLAMA="false"
    else
        # Use environment values or default to true (run tests if services available)
        export TEST_S3="${TEST_S3:-true}"
        export TEST_ELASTICSEARCH="${TEST_ELASTICSEARCH:-true}"
        export TEST_POSTGRES="${TEST_POSTGRES:-true}"
        export TEST_OLLAMA="${TEST_OLLAMA:-true}"
    fi
}

# Function to show usage
show_usage() {
    cat << EOF
${CYAN}DataKnobs Test Runner${NC}

Usage: $0 [OPTIONS] [PACKAGE|PATH] [-- PYTEST_ARGS]

Run unit and/or integration tests for DataKnobs packages with flexible pytest options.

PACKAGE|PATH can be:
  - A package name (e.g., 'data', 'config')
  - A test file path (e.g., 'packages/data/tests/test_backends/test_s3.py')
  - A test directory (e.g., 'packages/data/tests/integration/')

${YELLOW}Options:${NC}
    -t, --type TYPE          Test type: unit, integration, or both (default: both)
    -p, --package PACKAGE    Package or path to test
                            Can be a package name (e.g., data, config)
                            or a file/directory path (e.g., packages/data/tests/test_s3.py)
                            If not specified, tests all packages
    -s, --services          Start services for integration tests (auto by default)
    -n, --no-services       Don't start services (assume they're already running)
    --skip-integration      Skip integration tests (sets TEST_*=false)
    --only-integration      Only run integration tests (requires services)
    --no-cov                Disable coverage reporting
    --cov-report TYPE       Coverage report type: term, term-missing, html, xml, or combinations
                           (default: term-missing, use comma to combine: term-missing,html,xml)
    -h, --help              Show this help message

${YELLOW}Advanced Usage:${NC}
    Any arguments after -- are passed directly to pytest:
    $0 data -- -xvs --tb=short --pdb
    
    Common pytest options you can pass after --:
    -v, -vv, -vvv           Verbosity level
    -x                      Exit on first failure
    -s                      No capture, show print statements
    --tb=STYLE              Traceback style (auto, short, line, no, native, long)
    -k EXPRESSION           Only run tests matching expression
    -m MARKERS              Only run tests with specified markers
    --lf                    Rerun only failures from last run
    --ff                    Run failures first, then other tests
    --pdb                   Drop into debugger on failures
    --maxfail=N             Stop after N failures
    
${YELLOW}Examples:${NC}
    $0                                    # Run all tests with default settings
    $0 data                               # Test data package
    $0 -t unit data                       # Unit tests only for data package
    $0 packages/data/tests/test_s3.py    # Run specific test file
    $0 packages/data/tests/integration/  # Run all integration tests for data
    $0 data -- -xvs                       # Exit on first failure, verbose, no capture
    $0 data -- -vv --tb=short             # Very verbose with short tracebacks
    $0 data -- -k "test_s3"               # Run only tests matching "test_s3"
    $0 data -- -m "slow"                  # Run only tests marked as slow
    $0 data -- --lf                       # Rerun only last failures
    $0 data -- --pdb --maxfail=3          # Custom pytest args
    $0 -n data                            # Run without starting services (Docker)
    $0 --skip-integration data            # Skip integration tests, only run unit tests
    $0 --only-integration data            # Only run integration tests

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
            # Check if it's a file path or package name
            if [[ "$2" == *"/"* ]] || [ -f "$2" ] || [ -d "$2" ]; then
                TEST_PATH="$2"
            else
                PACKAGE="$2"
            fi
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
        --skip-integration)
            SKIP_INTEGRATION="true"
            shift
            ;;
        --only-integration)
            ONLY_INTEGRATION="true"
            TEST_TYPE="integration"
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
        --cov-report=*)
            COV_REPORT="${1#--cov-report=}"
            shift
            ;;
        # Help
        -h|--help)
            show_usage
            ;;
        # Separator for custom pytest args - everything after this goes to pytest
        --)
            shift
            PYTEST_ARGS="$@"
            break
            ;;
        *)
            # Check if it's a package name or file path
            if [ -z "$PACKAGE" ] && [ -z "$TEST_PATH" ]; then
                # Check if it looks like a file/directory path
                if [[ "$1" == *"/"* ]] || [ -f "$1" ] || [ -d "$1" ]; then
                    TEST_PATH="$1"
                else
                    PACKAGE="$1"
                fi
            else
                echo -e "${RED}Unknown option: $1${NC}"
                echo "Use -- to pass arguments to pytest"
                show_usage
            fi
            shift
            ;;
    esac
done

# Validate test type
if [[ "$TEST_TYPE" != "unit" && "$TEST_TYPE" != "integration" && "$TEST_TYPE" != "both" ]]; then
    echo -e "${RED}Invalid test type: $TEST_TYPE${NC}"
    echo "Must be one of: unit, integration, both"
    exit 1
fi

# Handle conflicting flags
if [ "$SKIP_INTEGRATION" = "true" ] && [ "$ONLY_INTEGRATION" = "true" ]; then
    echo -e "${RED}Error: Cannot use --skip-integration and --only-integration together${NC}"
    exit 1
fi

# Adjust test type based on flags
if [ "$SKIP_INTEGRATION" = "true" ]; then
    if [ "$TEST_TYPE" = "integration" ]; then
        echo -e "${RED}Error: Cannot skip integration tests when test type is 'integration'${NC}"
        echo "Remove --skip-integration or change test type"
        exit 1
    elif [ "$TEST_TYPE" = "both" ]; then
        # Silently change to unit tests only
        TEST_TYPE="unit"
    fi
fi

# Function to extract package name from a test path
extract_package_from_path() {
    local path=$1
    # Convert to absolute path if relative
    if [[ "$path" != /* ]]; then
        path="$ROOT_DIR/$path"
    fi
    
    # Extract package name if path is under packages/
    if [[ "$path" == *"/packages/"* ]]; then
        local package_part="${path#*/packages/}"
        echo "${package_part%%/*}"
    else
        echo ""
    fi
}

# Function to run tests for a specific file or directory
run_path_tests() {
    local path=$1
    echo -e "${YELLOW}Running tests from path: $path${NC}"
    
    # Convert to absolute path if relative
    if [[ "$path" != /* ]]; then
        # First check if the path exists relative to current directory
        if [ -e "$path" ]; then
            path="$(cd "$(dirname "$path")" && pwd)/$(basename "$path")"
        # Otherwise try relative to ROOT_DIR
        elif [ -e "$ROOT_DIR/$path" ]; then
            path="$ROOT_DIR/$path"
        else
            echo -e "${RED}Path not found: $path${NC}"
            return 1
        fi
    elif [ ! -e "$path" ]; then
        echo -e "${RED}Path not found: $path${NC}"
        return 1
    fi
    
    # Always set environment variables for tests (they control which tests are skipped)
    set_integration_env_vars
    
    # Check if it's an integration test path to determine if services are needed
    # Skip service startup if we're explicitly skipping integration tests
    if [ "$SKIP_INTEGRATION" != "true" ] && ([[ "$path" == *"/integration"* ]] || [[ "$path" == *"/integration/"* ]] || [[ "$path" == *"/tests"* ]]); then
        # Start services if needed using manage-services.sh
        if [ "$START_SERVICES" = "auto" ] || [ "$START_SERVICES" = "yes" ]; then
            if [ "$IN_DOCKER" = true ]; then
                echo -e "${BLUE}Running in Docker container, checking service connectivity...${NC}"
                "$SCRIPT_DIR/manage-services.sh" ensure >/dev/null 2>&1 || {
                    echo -e "${YELLOW}Warning: Some services may not be reachable from container${NC}"
                }
            else
                if [ "$START_SERVICES" = "yes" ] || [ "$START_SERVICES" = "auto" ]; then
                    echo -e "${YELLOW}Ensuring services are running for integration tests...${NC}"
                    "$SCRIPT_DIR/manage-services.sh" ensure || {
                        echo -e "${RED}Failed to ensure services are running${NC}"
                        return 1
                    }
                    SERVICES_STARTED=true
                fi
            fi
        fi
    fi
    
    # Build coverage args if enabled
    local cov_args=""
    if [ "$COVERAGE" = "yes" ]; then
        # Try to extract package name for coverage
        local package=$(extract_package_from_path "$path")
        if [ -n "$package" ]; then
            # Special case: legacy package is named "dataknobs" not "dataknobs_legacy"
            if [ "$package" = "legacy" ]; then
                cov_args="--cov=packages/$package/src/dataknobs"
            else
                cov_args="--cov=packages/$package/src/dataknobs_${package}"
            fi
        else
            # Fall back to covering the test path itself
            cov_args="--cov=$path"
        fi
        
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
    local cmd="pytest $path $cov_args $PYTEST_ARGS --color=yes"
    execute_pytest "$cmd" "tests in $path"
}

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
        # Special case: legacy package is named "dataknobs" not "dataknobs_legacy"
        if [ "$package" = "legacy" ]; then
            cov_args="--cov=packages/$package/src/dataknobs"
        else
            cov_args="--cov=packages/$package/src/dataknobs_${package}"
        fi
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
    execute_pytest "$cmd" "unit tests for $package"
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
    
    # Start services if needed using manage-services.sh
    if [ "$START_SERVICES" = "auto" ] || [ "$START_SERVICES" = "yes" ]; then
        if [ "$IN_DOCKER" = true ]; then
            echo -e "${BLUE}Running in Docker container, checking service connectivity...${NC}"
            # Use manage-services.sh to check service connectivity from within container
            "$SCRIPT_DIR/manage-services.sh" ensure >/dev/null 2>&1 || {
                echo -e "${YELLOW}Warning: Some services may not be reachable from container${NC}"
            }
            START_SERVICES="no"  # Don't try to start services from within container
        else
            if [ "$START_SERVICES" = "auto" ]; then
                START_SERVICES="yes"  # Let manage-services.sh decide if they're already running
            fi
            
            if [ "$START_SERVICES" = "yes" ]; then
                echo -e "${YELLOW}Ensuring services are running for integration tests...${NC}"
                "$SCRIPT_DIR/manage-services.sh" ensure || {
                    echo -e "${RED}Failed to ensure services are running${NC}"
                    return 1
                }
                SERVICES_STARTED=true
            fi
        fi
    fi
    
    # Set environment variables for tests
    set_integration_env_vars

    # Build coverage args if enabled
    local cov_args=""
    if [ "$COVERAGE" = "yes" ]; then
        # Special case: legacy package is named "dataknobs" not "dataknobs_legacy"
        if [ "$package" = "legacy" ]; then
            cov_args="--cov=packages/$package/src/dataknobs --cov-append"
        else
            cov_args="--cov=packages/$package/src/dataknobs_${package} --cov-append"
        fi
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
    execute_pytest "$cmd" "integration tests for $package"
}

# Function to run combined tests with coverage
run_combined_tests() {
    local package=$1
    echo -e "${YELLOW}Running all tests for package: $package${NC}"
    
    local test_path="packages/$package/tests"
    
    # Check if we need to start services for integration tests
    # Skip service startup if we're explicitly skipping integration tests
    if [ "$SKIP_INTEGRATION" != "true" ] && [ -d "$test_path/integration" ]; then
        # Start services if needed using manage-services.sh
        if [ "$START_SERVICES" = "auto" ] || [ "$START_SERVICES" = "yes" ]; then
            if [ "$IN_DOCKER" = true ]; then
                echo -e "${BLUE}Running in Docker container, checking service connectivity...${NC}"
                # Use manage-services.sh to check service connectivity from within container
                "$SCRIPT_DIR/manage-services.sh" ensure >/dev/null 2>&1 || {
                    echo -e "${YELLOW}Warning: Some services may not be reachable from container${NC}"
                }
                START_SERVICES="no"  # Don't try to start services from within container
            else
                if [ "$START_SERVICES" = "auto" ]; then
                    START_SERVICES="yes"  # Let manage-services.sh decide if they're already running
                fi
                
                if [ "$START_SERVICES" = "yes" ]; then
                    echo -e "${YELLOW}Ensuring services are running for integration tests...${NC}"
                    "$SCRIPT_DIR/manage-services.sh" ensure || {
                        echo -e "${RED}Failed to ensure services are running${NC}"
                        echo -e "${YELLOW}Integration tests may fail without services${NC}"
                    }
                    SERVICES_STARTED=true
                fi
            fi
        fi
        
        # Set environment variables for integration tests
        set_integration_env_vars
    fi

    # Build coverage args if enabled
    local cov_args=""
    if [ "$COVERAGE" = "yes" ]; then
        # Special case: legacy package is named "dataknobs" not "dataknobs_legacy"
        if [ "$package" = "legacy" ]; then
            cov_args="--cov=packages/$package/src/dataknobs"
        else
            cov_args="--cov=packages/$package/src/dataknobs_${package}"
        fi
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
    execute_pytest "$cmd" "tests for $package"
}

# Main execution
echo -e "${GREEN}DataKnobs Test Runner${NC}"
echo "======================================"
echo -e "Test type: ${BLUE}$TEST_TYPE${NC}"
if [ "$IN_DOCKER" = true ]; then
    echo -e "Environment: ${CYAN}Docker Container${NC}"
fi

# Determine what to test (file path vs package)
if [ -n "$TEST_PATH" ]; then
    # Test specific file or directory path
    echo -e "Test path: ${BLUE}$TEST_PATH${NC}"
    # For file paths, we'll run them directly, not through package logic
    USE_PATH_MODE=true
elif [ -n "$PACKAGE" ]; then
    # Test specific package
    if [ ! -d "$ROOT_DIR/packages/$PACKAGE" ]; then
        echo -e "${RED}Package not found: $PACKAGE${NC}"
        exit 1
    fi
    PACKAGES=("$PACKAGE")
    echo -e "Package: ${BLUE}$PACKAGE${NC}"
    USE_PATH_MODE=false
else
    # Discover packages based on test type
    PACKAGES=($(discover_test_packages "$TEST_TYPE"))
    echo -e "Packages: ${BLUE}${PACKAGES[*]}${NC}"
    USE_PATH_MODE=false
fi

if [ -n "$PYTEST_ARGS" ]; then
    echo -e "Pytest args: ${CYAN}$PYTEST_ARGS${NC}"
fi
echo ""

# Track overall test result
OVERALL_RESULT=0
SERVICES_STARTED=false

# Function to cleanup services if we started them
cleanup_services() {
    if [ "$SERVICES_STARTED" = true ] && [ "$IN_DOCKER" = false ]; then
        if [ -f "/tmp/.dataknobs_services_started_$$" ]; then
            echo -e "\n${YELLOW}Cleaning up services...${NC}"
            "$SCRIPT_DIR/manage-services.sh" stop
        fi
    fi
}

# Set trap for cleanup on exit
trap cleanup_services EXIT INT TERM

# Run tests based on mode
if [ "$USE_PATH_MODE" = true ]; then
    # Run tests for the specified path
    run_path_tests "$TEST_PATH" || OVERALL_RESULT=$?
else
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
fi

# Summary
echo ""
echo "======================================"
if [ $OVERALL_RESULT -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed!${NC}"
else
    echo -e "${RED}✗ Some tests failed${NC}"
fi

exit $OVERALL_RESULT