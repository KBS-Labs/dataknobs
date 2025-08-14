#!/bin/bash
# Test runner with options for unit/integration/both tests

set -e

# Get the script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
TEST_TYPE="both"
PACKAGE=""
VERBOSE=""
START_SERVICES="auto"
EXTRA_ARGS=""

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS] [PACKAGE]

Run unit and/or integration tests for DataKnobs packages.

Options:
    -t, --type TYPE        Test type: unit, integration, or both (default: both)
    -p, --package PACKAGE  Package to test (e.g., data, config, structures)
                          If not specified, tests all packages
    -s, --services         Start services for integration tests (auto by default)
    -n, --no-services      Don't start services (assume they're already running)
    -v, --verbose          Run tests in verbose mode
    -k EXPRESSION          Only run tests matching the expression (pytest -k)
    -x, --exitfirst        Exit on first failure
    -h, --help             Show this help message

Examples:
    $0                           # Run all tests for all packages
    $0 -t unit                   # Run only unit tests for all packages
    $0 -t integration data       # Run integration tests for data package
    $0 -t unit -p config         # Run unit tests for config package
    $0 -v -k "test_s3"          # Run tests matching "test_s3" verbosely
    $0 -n -t integration         # Run integration tests without starting services

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
        -v|--verbose)
            VERBOSE="-v"
            shift
            ;;
        -k)
            EXTRA_ARGS="$EXTRA_ARGS -k $2"
            shift 2
            ;;
        -x|--exitfirst)
            EXTRA_ARGS="$EXTRA_ARGS -x"
            shift
            ;;
        -h|--help)
            show_usage
            ;;
        *)
            # Assume it's a package name if no package specified yet
            if [ -z "$PACKAGE" ]; then
                PACKAGE="$1"
            else
                echo -e "${RED}Unknown option: $1${NC}"
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
    
    # Run tests
    if command -v uv &> /dev/null; then
        uv run pytest "$test_path" $exclude_args $VERBOSE $EXTRA_ARGS \
            --tb=short \
            --color=yes || return $?
    else
        pytest "$test_path" $exclude_args $VERBOSE $EXTRA_ARGS \
            --tb=short \
            --color=yes || return $?
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
    
    # Start services if needed
    if [ "$START_SERVICES" = "auto" ] || [ "$START_SERVICES" = "yes" ]; then
        if [ "$START_SERVICES" = "auto" ]; then
            # Check if services are already running
            if docker ps | grep -q "dataknobs.*postgres\|dataknobs.*elasticsearch\|dataknobs.*localstack"; then
                echo -e "${BLUE}Services appear to be running, skipping startup${NC}"
            else
                START_SERVICES="yes"
            fi
        fi
        
        if [ "$START_SERVICES" = "yes" ]; then
            echo -e "${YELLOW}Starting services for integration tests...${NC}"
            "$SCRIPT_DIR/run-integration-tests.sh" --services-only || {
                echo -e "${RED}Failed to start services${NC}"
                return 1
            }
        fi
    fi
    
    # Set environment variables for tests
    export POSTGRES_HOST=localhost
    export POSTGRES_PORT=5432
    export POSTGRES_USER=postgres
    export POSTGRES_PASSWORD=postgres
    export POSTGRES_DB=dataknobs_test
    export ELASTICSEARCH_HOST=localhost
    export ELASTICSEARCH_PORT=9200
    export AWS_ENDPOINT_URL=http://localhost:4566
    export AWS_ACCESS_KEY_ID=test
    export AWS_SECRET_ACCESS_KEY=test
    export AWS_DEFAULT_REGION=us-east-1
    export LOCALSTACK_ENDPOINT=http://localhost:4566
    
    # Run tests - include all integration tests, not just marked ones
    if command -v uv &> /dev/null; then
        uv run pytest "$test_path" $VERBOSE $EXTRA_ARGS \
            --tb=short \
            --color=yes || return $?
    else
        pytest "$test_path" $VERBOSE $EXTRA_ARGS \
            --tb=short \
            --color=yes || return $?
    fi
}

# Main execution
echo -e "${GREEN}DataKnobs Test Runner${NC}"
echo "======================================"
echo -e "Test type: ${BLUE}$TEST_TYPE${NC}"

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
            # Run unit tests first
            if [ -d "$ROOT_DIR/packages/$pkg/tests" ]; then
                run_unit_tests "$pkg" || OVERALL_RESULT=$?
            fi
            echo ""
            # Then run integration tests
            if [ -d "$ROOT_DIR/packages/$pkg/tests/integration" ]; then
                run_integration_tests "$pkg" || OVERALL_RESULT=$?
            fi
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