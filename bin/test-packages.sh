#!/usr/bin/env bash
# Run tests for dataknobs packages

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get the root directory
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Default values
PACKAGE=""
VERBOSE=false
COVERAGE=false

# Usage function
usage() {
    echo "Usage: $0 [OPTIONS] [PACKAGE]"
    echo ""
    echo "Run tests for dataknobs packages"
    echo ""
    echo "Arguments:"
    echo "  PACKAGE               Specific package to test (e.g., 'common', 'utils')"
    echo "                        If not specified, tests all packages"
    echo ""
    echo "Options:"
    echo "  -v, --verbose         Run tests in verbose mode"
    echo "  -c, --coverage        Generate coverage report"
    echo "  -h, --help            Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                    # Test all packages"
    echo "  $0 structures         # Test only structures package"
    echo "  $0 -v -c utils        # Test utils with verbose output and coverage"
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -c|--coverage)
            COVERAGE=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            if [[ -z "$PACKAGE" ]]; then
                PACKAGE="$1"
            else
                echo "Unknown option: $1"
                usage
            fi
            shift
            ;;
    esac
done

# Test order matters due to dependencies
ALL_PACKAGES=(
    "common"
    "structures" 
    "xization"
    "utils"
)

# Determine which packages to test
if [[ -n "$PACKAGE" ]]; then
    if [[ ! -d "packages/$PACKAGE" ]]; then
        echo -e "${RED}Error: Package '$PACKAGE' not found${NC}"
        exit 1
    fi
    PACKAGES=("$PACKAGE")
else
    PACKAGES=("${ALL_PACKAGES[@]}")
fi

echo -e "${YELLOW}Running tests for dataknobs packages...${NC}"

# Keep track of results
declare -A TEST_RESULTS
TOTAL_TESTS=0
FAILED_PACKAGES=()

# Run tests for each package
for package in "${PACKAGES[@]}"; do
    echo -e "\n${YELLOW}Testing dataknobs-$package...${NC}"
    
    cd "packages/$package"
    
    # Build pytest command
    PYTEST_CMD="pytest"
    if [[ "$VERBOSE" == true ]]; then
        PYTEST_CMD="$PYTEST_CMD -v"
    fi
    if [[ "$COVERAGE" == true ]]; then
        PYTEST_CMD="$PYTEST_CMD --cov=dataknobs_${package//-/_} --cov-report=term-missing"
    fi
    
    # Run tests and capture result
    if $PYTEST_CMD; then
        echo -e "${GREEN}✓ Tests passed for dataknobs-$package${NC}"
        TEST_RESULTS[$package]="PASSED"
    else
        echo -e "${RED}✗ Tests failed for dataknobs-$package${NC}"
        TEST_RESULTS[$package]="FAILED"
        FAILED_PACKAGES+=("$package")
    fi
    
    cd "$ROOT_DIR"
done

# Summary
echo -e "\n${YELLOW}Test Summary:${NC}"
echo -e "${YELLOW}=============${NC}"

for package in "${PACKAGES[@]}"; do
    if [[ "${TEST_RESULTS[$package]}" == "PASSED" ]]; then
        echo -e "${GREEN}✓ dataknobs-$package: PASSED${NC}"
    else
        echo -e "${RED}✗ dataknobs-$package: FAILED${NC}"
    fi
done

if [[ ${#FAILED_PACKAGES[@]} -eq 0 ]]; then
    echo -e "\n${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "\n${RED}Tests failed for: ${FAILED_PACKAGES[*]}${NC}"
    exit 1
fi