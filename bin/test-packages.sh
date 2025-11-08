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

# Source the package discovery utility
source "$ROOT_DIR/bin/package-discovery.sh"

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
    echo "Available packages:"
    local all_pkgs=($(discover_packages))
    echo "  ${all_pkgs[*]}"
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

# Get all packages in test order (dependency order)
ALL_PACKAGES=($(get_packages_in_order))

# Determine which packages to test
if [[ -n "$PACKAGE" ]]; then
    if ! package_exists "$PACKAGE"; then
        echo -e "${RED}Error: Package '$PACKAGE' not found${NC}"
        echo "Available packages: ${ALL_PACKAGES[*]}"
        exit 1
    fi
    PACKAGES=("$PACKAGE")
else
    PACKAGES=("${ALL_PACKAGES[@]}")
fi

echo -e "${YELLOW}Running tests for dataknobs packages...${NC}"
echo -e "${YELLOW}Testing packages: ${PACKAGES[*]}${NC}"

# Keep track of results (using parallel arrays for bash 3.2 compatibility)
TEST_PACKAGES=()
TEST_STATUSES=()
TOTAL_TESTS=0
FAILED_PACKAGES=()

# Run tests for each package
for package in "${PACKAGES[@]}"; do
    echo -e "\n${YELLOW}Testing dataknobs-$package...${NC}"
    
    cd "packages/$package"
    
    # Build pytest command (use uv run for virtual environment)
    PYTEST_CMD="uv run pytest"
    if [[ "$VERBOSE" == true ]]; then
        PYTEST_CMD="$PYTEST_CMD -v"
    fi
    if [[ "$COVERAGE" == true ]]; then
        # Handle package name conversion (e.g., config -> dataknobs_config)
        # Special case: legacy package is named "dataknobs" not "dataknobs_legacy"
        if [[ "$package" == "legacy" ]]; then
            pkg_module="dataknobs"
        else
            pkg_module="dataknobs_${package//-/_}"
        fi

        # Check if src directory exists and adjust coverage path
        if [[ -d "src/$pkg_module" ]]; then
            PYTEST_CMD="$PYTEST_CMD --cov=src/$pkg_module --cov-report=term-missing"
        elif [[ "$package" == "legacy" ]]; then
            PYTEST_CMD="$PYTEST_CMD --cov=src/dataknobs --cov-report=term-missing"
        elif [[ -d "src/dataknobs_${package}" ]]; then
            PYTEST_CMD="$PYTEST_CMD --cov=src/dataknobs_${package} --cov-report=term-missing"
        else
            PYTEST_CMD="$PYTEST_CMD --cov=$pkg_module --cov-report=term-missing"
        fi
    fi
    
    # Run tests and capture result
    if $PYTEST_CMD; then
        echo -e "${GREEN}✓ Tests passed for dataknobs-$package${NC}"
        TEST_PACKAGES+=("$package")
        TEST_STATUSES+=("PASSED")
    else
        echo -e "${RED}✗ Tests failed for dataknobs-$package${NC}"
        TEST_PACKAGES+=("$package")
        TEST_STATUSES+=("FAILED")
        FAILED_PACKAGES+=("$package")
    fi
    
    cd "$ROOT_DIR"
done

# Summary
echo -e "\n${YELLOW}Test Summary:${NC}"
echo -e "${YELLOW}=============${NC}"

# Display results using parallel arrays
for i in "${!TEST_PACKAGES[@]}"; do
    package="${TEST_PACKAGES[$i]}"
    status="${TEST_STATUSES[$i]}"
    if [[ "$status" == "PASSED" ]]; then
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