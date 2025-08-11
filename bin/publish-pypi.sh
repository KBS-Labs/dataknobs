#!/usr/bin/env bash
# Script to publish packages to PyPI or TestPyPI
#
# Usage:
#   ./bin/publish-pypi.sh          # Publish to PyPI
#   ./bin/publish-pypi.sh --test   # Publish to TestPyPI
#   ./bin/publish-pypi.sh --help   # Show this help

set -euo pipefail

# Show help if requested
if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
    echo "Dataknobs PyPI Publishing Tool"
    echo ""
    echo "Usage:"
    echo "  $0          # Publish to PyPI"
    echo "  $0 --test   # Publish to TestPyPI"
    echo "  $0 --help   # Show this help"
    echo ""
    echo "Authentication methods:"
    echo "  1. Create ~/.pypirc file with your PyPI token"
    echo "  2. Set UV_PUBLISH_TOKEN environment variable"
    echo ""
    exit 0
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Get the root directory
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo -e "${CYAN}Dataknobs PyPI Publishing Tool${NC}"
echo ""

# Check if using TestPyPI
TEST_MODE=false
if [ "${1:-}" = "--test" ] || [ "${1:-}" = "-t" ]; then
    TEST_MODE=true
    echo -e "${YELLOW}Running in TEST mode (TestPyPI)${NC}"
    echo ""
fi

# Check for authentication
PYPIRC_FILE="$HOME/.pypirc"
if [ -f "$PYPIRC_FILE" ]; then
    echo -e "${GREEN}✓ Found .pypirc file${NC}"
    # Extract token from .pypirc if not already in environment
    if [ -z "${UV_PUBLISH_TOKEN:-}" ] && [ -z "${UV_PUBLISH_PASSWORD:-}" ]; then
        # Try to extract token from .pypirc (handle spaces/tabs in formatting)
        if [ "$TEST_MODE" = true ]; then
            # Extract testpypi token
            PYPI_TOKEN=$(grep -A5 "\[testpypi\]" "$PYPIRC_FILE" | grep "password" | cut -d= -f2 | xargs)
        else
            # Extract pypi token (handle indentation)
            PYPI_TOKEN=$(grep -A5 "\[pypi\]" "$PYPIRC_FILE" | grep "password" | cut -d= -f2 | xargs)
        fi
        
        if [ -n "$PYPI_TOKEN" ]; then
            export UV_PUBLISH_TOKEN="$PYPI_TOKEN"
            echo -e "${GREEN}✓ Extracted token from .pypirc${NC}"
            # Debug: Show first few chars of token to verify it was extracted
            echo -e "${BLUE}  Token starts with: ${PYPI_TOKEN:0:10}...${NC}"
        else
            echo -e "${YELLOW}⚠ Could not extract token from .pypirc, will prompt for credentials${NC}"
        fi
    fi
elif [ -n "${UV_PUBLISH_TOKEN:-}" ] || [ -n "${UV_PUBLISH_PASSWORD:-}" ]; then
    echo -e "${GREEN}✓ Using environment variables for authentication${NC}"
else
    echo -e "${RED}Error: PyPI authentication not configured${NC}"
    echo ""
    echo "Please configure authentication using one of these methods:"
    echo ""
    echo "Method 1: Create ~/.pypirc file with:"
    echo "  [pypi]"
    echo "  username = __token__"
    echo "  password = pypi-..."
    echo ""
    echo "Method 2: Set environment variables:"
    echo "  export UV_PUBLISH_TOKEN='pypi-...'"
    echo "  OR"
    echo "  export UV_PUBLISH_USERNAME='__token__'"
    echo "  export UV_PUBLISH_PASSWORD='pypi-...'"
    exit 1
fi

# Package directories in order (common first, legacy last)
PACKAGES=(
    "packages/common"
    "packages/structures"
    "packages/utils"
    "packages/xization"
    "packages/legacy"
)

# Function to get version from pyproject.toml
get_version() {
    local package_dir=$1
    grep '^version = ' "$package_dir/pyproject.toml" | cut -d'"' -f2
}

# Function to build package
build_package() {
    local package_dir=$1
    local package_name=$(basename "$package_dir")
    
    echo -e "${CYAN}Building $package_name...${NC}"
    # Build from package directory, artifacts go to root dist/
    (cd "$package_dir" && uv build)
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Built $package_name${NC}"
        return 0
    else
        echo -e "${RED}✗ Failed to build $package_name${NC}"
        return 1
    fi
}

# Function to publish package
publish_package() {
    local package_dir=$1
    local package_name=$(basename "$package_dir")
    local version=$(get_version "$package_dir")
    
    # Package artifacts are in root dist/ directory
    # Legacy package uses "dataknobs" without underscore, others use "dataknobs_"
    if [ "$package_name" = "legacy" ]; then
        local wheel_file="${ROOT_DIR}/dist/dataknobs-${version}-py3-none-any.whl"
        local tar_file="${ROOT_DIR}/dist/dataknobs-${version}.tar.gz"
    else
        local wheel_file="${ROOT_DIR}/dist/dataknobs_${package_name}-${version}-py3-none-any.whl"
        local tar_file="${ROOT_DIR}/dist/dataknobs_${package_name}-${version}.tar.gz"
    fi
    
    # Check if artifacts exist
    if [ ! -f "$wheel_file" ] || [ ! -f "$tar_file" ]; then
        echo -e "${RED}✗ Build artifacts not found for $package_name v$version${NC}"
        echo "  Expected: $wheel_file"
        echo "  Expected: $tar_file"
        echo "  Run build first or check the dist/ directory"
        return 1
    fi
    
    if [ "$TEST_MODE" = true ]; then
        echo -e "${CYAN}Publishing $package_name v$version to TestPyPI...${NC}"
        PUBLISH_ARGS="--publish-url https://test.pypi.org/legacy/"
        INDEX_URL="https://test.pypi.org/simple/"
    else
        echo -e "${CYAN}Publishing $package_name v$version to PyPI...${NC}"
        PUBLISH_ARGS=""
        INDEX_URL="https://pypi.org/simple/"
    fi
    
    # Publish the specific artifacts with authentication
    if [ -n "${UV_PUBLISH_TOKEN:-}" ]; then
        # Use token authentication
        echo -e "${BLUE}  Using token authentication (token starts with: ${UV_PUBLISH_TOKEN:0:10}...)${NC}"
        uv publish $PUBLISH_ARGS --token "${UV_PUBLISH_TOKEN}" "$wheel_file" "$tar_file"
    elif [ -n "${UV_PUBLISH_PASSWORD:-}" ]; then
        # Use username/password authentication
        uv publish $PUBLISH_ARGS --username "${UV_PUBLISH_USERNAME:-__token__}" --password "${UV_PUBLISH_PASSWORD}" "$wheel_file" "$tar_file"
    else
        # No credentials in environment, uv will prompt
        uv publish $PUBLISH_ARGS "$wheel_file" "$tar_file"
    fi
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Published $package_name v$version${NC}"
        return 0
    else
        # Check if it failed because package already exists
        if [ -n "${UV_PUBLISH_TOKEN:-}" ]; then
            uv publish $PUBLISH_ARGS --token "${UV_PUBLISH_TOKEN}" "$wheel_file" "$tar_file" 2>&1 | grep -q "already exists"
        else
            uv publish $PUBLISH_ARGS "$wheel_file" "$tar_file" 2>&1 | grep -q "already exists"
        fi
        if [ $? -eq 0 ]; then
            echo -e "${YELLOW}⚠ $package_name v$version already exists, skipping${NC}"
            return 0
        else
            echo -e "${RED}✗ Failed to publish $package_name${NC}"
            return 1
        fi
    fi
}

# Check if dist directory exists and has artifacts
if [ -d "${ROOT_DIR}/dist" ]; then
    echo -e "${BLUE}Found dist/ directory with artifacts:${NC}"
    # Count artifacts for each package
    for package_dir in "${PACKAGES[@]}"; do
        package_name=$(basename "$package_dir")
        version=$(get_version "$package_dir")
        
        if [ "$package_name" = "legacy" ]; then
            wheel_file="${ROOT_DIR}/dist/dataknobs-${version}-py3-none-any.whl"
        else
            wheel_file="${ROOT_DIR}/dist/dataknobs_${package_name}-${version}-py3-none-any.whl"
        fi
        
        if [ -f "$wheel_file" ]; then
            echo -e "  ${GREEN}✓${NC} $package_name v$version"
        else
            echo -e "  ${RED}✗${NC} $package_name v$version (not built)"
        fi
    done
    echo ""
fi

# Ask what to do
echo -e "${CYAN}What would you like to do?${NC}"
echo "1) Publish already built packages (recommended if dist/ has artifacts)"
echo "2) Build and publish all packages"
echo "3) Build all packages only"
echo "4) Select specific packages to publish"
echo "5) Exit"
echo -n "Choice (1-5): "
read choice

case "$choice" in
    1)
        # Publish only (already built)
        echo -e "\n${CYAN}Publishing all packages...${NC}"
        for package_dir in "${PACKAGES[@]}"; do
            publish_package "$package_dir" || exit 1
        done
        ;;
    
    2)
        # Build and publish all
        echo -e "\n${CYAN}Building all packages...${NC}"
        for package_dir in "${PACKAGES[@]}"; do
            build_package "$package_dir" || exit 1
        done
        
        echo -e "\n${CYAN}Publishing all packages...${NC}"
        for package_dir in "${PACKAGES[@]}"; do
            publish_package "$package_dir" || exit 1
        done
        ;;
    
    3)
        # Build only
        echo -e "\n${CYAN}Building all packages...${NC}"
        for package_dir in "${PACKAGES[@]}"; do
            build_package "$package_dir" || exit 1
        done
        ;;
    
    4)
        # Select specific packages to publish
        for package_dir in "${PACKAGES[@]}"; do
            package_name=$(basename "$package_dir")
            version=$(get_version "$package_dir")
            
            # Check if artifacts exist (legacy uses different naming)
            if [ "$package_name" = "legacy" ]; then
                wheel_file="${ROOT_DIR}/dist/dataknobs-${version}-py3-none-any.whl"
            else
                wheel_file="${ROOT_DIR}/dist/dataknobs_${package_name}-${version}-py3-none-any.whl"
            fi
            
            if [ -f "$wheel_file" ]; then
                echo -n "Publish $package_name v$version? (y/n) "
                read REPLY
                if [[ $REPLY =~ ^[Yy]$ ]]; then
                    publish_package "$package_dir" || continue
                fi
            else
                echo -e "${YELLOW}No artifacts found for $package_name v$version, skipping${NC}"
            fi
        done
        ;;
    
    5)
        echo "Exiting..."
        exit 0
        ;;
    
    *)
        echo -e "${RED}Invalid choice${NC}"
        exit 1
        ;;
esac

echo -e "\n${GREEN}✅ Publishing complete!${NC}"

# Verify packages on PyPI
if [ "$TEST_MODE" = true ]; then
    echo -e "\n${CYAN}Verifying packages on TestPyPI:${NC}"
    for package_dir in "${PACKAGES[@]}"; do
        package_name=$(basename "$package_dir")
        version=$(get_version "$package_dir")
        echo -e "  dataknobs-$package_name v$version: https://test.pypi.org/project/dataknobs-$package_name/"
    done
else
    echo -e "\n${CYAN}Verifying packages on PyPI:${NC}"
    for package_dir in "${PACKAGES[@]}"; do
        package_name=$(basename "$package_dir")
        version=$(get_version "$package_dir")
        echo -e "  dataknobs-$package_name v$version: https://pypi.org/project/dataknobs-$package_name/"
    done
fi