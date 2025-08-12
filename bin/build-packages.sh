#!/usr/bin/env bash
# Build all dataknobs packages

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get the root directory
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Source the package discovery utility
source "$ROOT_DIR/bin/package-discovery.sh"

echo -e "${YELLOW}Building dataknobs packages...${NC}"

# Get packages in dependency order
PACKAGES=($(get_packages_in_order))

echo -e "${YELLOW}Discovered packages (in build order): ${PACKAGES[*]}${NC}"

# Clean previous builds in root dist directory
echo -e "${YELLOW}Cleaning previous builds...${NC}"
rm -rf dist
mkdir -p dist

# Build each package
for package in "${PACKAGES[@]}"; do
    echo -e "\n${YELLOW}Building dataknobs-$package...${NC}"
    
    cd "packages/$package"
    
    # Build the package
    if uv build; then
        echo -e "${GREEN}✓ Successfully built dataknobs-$package${NC}"
        # Move built artifacts to root dist directory
        if [[ -d "dist" ]]; then
            mv dist/* "$ROOT_DIR/dist/" 2>/dev/null || true
            rmdir dist 2>/dev/null || true
        fi
    else
        echo -e "${RED}✗ Failed to build dataknobs-$package${NC}"
        exit 1
    fi
    
    cd "$ROOT_DIR"
done

echo -e "\n${GREEN}All packages built successfully!${NC}"

# List all built packages
echo -e "\n${YELLOW}Built packages:${NC}"
find dist -name "*.whl" -o -name "*.tar.gz" 2>/dev/null | sort || echo "No packages found"