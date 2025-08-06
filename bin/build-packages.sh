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

echo -e "${YELLOW}Building dataknobs packages...${NC}"

# Build order matters due to dependencies
PACKAGES=(
    "common"
    "structures" 
    "xization"
    "utils"
    "legacy"
)

# Clean previous builds
echo -e "${YELLOW}Cleaning previous builds...${NC}"
for package in "${PACKAGES[@]}"; do
    rm -rf "packages/$package/dist"
done

# Build each package
for package in "${PACKAGES[@]}"; do
    echo -e "\n${YELLOW}Building dataknobs-$package...${NC}"
    
    cd "packages/$package"
    
    # Build the package
    if uv build; then
        echo -e "${GREEN}✓ Successfully built dataknobs-$package${NC}"
    else
        echo -e "${RED}✗ Failed to build dataknobs-$package${NC}"
        exit 1
    fi
    
    cd "$ROOT_DIR"
done

echo -e "\n${GREEN}All packages built successfully!${NC}"

# List all built packages
echo -e "\n${YELLOW}Built packages:${NC}"
find packages/*/dist -name "*.whl" -o -name "*.tar.gz" | sort