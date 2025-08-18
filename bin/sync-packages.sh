#!/usr/bin/env bash
# Sync all workspace packages - ensures all discovered packages are installed
# This supplements uv sync to handle dynamic package discovery

set -euo pipefail

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get the root directory
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Source package discovery utility
source "$ROOT_DIR/bin/package-discovery.sh"

echo -e "${BLUE}Syncing dataknobs workspace packages...${NC}"

# Check if virtual environment exists and is valid
if [ -d ".venv" ] && [ -L ".venv/bin/python" ]; then
    # Check if the symlink target exists
    if [ ! -e ".venv/bin/python" ]; then
        echo -e "${YELLOW}Virtual environment has broken symlink, recreating...${NC}"
        rm -rf .venv
    fi
fi

# First run uv sync to get dependencies
echo -e "${YELLOW}Running uv sync for dependencies...${NC}"
uv sync --all-packages

# Now install all discovered packages in editable mode
echo -e "${YELLOW}Installing discovered packages in editable mode...${NC}"
PACKAGES=($(discover_packages))

INSTALL_CMD="uv pip install"
for package in "${PACKAGES[@]}"; do
    if [ -d "packages/$package" ]; then
        INSTALL_CMD="$INSTALL_CMD -e packages/$package"
        echo -e "${BLUE}  - Found package: $package${NC}"
    fi
done

# Execute the install command
echo -e "${YELLOW}Installing packages...${NC}"
eval $INSTALL_CMD

echo -e "${GREEN}✓ All packages synced successfully!${NC}"

# Verify installation
echo -e "${YELLOW}Verifying installation...${NC}"
INSTALLED_COUNT=$(uv pip list | grep -c "dataknobs-" || true)
echo -e "${GREEN}✓ Found $INSTALLED_COUNT dataknobs packages installed${NC}"