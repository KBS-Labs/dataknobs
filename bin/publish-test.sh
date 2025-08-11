#!/usr/bin/env bash
# Script to publish packages to TestPyPI for testing

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Get the root directory
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo -e "${CYAN}Publishing Dataknobs packages to TestPyPI${NC}"
echo -e "${YELLOW}This will publish pre-release versions for testing${NC}"
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo -e "${RED}Error: uv is not installed${NC}"
    echo "Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Function to publish a package
publish_package() {
    local package_name=$1
    local package_dir="packages/$package_name"
    
    if [ ! -d "$package_dir" ]; then
        echo -e "${RED}Error: Package directory $package_dir not found${NC}"
        return 1
    fi
    
    echo -e "\n${YELLOW}Publishing $package_name...${NC}"
    cd "$package_dir"
    
    # Build the package
    echo -e "${CYAN}Building $package_name...${NC}"
    uv build
    
    # Check if dist directory exists
    if [ ! -d "dist" ]; then
        echo -e "${RED}Error: No dist directory found for $package_name${NC}"
        cd "$ROOT_DIR"
        return 1
    fi
    
    # Publish to TestPyPI
    echo -e "${CYAN}Publishing to TestPyPI...${NC}"
    echo -e "${YELLOW}You may be prompted for TestPyPI credentials${NC}"
    
    # Use twine for now as uv publish support for TestPyPI is still evolving
    if command -v twine &> /dev/null; then
        twine upload --repository testpypi dist/*
    else
        # Fallback to uv publish when it supports custom indexes
        echo -e "${YELLOW}Note: Install twine for TestPyPI upload: pip install twine${NC}"
        echo -e "${YELLOW}Or use: uv publish --index-url https://test.pypi.org/legacy/${NC}"
        # This is experimental and may not work yet
        # uv publish --index-url https://test.pypi.org/legacy/
    fi
    
    echo -e "${GREEN}✓ $package_name published to TestPyPI${NC}"
    cd "$ROOT_DIR"
}

# Check for TestPyPI credentials
if [ ! -f "$HOME/.pypirc" ]; then
    echo -e "${YELLOW}Warning: No .pypirc file found${NC}"
    echo "You'll need TestPyPI credentials. Create an account at:"
    echo "https://test.pypi.org/account/register/"
    echo ""
    echo "Then create ~/.pypirc with:"
    cat << 'EOF'
[distutils]
index-servers =
    testpypi

[testpypi]
repository = https://test.pypi.org/legacy/
username = your-testpypi-username
password = your-testpypi-password
EOF
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Install twine if not present
if ! command -v twine &> /dev/null; then
    echo -e "${YELLOW}Installing twine for TestPyPI upload...${NC}"
    uv pip install twine
fi

# Array of packages to publish
PACKAGES=(
    "common"
    "structures"
    "utils"
    "xization"
    "legacy"
)

# Ask which packages to publish
echo -e "${CYAN}Which packages do you want to publish to TestPyPI?${NC}"
echo "1) All packages"
echo "2) Select packages"
read -p "Choice (1 or 2): " choice

if [ "$choice" == "2" ]; then
    selected_packages=()
    for package in "${PACKAGES[@]}"; do
        read -p "Publish $package? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            selected_packages+=("$package")
        fi
    done
    PACKAGES=("${selected_packages[@]}")
fi

# Publish selected packages
for package in "${PACKAGES[@]}"; do
    publish_package "$package"
done

echo -e "\n${GREEN}✅ Publishing complete!${NC}"
echo -e "${CYAN}Test installation from TestPyPI with:${NC}"
echo "pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ dataknobs-structures"
echo ""
echo -e "${YELLOW}Note: Use --extra-index-url to get dependencies from PyPI${NC}"
