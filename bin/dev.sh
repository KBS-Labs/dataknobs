#!/usr/bin/env bash
# Development helper script for dataknobs

set -euo pipefail

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

# Usage function
usage() {
    echo -e "${CYAN}dataknobs development helper${NC}"
    echo ""
    echo "Usage: $0 COMMAND [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  setup         Set up development environment"
    echo "  build         Build all packages"
    echo "  install       Install packages in dev mode"
    echo "  test          Run all tests"
    echo "  lint          Run linting and type checking"
    echo "  clean         Clean build artifacts and caches"
    echo "  release       Prepare packages for release"
    echo ""
    echo "Options:"
    echo "  -h, --help    Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 setup      # Set up fresh development environment"
    echo "  $0 test       # Run all tests"
    echo "  $0 build      # Build all packages"
    exit 0
}

# Setup development environment
setup() {
    echo -e "${YELLOW}Setting up development environment...${NC}"
    
    # Create virtual environment if it doesn't exist
    if [[ ! -d "venv" ]]; then
        echo -e "${YELLOW}Creating virtual environment...${NC}"
        uv venv
    fi
    
    # Activate and install
    source venv/bin/activate
    
    # Install all packages in dev mode
    echo -e "${YELLOW}Installing packages in development mode...${NC}"
    "$ROOT_DIR/bin/install-packages.sh" -m dev
    
    # Install dev dependencies
    echo -e "${YELLOW}Installing development dependencies...${NC}"
    uv pip install pytest pytest-cov pytest-mock mypy ruff
    
    echo -e "${GREEN}✓ Development environment ready!${NC}"
    echo -e "${CYAN}Activate with: source venv/bin/activate${NC}"
}

# Build all packages
build() {
    echo -e "${YELLOW}Building all packages...${NC}"
    "$ROOT_DIR/bin/build-packages.sh"
}

# Install packages
install() {
    echo -e "${YELLOW}Installing packages in development mode...${NC}"
    "$ROOT_DIR/bin/install-packages.sh" -m dev "$@"
}

# Run tests
test() {
    echo -e "${YELLOW}Running tests...${NC}"
    "$ROOT_DIR/bin/test-packages.sh" "$@"
}

# Run linting and type checking
lint() {
    echo -e "${YELLOW}Running linting and type checking...${NC}"
    
    PACKAGES=(
        "common"
        "structures" 
        "xization"
        "utils"
    )
    
    for package in "${PACKAGES[@]}"; do
        echo -e "\n${YELLOW}Checking dataknobs-$package...${NC}"
        
        # Run ruff
        echo -e "${BLUE}Running ruff...${NC}"
        if ruff check "packages/$package/src"; then
            echo -e "${GREEN}✓ Ruff check passed${NC}"
        else
            echo -e "${RED}✗ Ruff check failed${NC}"
        fi
        
        # Run mypy
        echo -e "${BLUE}Running mypy...${NC}"
        if mypy "packages/$package/src"; then
            echo -e "${GREEN}✓ Type check passed${NC}"
        else
            echo -e "${RED}✗ Type check failed${NC}"
        fi
    done
}

# Clean build artifacts
clean() {
    echo -e "${YELLOW}Cleaning build artifacts...${NC}"
    
    # Remove dist directories
    find packages -name "dist" -type d -exec rm -rf {} + 2>/dev/null || true
    
    # Remove egg-info directories
    find packages -name "*.egg-info" -type d -exec rm -rf {} + 2>/dev/null || true
    
    # Remove __pycache__ directories
    find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    
    # Remove .pyc files
    find . -name "*.pyc" -delete 2>/dev/null || true
    
    # Remove .pytest_cache directories
    find . -name ".pytest_cache" -type d -exec rm -rf {} + 2>/dev/null || true
    
    # Remove .mypy_cache directories
    find . -name ".mypy_cache" -type d -exec rm -rf {} + 2>/dev/null || true
    
    # Remove coverage files
    find . -name ".coverage" -delete 2>/dev/null || true
    find . -name "coverage.xml" -delete 2>/dev/null || true
    
    echo -e "${GREEN}✓ Cleaned build artifacts${NC}"
}

# Prepare for release
release() {
    echo -e "${YELLOW}Preparing for release...${NC}"
    
    # Run tests first
    echo -e "${YELLOW}Running tests...${NC}"
    if ! "$ROOT_DIR/bin/test-packages.sh"; then
        echo -e "${RED}Tests failed! Fix before releasing.${NC}"
        exit 1
    fi
    
    # Run linting
    echo -e "${YELLOW}Running linting...${NC}"
    lint
    
    # Build packages
    echo -e "${YELLOW}Building packages...${NC}"
    "$ROOT_DIR/bin/build-packages.sh"
    
    echo -e "${GREEN}✓ Packages ready for release!${NC}"
    echo -e "${CYAN}Next steps:${NC}"
    echo -e "  1. Update version numbers in pyproject.toml files"
    echo -e "  2. Commit and tag the release"
    echo -e "  3. Push to repository"
    echo -e "  4. Publish to PyPI with: uv publish"
}

# Main command handling
if [[ $# -eq 0 ]]; then
    usage
fi

case $1 in
    setup)
        shift
        setup "$@"
        ;;
    build)
        shift
        build "$@"
        ;;
    install)
        shift
        install "$@"
        ;;
    test)
        shift
        test "$@"
        ;;
    lint)
        shift
        lint "$@"
        ;;
    clean)
        shift
        clean "$@"
        ;;
    release)
        shift
        release "$@"
        ;;
    -h|--help)
        usage
        ;;
    *)
        echo -e "${RED}Unknown command: $1${NC}"
        usage
        ;;
esac