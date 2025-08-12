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

# Source package discovery utility
source "$ROOT_DIR/bin/package-discovery.sh"

# Usage function
usage() {
    echo -e "${CYAN}dataknobs development helper${NC}"
    echo ""
    echo "Usage: $0 COMMAND [OPTIONS] [TARGETS...]"
    echo ""
    echo "Commands:"
    echo "  setup         Set up development environment"
    echo "  build         Build all packages"
    echo "  install       Install packages in dev mode"
    echo "  test          Run tests (packages, directories, or files)"
    echo "  lint          Run linting and type checking (packages, directories, or files)"
    echo "  validate      Validate code quality and catch errors"
    echo "  clean         Clean build artifacts and caches"
    echo "  release       Prepare packages for release"
    echo "  docs          Serve documentation locally"
    echo "  docs-build    Build documentation for deployment"
    echo ""
    echo "Options:"
    echo "  -h, --help    Show this help message"
    echo ""
    echo "Targets (for test, lint, validate):"
    echo "  - Package names: 'common', 'utils', 'structures', 'xization', 'config'"
    echo "  - Directory paths: 'packages/utils/src'"
    echo "  - File paths: 'packages/utils/src/dataknobs_utils/file_utils.py'"
    echo ""
    echo "Examples:"
    echo "  $0 setup                          # Set up fresh development environment"
    echo "  $0 test                           # Run all tests"
    echo "  $0 test utils                     # Test utils package"
    echo "  $0 test packages/utils/tests      # Test specific directory"
    echo "  $0 lint packages/utils/src        # Lint specific directory"
    echo "  $0 lint myfile.py                 # Lint specific file"
    echo "  $0 build                          # Build all packages"
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
    
    # Check if arguments are packages, directories, or files
    if [[ $# -eq 0 ]]; then
        # No arguments, test all packages
        "$ROOT_DIR/bin/test-packages.sh"
    else
        # Check first argument to determine type
        arg="$1"
        if [[ -d "packages/$arg" ]]; then
            # It's a package name
            "$ROOT_DIR/bin/test-packages.sh" "$@"
        elif [[ -d "$arg" ]]; then
            # It's a directory path
            echo -e "${BLUE}Testing directory: $arg${NC}"
            pytest "$arg" -v
        elif [[ -f "$arg" ]]; then
            # It's a file path
            echo -e "${BLUE}Testing file: $arg${NC}"
            pytest "$arg" -v
        else
            # Try as package name anyway
            "$ROOT_DIR/bin/test-packages.sh" "$@"
        fi
    fi
}

# Run linting and type checking
lint() {
    echo -e "${YELLOW}Running linting and type checking...${NC}"
    
    # Get packages dynamically (excluding legacy for linting)
    ALL_PACKAGES=($(discover_packages))
    PACKAGES=()
    for pkg in "${ALL_PACKAGES[@]}"; do
        if [[ "$pkg" != "legacy" ]]; then
            PACKAGES+=("$pkg")
        fi
    done
    
    # Determine what to lint
    if [[ $# -eq 0 ]]; then
        # No arguments, lint all packages
        TARGETS=()
        for package in "${PACKAGES[@]}"; do
            TARGETS+=("packages/$package/src")
        done
    else
        # Process arguments
        TARGETS=()
        for arg in "$@"; do
            if [[ -d "packages/$arg/src" ]]; then
                # It's a package name
                TARGETS+=("packages/$arg/src")
            elif [[ -d "$arg" ]]; then
                # It's a directory
                TARGETS+=("$arg")
            elif [[ -f "$arg" ]]; then
                # It's a file
                TARGETS+=("$arg")
            else
                echo -e "${RED}Warning: '$arg' not found as package, directory, or file${NC}"
            fi
        done
        
        if [[ ${#TARGETS[@]} -eq 0 ]]; then
            echo -e "${RED}No valid targets found${NC}"
            return 1
        fi
    fi
    
    # Run linting on each target
    for target in "${TARGETS[@]}"; do
        echo -e "\n${YELLOW}Checking $target...${NC}"
        
        # Run ruff check (no auto-fix during linting)
        echo -e "${BLUE}Running ruff check...${NC}"
        if ruff check "$target" --no-fix --config "$ROOT_DIR/pyproject.toml"; then
            echo -e "${GREEN}✓ Ruff check passed${NC}"
        else
            echo -e "${RED}✗ Ruff check failed${NC}"
        fi
        
        # Run ruff format check
        echo -e "${BLUE}Running ruff format check...${NC}"
        if ruff format --check "$target" --config "$ROOT_DIR/pyproject.toml"; then
            echo -e "${GREEN}✓ Ruff format check passed${NC}"
        else
            echo -e "${RED}✗ Ruff format check failed${NC}"
        fi
        
        # Run mypy with workspace configuration
        echo -e "${BLUE}Running mypy...${NC}"
        # For individual files, skip following imports to avoid checking the whole codebase
        if [[ -f "$target" ]]; then
            # Single file - don't follow imports
            if mypy "$target" --config-file "$ROOT_DIR/pyproject.toml" --follow-imports=skip; then
                echo -e "${GREEN}✓ Type check passed${NC}"
            else
                echo -e "${RED}✗ Type check failed${NC}"
            fi
        else
            # Directory or package - normal behavior
            if mypy "$target" --config-file "$ROOT_DIR/pyproject.toml"; then
                echo -e "${GREEN}✓ Type check passed${NC}"
            else
                echo -e "${RED}✗ Type check failed${NC}"
            fi
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

# Validate code
validate() {
    echo -e "${YELLOW}Validating code...${NC}"
    "$ROOT_DIR/bin/validate.sh" "$@"
}

# Prepare for release
release() {
    echo -e "${YELLOW}Preparing for release...${NC}"
    
    # Run validation first
    echo -e "${YELLOW}Running validation...${NC}"
    if ! "$ROOT_DIR/bin/validate.sh"; then
        echo -e "${RED}Validation failed! Fix before releasing.${NC}"
        exit 1
    fi
    
    # Run tests
    echo -e "${YELLOW}Running tests...${NC}"
    if ! "$ROOT_DIR/bin/test-packages.sh"; then
        echo -e "${RED}Tests failed! Fix before releasing.${NC}"
        exit 1
    fi
    
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

# Serve documentation locally
docs() {
    echo -e "${YELLOW}Starting documentation server...${NC}"
    
    # Check if MkDocs is installed
    if ! command -v mkdocs &> /dev/null; then
        echo -e "${YELLOW}Installing MkDocs dependencies...${NC}"
        uv pip install mkdocs mkdocs-material "mkdocstrings[python]" \
            mkdocs-monorepo-plugin mkdocs-awesome-pages-plugin \
            mkdocs-git-revision-date-localized-plugin
    fi
    
    # Check if packages are installed (needed for API docs)
    echo -e "${YELLOW}Ensuring packages are installed for API documentation...${NC}"
    uv pip install -e packages/common -e packages/structures \
       -e packages/utils -e packages/xization -e packages/legacy \
       -e packages/config
    
    echo -e "${GREEN}Starting documentation server at http://localhost:8000${NC}"
    echo -e "${CYAN}Press Ctrl+C to stop${NC}"
    mkdocs serve --dev-addr 0.0.0.0:8000
}

# Build documentation
docs_build() {
    echo -e "${YELLOW}Building documentation...${NC}"
    
    # Check if MkDocs is installed
    if ! command -v mkdocs &> /dev/null; then
        echo -e "${YELLOW}Installing MkDocs dependencies...${NC}"
        uv pip install mkdocs mkdocs-material "mkdocstrings[python]" \
            mkdocs-monorepo-plugin mkdocs-awesome-pages-plugin \
            mkdocs-git-revision-date-localized-plugin
    fi
    
    # Check if packages are installed (needed for API docs)
    echo -e "${YELLOW}Ensuring packages are installed for API documentation...${NC}"
    uv pip install -e packages/common -e packages/structures \
       -e packages/utils -e packages/xization -e packages/legacy \
       -e packages/config
    
    # Clean previous build
    echo -e "${YELLOW}Cleaning previous build...${NC}"
    rm -rf site/
    
    # Build documentation
    echo -e "${YELLOW}Building documentation...${NC}"
    if mkdocs build --strict; then
        echo -e "${GREEN}✓ Documentation built successfully!${NC}"
        echo -e "${CYAN}Output directory: ./site${NC}"
        echo ""
        echo -e "To preview locally, run:"
        echo -e "  python -m http.server -d site 8000"
        echo ""
        echo -e "To deploy to GitHub Pages, run:"
        echo -e "  mkdocs gh-deploy"
    else
        echo -e "${RED}✗ Documentation build failed!${NC}"
        exit 1
    fi
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
    validate)
        shift
        validate "$@"
        ;;
    clean)
        shift
        clean "$@"
        ;;
    release)
        shift
        release "$@"
        ;;
    docs)
        shift
        docs "$@"
        ;;
    docs-build)
        shift
        docs_build "$@"
        ;;
    -h|--help)
        usage
        ;;
    *)
        echo -e "${RED}Unknown command: $1${NC}"
        usage
        ;;
esac
