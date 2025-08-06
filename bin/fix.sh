#!/usr/bin/env bash
# Auto-fix code issues with ruff

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
FORMAT_ONLY=false

# Usage function
usage() {
    echo "Usage: $0 [OPTIONS] [PACKAGE]"
    echo ""
    echo "Auto-fix code issues using ruff"
    echo ""
    echo "Arguments:"
    echo "  PACKAGE               Specific package to fix (e.g., 'common', 'utils')"
    echo "                        If not specified, fixes all packages"
    echo ""
    echo "Options:"
    echo "  -f, --format-only     Only run formatting (skip linting fixes)"
    echo "  -h, --help            Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                    # Fix all packages"
    echo "  $0 structures         # Fix only structures package"
    echo "  $0 -f                 # Format all packages"
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -f|--format-only)
            FORMAT_ONLY=true
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

# All packages
ALL_PACKAGES=(
    "common"
    "structures" 
    "xization"
    "utils"
    "legacy"
)

# Determine which packages to fix
if [[ -n "$PACKAGE" ]]; then
    if [[ ! -d "packages/$PACKAGE" ]]; then
        echo -e "${RED}Error: Package '$PACKAGE' not found${NC}"
        exit 1
    fi
    PACKAGES=("$PACKAGE")
else
    PACKAGES=("${ALL_PACKAGES[@]}")
fi

echo -e "${YELLOW}Fixing code issues in dataknobs packages...${NC}"

# Fix each package
for package in "${PACKAGES[@]}"; do
    echo -e "\n${YELLOW}Fixing dataknobs-$package...${NC}"
    
    PACKAGE_DIR="packages/$package"
    
    if [[ "$FORMAT_ONLY" != true ]]; then
        # Run ruff check with auto-fix
        echo -e "${BLUE}Running ruff auto-fix...${NC}"
        # Use --unsafe-fixes=false to prevent breaking changes
        if ruff check "$PACKAGE_DIR/src" --fix --unsafe-fixes=false --config "$ROOT_DIR/pyproject.toml"; then
            echo -e "${GREEN}✓ Ruff auto-fix completed${NC}"
        else
            echo -e "${YELLOW}⚠ Some issues remain that need manual fixing${NC}"
        fi
    fi
    
    # Run ruff format
    echo -e "${BLUE}Running ruff format...${NC}"
    if ruff format "$PACKAGE_DIR/src" --config "$ROOT_DIR/pyproject.toml"; then
        echo -e "${GREEN}✓ Code formatted${NC}"
    else
        echo -e "${RED}✗ Format failed${NC}"
        exit 1
    fi
    
    # Also format tests if they exist
    if [[ -d "$PACKAGE_DIR/tests" ]]; then
        echo -e "${BLUE}Formatting tests...${NC}"
        if [[ "$FORMAT_ONLY" != true ]]; then
            ruff check "$PACKAGE_DIR/tests" --fix --config "$ROOT_DIR/pyproject.toml" || true
        fi
        ruff format "$PACKAGE_DIR/tests" --config "$ROOT_DIR/pyproject.toml"
    fi
done

echo -e "\n${GREEN}All fixes applied!${NC}"

# Suggest next steps
echo -e "\n${YELLOW}Next steps:${NC}"
echo -e "  1. Review the changes: git diff"
echo -e "  2. Run tests: ./bin/test-packages.sh"
echo -e "  3. Run full lint check: ./bin/dev.sh lint"