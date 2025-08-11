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
TARGETS=()
FORMAT_ONLY=false

# Usage function
usage() {
    echo "Usage: $0 [OPTIONS] [TARGETS...]"
    echo ""
    echo "Auto-fix code issues using ruff"
    echo ""
    echo "Arguments:"
    echo "  TARGETS               Packages, directories, or files to fix"
    echo "                        Can be:"
    echo "                        - Package name (e.g., 'common', 'utils')"
    echo "                        - Directory path (e.g., 'packages/utils/src')"
    echo "                        - File path (e.g., 'packages/utils/src/dataknobs_utils/file_utils.py')"
    echo "                        If not specified, fixes all packages"
    echo ""
    echo "Options:"
    echo "  -f, --format-only     Only run formatting (skip linting fixes)"
    echo "  -h, --help            Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                        # Fix all packages"
    echo "  $0 structures                             # Fix only structures package"
    echo "  $0 packages/utils/src                     # Fix specific directory"
    echo "  $0 packages/utils/src/dataknobs_utils/*.py  # Fix specific files"
    echo "  $0 -f                                     # Format all packages"
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
            # Add to targets list
            TARGETS+=("$1")
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

# Determine what to fix
FIX_TARGETS=()

if [[ ${#TARGETS[@]} -eq 0 ]]; then
    # No targets specified, fix all packages
    for package in "${ALL_PACKAGES[@]}"; do
        if [[ -d "packages/$package/src" ]]; then
            FIX_TARGETS+=("packages/$package/src")
        fi
        if [[ -d "packages/$package/tests" ]]; then
            FIX_TARGETS+=("packages/$package/tests")
        fi
    done
else
    # Process specified targets
    for target in "${TARGETS[@]}"; do
        if [[ -d "packages/$target" ]]; then
            # It's a package name
            if [[ -d "packages/$target/src" ]]; then
                FIX_TARGETS+=("packages/$target/src")
            fi
            if [[ -d "packages/$target/tests" ]]; then
                FIX_TARGETS+=("packages/$target/tests")
            fi
        elif [[ -d "$target" ]]; then
            # It's a directory
            FIX_TARGETS+=("$target")
        elif [[ -f "$target" ]]; then
            # It's a file
            FIX_TARGETS+=("$target")
        else
            # Try glob expansion
            shopt -s nullglob
            files=($target)
            shopt -u nullglob
            if [[ ${#files[@]} -gt 0 ]]; then
                FIX_TARGETS+=("${files[@]}")
            else
                echo -e "${YELLOW}Warning: Target '$target' not found${NC}"
            fi
        fi
    done
fi

if [[ ${#FIX_TARGETS[@]} -eq 0 ]]; then
    echo -e "${RED}No valid targets found to fix${NC}"
    exit 1
fi

echo -e "${YELLOW}Fixing code issues...${NC}"

# Fix each target
for target in "${FIX_TARGETS[@]}"; do
    echo -e "\n${YELLOW}Fixing $target...${NC}"
    
    if [[ "$FORMAT_ONLY" != true ]]; then
        # Run ruff check with auto-fix
        echo -e "${BLUE}Running ruff auto-fix...${NC}"
        # Use --no-unsafe-fixes to prevent breaking changes
        if ruff check "$target" --fix --no-unsafe-fixes --config "$ROOT_DIR/pyproject.toml"; then
            echo -e "${GREEN}✓ Ruff auto-fix completed${NC}"
        else
            echo -e "${YELLOW}⚠ Some issues remain that need manual fixing${NC}"
        fi
    fi
    
    # Run ruff format
    echo -e "${BLUE}Running ruff format...${NC}"
    if ruff format "$target" --config "$ROOT_DIR/pyproject.toml"; then
        echo -e "${GREEN}✓ Code formatted${NC}"
    else
        echo -e "${RED}✗ Format failed${NC}"
        exit 1
    fi
done

echo -e "\n${GREEN}All fixes applied!${NC}"

# Suggest next steps
echo -e "\n${YELLOW}Next steps:${NC}"
echo -e "  1. Review the changes: git diff"
echo -e "  2. Run tests: ./bin/test-packages.sh"
echo -e "  3. Run full lint check: ./bin/dev.sh lint"
