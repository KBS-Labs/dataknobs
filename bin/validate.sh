#!/usr/bin/env bash
# Validate code before commits or releases

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
QUICK=false
FIX=false

# Usage function
usage() {
    echo "Usage: $0 [OPTIONS] [PACKAGE]"
    echo ""
    echo "Validate code quality and catch common errors"
    echo ""
    echo "Arguments:"
    echo "  PACKAGE               Specific package to validate"
    echo "                        If not specified, validates all packages"
    echo ""
    echo "Options:"
    echo "  -q, --quick           Quick validation (skip slow checks)"
    echo "  -f, --fix             Attempt to auto-fix issues"
    echo "  -h, --help            Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                    # Validate all packages"
    echo "  $0 utils              # Validate only utils package"
    echo "  $0 -f                 # Validate and fix issues"
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -q|--quick)
            QUICK=true
            shift
            ;;
        -f|--fix)
            FIX=true
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

# Determine which packages to validate
if [[ -n "$PACKAGE" ]]; then
    if [[ ! -d "packages/$PACKAGE" ]]; then
        echo -e "${RED}Error: Package '$PACKAGE' not found${NC}"
        exit 1
    fi
    PACKAGES=("$PACKAGE")
else
    PACKAGES=("${ALL_PACKAGES[@]}")
fi

echo -e "${YELLOW}Validating dataknobs packages...${NC}"

# Track overall status
FAILED=false

# 1. Check Python syntax
echo -e "\n${BLUE}1. Checking Python syntax...${NC}"
for package in "${PACKAGES[@]}"; do
    echo -e "${YELLOW}  Checking $package...${NC}"
    
    # Find all Python files
    while IFS= read -r -d '' file; do
        if ! python -m py_compile "$file" 2>/dev/null; then
            echo -e "${RED}    ✗ Syntax error in $file${NC}"
            FAILED=true
        fi
    done < <(find "packages/$package/src" -name "*.py" -print0)
done

if [[ "$FAILED" == false ]]; then
    echo -e "${GREEN}  ✓ All Python files have valid syntax${NC}"
fi

# 2. Run ruff linting
echo -e "\n${BLUE}2. Running ruff linting...${NC}"
RUFF_CMD="ruff check"
if [[ "$FIX" == true ]]; then
    RUFF_CMD="$RUFF_CMD --fix"
fi

for package in "${PACKAGES[@]}"; do
    echo -e "${YELLOW}  Checking $package...${NC}"
    
    if ! $RUFF_CMD "packages/$package/src" --config "$ROOT_DIR/pyproject.toml" 2>&1 | grep -E "(error|Error)"; then
        echo -e "${GREEN}    ✓ Ruff checks passed${NC}"
    else
        echo -e "${RED}    ✗ Ruff found issues${NC}"
        FAILED=true
    fi
done

# 3. Check imports
echo -e "\n${BLUE}3. Checking imports...${NC}"
for package in "${PACKAGES[@]}"; do
    echo -e "${YELLOW}  Checking $package...${NC}"
    
    # Try to import the package
    PACKAGE_NAME="dataknobs_${package//-/_}"
    if python -c "import $PACKAGE_NAME" 2>/dev/null; then
        echo -e "${GREEN}    ✓ Package imports successfully${NC}"
    else
        echo -e "${RED}    ✗ Failed to import $PACKAGE_NAME${NC}"
        FAILED=true
    fi
done

# 4. Type checking with mypy (unless quick mode)
if [[ "$QUICK" != true ]]; then
    echo -e "\n${BLUE}4. Running mypy type checking...${NC}"
    for package in "${PACKAGES[@]}"; do
        echo -e "${YELLOW}  Checking $package...${NC}"
        
        if mypy "packages/$package/src" --config-file "$ROOT_DIR/pyproject.toml" 2>&1 | grep -E "(error|Error)"; then
            echo -e "${RED}    ✗ Type errors found${NC}"
            FAILED=true
        else
            echo -e "${GREEN}    ✓ Type checks passed${NC}"
        fi
    done
fi

# 5. Check for common issues
echo -e "\n${BLUE}5. Checking for common issues...${NC}"

# Check for print statements (except in __init__.py)
echo -e "${YELLOW}  Checking for print statements...${NC}"
if find packages/*/src -name "*.py" ! -name "__init__.py" -exec grep -l "print(" {} \; | grep -v test; then
    echo -e "${RED}    ✗ Found print statements (use logging instead)${NC}"
    FAILED=true
else
    echo -e "${GREEN}    ✓ No print statements found${NC}"
fi

# Check for TODO/FIXME comments
echo -e "${YELLOW}  Checking for TODO/FIXME comments...${NC}"
TODO_COUNT=$(find packages/*/src -name "*.py" -exec grep -c "TODO\|FIXME" {} + | awk -F: '{sum += $2} END {print sum}')
if [[ "$TODO_COUNT" -gt 0 ]]; then
    echo -e "${YELLOW}    ⚠ Found $TODO_COUNT TODO/FIXME comments${NC}"
fi

# Summary
echo -e "\n${YELLOW}Validation Summary:${NC}"
echo -e "${YELLOW}==================${NC}"

if [[ "$FAILED" == true ]]; then
    echo -e "${RED}❌ Validation failed!${NC}"
    echo -e "\nTo fix issues:"
    echo -e "  1. Run: ./bin/validate.sh -f"
    echo -e "  2. Run: ./bin/fix.sh"
    echo -e "  3. Fix remaining issues manually"
    exit 1
else
    echo -e "${GREEN}✅ All validations passed!${NC}"
    exit 0
fi