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

# Source the package discovery utility
source "$ROOT_DIR/bin/package-discovery.sh"

# Default values
TARGETS=()
QUICK=false
FIX=false

# Usage function
usage() {
    echo "Usage: $0 [OPTIONS] [TARGETS...]"
    echo ""
    echo "Validate code quality and catch common errors"
    echo ""
    echo "Arguments:"
    echo "  TARGETS               Packages, directories, or files to validate"
    echo "                        Can be:"
    echo "                        - Package name (e.g., 'common', 'utils')"
    echo "                        - Directory path (e.g., 'packages/utils/src')"
    echo "                        - File path (e.g., 'packages/utils/src/dataknobs_utils/file_utils.py')"
    echo "                        If not specified, validates all packages"
    echo ""
    echo "Options:"
    echo "  -q, --quick           Quick validation (skip slow checks)"
    echo "  -f, --fix             Attempt to auto-fix issues"
    echo "  -h, --help            Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                        # Validate all packages"
    echo "  $0 utils                                  # Validate only utils package"
    echo "  $0 packages/utils/src                     # Validate specific directory"
    echo "  $0 packages/utils/src/dataknobs_utils/*.py  # Validate specific files"
    echo "  $0 -f                                     # Validate and fix issues"
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
            # Add to targets list
            TARGETS+=("$1")
            shift
            ;;
    esac
done

# Get all packages dynamically
ALL_PACKAGES=($(discover_packages))

# Determine what to validate
VALIDATE_TARGETS=()
VALIDATE_PACKAGES=()

if [[ ${#TARGETS[@]} -eq 0 ]]; then
    # No targets specified, validate all packages
    VALIDATE_PACKAGES=("${ALL_PACKAGES[@]}")
    for package in "${ALL_PACKAGES[@]}"; do
        if [[ -d "packages/$package/src" ]]; then
            VALIDATE_TARGETS+=("packages/$package/src")
        fi
    done
else
    # Process specified targets
    for target in "${TARGETS[@]}"; do
        if [[ -d "packages/$target" ]]; then
            # It's a package name
            VALIDATE_PACKAGES+=("$target")
            if [[ -d "packages/$target/src" ]]; then
                VALIDATE_TARGETS+=("packages/$target/src")
            fi
        elif [[ -d "$target" ]]; then
            # It's a directory
            VALIDATE_TARGETS+=("$target")
        elif [[ -f "$target" ]]; then
            # It's a file
            VALIDATE_TARGETS+=("$target")
        else
            # Try glob expansion
            shopt -s nullglob
            files=($target)
            shopt -u nullglob
            if [[ ${#files[@]} -gt 0 ]]; then
                VALIDATE_TARGETS+=("${files[@]}")
            else
                echo -e "${YELLOW}Warning: Target '$target' not found${NC}"
            fi
        fi
    done
fi

if [[ ${#VALIDATE_TARGETS[@]} -eq 0 ]]; then
    echo -e "${RED}No valid targets found to validate${NC}"
    exit 1
fi

echo -e "${YELLOW}Validating targets...${NC}"

# Track overall status
FAILED=false

# 1. Check Python syntax
echo -e "\n${BLUE}1. Checking Python syntax...${NC}"
for target in "${VALIDATE_TARGETS[@]}"; do
    echo -e "${YELLOW}  Checking $target...${NC}"
    
    if [[ -f "$target" ]]; then
        # Single file
        if ! python -m py_compile "$target" 2>/dev/null; then
            echo -e "${RED}    ✗ Syntax error in $target${NC}"
            FAILED=true
        fi
    elif [[ -d "$target" ]]; then
        # Directory - find all Python files
        while IFS= read -r -d '' file; do
            if ! python -m py_compile "$file" 2>/dev/null; then
                echo -e "${RED}    ✗ Syntax error in $file${NC}"
                FAILED=true
            fi
        done < <(find "$target" -name "*.py" -print0)
    fi
done

if [[ "$FAILED" == false ]]; then
    echo -e "${GREEN}  ✓ All Python files have valid syntax${NC}"
fi

# 2. Run ruff linting
echo -e "\n${BLUE}2. Running ruff linting...${NC}"

for target in "${VALIDATE_TARGETS[@]}"; do
    echo -e "${YELLOW}  Checking $target...${NC}"
    
    if [[ "$FIX" == true ]]; then
        # Run ruff with auto-fix (matching fix.sh behavior)
        if ruff check "$target" --fix --no-unsafe-fixes --config "$ROOT_DIR/pyproject.toml"; then
            echo -e "${GREEN}    ✓ Ruff checks passed${NC}"
        else
            echo -e "${YELLOW}    ⚠ Some issues remain that need manual fixing${NC}"
            FAILED=true
        fi
    else
        # Run ruff without fixing
        if ruff check "$target" --no-fix --config "$ROOT_DIR/pyproject.toml"; then
            echo -e "${GREEN}    ✓ Ruff checks passed${NC}"
        else
            echo -e "${RED}    ✗ Ruff found issues${NC}"
            FAILED=true
        fi
    fi
done

# 3. Check imports (only for packages)
if [[ ${#VALIDATE_PACKAGES[@]} -gt 0 ]]; then
    echo -e "\n${BLUE}3. Checking imports...${NC}"
    for package in "${VALIDATE_PACKAGES[@]}"; do
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
fi

# 4. Type checking with mypy (unless quick mode)
if [[ "$QUICK" != true ]]; then
    echo -e "\n${BLUE}4. Running mypy type checking...${NC}"
    for target in "${VALIDATE_TARGETS[@]}"; do
        echo -e "${YELLOW}  Checking $target...${NC}"
        
        # For individual files, skip following imports to avoid checking the whole codebase
        if [[ -f "$target" ]]; then
            # Single file - don't follow imports
            if mypy "$target" --config-file "$ROOT_DIR/pyproject.toml" --follow-imports=skip 2>&1 | grep -E "(error|Error)"; then
                echo -e "${RED}    ✗ Type errors found${NC}"
                FAILED=true
            else
                echo -e "${GREEN}    ✓ Type checks passed${NC}"
            fi
        else
            # Directory or package - normal behavior
            if mypy "$target" --config-file "$ROOT_DIR/pyproject.toml" 2>&1 | grep -E "(error|Error)"; then
                echo -e "${RED}    ✗ Type errors found${NC}"
                FAILED=true
            else
                echo -e "${GREEN}    ✓ Type checks passed${NC}"
            fi
        fi
    done
fi

# 5. Check for common issues
echo -e "\n${BLUE}5. Checking for common issues...${NC}"

# Check for print statements (except in __init__.py)
echo -e "${YELLOW}  Checking for print statements...${NC}"
HAS_PRINTS=false
for target in "${VALIDATE_TARGETS[@]}"; do
    if [[ -f "$target" ]]; then
        # Single file
        if [[ "$(basename "$target")" != "__init__.py" ]] && grep -q "print(" "$target"; then
            echo -e "${RED}    ✗ Found print statement in $target${NC}"
            HAS_PRINTS=true
        fi
    elif [[ -d "$target" ]]; then
        # Directory
        if find "$target" -name "*.py" ! -name "__init__.py" -exec grep -l "print(" {} \; | grep -v test | head -n 1 > /dev/null; then
            echo -e "${RED}    ✗ Found print statements in $target (use logging instead)${NC}"
            HAS_PRINTS=true
        fi
    fi
done

if [[ "$HAS_PRINTS" == false ]]; then
    echo -e "${GREEN}    ✓ No print statements found${NC}"
else
    FAILED=true
fi

# Check for TODO/FIXME comments
echo -e "${YELLOW}  Checking for TODO/FIXME comments...${NC}"
TODO_COUNT=0
for target in "${VALIDATE_TARGETS[@]}"; do
    if [[ -f "$target" ]]; then
        count=$(grep -c "TODO\|FIXME" "$target" 2>/dev/null || echo 0)
        TODO_COUNT=$((TODO_COUNT + count))
    elif [[ -d "$target" ]]; then
        count=$(find "$target" -name "*.py" -exec grep -c "TODO\|FIXME" {} + 2>/dev/null | awk -F: '{sum += $2} END {print sum}' || echo 0)
        TODO_COUNT=$((TODO_COUNT + count))
    fi
done

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
