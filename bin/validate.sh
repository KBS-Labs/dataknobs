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
STATS=false
ALL_ERRORS=false

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
    echo "  -s, --stats           Show detailed error statistics"
    echo "  -a, --all-errors      Show all errors (bypass suppression rules)"
    echo "  -h, --help            Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                        # Validate all packages"
    echo "  $0 utils                                  # Validate only utils package"
    echo "  $0 packages/utils/src                     # Validate specific directory"
    echo "  $0 packages/utils/src/dataknobs_utils/*.py  # Validate specific files"
    echo "  $0 -f                                     # Validate and fix issues"
    echo "  $0 -s data                                # Show error statistics for data package"
    echo "  $0 -s -a data                             # Show ALL error statistics (including suppressed)"
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
        -s|--stats)
            STATS=true
            shift
            ;;
        -a|--all-errors)
            ALL_ERRORS=true
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

# Stats mode - show error statistics and exit
if [[ "$STATS" == true ]]; then
    echo -e "${BLUE}Error Statistics for targets:${NC}"
    echo -e "${YELLOW}==============================${NC}"
    
    # Ruff statistics
    echo -e "\n${BLUE}Ruff Linting Statistics:${NC}"
    for target in "${VALIDATE_TARGETS[@]}"; do
        echo -e "${YELLOW}  $target:${NC}"
        if [[ "$ALL_ERRORS" == true ]]; then
            # Show all errors without config (no suppression)
            uv run ruff check "$target" --statistics 2>/dev/null || true
        else
            # Normal mode with configured suppressions from pyproject.toml
            uv run ruff check "$target" --statistics --config "$ROOT_DIR/pyproject.toml" 2>/dev/null || true
        fi
    done
    
    # MyPy statistics
    if [[ "$QUICK" != true ]]; then
        echo -e "\n${BLUE}MyPy Type Checking Statistics:${NC}"
        for target in "${VALIDATE_TARGETS[@]}"; do
            echo -e "${YELLOW}  $target:${NC}"
            # Count errors by type
            if [[ "$ALL_ERRORS" == true ]]; then
                # Use pyproject.toml for comprehensive checking (more errors shown)
                uv run mypy "$target" --config-file "$ROOT_DIR/pyproject.toml" 2>&1 | \
                    grep "error:" | \
                    sed 's/.*error: //' | \
                    sed 's/  \[/\n[/' | \
                    grep '^\[' | \
                    sed 's/\[//' | \
                    sed 's/\]//' | \
                    sort | uniq -c | sort -rn || echo "    No type errors found"
            else
                # Use mypy.ini for focused checking (fewer errors shown)
                uv run mypy "$target" --config-file "$ROOT_DIR/mypy.ini" 2>&1 | \
                    grep "error:" | \
                    sed 's/.*error: //' | \
                    sed 's/  \[/\n[/' | \
                    grep '^\[' | \
                    sed 's/\[//' | \
                    sed 's/\]//' | \
                    sort | uniq -c | sort -rn || echo "    No type errors found"
            fi
        done
        
        # Show total mypy errors
        echo -e "\n${BLUE}Total MyPy Errors:${NC}"
        for target in "${VALIDATE_TARGETS[@]}"; do
            if [[ "$ALL_ERRORS" == true ]]; then
                ERROR_COUNT=$(uv run mypy "$target" --config-file "$ROOT_DIR/pyproject.toml" 2>&1 | grep -c "error:" || echo "0")
                echo -e "  ${YELLOW}$target:${NC} $ERROR_COUNT errors (comprehensive)"
            else
                ERROR_COUNT=$(uv run mypy "$target" --config-file "$ROOT_DIR/mypy.ini" 2>&1 | grep -c "error:" || echo "0")
                echo -e "  ${YELLOW}$target:${NC} $ERROR_COUNT errors (focused)"
            fi
        done
    fi
    
    # TODO/FIXME count
    echo -e "\n${BLUE}TODO/FIXME Comments:${NC}"
    for target in "${VALIDATE_TARGETS[@]}"; do
        if [[ -f "$target" ]]; then
            count=$(grep -c "TODO\|FIXME" "$target" 2>/dev/null || echo 0)
            echo -e "  ${YELLOW}$target:${NC} $count"
        elif [[ -d "$target" ]]; then
            # Temporarily disable pipefail since grep -c returns exit 1 when count is 0
            set +o pipefail
            count=$(find "$target" -name "*.py" -exec grep -c "TODO\|FIXME" {} + 2>/dev/null | awk -F: '{sum += $2} END {print sum ? sum : 0}')
            set -o pipefail
            echo -e "  ${YELLOW}$target:${NC} $count"
        fi
    done
    
    exit 0
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
        if ! uv run python -m py_compile "$target" 2>/dev/null; then
            echo -e "${RED}    ✗ Syntax error in $target${NC}"
            FAILED=true
        fi
    elif [[ -d "$target" ]]; then
        # Directory - find all Python files
        while IFS= read -r -d '' file; do
            if ! uv run python -m py_compile "$file" 2>/dev/null; then
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
        if [[ "$ALL_ERRORS" == true ]]; then
            # No config = show all errors
            if uv run ruff check "$target" --fix --no-unsafe-fixes; then
                echo -e "${GREEN}    ✓ Ruff checks passed${NC}"
            else
                echo -e "${YELLOW}    ⚠ Some issues remain that need manual fixing${NC}"
                FAILED=true
            fi
        else
            # Use config for suppressions
            if uv run ruff check "$target" --fix --no-unsafe-fixes --config "$ROOT_DIR/pyproject.toml"; then
                echo -e "${GREEN}    ✓ Ruff checks passed${NC}"
            else
                echo -e "${YELLOW}    ⚠ Some issues remain that need manual fixing${NC}"
                FAILED=true
            fi
        fi
    else
        # Run ruff without fixing
        if [[ "$ALL_ERRORS" == true ]]; then
            # No config = show all errors
            if uv run ruff check "$target" --no-fix; then
                echo -e "${GREEN}    ✓ Ruff checks passed${NC}"
            else
                echo -e "${RED}    ✗ Ruff found issues${NC}"
                FAILED=true
            fi
        else
            # Use config for suppressions
            if uv run ruff check "$target" --no-fix --config "$ROOT_DIR/pyproject.toml"; then
                echo -e "${GREEN}    ✓ Ruff checks passed${NC}"
            else
                echo -e "${RED}    ✗ Ruff found issues${NC}"
                FAILED=true
            fi
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
        if uv run python -c "import $PACKAGE_NAME" 2>/dev/null; then
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
        
        # Choose config based on all-errors flag
        if [[ "$ALL_ERRORS" == true ]]; then
            MYPY_CONFIG="$ROOT_DIR/pyproject.toml"
        else
            MYPY_CONFIG="$ROOT_DIR/mypy.ini"
        fi
        
        # For individual files, skip following imports to avoid checking the whole codebase
        if [[ -f "$target" ]]; then
            # Single file - don't follow imports
            if uv run mypy "$target" --config-file "$MYPY_CONFIG" --follow-imports=skip 2>&1 | grep -E "(error|Error)"; then
                echo -e "${RED}    ✗ Type errors found${NC}"
                FAILED=true
            else
                echo -e "${GREEN}    ✓ Type checks passed${NC}"
            fi
        else
            # Directory or package - normal behavior
            if uv run mypy "$target" --config-file "$MYPY_CONFIG" 2>&1 | grep -E "(error|Error)"; then
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

# Check for print statements (with exceptions for legitimate uses)
echo -e "${YELLOW}  Checking for print statements...${NC}"
HAS_PRINTS=false
PRINT_FILES=()

# Files that are allowed to have print statements
# CLI tools and debuggers need to output to users
PRINT_EXCEPTIONS=(
    "*/cli/main.py"       # CLI interface uses Rich console.print
    "*/api/advanced.py"   # Debugger class needs user output
)

# Function to check if a file should be excluded
should_exclude_file() {
    local file="$1"
    for exception in "${PRINT_EXCEPTIONS[@]}"; do
        if [[ "$file" == $exception ]]; then
            return 0  # Should exclude (true)
        fi
    done
    return 1  # Should not exclude (false)
}

for target in "${VALIDATE_TARGETS[@]}"; do
    if [[ -f "$target" ]]; then
        # Single file
        if [[ "$(basename "$target")" != "__init__.py" ]] && \
           [[ "$target" != *test* ]] && \
           ! should_exclude_file "$target" && \
           grep -q "print(" "$target"; then
            PRINT_FILES+=("$target")
            HAS_PRINTS=true
        fi
    elif [[ -d "$target" ]]; then
        # Directory - find all files with print statements
        while IFS= read -r file; do
            # Check if this file should be excluded
            if ! should_exclude_file "$file"; then
                PRINT_FILES+=("$file")
                HAS_PRINTS=true
            fi
        done < <(find "$target" -name "*.py" ! -name "__init__.py" ! -path "*/test*" -exec grep -l "print(" {} \; 2>/dev/null)
    fi
done

if [[ "$HAS_PRINTS" == false ]]; then
    echo -e "${GREEN}    ✓ No print statements found${NC}"
else
    echo -e "${RED}    ✗ Found print statements (use logging instead):${NC}"
    # Show up to 10 files with print statements
    shown=0
    for file in "${PRINT_FILES[@]}"; do
        if [[ $shown -lt 10 ]]; then
            # Show the file and line numbers where print statements appear
            echo -e "${RED}      - $file:${NC}"
            grep -n "print(" "$file" | head -3 | while IFS=: read -r line_num line_content; do
                # Trim whitespace and show a preview
                trimmed=$(echo "$line_content" | sed 's/^[[:space:]]*//' | cut -c1-60)
                echo -e "${RED}        Line $line_num: $trimmed${NC}"
            done
            ((shown++))
        fi
    done
    if [[ ${#PRINT_FILES[@]} -gt 10 ]]; then
        echo -e "${RED}      ... and $((${#PRINT_FILES[@]} - 10)) more files${NC}"
    fi
    FAILED=true
fi

# Check for TODO/FIXME comments
echo -e "${YELLOW}  Checking for TODO/FIXME comments...${NC}"
TODO_COUNT=0
TODO_FILES=()

for target in "${VALIDATE_TARGETS[@]}"; do
    if [[ -f "$target" ]]; then
        # grep -c returns exit 1 when count is 0, so we use || echo 0
        count=$(grep -c "TODO\|FIXME" "$target" 2>/dev/null || echo 0)
        if [[ $count -gt 0 ]]; then
            TODO_FILES+=("$target:$count")
            TODO_COUNT=$((TODO_COUNT + count))
        fi
    elif [[ -d "$target" ]]; then
        # Find files with TODO/FIXME and their counts
        while IFS=: read -r file count; do
            if [[ -n "$file" ]] && [[ "$count" -gt 0 ]]; then
                TODO_FILES+=("$file:$count")
                TODO_COUNT=$((TODO_COUNT + count))
            fi
        done < <(find "$target" -name "*.py" -exec grep -c "TODO\|FIXME" {} + 2>/dev/null | grep -v ":0$")
    fi
done

if [[ "$TODO_COUNT" -eq 0 ]]; then
    echo -e "${GREEN}    ✓ No TODO/FIXME comments found${NC}"
elif [[ "$TODO_COUNT" -gt 0 ]]; then
    echo -e "${YELLOW}    ⚠ Found $TODO_COUNT TODO/FIXME comments:${NC}"
    # Show up to 10 files with TODO/FIXME
    shown=0
    for file_info in "${TODO_FILES[@]}"; do
        if [[ $shown -lt 10 ]]; then
            file="${file_info%:*}"
            count="${file_info##*:}"
            echo -e "${YELLOW}      - $file ($count occurrences):${NC}"
            # Show first 3 TODO/FIXME comments with line numbers
            grep -n "TODO\|FIXME" "$file" | head -3 | while IFS=: read -r line_num line_content; do
                # Trim whitespace and show a preview
                trimmed=$(echo "$line_content" | sed 's/^[[:space:]]*//' | cut -c1-60)
                echo -e "${YELLOW}        Line $line_num: $trimmed${NC}"
            done
            ((shown++))
        fi
    done
    if [[ ${#TODO_FILES[@]} -gt 10 ]]; then
        echo -e "${YELLOW}      ... and $((${#TODO_FILES[@]} - 10)) more files${NC}"
    fi
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
