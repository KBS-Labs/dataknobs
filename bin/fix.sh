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

# Source the package discovery utility
source "$ROOT_DIR/bin/package-discovery.sh"

# Default values
TARGETS=()

# Usage function
usage() {
    echo "Usage: $0 [OPTIONS] [TARGETS...]"
    echo ""
    echo "Auto-fix lint findings using ruff"
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
    echo "  -h, --help            Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                        # Fix all packages"
    echo "  $0 structures                             # Fix only structures package"
    echo "  $0 packages/utils/src                     # Fix specific directory"
    echo "  $0 packages/utils/src/dataknobs_utils/*.py  # Fix specific files"
    echo ""
    echo "Formatting is not run here and is not enforced by any check. See"
    echo "'dk format' if you want it."
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
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
# Word splitting is intended: discover_packages emits a space-separated
# list. Collected with a read loop rather than the mapfile shellcheck
# suggests, because mapfile is bash 4+ and these scripts run on the
# stock macOS bash 3.2. The loop is also glob-safe, which the bare
# arr=($(cmd)) form is not.
ALL_PACKAGES=()
while IFS= read -r _pkg; do
    [[ -n "$_pkg" ]] && ALL_PACKAGES+=("$_pkg")
done < <(discover_packages | tr ' ' '\n')

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
    # The code belonging to no package. validate.sh reports findings here now,
    # so this is where they get fixed — without it, the one entry point named
    # "fix" would be the one that cannot fix them. The two default sets are
    # still not identical: this one also covers packages/*/tests, which
    # validate.sh does not yet lint.
    for workspace_target in $(workspace_targets); do
        FIX_TARGETS+=("$workspace_target")
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
            # Unquoted on purpose: $target is a glob and this line is what
            # expands it. nullglob above turns a non-match into zero words.
            # shellcheck disable=SC2206
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
    
    # Through `uv run`: the workspace pins the ruff that produced the findings
    # being fixed here. A bare `ruff` resolves against PATH, which this
    # workspace does not populate — so this step printed "some issues remain
    # that need manual fixing", which is what ruff-ran-and-stopped-short looks
    # like, and exited without having read a file.
    #
    # --no-unsafe-fixes: no behaviour-changing rewrites without a human.
    echo -e "${BLUE}Running ruff auto-fix...${NC}"
    if uv run ruff check "$target" --fix --no-unsafe-fixes --config "$ROOT_DIR/pyproject.toml"; then
        echo -e "${GREEN}✓ Ruff auto-fix completed${NC}"
    else
        echo -e "${YELLOW}⚠ Some issues remain that need manual fixing${NC}"
    fi
done

echo -e "\n${GREEN}All fixes applied!${NC}"

# Suggest next steps. These name entry points rather than tool invocations:
# a command spelled out here is a target set and a config that nothing keeps
# in step with the gate, and validate.sh is what sent you here.
echo -e "\n${YELLOW}Next steps:${NC}"
echo -e "  1. Review the changes: git diff"
echo -e "  2. Re-run validation: ./bin/validate.sh"
echo -e "  3. Run tests: ./bin/test.sh"
