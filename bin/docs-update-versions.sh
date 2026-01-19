#!/usr/bin/env bash
# Update documentation version table from packages.json
# This script reads versions from .dataknobs/packages.json and updates
# the version table in docs/index.md to keep them in sync.

set -euo pipefail

# Get the root directory
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Files
PACKAGES_JSON=".dataknobs/packages.json"
DOCS_INDEX="docs/index.md"

# Check mode (--check flag)
CHECK_MODE=false
if [[ "${1:-}" == "--check" ]]; then
    CHECK_MODE=true
fi

# Verify required files exist
if [[ ! -f "$PACKAGES_JSON" ]]; then
    echo -e "${RED}Error: $PACKAGES_JSON not found${NC}" >&2
    exit 1
fi

if [[ ! -f "$DOCS_INDEX" ]]; then
    echo -e "${RED}Error: $DOCS_INDEX not found${NC}" >&2
    exit 1
fi

# Build the expected version table content from packages.json
# Uses jq to extract package info and format as markdown table rows
build_version_table() {
    # Order of packages in documentation (matches current docs/index.md order)
    local pkg_order=("config" "data" "fsm" "llm" "bots" "structures" "utils" "xization" "common" "legacy")

    for pkg in "${pkg_order[@]}"; do
        local pypi_name version description link

        # Extract package info from JSON
        pypi_name=$(jq -r ".packages[] | select(.name == \"$pkg\") | .pypi_name" "$PACKAGES_JSON")
        version=$(jq -r ".packages[] | select(.name == \"$pkg\") | .version" "$PACKAGES_JSON")
        description=$(jq -r ".packages[] | select(.name == \"$pkg\") | .description" "$PACKAGES_JSON")

        # Skip if package not found
        if [[ -z "$pypi_name" || "$pypi_name" == "null" ]]; then
            continue
        fi

        # Build link path
        link="packages/$pkg/index.md"

        # Special handling for legacy package description
        if [[ "$pkg" == "legacy" ]]; then
            description="Legacy compatibility package (deprecated)"
        fi

        # Output table row
        echo "| [$pypi_name]($link) | $description | $version |"
    done
}

# Extract current version table from docs/index.md
extract_current_table() {
    # Extract lines between the table header and the next section
    # The table starts with "| Package |" and ends before "## Quick Installation"
    awk '/^\| Package \| Description \| Version \|$/,/^$/ { if (/^\|/) print }' "$DOCS_INDEX" | tail -n +3
}

# Update docs/index.md with new version table
update_docs() {
    local table_file="$1"
    local temp_file
    temp_file=$(mktemp)

    # Process the file using awk and a table file
    awk -v table_file="$table_file" '
        BEGIN { in_table = 0; header_done = 0 }
        /^\| Package \| Description \| Version \|$/ {
            print
            in_table = 1
            next
        }
        in_table == 1 && /^\|[-]+\|/ {
            print
            header_done = 1
            # Read and output the new table from file
            while ((getline line < table_file) > 0) {
                print line
            }
            close(table_file)
            next
        }
        in_table == 1 && header_done == 1 && /^\|/ {
            next  # Skip old data rows
        }
        in_table == 1 && !/^\|/ {
            in_table = 0
            header_done = 0
            print
            next
        }
        { print }
    ' "$DOCS_INDEX" > "$temp_file"

    mv "$temp_file" "$DOCS_INDEX"
}

# Main execution
main() {
    echo -e "${CYAN}Documentation Version Sync${NC}"
    echo ""

    # Build expected table to temp file
    local expected_table_file
    expected_table_file=$(mktemp)
    build_version_table > "$expected_table_file"

    # Extract current table to temp file
    local current_table_file
    current_table_file=$(mktemp)
    extract_current_table > "$current_table_file"

    # Compare tables
    if diff -q "$expected_table_file" "$current_table_file" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Documentation versions are in sync${NC}"
        rm -f "$expected_table_file" "$current_table_file"
        exit 0
    fi

    if $CHECK_MODE; then
        echo -e "${RED}✗ Documentation versions are out of sync${NC}"
        echo ""
        echo -e "${YELLOW}Expected (from packages.json):${NC}"
        cat "$expected_table_file"
        echo ""
        echo -e "${YELLOW}Current (in docs/index.md):${NC}"
        cat "$current_table_file"
        echo ""
        echo -e "${CYAN}Run 'bin/docs-update-versions.sh' to update documentation${NC}"
        rm -f "$expected_table_file" "$current_table_file"
        exit 1
    fi

    # Update mode
    echo -e "${YELLOW}Updating docs/index.md with current versions...${NC}"
    update_docs "$expected_table_file"
    echo -e "${GREEN}✓ Documentation updated successfully${NC}"

    # Show what changed
    echo ""
    echo -e "${CYAN}Updated version table:${NC}"
    cat "$expected_table_file"

    rm -f "$expected_table_file" "$current_table_file"
}

main "$@"
