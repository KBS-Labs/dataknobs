#!/usr/bin/env bash
# Update documentation versions from packages.json
# This script reads versions from .dataknobs/packages.json and updates:
# 1. The version table in docs/index.md
# 2. The requirements.txt example in docs/installation.md

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
DOCS_INSTALL="docs/installation.md"

# Check mode (--check flag)
CHECK_MODE=false
if [[ "${1:-}" == "--check" ]]; then
    CHECK_MODE=true
fi

# Track if any updates are needed
UPDATES_NEEDED=false

# Verify required files exist
if [[ ! -f "$PACKAGES_JSON" ]]; then
    echo -e "${RED}Error: $PACKAGES_JSON not found${NC}" >&2
    exit 1
fi

if [[ ! -f "$DOCS_INDEX" ]]; then
    echo -e "${RED}Error: $DOCS_INDEX not found${NC}" >&2
    exit 1
fi

if [[ ! -f "$DOCS_INSTALL" ]]; then
    echo -e "${RED}Error: $DOCS_INSTALL not found${NC}" >&2
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
update_docs_index() {
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

# Get version for a package from packages.json
get_package_version() {
    local pkg_name="$1"
    jq -r ".packages[] | select(.pypi_name == \"$pkg_name\") | .version" "$PACKAGES_JSON"
}

# Update version references in installation.md
# Updates lines matching pattern: dataknobs-<name>>=X.Y.Z
update_installation_versions() {
    local temp_file
    temp_file=$(mktemp)

    # Process the file, updating version numbers for dataknobs packages
    while IFS= read -r line; do
        if [[ "$line" =~ ^dataknobs-([a-z]+)\>=([0-9]+\.[0-9]+\.[0-9]+)$ ]]; then
            local pkg_name="dataknobs-${BASH_REMATCH[1]}"
            local current_version
            current_version=$(get_package_version "$pkg_name")
            if [[ -n "$current_version" && "$current_version" != "null" ]]; then
                echo "${pkg_name}>=${current_version}"
            else
                echo "$line"
            fi
        else
            echo "$line"
        fi
    done < "$DOCS_INSTALL" > "$temp_file"

    mv "$temp_file" "$DOCS_INSTALL"
}

# Check if installation.md versions are in sync
check_installation_versions() {
    local out_of_sync=false
    local pkg_order=("config" "data" "fsm" "llm" "bots" "structures" "utils" "xization")

    for pkg in "${pkg_order[@]}"; do
        local pypi_name="dataknobs-$pkg"
        local expected_version
        expected_version=$(get_package_version "$pypi_name")

        if [[ -z "$expected_version" || "$expected_version" == "null" ]]; then
            continue
        fi

        # Check if installation.md has the correct version
        if ! grep -q "^${pypi_name}>=${expected_version}$" "$DOCS_INSTALL"; then
            out_of_sync=true
            if $CHECK_MODE; then
                local current_line
                current_line=$(grep "^${pypi_name}>=" "$DOCS_INSTALL" 2>/dev/null || echo "not found")
                echo -e "  ${YELLOW}${pypi_name}${NC}: expected >=${expected_version}, found: ${current_line}"
            fi
        fi
    done

    if $out_of_sync; then
        return 1
    fi
    return 0
}

# Main execution
main() {
    echo -e "${CYAN}Documentation Version Sync${NC}"
    echo ""

    local has_errors=false

    # ===========================================
    # Check 1: docs/index.md version table
    # ===========================================
    echo -e "${CYAN}Checking docs/index.md version table...${NC}"

    # Build expected table to temp file
    local expected_table_file
    expected_table_file=$(mktemp)
    build_version_table > "$expected_table_file"

    # Extract current table to temp file
    local current_table_file
    current_table_file=$(mktemp)
    extract_current_table > "$current_table_file"

    # Compare tables
    local index_in_sync=true
    if ! diff -q "$expected_table_file" "$current_table_file" > /dev/null 2>&1; then
        index_in_sync=false
        has_errors=true
    fi

    if $index_in_sync; then
        echo -e "${GREEN}  ✓ docs/index.md versions are in sync${NC}"
    else
        if $CHECK_MODE; then
            echo -e "${RED}  ✗ docs/index.md versions are out of sync${NC}"
            echo ""
            echo -e "${YELLOW}  Expected (from packages.json):${NC}"
            cat "$expected_table_file" | sed 's/^/    /'
            echo ""
            echo -e "${YELLOW}  Current (in docs/index.md):${NC}"
            cat "$current_table_file" | sed 's/^/    /'
        else
            echo -e "${YELLOW}  Updating docs/index.md with current versions...${NC}"
            update_docs_index "$expected_table_file"
            echo -e "${GREEN}  ✓ docs/index.md updated successfully${NC}"
        fi
    fi

    echo ""

    # ===========================================
    # Check 2: docs/installation.md requirements
    # ===========================================
    echo -e "${CYAN}Checking docs/installation.md requirements...${NC}"

    local install_in_sync=true
    if ! check_installation_versions 2>/dev/null; then
        install_in_sync=false
        has_errors=true
    fi

    if $install_in_sync; then
        echo -e "${GREEN}  ✓ docs/installation.md versions are in sync${NC}"
    else
        if $CHECK_MODE; then
            echo -e "${RED}  ✗ docs/installation.md versions are out of sync${NC}"
            check_installation_versions  # Print details
        else
            echo -e "${YELLOW}  Updating docs/installation.md with current versions...${NC}"
            update_installation_versions
            echo -e "${GREEN}  ✓ docs/installation.md updated successfully${NC}"
        fi
    fi

    # Cleanup
    rm -f "$expected_table_file" "$current_table_file"

    echo ""

    # Final status
    if $CHECK_MODE && $has_errors; then
        echo -e "${RED}✗ Documentation versions are out of sync${NC}"
        echo -e "${CYAN}Run 'bin/docs-update-versions.sh' to update documentation${NC}"
        exit 1
    fi

    if ! $has_errors; then
        echo -e "${GREEN}✓ All documentation versions are in sync${NC}"
    else
        echo -e "${GREEN}✓ Documentation updated successfully${NC}"
    fi
}

main "$@"
