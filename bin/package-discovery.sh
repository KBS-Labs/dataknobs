#!/usr/bin/env bash
# Package discovery utility for dataknobs
# This script automatically discovers packages in the packages/ directory
# and provides functions for other scripts to use

set -euo pipefail

# Get the root directory
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Function to discover all packages
discover_packages() {
    local packages=()
    
    # Find all directories in packages/ that have a pyproject.toml
    for dir in "$ROOT_DIR"/packages/*/; do
        if [[ -f "$dir/pyproject.toml" ]]; then
            package_name=$(basename "$dir")
            packages+=("$package_name")
        fi
    done
    
    # Return sorted array
    IFS=$'\n' sorted=($(sort <<<"${packages[*]}"))
    unset IFS
    echo "${sorted[@]}"
}

# Function to get packages in dependency order
# This reads from pyproject.toml to determine dependencies
get_packages_in_order() {
    local ordered_packages=()
    local remaining_packages=($(discover_packages))
    local max_iterations=10
    local iterations=0
    
    # First, add packages with no internal dependencies
    while [[ ${#remaining_packages[@]} -gt 0 ]] && [[ $iterations -lt $max_iterations ]]; do
        local added_this_round=false
        local new_remaining=()
        
        for package in "${remaining_packages[@]}"; do
            local has_unmet_deps=false
            local pyproject="$ROOT_DIR/packages/$package/pyproject.toml"
            
            # Check if this package depends on any dataknobs packages not yet added
            if [[ -f "$pyproject" ]]; then
                # Extract dependencies (simplified - just looking for dataknobs- packages)
                local deps=$(grep -E "dataknobs-" "$pyproject" 2>/dev/null | grep -v "^name = " || true)
                
                if [[ ${#ordered_packages[@]} -gt 0 ]]; then
                    for ordered in "${ordered_packages[@]}"; do
                        # Remove already ordered packages from deps check
                        deps=$(echo "$deps" | grep -v "dataknobs-$ordered" || true)
                    done
                fi
                
                # If there are still dataknobs dependencies, this package must wait
                if echo "$deps" | grep -q "dataknobs-"; then
                    has_unmet_deps=true
                fi
            fi
            
            if [[ "$has_unmet_deps" == false ]]; then
                ordered_packages+=("$package")
                added_this_round=true
            else
                new_remaining+=("$package")
            fi
        done
        
        if [[ ${#new_remaining[@]} -gt 0 ]]; then
            remaining_packages=("${new_remaining[@]}")
        else
            remaining_packages=()
        fi
        iterations=$((iterations + 1))
        
        # If nothing was added this round and we still have packages, we have a circular dependency
        if [[ "$added_this_round" == false ]] && [[ ${#remaining_packages[@]} -gt 0 ]]; then
            echo "Warning: Possible circular dependency detected. Adding remaining packages in alphabetical order." >&2
            ordered_packages+=("${remaining_packages[@]}")
            break
        fi
    done
    
    echo "${ordered_packages[@]}"
}

# Function to check if a package exists
package_exists() {
    local package="$1"
    [[ -d "$ROOT_DIR/packages/$package" ]] && [[ -f "$ROOT_DIR/packages/$package/pyproject.toml" ]]
}

# Function to get package version
get_package_version() {
    local package="$1"
    local pyproject="$ROOT_DIR/packages/$package/pyproject.toml"
    
    if [[ -f "$pyproject" ]]; then
        grep -E "^version = " "$pyproject" | cut -d'"' -f2 || echo "unknown"
    else
        echo "unknown"
    fi
}

# If sourced with arguments, execute the requested function
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    # Script is being sourced - functions are available
    true
else
    # Script is being executed directly
    case "${1:-}" in
        list)
            discover_packages
            ;;
        ordered)
            get_packages_in_order
            ;;
        exists)
            if [[ -z "${2:-}" ]]; then
                echo "Usage: $0 exists <package-name>"
                exit 1
            fi
            if package_exists "$2"; then
                echo "yes"
                exit 0
            else
                echo "no"
                exit 1
            fi
            ;;
        version)
            if [[ -z "${2:-}" ]]; then
                echo "Usage: $0 version <package-name>"
                exit 1
            fi
            get_package_version "$2"
            ;;
        *)
            echo "Usage: $0 {list|ordered|exists <package>|version <package>}"
            echo ""
            echo "Commands:"
            echo "  list              List all discovered packages"
            echo "  ordered           List packages in dependency order"
            echo "  exists <package>  Check if a package exists"
            echo "  version <package> Get package version"
            echo ""
            echo "This script can also be sourced to use its functions:"
            echo "  source $0"
            echo "  packages=(\$(discover_packages))"
            exit 1
            ;;
    esac
fi