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
    # A read loop, not the mapfile shellcheck suggests: mapfile is bash 4+
    # and these scripts run on the stock macOS bash 3.2.
    local sorted=() name
    while IFS= read -r name; do
        [[ -n "$name" ]] && sorted+=("$name")
    done < <(printf '%s\n' "${packages[@]:-}" | sort)
    echo "${sorted[@]:-}"
}

# Function to list the first-party code that belongs to no package
#
# The loop above answers "which packages are there", and every caller that
# wanted "which code do we check" had to answer the rest for itself. There was
# only one such directory for a long time, so each caller wrote it out and the
# copies agreed by accident. They stopped: bin/ — the scripts deciding whether a
# pull request passes — was in none of them, and adding it to one would have
# left the others behind.
#
# Emitted as a plain word list, in the shape callers already build their target
# arrays from. Order is stable and deliberate: tests/ first, as the widest and
# the one that was already here.
workspace_targets() {
    local targets=()

    [[ -d "$ROOT_DIR/tests" ]] && targets+=("tests")
    [[ -d "$ROOT_DIR/bin" ]] && targets+=("bin")
    [[ -d "$ROOT_DIR/src" ]] && targets+=("src")
    [[ -f "$ROOT_DIR/conftest.py" ]] && targets+=("conftest.py")

    # An empty result is legitimate — a checkout with none of these — and must
    # not look like failure to a caller running under `set -e`.
    echo "${targets[@]:-}"
}

# Function to get packages in dependency order
# This reads from pyproject.toml to determine dependencies
get_packages_in_order() {
    local ordered_packages=()
    local remaining_packages=() _pkg
    # Declared on its own line before the assignment: `local x=$(cmd)` reports
    # `local`'s status rather than the command's (SC2155), and the process
    # substitution this replaces reported no status at all. Either way a failing
    # discovery came back as an empty list, and this function's contract -- the
    # dependency order every caller installs in -- has no way to tell that from
    # a workspace with no packages.
    local discovered
    discovered=$(discover_packages)
    while IFS= read -r _pkg; do
        [[ -n "$_pkg" ]] && remaining_packages+=("$_pkg")
    done <<< "${discovered// /$'\n'}"
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
                # Exclude self-references (package referring to itself with extras like dataknobs-foo[extra])
                local deps
                deps=$(grep -E "dataknobs-" "$pyproject" 2>/dev/null | \
                            grep -v "^name = " | \
                            grep -v "dataknobs-$package\[" | \
                            grep -v "dataknobs-$package\"" | \
                            grep -v "dataknobs-$package " || true)
                
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
        workspace-targets)
            workspace_targets
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
            echo "Usage: $0 {list|ordered|workspace-targets|exists <package>|version <package>}"
            echo ""
            echo "Commands:"
            echo "  list              List all discovered packages"
            echo "  ordered           List packages in dependency order"
            echo "  workspace-targets List the first-party code belonging to no package"
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