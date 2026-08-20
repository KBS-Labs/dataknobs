#!/usr/bin/env bash
# Package discovery utility for dataknobs
# This script automatically discovers packages in the packages/ directory
# and provides functions for other scripts to use

set -euo pipefail

# Get the root directory
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Function to discover all packages
discover_packages() {
    # `dir` and `package_name` declared here rather than left to the loop: this
    # file is sourced, so a loop variable that is not local is an assignment in
    # whatever shell sourced it. `for package in ...` is what nine of the
    # callers do, and `dir` is as common a name as shell has.
    local packages=() dir package_name

    # Find all directories in packages/ that have a pyproject.toml
    #
    # `|| return 1` rather than the inherited `set -e`, on every command here
    # that can fail. Callers capture this function through `$(...)` -- the only
    # way to read its output -- and bash unsets errexit inside a command
    # substitution, so a command failing in here aborts nothing. Execution
    # reaches the echo below, which succeeds, and the caller is handed exit 0
    # and an empty list. The callers were taught to examine a status; this is
    # the half that produces one.
    for dir in "$ROOT_DIR"/packages/*/; do
        if [[ -f "$dir/pyproject.toml" ]]; then
            package_name=$(basename "$dir") || return 1
            packages+=("$package_name")
        fi
    done

    # Return sorted array
    # A read loop, not the mapfile shellcheck suggests: mapfile is bash 4+
    # and these scripts run on the stock macOS bash 3.2.
    #
    # Sorted through a captured assignment rather than `done < <(...)`, for the
    # same reason the callers stopped using one: a process substitution reports
    # no status anywhere, so a failing sort arrived as a short list. That was
    # this defect one level down, inside the function the callers were fixed to
    # interrogate.
    local sorted_output
    sorted_output=$(printf '%s\n' "${packages[@]:-}" | sort) || return 1

    local sorted=() name
    while IFS= read -r name; do
        [[ -n "$name" ]] && sorted+=("$name")
    done <<< "$sorted_output"
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
    # The purpose-built cell. Tracked first-party Python like the three above,
    # and linted for the same reason -- but it is here specifically because half
    # of what it is for is that the gate finds nothing in it. It is dirty only
    # under quality-fixture/ruff.toml, which nothing but the guards over
    # bin/quality-contract.py ever runs; dropping it from this list would make
    # the clean half of that claim a measurement of nothing.
    [[ -d "$ROOT_DIR/quality-fixture" ]] && targets+=("quality-fixture")
    [[ -f "$ROOT_DIR/conftest.py" ]] && targets+=("conftest.py")

    # An empty result is legitimate — a checkout with none of these — and must
    # not look like failure to a caller running under `set -e`.
    echo "${targets[@]:-}"
}

# Function to list the packages whose tests/ the linter has been promoted onto
#
# The second of a promotion's two declarations. The first is the quality
# contract moving that package's `packages/<pkg>/tests` cell from the deferred
# tier to `checked` with a ceiling of 0; this one is what makes the ceiling a
# measurement of something, by putting the directory in front of the linter.
#
# Add a name here in the SAME change that moves the cell. Not before — the
# contract would then defer a directory the linter reads, so a finding arriving
# there is counted against a backlog that is supposed to be shrinking. Not
# after — a ceiling of zero over files nothing opens, which is the shape 2c
# shipped and the shape this whole program exists to make impossible. Both
# directions fail test_a_checked_cell_is_one_the_linter_actually_reaches, which
# compares the two declarations against each other rather than trusting either.
#
# Deliberately NOT derived from the contract. A target set read out of the file
# whose ceilings it is supposed to justify makes both coverage guards
# tautologies — they would compare a thing against itself and pass by
# construction. Two declarations that can disagree are the point.
#
# bin/fix.sh needs no equivalent: it already reaches every packages/*/tests,
# which is why a promoted cell arrives with its remedy already in place.
lint_promoted_test_packages() {
    echo "fsm legacy config structures utils xization llm common bots data"
}

# Function to list the per-package directories the linter reaches beyond src
#
# The sibling of lint_promoted_test_packages, and it lists directories rather
# than packages because that is the shape of the cells: the contract holds
# `packages/*/examples` as one cell across every package, not one cell per
# package, so a promotion here is repo-wide by construction and a per-package
# list would be a second answer to a question the contract asks once.
#
# The same both-directions rule applies — a name arrives here in the SAME
# change that moves the cell, and test_a_checked_cell_is_one_the_linter_actually
# _reaches fails either half on its own.
#
# Unlike the tests promotion, this one has to widen bin/fix.sh as well: a bare
# fix.sh reaches every packages/*/tests already, and reaches none of these. A
# checked cell the fix pass never opens is a red gate whose stated remedy
# cannot touch it, which test_every_lint_ceiling_is_reachable_by_the_fix holds.
lint_promoted_subdirs() {
    echo "examples scripts benchmarks"
}

# Function to list the per-package directories the formatter covers
#
# The formatter's population is NOT the linter's, and the difference is
# declared rather than incidental. `bin/validate.sh` lints packages/*/src plus
# the tests/ of each package in `lint_promoted_test_packages`; every other
# per-package directory sits in the quality contract's deferred tier for ruff.
# The contract holds `format` to a ceiling of 0 on all ten of its cells, so the
# formatter reaches every one of these.
#
# Kept here, beside workspace_targets, because three entry points need the same
# answer: the check in validate.sh, its write side in fix.sh, and `dk format`.
# They had three answers before this existed — the check and `dk format` read
# packages/*/src, fix.sh also read packages/*/tests, and none of them reached
# examples, scripts, benchmarks or docs. So the check passed over 874 files it
# never opened, and 42 of those had no local command that could format them.
#
# tests/test_toolchain_consistency.py compares this against the contract's
# enforced format cells, so a cell added there fails until it is named here.
format_subdirs() {
    echo "src tests examples scripts benchmarks docs"
}

# Function to list every path the formatter covers by default
#
# The whole population: each package's format directories, plus the code that
# belongs to no package. Callers narrowing to named packages compose
# format_subdirs themselves; this is the answer when nothing is named.
format_targets() {
    local targets=() package subdir
    local discovered
    discovered=$(discover_packages) || return 1

    while IFS= read -r package; do
        [[ -z "$package" ]] && continue
        for subdir in $(format_subdirs); do
            [[ -d "$ROOT_DIR/packages/$package/$subdir" ]] && targets+=("packages/$package/$subdir")
        done
    done <<< "${discovered// /$'\n'}"

    local workspace_target
    for workspace_target in $(workspace_targets); do
        targets+=("$workspace_target")
    done

    # Empty is legitimate and must not read as failure under `set -e`, for the
    # reason workspace_targets says.
    echo "${targets[@]:-}"
}

# Function to get packages in dependency order
# This reads from pyproject.toml to determine dependencies
get_packages_in_order() {
    local ordered_packages=()
    # `package` and `ordered` for the same reason as the two in the function
    # above: both name the loops below, and both were the caller's until now.
    local remaining_packages=() _pkg package ordered
    # Declared on its own line before the assignment: `local x=$(cmd)` reports
    # `local`'s status rather than the command's (SC2155), and the process
    # substitution this replaces reported no status at all. Either way a failing
    # discovery came back as an empty list, and this function's contract -- the
    # dependency order every caller installs in -- has no way to tell that from
    # a workspace with no packages.
    # `|| return 1` for the same reason, one level up: every caller reaches
    # this function through `$(get_packages_in_order)` too, so errexit is unset
    # in its frame as well and a failing capture would not abort it either.
    local discovered
    discovered=$(discover_packages) || return 1
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
    
    # `:-` as on the two echoes above: bash 3.2 treats an empty array as unset
    # under `set -u`, so a workspace with no packages died here rather than
    # reporting that it has none -- which is the answer this whole function
    # exists to let a caller tell apart from a failure.
    echo "${ordered_packages[@]:-}"
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
        format-subdirs)
            format_subdirs
            ;;
        format-targets)
            format_targets
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