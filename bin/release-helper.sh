#!/usr/bin/env bash
# DataKnobs Release Helper Tool
# Streamlines the release process with automated checks and updates

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Get the root directory
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Source package discovery utility
source "$ROOT_DIR/bin/package-discovery.sh"

# Usage function
show_usage() {
    cat << EOF
$(echo -e "${BOLD}${CYAN}DataKnobs Release Helper${NC}")

$(echo -e "${BOLD}Usage:${NC}") $0 <command> [options]

$(echo -e "${BOLD}Commands:${NC}")
  $(echo -e "${CYAN}check${NC}")      Check what changed since last release
  $(echo -e "${CYAN}changes${NC}")    List all commits for a package or all packages
  $(echo -e "${CYAN}diffs${NC}")      Browse commit diffs interactively
  $(echo -e "${CYAN}bump${NC}")       Bump package versions interactively
  $(echo -e "${CYAN}notes${NC}")      Generate release notes from commits
  $(echo -e "${CYAN}tag${NC}")        Create release tags (calls tag-releases.sh)
  $(echo -e "${CYAN}publish${NC}")    Publish to PyPI (calls publish-pypi.sh)
  $(echo -e "${CYAN}verify${NC}")     Verify packages can be installed from PyPI
  $(echo -e "${CYAN}all${NC}")        Run complete release process interactively

$(echo -e "${BOLD}Examples:${NC}")
  $0 check            # See what changed
  $0 changes          # List all commits for all packages
  $0 changes core     # List commits for a specific package
  $0 diffs            # Browse diffs interactively
  $0 diffs core       # Browse diffs for a specific package
  $0 diffs --no-pager # Browse diffs without pager
  $0 bump             # Update versions
  $0 notes            # Generate changelog entries
  $0 all              # Full guided release

$(echo -e "${BOLD}Quick Release Flow:${NC}")
  1. dk pr            # Ensure quality checks pass
  2. $0 check         # Review changes
  3. $0 bump          # Update versions
  4. $0 notes         # Generate release notes
  5. git commit & PR  # Create release PR
  6. $0 tag           # After merge, create tags
  7. $0 publish       # Publish to PyPI
  8. $0 verify        # Verify installation

EOF
    exit 0
}

# Function to get version from pyproject.toml
get_version() {
    local package_dir=$1
    local pyproject="$package_dir/pyproject.toml"
    
    if [ ! -f "$pyproject" ]; then
        echo "0.0.0"
        return
    fi
    
    grep '^version = ' "$pyproject" | cut -d'"' -f2
}

# Function to get last tag for a package
get_last_tag() {
    local package=$1
    # Get all tags for this package, sort by version
    git tag -l "${package}/v*" 2>/dev/null | sort -V | tail -1
}

# Function to list all changes (detailed commit list)
list_changes() {
    local target_package="${1:-}"

    # Get all packages
    PACKAGES=($(get_packages_in_order))

    if [ -n "$target_package" ] && [ "$target_package" != "all" ]; then
        # Validate package exists
        local found=false
        for pkg in "${PACKAGES[@]}"; do
            if [ "$pkg" = "$target_package" ]; then
                found=true
                break
            fi
        done

        if [ "$found" = false ]; then
            echo -e "${RED}Error: Unknown package '${target_package}'${NC}"
            echo -e "Available packages: ${PACKAGES[*]}"
            exit 1
        fi

        # Show changes for specific package
        local package_dir="packages/$target_package"
        local last_tag=$(get_last_tag "$target_package")
        local current_version=$(get_version "$package_dir")

        echo -e "${BOLD}${CYAN}Changes for ${target_package}${NC}"
        echo -e "Current version: v${current_version}"

        if [ -z "$last_tag" ]; then
            echo -e "Last release: ${YELLOW}none (new package)${NC}"
            echo ""
            echo -e "${BOLD}All commits:${NC}"
            git log --oneline --no-merges -- "$package_dir" 2>/dev/null || echo "  No commits found"
        else
            local tag_version="${last_tag#*/v}"
            echo -e "Last release: v${tag_version} (${last_tag})"
            echo ""
            echo -e "${BOLD}Commits since ${last_tag}:${NC}"
            local changes=$(git log --oneline --no-merges "${last_tag}..HEAD" -- "$package_dir" 2>/dev/null)
            if [ -n "$changes" ]; then
                echo "$changes"
            else
                echo -e "  ${GREEN}No changes since last release${NC}"
            fi
        fi
    else
        # Show changes for all packages
        echo -e "${BOLD}${CYAN}Changes for all packages${NC}"
        echo ""

        local any_changes=false

        for package in "${PACKAGES[@]}"; do
            local package_dir="packages/$package"
            local last_tag=$(get_last_tag "$package")
            local current_version=$(get_version "$package_dir")

            local changes=""
            if [ -z "$last_tag" ]; then
                changes=$(git log --oneline --no-merges -- "$package_dir" 2>/dev/null)
                if [ -n "$changes" ]; then
                    echo -e "${BOLD}${package}${NC} (v${current_version}) - ${YELLOW}new package${NC}"
                    echo "$changes" | sed 's/^/  /'
                    echo ""
                    any_changes=true
                fi
            else
                changes=$(git log --oneline --no-merges "${last_tag}..HEAD" -- "$package_dir" 2>/dev/null)
                if [ -n "$changes" ]; then
                    local tag_version="${last_tag#*/v}"
                    echo -e "${BOLD}${package}${NC} (v${tag_version} → v${current_version})"
                    echo "$changes" | sed 's/^/  /'
                    echo ""
                    any_changes=true
                fi
            fi
        done

        if [ "$any_changes" = false ]; then
            echo -e "${GREEN}No changes detected since last releases${NC}"
        fi
    fi
}

# Function to interactively browse diffs
browse_diffs() {
    local arg_package="${1:-}"
    local arg_commit="${2:-}"
    local arg_file="${3:-}"
    local use_pager="${4:-true}"

    # Get all packages
    PACKAGES=($(get_packages_in_order))

    # Build list of packages with changes
    local packages_with_changes=()
    for package in "${PACKAGES[@]}"; do
        local package_dir="packages/$package"
        local last_tag=$(get_last_tag "$package")
        local changes=""

        if [ -z "$last_tag" ]; then
            changes=$(git log --oneline --no-merges -- "$package_dir" 2>/dev/null)
        else
            changes=$(git log --oneline --no-merges "${last_tag}..HEAD" -- "$package_dir" 2>/dev/null)
        fi

        if [ -n "$changes" ]; then
            packages_with_changes+=("$package")
        fi
    done

    if [ ${#packages_with_changes[@]} -eq 0 ]; then
        echo -e "${GREEN}No changes detected since last releases${NC}"
        return
    fi

    # Main interactive loop
    while true; do
        local selected_package=""
        local selected_commit=""
        local selected_file=""

        # Package selection
        if [ -n "$arg_package" ]; then
            # Validate provided package
            local found=false
            for pkg in "${packages_with_changes[@]}"; do
                if [ "$pkg" = "$arg_package" ]; then
                    found=true
                    break
                fi
            done

            if [ "$found" = false ]; then
                echo -e "${RED}Error: Package '${arg_package}' not found or has no changes${NC}"
                echo -e "Packages with changes: ${packages_with_changes[*]}"
                return 1
            fi
            selected_package="$arg_package"
        else
            # Interactive package selection
            echo -e "${BOLD}${CYAN}Select a package:${NC}"
            local i=1
            for pkg in "${packages_with_changes[@]}"; do
                echo "  $i) $pkg"
                ((i++))
            done
            echo "  q) Quit"
            echo -n "Choice: "
            read -r choice

            if [ "$choice" = "q" ] || [ "$choice" = "Q" ]; then
                return 0
            fi

            if ! [[ "$choice" =~ ^[0-9]+$ ]] || [ "$choice" -lt 1 ] || [ "$choice" -gt ${#packages_with_changes[@]} ]; then
                echo -e "${RED}Invalid choice${NC}"
                continue
            fi

            selected_package="${packages_with_changes[$((choice - 1))]}"
        fi

        echo ""

        # Get commits for selected package
        local package_dir="packages/$selected_package"
        local last_tag=$(get_last_tag "$selected_package")
        local commits=()
        local commit_messages=()

        if [ -z "$last_tag" ]; then
            while IFS= read -r line; do
                commits+=("${line%% *}")
                commit_messages+=("${line#* }")
            done < <(git log --oneline --no-merges -- "$package_dir" 2>/dev/null)
        else
            while IFS= read -r line; do
                commits+=("${line%% *}")
                commit_messages+=("${line#* }")
            done < <(git log --oneline --no-merges "${last_tag}..HEAD" -- "$package_dir" 2>/dev/null)
        fi

        if [ ${#commits[@]} -eq 0 ]; then
            echo -e "${YELLOW}No commits found for ${selected_package}${NC}"
            if [ -n "$arg_package" ]; then
                return 0
            fi
            continue
        fi

        # Commit selection loop
        while true; do
            if [ -n "$arg_commit" ]; then
                # Validate provided commit
                local found=false
                for c in "${commits[@]}"; do
                    if [ "$c" = "$arg_commit" ]; then
                        found=true
                        break
                    fi
                done

                if [ "$found" = false ]; then
                    echo -e "${RED}Error: Commit '${arg_commit}' not found in ${selected_package}${NC}"
                    return 1
                fi
                selected_commit="$arg_commit"
            else
                # Interactive commit selection
                echo -e "${BOLD}${CYAN}Commits for ${selected_package}:${NC}"
                local i=1
                for idx in "${!commits[@]}"; do
                    echo "  $i) ${commits[$idx]} ${commit_messages[$idx]}"
                    ((i++))
                done
                echo "  b) Back to packages"
                echo "  q) Quit"
                echo -n "Choice: "
                read -r choice

                if [ "$choice" = "q" ] || [ "$choice" = "Q" ]; then
                    return 0
                fi

                if [ "$choice" = "b" ] || [ "$choice" = "B" ]; then
                    if [ -n "$arg_package" ]; then
                        return 0
                    fi
                    echo ""
                    break
                fi

                if ! [[ "$choice" =~ ^[0-9]+$ ]] || [ "$choice" -lt 1 ] || [ "$choice" -gt ${#commits[@]} ]; then
                    echo -e "${RED}Invalid choice${NC}"
                    continue
                fi

                selected_commit="${commits[$((choice - 1))]}"
            fi

            echo ""

            # Get files for selected commit in selected package
            local files=()
            while IFS= read -r file; do
                files+=("$file")
            done < <(git show "$selected_commit" --name-only --format="" | grep "^packages/$selected_package/" 2>/dev/null)

            if [ ${#files[@]} -eq 0 ]; then
                echo -e "${YELLOW}No files found for commit ${selected_commit} in ${selected_package}${NC}"
                if [ -n "$arg_commit" ]; then
                    return 0
                fi
                continue
            fi

            # File selection loop
            while true; do
                if [ -n "$arg_file" ]; then
                    # Find matching file
                    local found=false
                    for f in "${files[@]}"; do
                        if [ "$f" = "$arg_file" ] || [[ "$f" == *"$arg_file" ]]; then
                            selected_file="$f"
                            found=true
                            break
                        fi
                    done

                    if [ "$found" = false ]; then
                        echo -e "${RED}Error: File '${arg_file}' not found in commit${NC}"
                        return 1
                    fi
                else
                    # Interactive file selection
                    echo -e "${BOLD}${CYAN}Files in commit ${selected_commit}:${NC}"
                    local i=1
                    for file in "${files[@]}"; do
                        # Show relative path from package dir for readability
                        local display_file="${file#packages/$selected_package/}"
                        echo "  $i) $display_file"
                        ((i++))
                    done
                    echo "  b) Back to commits"
                    echo "  q) Quit"
                    echo -n "Choice: "
                    read -r choice

                    if [ "$choice" = "q" ] || [ "$choice" = "Q" ]; then
                        return 0
                    fi

                    if [ "$choice" = "b" ] || [ "$choice" = "B" ]; then
                        if [ -n "$arg_commit" ]; then
                            return 0
                        fi
                        echo ""
                        break
                    fi

                    if ! [[ "$choice" =~ ^[0-9]+$ ]] || [ "$choice" -lt 1 ] || [ "$choice" -gt ${#files[@]} ]; then
                        echo -e "${RED}Invalid choice${NC}"
                        continue
                    fi

                    selected_file="${files[$((choice - 1))]}"
                fi

                # Get commit message for display
                local commit_msg=$(git log -1 --format="%s" "$selected_commit" 2>/dev/null)

                # Display the diff
                echo ""
                echo -e "${BOLD}${MAGENTA}════════════════════════════════════════${NC}"
                echo -e "${BOLD}Commit:${NC} ${selected_commit}"
                echo -e "${BOLD}Message:${NC} ${commit_msg}"
                echo -e "${BOLD}File:${NC} ${selected_file}"
                echo -e "${BOLD}${MAGENTA}════════════════════════════════════════${NC}"
                echo ""

                if [ "$use_pager" = "true" ]; then
                    git show "$selected_commit" -- "$selected_file" | less -R
                else
                    git show "$selected_commit" -- "$selected_file"
                fi

                # If all arguments were provided, exit after showing diff
                if [ -n "$arg_package" ] && [ -n "$arg_commit" ] && [ -n "$arg_file" ]; then
                    return 0
                fi

                # Post-diff options
                echo ""
                echo -e "${CYAN}Options:${NC}"
                echo "  n) Next file"
                echo "  p) Toggle pager (currently: $use_pager)"
                echo "  b) Back to file list"
                echo "  c) Back to commits"
                echo "  k) Back to packages"
                echo "  q) Quit"
                echo -n "Choice: "
                read -r choice

                case "$choice" in
                    n|N)
                        # Find next file
                        local current_idx=0
                        for idx in "${!files[@]}"; do
                            if [ "${files[$idx]}" = "$selected_file" ]; then
                                current_idx=$idx
                                break
                            fi
                        done

                        local next_idx=$((current_idx + 1))
                        if [ $next_idx -ge ${#files[@]} ]; then
                            echo -e "${YELLOW}No more files in this commit${NC}"
                            continue
                        fi

                        selected_file="${files[$next_idx]}"
                        ;;
                    p|P)
                        if [ "$use_pager" = "true" ]; then
                            use_pager="false"
                            echo -e "${YELLOW}Pager disabled${NC}"
                        else
                            use_pager="true"
                            echo -e "${GREEN}Pager enabled${NC}"
                        fi
                        ;;
                    b|B)
                        echo ""
                        continue 2  # Back to file selection in file loop
                        ;;
                    c|C)
                        if [ -n "$arg_commit" ]; then
                            return 0
                        fi
                        echo ""
                        break 2  # Back to commit selection
                        ;;
                    k|K)
                        if [ -n "$arg_package" ]; then
                            return 0
                        fi
                        echo ""
                        break 3  # Back to package selection
                        ;;
                    q|Q)
                        return 0
                        ;;
                    *)
                        echo -e "${RED}Invalid choice${NC}"
                        ;;
                esac
            done

            # Break out of commit loop if we need to go back to packages
            if [ -n "$arg_commit" ]; then
                return 0
            fi
        done

        # If package was provided as argument, exit after processing
        if [ -n "$arg_package" ]; then
            return 0
        fi
    done
}

# Function to check what changed
check_changes() {
    echo -e "${CYAN}Checking changes since last release...${NC}"
    echo ""
    
    local has_changes=false
    
    # Get all packages
    PACKAGES=($(get_packages_in_order))
    
    for package in "${PACKAGES[@]}"; do
        local package_dir="packages/$package"
        local current_version=$(get_version "$package_dir")
        local last_tag=$(get_last_tag "$package")
        
        if [ -z "$last_tag" ]; then
            # No previous tag, check all commits
            local changes=$(git log --oneline --no-merges -- "$package_dir" 2>/dev/null | head -5)
            if [ -n "$changes" ]; then
                echo -e "${BOLD}${package}${NC} (no previous release → v${current_version})"
                echo -e "${YELLOW}  New package!${NC}"
                has_changes=true
            fi
        else
            # Check changes since last tag
            local changes=$(git log --oneline --no-merges "${last_tag}..HEAD" -- "$package_dir" 2>/dev/null)
            # Extract version from tag (package/vX.Y.Z format)
            local tag_version="${last_tag#*/v}"
            
            # Determine recommended version based on changes
            local recommended_version="$current_version"
            if [ -n "$changes" ]; then
                # Parse current version components
                IFS='.' read -r major minor patch <<< "$tag_version"
                
                # Categorize changes to determine version bump
                local features=$(echo "$changes" | grep -i "feat\|add\|new" | wc -l)
                local fixes=$(echo "$changes" | grep -i "fix\|bug\|patch" | wc -l)
                local breaking=$(echo "$changes" | grep -i "breaking\|!:" | wc -l)
                
                if [ $breaking -gt 0 ]; then
                    recommended_version="$((major + 1)).0.0"
                elif [ $features -gt 0 ]; then
                    recommended_version="${major}.$((minor + 1)).0"
                elif [ $fixes -gt 0 ]; then
                    recommended_version="${major}.${minor}.$((patch + 1))"
                else
                    # Other changes - default to patch
                    recommended_version="${major}.${minor}.$((patch + 1))"
                fi
            fi
            
            # Only show if there are changes OR if versions differ
            if [ -n "$changes" ] || [ "$tag_version" != "$current_version" ]; then
                if [ "$recommended_version" != "$current_version" ] && [ -n "$changes" ]; then
                    # Show recommended version bump
                    echo -e "${BOLD}${package}${NC} (v${tag_version} → v${current_version}, recommend v${recommended_version})"
                else
                    # Current version matches recommendation or no changes
                    echo -e "${BOLD}${package}${NC} (v${tag_version} → v${current_version})"
                fi
                
                if [ -n "$changes" ]; then
                    # Show change type and recommendation
                    if [ "$recommended_version" != "$current_version" ]; then
                        if [ $breaking -gt 0 ]; then
                            echo -e "  ${RED}Breaking changes detected - Major version bump to v${recommended_version} recommended${NC}"
                        elif [ $features -gt 0 ]; then
                            echo -e "  ${YELLOW}New features detected - Minor version bump to v${recommended_version} recommended${NC}"
                        elif [ $fixes -gt 0 ]; then
                            echo -e "  ${GREEN}Bug fixes detected - Patch version bump to v${recommended_version} recommended${NC}"
                        else
                            echo -e "  ${BLUE}Other changes detected - Patch version bump to v${recommended_version} recommended${NC}"
                        fi
                    else
                        # Version already correctly bumped
                        if [ $breaking -gt 0 ]; then
                            echo -e "  ${GREEN}✓ Breaking changes - Major version already bumped${NC}"
                        elif [ $features -gt 0 ]; then
                            echo -e "  ${GREEN}✓ New features - Minor version already bumped${NC}"
                        elif [ $fixes -gt 0 ]; then
                            echo -e "  ${GREEN}✓ Bug fixes - Patch version already bumped${NC}"
                        else
                            echo -e "  ${GREEN}✓ Version already updated${NC}"
                        fi
                    fi
                    
                    echo -e "  Changes: $features features, $fixes fixes, $breaking breaking"
                elif [ "$tag_version" != "$current_version" ]; then
                    # Version was already bumped but no new commits
                    echo -e "  ${CYAN}Version already bumped (no new commits since bump)${NC}"
                fi
                echo ""
                has_changes=true
            fi
        fi
    done
    
    if [ "$has_changes" = false ]; then
        echo -e "${GREEN}No changes detected since last release${NC}"
    fi
}

# Function to bump version
bump_version() {
    local package=$1
    local package_dir="packages/$package"
    local pyproject="$package_dir/pyproject.toml"
    local current_version=$(get_version "$package_dir")
    
    # Parse version components
    IFS='.' read -r major minor patch <<< "$current_version"
    
    # Determine recommendation based on changes
    local last_tag=$(get_last_tag "$package")
    local recommendation="patch"
    local default_choice="1"
    
    if [ -n "$last_tag" ]; then
        local changes=$(git log --oneline --no-merges "${last_tag}..HEAD" -- "$package_dir" 2>/dev/null)
        if [ -n "$changes" ]; then
            local features=$(echo "$changes" | grep -i "feat\|add\|new" | wc -l)
            local fixes=$(echo "$changes" | grep -i "fix\|bug\|patch" | wc -l)
            local breaking=$(echo "$changes" | grep -i "breaking\|!:" | wc -l)
            
            if [ $breaking -gt 0 ]; then
                recommendation="major"
                default_choice="3"
            elif [ $features -gt 0 ]; then
                recommendation="minor"
                default_choice="2"
            elif [ $fixes -gt 0 ]; then
                recommendation="patch"
                default_choice="1"
            fi
        fi
    else
        # New package, recommend minor version for initial release features
        recommendation="minor"
        default_choice="2"
    fi
    
    echo -e "\n${CYAN}Package: ${package}${NC}"
    echo -e "Current version: ${current_version}"
    echo -e "${YELLOW}Recommendation: ${recommendation} version bump${NC}"
    echo ""
    echo "Select version bump:"
    
    # Show options with recommendation marker
    if [ "$recommendation" = "patch" ]; then
        echo -e "  1) Patch (${major}.${minor}.$((patch + 1))) ${GREEN}[RECOMMENDED]${NC}"
    else
        echo "  1) Patch (${major}.${minor}.$((patch + 1)))"
    fi
    
    if [ "$recommendation" = "minor" ]; then
        echo -e "  2) Minor (${major}.$((minor + 1)).0) ${GREEN}[RECOMMENDED]${NC}"
    else
        echo "  2) Minor (${major}.$((minor + 1)).0)"
    fi
    
    if [ "$recommendation" = "major" ]; then
        echo -e "  3) Major ($((major + 1)).0.0) ${GREEN}[RECOMMENDED]${NC}"
    else
        echo "  3) Major ($((major + 1)).0.0)"
    fi
    
    echo "  4) Custom"
    echo "  5) Skip"
    echo -n "Choice [1-5] (default: $default_choice): "
    read -r choice
    
    # Use default if empty
    if [ -z "$choice" ]; then
        choice="$default_choice"
    fi
    
    local new_version=""
    case "$choice" in
        1) new_version="${major}.${minor}.$((patch + 1))" ;;
        2) new_version="${major}.$((minor + 1)).0" ;;
        3) new_version="$((major + 1)).0.0" ;;
        4)
            echo -n "Enter new version: "
            read -r new_version
            ;;
        5)
            echo -e "${YELLOW}Skipping ${package}${NC}"
            return
            ;;
        *)
            echo -e "${RED}Invalid choice${NC}"
            return
            ;;
    esac
    
    if [ -n "$new_version" ]; then
        # Update version in pyproject.toml
        sed -i.bak "s/^version = \"${current_version}\"/version = \"${new_version}\"/" "$pyproject"
        rm "${pyproject}.bak"

        # Update version in packages.json
        local packages_json="$ROOT_DIR/.dataknobs/packages.json"
        if [ -f "$packages_json" ]; then
            python3 << EOF
import json
import sys

packages_json = "$packages_json"
package_name = "$package"
new_version = "$new_version"

try:
    with open(packages_json, 'r') as f:
        data = json.load(f)

    # Find and update the package version
    for pkg in data.get('packages', []):
        if pkg['name'] == package_name:
            pkg['version'] = new_version
            break

    with open(packages_json, 'w') as f:
        json.dump(data, f, indent=2)
        f.write('\n')  # Add trailing newline

except Exception as e:
    print(f"Warning: Could not update packages.json: {e}", file=sys.stderr)
    sys.exit(1)
EOF
            if [ $? -eq 0 ]; then
                echo -e "${GREEN}✓ Updated ${package} to v${new_version} (pyproject.toml and packages.json)${NC}"
            else
                echo -e "${YELLOW}⚠ Updated ${package} to v${new_version} in pyproject.toml but packages.json update failed${NC}"
            fi
        else
            echo -e "${GREEN}✓ Updated ${package} to v${new_version}${NC}"
        fi
    fi
}

# Function to bump versions interactively
bump_versions() {
    echo -e "${CYAN}Bumping package versions...${NC}"
    echo ""
    
    # First show what changed
    check_changes
    echo ""
    echo -e "${YELLOW}─────────────────────────────${NC}"
    
    # Get all packages
    PACKAGES=($(get_packages_in_order))
    
    echo -e "${BOLD}Update versions?${NC}"
    echo "  1) Update all changed packages"
    echo "  2) Update all changed packages except specific ones"
    echo "  3) Select specific packages to update"
    echo "  4) Skip version updates"
    echo -n "Choice [1-4]: "
    read -r choice
    
    case "$choice" in
        1)
            # Update all changed packages
            for package in "${PACKAGES[@]}"; do
                local last_tag=$(get_last_tag "$package")
                local package_dir="packages/$package"
                
                if [ -z "$last_tag" ]; then
                    # New package
                    bump_version "$package"
                else
                    # Check if package has changes
                    local changes=$(git log --oneline "${last_tag}..HEAD" -- "$package_dir" 2>/dev/null)
                    if [ -n "$changes" ]; then
                        bump_version "$package"
                    fi
                fi
            done
            ;;
        2)
            # Update all changed packages except selected ones
            echo -e "\n${CYAN}Select packages to EXCLUDE from version bumping:${NC}"
            local excluded_packages=()
            
            # Show packages with changes
            for package in "${PACKAGES[@]}"; do
                local last_tag=$(get_last_tag "$package")
                local package_dir="packages/$package"
                local has_changes=false
                
                if [ -z "$last_tag" ]; then
                    has_changes=true
                else
                    local changes=$(git log --oneline "${last_tag}..HEAD" -- "$package_dir" 2>/dev/null)
                    if [ -n "$changes" ]; then
                        has_changes=true
                    fi
                fi
                
                if [ "$has_changes" = true ]; then
                    echo -n "  Exclude ${package}? (y/n) [n]: "
                    read -r reply
                    if [[ $reply =~ ^[Yy]$ ]]; then
                        excluded_packages+=("$package")
                        echo -e "    ${YELLOW}→ ${package} will be skipped${NC}"
                    fi
                fi
            done
            
            echo ""
            # Now bump versions for non-excluded packages
            for package in "${PACKAGES[@]}"; do
                # Check if package is excluded
                local is_excluded=false
                for excluded in "${excluded_packages[@]}"; do
                    if [ "$package" = "$excluded" ]; then
                        is_excluded=true
                        break
                    fi
                done
                
                if [ "$is_excluded" = true ]; then
                    echo -e "${YELLOW}Skipping ${package} (excluded)${NC}"
                    continue
                fi
                
                local last_tag=$(get_last_tag "$package")
                local package_dir="packages/$package"
                
                if [ -z "$last_tag" ]; then
                    # New package
                    bump_version "$package"
                else
                    # Check if package has changes
                    local changes=$(git log --oneline "${last_tag}..HEAD" -- "$package_dir" 2>/dev/null)
                    if [ -n "$changes" ]; then
                        bump_version "$package"
                    fi
                fi
            done
            ;;
        3)
            # Select specific packages
            for package in "${PACKAGES[@]}"; do
                bump_version "$package"
            done
            ;;
        4)
            echo -e "${YELLOW}Skipping version updates${NC}"
            ;;
        *)
            echo -e "${RED}Invalid choice${NC}"
            ;;
    esac
}

# Function to generate release notes
generate_notes() {
    echo -e "${CYAN}Generating release notes...${NC}"
    echo ""
    
    local changelog_file="docs/changelog.md"
    local temp_file="/tmp/changelog_new.md"
    local date=$(date +%Y-%m-%d)
    
    # Start with existing content up to [Unreleased]
    sed '/^## \[Unreleased\]/q' "$changelog_file" > "$temp_file"
    echo "" >> "$temp_file"
    
    # Add new release section
    echo "## Release - $date" >> "$temp_file"
    echo "" >> "$temp_file"
    
    # Get all packages
    PACKAGES=($(get_packages_in_order))
    
    for package in "${PACKAGES[@]}"; do
        local package_dir="packages/$package"
        local version=$(get_version "$package_dir")
        local last_tag=$(get_last_tag "$package")
        
        local changes=""
        if [ -z "$last_tag" ]; then
            # New package - get all commits
            changes=$(git log --oneline --no-merges -- "$package_dir" 2>/dev/null)
        else
            # Get changes since last tag
            changes=$(git log --oneline --no-merges "${last_tag}..HEAD" -- "$package_dir" 2>/dev/null)
        fi
        
        if [ -n "$changes" ]; then
            echo "### dataknobs-${package} [${version}]" >> "$temp_file"
            echo "" >> "$temp_file"
            
            # Categorize changes
            local added=$(echo "$changes" | grep -i "add\|new\|feat" || true)
            local changed=$(echo "$changes" | grep -i "update\|change\|improve\|enhance" || true)
            local fixed=$(echo "$changes" | grep -i "fix\|bug\|patch\|correct" || true)
            local breaking=$(echo "$changes" | grep -i "breaking\|!:" || true)
            
            if [ -n "$added" ]; then
                echo "#### Added" >> "$temp_file"
                echo "$added" | while read -r line; do
                    echo "- ${line#* }" >> "$temp_file"
                done
                echo "" >> "$temp_file"
            fi
            
            if [ -n "$changed" ]; then
                echo "#### Changed" >> "$temp_file"
                echo "$changed" | while read -r line; do
                    echo "- ${line#* }" >> "$temp_file"
                done
                echo "" >> "$temp_file"
            fi
            
            if [ -n "$fixed" ]; then
                echo "#### Fixed" >> "$temp_file"
                echo "$fixed" | while read -r line; do
                    echo "- ${line#* }" >> "$temp_file"
                done
                echo "" >> "$temp_file"
            fi
            
            if [ -n "$breaking" ]; then
                echo "#### Breaking Changes" >> "$temp_file"
                echo "$breaking" | while read -r line; do
                    echo "- ${line#* }" >> "$temp_file"
                done
                echo "" >> "$temp_file"
            fi
        fi
    done
    
    # Add the rest of the original changelog
    sed '1,/^## \[Unreleased\]/d' "$changelog_file" >> "$temp_file"
    
    # Show preview
    echo -e "${YELLOW}Preview of new changelog entries:${NC}"
    echo "─────────────────────────────"
    sed -n '/^## Release - '"$date"'/,/^## /p' "$temp_file" | head -50
    echo "─────────────────────────────"
    echo ""
    echo -n "Apply these changes to changelog? (y/n): "
    read -r reply
    
    if [[ $reply =~ ^[Yy]$ ]]; then
        cp "$temp_file" "$changelog_file"
        echo -e "${GREEN}✓ Updated ${changelog_file}${NC}"
        echo -e "${YELLOW}Please review and edit the changelog as needed${NC}"
    else
        echo -e "${YELLOW}Changelog not updated${NC}"
    fi
    
    rm -f "$temp_file"
}

# Function to verify installation
verify_installation() {
    echo -e "${CYAN}Verifying package installation...${NC}"
    echo ""
    
    # Create temporary virtual environment
    local temp_dir="/tmp/dataknobs_verify_$$"
    echo -e "${YELLOW}Creating test environment...${NC}"
    python3 -m venv "$temp_dir"
    source "$temp_dir/bin/activate"
    
    # Get all packages
    PACKAGES=($(get_packages_in_order))
    
    echo -e "${YELLOW}Testing package installations...${NC}"
    local all_success=true
    
    for package in "${PACKAGES[@]}"; do
        local package_name="dataknobs-${package}"
        if [ "$package" = "legacy" ]; then
            package_name="dataknobs"
        fi
        
        echo -n "  ${package_name}: "
        
        # Try to install
        if pip install "$package_name" --index-url https://pypi.org/simple/ >/dev/null 2>&1; then
            # Try to import
            local import_name="dataknobs_${package//-/_}"
            if [ "$package" = "legacy" ]; then
                import_name="dataknobs"
            fi
            
            if python -c "import ${import_name}; print(${import_name}.__version__)" 2>/dev/null; then
                echo -e "${GREEN}✓${NC}"
            else
                echo -e "${RED}✗ (import failed)${NC}"
                all_success=false
            fi
        else
            echo -e "${YELLOW}⊘ (not on PyPI)${NC}"
        fi
    done
    
    # Clean up
    deactivate
    rm -rf "$temp_dir"
    
    if [ "$all_success" = true ]; then
        echo -e "\n${GREEN}✓ All packages verified successfully${NC}"
    else
        echo -e "\n${YELLOW}⚠ Some packages had issues${NC}"
    fi
}

# Function for complete release process
full_release() {
    echo -e "${BOLD}${CYAN}DataKnobs Full Release Process${NC}"
    echo "═══════════════════════════════════════"
    echo ""
    
    # Step 1: Check quality
    echo -e "${BOLD}Step 1: Quality Check${NC}"
    echo -n "Run quality checks? (y/n): "
    read -r reply
    if [[ $reply =~ ^[Yy]$ ]]; then
        "$ROOT_DIR/bin/dk" pr
        echo ""
        echo -n "Did all checks pass? (y/n): "
        read -r reply
        if [[ ! $reply =~ ^[Yy]$ ]]; then
            echo -e "${RED}Please fix issues before continuing${NC}"
            exit 1
        fi
    fi
    
    # Step 2: Check changes
    echo -e "\n${BOLD}Step 2: Review Changes${NC}"
    check_changes
    echo ""
    echo -n "Continue with release? (y/n): "
    read -r reply
    if [[ ! $reply =~ ^[Yy]$ ]]; then
        exit 0
    fi
    
    # Step 3: Bump versions
    echo -e "\n${BOLD}Step 3: Version Updates${NC}"
    bump_versions
    
    # Step 4: Generate notes
    echo -e "\n${BOLD}Step 4: Release Notes${NC}"
    generate_notes
    
    # Step 5: Commit changes
    echo -e "\n${BOLD}Step 5: Commit Changes${NC}"
    echo "Review the changes:"
    git diff --stat
    echo ""
    echo -n "Commit these changes? (y/n): "
    read -r reply
    if [[ $reply =~ ^[Yy]$ ]]; then
        echo -n "Enter commit message: "
        read -r message
        git add -A
        git commit -m "$message"
        echo -e "${GREEN}✓ Changes committed${NC}"
        echo -e "${YELLOW}Next: Push and create PR on GitHub${NC}"
    fi
    
    echo -e "\n${GREEN}Release preparation complete!${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Push branch and create PR"
    echo "  2. After merge: $0 tag"
    echo "  3. Build: bin/build-packages.sh"
    echo "  4. Publish: $0 publish"
    echo "  5. Verify: $0 verify"
}

# Main command handling
case "${1:-help}" in
    check)
        check_changes
        ;;
    changes)
        shift
        list_changes "${1:-}"
        ;;
    diffs)
        shift
        # Parse arguments for diffs command
        diffs_package=""
        diffs_commit=""
        diffs_file=""
        diffs_pager="true"

        while [ $# -gt 0 ]; do
            case "$1" in
                --no-pager)
                    diffs_pager="false"
                    ;;
                *)
                    # Positional arguments: package, commit, file
                    if [ -z "$diffs_package" ]; then
                        diffs_package="$1"
                    elif [ -z "$diffs_commit" ]; then
                        diffs_commit="$1"
                    elif [ -z "$diffs_file" ]; then
                        diffs_file="$1"
                    fi
                    ;;
            esac
            shift
        done

        browse_diffs "$diffs_package" "$diffs_commit" "$diffs_file" "$diffs_pager"
        ;;
    bump)
        bump_versions
        ;;
    notes)
        generate_notes
        ;;
    tag)
        echo -e "${CYAN}Creating release tags...${NC}"
        "$ROOT_DIR/bin/tag-releases.sh"
        ;;
    publish)
        echo -e "${CYAN}Publishing to PyPI...${NC}"
        shift
        "$ROOT_DIR/bin/publish-pypi.sh" "$@"
        ;;
    verify)
        verify_installation
        ;;
    all)
        full_release
        ;;
    help|-h|--help)
        show_usage
        ;;
    *)
        echo -e "${RED}Unknown command: $1${NC}"
        show_usage
        ;;
esac