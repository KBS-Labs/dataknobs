#!/usr/bin/env bash
# Script to tag releases for each package

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Get the root directory
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Source package discovery utility
source "$ROOT_DIR/bin/package-discovery.sh"

echo -e "${CYAN}Dataknobs Release Tagging Tool${NC}"
echo ""

# Function to get version from pyproject.toml
get_version() {
    local package_dir=$1
    local pyproject="$package_dir/pyproject.toml"
    
    if [ ! -f "$pyproject" ]; then
        echo "unknown"
        return
    fi
    
    grep '^version = ' "$pyproject" | cut -d'"' -f2
}

# Function to check if tag exists
tag_exists() {
    local tag=$1
    git tag -l "$tag" | grep -q "$tag"
}

# Function to create tag
create_tag() {
    local package=$1
    local version=$2
    local tag="${package}/v${version}"
    
    if tag_exists "$tag"; then
        echo -e "${YELLOW}⚠ Tag $tag already exists${NC}"
        return 1
    fi
    
    echo -e "${CYAN}Creating tag: $tag${NC}"
    
    # Create annotated tag with message
    local message="Release ${package} v${version}

"
    
    # Add package description from pyproject.toml if available
    local pyproject="packages/$package/pyproject.toml"
    if [ -f "$pyproject" ]; then
        local description=$(grep '^description = ' "$pyproject" | cut -d'"' -f2)
        if [ -n "$description" ]; then
            message+="- $description"
        else
            message+="- Package: dataknobs-$package"
        fi
    else
        message+="- Package: dataknobs-$package"
    fi
    
    git tag -a "$tag" -m "$message"
    echo -e "${GREEN}✓ Tagged: $tag${NC}"
}

# Check git status
if [ -n "$(git status --porcelain)" ]; then
    echo -e "${RED}Error: Working directory has uncommitted changes${NC}"
    echo "Please commit or stash your changes before tagging"
    exit 1
fi

# Make sure we're on the right branch
CURRENT_BRANCH=$(git branch --show-current)
echo -e "${BLUE}Current branch: $CURRENT_BRANCH${NC}"

if [ "$CURRENT_BRANCH" != "main" ] && [ "$CURRENT_BRANCH" != "master" ]; then
    echo -e "${YELLOW}Warning: Not on main/master branch${NC}"
    echo -n "Continue on $CURRENT_BRANCH? (y/n) "
    read REPLY
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Get package information dynamically
PACKAGE_NAMES=($(get_packages_in_order))
PACKAGE_DIRS=()
for pkg in "${PACKAGE_NAMES[@]}"; do
    PACKAGE_DIRS+=("packages/$pkg")
done

# Display current versions
echo -e "\n${CYAN}Current Package Versions:${NC}"
echo "----------------------------------------"
for i in "${!PACKAGE_NAMES[@]}"; do
    package="${PACKAGE_NAMES[$i]}"
    dir="${PACKAGE_DIRS[$i]}"
    version=$(get_version "$dir")
    tag="${package}/v${version}"
    
    printf "%-15s v%-10s" "$package:" "$version"
    
    if tag_exists "$tag"; then
        echo -e " ${GREEN}[tagged]${NC}"
    else
        echo -e " ${YELLOW}[not tagged]${NC}"
    fi
done
echo "----------------------------------------"

# Ask what to tag
echo -e "\n${CYAN}What would you like to tag?${NC}"
echo "1) All untagged packages"
echo "2) Select specific packages"
echo "3) Create custom tag"
echo "4) Exit"
echo -n "Choice (1-4): "
read -r choice

case "$choice" in
    1)
        # Tag all untagged packages
        for i in "${!PACKAGE_NAMES[@]}"; do
            package="${PACKAGE_NAMES[$i]}"
            dir="${PACKAGE_DIRS[$i]}"
            version=$(get_version "$dir")
            tag="${package}/v${version}"
            
            if ! tag_exists "$tag"; then
                create_tag "$package" "$version"
            fi
        done
        ;;
    
    2)
        # Select specific packages
        for i in "${!PACKAGE_NAMES[@]}"; do
            package="${PACKAGE_NAMES[$i]}"
            dir="${PACKAGE_DIRS[$i]}"
            version=$(get_version "$dir")
            tag="${package}/v${version}"
            
            if tag_exists "$tag"; then
                echo -e "${package}: v${version} ${GREEN}[already tagged]${NC}"
            else
                echo -n "Tag ${package} v${version}? (y/n) "
                read REPLY
                if [[ $REPLY =~ ^[Yy]$ ]]; then
                    create_tag "$package" "$version"
                fi
            fi
        done
        ;;
    
    3)
        # Custom tag
        echo -n "Enter package name: "
        read package
        echo -n "Enter version (without 'v'): "
        read version
        
        if [ -z "$package" ] || [ -z "$version" ]; then
            echo -e "${RED}Error: Package name and version required${NC}"
            exit 1
        fi
        
        create_tag "$package" "$version"
        ;;
    
    4)
        echo "Exiting..."
        exit 0
        ;;
    
    *)
        echo -e "${RED}Invalid choice${NC}"
        exit 1
        ;;
esac

# Ask about pushing tags
echo ""
echo -n "Push tags to remote? (y/n) "
read REPLY
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${CYAN}Pushing tags to remote...${NC}"
    git push --tags
    echo -e "${GREEN}✓ Tags pushed to remote${NC}"
else
    echo -e "${YELLOW}Tags created locally. Push with: git push --tags${NC}"
fi

echo -e "\n${GREEN}✅ Tagging complete!${NC}"

# Show recent tags
echo -e "\n${CYAN}Recent tags:${NC}"
git tag --sort=-creatordate | head -10
