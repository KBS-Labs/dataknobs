#!/bin/bash
# Deploy documentation to GitHub Pages
#
# This script builds and deploys the documentation to GitHub Pages
# The documentation will be available at:
#   https://<username>.github.io/<repository>/
#
# Prerequisites:
#   - Git repository with GitHub remote
#   - GitHub Pages enabled for the repository
#   - MkDocs and dependencies installed

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}📚 Dataknobs Documentation Deployment${NC}"
echo "========================================"

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo -e "${RED}❌ Error: Not in a git repository${NC}"
    exit 1
fi

# Check for uncommitted changes
if [ -n "$(git status --porcelain)" ]; then
    echo -e "${YELLOW}⚠️  Warning: You have uncommitted changes${NC}"
    echo -n "Continue anyway? (y/n) "
    read REPLY
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Get repository information
REMOTE_URL=$(git config --get remote.origin.url)
if [ -z "$REMOTE_URL" ]; then
    echo -e "${RED}❌ Error: No remote origin found${NC}"
    echo "Please add a GitHub remote: git remote add origin <url>"
    exit 1
fi

# Extract repository info from remote URL
if [[ $REMOTE_URL =~ github.com[:/]([^/]+)/([^/.]+)(\.git)?$ ]]; then
    GITHUB_USER="${BASH_REMATCH[1]}"
    GITHUB_REPO="${BASH_REMATCH[2]}"
    echo -e "${BLUE}Repository: ${GITHUB_USER}/${GITHUB_REPO}${NC}"
    echo -e "${BLUE}Docs URL: https://${GITHUB_USER}.github.io/${GITHUB_REPO}/${NC}"
else
    echo -e "${YELLOW}⚠️  Warning: Could not parse GitHub repository from remote URL${NC}"
fi

# Install dependencies if needed
if ! command -v mkdocs &> /dev/null; then
    echo -e "${YELLOW}Installing MkDocs dependencies...${NC}"
    uv pip install mkdocs mkdocs-material mkdocstrings[python] \
        mkdocs-monorepo-plugin mkdocs-awesome-pages-plugin \
        mkdocs-git-revision-date-localized-plugin
fi

# Build documentation first
echo -e "\n${CYAN}Building documentation...${NC}"
NO_MKDOCS_2_WARNING=1 mkdocs build --strict

if [ ! -d "site" ]; then
    echo -e "${RED}❌ Documentation build failed!${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Documentation built successfully${NC}"

# Deploy to GitHub Pages
echo -e "\n${CYAN}Deploying to GitHub Pages...${NC}"
echo "This will:"
echo "  1. Create/update the 'gh-pages' branch"
echo "  2. Push the documentation to GitHub"
echo "  3. Make it available at the URL shown above"
echo ""
echo -n "Deploy to GitHub Pages? (y/n) "
read REPLY

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${CYAN}Deploying...${NC}"

    # Deploy with mkdocs
    NO_MKDOCS_2_WARNING=1 mkdocs gh-deploy --force --clean --verbose
    
    echo -e "\n${GREEN}✅ Documentation deployed successfully!${NC}"
    echo ""
    echo "Your documentation is now available at:"
    echo -e "${CYAN}https://${GITHUB_USER}.github.io/${GITHUB_REPO}/${NC}"
    echo ""
    echo "Note: It may take a few minutes for GitHub Pages to update."
    echo ""
    echo "To configure GitHub Pages:"
    echo "  1. Go to https://github.com/${GITHUB_USER}/${GITHUB_REPO}/settings/pages"
    echo "  2. Under 'Source', select 'Deploy from a branch'"
    echo "  3. Select 'gh-pages' branch and '/ (root)' folder"
    echo "  4. Click 'Save'"
else
    echo -e "${YELLOW}Deployment cancelled${NC}"
    exit 0
fi