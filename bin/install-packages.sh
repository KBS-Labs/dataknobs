#!/usr/bin/env bash
# Install dataknobs packages in development or production mode

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

# Default values
MODE="dev"
VENV_NAME=""
FORCE_REINSTALL=false

# Usage function
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -m, --mode MODE       Installation mode: 'dev' (default) or 'prod'"
    echo "  -e, --env NAME        Virtual environment name (creates new if doesn't exist)"
    echo "  -f, --force           Force reinstall packages"
    echo "  -h, --help            Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                    # Install in dev mode with current environment"
    echo "  $0 -m prod -e venv    # Install in prod mode in 'venv' environment"
    echo "  $0 -f                 # Force reinstall in dev mode"
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--mode)
            MODE="$2"
            shift 2
            ;;
        -e|--env)
            VENV_NAME="$2"
            shift 2
            ;;
        -f|--force)
            FORCE_REINSTALL=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Validate mode
if [[ "$MODE" != "dev" && "$MODE" != "prod" ]]; then
    echo -e "${RED}Error: Invalid mode '$MODE'. Must be 'dev' or 'prod'${NC}"
    exit 1
fi

echo -e "${YELLOW}Installing dataknobs packages in ${BLUE}$MODE${YELLOW} mode...${NC}"

# Handle virtual environment
if [[ -n "$VENV_NAME" ]]; then
    if [[ ! -d "$VENV_NAME" ]]; then
        echo -e "${YELLOW}Creating virtual environment: $VENV_NAME${NC}"
        uv venv "$VENV_NAME"
    fi
    echo -e "${YELLOW}Activating virtual environment: $VENV_NAME${NC}"
    source "$VENV_NAME/bin/activate"
fi

# Install order matters due to dependencies
PACKAGES=(
    "common"
    "structures" 
    "xization"
    "utils"
    "legacy"
)

if [[ "$MODE" == "dev" ]]; then
    echo -e "${YELLOW}Installing in development mode (editable)...${NC}"
    
    # In dev mode, use uv pip install -e
    for package in "${PACKAGES[@]}"; do
        echo -e "\n${YELLOW}Installing dataknobs-$package...${NC}"
        
        INSTALL_CMD="uv pip install -e packages/$package"
        if [[ "$FORCE_REINSTALL" == true ]]; then
            INSTALL_CMD="$INSTALL_CMD --force-reinstall"
        fi
        
        if $INSTALL_CMD; then
            echo -e "${GREEN}✓ Successfully installed dataknobs-$package${NC}"
        else
            echo -e "${RED}✗ Failed to install dataknobs-$package${NC}"
            exit 1
        fi
    done
else
    echo -e "${YELLOW}Installing in production mode...${NC}"
    
    # In prod mode, build first if needed
    if [[ ! -d "packages/common/dist" ]]; then
        echo -e "${YELLOW}No built packages found. Building first...${NC}"
        "$ROOT_DIR/bin/build-packages.sh"
    fi
    
    # Install from built wheels
    for package in "${PACKAGES[@]}"; do
        echo -e "\n${YELLOW}Installing dataknobs-$package...${NC}"
        
        # Find the latest wheel
        WHEEL=$(find "packages/$package/dist" -name "*.whl" | sort -V | tail -n1)
        
        if [[ -z "$WHEEL" ]]; then
            echo -e "${RED}No wheel found for dataknobs-$package${NC}"
            exit 1
        fi
        
        INSTALL_CMD="uv pip install $WHEEL"
        if [[ "$FORCE_REINSTALL" == true ]]; then
            INSTALL_CMD="$INSTALL_CMD --force-reinstall"
        fi
        
        if $INSTALL_CMD; then
            echo -e "${GREEN}✓ Successfully installed dataknobs-$package${NC}"
        else
            echo -e "${RED}✗ Failed to install dataknobs-$package${NC}"
            exit 1
        fi
    done
fi

echo -e "\n${GREEN}All packages installed successfully!${NC}"

# Show installed packages
echo -e "\n${YELLOW}Installed dataknobs packages:${NC}"
uv pip list | grep dataknobs