#!/bin/bash
# Check if Ollama is running and accessible

# Colors for output
if [ -t 1 ] && [ -n "${TERM:-}" ] && [ "${TERM}" != "dumb" ]; then
    GREEN='\033[0;32m'
    RED='\033[0;31m'
    YELLOW='\033[1;33m'
    NC='\033[0m' # No Color
else
    GREEN=''
    RED=''
    YELLOW=''
    NC=''
fi

OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"

# Check if Ollama is running
if curl -s "${OLLAMA_HOST}/api/tags" > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Ollama is running${NC} at ${OLLAMA_HOST}"

    # Try to get list of models
    MODELS=$(curl -s "${OLLAMA_HOST}/api/tags" 2>/dev/null)
    if [ $? -eq 0 ] && [ -n "$MODELS" ]; then
        MODEL_COUNT=$(echo "$MODELS" | grep -o '"name"' | wc -l | tr -d ' ')
        if [ "$MODEL_COUNT" -gt 0 ]; then
            echo -e "${GREEN}✓ ${MODEL_COUNT} model(s) available${NC}"
        else
            echo -e "${YELLOW}⚠ No models installed${NC}"
            echo -e "  Install models with: ${YELLOW}ollama pull <model-name>${NC}"
        fi
    fi

    exit 0
else
    echo -e "${RED}✗ Ollama is not running or not accessible${NC}"
    echo ""
    echo "To fix this:"
    echo "  1. Install Ollama:"
    echo "     - macOS: ${YELLOW}brew install ollama${NC}"
    echo "     - Linux: ${YELLOW}curl -fsSL https://ollama.ai/install.sh | sh${NC}"
    echo "     - Windows: Download from https://ollama.ai/download"
    echo ""
    echo "  2. Start Ollama:"
    echo "     ${YELLOW}ollama serve${NC}"
    echo ""
    echo "  Or skip Ollama tests:"
    echo "     ${YELLOW}export TEST_OLLAMA=false${NC}"
    echo "     ${YELLOW}dk test${NC}"
    echo ""
    exit 1
fi
