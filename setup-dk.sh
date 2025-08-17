#!/bin/bash
# Setup script to install dk command globally

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DK_SCRIPT="$SCRIPT_DIR/bin/dk"

echo "DataKnobs dk command setup"
echo "=========================="
echo ""

# Check if dk script exists
if [ ! -f "$DK_SCRIPT" ]; then
    echo "Error: dk script not found at $DK_SCRIPT"
    exit 1
fi

# Determine install location
if [ -d "$HOME/.local/bin" ]; then
    INSTALL_DIR="$HOME/.local/bin"
elif [ -d "$HOME/bin" ]; then
    INSTALL_DIR="$HOME/bin"
else
    INSTALL_DIR="$HOME/.local/bin"
    echo "Creating $INSTALL_DIR directory..."
    mkdir -p "$INSTALL_DIR"
fi

# Create symlink
echo "Creating symlink in $INSTALL_DIR..."
ln -sf "$DK_SCRIPT" "$INSTALL_DIR/dk"

# Check if directory is in PATH
if [[ ":$PATH:" != *":$INSTALL_DIR:"* ]]; then
    echo ""
    echo "⚠️  Warning: $INSTALL_DIR is not in your PATH"
    echo ""
    echo "Add this to your shell config (.bashrc, .zshrc, etc.):"
    echo "  export PATH=\"\$PATH:$INSTALL_DIR\""
    echo ""
else
    echo "✓ dk command installed successfully!"
    echo ""
    echo "You can now use 'dk' from anywhere in this project."
    echo "Try: dk help"
fi