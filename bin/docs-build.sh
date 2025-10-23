#!/bin/bash
# Build documentation for deployment

set -e

echo "📚 Building Dataknobs documentation..."
echo "======================================"

# Install dependencies if needed
if ! command -v mkdocs &> /dev/null; then
    echo "Installing MkDocs dependencies..."
    uv pip install mkdocs mkdocs-material mkdocstrings[python] mkdocs-monorepo-plugin mkdocs-awesome-pages-plugin mkdocs-git-revision-date-localized-plugin
fi

# Clean previous build
echo "Cleaning previous build..."
rm -rf site/

# Build documentation
echo "Building documentation..."
# Suppress DeprecationWarning from asttokens (incompatible with astroid 4.x)
PYTHONWARNINGS="ignore::DeprecationWarning:asttokens" mkdocs build --strict

# Check if build was successful
if [ -d "site" ]; then
    echo "✅ Documentation built successfully!"
    echo "Output directory: ./site"
    echo ""
    echo "To preview locally, run:"
    echo "  python -m http.server -d site 8000"
    echo ""
    echo "To deploy to GitHub Pages, run:"
    echo "  mkdocs gh-deploy"
else
    echo "❌ Documentation build failed!"
    exit 1
fi