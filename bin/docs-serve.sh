#!/bin/bash
# Serve documentation locally with hot reload

set -e

echo "📚 Starting Dataknobs documentation server..."
echo "================================================"

# Install dependencies if needed
if ! command -v mkdocs &> /dev/null; then
    echo "Installing MkDocs dependencies..."
    uv pip install mkdocs mkdocs-material mkdocstrings[python] mkdocs-monorepo-plugin mkdocs-awesome-pages-plugin mkdocs-git-revision-date-localized-plugin
fi

# Serve documentation
echo "Starting documentation server at http://localhost:8000"
echo "Press Ctrl+C to stop"
NO_MKDOCS_2_WARNING=1 mkdocs serve --dev-addr localhost:8000

echo "✅ Documentation server stopped"