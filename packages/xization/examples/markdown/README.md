# Markdown Chunking Examples

This directory contains example files demonstrating the markdown chunking utilities.

## Files

### Example Scripts

- **example_usage.py**: Comprehensive examples showing various features
  - Basic parsing and chunking
  - Heading inclusion options
  - Different output formats
  - Chunk size control
  - File streaming
  - JSON export
  - RAG use case simulation

### Example Documents

- **example_document.md**: Sample markdown document with multiple heading levels
  - Demonstrates hierarchical structure
  - Complex nested headings
  - Real-world RAG system documentation

- **constructs_example.md**: Example showing special markdown constructs
  - Code blocks (fenced with language tags)
  - Lists (ordered and unordered)
  - Tables
  - Horizontal rules
  - Demonstrates atomic construct handling

## Running the Examples

From the `examples/markdown` directory:

```bash
# Run all examples
uv run python example_usage.py

# Or run from package root
cd ../..
uv run python examples/markdown/example_usage.py
```

## CLI Examples

### Analyze a document

```bash
uv run python scripts/md_cli.py info examples/markdown/example_document.md
```

### Chunk a document

```bash
# Basic chunking
uv run python scripts/md_cli.py chunk examples/markdown/example_document.md

# With custom parameters
uv run python scripts/md_cli.py chunk \
  examples/markdown/constructs_example.md \
  --max-size 500 \
  --overlap 50 \
  --show-metadata

# Output as JSON
uv run python scripts/md_cli.py chunk \
  examples/markdown/example_document.md \
  --output-format json \
  --output chunks.json
```

### View tree structure

```bash
uv run python scripts/md_cli.py parse \
  examples/markdown/example_document.md \
  --show-tree
```

## Creating Your Own Examples

You can create your own markdown files to test with different:
- Heading structures
- Content types (code, tables, lists)
- Document sizes
- Nesting depths

The utilities will preserve the semantic structure and generate appropriate chunks for RAG applications.
