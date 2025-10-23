# dataknobs-xization

Text normalization and tokenization tools.

## Installation

```bash
pip install dataknobs-xization
```

## Features

- **Markdown Chunking**: Parse and chunk markdown documents for RAG applications
  - Preserves heading hierarchy and semantic structure
  - Supports code blocks, tables, lists, and other markdown constructs
  - Streaming support for large documents
  - Flexible configuration for chunk size, overlap, and heading inclusion
- **Text Normalization**: Standardize text for consistent processing
- **Masking Tokenizer**: Advanced tokenization with masking capabilities
- **Annotations**: Text annotation system
- **Authorities**: Authority management for text processing
- **Lexicon**: Lexicon-based text analysis

## Usage

### Markdown Chunking

```python
from dataknobs_xization import parse_markdown, chunk_markdown_tree

# Parse markdown into tree structure
markdown_text = """
# User Guide
## Installation
Install the package using pip.
"""

tree = parse_markdown(markdown_text)

# Generate chunks for RAG
chunks = chunk_markdown_tree(tree, max_chunk_size=500)

for chunk in chunks:
    print(f"Headings: {chunk.metadata.get_heading_path()}")
    print(f"Text: {chunk.text}\n")
```

For more details, see the [Markdown Chunking documentation](docs/markdown/MARKDOWN_CHUNKING.md).

### Text Normalization and Tokenization

```python
from dataknobs_xization import normalize, MaskingTokenizer

# Text normalization
normalized = normalize.normalize_text("Hello, World!")

# Tokenization with masking
tokenizer = MaskingTokenizer()
tokens = tokenizer.tokenize("This is a sample text.")

# Working with annotations
from dataknobs_xization import annotations
doc = annotations.create_document("Sample text", {"metadata": "value"})
```

## Dependencies

This package depends on:
- `dataknobs-common`
- `dataknobs-structures`
- `dataknobs-utils`
- nltk

## License

See LICENSE file in the root repository.