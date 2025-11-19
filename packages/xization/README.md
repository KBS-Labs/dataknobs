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
- **Content Transformation**: Convert JSON, YAML, and CSV to markdown for RAG ingestion
  - Generic conversion that preserves structure through headings
  - Custom schemas for specialized formatting
  - Configurable formatting options
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

### Content Transformation

Convert structured data (JSON, YAML, CSV) to well-formatted markdown for RAG ingestion:

```python
from dataknobs_xization import ContentTransformer, json_to_markdown

# Quick conversion
data = [
    {"name": "Chain of Thought", "description": "Step by step reasoning"},
    {"name": "Few-Shot", "description": "Learning from examples"}
]
markdown = json_to_markdown(data, title="Prompt Patterns")

# Or use the transformer class for more control
transformer = ContentTransformer(
    base_heading_level=2,
    include_field_labels=True,
    code_block_fields=["example", "code"],
    list_fields=["steps", "items"]
)

# Transform JSON
result = transformer.transform_json(data)

# Transform YAML
result = transformer.transform_yaml("config.yaml")

# Transform CSV
result = transformer.transform_csv("data.csv", title_field="name")
```

#### Custom Schemas

Register schemas for specialized formatting of known data structures:

```python
transformer = ContentTransformer()

# Register a schema for prompt patterns
transformer.register_schema("pattern", {
    "title_field": "name",
    "description_field": "description",
    "sections": [
        {"field": "use_case", "heading": "When to Use"},
        {"field": "example", "heading": "Example", "format": "code", "language": "python"},
        {"field": "variations", "heading": "Variations", "format": "list"}
    ],
    "metadata_fields": ["category", "difficulty"]
})

# Use the schema
patterns = [
    {
        "name": "Chain of Thought",
        "description": "Prompting technique for complex reasoning",
        "use_case": "Multi-step problems requiring logical reasoning",
        "example": "Let's think step by step...",
        "category": "reasoning",
        "difficulty": "intermediate"
    }
]

markdown = transformer.transform_json(patterns, schema="pattern")
```

#### Convenience Functions

```python
from dataknobs_xization import json_to_markdown, yaml_to_markdown, csv_to_markdown

# Quick conversions
md = json_to_markdown(data, title="My Data")
md = yaml_to_markdown("config.yaml", title="Config")
md = csv_to_markdown("data.csv", title_field="name")
```

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