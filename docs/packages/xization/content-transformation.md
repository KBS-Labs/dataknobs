# Content Transformation

The Content Transformation module provides tools for converting structured data formats (JSON, YAML, CSV) into well-formatted markdown suitable for RAG ingestion and chunking.

## Overview

When building RAG (Retrieval-Augmented Generation) systems, you often need to ingest structured data like JSON configuration files, YAML documentation, or CSV datasets. The `ContentTransformer` class converts these formats into markdown with appropriate heading hierarchies, enabling the markdown chunker to create semantic boundaries around logical content units.

## Quick Start

```python
from dataknobs_xization import ContentTransformer, json_to_markdown

# Simple conversion
data = {"name": "My Item", "description": "A description"}
markdown = json_to_markdown(data)

# Or use the transformer class
transformer = ContentTransformer()
markdown = transformer.transform_json(data)
```

## ContentTransformer Class

### Initialization

```python
from dataknobs_xization import ContentTransformer

transformer = ContentTransformer(
    base_heading_level=2,           # Starting heading level (default: 2)
    include_field_labels=True,      # Bold field names in output (default: True)
    code_block_fields=["example"],  # Fields to render as code blocks
    list_fields=["steps", "items"]  # Fields to render as bullet lists
)
```

### Default Field Handling

By default, the transformer treats certain field names specially:

**Code Block Fields** (rendered as fenced code blocks):
- `example`
- `code`
- `snippet`

**List Fields** (rendered as bullet lists):
- `items`
- `steps`
- `objectives`
- `symptoms`
- `solutions`

You can customize these lists during initialization.

## Generic Transformation

Without a schema, the transformer uses intelligent defaults:

```python
data = {
    "name": "Chain of Thought",
    "description": "Step by step reasoning technique",
    "steps": ["Break down problem", "Show work", "Conclude"],
    "example": "Let's think step by step..."
}

result = transformer.transform_json(data)
```

Output:
```markdown
## Chain of Thought

**Description**: Step by step reasoning technique

### Steps

- Break down problem
- Show work
- Conclude

### Example

```
Let's think step by step...
```

---
```

### Title Detection

The transformer automatically detects title fields in this order:
1. `name`
2. `title`
3. `id`
4. `key`

### Nested Structures

Nested dictionaries become subsections:

```python
data = {
    "name": "Config",
    "database": {
        "host": "localhost",
        "port": 5432
    }
}
```

Produces:
```markdown
## Config

### Database

**Host**: localhost

**Port**: 5432
```

## Custom Schemas

For specialized formatting, register custom schemas:

```python
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
```

### Schema Definition

| Key | Description |
|-----|-------------|
| `title_field` | Field to use as the main heading (required) |
| `description_field` | Field for intro text without a heading |
| `sections` | List of section definitions |
| `metadata_fields` | Fields to render as bold key-value pairs |

### Section Formats

| Format | Description |
|--------|-------------|
| `text` (default) | Plain text paragraph |
| `code` | Fenced code block (optionally with `language`) |
| `list` | Bullet list |
| `subsections` | Nested key-value pairs or list of items |

### Example Usage

```python
patterns = [
    {
        "name": "Chain of Thought",
        "description": "A prompting technique for complex reasoning",
        "use_case": "Use for multi-step problems requiring logical reasoning",
        "example": "Let's think step by step:\n1. First, ...\n2. Then, ...",
        "variations": ["Zero-shot CoT", "Manual CoT", "Self-consistency"],
        "category": "reasoning",
        "difficulty": "intermediate"
    }
]

markdown = transformer.transform_json(patterns, schema="pattern")
```

Output:
```markdown
## Chain of Thought

**Category**: reasoning
**Difficulty**: intermediate

A prompting technique for complex reasoning

### When to Use

Use for multi-step problems requiring logical reasoning

### Example

```python
Let's think step by step:
1. First, ...
2. Then, ...
```

### Variations

- Zero-shot CoT
- Manual CoT
- Self-consistency

---
```

## YAML Transformation

```python
# From YAML string
yaml_content = """
name: My Config
settings:
  timeout: 30
  retries: 3
"""
markdown = transformer.transform_yaml(yaml_content)

# From YAML file
markdown = transformer.transform_yaml("config.yaml")

# With schema
markdown = transformer.transform_yaml("config.yaml", schema="config")
```

## CSV Transformation

```python
# From CSV string
csv_content = "name,value,description\nItem1,100,First item\nItem2,200,Second item"
markdown = transformer.transform_csv(csv_content)

# From CSV file
markdown = transformer.transform_csv("data.csv")

# With custom title field
markdown = transformer.transform_csv("data.csv", title_field="name")

# With document title
markdown = transformer.transform_csv("data.csv", title="My Dataset")
```

Each row becomes a section with the first column (or `title_field`) as the heading.

## Convenience Functions

For quick one-off conversions:

```python
from dataknobs_xization import json_to_markdown, yaml_to_markdown, csv_to_markdown

# JSON to markdown
md = json_to_markdown(data, title="Document Title", base_heading_level=2)

# YAML to markdown
md = yaml_to_markdown("config.yaml", title="Configuration")

# CSV to markdown
md = csv_to_markdown("data.csv", title="Data", title_field="name")
```

## Integration with RAG

### With RAGKnowledgeBase

The `RAGKnowledgeBase` class provides direct methods for loading JSON, YAML, and CSV:

```python
from dataknobs_bots.knowledge import RAGKnowledgeBase
from dataknobs_xization import ContentTransformer

# Create knowledge base
kb = RAGKnowledgeBase(...)

# Create transformer with custom schema
transformer = ContentTransformer()
transformer.register_schema("pattern", {...})

# Load JSON directly
await kb.load_json_document(
    "patterns.json",
    schema="pattern",
    transformer=transformer,
    metadata={"source": "patterns"}
)

# Load YAML
await kb.load_yaml_document("config.yaml", metadata={"type": "config"})

# Load CSV
await kb.load_csv_document("data.csv", title_field="name")
```

### Why Convert to Markdown?

1. **Semantic Chunking**: Markdown headings create natural boundaries for chunks
2. **Hierarchy Preservation**: Nested structures become heading hierarchies
3. **RAG Optimization**: Chunks maintain context through heading paths
4. **Consistent Processing**: All content goes through the same chunking pipeline

## API Reference

### ContentTransformer

```python
class ContentTransformer:
    def __init__(
        self,
        base_heading_level: int = 2,
        include_field_labels: bool = True,
        code_block_fields: list[str] | None = None,
        list_fields: list[str] | None = None,
    )

    def register_schema(self, name: str, schema: dict[str, Any]) -> None

    def transform(
        self,
        content: Any,
        format: str = "json",
        schema: str | None = None,
        title: str | None = None,
    ) -> str

    def transform_json(
        self,
        data: dict[str, Any] | list[Any],
        schema: str | None = None,
        title: str | None = None,
    ) -> str

    def transform_yaml(
        self,
        content: str | Path,
        schema: str | None = None,
        title: str | None = None,
    ) -> str

    def transform_csv(
        self,
        content: str | Path,
        title: str | None = None,
        title_field: str | None = None,
    ) -> str
```

### Convenience Functions

```python
def json_to_markdown(
    data: dict[str, Any] | list[Any],
    title: str | None = None,
    base_heading_level: int = 2,
) -> str

def yaml_to_markdown(
    content: str | Path,
    title: str | None = None,
    base_heading_level: int = 2,
) -> str

def csv_to_markdown(
    content: str | Path,
    title: str | None = None,
    title_field: str | None = None,
    base_heading_level: int = 2,
) -> str
```
