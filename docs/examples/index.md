# Examples

Practical examples of using Dataknobs packages.

## Quick Examples

- [Basic Tree Operations](basic-tree.md)
- [Document Processing](document-processing.md)
- [Text Normalization](text-normalization.md)
- [Elasticsearch Integration](elasticsearch-integration.md)

## Complete Applications

### Document Analysis Pipeline

```python
from dataknobs_structures import Tree, Text, TextMetaData
from dataknobs_xization import basic_normalization_fn
from dataknobs_utils import json_utils

def analyze_document(content, metadata):
    # Create document
    meta = TextMetaData(**metadata)
    doc = Text(content, meta)
    
    # Normalize text
    normalized = basic_normalization_fn(doc.content)
    
    # Extract structure
    tree = Tree()
    # ... build tree from document structure
    
    return {
        "original": doc.content,
        "normalized": normalized,
        "metadata": metadata
    }
```
