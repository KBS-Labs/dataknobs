"""Core data structures for AI knowledge bases and data processing.

The dataknobs-structures package provides fundamental data structures for building
AI applications, knowledge bases, and data processing pipelines. It includes tree
structures, document containers, record stores, and conditional dictionaries.

## Modules

### Tree - Hierarchical data structures
The Tree class provides a flexible node-based tree structure with:
- Parent-child relationships with bidirectional links
- Depth-first and breadth-first traversal
- Node search and filtering
- Path finding and common ancestor detection
- Graphviz visualization support

### Document - Text with metadata
Classes for managing text documents with associated metadata:
- Text: Container combining content and metadata
- TextMetaData: Structured metadata with IDs and labels
- MetaData: Generic key-value metadata container

### RecordStore - Tabular data management
A flexible record store that represents data as:
- List of dictionaries (Python native)
- pandas DataFrame (data analysis)
- TSV/CSV files (disk persistence)

### cdict - Conditional dictionary
A dictionary that validates items before acceptance using the strategy pattern.
Rejected items are tracked separately for inspection.

## Quick Examples

### Using Tree
```python
from dataknobs_structures import Tree

# Build tree structure
root = Tree("root")
child1 = root.add_child("child1")
child2 = root.add_child("child2")
grandchild = child1.add_child("grandchild")

# Search and traverse
found = root.find_nodes(lambda n: "child" in str(n.data))
edges = root.get_edges()  # All parent-child pairs
```

### Using Text with metadata
```python
from dataknobs_structures import Text, TextMetaData

# Create document with metadata
metadata = TextMetaData(
    text_id="doc_001",
    text_label="article",
    author="Alice",
    category="technology"
)
doc = Text("This is the document content...", metadata)

print(doc.text_id)     # "doc_001"
print(doc.text_label)  # "article"
print(doc.metadata.get_value("author"))  # "Alice"
```

### Using RecordStore
```python
from dataknobs_structures import RecordStore

# Create store with disk backing
store = RecordStore("/data/users.tsv")

# Add records
store.add_rec({"id": 1, "name": "Alice", "age": 30})
store.add_rec({"id": 2, "name": "Bob", "age": 25})

# Access as DataFrame or list
df = store.df
records = store.records

# Persist changes
store.save()
```

### Using cdict
```python
from dataknobs_structures import cdict

# Only accept positive integers
positive = cdict(lambda d, k, v: isinstance(v, int) and v > 0)
positive['a'] = 5    # Accepted
positive['b'] = -1   # Rejected

print(positive)          # {'a': 5}
print(positive.rejected)  # {'b': -1}
```

## Design Philosophy

The structures in this package are designed to be:
1. **Simple** - Easy to understand and use with minimal boilerplate
2. **Flexible** - Support multiple representations and use cases
3. **Composable** - Work well together in larger systems
4. **Practical** - Solve real problems in AI and data workflows

## Installation

```bash
pip install dataknobs-structures
```

For more detailed documentation, see the individual class and function docstrings.
"""

from dataknobs_structures.conditional_dict import cdict
from dataknobs_structures.document import Text, TextMetaData
from dataknobs_structures.record_store import RecordStore
from dataknobs_structures.tree import Tree, build_tree_from_string

__version__ = "1.0.3"

__all__ = [
    "RecordStore",
    "Text",
    "TextMetaData",
    "Tree",
    "build_tree_from_string",
    "cdict",
]
