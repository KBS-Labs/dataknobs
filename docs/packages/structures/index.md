# Dataknobs Structures

The `dataknobs-structures` package provides core data structures for building AI knowledge base systems.

## Installation

```bash
pip install dataknobs-structures
```

## Overview

This package includes several key data structures:

- **Tree**: Hierarchical tree structure for organizing data
- **Document**: Text documents with metadata
- **RecordStore**: Key-value storage with record management
- **ConditionalDict**: Dictionary with conditional acceptance logic

## Quick Start

### Tree Structure

The `Tree` class provides a flexible hierarchical data structure:

```python
from dataknobs_structures import Tree

# Create a tree
root = Tree("root_data")

# Add children
child1 = root.add_child("child1_data")
child2 = root.add_child("child2_data")

# Navigate the tree. `children` answers a snapshot tuple, not the live list,
# and printing a node renders the subtree standing under it.
print(root.children)      # (child1_data, child2_data)
print(child1.parent.data) # root_data
print(root.depth)         # 0

# Tree operations
all_descendants = root.find_nodes(lambda node: True, include_self=False)
leaves = root.collect_terminal_nodes()
```

### Documents

Work with text documents and metadata:

```python
from dataknobs_structures import Text, TextMetaData

# Create metadata
metadata = TextMetaData(
    text_id=1,
    text_label="document",
    source="input.txt",
    timestamp="2024-01-01"
)

# Create a document
doc = Text("This is the document content", metadata)

# Access properties. The content is `.text`; `text_id` and `text_label` are
# promoted from the metadata, and the rest stays in `.data`.
print(doc.text)           # This is the document content
print(doc.text_id)        # 1
print(doc.metadata.data)
# {'text_id': 1, 'text_label': 'document', 'source': 'input.txt', 'timestamp': '2024-01-01'}
```

### Record Store

Manage collections of records:

`RecordStore` is a **row** store, not a key-value store: records are appended
dicts, kept in step with a pandas DataFrame and optionally a TSV on disk.

```python
from dataknobs_structures import RecordStore

# The backing file path is required and positional; pass None for memory only.
store = RecordStore(None)

# Add records -- each is a row, and there is no key to address it by.
store.add_rec({"user": "alice", "score": 100})
store.add_rec({"user": "bob", "score": 95})

print(len(store.records))          # 2
print(store.records[0]["user"])    # alice

# The same rows as a DataFrame, built on demand.
print(list(store.df.columns))      # ['user', 'score']
print(store.df["score"].max())     # 100

# Selecting and updating go through the DataFrame or the list; the store
# exposes no get/update/delete of its own. `clear` empties it.
store.clear()
print(store.records)               # []
```

Give it a path instead of `None` and `save()` writes the TSV, `restore()` reads
it back:

```python
import tempfile
from pathlib import Path

from dataknobs_structures import RecordStore

path = Path(tempfile.mkdtemp()) / "results.tsv"
store = RecordStore(str(path))
store.add_rec({"user": "alice", "score": 100})
store.save()

# repr, so the tab that makes it a TSV is visible rather than looking like
# run-together spaces.
print(repr(path.read_text()))
# 'user\tscore\nalice\t100\n'

reloaded = RecordStore(str(path))
reloaded.restore()
print(reloaded.records)            # [{'user': 'alice', 'score': 100}]
```

### Conditional Dictionary

A dictionary that conditionally accepts items:

```python
from dataknobs_structures import cdict

# Create with acceptance function
def accept_positive(d, key, value):
    """Only accept positive numbers"""
    return isinstance(value, (int, float)) and value > 0

cd = cdict(accept_positive)
cd["a"] = 10   # Accepted
cd["b"] = -5   # Rejected

print(cd)  # {'a': 10}
print(cd.rejected)  # {'b': -5}
```

## Key Features

### Tree Features

- **Hierarchical Structure**: Build complex nested structures
- **Traversal Methods**: Various ways to navigate and search the tree
- **Flexible Data**: Store any Python object as node data
- **Parent-Child Relationships**: Automatic relationship management

### Document Features

- **Metadata Management**: Rich metadata support for documents
- **Text Processing Ready**: Designed to work with text processing pipelines
- **Extensible**: Easy to extend with custom document types

### RecordStore Features

- **CRUD Operations**: Complete Create, Read, Update, Delete support
- **Batch Operations**: Process multiple records efficiently
- **Serialization**: Save and load record stores
- **Query Support**: Filter and search records

## Advanced Usage

### Tree Serialization

```python
import json

from dataknobs_structures import Tree

# Create a tree
tree = Tree({"type": "root", "value": 100})
tree.add_child({"type": "child", "value": 50})

# There is no to_dict()/from_dict(). Walk the tree to build the shape you
# want: a node exposes `data` and `children`, and `children` is None until a
# node first holds a child.
def to_record(node):
    return {
        "data": node.data,
        "children": [to_record(child) for child in node.children or ()],
    }

def from_record(record):
    node = Tree(record["data"])
    for child in record["children"]:
        node.add_child(from_record(child))
    return node

# Save to JSON
with open("tree.json", "w") as f:
    json.dump(to_record(tree), f)

# Load from JSON and reconstruct
with open("tree.json") as f:
    new_tree = from_record(json.load(f))

# The built-in round trip is the parenthesized string form, but it rebuilds
# every node's data as a string -- so it suits string payloads, not the dicts
# used here.
#   build_tree_from_string(tree.as_string())
```

### Custom Document Types

```python
from dataknobs_structures import Text, TextMetaData

class Article(Text):
    """Custom document type for articles"""
    
    def __init__(self, title, content, author, metadata=None):
        if metadata is None:
            metadata = TextMetaData(text_id=None)
        super().__init__(content, metadata)
        self.title = title
        self.author = author
    
    @property
    def word_count(self):
        return len(self.text.split())
    
    @property
    def summary(self):
        """Return first 100 characters as summary"""
        return self.text[:100] + "..." if len(self.text) > 100 else self.text

# Use custom document
article = Article(
    title="Introduction to Dataknobs",
    content="Dataknobs is a powerful library for knowledge management...",
    author="John Doe"
)

print(f"Title: {article.title}")
print(f"Word count: {article.word_count}")
print(f"Summary: {article.summary}")
```

## API Reference

For complete API documentation, see the [Structures API Reference](../../api/dataknobs-structures.md).

## Integration with Other Packages

The structures package is designed to work seamlessly with other Dataknobs packages:

```python
from dataknobs_structures import Tree
from dataknobs_utils import json_utils
from dataknobs_xization import normalize

# Create structure
tree = Tree("root")
child = tree.add_child({"text": "Hello WORLD!"})

# Use with utils -- get_value reads into a loaded object using indexed dot
# notation; there is no to_json() helper, so serialize with the stdlib.
print(json_utils.get_value(child.data, "text"))   # Hello WORLD!

# Use with text processing
normalized = normalize.basic_normalization_fn(child.data["text"])
child.data["normalized"] = normalized
```

## Best Practices

1. **Use Type Hints**: Always use type hints for better code clarity
2. **Handle Metadata**: Always include metadata for documents when possible
3. **Error Handling**: Wrap operations in try-except blocks for production code
4. **Memory Management**: Be mindful of tree size for large hierarchies
5. **Serialization**: Use built-in serialization methods for persistence

## Next Steps

- Explore [Tree API Documentation](tree.md)
- Learn about [Document Processing](document.md)
- See [Integration Examples](../../examples/index.md)