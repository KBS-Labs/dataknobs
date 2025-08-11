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

# Navigate the tree
print(root.children)  # List of child nodes
print(child1.parent)  # Parent node
print(root.depth)     # Depth in tree (0 for root)

# Tree operations
all_descendants = root.get_all_descendants()
leaves = root.get_leaves()
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

# Access properties
print(doc.text)       # Document text
print(doc.text_id)    # ID from metadata
print(doc.metadata.data)  # All metadata
```

### Record Store

Manage collections of records:

```python
from dataknobs_structures import RecordStore

# Create a record store
store = RecordStore()

# Add records
store.add_record("key1", {"data": "value1"})
store.add_record("key2", {"data": "value2"})

# Retrieve records
record = store.get_record("key1")
all_records = store.get_all_records()

# Update records
store.update_record("key1", {"data": "updated_value"})

# Delete records
store.delete_record("key2")
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
from dataknobs_structures import Tree
import json

# Create a tree
tree = Tree({"type": "root", "value": 100})
tree.add_child({"type": "child", "value": 50})

# Serialize to dict
tree_dict = tree.to_dict()

# Save to JSON
with open("tree.json", "w") as f:
    json.dump(tree_dict, f)

# Load from JSON
with open("tree.json", "r") as f:
    tree_dict = json.load(f)
    
# Reconstruct tree
new_tree = Tree.from_dict(tree_dict)
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

# Use with utils
json_str = json_utils.to_json(tree.to_dict())

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