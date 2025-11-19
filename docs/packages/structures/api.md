# Structures API Reference

Complete API reference for the `dataknobs-structures` package.

> **📖 Also see:** [Auto-generated API Reference](../../api/reference/structures.md) - Complete documentation from source code docstrings

This page provides curated examples and usage patterns. The auto-generated reference provides exhaustive technical documentation with all methods, parameters, and type annotations.

---

## Package Imports

```python
from dataknobs_structures import (
    Tree, build_tree_from_string,
    Text, TextMetaData,
    RecordStore,
    cdict
)
```

## Package Information

```python
from dataknobs_structures import __version__
print(__version__)  # "1.0.0"
```

## Classes Overview

| Class | Purpose | Module |
|-------|---------|---------|
| [`Tree`](tree.md) | Hierarchical tree structure | `dataknobs_structures.tree` |
| [`Text`](document.md) | Text document with metadata | `dataknobs_structures.document` |
| [`TextMetaData`](document.md) | Document metadata container | `dataknobs_structures.document` |
| [`RecordStore`](record-store.md) | Record collection management | `dataknobs_structures.record_store` |
| [`cdict`](conditional-dict.md) | Conditional dictionary | `dataknobs_structures.conditional_dict` |

## Functions Overview

| Function | Purpose | Module |
|----------|---------|---------|
| [`build_tree_from_string`](tree.md) | Create tree from string representation | `dataknobs_structures.tree` |

## Module: tree

### Classes

#### Tree
```python
class Tree:
    def __init__(self, data: Any, parent: Union[Tree, Any] = None, child_pos: Optional[int] = None)
```

**Properties:**
- `data: Any` - Node data
- `children: Optional[List[Tree]]` - Child nodes
- `parent: Optional[Tree]` - Parent node  
- `root: Tree` - Root of tree
- `depth: int` - Node depth (0-based)
- `num_children: int` - Number of children
- `sibnum: int` - Sibling position
- `num_siblings: int` - Total siblings including self
- `next_sibling: Optional[Tree]` - Next sibling
- `prev_sibling: Optional[Tree]` - Previous sibling

**Methods:**
- `add_child(node_or_data, child_pos=None) -> Tree`
- `add_edge(parent_node_or_data, child_node_or_data) -> Tuple[Tree, Tree]`
- `prune() -> Optional[Tree]`
- `find_nodes(accept_node_fn, traversal="dfs", include_self=True, only_first=False, highest_only=False) -> List[Tree]`
- `collect_terminal_nodes(accept_node_fn=None) -> List[Tree]`
- `get_edges(traversal="bfs", include_self=True, as_data=True) -> List[Tuple[Union[Tree, Any], Union[Tree, Any]]]`
- `get_path() -> List[Tree]`
- `is_ancestor(other, self_is_ancestor=False) -> bool`
- `find_deepest_common_ancestor(other) -> Optional[Tree]`
- `has_children() -> bool`
- `has_parent() -> bool`
- `get_deepest_left() -> Tree`
- `get_deepest_right() -> Tree`
- `as_string(delim=" ", multiline=False) -> str`
- `build_dot(node_name_fn=None, **kwargs) -> graphviz.Digraph`

### Functions

#### build_tree_from_string
```python
def build_tree_from_string(from_string: str) -> Tree
```
Builds a tree from its string representation.

## Module: document

### Classes

#### MetaData
```python
class MetaData:
    def __init__(self, key_data: dict[str, Any], **kwargs: Any)
```

**Properties:**
- `data: dict[str, Any]` - Complete metadata

**Methods:**
- `get_value(attribute: str, missing: str | None = None) -> Any`

#### TextMetaData
```python
class TextMetaData(MetaData):
    def __init__(self, text_id: Any, text_label: str = "text", **kwargs: Any)
```

**Properties:**
- `text_id: Any` - Text identifier
- `text_label: str | Any` - Text label
- Inherits all `MetaData` properties

#### Text
```python
class Text:
    def __init__(self, text: str, metadata: TextMetaData | None)
```

**Properties:**
- `text: str` - Document content
- `text_id: Any` - Text ID from metadata
- `text_label: str` - Text label from metadata
- `metadata: TextMetaData` - Document metadata

### Constants

- `TEXT_ID_ATTR = "text_id"`
- `TEXT_LABEL_ATTR = "text_label"`
- `TEXT_LABEL = "text"`

## Module: record_store

### Classes

#### RecordStore
```python
class RecordStore:
    def __init__(self, tsv_fpath: Optional[str], df: Optional[pd.DataFrame] = None, sep: str = "\t")
```

**Properties:**
- `df: Optional[pd.DataFrame]` - Records as DataFrame
- `records: List[Dict[str, Any]]` - Records as list of dicts

**Methods:**
- `add_rec(rec: Dict[str, Any]) -> None`
- `clear() -> None`
- `save() -> None`
- `restore(df: Optional[pd.DataFrame] = None) -> None`

## Module: conditional_dict

### Classes

#### cdict
```python
class cdict(dict):
    def __init__(self, accept_fn: Callable[[Dict, Any, Any], bool], *args: Any, **kwargs: Any)
```

**Properties:**
- `rejected: Dict` - Rejected key-value pairs
- Inherits all standard `dict` properties

**Methods:**
- `__setitem__(key: Any, value: Any) -> None`
- `setdefault(key: Any, default: Any = None) -> Any`
- `update(*args: Any, **kwargs: Any) -> None`
- Inherits all standard `dict` methods

## Type Annotations

### Common Types

```python
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
```

### Tree Types

```python
TreeNode = Tree
TreeData = Any
AcceptNodeFunction = Callable[[Tree], bool]
```

### Document Types

```python
TextContent = str
TextID = Any
MetadataDict = Dict[str, Any]
```

### RecordStore Types

```python
Record = Dict[str, Any]
RecordList = List[Record]
```

### ConditionalDict Types

```python
AcceptFunction = Callable[[Dict, Any, Any], bool]
```

## Error Handling

### Common Exceptions

Most methods follow standard Python exception patterns:

- `TypeError`: Invalid argument types
- `ValueError`: Invalid argument values
- `KeyError`: Missing dictionary keys
- `IndexError`: Invalid list/array indices
- `AttributeError`: Missing attributes

### Tree-Specific Errors

```python
# Node operations
try:
    node = tree.find_nodes(lambda n: n.data == "target")[0]
except IndexError:
    print("No matching nodes found")

# Parent-child relationships
if tree.has_parent():
    parent = tree.parent
else:
    print("Node is root")
```

### RecordStore Errors

```python
try:
    store = RecordStore("/invalid/path/file.tsv")
    store.save()
except (PermissionError, FileNotFoundError) as e:
    print(f"File operation failed: {e}")
```

## Usage Patterns

### Factory Functions

```python
def create_document(content: str, doc_id: Any, doc_type: str = "text") -> Text:
    """Factory function for creating documents"""
    metadata = TextMetaData(text_id=doc_id, text_label=doc_type)
    return Text(content, metadata)

def create_tree_from_data(data: List[Dict]) -> Tree:
    """Factory function for creating trees from structured data"""
    if not data:
        return Tree(None)
    
    root_data = data[0]
    root = Tree(root_data)
    
    for item in data[1:]:
        root.add_child(item)
    
    return root
```

### Validation Helpers

```python
def create_type_validator(expected_type):
    """Create type validation function for cdict"""
    def validate(d, key, value):
        return isinstance(value, expected_type)
    return validate

def create_range_validator(min_val, max_val):
    """Create range validation function for numeric values"""
    def validate(d, key, value):
        return isinstance(value, (int, float)) and min_val <= value <= max_val
    return validate
```

### Serialization Helpers

```python
import json

def serialize_tree(tree: Tree) -> str:
    """Serialize tree to JSON string"""
    def tree_to_dict(node):
        result = {"data": node.data}
        if node.children:
            result["children"] = [tree_to_dict(child) for child in node.children]
        return result
    
    return json.dumps(tree_to_dict(tree))

def serialize_records(store: RecordStore) -> str:
    """Serialize record store to JSON string"""
    return json.dumps(store.records)
```

## Integration Examples

### Cross-Module Usage

```python
from dataknobs_structures import Tree, Text, TextMetaData, RecordStore, cdict

# Store documents in tree structure
doc_tree = Tree("Document Collection")

documents = [
    Text("First document", TextMetaData(1, "article")),
    Text("Second document", TextMetaData(2, "report")),
    Text("Third document", TextMetaData(3, "note"))
]

for doc in documents:
    doc_tree.add_child(doc)

# Store document metadata in RecordStore
metadata_store = RecordStore("doc_metadata.tsv")
for doc in documents:
    metadata_store.add_rec(doc.metadata.data)

# Use conditional dict for validated configuration
def validate_config(d, key, value):
    valid_keys = ["max_docs", "auto_save", "format"]
    return key in valid_keys

config = cdict(validate_config)
config["max_docs"] = 100
config["auto_save"] = True
config["invalid_key"] = "rejected"  # Will be rejected
```

## Version Compatibility

### Version 1.0.0

- All classes and functions documented above
- Stable API - no breaking changes planned
- Compatible with Python 3.8+

### Migration Notes

When upgrading from earlier versions:

- Import paths remain the same
- All existing functionality preserved
- New optional parameters added (backward compatible)

## Testing Utilities

### Test Helpers

```python
def create_test_tree():
    """Create a test tree for unit tests"""
    root = Tree("root")
    child1 = root.add_child("child1")
    child2 = root.add_child("child2")
    child1.add_child("grandchild1")
    return root

def create_test_documents():
    """Create test documents"""
    docs = []
    for i in range(3):
        meta = TextMetaData(i, f"test_doc_{i}")
        doc = Text(f"Content of document {i}", meta)
        docs.append(doc)
    return docs
```

## Performance Notes

### Tree Operations

- Tree traversal: O(n) for n nodes
- Node insertion: O(1) 
- Node search: O(n) worst case
- Path operations: O(depth)

### RecordStore Operations

- Record addition: O(1)
- DataFrame conversion: O(n)
- File I/O: O(n) for n records

### Conditional Dict Operations

- Validation overhead: depends on validation function complexity
- Standard dict operations: same as built-in dict

## See Also

- [Tree Documentation](tree.md)
- [Document Documentation](document.md)
- [RecordStore Documentation](record-store.md)
- [Conditional Dictionary Documentation](conditional-dict.md)
- [Package Overview](index.md)