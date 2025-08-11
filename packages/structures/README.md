# dataknobs-structures

Data structures for AI knowledge bases.

## Installation

```bash
pip install dataknobs-structures
```

## Features

- **ConditionalDict**: Dictionary with conditional value retrieval
- **Document**: Document representation with metadata
- **RecordStore**: Efficient record storage and retrieval
- **Tree**: Tree data structure with various traversal methods

## Usage

```python
from dataknobs_structures import Tree, Document

# Create a tree structure
tree = Tree()
tree.add_node("root", "Root Node")
tree.add_node("child1", "Child 1", parent="root")

# Create a document
doc = Document(
    content="Sample document content",
    metadata={"author": "John Doe", "date": "2024-01-01"}
)
```

## License

See LICENSE file in the root repository.