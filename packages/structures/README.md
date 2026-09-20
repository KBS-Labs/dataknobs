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
from dataknobs_structures import Tree
from dataknobs_structures import Text, TextMetaData

# A node IS the tree: there is no container to create, and the root is simply
# the node with no parent. `add_child` takes the data and returns the new node.
tree = Tree("Root Node")
child = tree.add_child("Child 1")

print(tree.as_string())      # (Root Node Child 1)
print(child.parent.data)     # Root Node
print(tree.num_children)     # 1

# Create a document. text_id is required and comes first; anything else is
# kept as free-form metadata and read back through get_value.
doc = Text(
    "Sample document content",
    TextMetaData("doc_001", author="John Doe", date="2024-01-01"),
)

print(doc.text)                              # Sample document content
print(doc.text_id)                           # doc_001
print(doc.metadata.get_value("author"))      # John Doe
```

## License

Licensed under the [Apache License, Version 2.0](LICENSE); see [NOTICE](NOTICE)
for attribution requirements.

Versions released before this change remain available under the MIT License,
preserved in [LICENSES/MIT-historical.txt](../../LICENSES/MIT-historical.txt).
