# Quick Start Guide

## Prerequisites

- Python 3.10 or higher
- pip or uv package manager

## Installation

### Install all packages

```bash
pip install dataknobs-structures dataknobs-utils dataknobs-xization
```

### Install specific packages

```bash
# Core data structures only
pip install dataknobs-structures

# Utility functions only
pip install dataknobs-utils

# Text processing only
pip install dataknobs-xization
```

## Basic Usage

### Working with Trees

```python
from dataknobs_structures import Tree, build_tree_from_string

# Create a tree from a string
tree_str = "root -> child1, child2\nchild1 -> leaf1"
tree = build_tree_from_string(tree_str)

# Access tree nodes
root = tree.root
children = root.children
```

### Text Processing

```python
from dataknobs_xization import basic_normalization_fn

# Normalize text
text = "  Hello   World!  "
normalized = basic_normalization_fn(text)
print(normalized)  # "hello world!"
```

### Using Utilities

```python
from dataknobs_utils import json_utils

# Work with JSON data
data = {"key": "value", "nested": {"inner": "data"}}
value = json_utils.get_value(data, "nested.inner")
print(value)  # "data"
```

## Next Steps

- Read the [Basic Usage Guide](basic-usage.md) for more detailed examples
- Explore [Advanced Usage](advanced-usage.md) for complex scenarios
- Check out [Best Practices](best-practices.md) for production use
- See [Examples](../examples/index.md) for real-world use cases
