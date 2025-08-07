# Getting Started with Dataknobs

This guide will help you get up and running with Dataknobs in just a few minutes.

## Prerequisites

- Python 3.10 or higher
- pip or uv package manager

## Installation

### Installing Individual Packages (Recommended)

Install only the packages you need:

```bash
# For data structures
pip install dataknobs-structures

# For utilities
pip install dataknobs-utils

# For text processing
pip install dataknobs-xization

# Or install all main packages
pip install dataknobs-structures dataknobs-utils dataknobs-xization
```

### Installing with uv

If you're using the `uv` package manager:

```bash
uv pip install dataknobs-structures dataknobs-utils dataknobs-xization
```

### Installing from Source

Clone the repository and install in development mode:

```bash
git clone https://github.com/yourusername/dataknobs.git
cd dataknobs
uv sync  # or pip install -e packages/structures
```

## Basic Usage

### Working with Trees

```python
from dataknobs_structures import Tree

# Create a tree
root = Tree("root")
child1 = root.add_child("child1")
child2 = root.add_child("child2")
grandchild = child1.add_child("grandchild")

# Navigate the tree
print(root.children)  # Access children
print(child1.parent)  # Access parent
print(root.get_all_descendants())  # Get all descendants
```

### Document Processing

```python
from dataknobs_structures import Text, TextMetaData

# Create a document with metadata
metadata = TextMetaData(text_id=1, source="example.txt")
doc = Text("This is my document content", metadata)

print(doc.text)  # Access text
print(doc.text_id)  # Access metadata
```

### JSON Utilities

```python
from dataknobs_utils import json_utils

# Navigate nested JSON
data = {
    "users": {
        "john": {
            "age": 30,
            "email": "john@example.com"
        }
    }
}

# Get nested values easily
age = json_utils.get_value(data, "users.john.age")
print(age)  # Output: 30

# Build tree from JSON string
json_str = '{"a": {"b": {"c": "value"}}}'
tree = json_utils.build_tree_from_string(json_str)
```

### Text Normalization

```python
from dataknobs_xization import normalize

# Basic normalization
text = "Hello WORLD! How are YOU?"
normalized = normalize.basic_normalization_fn(text)
print(normalized)  # Output: "hello world! how are you?"

# Advanced normalization options
text = "CamelCaseExample"
expanded = normalize.expand_camelcase_fn(text)
print(expanded)  # Output: "Camel Case Example"
```

### Text Tokenization

```python
from dataknobs_xization.masking_tokenizer import TextFeatures
from dataknobs_structures import Text, TextMetaData

# Create a document
metadata = TextMetaData(text_id=1)
doc = Text("Hello World 123!", metadata)

# Extract text features
features = TextFeatures(doc)
print(features.text)  # Original text
print(features.cdf)  # Character dataframe with features
```

## Next Steps

Now that you have Dataknobs installed and understand the basics:

1. **Explore the Packages**: Learn about each package's capabilities in the [Package Documentation](packages/index.md)
2. **Read the User Guide**: Dive deeper into [advanced usage patterns](user-guide/advanced-usage.md)
3. **Check out Examples**: See real-world [usage examples](examples/index.md)
4. **API Reference**: Explore the complete [API documentation](api/index.md)

## Getting Help

- **GitHub Issues**: Report bugs or request features on [GitHub](https://github.com/yourusername/dataknobs/issues)
- **Documentation**: This documentation site has comprehensive guides and API references
- **Examples**: Check the [examples directory](https://github.com/yourusername/dataknobs/tree/main/examples) in the repository

## Common Issues

### Import Errors

If you encounter import errors, make sure you're using the new package names:

```python
# ❌ Old way (deprecated)
from dataknobs.structures import Tree

# ✅ New way
from dataknobs_structures import Tree
```

### Missing Dependencies

Some functionality requires additional packages:

```bash
# For pandas functionality
pip install pandas

# For elasticsearch integration  
pip install elasticsearch

# For LLM utilities
pip install openai
```

See [Installation Guide](installation.md) for complete dependency information.