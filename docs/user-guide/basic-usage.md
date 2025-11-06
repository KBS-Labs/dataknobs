# Basic Usage

This guide covers the fundamental features of Dataknobs packages.

## Data Structures

### Trees

Trees are hierarchical data structures used for representing relationships.

```python
from dataknobs_structures import Tree

# Create a tree manually
tree = Tree()
root = tree.add_node("root")
child1 = tree.add_child(root, "child1")
child2 = tree.add_child(root, "child2")
leaf = tree.add_child(child1, "leaf")

# Traverse the tree
for node in tree.traverse():
    print(f"Node: {node.value}, Level: {node.level}")
```

### Documents

Documents represent text with metadata and structure.

```python
from dataknobs_structures import Text, TextMetaData

# Create a document with metadata
metadata = TextMetaData(
    source="example.txt",
    created_at="2024-01-01",
    author="John Doe"
)

text = Text("This is the document content.", metadata)
print(f"Content: {text.content}")
print(f"Source: {text.metadata.source}")
```

### Conditional Dictionaries

Conditional dictionaries allow filtering of key-value pairs.

```python
from dataknobs_structures import cdict

# Create a conditional dict that only accepts string values
def accept_strings(d, k, v):
    return isinstance(v, str)

cd = cdict(accept_strings, {"name": "Alice", "age": 30})
# Only "name" will be stored
```

## Utilities

### JSON Utilities

```python
from dataknobs_utils import json_utils

data = {
    "users": [
        {"name": "Alice", "age": 30},
        {"name": "Bob", "age": 25}
    ]
}

# Get nested values
first_user = json_utils.get_value(data, "users.0")
print(first_user)  # {"name": "Alice", "age": 30}

# Set nested values
json_utils.set_value(data, "users.0.age", 31)
```

### File Utilities

```python
from dataknobs_utils import file_utils

# Read and write files
content = file_utils.read_file("input.txt")
processed = content.upper()
file_utils.write_file("output.txt", processed)

# Work with JSON files
data = file_utils.read_json("config.json")
data["updated"] = True
file_utils.write_json("config.json", data)
```

## Text Processing

### Normalization

```python
from dataknobs_xization import basic_normalization_fn

# Basic text normalization
text = "  HELLO   World!!!  "
normalized = basic_normalization_fn(text)
print(normalized)  # "hello world!"

# Custom normalization
def custom_normalize(text):
    return text.lower().replace("!", "").strip()

result = custom_normalize("Hello World!")
print(result)  # "hello world"
```

### Tokenization

```python
from dataknobs_xization import masking_tokenizer

# Tokenize text with masking
text = "John Doe lives at 123 Main St"
tokenizer = masking_tokenizer.MaskingTokenizer()
tokens = tokenizer.tokenize(text)

# Tokens will include masked versions for sensitive data
for token in tokens:
    print(f"Token: {token.value}, Type: {token.type}")
```

## Working with RecordStore

```python
from dataknobs_structures import RecordStore

# Create a record store
store = RecordStore()

# Add records
store.add_record("user:1", {"name": "Alice", "age": 30})
store.add_record("user:2", {"name": "Bob", "age": 25})

# Retrieve records
user1 = store.get_record("user:1")
print(user1)  # {"name": "Alice", "age": 30}

# Query records
young_users = store.query(lambda r: r.get("age", 0) < 30)
for user in young_users:
    print(user)
```

## Error Handling

All packages include proper error handling:

```python
from dataknobs_structures import Tree

try:
    tree = Tree()
    # Attempt to access non-existent node
    node = tree.get_node("nonexistent")
except KeyError as e:
    print(f"Node not found: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Beyond the Basics

Dataknobs includes powerful packages for more advanced use cases:

**For AI Applications:**
- [Bots Package](../packages/bots/index.md) - Build intelligent chatbots with memory and RAG
- [LLM Package](../packages/llm/index.md) - Integrate language models with prompt management

**For Data Engineering:**
- [FSM Package](../packages/fsm/index.md) - Orchestrate complex workflows with finite state machines
- [Data Package](../packages/data/index.md) - Unified interface across PostgreSQL, Elasticsearch, S3, and more
- [Config Package](../packages/config/index.md) - Environment-aware configuration management

## Next Steps

- Explore [Advanced Usage](advanced-usage.md) for complex scenarios and heavier packages
- Read [Best Practices](best-practices.md) for production deployments
- Check the [API Reference](../api/index.md) for detailed documentation
- Browse [Examples](../examples/index.md) for real-world use cases
