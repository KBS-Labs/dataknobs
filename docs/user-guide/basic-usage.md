# Basic Usage

This guide covers the fundamental features of Dataknobs packages.

## Data Structures

### Trees

Trees are hierarchical data structures used for representing relationships.

```python
from dataknobs_structures import Tree

# A Tree node is itself the tree -- there is no separate container, and the
# root is simply the node whose parent is None.
root = Tree("root")
child1 = root.add_child("child1")
child2 = root.add_child("child2")
leaf = child1.add_child("leaf")

# Walk it: find_nodes takes a predicate, so `lambda n: True` visits everything
for node in root.find_nodes(lambda n: True):
    print(f"Node: {node.data}, Depth: {node.depth}")
# Node: root, Depth: 0
# Node: child1, Depth: 1
# Node: leaf, Depth: 2
# Node: child2, Depth: 1
```

### Documents

Documents represent text with metadata and structure.

```python
from dataknobs_structures import Text, TextMetaData

# text_id is required and positional; text_label defaults to "text".
# Anything else you pass is kept as free-form metadata.
metadata = TextMetaData(
    "doc_001",
    text_label="article",
    source="example.txt",
    created_at="2024-01-01",
    author="John Doe",
)

text = Text("This is the document content.", metadata)
print(f"Content: {text.text}")     # Content: This is the document content.
print(f"Id:      {text.text_id}")  # Id:      doc_001
print(f"Label:   {text.text_label}")  # Label:   article
# Free-form keys are read through get_value, not as attributes
print(f"Source:  {text.metadata.get_value('source')}")  # Source:  example.txt
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

# Indexed dot notation: list positions are [n], not .n
first_user = json_utils.get_value(data, "users[0]")
print(first_user)  # {'name': 'Alice', 'age': 30}

print(json_utils.get_value(data, "users[0].name"))  # Alice

# A path that matches nothing answers the default rather than raising
print(json_utils.get_value(data, "users[9].name", "unknown"))  # unknown

# There is no set_value -- this module reads and indexes JSON; assign
# through the object itself when you need to change it.
data["users"][0]["age"] = 31
```

### File Utilities

```python
from dataknobs_utils import file_utils

# This module streams lines rather than reading whole files: fileline_generator
# yields one line at a time (and transparently handles gzip), write_lines
# writes a list back out.
lines = [line.rstrip("\n") for line in file_utils.fileline_generator("input.txt")]
file_utils.write_lines("output.txt", [line.upper() for line in lines])

# Walk a directory tree
for path in file_utils.filepath_generator("data/", descend=True):
    print(path)

# There is no read_json/write_json helper here -- use the standard library,
# or dataknobs_utils.json_utils for path-based access into a loaded object.
import json

with open("config.json") as handle:
    data = json.load(handle)
data["updated"] = True
with open("config.json", "w") as handle:
    json.dump(data, handle)
```

## Text Processing

### Normalization

```python
from dataknobs_xization.normalize import basic_normalization_fn

# Only three transforms are on by default: lowercasing, camelCase expansion and
# smart-quote simplification. Whitespace and symbols are left exactly as found.
text = "  HELLO   World!!!  "
print(repr(basic_normalization_fn(text)))
# '  hello   world!!!  '

print(basic_normalization_fn("parseHTTPResponse"))
# parse http response

# The rest are opt-in, and `do_all` turns on every one of them.
print(repr(basic_normalization_fn(text, squash_whitespace=True)))
# 'hello world!!!'

print(repr(basic_normalization_fn(text, do_all=True)))
# 'hello world'
```

### Tokenization

```python
from dataknobs_xization import masking_tokenizer

# Tokenizing goes through TextFeatures, which builds per-character feature
# masks and derives tokens from them. The mark_* flags choose which character
# classes are marked; emoji_data=None skips loading the emoji tables.
text = "John Doe lives at 123 Main St"
features = masking_tokenizer.TextFeatures(
    text, mark_alpha=True, mark_digit=True, emoji_data=None
)

for token in features.get_tokens():
    print(f"Token: {token.token_text}, Position: {token.token_pos}")
# Token: John, Position: (0, 4)
# Token: Doe, Position: (5, 8)
# Token: lives, Position: (9, 14)
# Token: at, Position: (15, 17)
# Token: 123, Position: (18, 21)
# Token: Main, Position: (22, 26)
# Token: St, Position: (27, 29)

# Tokens are doubly linked and carry a normalized form, so a pass can walk
# forward from the first one without re-tokenizing.
first = features.build_first_token(normalize_fn=str.lower)
print(first.token_text, "->", first.norm_text)   # John -> john
print(first.next_token.token_text)               # Doe
```

## Working with RecordStore

```python
from dataknobs_structures import RecordStore

# A RecordStore is backed by a TSV path, which is required -- pass None for an
# in-memory store with no file behind it. Records are rows, not keyed entries.
store = RecordStore(None)

store.add_rec({"id": "user:1", "name": "Alice", "age": 30})
store.add_rec({"id": "user:2", "name": "Bob", "age": 25})

# Read them back as a list of dicts, or as a pandas DataFrame
print(store.records[0])            # {'id': 'user:1', 'name': 'Alice', 'age': 30}
print(len(store.records))          # 2

# There is no query() -- filter the DataFrame, or the list
young = [r for r in store.records if r["age"] < 30]
print([r["name"] for r in young])  # ['Bob']

# With a path, save() writes the TSV and restore() reads it back
# store = RecordStore("/data/users.tsv")
# store.save()
```

## Error Handling

All packages include proper error handling:

```python
from dataknobs_common.exceptions import ValidationError
from dataknobs_structures import Tree

root = Tree("root")
child = root.add_child("child")

# A search that matches nothing returns an empty list rather than raising
found = root.find_nodes(lambda n: n.data == "nonexistent")
if not found:
    print("No matching node")

# Writes that would make a node its own ancestor are refused, and the tree
# is left exactly as it was
try:
    child.add_child(root)
except ValidationError as e:
    print(f"Refused: {e}")
    print(e.context)  # {'child': 'root', 'parent': 'child'}
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
