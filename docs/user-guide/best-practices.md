# Best Practices

Guidelines and recommendations for using Dataknobs in production.

## Installation and Deployment

### Version Pinning

Always pin exact versions in production:

```toml
# pyproject.toml
[tool.uv.dependencies]
dataknobs-structures = "==1.0.0"
dataknobs-utils = "==1.0.0"
dataknobs-xization = "==1.0.0"
```

### Virtual Environments

Always use virtual environments:

```bash
# Using uv
uv venv
source .venv/bin/activate
uv pip install dataknobs-structures

# Using standard venv
python -m venv venv
source venv/bin/activate
pip install dataknobs-structures
```

## Code Organization

### Import Organization

```python
# Good: Group imports by package
from dataknobs_structures import Tree, Text, TextMetaData
from dataknobs_utils import json_utils, file_utils
from dataknobs_xization.normalize import basic_normalization_fn

# Avoid: Scattered imports
from dataknobs_structures import Tree
from dataknobs_utils import json_utils
from dataknobs_structures import Text  # Scattered
```

### Module Structure

```python
# project/
#   ├── core/
#   │   ├── __init__.py
#   │   ├── models.py      # Data models using dataknobs_structures
#   │   └── processors.py  # Processing logic using dataknobs_xization
#   ├── utils/
#   │   ├── __init__.py
#   │   └── helpers.py     # Utilities using dataknobs_utils
#   └── main.py
```

## Error Handling

### Graceful Degradation

```python
import logging

from pyparsing import ParseException

from dataknobs_structures import Tree, build_tree_from_string

logger = logging.getLogger(__name__)

def safe_tree_operation(tree_string):
    try:
        return build_tree_from_string(tree_string)
    except ParseException as e:
        # Unbalanced parentheses raise pyparsing's ParseException, not
        # ValueError -- the parser's exception reaches the caller unchanged.
        logger.warning("Invalid tree format: %s", e)
        return Tree(None)  # Tree requires data; there is no empty Tree()
    except Exception as e:
        logger.error("Unexpected error: %s", e)
        raise  # Re-raise unexpected errors

# Note what does NOT raise: input that does not start with "(" is taken as a
# single node's data, so a malformed string often parses successfully into a
# childless node rather than failing. Check num_children if you expected one.
suspicious = build_tree_from_string("root -> a, b")
print(suspicious.data, suspicious.num_children)   # root -> a, b 0
```

### Input Validation

```python
def process_document(text, metadata=None):
    # Validate inputs
    if not isinstance(text, str):
        raise TypeError(f"Expected str, got {type(text)}")
    
    if not text.strip():
        raise ValueError("Text cannot be empty")
    
    if metadata and not isinstance(metadata, dict):
        raise TypeError(f"Metadata must be dict, got {type(metadata)}")
    
    # Process
    from dataknobs_structures import Text, TextMetaData
    meta = TextMetaData(**metadata) if metadata else TextMetaData()
    return Text(text, meta)
```

## Performance

### Lazy Loading

```python
class DocumentProcessor:
    def __init__(self):
        self._tokenizer = None
        self._normalizer = None
    
    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from dataknobs_xization import masking_tokenizer
            # Tokenizing is per-text: TextFeatures takes the text itself, so
            # what is worth caching is a configured factory, not an object.
            self._tokenizer = lambda text: masking_tokenizer.TextFeatures(
                text, mark_alpha=True, mark_digit=True, emoji_data=None
            ).get_tokens()
        return self._tokenizer
    
    @property
    def normalizer(self):
        if self._normalizer is None:
            from dataknobs_xization.normalize import basic_normalization_fn
            self._normalizer = basic_normalization_fn
        return self._normalizer
```

### Batch Processing

```python
def process_documents_batch(documents, batch_size=100):
    from dataknobs_xization.normalize import basic_normalization_fn
    
    results = []
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        
        # Process batch
        batch_results = [basic_normalization_fn(doc) for doc in batch]
        results.extend(batch_results)
        
        # Optional: yield for streaming
        # yield batch_results
    
    return results
```

## Memory Management

### Resource Cleanup

```python
from contextlib import contextmanager

@contextmanager
def large_tree_processor(tree_data):
    from dataknobs_structures import build_tree_from_string

    # There is no Tree.from_data and no tree.clear(). The parenthesized string
    # form is the bulk constructor the package exports, and a tree is ordinary
    # garbage: dropping the last reference to the root frees it.
    tree = build_tree_from_string(tree_data)
    try:
        yield tree
    finally:
        # Detach the children so the subtrees are collectable even if a caller
        # kept a reference to the root. Iterate a snapshot, not the live list.
        for child in tree.children or ():
            child.prune()

# Usage
with large_tree_processor("(root (a a1 a2) b)") as tree:
    print(tree.as_string())   # (root (a a1 a2) b)
```

### Generator Patterns

```python
def read_large_dataset(filepath):
    from dataknobs_utils import file_utils
    
    with open(filepath, 'r') as f:
        for line in f:
            # Process line by line instead of loading all
            data = json.loads(line)
            yield process_document(data)

# Memory-efficient processing
for processed in read_large_dataset("large_file.jsonl"):
    handle_document(processed)
```

## Testing

### Unit Testing

```python
import pytest
from dataknobs_structures import Tree, build_tree_from_string

class TestTreeOperations:
    def test_tree_creation(self):
        # Tree takes its data positionally; a node with no parent is a root
        root = Tree("root")
        assert root.parent is None
        assert root.data == "root"
        assert root.children is None       # None until it first holds a child

    def test_tree_traversal(self):
        tree = build_tree_from_string("(root a b)")
        nodes = tree.find_nodes(lambda n: True)
        assert len(nodes) == 3
        assert nodes[0].data == "root"

    @pytest.mark.parametrize("input_str,expected_nodes", [
        ("root", 1),
        ("(root a)", 2),
        ("(root (a c) b)", 4),
    ])
    def test_tree_sizes(self, input_str, expected_nodes):
        tree = build_tree_from_string(input_str)
        assert len(tree.find_nodes(lambda n: True)) == expected_nodes
```

### Integration Testing

```python
def test_full_pipeline():
    from dataknobs_structures import Text, TextMetaData
    from dataknobs_xization.normalize import basic_normalization_fn
    from dataknobs_utils import json_utils
    
    # Create document -- text_id is required and positional
    metadata = TextMetaData("test_doc", source="test")
    doc = Text("  TEST Document  ", metadata)

    # Process. The content is doc.text, and basic_normalization_fn lowercases
    # without stripping or collapsing whitespace.
    normalized = basic_normalization_fn(doc.text)

    # Store result. Free-form metadata keys are read through get_value.
    result = {
        "original": doc.text,
        "normalized": normalized,
        "metadata": {"source": doc.metadata.get_value("source")},
    }

    # Verify
    assert json_utils.get_value(result, "normalized") == "  test document  "
    assert json_utils.get_value(result, "metadata.source") == "test"
```

## Logging

### Structured Logging

```python
import logging
import json
from datetime import datetime

class StructuredLogger:
    def __init__(self, name):
        self.logger = logging.getLogger(name)
    
    def log_operation(self, operation, data, level=logging.INFO):
        log_entry = {
            "operation": operation,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        self.logger.log(level, json.dumps(log_entry))

# Usage
logger = StructuredLogger(__name__)
logger.log_operation(
    "tree_processed",
    {"nodes": 100, "depth": 5, "time_ms": 45}
)
```

## Security

### Input Sanitization

```python
from dataknobs_xization.normalize import basic_normalization_fn
import re

def sanitize_input(text):
    # Remove potential injection patterns
    text = re.sub(r'[<>"\']', '', text)
    
    # Normalize
    text = basic_normalization_fn(text)
    
    # Length limit
    max_length = 10000
    if len(text) > max_length:
        text = text[:max_length]
    
    return text
```

### Sensitive Data Handling

```python
import re

from dataknobs_xization import masking_tokenizer

class SecureProcessor:
    # There is no add_pattern hook to register these with; the patterns live
    # here and are applied to the text the tokenizer yields.
    SENSITIVE = (
        re.compile(r"^\d{3}-\d{2}-\d{4}$"),   # SSN
        re.compile(r"^\d{16}$"),               # card number
    )

    def process_sensitive(self, text):
        features = masking_tokenizer.TextFeatures(
            text, mark_alpha=True, mark_digit=True, emoji_data=None
        )
        return " ".join(
            "[REDACTED]"
            if any(p.match(token.token_text) for p in self.SENSITIVE)
            else token.token_text
            for token in features.get_tokens()
        )

# Note: the tokenizer splits on punctuation, so "123-45-6789" arrives as three
# tokens and the SSN pattern above will not match it. Redact over the raw text
# with re.sub when the thing you are hiding spans a delimiter.
```

## Monitoring

### Performance Metrics

```python
import time
from contextlib import contextmanager

@contextmanager
def measure_time(operation_name):
    start = time.time()
    try:
        yield
    finally:
        duration = time.time() - start
        print(f"{operation_name} took {duration:.3f}s")
        
        # Send to monitoring system
        send_metric(f"dataknobs.{operation_name}.duration", duration)

# Usage
with measure_time("tree_processing"):
    process_large_tree(tree)
```

## Migration from Legacy

If migrating from the legacy `dataknobs` package:

```python
# Old code
try:
    from dataknobs.structures import Tree  # Legacy
except ImportError:
    from dataknobs_structures import Tree  # New

# Better: Use only new packages
from dataknobs_structures import Tree
```

## Common Pitfalls

### Avoid These Patterns

```python
from dataknobs_structures import build_tree_from_string

# Removing nodes while walking is safe here, and it is worth knowing why
# rather than copying the usual "collect first" rule by reflex:
#   - `children` answers a snapshot tuple, not the live list, so pruning a
#     child cannot make the iteration skip its sibling;
#   - `find_nodes` returns a materialised list, not a lazy iterator.
# There is no remove_node; a node detaches itself with prune().
tree = build_tree_from_string("(root a remove b remove c)")
for node in tree.find_nodes(lambda n: n.data == "remove"):
    node.prune()
print(tree.as_string())   # (root a b c)

# The rule still applies to anything you iterate lazily or mutate in place,
# so collect first when the sequence is a generator rather than a list.
```

```python
from dataknobs_utils.elasticsearch_utils import SimplifiedElasticsearchIndex

# There is no module-level elasticsearch_utils.search(); searching goes
# through an index object, and the body is an Elasticsearch query dict.
index = SimplifiedElasticsearchIndex("my-index")

# search() answers a ServerResponse, not a dict -- read .result, and check
# .succeeded rather than assuming the call reached the server.
# Bad: Not handling empty results
response = index.search({"query": {"match_all": {}}})
first = response.result["hits"]["hits"][0]  # May fail!

# Good: Check first
response = index.search({"query": {"match_all": {}}})
hits = response.result["hits"]["hits"] if response.succeeded else []
if hits:
    first = hits[0]
else:
    handle_empty_result()
```

## Recommended Tools

- **Linting**: `ruff` for fast Python linting
- **Type Checking**: `mypy` for static type analysis
- **Testing**: `pytest` with coverage reporting
- **Documentation**: `mkdocs` with Material theme
- **Package Management**: `uv` for fast, reliable dependency management

## Support and Resources

- [API Reference](../api/index.md)
- [Examples](../examples/index.md)
- [GitHub Issues](https://github.com/kbs-labs/dataknobs/issues)
- [Contributing Guide](../development/contributing.md)
