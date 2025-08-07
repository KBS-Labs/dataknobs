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
from dataknobs_xization import basic_normalization_fn

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
from dataknobs_structures import Tree
import logging

logger = logging.getLogger(__name__)

def safe_tree_operation(tree_string):
    try:
        tree = build_tree_from_string(tree_string)
        return tree
    except ValueError as e:
        logger.warning(f"Invalid tree format: {e}")
        return Tree()  # Return empty tree
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise  # Re-raise unexpected errors
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
            self._tokenizer = masking_tokenizer.MaskingTokenizer()
        return self._tokenizer
    
    @property
    def normalizer(self):
        if self._normalizer is None:
            from dataknobs_xization import basic_normalization_fn
            self._normalizer = basic_normalization_fn
        return self._normalizer
```

### Batch Processing

```python
def process_documents_batch(documents, batch_size=100):
    from dataknobs_xization import basic_normalization_fn
    
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
    from dataknobs_structures import Tree
    
    tree = None
    try:
        tree = Tree.from_data(tree_data)
        yield tree
    finally:
        # Cleanup
        if tree:
            tree.clear()
        del tree

# Usage
with large_tree_processor(data) as tree:
    process_tree(tree)
# Tree is automatically cleaned up
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
from dataknobs_structures import Tree

class TestTreeOperations:
    def test_tree_creation(self):
        tree = Tree()
        assert tree.root is None
        
        tree.add_root("root")
        assert tree.root.value == "root"
    
    def test_tree_traversal(self):
        tree = build_tree_from_string("root -> a, b")
        nodes = list(tree.traverse())
        assert len(nodes) == 3
        assert nodes[0].value == "root"
    
    @pytest.mark.parametrize("input_str,expected_nodes", [
        ("root", 1),
        ("root -> a", 2),
        ("root -> a, b\na -> c", 4),
    ])
    def test_tree_sizes(self, input_str, expected_nodes):
        tree = build_tree_from_string(input_str)
        assert len(list(tree.traverse())) == expected_nodes
```

### Integration Testing

```python
def test_full_pipeline():
    from dataknobs_structures import Text, TextMetaData
    from dataknobs_xization import basic_normalization_fn
    from dataknobs_utils import json_utils
    
    # Create document
    metadata = TextMetaData(source="test")
    doc = Text("  TEST Document  ", metadata)
    
    # Process
    normalized = basic_normalization_fn(doc.content)
    
    # Store result
    result = {
        "original": doc.content,
        "normalized": normalized,
        "metadata": {"source": doc.metadata.source}
    }
    
    # Verify
    assert json_utils.get_value(result, "normalized") == "test document"
    assert json_utils.get_value(result, "metadata.source") == "test"
```

## Logging

### Structured Logging

```python
import logging
import json

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
from dataknobs_xization import basic_normalization_fn
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
from dataknobs_xization import masking_tokenizer

class SecureProcessor:
    def __init__(self):
        self.tokenizer = masking_tokenizer.MaskingTokenizer()
        self.tokenizer.add_pattern(r'\b\d{3}-\d{2}-\d{4}\b', 'SSN')
        self.tokenizer.add_pattern(r'\b\d{16}\b', 'CREDIT_CARD')
    
    def process_sensitive(self, text):
        # Mask sensitive data
        tokens = self.tokenizer.tokenize(text)
        masked = ' '.join([
            '[REDACTED]' if t.type in ['SSN', 'CREDIT_CARD'] else t.value
            for t in tokens
        ])
        return masked
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
# Bad: Modifying trees during traversal
for node in tree.traverse():
    if node.value == "remove":
        tree.remove_node(node)  # Don't do this!

# Good: Collect then modify
nodes_to_remove = [n for n in tree.traverse() if n.value == "remove"]
for node in nodes_to_remove:
    tree.remove_node(node)

# Bad: Not handling empty results
result = elasticsearch_utils.search(query)
first = result[0]  # May fail!

# Good: Check first
result = elasticsearch_utils.search(query)
if result:
    first = result[0]
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
