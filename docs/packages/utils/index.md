# Dataknobs Utils

The `dataknobs-utils` package provides utility functions and helper classes for common data processing tasks.

## Installation

```bash
pip install dataknobs-utils
```

## Overview

The Utils package includes utilities for:

- **JSON Processing**: Advanced JSON manipulation and streaming
- **File Operations**: File system utilities and helpers
- **Elasticsearch Integration**: Elasticsearch client and query builders
- **LLM Utilities**: Large Language Model prompt management
- **Data Processing**: Pandas, XML, SQL, and other data utilities

## Package Structure

```
dataknobs-utils/
├── src/
│   └── dataknobs_utils/
│       ├── __init__.py
│       ├── elasticsearch_utils.py
│       ├── emoji_utils.py
│       ├── file_utils.py
│       ├── json_extractor.py
│       ├── json_utils.py
│       ├── llm_utils.py
│       ├── pandas_utils.py
│       ├── requests_utils.py
│       ├── resource_utils.py
│       ├── sql_utils.py
│       ├── stats_utils.py
│       ├── subprocess_utils.py
│       ├── sys_utils.py
│       └── xml_utils.py
└── tests/
```

## Quick Start

### JSON Processing

```python
from dataknobs_utils import json_utils

# Stream large JSON files
def process_item(item, path):
    print(f"Path: {path}, Item: {item}")

json_utils.stream_json_data("large_file.json", process_item)

# Extract values with path notation
data = {"users": [{"name": "Alice"}, {"name": "Bob"}]}
names = json_utils.get_value(data, "users[*].name")
print(names)  # ["Alice", "Bob"]
```

### File Operations

```python
from dataknobs_utils import file_utils

# Generate file paths recursively
for filepath in file_utils.filepath_generator("/data", descend=True):
    print(filepath)

# Read lines from files (handles gzip)
for line in file_utils.fileline_generator("data.txt.gz"):
    process_line(line)
```

### Elasticsearch Integration

```python
from dataknobs_utils import elasticsearch_utils

# Build queries
query = elasticsearch_utils.build_field_query_dict(
    fields=["title", "content"],
    text="python programming"
)

# Create index wrapper
index = elasticsearch_utils.ElasticsearchIndex(
    request_helper=None,  # Will use localhost
    table_settings=[]
)

# Search
response = index.search(query)
if response.succeeded:
    hits_df = response.extra.get("hits_df")
```

### LLM Utilities

```python
from dataknobs_utils import llm_utils

# Create prompt message
message = llm_utils.PromptMessage(
    role="user",
    content="Explain Python functions",
    metadata={"model": "gpt-4", "temperature": 0.7}
)

# Build prompt tree
prompt_tree = llm_utils.PromptTree(message=message)
response_node = prompt_tree.add_message(
    role="assistant",
    content="Python functions are reusable blocks of code..."
)

# Get conversation context
messages = prompt_tree.get_messages()
```

## Module Overview

### Core Modules

| Module | Purpose | Key Features |
|--------|---------|--------------|
| [`json_utils`](json-utils.md) | JSON processing | Streaming, path extraction, schema analysis |
| [`file_utils`](file-utils.md) | File operations | Path generation, line reading, compression |
| [`elasticsearch_utils`](elasticsearch.md) | Elasticsearch | Query building, indexing, search |
| [`llm_utils`](llm-utils.md) | LLM integration | Prompt trees, message management |

### Supporting Modules

| Module | Purpose |
|--------|---------|
| `emoji_utils` | Emoji detection and processing |
| `pandas_utils` | Pandas DataFrame utilities |
| `requests_utils` | HTTP request helpers |
| `resource_utils` | Resource loading utilities |
| `sql_utils` | SQL query helpers |
| `stats_utils` | Statistical utilities |
| `subprocess_utils` | Process execution |
| `sys_utils` | System utilities |
| `xml_utils` | XML processing |

## Advanced Usage

### JSON Schema Analysis

```python
from dataknobs_utils.json_utils import JsonSchemaBuilder

# Analyze JSON structure
builder = JsonSchemaBuilder(
    json_data="data.json",
    keep_unique_values=True,
    invert_uniques=True
)

schema = builder.schema
df = schema.df  # Schema as DataFrame

# Extract specific values
values = schema.extract_values(".users[].name", "data.json")
```

### Streaming Data Processing

```python
from dataknobs_utils.json_utils import stream_record_paths
import io

# Process JSON records
output = io.StringIO()
stream_record_paths(
    json_data="records.json",
    output_stream=output,
    line_builder_fn=lambda rid, lid, path, val: f"{rid},{path},{val}"
)
```

### Elasticsearch Batch Operations

```python
from dataknobs_utils.elasticsearch_utils import add_batch_data
import io

# Prepare batch data for indexing
records = [
    {"id": 1, "title": "Document 1"},
    {"id": 2, "title": "Document 2"}
]

batch_file = io.StringIO()
next_id = add_batch_data(
    batchfile=batch_file,
    record_generator=iter(records),
    idx_name="documents"
)
```

### Prompt Tree Management

```python
from dataknobs_utils.llm_utils import PromptTree

# Create conversation tree
root = PromptTree(role="system", content="You are a helpful assistant")
user_msg = root.add_message(role="user", content="What is Python?")
assistant_msg = user_msg.add_message(
    role="assistant", 
    content="Python is a programming language..."
)

# Branch conversation
followup = user_msg.add_message(role="user", content="Show me an example")

# Get conversation paths
messages = assistant_msg.get_messages()  # System + User + Assistant
followup_messages = followup.get_messages()  # System + User (different path)
```

## Integration Examples

### With Structures Package

```python
from dataknobs_structures import RecordStore, Tree
from dataknobs_utils import json_utils

# Load JSON data into RecordStore
def load_json_to_store(json_file, store):
    def visitor(item, path):
        if isinstance(item, dict):
            store.add_rec(item)
    
    json_utils.stream_json_data(json_file, visitor)

store = RecordStore("processed_data.tsv")
load_json_to_store("input.json", store)

# Build tree from JSON structure
json_data = {"name": "root", "children": [{"name": "child1"}]}
tree = Tree(json_data["name"])
for child_data in json_data.get("children", []):
    tree.add_child(child_data["name"])
```

### With Xization Package

```python
from dataknobs_utils import llm_utils
from dataknobs_xization import normalize

# Normalize prompt content
message = llm_utils.PromptMessage(
    role="user",
    content="HELLO WORLD! This is a TEST."
)

# Apply normalization
normalized_content = normalize.basic_normalization_fn(
    message.content,
    lowercase=True,
    squash_whitespace=True
)

# Update message
message.content = normalized_content
```

## Error Handling

### JSON Processing Errors

```python
from dataknobs_utils import json_utils

try:
    result = json_utils.get_value(data, "invalid.path")
except Exception as e:
    print(f"Path extraction failed: {e}")
    result = None
```

### File Operation Errors

```python
from dataknobs_utils import file_utils

try:
    for filepath in file_utils.filepath_generator("/nonexistent"):
        print(filepath)
except (FileNotFoundError, PermissionError) as e:
    print(f"File access error: {e}")
```

### Elasticsearch Errors

```python
from dataknobs_utils import elasticsearch_utils

# Check if Elasticsearch is available
index = elasticsearch_utils.ElasticsearchIndex(None, [])
if index.is_up():
    print("Elasticsearch is running")
else:
    print("Elasticsearch unavailable")
```

## Performance Considerations

### JSON Streaming

- Use streaming for large JSON files to minimize memory usage
- Implement selective processing with visitor patterns
- Consider parallel processing for independent data streams

### File Operations

- Use generators for large directory traversals
- Handle compressed files efficiently with built-in support
- Batch file operations when possible

### Elasticsearch

- Use bulk operations for inserting multiple documents
- Optimize query structure for performance
- Consider connection pooling for high-throughput scenarios

## Best Practices

1. **Use Streaming**: Process large files with streaming APIs
2. **Error Handling**: Always handle file and network errors
3. **Resource Management**: Close files and connections properly
4. **Batch Operations**: Group operations for better performance
5. **Logging**: Use appropriate logging for debugging

## Configuration

### JSON Processing

```python
# Configure timeouts and limits
from dataknobs_utils.json_utils import JsonSchemaBuilder

builder = JsonSchemaBuilder(
    json_data="large_file.json",
    keep_unique_values=1000,  # Limit unique values
    timeout=30  # 30 second timeout
)
```

### Elasticsearch

```python
# Configure connection
from dataknobs_utils.elasticsearch_utils import ElasticsearchIndex

index = ElasticsearchIndex(
    request_helper=None,
    table_settings=[],
    elasticsearch_ip="localhost",
    elasticsearch_port=9200
)
```

## Testing

The package includes comprehensive tests:

```bash
# Run all tests
python -m pytest tests/

# Run specific module tests
python -m pytest tests/test_json_utils.py
python -m pytest tests/test_file_utils.py
```

## API Reference

For complete API documentation, see the [Utils API Reference](api.md).

## See Also

- [JSON Utilities](json-utils.md) - Comprehensive JSON processing
- [File Utilities](file-utils.md) - File system operations
- [Elasticsearch Integration](elasticsearch.md) - Search and indexing
- [LLM Utilities](llm-utils.md) - Language model support
- [Integration Examples](../../examples/index.md)