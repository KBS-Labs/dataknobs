# JSON Utils Documentation

The `json_utils` module provides advanced JSON processing capabilities including streaming, path extraction, schema analysis, and data transformation.

## Overview

JSON Utils offers:

- **Path-based Value Extraction**: Extract values using dot notation with array indexing
- **JSON Streaming**: Process large JSON files without loading into memory  
- **Schema Analysis**: Analyze JSON structure and data types
- **Record Processing**: Extract records from nested JSON structures
- **Data Transformation**: Convert between JSON and other formats

## Core Functions

### get_value()

Extract values from JSON objects using path notation.

```python
from dataknobs_utils.json_utils import get_value

data = {
    "users": [
        {"name": "Alice", "age": 30},
        {"name": "Bob", "age": 25}
    ],
    "config": {"debug": True}
}

# Simple path
name = get_value(data, "config.debug")  # True

# Array indexing  
first_user = get_value(data, "users[0].name")  # "Alice"

# Wildcard extraction
all_names = get_value(data, "users[*].name")  # ["Alice", "Bob"]

# First match extraction
any_name = get_value(data, "users[?].name")  # "Alice" (first found)

# Default values
missing = get_value(data, "nonexistent.path", "default")  # "default"
```

### stream_json_data()

Stream JSON data for memory-efficient processing.

```python
from dataknobs_utils.json_utils import stream_json_data

def process_item(item, path):
    """Visitor function called for each JSON value"""
    print(f"Path: {path}, Value: {item}")

# Stream from file
stream_json_data("large_file.json", process_item)

# Stream from URL
stream_json_data("https://api.example.com/data.json", process_item)

# Stream from string
json_string = '{"key": "value", "numbers": [1, 2, 3]}'
stream_json_data(json_string, process_item)
```

### build_jq_path()

Convert stream paths to jq-style path strings.

```python
from dataknobs_utils.json_utils import build_jq_path

# Convert path tuple to jq path
path_tuple = ("users", 0, "profile", "email")
jq_path = build_jq_path(path_tuple)
print(jq_path)  # ".users[0].profile.email"

# Without list indices
jq_path = build_jq_path(path_tuple, keep_list_idxs=False) 
print(jq_path)  # ".users[].profile.email"
```

## JSON Schema Analysis

### JsonSchemaBuilder

Analyze JSON structure and build schemas.

```python
from dataknobs_utils.json_utils import JsonSchemaBuilder

# Analyze JSON file
builder = JsonSchemaBuilder(
    json_data="data.json",
    keep_unique_values=True,     # Track unique values
    invert_uniques=True,         # Track paths to values
    keep_list_idxs=False        # Generalize array indices
)

# Get schema
schema = builder.schema

# Schema as DataFrame
df = schema.df
print(df.head())
# Columns: jq_path, value_type, value_count, unique_count

# Extract unique values for a path
unique_names = schema.get_values(".users[].name")
print(unique_names)  # {"Alice", "Bob", "Charlie"}
```

### JsonSchema Class

Work with generated schemas.

```python
from dataknobs_utils.json_utils import JsonSchema, ValuesIndex

# Create empty schema
schema = JsonSchema()

# Add path information
schema.add_path(".user.name", "str", value="Alice")
schema.add_path(".user.age", "int", value=30)
schema.add_path(".user.age", "int", value=25)  # Another instance

# Get schema DataFrame
df = schema.df
print(df)
# Shows: jq_path, value_type, value_count, unique_count

# Extract values from actual JSON
values = schema.extract_values(".user.name", "users.json")
```

## Record Processing

### stream_record_paths()

Extract records from JSON streams.

```python
from dataknobs_utils.json_utils import stream_record_paths
import io

# Output stream
output = io.StringIO()

# Extract records with custom formatting
def format_record(rec_id, line_num, jq_path, value):
    return f"{rec_id},{line_num},{jq_path},{value}"

stream_record_paths(
    json_data="records.json",
    output_stream=output,
    line_builder_fn=format_record
)

# Get results
output.seek(0)
results = output.read()
print(results)
```

### get_records_df()

Get records as a pandas DataFrame.

```python
from dataknobs_utils.json_utils import get_records_df

# Process JSON to DataFrame
df = get_records_df("data.json")
print(df.columns)  # ["rec_id", "line_num", "jq_path", "item"]

# Analyze record structure
record_counts = df.groupby("rec_id").size()
path_frequency = df["jq_path"].value_counts()
```

## Data Transformation

### Squashing and Exploding

Transform JSON between nested and flat representations.

```python
from dataknobs_utils.json_utils import collect_squashed, explode

# Squash nested JSON to flat key-value pairs
nested_data = {
    "user": {
        "profile": {"name": "Alice", "age": 30},
        "preferences": {"theme": "dark"}
    }
}

# Convert to flat structure
squashed = collect_squashed('{"user": {"name": "Alice", "age": 30}}')
print(squashed)
# {".user.name": "Alice", ".user.age": 30}

# Convert back to nested
exploded = explode(squashed)
print(exploded)
# {"user": {"name": "Alice", "age": 30}}
```

### path_to_dict()

Convert paths and values to nested dictionaries.

```python
from dataknobs_utils.json_utils import path_to_dict

# Build nested dict from path
result = {}
path_to_dict(".users[0].name", "Alice", result)
path_to_dict(".users[0].age", 30, result)
path_to_dict(".users[1].name", "Bob", result)

print(result)
# {"users": [{"name": "Alice", "age": 30}, {"name": "Bob"}]}
```

## Advanced Processing

### squash_data()

Process JSON with custom builder functions.

```python
from dataknobs_utils.json_utils import squash_data

results = []

def collect_strings(jq_path, item):
    """Collect only string values"""
    if isinstance(item, str):
        results.append((jq_path, item))

# Process with filtering
squash_data(
    builder_fn=collect_strings,
    json_data="mixed_data.json",
    prune_at=["metadata"]  # Skip metadata branches
)

print(results)  # List of (path, string_value) tuples
```

### Filtering and Pruning

```python
from dataknobs_utils.json_utils import squash_data

# Prune specific paths during processing
prune_config = [
    "metadata",           # Skip any "metadata" keys
    ("logs", 2),         # Skip "logs" at depth 2
    3                    # Skip anything at depth 3
]

def process_filtered(jq_path, item):
    print(f"Processing: {jq_path} = {item}")

squash_data(
    builder_fn=process_filtered,
    json_data="data.json",
    prune_at=prune_config
)
```

## Streaming Patterns

### Large File Processing

```python
from dataknobs_utils.json_utils import stream_json_data
from collections import defaultdict

# Aggregate data while streaming
stats = defaultdict(int)

def count_types(item, path):
    """Count value types while streaming"""
    item_type = type(item).__name__
    stats[item_type] += 1

# Process large file without loading into memory
stream_json_data("very_large_file.json", count_types)

print(f"Type distribution: {dict(stats)}")
```

### Selective Processing

```python
import json

# Only process specific paths
target_paths = {".users[].email", ".users[].profile.settings"}
collected = []

def selective_visitor(item, path):
    jq_path = build_jq_path(path, keep_list_idxs=False)
    if jq_path in target_paths:
        collected.append((jq_path, item))

stream_json_data("users.json", selective_visitor)
```

## Integration Examples

### With RecordStore

```python
from dataknobs_structures import RecordStore
from dataknobs_utils.json_utils import stream_json_data

# Stream JSON into RecordStore
store = RecordStore("extracted_records.tsv")

def extract_records(item, path):
    """Extract user records from JSON stream"""
    if isinstance(item, dict) and "user_id" in item:
        store.add_rec(item)

stream_json_data("users.json", extract_records)
store.save()
```

### With Pandas

```python
import pandas as pd
from dataknobs_utils.json_utils import JsonSchemaBuilder

# Analyze JSON schema with pandas
builder = JsonSchemaBuilder("data.json", keep_unique_values=True)
schema_df = builder.schema.df

# Analyze value types
type_distribution = schema_df.groupby("value_type")["value_count"].sum()
print(type_distribution)

# Find paths with high cardinality
high_cardinality = schema_df[schema_df["unique_count"] > 100]
print(high_cardinality)
```

## Performance Tips

### Memory Management

```python
# For large files, use streaming instead of loading
# DON'T do this for large files:
# with open("huge_file.json") as f:
#     data = json.load(f)

# DO this instead:
def process_large_file():
    count = 0
    def counter(item, path):
        nonlocal count
        count += 1
    
    stream_json_data("huge_file.json", counter)
    return count
```

### Efficient Path Extraction

```python
# Batch path extractions
paths_to_extract = [".user.name", ".user.email", ".user.age"]
extracted = {}

for path in paths_to_extract:
    extracted[path] = get_value(data, path)

# Better than multiple separate calls for complex data
```

## Error Handling

```python
from dataknobs_utils.json_utils import get_value, stream_json_data

# Handle missing paths gracefully
def safe_extract(data, path):
    try:
        return get_value(data, path)
    except Exception as e:
        print(f"Failed to extract {path}: {e}")
        return None

# Handle streaming errors
def safe_stream_processor(json_file):
    def error_handler(item, path):
        try:
            # Process item
            process_item(item, path)
        except Exception as e:
            print(f"Error processing {path}: {e}")
    
    try:
        stream_json_data(json_file, error_handler, timeout=30)
    except Exception as e:
        print(f"Stream processing failed: {e}")
```

## Configuration

### Timeouts and Limits

```python
from dataknobs_utils.json_utils import JsonSchemaBuilder

# Configure processing limits
builder = JsonSchemaBuilder(
    json_data="data.json",
    keep_unique_values=1000,    # Limit unique value tracking
    timeout=60,                 # 60 second timeout for URLs
    values_limit=500           # Stop after 500 unique values per path
)
```

### URL Processing

```python
# Stream from URLs with custom timeout
def process_api_data():
    def api_processor(item, path):
        if isinstance(item, dict) and "id" in item:
            print(f"Processing record: {item['id']}")
    
    stream_json_data(
        "https://api.example.com/large-dataset.json",
        api_processor,
        timeout=120  # 2 minute timeout
    )
```

## Best Practices

1. **Use Streaming**: For large JSON files, always use streaming APIs
2. **Path Validation**: Validate paths before extraction in production code
3. **Error Handling**: Always handle missing paths and network errors
4. **Memory Monitoring**: Monitor memory usage when processing large datasets
5. **Timeout Configuration**: Set appropriate timeouts for URL-based processing

## See Also

- [File Utils](file-utils.md) - File system utilities
- [Utils Overview](index.md) - Complete utils package documentation
- [Examples](../../examples/json-processing.md) - Detailed usage examples