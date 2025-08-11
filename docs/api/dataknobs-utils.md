# dataknobs-utils API Reference

Complete API documentation for the `dataknobs_utils` package.

## Package Information

- **Package Name**: `dataknobs_utils`
- **Version**: 1.0.0
- **Description**: Utility functions for dataknobs packages
- **Python Requirements**: >=3.8

## Installation

```bash
pip install dataknobs-utils
```

## Import Statement

```python
from dataknobs_utils import (
    elasticsearch_utils,
    emoji_utils,
    file_utils,
    json_extractor,
    json_utils,
    llm_utils,
    pandas_utils,
    requests_utils,
    resource_utils,
    sql_utils,
    stats_utils,
    subprocess_utils,
    sys_utils,
    xml_utils,
)
```

## Module Documentation

### elasticsearch_utils

#### Classes

##### TableSettings
::: dataknobs_utils.elasticsearch_utils.TableSettings
    options:
      show_source: true
      show_root_heading: true
      show_root_toc_entry: false

##### ElasticsearchIndex
::: dataknobs_utils.elasticsearch_utils.ElasticsearchIndex
    options:
      show_source: true
      show_root_heading: true
      show_root_toc_entry: false

#### Functions

##### build_field_query_dict
::: dataknobs_utils.elasticsearch_utils.build_field_query_dict
    options:
      show_source: true

##### build_phrase_query_dict
::: dataknobs_utils.elasticsearch_utils.build_phrase_query_dict
    options:
      show_source: true

##### build_hits_dataframe
::: dataknobs_utils.elasticsearch_utils.build_hits_dataframe
    options:
      show_source: true

##### build_aggs_dataframe
::: dataknobs_utils.elasticsearch_utils.build_aggs_dataframe
    options:
      show_source: true

##### decode_results
::: dataknobs_utils.elasticsearch_utils.decode_results
    options:
      show_source: true

##### add_batch_data
::: dataknobs_utils.elasticsearch_utils.add_batch_data
    options:
      show_source: true

##### batchfile_record_generator
::: dataknobs_utils.elasticsearch_utils.batchfile_record_generator
    options:
      show_source: true

##### collect_batchfile_values
::: dataknobs_utils.elasticsearch_utils.collect_batchfile_values
    options:
      show_source: true

##### collect_batchfile_records
::: dataknobs_utils.elasticsearch_utils.collect_batchfile_records
    options:
      show_source: true

### file_utils

#### Functions

##### filepath_generator
::: dataknobs_utils.file_utils.filepath_generator
    options:
      show_source: true

##### fileline_generator
::: dataknobs_utils.file_utils.fileline_generator
    options:
      show_source: true

##### write_lines
::: dataknobs_utils.file_utils.write_lines
    options:
      show_source: true

##### is_gzip_file
::: dataknobs_utils.file_utils.is_gzip_file
    options:
      show_source: true

### json_utils

#### Functions

::: dataknobs_utils.json_utils
    options:
      show_source: true
      show_root_heading: true
      show_root_toc_entry: false

### llm_utils

#### Classes

##### PromptMessage
::: dataknobs_utils.llm_utils.PromptMessage
    options:
      show_source: true
      show_root_heading: true
      show_root_toc_entry: false

#### Functions

##### get_value_by_key
::: dataknobs_utils.llm_utils.get_value_by_key
    options:
      show_source: true

### pandas_utils

#### Functions

::: dataknobs_utils.pandas_utils
    options:
      show_source: true
      show_root_heading: true
      show_root_toc_entry: false

### requests_utils

#### Classes and Functions

::: dataknobs_utils.requests_utils
    options:
      show_source: true
      show_root_heading: true
      show_root_toc_entry: false

### resource_utils

#### Functions

::: dataknobs_utils.resource_utils
    options:
      show_source: true
      show_root_heading: true
      show_root_toc_entry: false

### sql_utils

#### Functions

::: dataknobs_utils.sql_utils
    options:
      show_source: true
      show_root_heading: true
      show_root_toc_entry: false

### stats_utils

#### Functions

::: dataknobs_utils.stats_utils
    options:
      show_source: true
      show_root_heading: true
      show_root_toc_entry: false

### subprocess_utils

#### Functions

::: dataknobs_utils.subprocess_utils
    options:
      show_source: true
      show_root_heading: true
      show_root_toc_entry: false

### sys_utils

#### Functions

::: dataknobs_utils.sys_utils
    options:
      show_source: true
      show_root_heading: true
      show_root_toc_entry: false

### xml_utils

#### Functions

::: dataknobs_utils.xml_utils
    options:
      show_source: true
      show_root_heading: true
      show_root_toc_entry: false

### emoji_utils

#### Functions

::: dataknobs_utils.emoji_utils
    options:
      show_source: true
      show_root_heading: true
      show_root_toc_entry: false

### json_extractor

#### Classes and Functions

::: dataknobs_utils.json_extractor
    options:
      show_source: true
      show_root_heading: true
      show_root_toc_entry: false

## Usage Examples

### File Processing Example

```python
from dataknobs_utils import file_utils

# Generate all Python files in a directory
for filepath in file_utils.filepath_generator("/path/to/project"):
    if filepath.endswith(".py"):
        # Process each line
        for line in file_utils.fileline_generator(filepath):
            print(f"{filepath}: {line}")

# Write processed results
processed_lines = ["result 1", "result 2", "result 3"]
file_utils.write_lines("output.txt", processed_lines)
```

### Elasticsearch Example

```python
from dataknobs_utils import elasticsearch_utils

# Build a search query
query = elasticsearch_utils.build_field_query_dict(
    ["title", "content"], 
    "machine learning"
)

# Create index configuration
table_settings = elasticsearch_utils.TableSettings(
    "documents",
    {"number_of_shards": 1},
    {
        "properties": {
            "title": {"type": "text"},
            "content": {"type": "text"}
        }
    }
)

# Create and use index
index = elasticsearch_utils.ElasticsearchIndex(None, [table_settings])
results = index.search(query)
```

### LLM Utils Example

```python
from dataknobs_utils import llm_utils

# Create prompt message
message = llm_utils.PromptMessage(
    "user",
    "Analyze this data and provide insights",
    metadata={"priority": "high", "model": "gpt-4"}
)

# Access nested configuration
config = {
    "models": {
        "gpt4": {
            "temperature": 0.7,
            "max_tokens": 1000
        }
    }
}

temperature = llm_utils.get_value_by_key(
    config, "models.gpt4.temperature", 0.5
)
print(f"Temperature: {temperature}")  # 0.7
```

### Integration Example

```python
from dataknobs_utils import (
    file_utils, json_utils, elasticsearch_utils, llm_utils
)
import json

# Complete data processing pipeline
def process_documents(input_dir: str, es_index: str):
    """Process JSON documents and index in Elasticsearch."""
    
    # Step 1: Collect documents
    documents = []
    for filepath in file_utils.filepath_generator(input_dir):
        if filepath.endswith('.json'):
            for line in file_utils.fileline_generator(filepath):
                try:
                    doc = json.loads(line)
                    documents.append(doc)
                except json.JSONDecodeError:
                    continue
    
    # Step 2: Process with LLM utils for configuration
    config = json_utils.load_json_file("config.json")
    es_config = llm_utils.get_value_by_key(
        config, "elasticsearch.settings", {}
    )
    
    # Step 3: Index in Elasticsearch
    table_settings = elasticsearch_utils.TableSettings(
        es_index,
        es_config,
        {
            "properties": {
                "content": {"type": "text"},
                "timestamp": {"type": "date"}
            }
        }
    )
    
    index = elasticsearch_utils.ElasticsearchIndex(None, [table_settings])
    
    # Create batch file
    with open("batch_data.jsonl", "w") as f:
        elasticsearch_utils.add_batch_data(
            f, iter(documents), es_index
        )
    
    return len(documents)

# Usage
processed_count = process_documents("/data/input", "processed_docs")
print(f"Processed {processed_count} documents")
```

## Error Handling

All functions include appropriate error handling. Here are common patterns:

```python
from dataknobs_utils import file_utils, elasticsearch_utils

try:
    # File operations
    lines = list(file_utils.fileline_generator("data.txt"))
except FileNotFoundError:
    print("File not found")
except IOError as e:
    print(f"IO error: {e}")

try:
    # Elasticsearch operations
    index = elasticsearch_utils.ElasticsearchIndex(None, [])
    if not index.is_up():
        raise ConnectionError("Elasticsearch not available")
except ConnectionError:
    print("Cannot connect to Elasticsearch")
```

## Testing

Example test patterns for dataknobs_utils:

```python
import pytest
import tempfile
import os
from dataknobs_utils import file_utils, llm_utils

def test_file_operations():
    """Test file utility functions."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test file writing and reading
        test_file = os.path.join(temp_dir, "test.txt")
        test_lines = ["line1", "line2", "line3"]
        
        file_utils.write_lines(test_file, test_lines)
        read_lines = list(file_utils.fileline_generator(test_file))
        
        assert read_lines == sorted(test_lines)  # write_lines sorts

def test_llm_utils():
    """Test LLM utility functions."""
    # Test nested dictionary access
    data = {"a": {"b": {"c": "value"}}}
    
    result = llm_utils.get_value_by_key(data, "a.b.c")
    assert result == "value"
    
    result = llm_utils.get_value_by_key(data, "x.y.z", "default")
    assert result == "default"
    
    # Test PromptMessage
    msg = llm_utils.PromptMessage("user", "test", {"key": "value"})
    assert msg.role == "user"
    assert msg.content == "test"
    assert msg.metadata["key"] == "value"
```

## Performance Notes

- **File Utils**: Uses generators for memory-efficient processing of large files
- **Elasticsearch Utils**: Supports batch operations for better performance
- **JSON Utils**: Optimized for streaming JSON processing
- **Pandas Utils**: Efficient DataFrame operations with proper data types

## Dependencies

Core dependencies for dataknobs_utils:

```txt
pandas>=1.3.0
numpy>=1.20.0
requests>=2.25.0
psycopg2-binary>=2.8.6  # for SQL utils
elasticsearch>=7.0.0  # optional, for elasticsearch_utils
```

## Contributing

For contributing to dataknobs_utils:

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit pull request

See [Contributing Guide](../development/contributing.md) for detailed information.

## Changelog

### Version 1.0.0
- Initial release
- Core utility modules
- Elasticsearch integration
- File processing utilities
- LLM prompt management
- JSON processing tools

## License

See [License](../license.md) for license information.