# Utils Package API Reference

Complete API reference for the `dataknobs_utils` package.

## Package Overview

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

## Module Index

### Core Utilities
- [file_utils](file-utils.md) - File operations and path handling
- [json_utils](json-utils.md) - JSON processing and manipulation
- [llm_utils](llm-utils.md) - LLM prompt and message management

### Data Processing
- **elasticsearch_utils** - [Elasticsearch integration](elasticsearch.md)
- **pandas_utils** - DataFrame operations and utilities
- **stats_utils** - Statistical analysis functions
- **xml_utils** - XML parsing and processing

### System and Network
- **requests_utils** - HTTP request utilities and helpers
- **subprocess_utils** - Process execution utilities
- **sys_utils** - System information and utilities

### Specialized Tools
- **emoji_utils** - Emoji detection and processing
- **json_extractor** - Advanced JSON data extraction
- **resource_utils** - Resource file management
- **sql_utils** - SQL query utilities

## Quick Reference

### File Operations
```python
from dataknobs_utils import file_utils

# Generate file paths
for filepath in file_utils.filepath_generator("/data"):
    print(filepath)

# Read lines with compression support
for line in file_utils.fileline_generator("data.txt.gz"):
    process_line(line)

# Write lines with sorting
file_utils.write_lines("output.txt", lines)

# Check file compression
if file_utils.is_gzip_file("data.gz"):
    print("File is compressed")
```

### JSON Processing
```python
from dataknobs_utils import json_utils

# Process JSON files
data = json_utils.load_json_file("config.json")
processed = json_utils.process_json_data(data)
json_utils.save_json_file(processed, "output.json")
```

### LLM Integration
```python
from dataknobs_utils import llm_utils

# Create prompt messages
message = llm_utils.PromptMessage(
    "user", 
    "Analyze this data",
    metadata={"priority": "high"}
)

# Access nested configuration
value = llm_utils.get_value_by_key(
    config, "models.gpt4.temperature", 0.7
)
```

### Elasticsearch Operations
```python
from dataknobs_utils import elasticsearch_utils

# Build queries
query = elasticsearch_utils.build_field_query_dict(
    ["title", "content"], "search term"
)

phrase_query = elasticsearch_utils.build_phrase_query_dict(
    "content", "exact phrase", slop=2
)

# Work with results
hits_df = elasticsearch_utils.build_hits_dataframe(result)
results = elasticsearch_utils.decode_results(query_result)
```

### Data Processing
```python
from dataknobs_utils import pandas_utils, stats_utils

# DataFrame utilities
df = pandas_utils.process_dataframe(raw_data)
summary = pandas_utils.generate_summary(df)

# Statistical analysis
stats = stats_utils.calculate_statistics(data)
distribution = stats_utils.analyze_distribution(values)
```

## Module Details

### elasticsearch_utils

**Classes:**
- `TableSettings` - Elasticsearch table configuration
- `ElasticsearchIndex` - Index management wrapper

**Functions:**
- `build_field_query_dict()` - Create field-based queries
- `build_phrase_query_dict()` - Create phrase queries
- `build_hits_dataframe()` - Convert hits to DataFrame
- `build_aggs_dataframe()` - Convert aggregations to DataFrame
- `decode_results()` - Process query results
- `add_batch_data()` - Add records to batch file
- `batchfile_record_generator()` - Generate records from batch file
- `collect_batchfile_values()` - Collect field values
- `collect_batchfile_records()` - Load batch records as DataFrame

### file_utils

**Functions:**
- `filepath_generator()` - Generate file paths recursively
- `fileline_generator()` - Generate file lines with compression support
- `write_lines()` - Write sorted lines to file
- `is_gzip_file()` - Check if file is gzipped

### json_utils

**Functions:**
- JSON file loading and saving
- JSON data validation
- JSON schema operations
- Nested data manipulation

### llm_utils

**Classes:**
- `PromptMessage` - Message wrapper with metadata

**Functions:**
- `get_value_by_key()` - Deep dictionary value retrieval

### pandas_utils

**Functions:**
- DataFrame creation and manipulation
- Data type conversion utilities
- Summary statistics generation
- Data cleaning operations

### requests_utils

**Classes:**
- `RequestHelper` - HTTP request management
- `ServerResponse` - Response wrapper

**Functions:**
- HTTP method utilities
- Response processing
- Error handling

### stats_utils

**Functions:**
- Descriptive statistics calculation
- Distribution analysis
- Correlation analysis
- Statistical testing utilities

### xml_utils

**Functions:**
- XML parsing and validation
- XML to dictionary conversion
- XPath query utilities
- XML transformation functions

### emoji_utils

**Functions:**
- Emoji detection in text
- Emoji classification
- Unicode emoji utilities
- Text cleaning functions

### json_extractor

**Classes:**
- Advanced JSON data extraction
- Pattern-based extraction
- Schema inference

### resource_utils

**Functions:**
- Package resource access
- Resource file loading
- Path resolution utilities

### sql_utils

**Functions:**
- SQL query building
- Database connection utilities
- Result processing
- Query optimization helpers

### subprocess_utils

**Functions:**
- Process execution utilities
- Command building helpers
- Output capture and processing
- Error handling

### sys_utils

**Functions:**
- System information retrieval
- Environment variable utilities
- Platform detection
- Resource monitoring

## Common Usage Patterns

### Data Pipeline Integration
```python
from dataknobs_utils import (
    file_utils, json_utils, pandas_utils, 
    elasticsearch_utils, stats_utils
)

# Complete data processing pipeline
def process_data_pipeline(input_dir, output_dir):
    # Collect all data files
    data_files = list(file_utils.filepath_generator(input_dir))
    
    # Process each file
    all_data = []
    for filepath in data_files:
        if filepath.endswith('.json'):
            # Load and validate JSON
            data = json_utils.load_json_file(filepath)
            if json_utils.validate_schema(data, schema):
                all_data.extend(data)
    
    # Convert to DataFrame and analyze
    df = pandas_utils.create_dataframe(all_data)
    summary = stats_utils.generate_summary(df)
    
    # Index in Elasticsearch
    with open(f"{output_dir}/batch.jsonl", "w") as f:
        elasticsearch_utils.add_batch_data(
            f, iter(all_data), "processed_data"
        )
    
    return summary
```

### Configuration Management
```python
from dataknobs_utils import llm_utils, json_utils

class ConfigManager:
    def __init__(self, config_path):
        self.config = json_utils.load_json_file(config_path)
    
    def get_setting(self, path, default=None):
        return llm_utils.get_value_by_key(self.config, path, default)
    
    def get_database_config(self):
        return {
            "host": self.get_setting("database.host", "localhost"),
            "port": self.get_setting("database.port", 5432),
            "name": self.get_setting("database.name")
        }
    
    def get_elasticsearch_config(self):
        return {
            "host": self.get_setting("elasticsearch.host", "localhost"),
            "port": self.get_setting("elasticsearch.port", 9200),
            "index_settings": self.get_setting("elasticsearch.settings", {})
        }
```

### Error Handling Patterns
```python
from dataknobs_utils import file_utils, requests_utils
import logging

def safe_data_processing(input_path, output_path):
    """Process data with comprehensive error handling."""
    try:
        # Check input exists
        if not file_utils.filepath_exists(input_path):
            raise FileNotFoundError(f"Input path not found: {input_path}")
        
        processed_lines = []
        error_count = 0
        
        # Process each line safely
        for line in file_utils.fileline_generator(input_path):
            try:
                processed = process_line(line)
                processed_lines.append(processed)
            except Exception as e:
                logging.error(f"Error processing line: {e}")
                error_count += 1
        
        # Write results
        file_utils.write_lines(output_path, processed_lines)
        
        return {
            "processed": len(processed_lines),
            "errors": error_count,
            "success": True
        }
        
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        return {"success": False, "error": str(e)}
```

## Testing Utilities

```python
from dataknobs_utils import file_utils, json_utils
import tempfile
import os

def test_file_operations():
    """Test file utilities with temporary files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test data
        test_file = os.path.join(temp_dir, "test.txt")
        test_lines = ["line 1", "line 2", "line 3"]
        
        # Test writing
        file_utils.write_lines(test_file, test_lines)
        
        # Test reading
        read_lines = list(file_utils.fileline_generator(test_file))
        assert read_lines == sorted(test_lines)
        
        print("File operations test passed")

def test_json_operations():
    """Test JSON utilities."""
    test_data = {"key": "value", "nested": {"inner": "data"}}
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name
    
    try:
        # Test save/load
        json_utils.save_json_file(test_data, temp_path)
        loaded_data = json_utils.load_json_file(temp_path)
        assert loaded_data == test_data
        
        print("JSON operations test passed")
    finally:
        os.unlink(temp_path)
```

## Best Practices

1. **Error Handling**: Always use try-catch blocks for file and network operations
2. **Resource Management**: Use context managers for file operations
3. **Memory Efficiency**: Use generators for large datasets
4. **Configuration**: Use nested key access for complex configurations
5. **Testing**: Use temporary directories for testing file operations
6. **Logging**: Include appropriate logging for debugging
7. **Validation**: Validate input data before processing
8. **Documentation**: Include type hints and docstrings

## Version Information

- **Package Version**: 1.0.0
- **Python Compatibility**: 3.8+
- **Dependencies**: pandas, numpy, requests, elasticsearch (optional)

For detailed documentation of individual modules, see their respective documentation pages.