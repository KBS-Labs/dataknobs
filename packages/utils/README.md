# dataknobs-utils

Utility functions for dataknobs packages.

## Installation

```bash
pip install dataknobs-utils
```

## Features

### File Utilities
- File reading/writing with various formats (JSON, GZIP, etc.)
- Resource management
- Path utilities

### Data Processing
- **JSON utilities**: Schema extraction, value traversal, data transformation
- **Pandas utilities**: DataFrame operations and transformations
- **XML utilities**: XML parsing and manipulation
- **SQL utilities**: Database connection and query helpers

### Web & API
- **Requests utilities**: HTTP request helpers with retry logic
- **Elasticsearch utilities**: ES client helpers and query builders

### System & Process
- **System utilities**: Environment variable management
- **Subprocess utilities**: Process execution helpers
- **Stats utilities**: Statistical calculations

### Other
- **LLM utilities**: Utilities for working with language models
- **Emoji utilities**: Emoji processing and handling

## Usage

```python
from dataknobs_utils import json_utils, file_utils

# Read JSON file
data = json_utils.load_json_file("data.json")

# Extract nested values
value = json_utils.get_value(data, "path.to.nested[0].value")

# File operations
content = file_utils.read_text_file("example.txt")
file_utils.write_json_file("output.json", {"key": "value"})
```

## Dependencies

This package depends on:
- `dataknobs-common`
- `dataknobs-structures`
- pandas, requests, psycopg2-binary, lxml, beautifulsoup4, json-stream, scikit-learn

## License

See LICENSE file in the root repository.