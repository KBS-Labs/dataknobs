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
import json
from pathlib import Path

from dataknobs_utils import json_utils, file_utils

# Reading and writing whole files is stdlib; this package adds what sits on
# top of that -- addressing into a structure, and streaming a file by line.
with open("data.json") as handle:
    data = json.load(handle)

# Extract nested values. List elements are indexed with [n], not .n
value = json_utils.get_value(data, "path.to.nested[0].value")

# File operations: a line generator that transparently handles gzip, and a
# writer that takes the lines to put in it.
for line in file_utils.fileline_generator("example.txt"):
    process(line)
file_utils.write_lines("output.txt", ["one", "two"])

Path("output.json").write_text(json.dumps({"key": "value"}))
```

## Dependencies

This package depends on:
- `dataknobs-common`
- `dataknobs-structures`
- pandas, requests, psycopg2-binary, lxml, beautifulsoup4, json-stream, scikit-learn

## License

Licensed under the [Apache License, Version 2.0](LICENSE); see [NOTICE](NOTICE)
for attribution requirements.

Versions released before this change remain available under the MIT License,
preserved in [LICENSES/MIT-historical.txt](../../LICENSES/MIT-historical.txt).
