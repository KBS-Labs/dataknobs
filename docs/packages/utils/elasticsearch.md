# Elasticsearch Utilities API Documentation

The `elasticsearch_utils` module provides utilities for interacting with Elasticsearch, including query building, data processing, and index management.

## Overview

This module includes:

- Query building functions for different search types
- Data processing utilities for results
- Batch loading utilities
- Elasticsearch index wrapper class
- Integration with pandas DataFrames

## Query Building Functions

### build_field_query_dict()
```python
def build_field_query_dict(
    fields: Union[str, List[str]], 
    text: str, 
    operator: Optional[str] = None
) -> Dict[str, Any]
```

Build an elasticsearch field query to find text in specified field(s).

**Parameters:**
- `fields` (Union[str, List[str]]): The field or fields to query
- `text` (str): The text to find
- `operator` (Optional[str], default=None): The operator to use (e.g., "AND", "OR")

**Returns:** Dict representing the Elasticsearch query

**Example:**
```python
from dataknobs_utils import elasticsearch_utils

# Single field query
query = elasticsearch_utils.build_field_query_dict("title", "python programming")
# Returns: {"query": {"match": {"title": {"query": "python programming"}}}}

# Multi-field query
query = elasticsearch_utils.build_field_query_dict(
    ["title", "content"], 
    "machine learning"
)
# Returns multi_match query

# Query with operator
query = elasticsearch_utils.build_field_query_dict(
    "description", 
    "data science", 
    operator="AND"
)
```

### build_phrase_query_dict()
```python
def build_phrase_query_dict(
    field: str, 
    phrase: str, 
    slop: int = 0
) -> Dict[str, Any]
```

Build an elasticsearch phrase query to find exact phrases in a field.

**Parameters:**
- `field` (str): The field to query
- `phrase` (str): The phrase to find
- `slop` (int, default=0): The slop factor for phrase matching

**Returns:** Dict representing the phrase query

**Example:**
```python
# Exact phrase query
query = elasticsearch_utils.build_phrase_query_dict(
    "content", 
    "artificial intelligence"
)

# Phrase query with slop (allows words to be reordered)
query = elasticsearch_utils.build_phrase_query_dict(
    "title", 
    "machine learning algorithms", 
    slop=2
)
```

## Result Processing Functions

### build_hits_dataframe()
```python
def build_hits_dataframe(query_result: Dict[str, Any]) -> Optional[pd.DataFrame]
```

Build a pandas DataFrame from elasticsearch query result hits.

**Parameters:**
- `query_result` (Dict[str, Any]): The elasticsearch query result

**Returns:** DataFrame containing the hits, or None if no hits

**Example:**
```python
import pandas as pd
from dataknobs_utils import elasticsearch_utils

# Process search results
query_result = {
    "hits": {
        "hits": [
            {"_source": {"title": "Doc 1", "content": "Content 1"}},
            {"_source": {"title": "Doc 2", "content": "Content 2"}}
        ]
    }
}

df = elasticsearch_utils.build_hits_dataframe(query_result)
print(df.head())
```

### decode_results()
```python
def decode_results(query_result: Dict[str, Any]) -> Dict[str, pd.DataFrame]
```

Decode elasticsearch query results into DataFrames.

**Parameters:**
- `query_result` (Dict[str, Any]): The elasticsearch query result

**Returns:** Dict with "hits_df" and/or "aggs_df" DataFrames

**Example:**
```python
result = elasticsearch_utils.decode_results(query_result)
if "hits_df" in result:
    hits_df = result["hits_df"]
    print(f"Found {len(hits_df)} documents")
```

## Batch Loading Functions

### add_batch_data()
```python
def add_batch_data(
    batchfile: TextIO,
    record_generator: Any,
    idx_name: str,
    source_id_fieldname: str = "id",
    cur_id: int = 1,
) -> int
```

Add source records to a batch file for Elasticsearch bulk loading.

**Parameters:**
- `batchfile` (TextIO): File handle for batch data output
- `record_generator` (Any): Generator yielding record dictionaries
- `idx_name` (str): Name of the Elasticsearch index
- `source_id_fieldname` (str, default="id"): Field name for record ID in source
- `cur_id` (int, default=1): Starting ID for records

**Returns:** Next available ID after processing all records

**Example:**
```python
def data_generator():
    yield {"title": "Document 1", "content": "Content 1"}
    yield {"title": "Document 2", "content": "Content 2"}

with open("batch_data.jsonl", "w") as f:
    next_id = elasticsearch_utils.add_batch_data(
        f, data_generator(), "documents", "doc_id", 1
    )
    print(f"Next available ID: {next_id}")
```

### batchfile_record_generator()
```python
def batchfile_record_generator(batchfile_path: str) -> Generator[Any, None, None]
```

Generate records from an Elasticsearch batch file.

**Parameters:**
- `batchfile_path` (str): Path to the batch file

**Yields:** Each record dictionary

**Example:**
```python
# Read records from batch file
for record in elasticsearch_utils.batchfile_record_generator("batch_data.jsonl"):
    print(f"Record: {record['title']}")
```

### collect_batchfile_values()
```python
def collect_batchfile_values(
    batchfile_path: str, 
    fieldname: str, 
    default_value: Any = ""
) -> List[Any]
```

Collect all values for a specific field from a batch file.

**Parameters:**
- `batchfile_path` (str): Path to the batch file
- `fieldname` (str): Name of the field to collect
- `default_value` (Any, default=""): Default value if field doesn't exist

**Returns:** List of collected values

**Example:**
```python
# Collect all titles
titles = elasticsearch_utils.collect_batchfile_values("batch_data.jsonl", "title")
print(f"Found {len(titles)} titles: {titles[:5]}")

# Collect with default value
authors = elasticsearch_utils.collect_batchfile_values(
    "batch_data.jsonl", "author", "Unknown"
)
```

### collect_batchfile_records()
```python
def collect_batchfile_records(batchfile_path: str) -> pd.DataFrame
```

Collect all batch file records as a pandas DataFrame.

**Parameters:**
- `batchfile_path` (str): Path to the batch file

**Returns:** DataFrame containing all records

**Example:**
```python
# Load batch data into DataFrame
df = elasticsearch_utils.collect_batchfile_records("batch_data.jsonl")
print(f"Loaded {len(df)} records")
print(df.columns.tolist())
```

## Classes

### TableSettings
```python
class TableSettings:
    def __init__(
        self,
        table_name: str,
        data_settings: Dict[str, Any],
        data_mapping: Dict[str, Any],
    ) -> None
```

Container for Elasticsearch table settings.

**Properties:**
- `name` (str): Table name
- `settings` (Dict[str, Any]): Elasticsearch index settings
- `mapping` (Dict[str, Any]): Elasticsearch field mappings

**Example:**
```python
settings = elasticsearch_utils.TableSettings(
    "documents",
    {
        "number_of_shards": 1,
        "number_of_replicas": 0
    },
    {
        "properties": {
            "title": {"type": "text"},
            "content": {"type": "text"},
            "created_at": {"type": "date"}
        }
    }
)
```

### ElasticsearchIndex
```python
class ElasticsearchIndex:
    def __init__(
        self,
        request_helper: Optional[Any],
        table_settings: List[TableSettings],
        elasticsearch_ip: Optional[str] = None,
        elasticsearch_port: int = 9200,
        mock_requests: Optional[Any] = None,
    ) -> None
```

Wrapper for interacting with an Elasticsearch index.

**Parameters:**
- `request_helper` (Optional[Any]): Request helper instance
- `table_settings` (List[TableSettings]): List of table configurations
- `elasticsearch_ip` (Optional[str], default=None): Elasticsearch IP (defaults to localhost)
- `elasticsearch_port` (int, default=9200): Elasticsearch port
- `mock_requests` (Optional[Any], default=None): Mock requests for testing

#### Methods

##### is_up()
```python
def is_up(self) -> bool
```
Check if the Elasticsearch server is running.

##### search()
```python
def search(
    self,
    query: Dict[str, Any],
    table: Optional[str] = None,
    verbose: bool = False,
) -> Optional[Any]
```
Submit an Elasticsearch search query.

##### sql()
```python
def sql(
    self,
    query: str,
    fetch_size: int = 10000,
    columnar: bool = True,
    verbose: bool = False,
) -> Any
```
Submit an Elasticsearch SQL query.

**Example:**
```python
from dataknobs_utils import elasticsearch_utils, requests_utils

# Create table settings
table_settings = [
    elasticsearch_utils.TableSettings(
        "documents",
        {"number_of_shards": 1},
        {
            "properties": {
                "title": {"type": "text"},
                "content": {"type": "text"}
            }
        }
    )
]

# Create Elasticsearch index wrapper
request_helper = requests_utils.RequestHelper("localhost", 9200)
index = elasticsearch_utils.ElasticsearchIndex(request_helper, table_settings)

# Check if server is up
if index.is_up():
    print("Elasticsearch is running")
    
    # Perform search
    query = elasticsearch_utils.build_field_query_dict("title", "python")
    result = index.search(query)
    
    if result and result.succeeded:
        if "hits_df" in result.extra:
            df = result.extra["hits_df"]
            print(f"Found {len(df)} documents")
    
    # Perform SQL query
    sql_result = index.sql("SELECT title, content FROM documents LIMIT 10")
    if sql_result and sql_result.succeeded:
        if "df" in sql_result.extra:
            df = sql_result.extra["df"]
            print(df.head())
```

## Usage Patterns

### Document Indexing Pipeline
```python
from dataknobs_utils import elasticsearch_utils, file_utils
import json

# Set up Elasticsearch index
table_settings = [
    elasticsearch_utils.TableSettings(
        "documents",
        {"number_of_shards": 1, "number_of_replicas": 0},
        {
            "properties": {
                "title": {"type": "text", "analyzer": "english"},
                "content": {"type": "text", "analyzer": "english"},
                "tags": {"type": "keyword"}
            }
        }
    )
]

index = elasticsearch_utils.ElasticsearchIndex(None, table_settings)

# Create batch file for bulk loading
def document_generator():
    for filepath in file_utils.filepath_generator("/documents"):
        if filepath.endswith(".json"):
            for line in file_utils.fileline_generator(filepath):
                try:
                    doc = json.loads(line)
                    yield doc
                except json.JSONDecodeError:
                    continue

with open("elasticsearch_batch.jsonl", "w") as batch_file:
    elasticsearch_utils.add_batch_data(
        batch_file, document_generator(), "documents"
    )
```

### Search and Analysis
```python
# Multi-field search
query = elasticsearch_utils.build_field_query_dict(
    ["title", "content"], "machine learning"
)
results = index.search(query)

if results and results.succeeded and "hits_df" in results.extra:
    hits_df = results.extra["hits_df"]
    print(f"Found {len(hits_df)} relevant documents")
    
    # Analyze results
    top_titles = hits_df["title"].head(10)
    print("Top matching documents:")
    for title in top_titles:
        print(f"- {title}")

# Phrase search
phrase_query = elasticsearch_utils.build_phrase_query_dict(
    "content", "natural language processing"
)
phrase_results = index.search(phrase_query)
```

### Analytics with SQL
```python
# Aggregate queries using Elasticsearch SQL
sql_queries = [
    "SELECT COUNT(*) as total_docs FROM documents",
    "SELECT tags, COUNT(*) as count FROM documents GROUP BY tags ORDER BY count DESC LIMIT 10",
    "SELECT DATE_TRUNC('month', created_at) as month, COUNT(*) as docs_per_month FROM documents GROUP BY month ORDER BY month"
]

for query in sql_queries:
    result = index.sql(query)
    if result and result.succeeded and "df" in result.extra:
        print(f"Query: {query}")
        print(result.extra["df"])
        print("---")
```

## Error Handling

```python
try:
    # Check server connectivity
    if not index.is_up():
        raise ConnectionError("Elasticsearch server is not available")
    
    # Perform search with error handling
    query = elasticsearch_utils.build_field_query_dict("title", "search term")
    result = index.search(query, verbose=True)
    
    if result and result.succeeded:
        # Process successful result
        decoded = elasticsearch_utils.decode_results(result.result)
        if "hits_df" in decoded:
            print(f"Found {len(decoded['hits_df'])} results")
    else:
        print(f"Search failed: {result.error if result else 'Unknown error'}")
        
except ConnectionError as e:
    print(f"Connection error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Performance Considerations

- Use batch loading for large datasets to improve indexing performance
- Configure appropriate shard and replica settings for your use case
- Use SQL queries for complex aggregations when possible
- Consider using columnar format for better performance with wide result sets
- Monitor Elasticsearch cluster health regularly

## Integration Examples

### With Text Processing
```python
from dataknobs_utils import elasticsearch_utils
from dataknobs_xization import normalize

# Index normalized text
def normalized_document_generator():
    for doc in raw_documents:
        normalized_doc = {
            "id": doc["id"],
            "title": normalize.basic_normalization_fn(doc["title"]),
            "content": normalize.basic_normalization_fn(doc["content"]),
            "tags": doc.get("tags", [])
        }
        yield normalized_doc

with open("normalized_batch.jsonl", "w") as f:
    elasticsearch_utils.add_batch_data(
        f, normalized_document_generator(), "normalized_documents"
    )
```

### With File Processing
```python
from dataknobs_utils import elasticsearch_utils, file_utils

# Process and index files from directory
def file_content_generator(directory):
    for filepath in file_utils.filepath_generator(directory):
        if filepath.endswith(".txt"):
            content_lines = list(file_utils.fileline_generator(filepath))
            yield {
                "filepath": filepath,
                "content": "\n".join(content_lines),
                "line_count": len(content_lines)
            }

with open("file_content_batch.jsonl", "w") as f:
    elasticsearch_utils.add_batch_data(
        f, file_content_generator("/documents"), "file_contents"
    )
```